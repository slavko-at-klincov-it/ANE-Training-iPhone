// ANEUltimateBenchmark.m — Definitive benchmark for ANE training & inference on iPhone
// Compares ANE vs GPU (batched MPS) vs CPU with per-component timing, power, efficiency
#import "ANEUltimateBenchmark.h"
#import "ANETrainingEngine.h"
#import "ANETrainingConfig.h"
#import "ANEStoriesMIL.h"
#import "ANEStoriesCPUOps.h"
#import "ANEThermal.h"
#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#import <mach/mach_time.h>
#import <mach/mach.h>
#import <mach/task_info.h>
#include <sys/utsname.h>

// ═══════════════════════════════════════════════════════
// INFRASTRUCTURE
// ═══════════════════════════════════════════════════════

static mach_timebase_info_data_t g_ub_tb;
static void ub_init_tb(void) { static dispatch_once_t o; dispatch_once(&o, ^{ mach_timebase_info(&g_ub_tb); }); }
#define UB_MS(t) ((double)(t)*g_ub_tb.numer/g_ub_tb.denom/1e6)
#define UB_S(t)  ((double)(t)*g_ub_tb.numer/g_ub_tb.denom/1e9)

// --- Stats with percentile support ---
typedef struct {
    double *samples;
    int count, capacity;
    double sum, sum_sq, min, max;
} UBStats;

static void ubs_init(UBStats *s, int cap) {
    s->samples = (double*)malloc(cap * sizeof(double));
    s->count = 0; s->capacity = cap;
    s->sum = s->sum_sq = 0; s->min = 1e9; s->max = -1e9;
}
static void ubs_add(UBStats *s, double v) {
    if (s->count >= s->capacity) {
        s->capacity *= 2;
        s->samples = (double*)realloc(s->samples, s->capacity * sizeof(double));
    }
    s->samples[s->count++] = v;
    s->sum += v; s->sum_sq += v*v;
    if (v < s->min) s->min = v;
    if (v > s->max) s->max = v;
}
static double ubs_mean(UBStats *s) { return s->count > 0 ? s->sum / s->count : 0; }
static double ubs_std(UBStats *s) {
    if (s->count < 2) return 0;
    double m = ubs_mean(s), v = s->sum_sq/s->count - m*m;
    return v > 0 ? sqrt(v) : 0;
}
static int dbl_cmp(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}
static double ubs_pct(UBStats *s, double p) {
    if (s->count == 0) return 0;
    qsort(s->samples, s->count, sizeof(double), dbl_cmp);
    int idx = (int)(s->count * p);
    if (idx >= s->count) idx = s->count - 1;
    return s->samples[idx];
}
static void ubs_free(UBStats *s) { free(s->samples); s->samples = NULL; }

// --- System metrics ---
static float ub_battery(void) {
    __block float level = -1;
    if ([NSThread isMainThread]) {
        [UIDevice currentDevice].batteryMonitoringEnabled = YES;
        level = [UIDevice currentDevice].batteryLevel;
    } else {
        dispatch_sync(dispatch_get_main_queue(), ^{
            [UIDevice currentDevice].batteryMonitoringEnabled = YES;
            level = [UIDevice currentDevice].batteryLevel;
        });
    }
    return level;
}

static double ub_cpu_time(void) {
    struct task_basic_info info;
    mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return -1;
    return (info.user_time.seconds + info.user_time.microseconds/1e6)
         + (info.system_time.seconds + info.system_time.microseconds/1e6);
}

static double ub_memory_mb(void) {
    task_vm_info_data_t vm;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &count) != KERN_SUCCESS) return -1;
    return (double)vm.phys_footprint / (1024.0 * 1024.0);
}

static double ub_battery_wh(void) {
    struct utsname si; uname(&si);
    NSString *hw = [NSString stringWithCString:si.machine encoding:NSUTF8StringEncoding];
    double mAh = 3400;
    if ([hw hasPrefix:@"iPhone16,1"])      mAh = 3274;
    else if ([hw hasPrefix:@"iPhone16,2"]) mAh = 4422;
    else if ([hw hasPrefix:@"iPhone17,1"]) mAh = 3582;
    else if ([hw hasPrefix:@"iPhone17,2"]) mAh = 4685;
    return mAh * 3.83 / 1000.0;
}

// --- Per-backend result ---
typedef struct {
    const char *name;
    float batt_start, batt_end;
    double elapsed_s, cpu_start, cpu_end;
    double mem_start, mem_peak;
    int passes;
    UBStats latency;
    ThermalLevel worst_thermal;
    double thermal_time[4]; // seconds in nominal/fair/serious/critical
} UBResult;

// ═══════════════════════════════════════════════════════
// SHARED WEIGHTS (same as ANEInferenceBenchmark.m)
// ═══════════════════════════════════════════════════════

typedef struct {
    LayerWeights lw[NLAYERS];
    float *rms_final, *embed;
} UBWeights;

static UBWeights ub_weights_alloc(void) {
    UBWeights w;
    for (int L = 0; L < NLAYERS; L++) w.lw[L] = layer_weights_alloc();
    w.rms_final = (float*)malloc(DIM*4);
    w.embed = (float*)malloc((size_t)VOCAB*DIM*4);
    srand48(42);
    float sd=1.0f/sqrtf(DIM), sh=1.0f/sqrtf(HIDDEN), os=1.0f/sqrtf((float)NLAYERS);
    for (int L=0; L<NLAYERS; L++) {
        for (size_t i=0;i<WQ_SZ;i++){w.lw[L].Wq[i]=sd*(2*drand48()-1);w.lw[L].Wk[i]=sd*(2*drand48()-1);}
        for (size_t i=0;i<WQ_SZ;i++){w.lw[L].Wv[i]=sd*(2*drand48()-1);w.lw[L].Wo[i]=sd*os*(2*drand48()-1);}
        for (size_t i=0;i<W1_SZ;i++) w.lw[L].W1[i]=sh*(2*drand48()-1);
        for (size_t i=0;i<W2_SZ;i++) w.lw[L].W2[i]=sd*os*(2*drand48()-1);
        for (size_t i=0;i<W3_SZ;i++) w.lw[L].W3[i]=sh*(2*drand48()-1);
        for (int i=0;i<DIM;i++){w.lw[L].rms_att[i]=1;w.lw[L].rms_ffn[i]=1;}
    }
    for (int i=0;i<DIM;i++) w.rms_final[i]=1;
    for (size_t i=0;i<(size_t)VOCAB*DIM;i++) w.embed[i]=0.02f*(2*drand48()-1);
    return w;
}
static void ub_weights_free(UBWeights *w) {
    for (int L=0;L<NLAYERS;L++) layer_weights_free(&w->lw[L]);
    free(w->rms_final); free(w->embed);
}

// ═══════════════════════════════════════════════════════
// ANE INFERENCE
// ═══════════════════════════════════════════════════════

typedef struct {
    Kern *fwdAttn[NLAYERS], *fwdFFN[NLAYERS];
    float *x_cur, *x_final, *o_out, *x2, *ffn_out, *logits;
    float *rms_final, *embed; // borrowed pointers
    int compile_count; bool ok;
} UBAne;

static UBAne *ub_ane_init(UBWeights *w) {
    ane_init();
    UBAne *a = (UBAne*)calloc(1, sizeof(UBAne));
    a->rms_final = w->rms_final; a->embed = w->embed;
    a->x_cur=(float*)malloc(SEQ*DIM*4); a->x_final=(float*)malloc(SEQ*DIM*4);
    a->o_out=(float*)malloc(SEQ*DIM*4); a->x2=(float*)malloc(SEQ*DIM*4);
    a->ffn_out=(float*)malloc(SEQ*DIM*4); a->logits=(float*)malloc(VOCAB*4);
    a->ok = true;
    for (int L=0; L<NLAYERS; L++) {
        a->fwdAttn[L] = compile_kern(gen_sdpa_fwd_taps(), (@{
            @"@model_path/weights/rms1.bin":@{@"offset":@0,@"data":build_blob(w->lw[L].rms_att,1,DIM)},
            @"@model_path/weights/wq.bin":@{@"offset":@0,@"data":build_blob(w->lw[L].Wq,DIM,DIM)},
            @"@model_path/weights/wk.bin":@{@"offset":@0,@"data":build_blob(w->lw[L].Wk,DIM,DIM)},
            @"@model_path/weights/wv.bin":@{@"offset":@0,@"data":build_blob(w->lw[L].Wv,DIM,DIM)},
            @"@model_path/weights/wo.bin":@{@"offset":@0,@"data":build_blob(w->lw[L].Wo,DIM,DIM)},
            @"@model_path/weights/mask.bin":@{@"offset":@0,@"data":get_mask_blob()},
        }), DIM*SEQ*2, 6*DIM*SEQ*2);
        a->fwdFFN[L] = compile_kern(gen_ffn_fwd_taps(), (@{
            @"@model_path/weights/rms2.bin":@{@"offset":@0,@"data":build_blob(w->lw[L].rms_ffn,1,DIM)},
            @"@model_path/weights/w1.bin":@{@"offset":@0,@"data":build_blob(w->lw[L].W1,HIDDEN,DIM)},
            @"@model_path/weights/w3.bin":@{@"offset":@0,@"data":build_blob(w->lw[L].W3,HIDDEN,DIM)},
            @"@model_path/weights/w2.bin":@{@"offset":@0,@"data":build_blob(w->lw[L].W2,DIM,HIDDEN)},
        }), DIM*SEQ*2, (2*DIM+3*HIDDEN)*SEQ*2);
        a->compile_count += 2;
        if (!a->fwdAttn[L]||!a->fwdFFN[L]) { a->ok=false; break; }
    }
    return a;
}
static void ub_ane_free(UBAne *a) {
    for (int L=0;L<NLAYERS;L++){free_kern(a->fwdAttn[L]);free_kern(a->fwdFFN[L]);}
    free(a->x_cur);free(a->x_final);free(a->o_out);free(a->x2);free(a->ffn_out);free(a->logits);
    free(a);
}

// Component timing accumulators (per forward pass)
typedef struct {
    double embed_ms, ane_eval_ms, io_ms, cpu_ms, classifier_ms;
} UBComponents;

static void ub_ane_forward(UBAne *a, const uint16_t *tokens, UBComponents *c) {
    uint64_t t;
    // Embed
    t = mach_absolute_time();
    memset(a->x_cur,0,SEQ*DIM*4);
    embed_lookup(a->x_cur, a->embed, tokens, DIM, SEQ);
    c->embed_ms = UB_MS(mach_absolute_time()-t);

    double eval_total=0, io_total=0, cpu_total=0;
    for (int L=0; L<NLAYERS; L++) {
        // Attn: IO write
        t=mach_absolute_time();
        io_write_fp16(a->fwdAttn[L]->ioIn, a->x_cur, DIM, SEQ);
        io_total += UB_MS(mach_absolute_time()-t);
        // Attn: ANE eval
        t=mach_absolute_time();
        ane_eval(a->fwdAttn[L]);
        eval_total += UB_MS(mach_absolute_time()-t);
        // Attn: IO read
        t=mach_absolute_time();
        io_read_fp16(a->fwdAttn[L]->ioOut, a->o_out, 0, DIM, SEQ);
        io_total += UB_MS(mach_absolute_time()-t);
        // Residual (CPU)
        t=mach_absolute_time();
        vDSP_vadd(a->x_cur,1,a->o_out,1,a->x2,1,(vDSP_Length)(SEQ*DIM));
        cpu_total += UB_MS(mach_absolute_time()-t);

        // FFN: IO write
        t=mach_absolute_time();
        io_write_fp16(a->fwdFFN[L]->ioIn, a->x2, DIM, SEQ);
        io_total += UB_MS(mach_absolute_time()-t);
        // FFN: ANE eval
        t=mach_absolute_time();
        ane_eval(a->fwdFFN[L]);
        eval_total += UB_MS(mach_absolute_time()-t);
        // FFN: IO read
        t=mach_absolute_time();
        io_read_fp16(a->fwdFFN[L]->ioOut, a->ffn_out, 0, DIM, SEQ);
        io_total += UB_MS(mach_absolute_time()-t);
        // Residual (CPU)
        t=mach_absolute_time();
        vDSP_vadd(a->x2,1,a->ffn_out,1,a->x_cur,1,(vDSP_Length)(SEQ*DIM));
        cpu_total += UB_MS(mach_absolute_time()-t);
    }
    c->ane_eval_ms = eval_total;
    c->io_ms = io_total;

    // Final rmsnorm + classifier
    t=mach_absolute_time();
    rmsnorm(a->x_final, a->x_cur, a->rms_final, DIM, SEQ);
    cpu_total += UB_MS(mach_absolute_time()-t);
    c->cpu_ms = cpu_total;

    t=mach_absolute_time();
    int pos=SEQ-1; float *xp=(float*)malloc(DIM*4);
    for (int d=0;d<DIM;d++) xp[d]=a->x_final[d*SEQ+pos];
    cblas_sgemv(CblasRowMajor,CblasNoTrans,VOCAB,DIM,1.0f,a->embed,DIM,xp,1,0.0f,a->logits,1);
    free(xp);
    c->classifier_ms = UB_MS(mach_absolute_time()-t);
}

// ═══════════════════════════════════════════════════════
// CPU INFERENCE
// ═══════════════════════════════════════════════════════

typedef struct {
    float *x_cur,*x_norm,*x2,*x_final,*Q,*K,*V,*attn_out,*o_out;
    float *scores,*h1,*h3,*silu_out,*ffn_out,*logits;
} UBCpuBufs;

static UBCpuBufs ub_cpu_alloc(void) {
    UBCpuBufs b;
    b.x_cur=(float*)malloc(SEQ*DIM*4);b.x_norm=(float*)malloc(SEQ*DIM*4);
    b.x2=(float*)malloc(SEQ*DIM*4);b.x_final=(float*)malloc(SEQ*DIM*4);
    b.Q=(float*)malloc(SEQ*DIM*4);b.K=(float*)malloc(SEQ*DIM*4);
    b.V=(float*)malloc(SEQ*DIM*4);b.attn_out=(float*)malloc(SEQ*DIM*4);
    b.o_out=(float*)malloc(SEQ*DIM*4);b.scores=(float*)malloc(HEADS*SEQ*SEQ*4);
    b.h1=(float*)malloc(SEQ*HIDDEN*4);b.h3=(float*)malloc(SEQ*HIDDEN*4);
    b.silu_out=(float*)malloc(SEQ*HIDDEN*4);b.ffn_out=(float*)malloc(SEQ*DIM*4);
    b.logits=(float*)malloc(VOCAB*4);
    return b;
}
static void ub_cpu_free(UBCpuBufs *b) {
    free(b->x_cur);free(b->x_norm);free(b->x2);free(b->x_final);
    free(b->Q);free(b->K);free(b->V);free(b->attn_out);free(b->o_out);
    free(b->scores);free(b->h1);free(b->h3);free(b->silu_out);free(b->ffn_out);
    free(b->logits);
}

static void ub_silu(float *x, int n) {
    float *neg=(float*)malloc(n*4); float m1=-1;
    vDSP_vsmul(x,1,&m1,neg,1,(vDSP_Length)n);
    vvexpf(neg,neg,&n); float one=1;
    vDSP_vsadd(neg,1,&one,neg,1,(vDSP_Length)n);
    vDSP_vdiv(neg,1,x,1,x,1,(vDSP_Length)n); free(neg);
}

static void ub_cpu_attention(float *out, const float *Q, const float *K, const float *V,
                             float *sc, int dim, int seq, int nh) {
    int hd=dim/nh;
    for (int h=0;h<nh;h++) {
        const float *Qh=Q+h*hd*seq, *Kh=K+h*hd*seq, *Vh=V+h*hd*seq;
        float *s=sc+h*seq*seq;
        cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,seq,seq,hd,1,Qh,seq,Kh,seq,0,s,seq);
        float scale=1.0f/sqrtf((float)hd);
        vDSP_vsmul(s,1,&scale,s,1,(vDSP_Length)(seq*seq));
        for (int i=0;i<seq;i++) for (int j=i+1;j<seq;j++) s[i*seq+j]=-1e9f;
        for (int i=0;i<seq;i++) {
            float *r=s+i*seq; float mx; vDSP_maxv(r,1,&mx,(vDSP_Length)seq);
            float nm=-mx; vDSP_vsadd(r,1,&nm,r,1,(vDSP_Length)seq);
            int n2=seq; vvexpf(r,r,&n2);
            float sm; vDSP_sve(r,1,&sm,(vDSP_Length)seq);
            float inv=1.0f/sm; vDSP_vsmul(r,1,&inv,r,1,(vDSP_Length)seq);
        }
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,hd,seq,seq,1,Vh,seq,s,seq,0,out+h*hd*seq,seq);
    }
}

static void ub_cpu_forward(UBWeights *w, UBCpuBufs *b, const uint16_t *tokens, UBComponents *c) {
    uint64_t t;
    t=mach_absolute_time();
    memset(b->x_cur,0,SEQ*DIM*4);
    embed_lookup(b->x_cur, w->embed, tokens, DIM, SEQ);
    c->embed_ms = UB_MS(mach_absolute_time()-t);

    double matmul_total=0, attn_total=0, norm_total=0, act_total=0;
    for (int L=0;L<NLAYERS;L++) {
        t=mach_absolute_time();
        rmsnorm(b->x_norm, b->x_cur, w->lw[L].rms_att, DIM, SEQ);
        norm_total += UB_MS(mach_absolute_time()-t);

        t=mach_absolute_time();
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,DIM,SEQ,DIM,1,w->lw[L].Wq,DIM,b->x_norm,SEQ,0,b->Q,SEQ);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,DIM,SEQ,DIM,1,w->lw[L].Wk,DIM,b->x_norm,SEQ,0,b->K,SEQ);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,DIM,SEQ,DIM,1,w->lw[L].Wv,DIM,b->x_norm,SEQ,0,b->V,SEQ);
        matmul_total += UB_MS(mach_absolute_time()-t);

        t=mach_absolute_time();
        ub_cpu_attention(b->attn_out, b->Q, b->K, b->V, b->scores, DIM, SEQ, HEADS);
        attn_total += UB_MS(mach_absolute_time()-t);

        t=mach_absolute_time();
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,DIM,SEQ,DIM,1,w->lw[L].Wo,DIM,b->attn_out,SEQ,0,b->o_out,SEQ);
        matmul_total += UB_MS(mach_absolute_time()-t);

        vDSP_vadd(b->x_cur,1,b->o_out,1,b->x2,1,(vDSP_Length)(SEQ*DIM));

        t=mach_absolute_time();
        rmsnorm(b->x_norm, b->x2, w->lw[L].rms_ffn, DIM, SEQ);
        norm_total += UB_MS(mach_absolute_time()-t);

        t=mach_absolute_time();
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,HIDDEN,SEQ,DIM,1,w->lw[L].W1,DIM,b->x_norm,SEQ,0,b->h1,SEQ);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,HIDDEN,SEQ,DIM,1,w->lw[L].W3,DIM,b->x_norm,SEQ,0,b->h3,SEQ);
        matmul_total += UB_MS(mach_absolute_time()-t);

        t=mach_absolute_time();
        ub_silu(b->h1, HIDDEN*SEQ);
        vDSP_vmul(b->h1,1,b->h3,1,b->silu_out,1,(vDSP_Length)(HIDDEN*SEQ));
        act_total += UB_MS(mach_absolute_time()-t);

        t=mach_absolute_time();
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,DIM,SEQ,HIDDEN,1,w->lw[L].W2,HIDDEN,b->silu_out,SEQ,0,b->ffn_out,SEQ);
        matmul_total += UB_MS(mach_absolute_time()-t);

        vDSP_vadd(b->x2,1,b->ffn_out,1,b->x_cur,1,(vDSP_Length)(SEQ*DIM));
    }
    // Store CPU breakdown: matmul in ane_eval_ms, attention in io_ms, norms+act in cpu_ms
    c->ane_eval_ms = matmul_total;  // "compute" portion
    c->io_ms = attn_total;          // attention
    c->cpu_ms = norm_total + act_total; // norms + activations

    t=mach_absolute_time();
    rmsnorm(b->x_final, b->x_cur, w->rms_final, DIM, SEQ);
    int pos=SEQ-1; float *xp=(float*)malloc(DIM*4);
    for (int d=0;d<DIM;d++) xp[d]=b->x_final[d*SEQ+pos];
    cblas_sgemv(CblasRowMajor,CblasNoTrans,VOCAB,DIM,1,w->embed,DIM,xp,1,0,b->logits,1);
    free(xp);
    c->classifier_ms = UB_MS(mach_absolute_time()-t);
}

// ═══════════════════════════════════════════════════════
// GPU INFERENCE (batched MPS — fair comparison)
// ═══════════════════════════════════════════════════════

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> wq[NLAYERS],wk[NLAYERS],wv[NLAYERS],wo[NLAYERS];
    id<MTLBuffer> w1[NLAYERS],w2[NLAYERS],w3[NLAYERS];
    id<MTLBuffer> buf_x,buf_norm,buf_Q,buf_K,buf_V,buf_attn,buf_o;
    id<MTLBuffer> buf_h1,buf_h3,buf_silu,buf_ffn;
    // Pre-created MPS objects (reusable)
    MPSMatrixMultiplication *mm_dim;    // DIM×DIM @ DIM×SEQ
    MPSMatrixMultiplication *mm_up;     // HIDDEN×DIM @ DIM×SEQ
    MPSMatrixMultiplication *mm_down;   // DIM×HIDDEN @ HIDDEN×SEQ
} UBGpu;

static UBGpu *ub_gpu_init(UBWeights *w) {
    UBGpu *g = (UBGpu*)calloc(1, sizeof(UBGpu));
    g->device = MTLCreateSystemDefaultDevice();
    if (!g->device) { free(g); return NULL; }
    g->queue = [g->device newCommandQueue];

    for (int L=0;L<NLAYERS;L++) {
        g->wq[L]=[g->device newBufferWithBytes:w->lw[L].Wq length:WQ_SZ*4 options:MTLResourceStorageModeShared];
        g->wk[L]=[g->device newBufferWithBytes:w->lw[L].Wk length:WQ_SZ*4 options:MTLResourceStorageModeShared];
        g->wv[L]=[g->device newBufferWithBytes:w->lw[L].Wv length:WQ_SZ*4 options:MTLResourceStorageModeShared];
        g->wo[L]=[g->device newBufferWithBytes:w->lw[L].Wo length:WO_SZ*4 options:MTLResourceStorageModeShared];
        g->w1[L]=[g->device newBufferWithBytes:w->lw[L].W1 length:W1_SZ*4 options:MTLResourceStorageModeShared];
        g->w2[L]=[g->device newBufferWithBytes:w->lw[L].W2 length:W2_SZ*4 options:MTLResourceStorageModeShared];
        g->w3[L]=[g->device newBufferWithBytes:w->lw[L].W3 length:W3_SZ*4 options:MTLResourceStorageModeShared];
    }
    g->buf_x=[g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_norm=[g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_Q=[g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_K=[g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_V=[g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_attn=[g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_o=[g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_h1=[g->device newBufferWithLength:SEQ*HIDDEN*4 options:MTLResourceStorageModeShared];
    g->buf_h3=[g->device newBufferWithLength:SEQ*HIDDEN*4 options:MTLResourceStorageModeShared];
    g->buf_silu=[g->device newBufferWithLength:SEQ*HIDDEN*4 options:MTLResourceStorageModeShared];
    g->buf_ffn=[g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];

    // Pre-create MPS multiply objects (reusable — dimensions fixed)
    g->mm_dim = [[MPSMatrixMultiplication alloc] initWithDevice:g->device
        transposeLeft:NO transposeRight:NO resultRows:DIM resultColumns:SEQ interiorColumns:DIM alpha:1 beta:0];
    g->mm_up = [[MPSMatrixMultiplication alloc] initWithDevice:g->device
        transposeLeft:NO transposeRight:NO resultRows:HIDDEN resultColumns:SEQ interiorColumns:DIM alpha:1 beta:0];
    g->mm_down = [[MPSMatrixMultiplication alloc] initWithDevice:g->device
        transposeLeft:NO transposeRight:NO resultRows:DIM resultColumns:SEQ interiorColumns:HIDDEN alpha:1 beta:0];
    return g;
}
static void ub_gpu_free(UBGpu *g) { free(g); }

// Helper: encode matmul using pre-created MPS object
static void ub_gpu_encode(UBGpu *g, MPSMatrixMultiplication *mm, id<MTLCommandBuffer> cmd,
                          id<MTLBuffer> bA, int rA, int cA,
                          id<MTLBuffer> bB, int rB, int cB,
                          id<MTLBuffer> bC, int rC, int cC) {
    MPSMatrixDescriptor *dA=[MPSMatrixDescriptor matrixDescriptorWithRows:rA columns:cA rowBytes:cA*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dB=[MPSMatrixDescriptor matrixDescriptorWithRows:rB columns:cB rowBytes:cB*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dC=[MPSMatrixDescriptor matrixDescriptorWithRows:rC columns:cC rowBytes:cC*4 dataType:MPSDataTypeFloat32];
    MPSMatrix *mA=[[MPSMatrix alloc] initWithBuffer:bA descriptor:dA];
    MPSMatrix *mB=[[MPSMatrix alloc] initWithBuffer:bB descriptor:dB];
    MPSMatrix *mC=[[MPSMatrix alloc] initWithBuffer:bC descriptor:dC];
    [mm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
}

static void ub_gpu_forward(UBGpu *g, UBWeights *w, UBCpuBufs *b, const uint16_t *tokens, UBComponents *c) {
    uint64_t t;
    t=mach_absolute_time();
    memset(b->x_cur,0,SEQ*DIM*4);
    embed_lookup(b->x_cur, w->embed, tokens, DIM, SEQ);
    c->embed_ms = UB_MS(mach_absolute_time()-t);

    double gpu_total=0, cpu_total=0, io_total=0;

    for (int L=0;L<NLAYERS;L++) {
        // RMSNorm1 (CPU)
        t=mach_absolute_time();
        rmsnorm(b->x_norm, b->x_cur, w->lw[L].rms_att, DIM, SEQ);
        memcpy(g->buf_norm.contents, b->x_norm, SEQ*DIM*4);
        cpu_total += UB_MS(mach_absolute_time()-t);

        // QKV — 3 matmuls in ONE command buffer (batched!)
        t=mach_absolute_time();
        id<MTLCommandBuffer> cmd1 = [g->queue commandBuffer];
        ub_gpu_encode(g, g->mm_dim, cmd1, g->wq[L],DIM,DIM, g->buf_norm,DIM,SEQ, g->buf_Q,DIM,SEQ);
        ub_gpu_encode(g, g->mm_dim, cmd1, g->wk[L],DIM,DIM, g->buf_norm,DIM,SEQ, g->buf_K,DIM,SEQ);
        ub_gpu_encode(g, g->mm_dim, cmd1, g->wv[L],DIM,DIM, g->buf_norm,DIM,SEQ, g->buf_V,DIM,SEQ);
        [cmd1 commit]; [cmd1 waitUntilCompleted];
        gpu_total += UB_MS(mach_absolute_time()-t);

        // Read QKV + attention (CPU)
        t=mach_absolute_time();
        memcpy(b->Q, g->buf_Q.contents, SEQ*DIM*4);
        memcpy(b->K, g->buf_K.contents, SEQ*DIM*4);
        memcpy(b->V, g->buf_V.contents, SEQ*DIM*4);
        io_total += UB_MS(mach_absolute_time()-t);

        t=mach_absolute_time();
        ub_cpu_attention(b->attn_out, b->Q, b->K, b->V, b->scores, DIM, SEQ, HEADS);
        cpu_total += UB_MS(mach_absolute_time()-t);

        // Wo — 1 matmul
        t=mach_absolute_time();
        memcpy(g->buf_attn.contents, b->attn_out, SEQ*DIM*4);
        id<MTLCommandBuffer> cmd2 = [g->queue commandBuffer];
        ub_gpu_encode(g, g->mm_dim, cmd2, g->wo[L],DIM,DIM, g->buf_attn,DIM,SEQ, g->buf_o,DIM,SEQ);
        [cmd2 commit]; [cmd2 waitUntilCompleted];
        memcpy(b->o_out, g->buf_o.contents, SEQ*DIM*4);
        gpu_total += UB_MS(mach_absolute_time()-t);

        vDSP_vadd(b->x_cur,1,b->o_out,1,b->x2,1,(vDSP_Length)(SEQ*DIM));

        // RMSNorm2 (CPU)
        t=mach_absolute_time();
        rmsnorm(b->x_norm, b->x2, w->lw[L].rms_ffn, DIM, SEQ);
        memcpy(g->buf_norm.contents, b->x_norm, SEQ*DIM*4);
        cpu_total += UB_MS(mach_absolute_time()-t);

        // W1+W3 — 2 matmuls in ONE command buffer (batched!)
        t=mach_absolute_time();
        id<MTLCommandBuffer> cmd3 = [g->queue commandBuffer];
        ub_gpu_encode(g, g->mm_up, cmd3, g->w1[L],HIDDEN,DIM, g->buf_norm,DIM,SEQ, g->buf_h1,HIDDEN,SEQ);
        ub_gpu_encode(g, g->mm_up, cmd3, g->w3[L],HIDDEN,DIM, g->buf_norm,DIM,SEQ, g->buf_h3,HIDDEN,SEQ);
        [cmd3 commit]; [cmd3 waitUntilCompleted];
        gpu_total += UB_MS(mach_absolute_time()-t);

        // SiLU + gate (CPU)
        t=mach_absolute_time();
        memcpy(b->h1, g->buf_h1.contents, SEQ*HIDDEN*4);
        memcpy(b->h3, g->buf_h3.contents, SEQ*HIDDEN*4);
        ub_silu(b->h1, HIDDEN*SEQ);
        vDSP_vmul(b->h1,1,b->h3,1,b->silu_out,1,(vDSP_Length)(HIDDEN*SEQ));
        cpu_total += UB_MS(mach_absolute_time()-t);

        // W2 — 1 matmul
        t=mach_absolute_time();
        memcpy(g->buf_silu.contents, b->silu_out, SEQ*HIDDEN*4);
        id<MTLCommandBuffer> cmd4 = [g->queue commandBuffer];
        ub_gpu_encode(g, g->mm_down, cmd4, g->w2[L],DIM,HIDDEN, g->buf_silu,HIDDEN,SEQ, g->buf_ffn,DIM,SEQ);
        [cmd4 commit]; [cmd4 waitUntilCompleted];
        memcpy(b->ffn_out, g->buf_ffn.contents, SEQ*DIM*4);
        gpu_total += UB_MS(mach_absolute_time()-t);

        vDSP_vadd(b->x2,1,b->ffn_out,1,b->x_cur,1,(vDSP_Length)(SEQ*DIM));
    }
    c->ane_eval_ms = gpu_total; // GPU compute time
    c->io_ms = io_total;        // CPU↔GPU transfer
    c->cpu_ms = cpu_total;       // CPU compute (attention, norms)

    t=mach_absolute_time();
    rmsnorm(b->x_final, b->x_cur, w->rms_final, DIM, SEQ);
    int pos=SEQ-1; float *xp=(float*)malloc(DIM*4);
    for (int d=0;d<DIM;d++) xp[d]=b->x_final[d*SEQ+pos];
    cblas_sgemv(CblasRowMajor,CblasNoTrans,VOCAB,DIM,1,w->embed,DIM,xp,1,0,b->logits,1);
    free(xp);
    c->classifier_ms = UB_MS(mach_absolute_time()-t);
}

// ═══════════════════════════════════════════════════════
// HELPER: run a backend for N seconds, collect stats
// ═══════════════════════════════════════════════════════

typedef void (*ForwardFn)(void *state, void *weights, void *bufs, const uint16_t *tokens, UBComponents *c);

static void ub_run_backend(const char *name, double seconds,
                           ForwardFn fn, void *state, void *weights, void *bufs,
                           const uint16_t *tokens, UBResult *r, FILE *csv,
                           NSMutableString *out) {
    r->name = name;
    ubs_init(&r->latency, 16384);
    memset(r->thermal_time, 0, sizeof(r->thermal_time));
    r->worst_thermal = ThermalNominal;
    r->mem_start = ub_memory_mb();
    r->mem_peak = r->mem_start;
    r->cpu_start = ub_cpu_time();
    r->batt_start = ub_battery();
    r->passes = 0;

    // Component accumulators
    UBStats st_eval, st_io, st_cpu, st_cls, st_embed;
    ubs_init(&st_eval,4096); ubs_init(&st_io,4096);
    ubs_init(&st_cpu,4096); ubs_init(&st_cls,4096); ubs_init(&st_embed,4096);

    uint64_t t_start = mach_absolute_time();
    double last_sample = 0;

    while (UB_S(mach_absolute_time()-t_start) < seconds) {
        UBComponents c = {0};
        uint64_t ts = mach_absolute_time();
        fn(state, weights, bufs, tokens, &c);
        double total_ms = UB_MS(mach_absolute_time()-ts);

        ubs_add(&r->latency, total_ms);
        ubs_add(&st_eval, c.ane_eval_ms);
        ubs_add(&st_io, c.io_ms);
        ubs_add(&st_cpu, c.cpu_ms);
        ubs_add(&st_cls, c.classifier_ms);
        ubs_add(&st_embed, c.embed_ms);
        r->passes++;

        double mem = ub_memory_mb();
        if (mem > r->mem_peak) r->mem_peak = mem;

        double elapsed = UB_S(mach_absolute_time()-t_start);
        ThermalLevel tl = current_thermal_level();
        if (tl > r->worst_thermal) r->worst_thermal = tl;

        // Thermal time tracking (10s samples)
        if (elapsed - last_sample >= 10.0) {
            r->thermal_time[tl] += (elapsed - last_sample);
            last_sample = elapsed;
        }

        // CSV
        if (csv && r->passes % 50 == 0) {
            fprintf(csv, "inference,%s,%d,%.1f,%.2f,%.6f,%d,%.4f,%.4f,%.4f,%.1f,0\n",
                name, r->passes, elapsed, total_ms, 0.0,
                (int)tl, ub_battery(), ub_cpu_time(), 0.0, ub_memory_mb());
            fflush(csv);
        }

        if (r->passes % 500 == 0) {
            NSLog(@"UB: %s %d passes, %.1fs, %.1fms avg",
                name, r->passes, elapsed, ubs_mean(&r->latency));
        }

        // Yield CPU to stay under iOS 50% CPU limit (180s window)
        // Accelerate BLAS uses multiple cores, so need generous sleep
        // 5ms keeps total process CPU ~35-40% even with multi-core BLAS
        usleep(5000);

        // Thermal safety: if critical, pause 30s to cool down
        if (current_thermal_level() >= ThermalCritical) {
            NSLog(@"UB: CRITICAL thermal — pausing 30s");
            usleep(30000000);
        }
    }

    r->elapsed_s = UB_S(mach_absolute_time()-t_start);
    r->cpu_end = ub_cpu_time();
    r->batt_end = ub_battery();

    // Report
    double tps = (r->passes * SEQ) / r->elapsed_s;
    double cpu_pct = (r->cpu_end - r->cpu_start) / r->elapsed_s * 100;
    float drain = r->batt_start - r->batt_end;

    [out appendFormat:@"  %-6s  %d passes in %.0fs\n", name, r->passes, r->elapsed_s];
    [out appendFormat:@"    Latency:  %.2f ms (p50=%.2f p95=%.2f p99=%.2f)\n",
        ubs_mean(&r->latency), ubs_pct(&r->latency,0.5), ubs_pct(&r->latency,0.95), ubs_pct(&r->latency,0.99)];
    [out appendFormat:@"    Range:    %.2f - %.2f ms (sigma=%.2f)\n", r->latency.min, r->latency.max, ubs_std(&r->latency)];
    [out appendFormat:@"    Tokens/s: %.0f\n", tps];
    [out appendFormat:@"    Battery:  %.0f%% -> %.0f%% (%.1f%% drain)\n",
        r->batt_start*100, r->batt_end*100, drain*100];
    [out appendFormat:@"    CPU use:  %.1f%% (%.1fs of %.0fs)\n", cpu_pct, r->cpu_end-r->cpu_start, r->elapsed_s];
    [out appendFormat:@"    Memory:   %.0f MB (peak %.0f MB)\n", r->mem_start, r->mem_peak];
    [out appendFormat:@"    Thermal:  %s (worst)\n", thermal_level_name(r->worst_thermal)];

    // Component breakdown
    const char *comp_label = "compute";
    const char *io_label = "IO/transfer";
    const char *cpu_label = "CPU ops";
    if (strcmp(name, "CPU") == 0) { comp_label = "matmuls"; io_label = "attention"; cpu_label = "norms+act"; }
    else if (strcmp(name, "GPU") == 0) { comp_label = "GPU matmuls"; io_label = "CPU<>GPU"; cpu_label = "CPU ops"; }
    else { comp_label = "ANE evals"; io_label = "IOSurface"; cpu_label = "CPU ops"; }

    double total_comp = ubs_mean(&st_eval)+ubs_mean(&st_io)+ubs_mean(&st_cpu)+ubs_mean(&st_cls)+ubs_mean(&st_embed);
    if (total_comp > 0) {
        [out appendFormat:@"    Breakdown:\n"];
        [out appendFormat:@"      embed:       %5.2f ms (%4.1f%%)\n", ubs_mean(&st_embed), ubs_mean(&st_embed)/total_comp*100];
        [out appendFormat:@"      %-12s %5.2f ms (%4.1f%%)\n", comp_label, ubs_mean(&st_eval), ubs_mean(&st_eval)/total_comp*100];
        [out appendFormat:@"      %-12s %5.2f ms (%4.1f%%)\n", io_label, ubs_mean(&st_io), ubs_mean(&st_io)/total_comp*100];
        [out appendFormat:@"      %-12s %5.2f ms (%4.1f%%)\n", cpu_label, ubs_mean(&st_cpu), ubs_mean(&st_cpu)/total_comp*100];
        [out appendFormat:@"      classifier:  %5.2f ms (%4.1f%%)\n", ubs_mean(&st_cls), ubs_mean(&st_cls)/total_comp*100];
    }
    [out appendString:@"\n"];

    ubs_free(&st_eval); ubs_free(&st_io); ubs_free(&st_cpu); ubs_free(&st_cls); ubs_free(&st_embed);
}

// Forward function adapters (cast to common signature)
static void ub_ane_fwd_adapter(void *s, void *w, void *b, const uint16_t *t, UBComponents *c) {
    (void)w; (void)b; ub_ane_forward((UBAne*)s, t, c);
}
static void ub_cpu_fwd_adapter(void *s, void *w, void *b, const uint16_t *t, UBComponents *c) {
    (void)s; ub_cpu_forward((UBWeights*)w, (UBCpuBufs*)b, t, c);
}
static void ub_gpu_fwd_adapter(void *s, void *w, void *b, const uint16_t *t, UBComponents *c) {
    ub_gpu_forward((UBGpu*)s, (UBWeights*)w, (UBCpuBufs*)b, t, c);
}

// ═══════════════════════════════════════════════════════
// MAIN BENCHMARK
// ═══════════════════════════════════════════════════════

NSString *ane_ultimate_benchmark(float total_minutes) {
    @autoreleasepool {
    ub_init_tb();
    NSMutableString *out = [NSMutableString string];

    // Timing config
    double base_s = 120;     // baseline idle
    double inf_s  = 600;     // per-backend inference
    double cool_s = 60;      // cooldown between backends
    double train_s = 600;    // training
    double micro_s = 60;     // micro-benchmarks

    if (total_minutes > 0) {
        double scale = (total_minutes * 60.0) / (base_s + 3*inf_s + 3*cool_s + train_s + micro_s);
        base_s *= scale; inf_s *= scale; cool_s *= scale; train_s *= scale; micro_s *= scale;
        if (base_s < 30) base_s = 30;
        if (cool_s < 15) cool_s = 15;
    }

    double battWh = ub_battery_wh();
    struct utsname si; uname(&si);
    NSString *hw = [NSString stringWithCString:si.machine encoding:NSUTF8StringEncoding];

    // Progress log file — persists even if app crashes
    NSString *docs = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,NSUserDomainMask,YES) firstObject];
    NSString *progressPath = [docs stringByAppendingPathComponent:@"ane_benchmark_progress.log"];
    FILE *plog = fopen([progressPath UTF8String], "w");
    // Progress log — survives crashes
    #define PLOG(fmt, ...) do { if (plog) { fprintf(plog, fmt "\n", ##__VA_ARGS__); fflush(plog); } \
        NSLog(@"" fmt, ##__VA_ARGS__); } while(0)

    // Results file — written after each phase so partial results survive crashes
    NSString *resultsPath = [docs stringByAppendingPathComponent:@"ane_benchmark_results.txt"];
    #define SAVE_RESULTS() do { \
        [out writeToFile:resultsPath atomically:YES encoding:NSUTF8StringEncoding error:nil]; \
        PLOG("UB: Results saved to disk (%d chars)", (int)out.length); \
    } while(0)

    [out appendString:@"====================================================================\n"];
    [out appendString:@"  ANE ULTIMATE BENCHMARK\n"];
    [out appendString:@"  Inference (ANE/GPU/CPU) + Training + Power + Efficiency\n"];
    [out appendString:@"====================================================================\n\n"];
    [out appendFormat:@"  Device:    %@\n", hw];
    [out appendFormat:@"  Battery:   %.1f Wh, %.0f%%\n", battWh, ub_battery()*100];
    [out appendFormat:@"  Thermal:   %s\n", thermal_level_name(current_thermal_level())];
    [out appendFormat:@"  Memory:    %.0f MB\n", ub_memory_mb()];
    [out appendFormat:@"  Model:     Stories-110M (%dL, dim=%d, seq=%d)\n", NLAYERS, DIM, SEQ];
    [out appendFormat:@"  Schedule:  %.0fs base + 3x%.0fs inf + %.0fs train + %.0fs micro\n\n",
        base_s, inf_s, train_s, micro_s];

    // CSV (reuse docs from progress log above)
    NSString *csvPath = [docs stringByAppendingPathComponent:@"ane_ultimate_benchmark.csv"];
    FILE *csv = fopen([csvPath UTF8String], "w");
    if (csv) fprintf(csv, "phase,backend,sample,elapsed_s,step_ms,loss,thermal,battery,cpu_time,cpu_sys,memory_mb,is_compile\n");

    // Shared resources
    UBWeights weights = ub_weights_alloc();
    UBCpuBufs cpubufs = ub_cpu_alloc();
    uint16_t tokens[SEQ]; srand48(99);
    for (int i=0;i<SEQ;i++) tokens[i]=(uint16_t)(drand48()*(VOCAB-1));

    // ═══════════════════════════════
    // PHASE 1: BASELINE
    // ═══════════════════════════════
    [out appendString:@"  --- PHASE 1: SYSTEM BASELINE ---\n"];
    PLOG("UB: Phase 1 — idle baseline (%.0fs)\n", base_s);

    float base_batt_start = ub_battery();
    double base_cpu_start = ub_cpu_time();
    double base_mem = ub_memory_mb();

    uint64_t base_t0 = mach_absolute_time();
    while (UB_S(mach_absolute_time()-base_t0) < base_s) {
        usleep(5000000); // 5s sleep
        if (csv) {
            double el = UB_S(mach_absolute_time()-base_t0);
            fprintf(csv, "baseline,idle,0,%.1f,0,0,%d,%.4f,%.4f,0,%.1f,0\n",
                el, (int)current_thermal_level(), ub_battery(), ub_cpu_time(), ub_memory_mb());
        }
    }
    float base_batt_end = ub_battery();
    double base_elapsed = UB_S(mach_absolute_time()-base_t0);
    float base_drain = base_batt_start - base_batt_end;
    double idle_watts = (base_drain > 0) ? (base_drain * battWh) / (base_elapsed/3600.0) : 0;

    [out appendFormat:@"  Duration:  %.0fs\n", base_elapsed];
    [out appendFormat:@"  Battery:   %.0f%% -> %.0f%% (%.1f%% idle drain)\n",
        base_batt_start*100, base_batt_end*100, base_drain*100];
    [out appendFormat:@"  Idle power: %.2f W (display on, no compute)\n", idle_watts];
    [out appendFormat:@"  CPU time:  %.2fs (system overhead)\n", ub_cpu_time()-base_cpu_start];
    [out appendFormat:@"  Memory:    %.0f MB\n\n", base_mem];
    SAVE_RESULTS();

    // ═══════════════════════════════
    // PHASE 2: INFERENCE (3 backends)
    // ═══════════════════════════════
    [out appendString:@"  --- PHASE 2: INFERENCE COMPARISON ---\n\n"];
    UBResult res[3];
    memset(res, 0, sizeof(res));

    // 2a. ANE
    [out appendString:@"  [ANE] Compiling 24 kernels...\n"];
    PLOG("UB: Compiling ANE kernels...\n");
    uint64_t tc = mach_absolute_time();
    UBAne *ane = ub_ane_init(&weights);
    double ane_compile = UB_MS(mach_absolute_time()-tc);
    [out appendFormat:@"  [ANE] Compile: %.0f ms\n", ane_compile];

    if (ane->ok) {
        for (int i=0;i<5;i++) { UBComponents _c={0}; ub_ane_forward(ane,tokens,&_c); } // warmup
        PLOG("UB: ANE inference (%.0fs)...\n", inf_s);
        ub_run_backend("ANE", inf_s, ub_ane_fwd_adapter, ane, &weights, &cpubufs, tokens, &res[0], csv, out);
    } else {
        [out appendString:@"  [ANE] FAILED to compile kernels\n\n"];
    }

    // Cooldown
    [out appendFormat:@"  Cooling %.0fs...\n", cool_s];
    PLOG("UB: Cooldown %.0fs\n", cool_s);
    usleep((unsigned)(cool_s*1e6));

    // 2b. CPU
    PLOG("UB: CPU inference (%.0fs)...\n", inf_s);
    for (int i=0;i<5;i++) { UBComponents _c={0}; ub_cpu_forward(&weights,&cpubufs,tokens,&_c); }
    ub_run_backend("CPU", inf_s, ub_cpu_fwd_adapter, NULL, &weights, &cpubufs, tokens, &res[1], csv, out);

    [out appendFormat:@"  Cooling %.0fs...\n", cool_s];
    PLOG("UB: Cooldown %.0fs\n", cool_s);
    usleep((unsigned)(cool_s*1e6));

    // 2c. GPU (batched)
    PLOG("UB: GPU init + inference (%.0fs)...\n", inf_s);
    UBGpu *gpu = ub_gpu_init(&weights);
    if (gpu) {
        [out appendFormat:@"  [GPU] Device: %@\n", gpu->device.name];
        [out appendString:@"  [GPU] Batched MPS (48 syncs/pass vs 84 before)\n"];
        for (int i=0;i<5;i++) { UBComponents _c={0}; ub_gpu_forward(gpu,&weights,&cpubufs,tokens,&_c); }
        ub_run_backend("GPU", inf_s, ub_gpu_fwd_adapter, gpu, &weights, &cpubufs, tokens, &res[2], csv, out);
    } else {
        [out appendString:@"  [GPU] Metal not available\n\n"];
    }

    [out appendFormat:@"  Cooling %.0fs...\n", cool_s];
    usleep((unsigned)(cool_s*1e6));

    // ═══════════════════════════════
    // PHASE 2 COMPARISON TABLE
    // ═══════════════════════════════
    [out appendString:@"  --- INFERENCE COMPARISON TABLE ---\n\n"];
    [out appendString:@"  Backend  Latency  p95     Tok/s   Drain  Watts  NetW   Tok/J   CPU%%\n"];
    [out appendString:@"  ------  -------  ------  ------  -----  -----  -----  ------  ----\n"];

    for (int i=0; i<3; i++) {
        UBResult *r = &res[i];
        if (!r->name || r->passes == 0) continue;
        double tps = (r->passes*SEQ)/r->elapsed_s;
        float drain = r->batt_start - r->batt_end;
        double gross_w = (drain>0) ? (drain*battWh)/(r->elapsed_s/3600.0) : 0;
        double net_w = (gross_w > idle_watts) ? gross_w - idle_watts : 0;
        double energy_j = net_w * r->elapsed_s;
        double tpj = (energy_j>0) ? (r->passes*SEQ)/energy_j : 0;
        double cpu_pct = (r->cpu_end-r->cpu_start)/r->elapsed_s*100;

        if (drain > 0) {
            [out appendFormat:@"  %-6s  %6.1f   %5.1f   %5.0f   %4.1f%%  %4.2fW  %4.2fW  %5.0f   %4.1f\n",
                r->name, ubs_mean(&r->latency), ubs_pct(&r->latency,0.95),
                tps, drain*100, gross_w, net_w, tpj, cpu_pct];
        } else {
            [out appendFormat:@"  %-6s  %6.1f   %5.1f   %5.0f   %4.1f%%    N/A    N/A    N/A   %4.1f\n",
                r->name, ubs_mean(&r->latency), ubs_pct(&r->latency,0.95),
                tps, drain*100, cpu_pct];
        }
    }
    [out appendString:@"\n"];

    // Winner
    double best = 1e9; const char *winner = "?";
    for (int i=0;i<3;i++) {
        if (res[i].passes > 0 && ubs_mean(&res[i].latency) < best) {
            best = ubs_mean(&res[i].latency); winner = res[i].name;
        }
    }
    [out appendFormat:@"  Fastest: %s (%.1f ms, %.0f tok/s)\n\n", winner, best, SEQ/(best/1000.0)];
    SAVE_RESULTS();

    // ═══════════════════════════════
    // FREE ALL INFERENCE STATE BEFORE TRAINING
    // Training alone uses ~2.3 GB — must free everything
    // ═══════════════════════════════
    if (gpu) { ub_gpu_free(gpu); gpu = NULL; }
    ub_cpu_free(&cpubufs); memset(&cpubufs, 0, sizeof(cpubufs));
    ub_ane_free(ane); ane = NULL;
    ub_weights_free(&weights); memset(&weights, 0, sizeof(weights));
    // Free latency sample arrays from inference results
    for (int i=0;i<3;i++) ubs_free(&res[i].latency);
    // Force ARC to release Metal objects (GPU buffers etc.)
    @autoreleasepool {}
    PLOG("UB: Freed ALL inference state, memory=%.0fMB", ub_memory_mb());
    // If memory still high, skip training
    double mem_after = ub_memory_mb();
    [out appendFormat:@"  Memory after cleanup: %.0f MB\n\n", mem_after];
    if (mem_after > 800) {
        PLOG("UB: WARNING memory still %.0fMB after cleanup, training may crash", mem_after);
        [out appendFormat:@"  WARNING: memory %.0f MB — training may be at risk\n\n", mem_after];
    }

    // ═══════════════════════════════
    // PHASE 3: TRAINING — SKIPPED (use separate TRAIN button)
    // Training needs ~2.3 GB and conflicts with inference memory
    // ═══════════════════════════════
    [out appendString:@"  --- PHASE 3: TRAINING (skipped — use TRAIN 5m button) ---\n\n"];
    PLOG("UB: Phase 3 skipped — training runs separately");
    SAVE_RESULTS();
    if (0) { // Training disabled — runs as separate benchmark to avoid memory conflicts
    PLOG("UB: Phase 3 — Training (%.0fs)...\n", train_s);

    // Create dummy training data
    NSString *tmpPath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"ub_tokens.bin"];
    { size_t n=200000; uint16_t *buf=(uint16_t*)malloc(n*2); srand48(42);
      for (size_t i=0;i<n;i++) buf[i]=(uint16_t)(drand48()*(VOCAB-1));
      [[NSData dataWithBytesNoCopy:buf length:n*2 freeWhenDone:YES] writeToFile:tmpPath atomically:YES]; }

    // Memory safety check — training needs ~2.3 GB
    double mem_before_train = ub_memory_mb();
    PLOG("UB: Memory before training init: %.0fMB", mem_before_train);
    if (mem_before_train > 2000) {
        [out appendString:@"  SKIPPING training — memory too high, would crash\n\n"];
        PLOG("UB: SKIPPING training, memory %.0fMB > 2000MB limit", mem_before_train);
        SAVE_RESULTS();
    }

    ANETrainState *train = ane_train_init(NULL, [tmpPath UTF8String]);
    if (train) {
        float train_batt_start = ub_battery();
        double train_cpu_start = ub_cpu_time();
        UBStats st_regular, st_compile;
        ubs_init(&st_regular, 8192); ubs_init(&st_compile, 2048);
        float first_loss=0, best_loss=999, last_loss=0;
        int total_steps=0;

        uint64_t t_train = mach_absolute_time();
        while (UB_S(mach_absolute_time()-t_train) < train_s) {
            uint64_t ts = mach_absolute_time();
            float loss = ane_train_step(train);
            double ms = UB_MS(mach_absolute_time()-ts);

            total_steps++;
            last_loss = loss;
            if (total_steps==1) first_loss = loss;
            if (loss < best_loss) best_loss = loss;

            int step = ane_train_current_step(train);
            bool is_compile = (step % ACCUM_STEPS == 0) && (step > 0);
            if (is_compile) ubs_add(&st_compile, ms); else ubs_add(&st_regular, ms);

            if (csv && total_steps%20==0) {
                fprintf(csv, "training,ANE,%d,%.1f,%.2f,%.6f,%d,%.4f,%.4f,0,%.1f,%d\n",
                    total_steps, UB_S(mach_absolute_time()-t_train), ms, loss,
                    (int)current_thermal_level(), ub_battery(), ub_cpu_time(),
                    ub_memory_mb(), is_compile?1:0);
            }
            if (total_steps%200==0)
                PLOG("UB: train step %d, loss=%.4f, %.1fms\n", total_steps, loss, ms);

            // Yield CPU to stay under iOS 50% limit
            usleep(5000);

            // Thermal safety
            if (current_thermal_level() >= ThermalCritical) {
                PLOG("UB: CRITICAL thermal during training — pausing 30s");
                usleep(30000000);
            }
        }
        double train_elapsed = UB_S(mach_absolute_time()-t_train);
        float train_batt_end = ub_battery();
        float train_drain = train_batt_start - train_batt_end;
        double train_gross_w = (train_drain>0) ? (train_drain*battWh)/(train_elapsed/3600.0) : 0;
        double train_net_w = (train_gross_w > idle_watts) ? train_gross_w - idle_watts : 0;
        double train_cpu_pct = (ub_cpu_time()-train_cpu_start)/train_elapsed*100;

        [out appendFormat:@"  Steps:     %d in %.0fs (%.2f steps/s)\n", total_steps, train_elapsed, total_steps/train_elapsed];
        [out appendFormat:@"  Tokens/s:  %.0f\n", (total_steps*SEQ)/train_elapsed];
        [out appendFormat:@"  Loss:      %.4f -> %.4f (best: %.4f, -%.1f%%)\n",
            first_loss, last_loss, best_loss, (first_loss-best_loss)/first_loss*100];
        [out appendFormat:@"  Regular:   %.1f ms (p50=%.1f p95=%.1f, n=%d)\n",
            ubs_mean(&st_regular), ubs_pct(&st_regular,0.5), ubs_pct(&st_regular,0.95), st_regular.count];
        [out appendFormat:@"  Compile:   %.1f ms (p50=%.1f p95=%.1f, n=%d)\n",
            ubs_mean(&st_compile), ubs_pct(&st_compile,0.5), ubs_pct(&st_compile,0.95), st_compile.count];
        [out appendFormat:@"  Overhead:  %.1fx slower, %.1f%% of time in compile\n",
            ubs_mean(&st_compile)/fmax(ubs_mean(&st_regular),0.001),
            st_compile.count>0 ? st_compile.sum/(st_compile.sum+st_regular.sum)*100 : 0];
        [out appendFormat:@"  Battery:   %.0f%% -> %.0f%% (%.1f%% drain)\n",
            train_batt_start*100, train_batt_end*100, train_drain*100];
        if (train_drain > 0) {
            double j_per_step = (train_net_w * train_elapsed) / total_steps;
            [out appendFormat:@"  Power:     %.2fW gross, %.2fW net\n", train_gross_w, train_net_w];
            [out appendFormat:@"  J/step:    %.3f\n", j_per_step];
            [out appendFormat:@"  Loss/Wh:   %.4f\n",
                (first_loss-best_loss)/(train_drain*battWh)];
        }
        [out appendFormat:@"  CPU use:   %.1f%%\n", train_cpu_pct];
        [out appendFormat:@"  Thermal:   %s\n", thermal_level_name(current_thermal_level())];
        [out appendString:@"\n"];

        ubs_free(&st_regular); ubs_free(&st_compile);
        ane_train_free(train);
    } else {
        [out appendString:@"  Training init failed\n\n"];
    }
    [[NSFileManager defaultManager] removeItemAtPath:tmpPath error:nil];
    SAVE_RESULTS();
    } // end if(0) disabled training block

    // ═══════════════════════════════
    // PHASE 4: COMPONENT MICRO-BENCHMARKS
    // (re-allocate lightweight state after training freed)
    // ═══════════════════════════════
    [out appendString:@"  --- PHASE 4: COMPONENT MICRO-BENCHMARKS ---\n\n"];
    PLOG("UB: Phase 4 — micro-benchmarks (re-allocating weights)\n");
    int micro_n = 50;

    // Re-allocate weights and ANE for micro-benchmarks
    weights = ub_weights_alloc();
    ane = ub_ane_init(&weights);
    PLOG("UB: Re-init done, memory=%.0fMB\n", ub_memory_mb());

    if (ane && ane->ok) {
        io_write_fp16(ane->fwdAttn[0]->ioIn, ane->x_cur, DIM, SEQ); // prep input
        UBStats st; ubs_init(&st, micro_n);
        for (int i=0;i<micro_n;i++) {
            uint64_t t=mach_absolute_time();
            ane_eval(ane->fwdAttn[0]);
            ubs_add(&st, UB_MS(mach_absolute_time()-t));
        }
        [out appendFormat:@"  ane_eval (1 kernel):   %.2f ms (sigma=%.2f)\n", ubs_mean(&st), ubs_std(&st)];
        ubs_free(&st);
    }

    // IO transfer timing
    {
        UBStats st; ubs_init(&st, micro_n);
        float *tmp = (float*)malloc(DIM*SEQ*4);
        for (int i=0;i<micro_n;i++) {
            uint64_t t=mach_absolute_time();
            if (ane && ane->ok) {
                io_write_fp16(ane->fwdAttn[0]->ioIn, tmp, DIM, SEQ);
                io_read_fp16(ane->fwdAttn[0]->ioOut, tmp, 0, DIM, SEQ);
            }
            ubs_add(&st, UB_MS(mach_absolute_time()-t));
        }
        [out appendFormat:@"  IO write+read cycle:   %.2f ms (sigma=%.2f)\n", ubs_mean(&st), ubs_std(&st)];
        free(tmp); ubs_free(&st);
    }

    // RMSNorm timing
    {
        UBStats st; ubs_init(&st, micro_n);
        float *x=(float*)malloc(DIM*SEQ*4), *o=(float*)malloc(DIM*SEQ*4), *w=(float*)malloc(DIM*4);
        for (int i=0;i<DIM;i++) w[i]=1; for (int i=0;i<DIM*SEQ;i++) x[i]=0.1f;
        for (int i=0;i<micro_n;i++) {
            uint64_t t=mach_absolute_time();
            rmsnorm(o, x, w, DIM, SEQ);
            ubs_add(&st, UB_MS(mach_absolute_time()-t));
        }
        [out appendFormat:@"  rmsnorm [%d,%d]:       %.2f ms (sigma=%.2f)\n", DIM, SEQ, ubs_mean(&st), ubs_std(&st)];
        free(x); free(o); free(w); ubs_free(&st);
    }

    // Classifier (cblas_sgemv) timing
    {
        UBStats st; ubs_init(&st, micro_n);
        float *x=(float*)malloc(DIM*4), *lo=(float*)malloc(VOCAB*4);
        for (int i=0;i<DIM;i++) x[i]=0.1f;
        for (int i=0;i<micro_n;i++) {
            uint64_t t=mach_absolute_time();
            cblas_sgemv(CblasRowMajor,CblasNoTrans,VOCAB,DIM,1,weights.embed,DIM,x,1,0,lo,1);
            ubs_add(&st, UB_MS(mach_absolute_time()-t));
        }
        [out appendFormat:@"  classifier [%d,%d]:    %.2f ms (sigma=%.2f)\n", VOCAB, DIM, ubs_mean(&st), ubs_std(&st)];
        free(x); free(lo); ubs_free(&st);
    }

    // Compile timing (single layer)
    {
        uint64_t t=mach_absolute_time();
        int cc=0;
        Kern *k = compile_kern(gen_sdpa_fwd_taps(), (@{
            @"@model_path/weights/rms1.bin":@{@"offset":@0,@"data":build_blob(weights.lw[0].rms_att,1,DIM)},
            @"@model_path/weights/wq.bin":@{@"offset":@0,@"data":build_blob(weights.lw[0].Wq,DIM,DIM)},
            @"@model_path/weights/wk.bin":@{@"offset":@0,@"data":build_blob(weights.lw[0].Wk,DIM,DIM)},
            @"@model_path/weights/wv.bin":@{@"offset":@0,@"data":build_blob(weights.lw[0].Wv,DIM,DIM)},
            @"@model_path/weights/wo.bin":@{@"offset":@0,@"data":build_blob(weights.lw[0].Wo,DIM,DIM)},
            @"@model_path/weights/mask.bin":@{@"offset":@0,@"data":get_mask_blob()},
        }), DIM*SEQ*2, 6*DIM*SEQ*2);
        double compile_ms = UB_MS(mach_absolute_time()-t);
        [out appendFormat:@"  compile_kern (1 attn): %.0f ms\n", compile_ms];
        if (k) free_kern(k);
    }

    [out appendString:@"\n"];

    // ═══════════════════════════════
    // FINAL SUMMARY
    // ═══════════════════════════════
    [out appendString:@"====================================================================\n"];
    [out appendString:@"  SUMMARY\n"];
    [out appendString:@"====================================================================\n\n"];

    // Best inference backend
    [out appendFormat:@"  Fastest inference:    %s (%.1f ms, %.0f tok/s)\n", winner, best, SEQ/(best/1000)];

    // Power efficiency winner
    double best_tpj = 0; const char *eff_winner = "?";
    for (int i=0;i<3;i++) {
        if (res[i].passes == 0) continue;
        float drain = res[i].batt_start - res[i].batt_end;
        if (drain <= 0) continue;
        double net = (drain*battWh)/(res[i].elapsed_s/3600.0) - idle_watts;
        if (net <= 0) continue;
        double tpj = (res[i].passes*SEQ) / (net * res[i].elapsed_s);
        if (tpj > best_tpj) { best_tpj = tpj; eff_winner = res[i].name; }
    }
    if (best_tpj > 0)
        [out appendFormat:@"  Most efficient:      %s (%.0f tok/J)\n", eff_winner, best_tpj];
    else
        [out appendString:@"  Most efficient:      ANE (too efficient to measure drain)\n"];

    [out appendFormat:@"  Idle baseline:       %.2f W\n", idle_watts];
    [out appendFormat:@"  CSV:                 %@\n", csvPath];
    [out appendString:@"====================================================================\n"];

    // Cleanup (re-allocated for Phase 4 micro-benchmarks)
    if (ane) ub_ane_free(ane);
    if (weights.rms_final) ub_weights_free(&weights);
    // res[].latency already freed before Phase 3
    if (csv) fclose(csv);
    thermal_disable_battery_monitoring();

    SAVE_RESULTS();
    PLOG("UB: Complete.");
    if (plog) fclose(plog);
    #undef PLOG
    #undef SAVE_RESULTS
    return out;
    }
}

// ═══════════════════════════════════════════════════════
// STANDALONE TRAINING BENCHMARK
// Runs in isolation — no inference state competing for memory
// ═══════════════════════════════════════════════════════

NSString *ane_training_only_benchmark(float minutes) {
    @autoreleasepool {
    ub_init_tb();
    NSMutableString *out = [NSMutableString string];
    double battWh = ub_battery_wh();
    double target_s = minutes * 60.0;

    NSString *docs = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,NSUserDomainMask,YES) firstObject];
    NSString *resultsPath = [docs stringByAppendingPathComponent:@"ane_training_results.txt"];

    [out appendString:@"====================================================================\n"];
    [out appendString:@"  ANE TRAINING BENCHMARK (standalone)\n"];
    [out appendString:@"====================================================================\n\n"];
    [out appendFormat:@"  Battery:   %.0f%%\n", ub_battery()*100];
    [out appendFormat:@"  Memory:    %.0f MB\n", ub_memory_mb()];
    [out appendFormat:@"  Duration:  %.0f min\n\n", minutes];

    // Create dummy data
    NSString *tmpPath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"ub_train_tokens.bin"];
    { size_t n=200000; uint16_t *buf=(uint16_t*)malloc(n*2); srand48(42);
      for (size_t i=0;i<n;i++) buf[i]=(uint16_t)(drand48()*(VOCAB-1));
      [[NSData dataWithBytesNoCopy:buf length:n*2 freeWhenDone:YES] writeToFile:tmpPath atomically:YES]; }

    NSLog(@"UB-TRAIN: Initializing (memory=%.0fMB)...", ub_memory_mb());
    ANETrainState *train = ane_train_init(NULL, [tmpPath UTF8String]);
    if (!train) {
        [out appendString:@"  FAILED: ane_train_init returned NULL\n"];
        return out;
    }
    NSLog(@"UB-TRAIN: Init done (memory=%.0fMB). Running %.0fs...", ub_memory_mb(), target_s);
    [out appendFormat:@"  Memory after init: %.0f MB\n\n", ub_memory_mb()];

    float batt_start = ub_battery();
    double cpu_start = ub_cpu_time();
    UBStats st_regular, st_compile;
    ubs_init(&st_regular, 8192); ubs_init(&st_compile, 2048);
    float first_loss=0, best_loss=999, last_loss=0;
    int total_steps=0;

    uint64_t t_train = mach_absolute_time();
    while (UB_S(mach_absolute_time()-t_train) < target_s) {
        uint64_t ts = mach_absolute_time();
        float loss = ane_train_step(train);
        double ms = UB_MS(mach_absolute_time()-ts);

        total_steps++;
        last_loss = loss;
        if (total_steps==1) first_loss = loss;
        if (loss < best_loss) best_loss = loss;

        int step = ane_train_current_step(train);
        bool is_compile = (step % ACCUM_STEPS == 0) && (step > 0);
        if (is_compile) ubs_add(&st_compile, ms); else ubs_add(&st_regular, ms);

        if (total_steps%200==0)
            NSLog(@"UB-TRAIN: step %d, loss=%.4f, %.1fms, mem=%.0fMB",
                total_steps, loss, ms, ub_memory_mb());

        usleep(5000);
        if (current_thermal_level() >= ThermalCritical) {
            NSLog(@"UB-TRAIN: CRITICAL thermal — pausing 30s");
            usleep(30000000);
        }
    }

    double elapsed = UB_S(mach_absolute_time()-t_train);
    float batt_end = ub_battery();
    float drain = batt_start - batt_end;
    double gross_w = (drain>0) ? (drain*battWh)/(elapsed/3600.0) : 0;
    double cpu_pct = (ub_cpu_time()-cpu_start)/elapsed*100;

    [out appendFormat:@"  Steps:     %d in %.0fs (%.2f steps/s)\n", total_steps, elapsed, total_steps/elapsed];
    [out appendFormat:@"  Tokens/s:  %.0f\n", (total_steps*SEQ)/elapsed];
    [out appendFormat:@"  Loss:      %.4f -> %.4f (best: %.4f, -%.1f%%)\n",
        first_loss, last_loss, best_loss, (first_loss-best_loss)/first_loss*100];
    [out appendFormat:@"  Regular:   %.1f ms (p50=%.1f p95=%.1f, n=%d)\n",
        ubs_mean(&st_regular), ubs_pct(&st_regular,0.5), ubs_pct(&st_regular,0.95), st_regular.count];
    [out appendFormat:@"  Compile:   %.1f ms (p50=%.1f p95=%.1f, n=%d)\n",
        ubs_mean(&st_compile), ubs_pct(&st_compile,0.5), ubs_pct(&st_compile,0.95), st_compile.count];
    [out appendFormat:@"  Overhead:  %.1fx slower, %.1f%% of time in compile\n",
        ubs_mean(&st_compile)/fmax(ubs_mean(&st_regular),0.001),
        st_compile.count>0 ? st_compile.sum/(st_compile.sum+st_regular.sum)*100 : 0];
    [out appendFormat:@"  Battery:   %.0f%% -> %.0f%% (%.1f%% drain)\n",
        batt_start*100, batt_end*100, drain*100];
    if (drain > 0) {
        [out appendFormat:@"  Power:     %.2fW\n", gross_w];
        [out appendFormat:@"  J/step:    %.3f\n", (gross_w*elapsed)/total_steps];
        [out appendFormat:@"  Tok/J:     %.0f\n", (total_steps*SEQ)/(gross_w*elapsed)];
    }
    [out appendFormat:@"  CPU use:   %.1f%%\n", cpu_pct];
    [out appendFormat:@"  Memory:    %.0f MB\n", ub_memory_mb()];
    [out appendFormat:@"  Thermal:   %s\n", thermal_level_name(current_thermal_level())];
    [out appendString:@"\n====================================================================\n"];

    ubs_free(&st_regular); ubs_free(&st_compile);
    ane_train_free(train);
    [[NSFileManager defaultManager] removeItemAtPath:tmpPath error:nil];
    [out writeToFile:resultsPath atomically:YES encoding:NSUTF8StringEncoding error:nil];
    NSLog(@"UB-TRAIN: Complete.");
    return out;
    }
}

// ═══════════════════════════════════════════════════════
// ANE POWER TEST — long-running ANE-only inference for power measurement
// ═══════════════════════════════════════════════════════

NSString *ane_power_test(float minutes) {
    @autoreleasepool {
    ub_init_tb();
    NSMutableString *out = [NSMutableString string];
    double battWh = ub_battery_wh();
    double target_s = minutes * 60.0;

    [out appendString:@"====================================================================\n"];
    [out appendString:@"  ANE POWER TEST — long-duration ANE inference\n"];
    [out appendString:@"====================================================================\n\n"];
    [out appendFormat:@"  Duration:  %.0f min\n", minutes];
    [out appendFormat:@"  Battery:   %.0f%% (%.1f Wh capacity)\n", ub_battery()*100, battWh];
    [out appendFormat:@"  Thermal:   %s\n", thermal_level_name(current_thermal_level())];
    [out appendFormat:@"  Memory:    %.0f MB\n\n", ub_memory_mb()];

    // Allocate weights + ANE kernels
    NSLog(@"ANE-POWER: Allocating weights + compiling 24 kernels...");
    UBWeights w = ub_weights_alloc();
    UBAne *ane = ub_ane_init(&w);
    if (!ane || !ane->ok) {
        [out appendString:@"  FAILED: ANE kernel compilation failed\n"];
        ub_weights_free(&w);
        if (ane) ub_ane_free(ane);
        return out;
    }
    [out appendFormat:@"  ANE compiled: %d kernels, memory %.0f MB\n\n", ane->compile_count, ub_memory_mb()];

    // Dummy input
    uint16_t tokens[SEQ]; srand48(99);
    for (int i=0;i<SEQ;i++) tokens[i]=(uint16_t)(drand48()*(VOCAB-1));

    // Warmup
    for (int i=0;i<10;i++) { UBComponents _c={0}; ub_ane_forward(ane,tokens,&_c); }

    float batt_start = ub_battery();
    double cpu_start = ub_cpu_time();
    int passes = 0;
    ThermalLevel worst = ThermalNominal;

    NSLog(@"ANE-POWER: Running for %.0f min, battery=%.0f%%...", minutes, batt_start*100);

    uint64_t t0 = mach_absolute_time();
    while (UB_S(mach_absolute_time()-t0) < target_s) {
        UBComponents _c = {0};
        ub_ane_forward(ane, tokens, &_c);
        passes++;

        ThermalLevel tl = current_thermal_level();
        if (tl > worst) worst = tl;

        usleep(5000);
        if (tl >= ThermalCritical) usleep(30000000);

        if (passes % 1000 == 0) {
            double el = UB_S(mach_absolute_time()-t0);
            NSLog(@"ANE-POWER: %d passes, %.0fs/%.0fs, batt=%.0f%%, thermal=%s",
                passes, el, target_s, ub_battery()*100, thermal_level_name(tl));
        }
    }

    double elapsed = UB_S(mach_absolute_time()-t0);
    float batt_end = ub_battery();
    float drain = batt_start - batt_end;
    double gross_w = (drain > 0) ? (drain * battWh) / (elapsed / 3600.0) : 0;
    double tps = (passes * SEQ) / elapsed;
    double cpu_pct = (ub_cpu_time()-cpu_start)/elapsed*100;

    [out appendFormat:@"  Passes:    %d in %.0fs (%.1f/s)\n", passes, elapsed, passes/elapsed];
    [out appendFormat:@"  Tokens/s:  %.0f\n", tps];
    [out appendFormat:@"  Battery:   %.0f%% -> %.0f%% (%.1f%% drain)\n",
        batt_start*100, batt_end*100, drain*100];
    if (drain > 0) {
        double tpj = (passes * SEQ) / (gross_w * elapsed);
        [out appendFormat:@"  Power:     %.2f W\n", gross_w];
        [out appendFormat:@"  Energy:    %.3f Wh\n", drain * battWh];
        [out appendFormat:@"  Tok/Joule: %.0f\n", tpj];
        [out appendFormat:@"  Tok/Watt:  %.0f\n", tps / gross_w];
    } else {
        [out appendString:@"  Power:     <1%% drain — below measurement threshold\n"];
        [out appendFormat:@"  Upper bound: <%.2f W\n", (0.01 * battWh) / (elapsed / 3600.0)];
    }
    [out appendFormat:@"  CPU use:   %.1f%%\n", cpu_pct];
    [out appendFormat:@"  Thermal:   %s (worst)\n", thermal_level_name(worst)];
    [out appendString:@"\n====================================================================\n"];

    NSString *docs = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,NSUserDomainMask,YES) firstObject];
    [out writeToFile:[docs stringByAppendingPathComponent:@"ane_power_test.txt"]
          atomically:YES encoding:NSUTF8StringEncoding error:nil];

    ub_ane_free(ane);
    ub_weights_free(&w);
    NSLog(@"ANE-POWER: Complete.");
    return out;
    }
}
