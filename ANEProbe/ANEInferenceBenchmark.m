// ANEInferenceBenchmark.m — Inference benchmark: ANE vs GPU (MPS) vs CPU
// Compares Stories-110M forward pass on all three compute backends.
#import "ANEInferenceBenchmark.h"
#import "ANEInference.h"
#import "ANETrainingConfig.h"
#import "ANEStoriesMIL.h"
#import "ANEStoriesCPUOps.h"
#import "ANEThermal.h"
#import <Accelerate/Accelerate.h>
#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <mach/mach_time.h>
#import <mach/mach.h>
#include <sys/utsname.h>

// ===== Stats helper =====
typedef struct {
    double sum, sum_sq, min, max;
    int count;
} InfStats;

static void istats_init(InfStats *s) { s->sum = s->sum_sq = 0; s->min = 1e9; s->max = -1e9; s->count = 0; }
static void istats_add(InfStats *s, double v) { s->sum += v; s->sum_sq += v*v; if (v<s->min) s->min=v; if (v>s->max) s->max=v; s->count++; }
static double istats_mean(InfStats *s) { return s->count>0 ? s->sum/s->count : 0; }
static double istats_std(InfStats *s) { if (s->count<2) return 0; double m=istats_mean(s); double v=s->sum_sq/s->count-m*m; return v>0?sqrt(v):0; }

// ===== Shared weight state (used by all backends) =====
typedef struct {
    LayerWeights lw[NLAYERS];
    float *rms_final;
    float *embed;
} SharedWeights;

static SharedWeights alloc_shared_weights(void) {
    SharedWeights sw;
    for (int L = 0; L < NLAYERS; L++) sw.lw[L] = layer_weights_alloc();
    sw.rms_final = (float *)malloc(DIM * 4);
    sw.embed = (float *)malloc((size_t)VOCAB * DIM * 4);
    // Random init
    srand48(42);
    float sd = 1.0f/sqrtf(DIM), sh = 1.0f/sqrtf(HIDDEN), os = 1.0f/sqrtf((float)NLAYERS);
    for (int L = 0; L < NLAYERS; L++) {
        for (size_t i=0; i<WQ_SZ; i++) { sw.lw[L].Wq[i]=sd*(2*drand48()-1); sw.lw[L].Wk[i]=sd*(2*drand48()-1); }
        for (size_t i=0; i<WQ_SZ; i++) { sw.lw[L].Wv[i]=sd*(2*drand48()-1); sw.lw[L].Wo[i]=sd*os*(2*drand48()-1); }
        for (size_t i=0; i<W1_SZ; i++) sw.lw[L].W1[i]=sh*(2*drand48()-1);
        for (size_t i=0; i<W2_SZ; i++) sw.lw[L].W2[i]=sd*os*(2*drand48()-1);
        for (size_t i=0; i<W3_SZ; i++) sw.lw[L].W3[i]=sh*(2*drand48()-1);
        for (int i=0; i<DIM; i++) { sw.lw[L].rms_att[i]=1.0f; sw.lw[L].rms_ffn[i]=1.0f; }
    }
    for (int i=0; i<DIM; i++) sw.rms_final[i]=1.0f;
    for (size_t i=0; i<(size_t)VOCAB*DIM; i++) sw.embed[i]=0.02f*(2*drand48()-1);
    return sw;
}

static void free_shared_weights(SharedWeights *sw) {
    for (int L = 0; L < NLAYERS; L++) layer_weights_free(&sw->lw[L]);
    free(sw->rms_final); free(sw->embed);
}

// ===== CPU Forward Pass (pure Accelerate/BLAS) =====
// All matmuls via cblas_sgemm, all norms/activations via vDSP
// Layout: [channels, spatial] i.e. [DIM, SEQ] column-of-channels

typedef struct {
    float *x_cur, *x_norm, *x2, *x_final;
    float *Q, *K, *V, *attn_out, *o_out;
    float *scores;          // [HEADS, SEQ, SEQ] for attention
    float *h1, *h3, *silu_out, *ffn_out;
    float *logits;          // [VOCAB]
} CPUBufs;

static CPUBufs cpu_bufs_alloc(void) {
    CPUBufs b;
    b.x_cur    = (float*)malloc(SEQ*DIM*4);
    b.x_norm   = (float*)malloc(SEQ*DIM*4);
    b.x2       = (float*)malloc(SEQ*DIM*4);
    b.x_final  = (float*)malloc(SEQ*DIM*4);
    b.Q        = (float*)malloc(SEQ*DIM*4);
    b.K        = (float*)malloc(SEQ*DIM*4);
    b.V        = (float*)malloc(SEQ*DIM*4);
    b.attn_out = (float*)malloc(SEQ*DIM*4);
    b.o_out    = (float*)malloc(SEQ*DIM*4);
    b.scores   = (float*)malloc(HEADS*SEQ*SEQ*4);
    b.h1       = (float*)malloc(SEQ*HIDDEN*4);
    b.h3       = (float*)malloc(SEQ*HIDDEN*4);
    b.silu_out = (float*)malloc(SEQ*HIDDEN*4);
    b.ffn_out  = (float*)malloc(SEQ*DIM*4);
    b.logits   = (float*)malloc(VOCAB*4);
    return b;
}
static void cpu_bufs_free(CPUBufs *b) {
    free(b->x_cur); free(b->x_norm); free(b->x2); free(b->x_final);
    free(b->Q); free(b->K); free(b->V); free(b->attn_out); free(b->o_out);
    free(b->scores); free(b->h1); free(b->h3); free(b->silu_out); free(b->ffn_out);
    free(b->logits);
}

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
static void silu_inplace(float *x, int n) {
    float *neg = (float*)malloc(n*4);
    float neg1 = -1.0f;
    vDSP_vsmul(x, 1, &neg1, neg, 1, (vDSP_Length)n);
    vvexpf(neg, neg, &n);
    float one = 1.0f;
    vDSP_vsadd(neg, 1, &one, neg, 1, (vDSP_Length)n);  // neg = 1 + exp(-x)
    vDSP_vdiv(neg, 1, x, 1, x, 1, (vDSP_Length)n);     // x = x / (1+exp(-x))
    free(neg);
}

// CPU scaled dot-product attention
// Q,K,V in [DIM, SEQ] channel-first layout, multi-head: head_dim = DIM/HEADS
// Output: attn_out [DIM, SEQ]
static void cpu_attention(float *attn_out, const float *Q, const float *K, const float *V,
                          float *scores_buf, int dim, int seq, int nheads) {
    int hd = dim / nheads;
    // For each head, compute attention
    // Q,K,V are [DIM, SEQ] channel-first. Head h uses channels [h*hd .. (h+1)*hd-1]
    for (int h = 0; h < nheads; h++) {
        // Extract Q_h [hd, seq], K_h [hd, seq] — already contiguous in channel-first
        const float *Qh = Q + h * hd * seq;
        const float *Kh = K + h * hd * seq;
        const float *Vh = V + h * hd * seq;
        float *scores = scores_buf + h * seq * seq;

        // scores = Q_h^T @ K_h → [seq, seq]
        // Q_h is [hd, seq], Q_h^T is [seq, hd]
        // K_h is [hd, seq]
        // scores[i,j] = dot(Q[:,i], K[:,j])
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    seq, seq, hd, 1.0f,
                    Qh, seq, Kh, seq, 0.0f, scores, seq);

        // Scale by 1/sqrt(hd)
        float scale = 1.0f / sqrtf((float)hd);
        vDSP_vsmul(scores, 1, &scale, scores, 1, (vDSP_Length)(seq*seq));

        // Causal mask: set scores[i][j] = -inf for j > i
        for (int i = 0; i < seq; i++)
            for (int j = i+1; j < seq; j++)
                scores[i*seq + j] = -1e9f;

        // Softmax per row
        for (int i = 0; i < seq; i++) {
            float *row = scores + i * seq;
            float mx; vDSP_maxv(row, 1, &mx, (vDSP_Length)seq);
            float neg_mx = -mx; vDSP_vsadd(row, 1, &neg_mx, row, 1, (vDSP_Length)seq);
            int n = seq; vvexpf(row, row, &n);
            float sm; vDSP_sve(row, 1, &sm, (vDSP_Length)seq);
            float inv = 1.0f/sm; vDSP_vsmul(row, 1, &inv, row, 1, (vDSP_Length)seq);
        }

        // attn_out_h = V_h @ scores^T → [hd, seq]
        // V_h is [hd, seq], scores^T is [seq, seq]
        // We want: for each position i, attn[i] = sum_j scores[i][j] * V[:,j]
        // = V @ scores^T (row of scores = weights for position i)
        float *Ah = attn_out + h * hd * seq;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    hd, seq, seq, 1.0f,
                    Vh, seq, scores, seq, 0.0f, Ah, seq);
    }
}

// Full CPU forward pass
static void cpu_forward(SharedWeights *sw, CPUBufs *b, const uint16_t *tokens) {
    // Embed
    memset(b->x_cur, 0, SEQ*DIM*4);
    embed_lookup(b->x_cur, sw->embed, tokens, DIM, SEQ);

    for (int L = 0; L < NLAYERS; L++) {
        // RMSNorm1
        rmsnorm(b->x_norm, b->x_cur, sw->lw[L].rms_att, DIM, SEQ);

        // Q = Wq @ x_norm  [DIM,DIM] @ [DIM,SEQ] = [DIM,SEQ]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    DIM, SEQ, DIM, 1.0f, sw->lw[L].Wq, DIM, b->x_norm, SEQ, 0.0f, b->Q, SEQ);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    DIM, SEQ, DIM, 1.0f, sw->lw[L].Wk, DIM, b->x_norm, SEQ, 0.0f, b->K, SEQ);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    DIM, SEQ, DIM, 1.0f, sw->lw[L].Wv, DIM, b->x_norm, SEQ, 0.0f, b->V, SEQ);

        // Attention
        cpu_attention(b->attn_out, b->Q, b->K, b->V, b->scores, DIM, SEQ, HEADS);

        // O projection: o_out = Wo @ attn_out
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    DIM, SEQ, DIM, 1.0f, sw->lw[L].Wo, DIM, b->attn_out, SEQ, 0.0f, b->o_out, SEQ);

        // Residual: x2 = x_cur + o_out
        vDSP_vadd(b->x_cur, 1, b->o_out, 1, b->x2, 1, (vDSP_Length)(SEQ*DIM));

        // RMSNorm2
        rmsnorm(b->x_norm, b->x2, sw->lw[L].rms_ffn, DIM, SEQ);

        // FFN: h1 = W1 @ x_norm [HIDDEN,DIM]@[DIM,SEQ]=[HIDDEN,SEQ]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    HIDDEN, SEQ, DIM, 1.0f, sw->lw[L].W1, DIM, b->x_norm, SEQ, 0.0f, b->h1, SEQ);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    HIDDEN, SEQ, DIM, 1.0f, sw->lw[L].W3, DIM, b->x_norm, SEQ, 0.0f, b->h3, SEQ);

        // SiLU(h1) * h3
        silu_inplace(b->h1, HIDDEN*SEQ);
        vDSP_vmul(b->h1, 1, b->h3, 1, b->silu_out, 1, (vDSP_Length)(HIDDEN*SEQ));

        // ffn_out = W2 @ silu_out [DIM,HIDDEN]@[HIDDEN,SEQ]=[DIM,SEQ]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    DIM, SEQ, HIDDEN, 1.0f, sw->lw[L].W2, HIDDEN, b->silu_out, SEQ, 0.0f, b->ffn_out, SEQ);

        // Residual: x_cur = x2 + ffn_out
        vDSP_vadd(b->x2, 1, b->ffn_out, 1, b->x_cur, 1, (vDSP_Length)(SEQ*DIM));
    }

    // Final RMSNorm
    rmsnorm(b->x_final, b->x_cur, sw->rms_final, DIM, SEQ);

    // Classifier: single position logits (last position)
    int pos = SEQ - 1;
    float *x_pos = (float*)malloc(DIM*4);
    for (int d = 0; d < DIM; d++) x_pos[d] = b->x_final[d*SEQ + pos];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, VOCAB, DIM, 1.0f,
                sw->embed, DIM, x_pos, 1, 0.0f, b->logits, 1);
    free(x_pos);
}

// ===== GPU Forward Pass (MPS matmuls + CPU for norms/attention) =====

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    // Weight buffers (persistent)
    id<MTLBuffer> wq[NLAYERS], wk[NLAYERS], wv[NLAYERS], wo[NLAYERS];
    id<MTLBuffer> w1[NLAYERS], w2[NLAYERS], w3[NLAYERS];
    // Activation buffers (reused)
    id<MTLBuffer> buf_x, buf_norm, buf_Q, buf_K, buf_V;
    id<MTLBuffer> buf_attn, buf_o, buf_h1, buf_h3, buf_silu, buf_ffn;
} GPUState;

static GPUState *gpu_init(SharedWeights *sw) {
    GPUState *g = (GPUState *)calloc(1, sizeof(GPUState));
    g->device = MTLCreateSystemDefaultDevice();
    if (!g->device) { free(g); return NULL; }
    g->queue = [g->device newCommandQueue];

    // Upload weight matrices to GPU
    for (int L = 0; L < NLAYERS; L++) {
        g->wq[L] = [g->device newBufferWithBytes:sw->lw[L].Wq length:WQ_SZ*4 options:MTLResourceStorageModeShared];
        g->wk[L] = [g->device newBufferWithBytes:sw->lw[L].Wk length:WQ_SZ*4 options:MTLResourceStorageModeShared];
        g->wv[L] = [g->device newBufferWithBytes:sw->lw[L].Wv length:WQ_SZ*4 options:MTLResourceStorageModeShared];
        g->wo[L] = [g->device newBufferWithBytes:sw->lw[L].Wo length:WO_SZ*4 options:MTLResourceStorageModeShared];
        g->w1[L] = [g->device newBufferWithBytes:sw->lw[L].W1 length:W1_SZ*4 options:MTLResourceStorageModeShared];
        g->w2[L] = [g->device newBufferWithBytes:sw->lw[L].W2 length:W2_SZ*4 options:MTLResourceStorageModeShared];
        g->w3[L] = [g->device newBufferWithBytes:sw->lw[L].W3 length:W3_SZ*4 options:MTLResourceStorageModeShared];
    }

    // Activation buffers
    g->buf_x    = [g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_norm = [g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_Q    = [g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_K    = [g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_V    = [g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_attn = [g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_o    = [g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    g->buf_h1   = [g->device newBufferWithLength:SEQ*HIDDEN*4 options:MTLResourceStorageModeShared];
    g->buf_h3   = [g->device newBufferWithLength:SEQ*HIDDEN*4 options:MTLResourceStorageModeShared];
    g->buf_silu = [g->device newBufferWithLength:SEQ*HIDDEN*4 options:MTLResourceStorageModeShared];
    g->buf_ffn  = [g->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    return g;
}

static void gpu_free(GPUState *g) {
    // ARC handles Metal object release
    free(g);
}

// MPS matmul helper: C[M,N] = A[M,K] @ B[K,N]
// Reads from bufA, bufB, writes to bufC
static void mps_matmul(GPUState *g, id<MTLBuffer> bufA, id<MTLBuffer> bufB, id<MTLBuffer> bufC,
                       int M, int K, int N) {
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
                                   rowBytes:K*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N
                                   rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
                                   rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g->device transposeLeft:NO transposeRight:NO
        resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];

    id<MTLCommandBuffer> cmd = [g->queue commandBuffer];
    [mm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [cmd commit];
    [cmd waitUntilCompleted];
}

// GPU forward pass: MPS for matmuls, CPU for norms/attention/activations
static void gpu_forward(GPUState *g, SharedWeights *sw, CPUBufs *b, const uint16_t *tokens) {
    // Embed (CPU)
    memset(b->x_cur, 0, SEQ*DIM*4);
    embed_lookup(b->x_cur, sw->embed, tokens, DIM, SEQ);

    for (int L = 0; L < NLAYERS; L++) {
        // RMSNorm1 (CPU)
        rmsnorm(b->x_norm, b->x_cur, sw->lw[L].rms_att, DIM, SEQ);

        // Copy x_norm to GPU buffer
        memcpy(g->buf_norm.contents, b->x_norm, SEQ*DIM*4);

        // Q,K,V projections (GPU via MPS) — batch all 3 in sequence
        // Wq[DIM,DIM] @ x_norm[DIM,SEQ] = Q[DIM,SEQ]
        mps_matmul(g, g->wq[L], g->buf_norm, g->buf_Q, DIM, DIM, SEQ);
        mps_matmul(g, g->wk[L], g->buf_norm, g->buf_K, DIM, DIM, SEQ);
        mps_matmul(g, g->wv[L], g->buf_norm, g->buf_V, DIM, DIM, SEQ);

        // Read back Q,K,V for attention (CPU)
        memcpy(b->Q, g->buf_Q.contents, SEQ*DIM*4);
        memcpy(b->K, g->buf_K.contents, SEQ*DIM*4);
        memcpy(b->V, g->buf_V.contents, SEQ*DIM*4);

        // Attention (CPU — would need custom Metal kernel for GPU)
        cpu_attention(b->attn_out, b->Q, b->K, b->V, b->scores, DIM, SEQ, HEADS);

        // O projection (GPU)
        memcpy(g->buf_attn.contents, b->attn_out, SEQ*DIM*4);
        mps_matmul(g, g->wo[L], g->buf_attn, g->buf_o, DIM, DIM, SEQ);
        memcpy(b->o_out, g->buf_o.contents, SEQ*DIM*4);

        // Residual (CPU)
        vDSP_vadd(b->x_cur, 1, b->o_out, 1, b->x2, 1, (vDSP_Length)(SEQ*DIM));

        // RMSNorm2 (CPU)
        rmsnorm(b->x_norm, b->x2, sw->lw[L].rms_ffn, DIM, SEQ);
        memcpy(g->buf_norm.contents, b->x_norm, SEQ*DIM*4);

        // FFN projections (GPU)
        mps_matmul(g, g->w1[L], g->buf_norm, g->buf_h1, HIDDEN, DIM, SEQ);
        mps_matmul(g, g->w3[L], g->buf_norm, g->buf_h3, HIDDEN, DIM, SEQ);

        memcpy(b->h1, g->buf_h1.contents, SEQ*HIDDEN*4);
        memcpy(b->h3, g->buf_h3.contents, SEQ*HIDDEN*4);

        // SiLU + gate (CPU)
        silu_inplace(b->h1, HIDDEN*SEQ);
        vDSP_vmul(b->h1, 1, b->h3, 1, b->silu_out, 1, (vDSP_Length)(HIDDEN*SEQ));

        // W2 projection (GPU)
        memcpy(g->buf_silu.contents, b->silu_out, SEQ*HIDDEN*4);
        mps_matmul(g, g->w2[L], g->buf_silu, g->buf_ffn, DIM, HIDDEN, SEQ);
        memcpy(b->ffn_out, g->buf_ffn.contents, SEQ*DIM*4);

        // Residual (CPU)
        vDSP_vadd(b->x2, 1, b->ffn_out, 1, b->x_cur, 1, (vDSP_Length)(SEQ*DIM));
    }

    // Final RMSNorm (CPU)
    rmsnorm(b->x_final, b->x_cur, sw->rms_final, DIM, SEQ);

    // Classifier (CPU — single position)
    int pos = SEQ - 1;
    float *x_pos = (float*)malloc(DIM*4);
    for (int d = 0; d < DIM; d++) x_pos[d] = b->x_final[d*SEQ + pos];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, VOCAB, DIM, 1.0f,
                sw->embed, DIM, x_pos, 1, 0.0f, b->logits, 1);
    free(x_pos);
}

// ===== ANE Forward Pass (existing engine) =====
// Uses ane_inference_init / inf_forward via the public API.
// Since inf_forward is static in ANEInference.m, we use ane_generate with 1 token as proxy.
// Better: replicate the forward call here using compile_kern + ane_eval directly.

typedef struct {
    LayerWeights lw[NLAYERS];
    float *rms_final, *embed;
    Kern *fwdAttn[NLAYERS], *fwdFFN[NLAYERS];
    float *x_cur, *x_final, *o_out, *x2, *ffn_out, *logits;
    int compile_count;
    bool ok;
} ANEInfState;

static ANEInfState *ane_inf_init(SharedWeights *sw) {
    ane_init();
    ANEInfState *a = (ANEInfState *)calloc(1, sizeof(ANEInfState));

    // Share weights by copying pointers (we own the data via SharedWeights)
    for (int L = 0; L < NLAYERS; L++) {
        a->lw[L] = sw->lw[L];  // shallow copy — points to same memory
    }
    a->rms_final = sw->rms_final;
    a->embed = sw->embed;

    // Scratch
    a->x_cur   = (float*)malloc(SEQ*DIM*4);
    a->x_final = (float*)malloc(SEQ*DIM*4);
    a->o_out   = (float*)malloc(SEQ*DIM*4);
    a->x2      = (float*)malloc(SEQ*DIM*4);
    a->ffn_out = (float*)malloc(SEQ*DIM*4);
    a->logits  = (float*)malloc(VOCAB*4);

    // Compile 24 forward kernels
    a->ok = true;
    for (int L = 0; L < NLAYERS; L++) {
        a->fwdAttn[L] = compile_kern(gen_sdpa_fwd_taps(), (@{
            @"@model_path/weights/rms1.bin": @{@"offset":@0, @"data":build_blob(sw->lw[L].rms_att, 1, DIM)},
            @"@model_path/weights/wq.bin":   @{@"offset":@0, @"data":build_blob(sw->lw[L].Wq, DIM, DIM)},
            @"@model_path/weights/wk.bin":   @{@"offset":@0, @"data":build_blob(sw->lw[L].Wk, DIM, DIM)},
            @"@model_path/weights/wv.bin":   @{@"offset":@0, @"data":build_blob(sw->lw[L].Wv, DIM, DIM)},
            @"@model_path/weights/wo.bin":   @{@"offset":@0, @"data":build_blob(sw->lw[L].Wo, DIM, DIM)},
            @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        }), DIM*SEQ*2, 6*DIM*SEQ*2);

        a->fwdFFN[L] = compile_kern(gen_ffn_fwd_taps(), (@{
            @"@model_path/weights/rms2.bin": @{@"offset":@0, @"data":build_blob(sw->lw[L].rms_ffn, 1, DIM)},
            @"@model_path/weights/w1.bin":   @{@"offset":@0, @"data":build_blob(sw->lw[L].W1, HIDDEN, DIM)},
            @"@model_path/weights/w3.bin":   @{@"offset":@0, @"data":build_blob(sw->lw[L].W3, HIDDEN, DIM)},
            @"@model_path/weights/w2.bin":   @{@"offset":@0, @"data":build_blob(sw->lw[L].W2, DIM, HIDDEN)},
        }), DIM*SEQ*2, (2*DIM + 3*HIDDEN)*SEQ*2);

        a->compile_count += 2;
        if (!a->fwdAttn[L] || !a->fwdFFN[L]) { a->ok = false; break; }
    }
    return a;
}

static void ane_inf_free(ANEInfState *a) {
    for (int L = 0; L < NLAYERS; L++) {
        free_kern(a->fwdAttn[L]);
        free_kern(a->fwdFFN[L]);
    }
    // Don't free lw/rms_final/embed — owned by SharedWeights
    free(a->x_cur); free(a->x_final); free(a->o_out);
    free(a->x2); free(a->ffn_out); free(a->logits);
    free(a);
}

static void ane_inf_forward(ANEInfState *a, const uint16_t *tokens) {
    memset(a->x_cur, 0, SEQ*DIM*4);
    embed_lookup(a->x_cur, a->embed, tokens, DIM, SEQ);

    for (int L = 0; L < NLAYERS; L++) {
        io_write_fp16(a->fwdAttn[L]->ioIn, a->x_cur, DIM, SEQ);
        ane_eval(a->fwdAttn[L]);
        io_read_fp16(a->fwdAttn[L]->ioOut, a->o_out, 0, DIM, SEQ);
        vDSP_vadd(a->x_cur, 1, a->o_out, 1, a->x2, 1, (vDSP_Length)(SEQ*DIM));

        io_write_fp16(a->fwdFFN[L]->ioIn, a->x2, DIM, SEQ);
        ane_eval(a->fwdFFN[L]);
        io_read_fp16(a->fwdFFN[L]->ioOut, a->ffn_out, 0, DIM, SEQ);
        vDSP_vadd(a->x2, 1, a->ffn_out, 1, a->x_cur, 1, (vDSP_Length)(SEQ*DIM));
    }

    rmsnorm(a->x_final, a->x_cur, a->rms_final, DIM, SEQ);

    int pos = SEQ - 1;
    float *x_pos = (float*)malloc(DIM*4);
    for (int d = 0; d < DIM; d++) x_pos[d] = a->x_final[d*SEQ + pos];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, VOCAB, DIM, 1.0f,
                a->embed, DIM, x_pos, 1, 0.0f, a->logits, 1);
    free(x_pos);
}

// ===== MAIN BENCHMARK =====

NSString *ane_inference_benchmark(int iterations) {
    @autoreleasepool {
    NSMutableString *out = [NSMutableString string];
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    #define TBM(t) ((double)(t)*tb.numer/tb.denom/1e6)

    [out appendString:@"╔═══════════════════════════════════════════════════════════╗\n"];
    [out appendString:@"║        INFERENCE BENCHMARK: ANE vs GPU vs CPU             ║\n"];
    [out appendString:@"║        Stories-110M Forward Pass Comparison               ║\n"];
    [out appendString:@"╚═══════════════════════════════════════════════════════════╝\n\n"];

    [out appendFormat:@"  Model:       Stories-110M (%d layers, dim=%d, seq=%d)\n", NLAYERS, DIM, SEQ];
    [out appendFormat:@"  Iterations:  %d per backend\n", iterations];
    [out appendFormat:@"  Tokens/fwd:  %d\n\n", SEQ];

    // --- Shared weights ---
    [out appendString:@"  Allocating shared weights...\n"];
    SharedWeights sw = alloc_shared_weights();

    // --- Dummy input tokens ---
    uint16_t tokens[SEQ];
    srand48(99);
    for (int i = 0; i < SEQ; i++) tokens[i] = (uint16_t)(drand48() * (VOCAB-1));

    // --- CPU buffers (shared between CPU and GPU backends) ---
    CPUBufs bufs = cpu_bufs_alloc();

    // Enable battery monitoring
    thermal_enable_battery_monitoring();
    [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.1]];
    float battStart = thermal_battery_level();

    // ═══════════════════════════════
    // 1. ANE BENCHMARK
    // ═══════════════════════════════
    [out appendString:@"  ┌─── ANE (Apple Neural Engine) ──────────────────────────┐\n"];
    [out appendString:@"  │ Compiling 24 MIL kernels...\n"];
    fprintf(stderr, "INFBENCH: Compiling ANE kernels...\n");

    uint64_t t0 = mach_absolute_time();
    ANEInfState *ane = ane_inf_init(&sw);
    double ane_compile_ms = TBM(mach_absolute_time() - t0);

    InfStats ane_stats; istats_init(&ane_stats);

    if (!ane->ok) {
        [out appendString:@"  │ FAIL: ANE kernel compilation failed\n"];
        [out appendString:@"  └────────────────────────────────────────────────────────┘\n\n"];
    } else {
        [out appendFormat:@"  │ Compile: %.1f ms (%d kernels)\n", ane_compile_ms, ane->compile_count];

        // Warmup
        for (int i = 0; i < 3; i++) ane_inf_forward(ane, tokens);
        fprintf(stderr, "INFBENCH: Running ANE (%d iterations)...\n", iterations);
        for (int i = 0; i < iterations; i++) {
            uint64_t ts = mach_absolute_time();
            ane_inf_forward(ane, tokens);
            istats_add(&ane_stats, TBM(mach_absolute_time() - ts));
        }

        double ane_tps = SEQ / (istats_mean(&ane_stats) / 1000.0);
        [out appendFormat:@"  │ Latency:  %.2f ms (σ=%.2f, min=%.2f, max=%.2f)\n",
            istats_mean(&ane_stats), istats_std(&ane_stats), ane_stats.min, ane_stats.max];
        [out appendFormat:@"  │ Tokens/s: %.0f\n", ane_tps];
        [out appendString:@"  └────────────────────────────────────────────────────────┘\n\n"];
    }

    // ═══════════════════════════════
    // 2. GPU BENCHMARK (MPS)
    // ═══════════════════════════════
    [out appendString:@"  ┌─── GPU (Metal Performance Shaders) ────────────────────┐\n"];
    fprintf(stderr, "INFBENCH: Initializing GPU (MPS)...\n");

    t0 = mach_absolute_time();
    GPUState *gpu = gpu_init(&sw);
    double gpu_init_ms = TBM(mach_absolute_time() - t0);

    InfStats gpu_stats; istats_init(&gpu_stats);
    if (!gpu) {
        [out appendString:@"  │ FAIL: Metal device not available\n"];
    } else {
        [out appendFormat:@"  │ Init:    %.1f ms (Metal device + buffer upload)\n", gpu_init_ms];
        [out appendFormat:@"  │ Device:  %@\n", gpu->device.name];
        [out appendString:@"  │ Note:    matmuls on GPU, attention/norms on CPU\n"];

        // Warmup
        for (int i = 0; i < 3; i++) gpu_forward(gpu, &sw, &bufs, tokens);

        fprintf(stderr, "INFBENCH: Running GPU (%d iterations)...\n", iterations);
        for (int i = 0; i < iterations; i++) {
            uint64_t ts = mach_absolute_time();
            gpu_forward(gpu, &sw, &bufs, tokens);
            istats_add(&gpu_stats, TBM(mach_absolute_time() - ts));
        }

        double gpu_tps = SEQ / (istats_mean(&gpu_stats) / 1000.0);
        [out appendFormat:@"  │ Latency:  %.2f ms (σ=%.2f, min=%.2f, max=%.2f)\n",
            istats_mean(&gpu_stats), istats_std(&gpu_stats), gpu_stats.min, gpu_stats.max];
        [out appendFormat:@"  │ Tokens/s: %.0f\n", gpu_tps];
    }
    [out appendString:@"  └────────────────────────────────────────────────────────┘\n\n"];

    // ═══════════════════════════════
    // 3. CPU BENCHMARK (Accelerate)
    // ═══════════════════════════════
    [out appendString:@"  ┌─── CPU (Accelerate / BLAS) ────────────────────────────┐\n"];
    fprintf(stderr, "INFBENCH: Running CPU (%d iterations)...\n", iterations);

    // Warmup
    for (int i = 0; i < 3; i++) cpu_forward(&sw, &bufs, tokens);

    InfStats cpu_stats; istats_init(&cpu_stats);
    for (int i = 0; i < iterations; i++) {
        uint64_t ts = mach_absolute_time();
        cpu_forward(&sw, &bufs, tokens);
        istats_add(&cpu_stats, TBM(mach_absolute_time() - ts));
    }

    double cpu_tps = SEQ / (istats_mean(&cpu_stats) / 1000.0);
    [out appendFormat:@"  │ Latency:  %.2f ms (σ=%.2f, min=%.2f, max=%.2f)\n",
        istats_mean(&cpu_stats), istats_std(&cpu_stats), cpu_stats.min, cpu_stats.max];
    [out appendFormat:@"  │ Tokens/s: %.0f\n", cpu_tps];
    [out appendString:@"  └────────────────────────────────────────────────────────┘\n\n"];

    // ═══════════════════════════════
    // COMPARISON
    // ═══════════════════════════════
    [out appendString:@"  ┌─── COMPARISON ──────────────────────────────────────────┐\n"];
    [out appendString:@"  │                                                        │\n"];
    [out appendString:@"  │ Backend    Latency(ms)  Tokens/s   vs ANE   vs CPU     │\n"];
    [out appendString:@"  │ ───────── ──────────── ────────── ──────── ────────    │\n"];

    double ane_ms = ane->ok ? istats_mean(&ane_stats) : -1;
    double gpu_ms = gpu ? istats_mean(&gpu_stats) : -1;
    double cpu_ms = istats_mean(&cpu_stats);

    if (ane->ok) {
        [out appendFormat:@"  │ ANE       %8.2f     %7.0f    1.00x    %.2fx      │\n",
            ane_ms, SEQ/(ane_ms/1000.0), cpu_ms/ane_ms];
    }
    if (gpu) {
        [out appendFormat:@"  │ GPU(MPS)  %8.2f     %7.0f    %.2fx    %.2fx      │\n",
            gpu_ms, SEQ/(gpu_ms/1000.0),
            ane->ok ? ane_ms/gpu_ms : 0, cpu_ms/gpu_ms];
    }
    [out appendFormat:@"  │ CPU       %8.2f     %7.0f    %.2fx    1.00x      │\n",
        cpu_ms, cpu_tps,
        ane->ok ? ane_ms/cpu_ms : 0];

    [out appendString:@"  │                                                        │\n"];

    // Winner
    double best_ms = cpu_ms;
    const char *winner = "CPU";
    if (ane->ok && ane_ms < best_ms) { best_ms = ane_ms; winner = "ANE"; }
    if (gpu && gpu_ms < best_ms) { best_ms = gpu_ms; winner = "GPU (MPS)"; }
    [out appendFormat:@"  │ Winner:   %s (%.2f ms, %.0f tok/s)               \n",
        winner, best_ms, SEQ/(best_ms/1000.0)];
    [out appendString:@"  │                                                        │\n"];

    // Notes
    [out appendString:@"  │ Notes:                                                 │\n"];
    [out appendString:@"  │ - ANE: full forward pass on Neural Engine (FP16)       │\n"];
    [out appendString:@"  │ - GPU: MPS matmuls + CPU attention/norms (FP32)        │\n"];
    [out appendString:@"  │ - CPU: Accelerate BLAS matmuls + vDSP (FP32)           │\n"];
    if (gpu && ane->ok && gpu_ms > ane_ms) {
        [out appendString:@"  │ - GPU slower due to CPU-GPU sync per matmul           │\n"];
        [out appendString:@"  │   (84 round-trips: 7 matmuls x 12 layers)             │\n"];
    }
    [out appendString:@"  └────────────────────────────────────────────────────────┘\n\n"];

    // Battery
    float battEnd = thermal_battery_level();
    if (battStart >= 0 && battEnd >= 0) {
        [out appendFormat:@"  Battery: %.0f%% → %.0f%% (Δ%.1f%%)\n",
            battStart*100, battEnd*100, (battStart-battEnd)*100];
    }

    // Cleanup
    ane_inf_free(ane);
    if (gpu) gpu_free(gpu);
    cpu_bufs_free(&bufs);
    free_shared_weights(&sw);
    thermal_disable_battery_monitoring();

    fprintf(stderr, "INFBENCH: Done.\n");
    #undef TBM
    return out;
    }
}

// ═══════════════════════════════════════════════════════
// POWER BENCHMARK
// ═══════════════════════════════════════════════════════

// Read battery synchronously (ensure monitoring is on)
static float read_battery_sync(void) {
    // Force main thread to process battery enable
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

static double device_battery_wh_power(void) {
    struct utsname si; uname(&si);
    NSString *hw = [NSString stringWithCString:si.machine encoding:NSUTF8StringEncoding];
    double mAh = 3400;
    if ([hw hasPrefix:@"iPhone16,1"])      mAh = 3274;
    else if ([hw hasPrefix:@"iPhone16,2"]) mAh = 4422;
    else if ([hw hasPrefix:@"iPhone17,1"]) mAh = 3582;
    else if ([hw hasPrefix:@"iPhone17,2"]) mAh = 4685;
    return mAh * 3.83 / 1000.0;
}

NSString *ane_power_benchmark(float minutes_per_backend) {
    @autoreleasepool {
    NSMutableString *out = [NSMutableString string];
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    #define TBM2(t) ((double)(t)*tb.numer/tb.denom/1e6)
    #define TBS2(t) ((double)(t)*tb.numer/tb.denom/1e9)

    double battWh = device_battery_wh_power();

    [out appendString:@"╔═══════════════════════════════════════════════════════════╗\n"];
    [out appendString:@"║        POWER BENCHMARK: ANE vs GPU vs CPU                ║\n"];
    [out appendString:@"║        Energy · Watts · Tokens/Joule                     ║\n"];
    [out appendString:@"╚═══════════════════════════════════════════════════════════╝\n\n"];

    [out appendFormat:@"  Duration:    %.0f min per backend (%.0f min total)\n",
        minutes_per_backend, minutes_per_backend * 3];
    [out appendFormat:@"  Battery:     %.1f Wh capacity\n", battWh];

    // Check battery
    float batt = read_battery_sync();
    [out appendFormat:@"  Level:       %.0f%%\n", batt * 100];

    if (batt < 0) {
        [out appendString:@"\n  ERROR: Battery monitoring unavailable.\n"];
        [out appendString:@"  Run on a real device, unplugged from charger.\n"];
        return out;
    }
    if (batt < 0.15) {
        [out appendString:@"\n  WARNING: Battery low (<15%%). Results may be inaccurate\n"];
        [out appendString:@"  due to low-power mode throttling.\n"];
    }

    [out appendFormat:@"  Thermal:     %s\n\n", thermal_level_name(current_thermal_level())];

    // Shared weights & buffers
    SharedWeights sw = alloc_shared_weights();
    CPUBufs bufs = cpu_bufs_alloc();
    uint16_t tokens[SEQ];
    srand48(99);
    for (int i = 0; i < SEQ; i++) tokens[i] = (uint16_t)(drand48() * (VOCAB-1));

    double target_s = minutes_per_backend * 60.0;

    // Results per backend
    typedef struct {
        const char *name;
        float batt_start, batt_end;
        double elapsed_s;
        int fwd_passes;
        double total_ms;
        ThermalLevel worst_thermal;
    } BackendResult;
    BackendResult results[3];
    memset(results, 0, sizeof(results));

    // ═══════════════════════════════
    // 1. ANE
    // ═══════════════════════════════
    [out appendString:@"  Running ANE...\n"];
    fprintf(stderr, "POWER: Compiling ANE kernels...\n");
    ANEInfState *ane = ane_inf_init(&sw);

    if (ane->ok) {
        // Warmup
        for (int i = 0; i < 5; i++) ane_inf_forward(ane, tokens);

        // Wait a moment for thermal to settle, then read battery
        usleep(500000);
        results[0].name = "ANE";
        results[0].batt_start = read_battery_sync();
        results[0].worst_thermal = ThermalNominal;

        uint64_t t_start = mach_absolute_time();
        fprintf(stderr, "POWER: ANE running for %.0f min...\n", minutes_per_backend);

        while (TBS2(mach_absolute_time() - t_start) < target_s) {
            uint64_t ts = mach_absolute_time();
            ane_inf_forward(ane, tokens);
            results[0].total_ms += TBM2(mach_absolute_time() - ts);
            results[0].fwd_passes++;
            ThermalLevel tl = current_thermal_level();
            if (tl > results[0].worst_thermal) results[0].worst_thermal = tl;

            if (results[0].fwd_passes % 500 == 0) {
                fprintf(stderr, "POWER: ANE %d passes, %.1fs elapsed\n",
                    results[0].fwd_passes, TBS2(mach_absolute_time() - t_start));
            }
        }
        results[0].elapsed_s = TBS2(mach_absolute_time() - t_start);
        results[0].batt_end = read_battery_sync();
        [out appendFormat:@"  ANE done: %d passes in %.1fs, battery %.0f%%→%.0f%%\n",
            results[0].fwd_passes, results[0].elapsed_s,
            results[0].batt_start*100, results[0].batt_end*100];
    } else {
        [out appendString:@"  ANE: kernel compilation failed, skipping\n"];
    }
    ane_inf_free(ane);

    // Cool-down pause between backends
    [out appendString:@"  Cooling down 30s...\n"];
    fprintf(stderr, "POWER: Cooling down 30s before GPU...\n");
    usleep(30000000);

    // ═══════════════════════════════
    // 2. GPU (MPS)
    // ═══════════════════════════════
    [out appendString:@"  Running GPU (MPS)...\n"];
    GPUState *gpu = gpu_init(&sw);

    if (gpu) {
        for (int i = 0; i < 5; i++) gpu_forward(gpu, &sw, &bufs, tokens);

        usleep(500000);
        results[1].name = "GPU";
        results[1].batt_start = read_battery_sync();
        results[1].worst_thermal = ThermalNominal;

        uint64_t t_start = mach_absolute_time();
        fprintf(stderr, "POWER: GPU running for %.0f min...\n", minutes_per_backend);

        while (TBS2(mach_absolute_time() - t_start) < target_s) {
            uint64_t ts = mach_absolute_time();
            gpu_forward(gpu, &sw, &bufs, tokens);
            results[1].total_ms += TBM2(mach_absolute_time() - ts);
            results[1].fwd_passes++;
            ThermalLevel tl = current_thermal_level();
            if (tl > results[1].worst_thermal) results[1].worst_thermal = tl;

            if (results[1].fwd_passes % 100 == 0) {
                fprintf(stderr, "POWER: GPU %d passes, %.1fs elapsed\n",
                    results[1].fwd_passes, TBS2(mach_absolute_time() - t_start));
            }
        }
        results[1].elapsed_s = TBS2(mach_absolute_time() - t_start);
        results[1].batt_end = read_battery_sync();
        [out appendFormat:@"  GPU done: %d passes in %.1fs, battery %.0f%%→%.0f%%\n",
            results[1].fwd_passes, results[1].elapsed_s,
            results[1].batt_start*100, results[1].batt_end*100];
        gpu_free(gpu);
    }

    [out appendString:@"  Cooling down 30s...\n"];
    fprintf(stderr, "POWER: Cooling down 30s before CPU...\n");
    usleep(30000000);

    // ═══════════════════════════════
    // 3. CPU
    // ═══════════════════════════════
    [out appendString:@"  Running CPU...\n"];
    {
        for (int i = 0; i < 5; i++) cpu_forward(&sw, &bufs, tokens);

        usleep(500000);
        results[2].name = "CPU";
        results[2].batt_start = read_battery_sync();
        results[2].worst_thermal = ThermalNominal;

        uint64_t t_start = mach_absolute_time();
        fprintf(stderr, "POWER: CPU running for %.0f min...\n", minutes_per_backend);

        while (TBS2(mach_absolute_time() - t_start) < target_s) {
            uint64_t ts = mach_absolute_time();
            cpu_forward(&sw, &bufs, tokens);
            results[2].total_ms += TBM2(mach_absolute_time() - ts);
            results[2].fwd_passes++;
            ThermalLevel tl = current_thermal_level();
            if (tl > results[2].worst_thermal) results[2].worst_thermal = tl;

            if (results[2].fwd_passes % 500 == 0) {
                fprintf(stderr, "POWER: CPU %d passes, %.1fs elapsed\n",
                    results[2].fwd_passes, TBS2(mach_absolute_time() - t_start));
            }
        }
        results[2].elapsed_s = TBS2(mach_absolute_time() - t_start);
        results[2].batt_end = read_battery_sync();
        [out appendFormat:@"  CPU done: %d passes in %.1fs, battery %.0f%%→%.0f%%\n",
            results[2].fwd_passes, results[2].elapsed_s,
            results[2].batt_start*100, results[2].batt_end*100];
    }

    // ═══════════════════════════════
    // RESULTS
    // ═══════════════════════════════
    [out appendString:@"\n"];
    [out appendString:@"  ┌─── POWER COMPARISON ───────────────────────────────────┐\n"];
    [out appendString:@"  │                                                        │\n"];
    [out appendString:@"  │ Backend   Drain  Watts  Tok/s  Tok/J   Thermal        │\n"];
    [out appendString:@"  │ ──────── ────── ────── ────── ─────── ────────        │\n"];

    double best_tpj = 0;
    const char *most_efficient = "?";

    for (int i = 0; i < 3; i++) {
        BackendResult *r = &results[i];
        if (!r->name || r->fwd_passes == 0) continue;

        float drain = r->batt_start - r->batt_end;
        double energyWh = drain * battWh;
        double watts = (drain > 0) ? energyWh / (r->elapsed_s / 3600.0) : 0;
        double tps = (r->fwd_passes * SEQ) / r->elapsed_s;
        double tpj = (energyWh > 0) ? (r->fwd_passes * SEQ) / (energyWh * 3600.0) : 0;

        if (drain > 0) {
            [out appendFormat:@"  │ %-8s %4.1f%%  %5.2fW  %5.0f  %6.0f   %-8s        │\n",
                r->name, drain*100, watts, tps, tpj,
                thermal_level_name(r->worst_thermal)];
        } else {
            [out appendFormat:@"  │ %-8s %4.1f%%    N/A  %5.0f     N/A   %-8s        │\n",
                r->name, drain*100, tps,
                thermal_level_name(r->worst_thermal)];
        }

        if (tpj > best_tpj) { best_tpj = tpj; most_efficient = r->name; }
    }

    [out appendString:@"  │                                                        │\n"];
    [out appendFormat:@"  │ Most efficient: %s (%.0f tokens/Joule)              \n",
        most_efficient, best_tpj];
    [out appendString:@"  │                                                        │\n"];

    // Detailed breakdown
    [out appendString:@"  │ Drain = battery %% used                                │\n"];
    [out appendString:@"  │ Watts = energy / time (includes display+system)       │\n"];
    [out appendString:@"  │ Tok/J = tokens processed per Joule of energy          │\n"];
    [out appendString:@"  └────────────────────────────────────────────────────────┘\n"];

    // Data quality
    [out appendString:@"\n  ┌─── DATA QUALITY ────────────────────────────────────────┐\n"];
    for (int i = 0; i < 3; i++) {
        BackendResult *r = &results[i];
        if (!r->name) continue;
        float drain = r->batt_start - r->batt_end;
        const char *conf = "N/A";
        if (drain >= 0.05) conf = "HIGH";
        else if (drain >= 0.02) conf = "MEDIUM";
        else if (drain > 0) conf = "LOW";
        [out appendFormat:@"  │ %-8s  %d passes, %.1f%% drain → %s confidence    \n",
            r->name, r->fwd_passes, drain*100, conf];
    }
    [out appendString:@"  └────────────────────────────────────────────────────────┘\n"];

    cpu_bufs_free(&bufs);
    free_shared_weights(&sw);
    thermal_disable_battery_monitoring();

    fprintf(stderr, "POWER: Benchmark complete.\n");
    #undef TBM2
    #undef TBS2
    return out;
    }
}
