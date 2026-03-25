// ANEGPUTraining.m — GPU-based Stories-110M training via Metal Performance Shaders
// Forward: GPU matmuls + CPU attention/norms. Backward: GPU matmuls + CPU ops.
// NO kernel recompilation — weights in SharedMemory MTLBuffers, zero-copy Adam.
#import "ANEGPUTraining.h"
#import "ANETrainingConfig.h"
#import "ANEStoriesCPUOps.h"
#import "ANEThermal.h"
#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#import <mach/mach_time.h>
#import <mach/mach.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

static mach_timebase_info_data_t g_gpu_tb;
static void gpu_tb_init(void) { static dispatch_once_t o; dispatch_once(&o, ^{ mach_timebase_info(&g_gpu_tb); }); }
#define GPU_MS(t) ((double)(t)*g_gpu_tb.numer/g_gpu_tb.denom/1e6)
#define GPU_S(t)  ((double)(t)*g_gpu_tb.numer/g_gpu_tb.denom/1e9)

// ═══════════════════════════════════════════════════════
// TRAINING STATE
// ═══════════════════════════════════════════════════════

// Extended LayerActs to include attention scores for backward
typedef struct {
    float *layer_in;           // [DIM, SEQ] saved for rmsnorm1 backward
    float *xnorm;              // [DIM, SEQ]
    float *Q, *K, *V;          // [DIM, SEQ] each
    float *attn_out;           // [DIM, SEQ]
    float *scores;             // [HEADS * SEQ * SEQ] saved softmax output for attn backward
    float *o_out;              // [DIM, SEQ]
    float *x2;                 // [DIM, SEQ]
    float *x2norm;             // [DIM, SEQ]
    float *h1, *h3;            // [HIDDEN, SEQ]
    float *silu_out;           // [HIDDEN, SEQ]
    float *ffn_out;            // [DIM, SEQ]
} GPULayerActs;

struct GPUTrainState {
    // Metal objects
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;

    // Weight MTLBuffers per layer (SharedMemory — CPU writes, GPU reads)
    id<MTLBuffer> buf_wq[NLAYERS], buf_wk[NLAYERS], buf_wv[NLAYERS], buf_wo[NLAYERS];
    id<MTLBuffer> buf_w1[NLAYERS], buf_w2[NLAYERS], buf_w3[NLAYERS];

    // Pre-created MPS matmul objects (reused every step)
    MPSMatrixMultiplication *mm_dim;        // [DIM,DIM] @ [DIM,SEQ] → [DIM,SEQ]
    MPSMatrixMultiplication *mm_up;         // [HIDDEN,DIM] @ [DIM,SEQ] → [HIDDEN,SEQ]
    MPSMatrixMultiplication *mm_down;       // [DIM,HIDDEN] @ [HIDDEN,SEQ] → [DIM,SEQ]
    MPSMatrixMultiplication *mm_dim_T;      // [DIM,DIM]^T @ [DIM,SEQ] → [DIM,SEQ]
    MPSMatrixMultiplication *mm_up_T;       // [DIM,HIDDEN] (=W1^T shape) @ [HIDDEN,SEQ] → [DIM,SEQ]
    MPSMatrixMultiplication *mm_down_T;     // [HIDDEN,DIM] (=W2^T shape) @ [DIM,SEQ] → [HIDDEN,SEQ]
    // Gradient accum (beta=1)
    MPSMatrixMultiplication *mm_dW_dim;     // [DIM,SEQ] @ [SEQ,DIM] → [DIM,DIM] accum
    MPSMatrixMultiplication *mm_dW_up;      // [HIDDEN,SEQ] @ [SEQ,DIM] → [HIDDEN,DIM] accum
    MPSMatrixMultiplication *mm_dW_down;    // [DIM,SEQ] @ [SEQ,HIDDEN] → [DIM,HIDDEN] accum

    // Intermediate GPU buffers (reused between layers)
    id<MTLBuffer> buf_x, buf_norm;
    id<MTLBuffer> buf_Q, buf_K, buf_V, buf_attn, buf_o;
    id<MTLBuffer> buf_h1, buf_h3, buf_silu, buf_ffn;
    // Backward intermediates
    id<MTLBuffer> buf_dy, buf_dffn, buf_dsilu, buf_dh1, buf_dh3;
    id<MTLBuffer> buf_dx, buf_dx2, buf_do;
    id<MTLBuffer> buf_dq, buf_dk, buf_dv;

    // Gradient MTLBuffers per layer (for GPU-computed dW, SharedMemory)
    id<MTLBuffer> buf_gWq[NLAYERS], buf_gWk[NLAYERS], buf_gWv[NLAYERS], buf_gWo[NLAYERS];
    id<MTLBuffer> buf_gW1[NLAYERS], buf_gW2[NLAYERS], buf_gW3[NLAYERS];

    // CPU state — weight pointers alias into MTLBuffer.contents
    LayerWeights lw[NLAYERS];
    LayerAdam la[NLAYERS];
    GPULayerActs acts[NLAYERS];
    // RMS norm gradients (CPU-only, small)
    float *grms_att[NLAYERS], *grms_ffn[NLAYERS];

    // Global
    float *rms_final, *embed;
    float *grms_final, *gembed;
    AdamState arms_final, aembed;

    // Scratch
    float *x_cur, *x_final;
    float *logits, *dlogits;
    float *dy;

    // Token data
    uint16_t *token_data;
    size_t n_tokens, data_len;
    int data_fd;

    // Training state
    int step, adam_t, accum_count;
    float lr, last_loss;
};

// ═══════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════

static void gpu_silu_inplace(float *x, int n) {
    float *neg = (float*)malloc(n*4);
    float m1 = -1.0f;
    vDSP_vsmul(x, 1, &m1, neg, 1, (vDSP_Length)n);
    vvexpf(neg, neg, &n);
    float one = 1.0f;
    vDSP_vsadd(neg, 1, &one, neg, 1, (vDSP_Length)n);
    vDSP_vdiv(neg, 1, x, 1, x, 1, (vDSP_Length)n);
    free(neg);
}

// MPS encode helper
static void gpu_encode(GPUTrainState *s, MPSMatrixMultiplication *mm, id<MTLCommandBuffer> cmd,
                       id<MTLBuffer> bA, int rA, int cA,
                       id<MTLBuffer> bB, int rB, int cB,
                       id<MTLBuffer> bC, int rC, int cC) {
    MPSMatrixDescriptor *dA = [MPSMatrixDescriptor matrixDescriptorWithRows:rA columns:cA rowBytes:cA*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dB = [MPSMatrixDescriptor matrixDescriptorWithRows:rB columns:cB rowBytes:cB*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dC = [MPSMatrixDescriptor matrixDescriptorWithRows:rC columns:cC rowBytes:cC*4 dataType:MPSDataTypeFloat32];
    MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:bA descriptor:dA];
    MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:bB descriptor:dB];
    MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:bC descriptor:dC];
    [mm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
}

// CPU attention forward (reused from benchmark)
static void gpu_cpu_attention(float *out, const float *Q, const float *K, const float *V,
                              float *scores_out, int dim, int seq, int nh) {
    int hd = dim / nh;
    for (int h = 0; h < nh; h++) {
        const float *Qh = Q + h*hd*seq, *Kh = K + h*hd*seq, *Vh = V + h*hd*seq;
        float *sc = scores_out + h*seq*seq;
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, seq, seq, hd, 1, Qh, seq, Kh, seq, 0, sc, seq);
        float scale = 1.0f / sqrtf((float)hd);
        vDSP_vsmul(sc, 1, &scale, sc, 1, (vDSP_Length)(seq*seq));
        for (int i = 0; i < seq; i++)
            for (int j = i+1; j < seq; j++)
                sc[i*seq + j] = -1e9f;
        for (int i = 0; i < seq; i++) {
            float *r = sc + i*seq;
            float mx; vDSP_maxv(r, 1, &mx, (vDSP_Length)seq);
            float nm = -mx; vDSP_vsadd(r, 1, &nm, r, 1, (vDSP_Length)seq);
            int n2 = seq; vvexpf(r, r, &n2);
            float sm; vDSP_sve(r, 1, &sm, (vDSP_Length)seq);
            float inv = 1.0f/sm; vDSP_vsmul(r, 1, &inv, r, 1, (vDSP_Length)seq);
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, hd, seq, seq, 1, Vh, seq, sc, seq, 0, out+h*hd*seq, seq);
    }
}

// CPU attention backward — NEW
static void gpu_cpu_attention_backward(float *dQ, float *dK, float *dV,
                                       const float *do_out, const float *Q, const float *K, const float *V,
                                       const float *scores, int dim, int seq, int nh) {
    int hd = dim / nh;
    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < nh; h++) {
        const float *doh = do_out + h*hd*seq;
        const float *Qh = Q + h*hd*seq, *Kh = K + h*hd*seq, *Vh = V + h*hd*seq;
        const float *sch = scores + h*seq*seq;
        float *dQh = dQ + h*hd*seq, *dKh = dK + h*hd*seq, *dVh = dV + h*hd*seq;

        float *dsc = (float*)calloc(seq*seq, sizeof(float));

        // d_scores = V^T @ do (for softmax backward)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, seq, seq, hd, 1, Vh, seq, doh, seq, 0, dsc, seq);
        // dV = do @ scores
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, hd, seq, seq, 1, doh, seq, sch, seq, 0, dVh, seq);

        // Softmax backward: ds = scores * (ds - rowwise_dot(ds, scores))
        for (int i = 0; i < seq; i++) {
            float *dr = dsc + i*seq;
            const float *sr = sch + i*seq;
            float dot;
            vDSP_dotpr(dr, 1, sr, 1, &dot, (vDSP_Length)seq);
            float neg_dot = -dot;
            vDSP_vsadd(dr, 1, &neg_dot, dr, 1, (vDSP_Length)seq);
            vDSP_vmul(dr, 1, sr, 1, dr, 1, (vDSP_Length)seq);
        }
        // Causal mask
        for (int i = 0; i < seq; i++)
            for (int j = i+1; j < seq; j++)
                dsc[i*seq + j] = 0;
        // Scale
        vDSP_vsmul(dsc, 1, &scale, dsc, 1, (vDSP_Length)(seq*seq));

        // dQ = K @ d_scores^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, hd, seq, seq, 1, Kh, seq, dsc, seq, 0, dQh, seq);
        // dK = Q @ d_scores
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, hd, seq, seq, 1, Qh, seq, dsc, seq, 0, dKh, seq);

        free(dsc);
    }
}

// ═══════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════

GPUTrainState *gpu_train_init(const char *model_path, const char *data_path) {
    @autoreleasepool {
    gpu_tb_init();
    GPUTrainState *s = (GPUTrainState*)calloc(1, sizeof(GPUTrainState));
    s->lr = 3e-4f;
    s->last_loss = 999.0f;

    // Metal setup
    s->device = MTLCreateSystemDefaultDevice();
    if (!s->device) { fprintf(stderr, "[gpu-train] No Metal device\n"); free(s); return NULL; }
    s->queue = [s->device newCommandQueue];

    // Allocate weight MTLBuffers + alias to CPU pointers (zero-copy)
    for (int L = 0; L < NLAYERS; L++) {
        s->buf_wq[L] = [s->device newBufferWithLength:WQ_SZ*4 options:MTLResourceStorageModeShared];
        s->buf_wk[L] = [s->device newBufferWithLength:WQ_SZ*4 options:MTLResourceStorageModeShared];
        s->buf_wv[L] = [s->device newBufferWithLength:WQ_SZ*4 options:MTLResourceStorageModeShared];
        s->buf_wo[L] = [s->device newBufferWithLength:WO_SZ*4 options:MTLResourceStorageModeShared];
        s->buf_w1[L] = [s->device newBufferWithLength:W1_SZ*4 options:MTLResourceStorageModeShared];
        s->buf_w2[L] = [s->device newBufferWithLength:W2_SZ*4 options:MTLResourceStorageModeShared];
        s->buf_w3[L] = [s->device newBufferWithLength:W3_SZ*4 options:MTLResourceStorageModeShared];
        // Alias CPU pointers into MTLBuffer contents
        s->lw[L].Wq = (float*)s->buf_wq[L].contents;
        s->lw[L].Wk = (float*)s->buf_wk[L].contents;
        s->lw[L].Wv = (float*)s->buf_wv[L].contents;
        s->lw[L].Wo = (float*)s->buf_wo[L].contents;
        s->lw[L].W1 = (float*)s->buf_w1[L].contents;
        s->lw[L].W2 = (float*)s->buf_w2[L].contents;
        s->lw[L].W3 = (float*)s->buf_w3[L].contents;
        s->lw[L].rms_att = (float*)malloc(DIM*4);
        s->lw[L].rms_ffn = (float*)malloc(DIM*4);

        // Adam state
        s->la[L] = layer_adam_alloc();

        // Activations (CPU, saved for backward)
        GPULayerActs *a = &s->acts[L];
        a->layer_in = (float*)malloc(SEQ*DIM*4);
        a->xnorm = (float*)malloc(SEQ*DIM*4);
        a->Q = (float*)malloc(SEQ*DIM*4);
        a->K = (float*)malloc(SEQ*DIM*4);
        a->V = (float*)malloc(SEQ*DIM*4);
        a->attn_out = (float*)malloc(SEQ*DIM*4);
        a->scores = (float*)malloc(HEADS*SEQ*SEQ*4);
        a->o_out = (float*)malloc(SEQ*DIM*4);
        a->x2 = (float*)malloc(SEQ*DIM*4);
        a->x2norm = (float*)malloc(SEQ*DIM*4);
        a->h1 = (float*)malloc(SEQ*HIDDEN*4);
        a->h3 = (float*)malloc(SEQ*HIDDEN*4);
        a->silu_out = (float*)malloc(SEQ*HIDDEN*4);
        a->ffn_out = (float*)malloc(SEQ*DIM*4);

        // RMS norm gradients (small, CPU only)
        s->grms_att[L] = (float*)calloc(DIM, 4);
        s->grms_ffn[L] = (float*)calloc(DIM, 4);

        // Gradient MTLBuffers
        s->buf_gWq[L] = [s->device newBufferWithLength:WQ_SZ*4 options:MTLResourceStorageModeShared];
        s->buf_gWk[L] = [s->device newBufferWithLength:WQ_SZ*4 options:MTLResourceStorageModeShared];
        s->buf_gWv[L] = [s->device newBufferWithLength:WQ_SZ*4 options:MTLResourceStorageModeShared];
        s->buf_gWo[L] = [s->device newBufferWithLength:WO_SZ*4 options:MTLResourceStorageModeShared];
        s->buf_gW1[L] = [s->device newBufferWithLength:W1_SZ*4 options:MTLResourceStorageModeShared];
        s->buf_gW2[L] = [s->device newBufferWithLength:W2_SZ*4 options:MTLResourceStorageModeShared];
        s->buf_gW3[L] = [s->device newBufferWithLength:W3_SZ*4 options:MTLResourceStorageModeShared];
    }

    // Global weights
    s->rms_final = (float*)malloc(DIM*4);
    s->embed = (float*)malloc((size_t)VOCAB*DIM*4);
    s->grms_final = (float*)calloc(DIM, 4);
    s->gembed = (float*)calloc((size_t)VOCAB*DIM, 4);
    s->arms_final = adam_alloc(DIM);
    s->aembed = adam_alloc((size_t)VOCAB*DIM);

    // Scratch
    s->x_cur = (float*)malloc(SEQ*DIM*4);
    s->x_final = (float*)malloc(SEQ*DIM*4);
    s->logits = (float*)malloc(SEQ*VOCAB*4);
    s->dlogits = (float*)malloc(SEQ*VOCAB*4);
    s->dy = (float*)malloc(SEQ*DIM*4);

    // Random init weights (into aliased MTLBuffer memory)
    srand48(42);
    float sd = 1.0f/sqrtf(DIM), sh = 1.0f/sqrtf(HIDDEN), os = 1.0f/sqrtf((float)NLAYERS);
    for (int L = 0; L < NLAYERS; L++) {
        for (size_t i=0;i<WQ_SZ;i++){s->lw[L].Wq[i]=sd*(2*drand48()-1);s->lw[L].Wk[i]=sd*(2*drand48()-1);}
        for (size_t i=0;i<WQ_SZ;i++){s->lw[L].Wv[i]=sd*(2*drand48()-1);s->lw[L].Wo[i]=sd*os*(2*drand48()-1);}
        for (size_t i=0;i<W1_SZ;i++) s->lw[L].W1[i]=sh*(2*drand48()-1);
        for (size_t i=0;i<W2_SZ;i++) s->lw[L].W2[i]=sd*os*(2*drand48()-1);
        for (size_t i=0;i<W3_SZ;i++) s->lw[L].W3[i]=sh*(2*drand48()-1);
        for (int i=0;i<DIM;i++){s->lw[L].rms_att[i]=1;s->lw[L].rms_ffn[i]=1;}
    }
    for (int i=0;i<DIM;i++) s->rms_final[i]=1;
    for (size_t i=0;i<(size_t)VOCAB*DIM;i++) s->embed[i]=0.02f*(2*drand48()-1);

    // mmap token data
    s->data_fd = -1; s->token_data = NULL; s->n_tokens = 0;
    if (data_path) {
        s->data_fd = open(data_path, O_RDONLY);
        if (s->data_fd >= 0) {
            struct stat st; fstat(s->data_fd, &st);
            s->data_len = st.st_size;
            s->token_data = (uint16_t*)mmap(NULL, s->data_len, PROT_READ, MAP_PRIVATE, s->data_fd, 0);
            if (s->token_data == MAP_FAILED) { s->token_data = NULL; close(s->data_fd); s->data_fd = -1; }
            else s->n_tokens = s->data_len / 2;
        }
    }

    // GPU intermediate buffers
    s->buf_x = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_norm = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_Q = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_K = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_V = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_attn = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_o = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_h1 = [s->device newBufferWithLength:SEQ*HIDDEN*4 options:MTLResourceStorageModeShared];
    s->buf_h3 = [s->device newBufferWithLength:SEQ*HIDDEN*4 options:MTLResourceStorageModeShared];
    s->buf_silu = [s->device newBufferWithLength:SEQ*HIDDEN*4 options:MTLResourceStorageModeShared];
    s->buf_ffn = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_dy = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_dffn = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_dsilu = [s->device newBufferWithLength:SEQ*HIDDEN*4 options:MTLResourceStorageModeShared];
    s->buf_dh1 = [s->device newBufferWithLength:SEQ*HIDDEN*4 options:MTLResourceStorageModeShared];
    s->buf_dh3 = [s->device newBufferWithLength:SEQ*HIDDEN*4 options:MTLResourceStorageModeShared];
    s->buf_dx = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_dx2 = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_do = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_dq = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_dk = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];
    s->buf_dv = [s->device newBufferWithLength:SEQ*DIM*4 options:MTLResourceStorageModeShared];

    // Pre-create MPS matmul objects
    s->mm_dim = [[MPSMatrixMultiplication alloc] initWithDevice:s->device transposeLeft:NO transposeRight:NO
        resultRows:DIM resultColumns:SEQ interiorColumns:DIM alpha:1 beta:0];
    s->mm_up = [[MPSMatrixMultiplication alloc] initWithDevice:s->device transposeLeft:NO transposeRight:NO
        resultRows:HIDDEN resultColumns:SEQ interiorColumns:DIM alpha:1 beta:0];
    s->mm_down = [[MPSMatrixMultiplication alloc] initWithDevice:s->device transposeLeft:NO transposeRight:NO
        resultRows:DIM resultColumns:SEQ interiorColumns:HIDDEN alpha:1 beta:0];
    // For backward: instead of transpose MPS objects (which have tricky dimension semantics),
    // use non-transpose matmuls and swap operands. E.g. W^T @ x = (x^T @ W)^T
    // But simpler: just create separate non-transpose MPS for each backward shape.
    // W[DIM,DIM]^T @ x[DIM,SEQ] → [DIM,SEQ]: same as mm_dim (symmetric)
    s->mm_dim_T = s->mm_dim; // DIM x DIM is symmetric under transpose
    // W1[HIDDEN,DIM]^T @ dh[HIDDEN,SEQ] → [DIM,SEQ]: need [DIM,HIDDEN] @ [HIDDEN,SEQ]
    s->mm_up_T = [[MPSMatrixMultiplication alloc] initWithDevice:s->device transposeLeft:NO transposeRight:NO
        resultRows:DIM resultColumns:SEQ interiorColumns:HIDDEN alpha:1 beta:0];
    // W2[DIM,HIDDEN]^T @ dffn[DIM,SEQ] → [HIDDEN,SEQ]: need [HIDDEN,DIM] @ [DIM,SEQ]
    s->mm_down_T = [[MPSMatrixMultiplication alloc] initWithDevice:s->device transposeLeft:NO transposeRight:NO
        resultRows:HIDDEN resultColumns:SEQ interiorColumns:DIM alpha:1 beta:0];
    // Gradient accumulation: dW = dy @ x^T
    // dy[M,SEQ] @ x[N,SEQ]^T → [M,N]: use transposeRight
    s->mm_dW_dim = [[MPSMatrixMultiplication alloc] initWithDevice:s->device transposeLeft:NO transposeRight:YES
        resultRows:DIM resultColumns:DIM interiorColumns:SEQ alpha:1 beta:1];
    s->mm_dW_up = [[MPSMatrixMultiplication alloc] initWithDevice:s->device transposeLeft:NO transposeRight:YES
        resultRows:HIDDEN resultColumns:DIM interiorColumns:SEQ alpha:1 beta:1];
    s->mm_dW_down = [[MPSMatrixMultiplication alloc] initWithDevice:s->device transposeLeft:NO transposeRight:YES
        resultRows:DIM resultColumns:HIDDEN interiorColumns:SEQ alpha:1 beta:1];

    // Zero gradient buffers
    for (int L = 0; L < NLAYERS; L++) {
        memset(s->buf_gWq[L].contents, 0, WQ_SZ*4);
        memset(s->buf_gWk[L].contents, 0, WQ_SZ*4);
        memset(s->buf_gWv[L].contents, 0, WQ_SZ*4);
        memset(s->buf_gWo[L].contents, 0, WO_SZ*4);
        memset(s->buf_gW1[L].contents, 0, W1_SZ*4);
        memset(s->buf_gW2[L].contents, 0, W2_SZ*4);
        memset(s->buf_gW3[L].contents, 0, W3_SZ*4);
        memset(s->grms_att[L], 0, DIM*4);
        memset(s->grms_ffn[L], 0, DIM*4);
    }
    memset(s->grms_final, 0, DIM*4);
    memset(s->gembed, 0, (size_t)VOCAB*DIM*4);

    fprintf(stderr, "[gpu-train] Init complete. Device: %s\n", [s->device.name UTF8String]);
    return s;
    }
}

// ═══════════════════════════════════════════════════════
// TRAIN STEP
// ═══════════════════════════════════════════════════════

float gpu_train_step(GPUTrainState *s) {
    @autoreleasepool {

    // Sample random position
    size_t max_pos = s->n_tokens - SEQ - 1;
    size_t pos = (size_t)(drand48() * max_pos);
    uint16_t *input_tokens = s->token_data + pos;
    uint16_t *target_tokens = s->token_data + pos + 1;

    // ===== FORWARD =====
    memset(s->x_cur, 0, SEQ*DIM*4);
    embed_lookup(s->x_cur, s->embed, input_tokens, DIM, SEQ);

    for (int L = 0; L < NLAYERS; L++) {
        GPULayerActs *a = &s->acts[L];
        memcpy(a->layer_in, s->x_cur, SEQ*DIM*4);

        // RMSNorm1 (CPU)
        rmsnorm(a->xnorm, s->x_cur, s->lw[L].rms_att, DIM, SEQ);
        memcpy(s->buf_norm.contents, a->xnorm, SEQ*DIM*4);

        // QKV (GPU, batched)
        id<MTLCommandBuffer> cmd1 = [s->queue commandBuffer];
        gpu_encode(s, s->mm_dim, cmd1, s->buf_wq[L],DIM,DIM, s->buf_norm,DIM,SEQ, s->buf_Q,DIM,SEQ);
        gpu_encode(s, s->mm_dim, cmd1, s->buf_wk[L],DIM,DIM, s->buf_norm,DIM,SEQ, s->buf_K,DIM,SEQ);
        gpu_encode(s, s->mm_dim, cmd1, s->buf_wv[L],DIM,DIM, s->buf_norm,DIM,SEQ, s->buf_V,DIM,SEQ);
        [cmd1 commit]; [cmd1 waitUntilCompleted];

        memcpy(a->Q, s->buf_Q.contents, SEQ*DIM*4);
        memcpy(a->K, s->buf_K.contents, SEQ*DIM*4);
        memcpy(a->V, s->buf_V.contents, SEQ*DIM*4);

        // Attention (CPU)
        gpu_cpu_attention(a->attn_out, a->Q, a->K, a->V, a->scores, DIM, SEQ, HEADS);

        // Wo (GPU)
        memcpy(s->buf_attn.contents, a->attn_out, SEQ*DIM*4);
        id<MTLCommandBuffer> cmd2 = [s->queue commandBuffer];
        gpu_encode(s, s->mm_dim, cmd2, s->buf_wo[L],DIM,DIM, s->buf_attn,DIM,SEQ, s->buf_o,DIM,SEQ);
        [cmd2 commit]; [cmd2 waitUntilCompleted];
        memcpy(a->o_out, s->buf_o.contents, SEQ*DIM*4);

        // Residual
        vDSP_vadd(s->x_cur, 1, a->o_out, 1, a->x2, 1, (vDSP_Length)(SEQ*DIM));

        // RMSNorm2 (CPU)
        rmsnorm(a->x2norm, a->x2, s->lw[L].rms_ffn, DIM, SEQ);
        memcpy(s->buf_norm.contents, a->x2norm, SEQ*DIM*4);

        // W1, W3 (GPU, batched)
        id<MTLCommandBuffer> cmd3 = [s->queue commandBuffer];
        gpu_encode(s, s->mm_up, cmd3, s->buf_w1[L],HIDDEN,DIM, s->buf_norm,DIM,SEQ, s->buf_h1,HIDDEN,SEQ);
        gpu_encode(s, s->mm_up, cmd3, s->buf_w3[L],HIDDEN,DIM, s->buf_norm,DIM,SEQ, s->buf_h3,HIDDEN,SEQ);
        [cmd3 commit]; [cmd3 waitUntilCompleted];
        memcpy(a->h1, s->buf_h1.contents, SEQ*HIDDEN*4);
        memcpy(a->h3, s->buf_h3.contents, SEQ*HIDDEN*4);

        // SiLU + gate (CPU)
        memcpy(a->silu_out, a->h1, SEQ*HIDDEN*4);
        gpu_silu_inplace(a->silu_out, HIDDEN*SEQ);
        // silu_out = SiLU(h1) — now multiply by h3
        vDSP_vmul(a->silu_out, 1, a->h3, 1, a->silu_out, 1, (vDSP_Length)(HIDDEN*SEQ));

        // W2 (GPU)
        memcpy(s->buf_silu.contents, a->silu_out, SEQ*HIDDEN*4);
        id<MTLCommandBuffer> cmd4 = [s->queue commandBuffer];
        gpu_encode(s, s->mm_down, cmd4, s->buf_w2[L],DIM,HIDDEN, s->buf_silu,HIDDEN,SEQ, s->buf_ffn,DIM,SEQ);
        [cmd4 commit]; [cmd4 waitUntilCompleted];
        memcpy(a->ffn_out, s->buf_ffn.contents, SEQ*DIM*4);

        // Residual
        vDSP_vadd(a->x2, 1, a->ffn_out, 1, s->x_cur, 1, (vDSP_Length)(SEQ*DIM));
    }

    // Final RMSNorm + Classifier (CPU)
    rmsnorm(s->x_final, s->x_cur, s->rms_final, DIM, SEQ);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                VOCAB, SEQ, DIM, 1, s->embed, DIM, s->x_final, SEQ, 0, s->logits, SEQ);
    float loss = cross_entropy_loss(s->dlogits, s->logits, target_tokens, VOCAB, SEQ);
    s->last_loss = loss;

    // ===== BACKWARD =====

    // Classifier backward (CPU)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                DIM, SEQ, VOCAB, 1, s->embed, DIM, s->dlogits, SEQ, 0, s->dy, SEQ);

    // dEmbed accumulate (CPU)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                VOCAB, DIM, SEQ, 1, s->dlogits, SEQ, s->x_final, SEQ, 1, s->gembed, DIM);

    // Final RMSNorm backward (CPU)
    float *dx_rms = (float*)calloc(SEQ*DIM, 4);
    rmsnorm_bwd(dx_rms, s->grms_final, s->dy, s->x_cur, s->rms_final, DIM, SEQ);
    memcpy(s->dy, dx_rms, SEQ*DIM*4);
    free(dx_rms);

    // ===== BACKWARD (layers, reverse) =====
    float *dffn = (float*)malloc(SEQ*DIM*4);
    float *dsilu = (float*)malloc(SEQ*HIDDEN*4);
    float *dh1 = (float*)malloc(SEQ*HIDDEN*4);
    float *dh3 = (float*)malloc(SEQ*HIDDEN*4);
    float *dx_ffn = (float*)malloc(SEQ*DIM*4);
    float *dx2 = (float*)malloc(SEQ*DIM*4);
    float *do_out = (float*)malloc(SEQ*DIM*4);
    float *dQ = (float*)malloc(SEQ*DIM*4);
    float *dK = (float*)malloc(SEQ*DIM*4);
    float *dV = (float*)malloc(SEQ*DIM*4);
    float *dx_attn = (float*)malloc(SEQ*DIM*4);

    for (int L = NLAYERS - 1; L >= 0; L--) {
        GPULayerActs *a = &s->acts[L];

        memcpy(dffn, s->dy, SEQ*DIM*4);

        // --- FFN Backward ---
        // dsilu = W2^T @ dffn (CPU cblas — simpler than GPU transpose management)
        memcpy(dffn, s->dy, SEQ*DIM*4); // dffn = dy for this layer's ffn path
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    HIDDEN, SEQ, DIM, 1, s->lw[L].W2, HIDDEN, dffn, SEQ, 0, dsilu, SEQ);

        // SiLU backward chain rule (CPU)
        // d(SiLU(h1)*h3)/dh1 = h3 * SiLU'(h1), d/dh3 = SiLU(h1)
        // SiLU'(x) = sigmoid(x) * (1 + x*(1-sigmoid(x)))
        // But simpler: silu_out = SiLU(h1)*h3, dsilu comes from W2^T@dffn
        // dh3 = dsilu * SiLU(h1)
        // d_silu_h1 = dsilu * h3
        // dh1 = d_silu_h1 * SiLU'(h1) where SiLU'(x) = sig(x)(1+x(1-sig(x)))
        {
            // Compute sigmoid(h1) for SiLU derivative
            float *sig = (float*)malloc(HIDDEN*SEQ*4);
            float neg1 = -1; int n = HIDDEN*SEQ;
            vDSP_vsmul(a->h1, 1, &neg1, sig, 1, (vDSP_Length)n);
            vvexpf(sig, sig, &n);
            float one = 1;
            vDSP_vsadd(sig, 1, &one, sig, 1, (vDSP_Length)n);
            // sig = 1/(1+exp(-h1)) = sigmoid
            float *ones_arr = (float*)malloc(n*4);
            for (int i=0;i<n;i++) ones_arr[i]=1;
            vDSP_vdiv(sig, 1, ones_arr, 1, sig, 1, (vDSP_Length)n);
            free(ones_arr);

            // SiLU(h1) = h1 * sig
            float *silu_h1 = (float*)malloc(n*4);
            vDSP_vmul(a->h1, 1, sig, 1, silu_h1, 1, (vDSP_Length)n);

            // dh3 = dsilu * SiLU(h1)
            vDSP_vmul(dsilu, 1, silu_h1, 1, dh3, 1, (vDSP_Length)n);

            // d_silu_h1 = dsilu * h3
            float *d_silu_h1 = (float*)malloc(n*4);
            vDSP_vmul(dsilu, 1, a->h3, 1, d_silu_h1, 1, (vDSP_Length)n);

            // SiLU'(x) = sig(x) * (1 + x*(1-sig(x)))
            float *silu_deriv = (float*)malloc(n*4);
            // 1 - sig
            float neg1b = -1;
            vDSP_vsmul(sig, 1, &neg1b, silu_deriv, 1, (vDSP_Length)n);
            vDSP_vsadd(silu_deriv, 1, &one, silu_deriv, 1, (vDSP_Length)n); // 1-sig
            vDSP_vmul(silu_deriv, 1, a->h1, 1, silu_deriv, 1, (vDSP_Length)n); // x*(1-sig)
            vDSP_vsadd(silu_deriv, 1, &one, silu_deriv, 1, (vDSP_Length)n); // 1+x*(1-sig)
            vDSP_vmul(silu_deriv, 1, sig, 1, silu_deriv, 1, (vDSP_Length)n); // sig*(1+x*(1-sig))

            // dh1 = d_silu_h1 * silu_deriv
            vDSP_vmul(d_silu_h1, 1, silu_deriv, 1, dh1, 1, (vDSP_Length)n);

            free(sig); free(silu_h1); free(d_silu_h1); free(silu_deriv);
        }

        // dx_ffn = W1^T @ dh1 + W3^T @ dh3 (CPU cblas)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, SEQ, HIDDEN, 1, s->lw[L].W1, DIM, dh1, SEQ, 0, dx_ffn, SEQ);
        // + W3^T @ dh3
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, SEQ, HIDDEN, 1, s->lw[L].W3, DIM, dh3, SEQ, 1, dx_ffn, SEQ);

        // dW accumulation (CPU cblas, beta=1 to accumulate)
        // dW2 += dffn @ silu_out^T  [DIM,HIDDEN]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    DIM, HIDDEN, SEQ, 1, dffn, SEQ, a->silu_out, SEQ, 1, (float*)s->buf_gW2[L].contents, HIDDEN);
        // dW1 += dh1 @ x2norm^T  [HIDDEN,DIM]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    HIDDEN, DIM, SEQ, 1, dh1, SEQ, a->x2norm, SEQ, 1, (float*)s->buf_gW1[L].contents, DIM);
        // dW3 += dh3 @ x2norm^T  [HIDDEN,DIM]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    HIDDEN, DIM, SEQ, 1, dh3, SEQ, a->x2norm, SEQ, 1, (float*)s->buf_gW3[L].contents, DIM);

        // RMSNorm2 backward (CPU)
        memset(dx2, 0, SEQ*DIM*4);
        rmsnorm_bwd(dx2, s->grms_ffn[L], dx_ffn, a->x2, s->lw[L].rms_ffn, DIM, SEQ);
        // dx2 += dy (residual)
        vDSP_vadd(dx2, 1, s->dy, 1, dx2, 1, (vDSP_Length)(SEQ*DIM));

        // --- Attention Backward ---
        // do_out = Wo^T @ dx2 (CPU cblas)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, SEQ, DIM, 1, s->lw[L].Wo, DIM, dx2, SEQ, 0, do_out, SEQ);

        // dWo += dx2 @ attn_out^T (CPU cblas, beta=1)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    DIM, DIM, SEQ, 1, dx2, SEQ, a->attn_out, SEQ, 1, (float*)s->buf_gWo[L].contents, DIM);

        // Attention backward (CPU)
        gpu_cpu_attention_backward(dQ, dK, dV, do_out, a->Q, a->K, a->V, a->scores, DIM, SEQ, HEADS);

        // dx_attn = Wq^T@dQ + Wk^T@dK + Wv^T@dV (CPU cblas)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, SEQ, DIM, 1, s->lw[L].Wq, DIM, dQ, SEQ, 0, dx_attn, SEQ);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, SEQ, DIM, 1, s->lw[L].Wk, DIM, dK, SEQ, 1, dx_attn, SEQ);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, SEQ, DIM, 1, s->lw[L].Wv, DIM, dV, SEQ, 1, dx_attn, SEQ);

        // dWq, dWk, dWv += d{Q,K,V} @ xnorm^T (CPU cblas, beta=1)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    DIM, DIM, SEQ, 1, dQ, SEQ, a->xnorm, SEQ, 1, (float*)s->buf_gWq[L].contents, DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    DIM, DIM, SEQ, 1, dK, SEQ, a->xnorm, SEQ, 1, (float*)s->buf_gWk[L].contents, DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    DIM, DIM, SEQ, 1, dV, SEQ, a->xnorm, SEQ, 1, (float*)s->buf_gWv[L].contents, DIM);

        // RMSNorm1 backward (CPU)
        float *dx_rms1 = (float*)calloc(SEQ*DIM, 4);
        rmsnorm_bwd(dx_rms1, s->grms_att[L], dx_attn, a->layer_in, s->lw[L].rms_att, DIM, SEQ);
        // dy for next layer = dx_rms1 + dx2
        for (int i = 0; i < SEQ*DIM; i++) s->dy[i] = dx_rms1[i] + dx2[i];
        free(dx_rms1);
    }

    // Embedding backward
    embed_backward(s->gembed, s->dy, input_tokens, DIM, SEQ);

    free(dffn); free(dsilu); free(dh1); free(dh3); free(dx_ffn);
    free(dx2); free(do_out); free(dQ); free(dK); free(dV); free(dx_attn);

    s->accum_count++;
    s->step++;

    // ===== Adam update when accumulation complete =====
    if (s->accum_count >= ACCUM_STEPS) {
        float gsc = 1.0f / s->accum_count;
        s->adam_t++;
        float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;

        for (int L = 0; L < NLAYERS; L++) {
            // Scale GPU-computed gradients
            float *gWq=(float*)s->buf_gWq[L].contents, *gWk=(float*)s->buf_gWk[L].contents;
            float *gWv=(float*)s->buf_gWv[L].contents, *gWo=(float*)s->buf_gWo[L].contents;
            float *gW1=(float*)s->buf_gW1[L].contents, *gW2=(float*)s->buf_gW2[L].contents;
            float *gW3=(float*)s->buf_gW3[L].contents;

            vDSP_vsmul(gWq,1,&gsc,gWq,1,(vDSP_Length)WQ_SZ);
            vDSP_vsmul(gWk,1,&gsc,gWk,1,(vDSP_Length)WQ_SZ);
            vDSP_vsmul(gWv,1,&gsc,gWv,1,(vDSP_Length)WQ_SZ);
            vDSP_vsmul(gWo,1,&gsc,gWo,1,(vDSP_Length)WO_SZ);
            vDSP_vsmul(gW1,1,&gsc,gW1,1,(vDSP_Length)W1_SZ);
            vDSP_vsmul(gW2,1,&gsc,gW2,1,(vDSP_Length)W2_SZ);
            vDSP_vsmul(gW3,1,&gsc,gW3,1,(vDSP_Length)W3_SZ);

            // Scale CPU-computed rms gradients
            vDSP_vsmul(s->grms_att[L],1,&gsc,s->grms_att[L],1,(vDSP_Length)DIM);
            vDSP_vsmul(s->grms_ffn[L],1,&gsc,s->grms_ffn[L],1,(vDSP_Length)DIM);

            // Adam updates (writes directly into MTLBuffer.contents — zero-copy!)
            adam_update(s->lw[L].Wq, gWq, &s->la[L].Wq, s->adam_t, s->lr, b1, b2, eps);
            adam_update(s->lw[L].Wk, gWk, &s->la[L].Wk, s->adam_t, s->lr, b1, b2, eps);
            adam_update(s->lw[L].Wv, gWv, &s->la[L].Wv, s->adam_t, s->lr, b1, b2, eps);
            adam_update(s->lw[L].Wo, gWo, &s->la[L].Wo, s->adam_t, s->lr, b1, b2, eps);
            adam_update(s->lw[L].W1, gW1, &s->la[L].W1, s->adam_t, s->lr, b1, b2, eps);
            adam_update(s->lw[L].W2, gW2, &s->la[L].W2, s->adam_t, s->lr, b1, b2, eps);
            adam_update(s->lw[L].W3, gW3, &s->la[L].W3, s->adam_t, s->lr, b1, b2, eps);

            // RMS norm weight updates (CPU-only, separate AdamState needed)
            // For simplicity, use SGD for rms weights (they're 768-dim vectors)
            float rms_lr = s->lr;
            for (int i=0;i<DIM;i++) {
                s->lw[L].rms_att[i] -= rms_lr * s->grms_att[L][i];
                s->lw[L].rms_ffn[i] -= rms_lr * s->grms_ffn[L][i];
            }

            // Zero gradient buffers
            memset(s->buf_gWq[L].contents, 0, WQ_SZ*4);
            memset(s->buf_gWk[L].contents, 0, WQ_SZ*4);
            memset(s->buf_gWv[L].contents, 0, WQ_SZ*4);
            memset(s->buf_gWo[L].contents, 0, WO_SZ*4);
            memset(s->buf_gW1[L].contents, 0, W1_SZ*4);
            memset(s->buf_gW2[L].contents, 0, W2_SZ*4);
            memset(s->buf_gW3[L].contents, 0, W3_SZ*4);
            memset(s->grms_att[L], 0, DIM*4);
            memset(s->grms_ffn[L], 0, DIM*4);
        }

        // Global: rms_final + embed
        vDSP_vsmul(s->grms_final,1,&gsc,s->grms_final,1,(vDSP_Length)DIM);
        adam_update(s->rms_final, s->grms_final, &s->arms_final, s->adam_t, s->lr, b1, b2, eps);
        for (size_t i=0;i<(size_t)VOCAB*DIM;i++) s->gembed[i] *= gsc;
        adam_update(s->embed, s->gembed, &s->aembed, s->adam_t, s->lr, b1, b2, eps);

        memset(s->grms_final, 0, DIM*4);
        memset(s->gembed, 0, (size_t)VOCAB*DIM*4);
        s->accum_count = 0;

        // NO RECOMPILATION NEEDED! Weights updated in-place in MTLBuffer.
    }

    return loss;
    }
}

// ═══════════════════════════════════════════════════════
// ACCESSORS + FREE
// ═══════════════════════════════════════════════════════

int gpu_train_current_step(GPUTrainState *s) { return s->step; }
float gpu_train_current_loss(GPUTrainState *s) { return s->last_loss; }

void gpu_train_free(GPUTrainState *s) {
    if (!s) return;
    for (int L = 0; L < NLAYERS; L++) {
        // Don't free lw[L].Wq etc — they alias MTLBuffer.contents, ARC frees those
        free(s->lw[L].rms_att); free(s->lw[L].rms_ffn);
        layer_adam_free(&s->la[L]);
        GPULayerActs *a = &s->acts[L];
        free(a->layer_in); free(a->xnorm);
        free(a->Q); free(a->K); free(a->V);
        free(a->attn_out); free(a->scores);
        free(a->o_out); free(a->x2); free(a->x2norm);
        free(a->h1); free(a->h3); free(a->silu_out); free(a->ffn_out);
        free(s->grms_att[L]); free(s->grms_ffn[L]);
    }
    free(s->rms_final); free(s->embed);
    free(s->grms_final); free(s->gembed);
    adam_free(&s->arms_final); adam_free(&s->aembed);
    free(s->x_cur); free(s->x_final);
    free(s->logits); free(s->dlogits);
    free(s->dy);
    if (s->token_data && s->data_len > 0) munmap(s->token_data, s->data_len);
    if (s->data_fd >= 0) close(s->data_fd);
    free(s);
}

// ═══════════════════════════════════════════════════════
// TEST + BENCHMARK
// ═══════════════════════════════════════════════════════

NSString *gpu_training_test(void) {
    @autoreleasepool {
    NSMutableString *log = [NSMutableString string];
    [log appendString:@"=== GPU Training Engine Test ===\n"];

    // Create dummy data
    NSString *tmp = [NSTemporaryDirectory() stringByAppendingPathComponent:@"gpu_test_tokens.bin"];
    { size_t n=10000; uint16_t *b=(uint16_t*)malloc(n*2); srand48(123);
      for (size_t i=0;i<n;i++) b[i]=(uint16_t)(drand48()*(VOCAB-1));
      [[NSData dataWithBytesNoCopy:b length:n*2 freeWhenDone:YES] writeToFile:tmp atomically:YES]; }

    GPUTrainState *s = gpu_train_init(NULL, [tmp UTF8String]);
    if (!s) { [log appendString:@"FAIL: init returned NULL\n"]; return log; }
    [log appendFormat:@"Init OK. Device: %@\n", s->device.name];

    int nsteps = 20;
    float first_loss = 0, last_loss = 0;
    for (int i = 0; i < nsteps; i++) {
        uint64_t t = mach_absolute_time();
        float loss = gpu_train_step(s);
        double ms = GPU_MS(mach_absolute_time()-t);
        if (i == 0) first_loss = loss;
        last_loss = loss;
        if (i < 8 || i == nsteps-1 || (i+1)%ACCUM_STEPS==0)
            [log appendFormat:@"Step %d: loss=%.4f (%.0fms)%s\n", i, loss, ms,
                (i+1)%ACCUM_STEPS==0 ? @" [Adam update]" : @""];
    }
    [log appendFormat:@"\nLoss: %.4f -> %.4f (%s)\n", first_loss, last_loss,
        last_loss < first_loss ? "DECREASING OK" : "NOT DECREASING"];

    gpu_train_free(s);
    [[NSFileManager defaultManager] removeItemAtPath:tmp error:nil];
    [log appendString:@"\n=== GPU Test Complete ===\n"];
    return log;
    }
}

NSString *gpu_training_benchmark(float minutes) {
    @autoreleasepool {
    gpu_tb_init();
    NSMutableString *out = [NSMutableString string];

    [out appendString:@"====================================================================\n"];
    [out appendString:@"  GPU TRAINING BENCHMARK\n"];
    [out appendString:@"====================================================================\n\n"];

    // Dummy data
    NSString *tmp = [NSTemporaryDirectory() stringByAppendingPathComponent:@"gpu_bench_tokens.bin"];
    { size_t n=200000; uint16_t *b=(uint16_t*)malloc(n*2); srand48(42);
      for (size_t i=0;i<n;i++) b[i]=(uint16_t)(drand48()*(VOCAB-1));
      [[NSData dataWithBytesNoCopy:b length:n*2 freeWhenDone:YES] writeToFile:tmp atomically:YES]; }

    // Enable battery
    __block float bl = -1;
    dispatch_sync(dispatch_get_main_queue(), ^{
        [UIDevice currentDevice].batteryMonitoringEnabled = YES;
        bl = [UIDevice currentDevice].batteryLevel;
    });

    NSLog(@"GPU-BENCH: Init...");
    GPUTrainState *s = gpu_train_init(NULL, [tmp UTF8String]);
    if (!s) { [out appendString:@"  FAIL: init\n"]; return out; }
    [out appendFormat:@"  Device:  %@\n", s->device.name];
    [out appendFormat:@"  Battery: %.0f%%\n", bl*100];

    double target_s = minutes * 60;
    float batt_start = bl;
    int total_steps = 0;
    float first_loss = 0, best_loss = 999, last_loss = 0;

    NSLog(@"GPU-BENCH: Running %.0f min...", minutes);
    uint64_t t0 = mach_absolute_time();
    while (GPU_S(mach_absolute_time()-t0) < target_s) {
        float loss = gpu_train_step(s);
        total_steps++;
        last_loss = loss;
        if (total_steps==1) first_loss = loss;
        if (loss < best_loss) best_loss = loss;

        if (total_steps%100==0) {
            NSLog(@"GPU-BENCH: step %d, loss=%.4f", total_steps, loss);
        }
        usleep(5000);
        if (current_thermal_level() >= ThermalCritical) usleep(30000000);
    }

    double elapsed = GPU_S(mach_absolute_time()-t0);
    dispatch_sync(dispatch_get_main_queue(), ^{ bl = [UIDevice currentDevice].batteryLevel; });
    float batt_end = bl;
    float drain = batt_start - batt_end;

    [out appendFormat:@"  Steps:     %d in %.0fs (%.2f steps/s)\n", total_steps, elapsed, total_steps/elapsed];
    [out appendFormat:@"  Tokens/s:  %.0f\n", (total_steps*SEQ)/elapsed];
    [out appendFormat:@"  Loss:      %.4f -> %.4f (best: %.4f, -%.1f%%)\n",
        first_loss, last_loss, best_loss, (first_loss-best_loss)/first_loss*100];
    [out appendFormat:@"  Battery:   %.0f%% -> %.0f%% (%.1f%%)\n", batt_start*100, batt_end*100, drain*100];
    if (drain > 0) {
        double wh = 12.5; // iPhone 15 Pro
        double watts = (drain*wh)/(elapsed/3600);
        [out appendFormat:@"  Power:     %.2fW\n", watts];
        [out appendFormat:@"  Tok/J:     %.0f\n", (total_steps*SEQ)/(watts*elapsed)];
    }
    [out appendFormat:@"  Thermal:   %s\n", thermal_level_name(current_thermal_level())];
    [out appendString:@"\n  NO KERNEL RECOMPILATION (weights in MTLBuffer SharedMemory)\n"];
    [out appendString:@"\n====================================================================\n"];

    // Save
    NSString *docs = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,NSUserDomainMask,YES) firstObject];
    [out writeToFile:[docs stringByAppendingPathComponent:@"gpu_training_results.txt"]
          atomically:YES encoding:NSUTF8StringEncoding error:nil];

    gpu_train_free(s);
    [[NSFileManager defaultManager] removeItemAtPath:tmp error:nil];
    NSLog(@"GPU-BENCH: Complete.");
    return out;
    }
}
