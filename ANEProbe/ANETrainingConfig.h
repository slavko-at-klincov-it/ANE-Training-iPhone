// ANETrainingConfig.h — Model config and shared structures for iOS ANE training
// Ported from macOS ANE-Training/training/stories_config.h
#pragma once
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurfaceRef.h>
#import <mach/mach_time.h>
#include <math.h>
#include <arm_neon.h>

// Stories110M config (same as macOS — can be tuned for smaller models later)
#define DIM 768
#define HIDDEN 2048
#define HEADS 12
#define HD (DIM/HEADS)
#define SEQ 256
#define NLAYERS 12
#define VOCAB 32000
#define ACCUM_STEPS 4
#define SCORE_CH (HEADS*SEQ)
#define MAX_COMPILES 200

// Weight sizes per layer
#define WQ_SZ (DIM*DIM)
#define WO_SZ (DIM*DIM)
#define W1_SZ (HIDDEN*DIM)
#define W2_SZ (DIM*HIDDEN)
#define W3_SZ (HIDDEN*DIM)
#define LAYER_PARAMS (4*WQ_SZ + W1_SZ + W2_SZ + W3_SZ + 2*DIM)
#define TOTAL_PARAMS (NLAYERS * LAYER_PARAMS + DIM + VOCAB*DIM)  // +rms_final+embed

#define KERNELS_PER_LAYER 5
#define TOTAL_WEIGHT_KERNELS (KERNELS_PER_LAYER * NLAYERS)

// MIL boilerplate
#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

#define CONV_CONST \
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

// ANE kernel handle
typedef struct {
    void *model;        // _ANEInMemoryModel (bridged)
    IOSurfaceRef ioIn;
    IOSurfaceRef ioOut;
    void *request;      // _ANERequest (bridged)
    void *tmpDir;       // NSString (bridged)
} Kern;

// ANE globals
static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;

static void ane_init(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_D   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_I   = NSClassFromString(@"_ANEInMemoryModel");
        g_AR  = NSClassFromString(@"_ANERequest");
        g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        mach_timebase_info(&g_tb);
    });
}

static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// IOSurface with 16KB minimum (iOS ANE requirement)
static IOSurfaceRef make_surface(size_t bytes) {
    if (bytes < 16384) bytes = 16384;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Weight blob builder: float array → FP16 blob with ANE header
static NSData *build_blob(const float *w, int rows, int cols) {
    int ws = rows * cols * 2, tot = 128 + ws;
    uint8_t *b = (uint8_t*)calloc(tot, 1);
    b[0]=1; b[4]=2; b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
    *(uint32_t*)(b+72) = ws; *(uint32_t*)(b+80) = 128;
    _Float16 *fp16 = (_Float16*)(b+128);
    for (int i = 0; i < rows*cols; i++) fp16[i] = (_Float16)w[i];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// Transposed blob builder (for backward weight matrices)
static NSData *build_blob_t(const float *w, int rows, int cols) {
    int ws = cols * rows * 2, tot = 128 + ws;
    uint8_t *b = (uint8_t*)calloc(tot, 1);
    b[0]=1; b[4]=2; b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
    *(uint32_t*)(b+72) = ws; *(uint32_t*)(b+80) = 128;
    _Float16 *fp16 = (_Float16*)(b+128);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            fp16[j*rows+i] = (_Float16)w[i*cols+j];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// FP16 blob from raw data
static NSData *build_blob_fp16(_Float16 *d, int cnt) {
    int ws = cnt * 2, tot = 128 + ws;
    uint8_t *b = (uint8_t*)calloc(tot, 1);
    b[0]=1; b[4]=2; b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
    *(uint32_t*)(b+72) = ws; *(uint32_t*)(b+80) = 128;
    memcpy(b+128, d, ws);
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// NEON vectorized FP16↔FP32 conversion
static void cvt_f16_f32(float *dst, const _Float16 *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(src+i));
        vst1q_f32(dst+i,   vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst+i+4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; i++) dst[i] = (float)src[i];
}
static void cvt_f32_f16(_Float16 *dst, const float *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src+i)),
                                      vcvt_f16_f32(vld1q_f32(src+i+4)));
        vst1q_f16((__fp16*)(dst+i), h);
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}

// IOSurface I/O helpers
static void io_write_fp16(IOSurfaceRef s, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s), data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}
static void io_read_fp16(IOSurfaceRef s, float *data, int ch_off, int channels, int sp) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    cvt_f16_f32(data, (_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, channels * sp);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

// Compile MIL with weights into a kernel
static Kern *compile_kern(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes) {
    @autoreleasepool {
        NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
            @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
        if (!desc) { fprintf(stderr, "[compile] desc=NULL\n"); return NULL; }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        for (NSString *path in weights) {
            NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
            [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
        }

        NSError *e = nil;
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 9, @{}, &e)) {
            fprintf(stderr, "[compile] FAIL: %s\n", e ? [[e description] UTF8String] : "no error");
            [fm removeItemAtPath:td error:nil]; return NULL;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 9, @{}, &e)) {
            fprintf(stderr, "[compile] load FAIL\n");
            [fm removeItemAtPath:td error:nil]; return NULL;
        }

        Kern *k = (Kern*)calloc(1, sizeof(Kern));
        k->model = (void*)CFBridgingRetain(mdl);
        k->ioIn  = make_surface(ic_bytes);
        k->ioOut = make_surface(oc_bytes);
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
        k->request = (void*)CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0));
        k->tmpDir = (void*)CFBridgingRetain(td);
        return k;
    }
}

static void free_kern(Kern *k) {
    if (!k) return;
    id mdl = (__bridge id)k->model; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 9, &e);
    CFRelease(k->ioIn); CFRelease(k->ioOut);
    [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)k->tmpDir error:nil];
    CFRelease(k->model); CFRelease(k->request); CFRelease(k->tmpDir);
    free(k);
}

static BOOL ane_eval(Kern *k) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
}

// IOSurface to IOSurface direct memcpy
static void io_copy(IOSurfaceRef dst, int dst_ch, IOSurfaceRef src, int src_ch, int channels, int sp) {
    IOSurfaceLock(dst, 0, NULL);
    IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL);
    memcpy((_Float16*)IOSurfaceGetBaseAddress(dst) + dst_ch*sp,
           (_Float16*)IOSurfaceGetBaseAddress(src) + src_ch*sp,
           channels * sp * sizeof(_Float16));
    IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(dst, 0, NULL);
}

// Write float data at channel offset into IOSurface as FP16
static void io_write_fp16_at(IOSurfaceRef s, int ch_off, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}

// Compile count tracker
static int g_compile_count = 0;

// Kernel compile with compile count tracking (QoS=9 for iOS Background)
static Kern *compile_kern_mil_w(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes) {
    @autoreleasepool {
        NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
            @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
        if (!desc) { fprintf(stderr, "[compile] desc=NULL\n"); return NULL; }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        for (NSString *path in weights) {
            NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
            [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
        }

        NSError *e = nil;
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 9, @{}, &e)) {
            fprintf(stderr, "[compile] FAIL: %s\n", e ? [[e description] UTF8String] : "no error");
            [fm removeItemAtPath:td error:nil]; return NULL;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 9, @{}, &e)) {
            fprintf(stderr, "[compile] load FAIL\n");
            [fm removeItemAtPath:td error:nil]; return NULL;
        }
        __sync_fetch_and_add(&g_compile_count, 1);

        Kern *k = (Kern*)calloc(1, sizeof(Kern));
        k->model = (void*)CFBridgingRetain(mdl);
        k->ioIn  = make_surface(ic_bytes);
        k->ioOut = make_surface(oc_bytes);
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
        k->request = (void*)CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0));
        k->tmpDir = (void*)CFBridgingRetain(td);
        return k;
    }
}

// Per-layer weight and optimizer state
typedef struct {
    float *Wq, *Wk, *Wv, *Wo;
    float *W1, *W2, *W3;
    float *rms_att, *rms_ffn;
} LayerWeights;

typedef struct {
    float *m, *v;
    size_t n;
} AdamState;

typedef struct {
    AdamState Wq, Wk, Wv, Wo;
    AdamState W1, W2, W3;
    AdamState rms_att, rms_ffn;
} LayerAdam;

// Per-layer activation buffers (saved for backward)
typedef struct {
    float *layer_in;    // [DIM, SEQ] input to this layer (for rmsnorm1 bwd)
    float *xnorm;       // [DIM, SEQ] rmsnorm1 output
    float *Q, *K, *V;   // [DIM, SEQ] QKV projections
    float *attn_out;     // [DIM, SEQ] attention output (before Wo)
    float *o_out;        // [DIM, SEQ] Wo output
    float *x2;           // [DIM, SEQ] residual after attn
    float *x2norm;       // [DIM, SEQ] rmsnorm2 output
    float *h1, *h3;      // [HIDDEN, SEQ] FFN intermediates
    float *silu_out;     // [HIDDEN, SEQ] SiLU(h1)*h3
    float *ffn_out;      // [DIM, SEQ] FFN output
} LayerActs;

// Per-layer gradient accumulators
typedef struct {
    float *Wq, *Wk, *Wv, *Wo;
    float *W1, *W2, *W3;
    float *rms_att, *rms_ffn;
} LayerGrads;

// ANE kernels per layer
typedef struct {
    Kern *fwdAttn, *fwdFFN, *ffnBwd, *sdpaBwd1, *sdpaBwd2, *qkvBwd;
} LayerKernels;

// Alloc/free helpers
static AdamState adam_alloc(size_t n) { AdamState s; s.m=(float*)calloc(n,4); s.v=(float*)calloc(n,4); s.n=n; return s; }
static void adam_free(AdamState *s) { free(s->m); free(s->v); }

static LayerWeights layer_weights_alloc(void) {
    LayerWeights w;
    w.Wq=(float*)malloc(WQ_SZ*4); w.Wk=(float*)malloc(WQ_SZ*4);
    w.Wv=(float*)malloc(WQ_SZ*4); w.Wo=(float*)malloc(WO_SZ*4);
    w.W1=(float*)malloc(W1_SZ*4); w.W2=(float*)malloc(W2_SZ*4); w.W3=(float*)malloc(W3_SZ*4);
    w.rms_att=(float*)malloc(DIM*4); w.rms_ffn=(float*)malloc(DIM*4);
    return w;
}
static void layer_weights_free(LayerWeights *w) {
    free(w->Wq);free(w->Wk);free(w->Wv);free(w->Wo);
    free(w->W1);free(w->W2);free(w->W3);
    free(w->rms_att);free(w->rms_ffn);
}

static LayerAdam layer_adam_alloc(void) {
    LayerAdam a;
    a.Wq=adam_alloc(WQ_SZ); a.Wk=adam_alloc(WQ_SZ); a.Wv=adam_alloc(WQ_SZ); a.Wo=adam_alloc(WO_SZ);
    a.W1=adam_alloc(W1_SZ); a.W2=adam_alloc(W2_SZ); a.W3=adam_alloc(W3_SZ);
    a.rms_att=adam_alloc(DIM); a.rms_ffn=adam_alloc(DIM);
    return a;
}
static void layer_adam_free(LayerAdam *a) {
    adam_free(&a->Wq);adam_free(&a->Wk);adam_free(&a->Wv);adam_free(&a->Wo);
    adam_free(&a->W1);adam_free(&a->W2);adam_free(&a->W3);
    adam_free(&a->rms_att);adam_free(&a->rms_ffn);
}

static LayerActs layer_acts_alloc(void) {
    LayerActs a;
    a.layer_in=(float*)malloc(SEQ*DIM*4);
    a.xnorm=(float*)malloc(SEQ*DIM*4); a.Q=(float*)malloc(SEQ*DIM*4);
    a.K=(float*)malloc(SEQ*DIM*4); a.V=(float*)malloc(SEQ*DIM*4);
    a.attn_out=(float*)malloc(SEQ*DIM*4); a.o_out=(float*)malloc(SEQ*DIM*4);
    a.x2=(float*)malloc(SEQ*DIM*4); a.x2norm=(float*)malloc(SEQ*DIM*4);
    a.h1=(float*)malloc(SEQ*HIDDEN*4); a.h3=(float*)malloc(SEQ*HIDDEN*4);
    a.silu_out=(float*)malloc(SEQ*HIDDEN*4); a.ffn_out=(float*)malloc(SEQ*DIM*4);
    return a;
}
static void layer_acts_free(LayerActs *a) {
    free(a->layer_in);free(a->xnorm);free(a->Q);free(a->K);free(a->V);
    free(a->attn_out);free(a->o_out);free(a->x2);free(a->x2norm);
    free(a->h1);free(a->h3);free(a->silu_out);free(a->ffn_out);
}

static LayerGrads layer_grads_alloc(void) {
    LayerGrads g;
    g.Wq=(float*)calloc(WQ_SZ,4); g.Wk=(float*)calloc(WQ_SZ,4);
    g.Wv=(float*)calloc(WQ_SZ,4); g.Wo=(float*)calloc(WO_SZ,4);
    g.W1=(float*)calloc(W1_SZ,4); g.W2=(float*)calloc(W2_SZ,4); g.W3=(float*)calloc(W3_SZ,4);
    g.rms_att=(float*)calloc(DIM,4); g.rms_ffn=(float*)calloc(DIM,4);
    return g;
}
static void layer_grads_zero(LayerGrads *g) {
    memset(g->Wq,0,WQ_SZ*4);memset(g->Wk,0,WQ_SZ*4);
    memset(g->Wv,0,WQ_SZ*4);memset(g->Wo,0,WO_SZ*4);
    memset(g->W1,0,W1_SZ*4);memset(g->W2,0,W2_SZ*4);memset(g->W3,0,W3_SZ*4);
    memset(g->rms_att,0,DIM*4);memset(g->rms_ffn,0,DIM*4);
}
static void layer_grads_free(LayerGrads *g) {
    free(g->Wq);free(g->Wk);free(g->Wv);free(g->Wo);
    free(g->W1);free(g->W2);free(g->W3);
    free(g->rms_att);free(g->rms_ffn);
}
