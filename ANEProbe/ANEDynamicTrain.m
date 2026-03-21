// ANEDynamicTrain.m — Dynamic Spatial Packing Training Loop
// Eliminates 20ms recompile overhead by packing weights into the INPUT IOSurface.
//
// Key insight: instead of baking weights into MIL constants and recompiling each step,
// we pack weights into extra spatial positions of the input tensor. The MIL uses
// slice_by_size to separate activations from weights at runtime.
//
// Result: compile ONCE, then just update IOSurface with new weights each step.
//
// Test A: Per-channel scale (element-wise, proven in Phase 1.4)
//   Input: [1, CH, 1, SP + 1]  — last spatial = per-channel weight
//   MIL: slice activations [0:SP], slice weight [SP:SP+1], mul
//
// Test B: Full weight matrix via spatial packing
//   Input: [1, OUT_CH, 1, SP + IN_CH]  — extra spatial = weight rows
//   MIL: slice activations and weight matrix, reshape, matmul via conv
//   For y = W @ x: pack W[OUT_CH, IN_CH] as extra spatial columns
//
// Both tests run a training loop to minimize MSE loss with SGD.

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurfaceRef.h>
#import <mach/mach_time.h>
#include <math.h>
#include <arm_neon.h>

// ================================================================
// Private globals (prefixed dt_ to avoid collisions)
// ================================================================

static Class dt_D, dt_I, dt_R, dt_IO;
static mach_timebase_info_data_t dt_tb;

static void dt_ensure(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_LAZY);
        dt_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        dt_I  = NSClassFromString(@"_ANEInMemoryModel");
        dt_R  = NSClassFromString(@"_ANERequest");
        dt_IO = NSClassFromString(@"_ANEIOSurfaceObject");
        mach_timebase_info(&dt_tb);
    });
}

static double dt_ms(uint64_t t) { return (double)t * dt_tb.numer / dt_tb.denom / 1e6; }

static IOSurfaceRef dt_surface(size_t bytes) {
    if (bytes < 16384) bytes = 16384;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Minimal weight blob (needed even when no baked weights)
static NSData *dt_dummy_blob(void) {
    int ws = 2, tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0]=1; b[4]=2;
    b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
    *(uint32_t*)(b+72) = ws; *(uint32_t*)(b+80) = 128;
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// NEON FP16<->FP32
static void dt_f16_to_f32(float *dst, const _Float16 *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(src+i));
        vst1q_f32(dst+i,   vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst+i+4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; i++) dst[i] = (float)src[i];
}
static void dt_f32_to_f16(_Float16 *dst, const float *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src+i)),
                                      vcvt_f16_f32(vld1q_f32(src+i+4)));
        vst1q_f16((__fp16*)(dst+i), h);
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}

// ================================================================
// Compile + load helper (compile once, reuse)
// ================================================================

typedef struct {
    void *model;    // _ANEInMemoryModel (bridged)
    void *tmpDir;   // NSString (bridged)
} DynKernel;

static DynKernel *dt_compile(NSString *mil, NSMutableString *log) {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    NSData *wdata = dt_dummy_blob();

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(dt_D,
        @selector(modelWithMILText:weights:optionsPlist:),
        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
    if (!desc) { [log appendString:@"    desc=nil\n"]; return NULL; }

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(dt_I, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) { [log appendString:@"    model=nil\n"]; return NULL; }

    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(compileWithQoS:options:error:), 9, @{}, &e);
    if (!ok) { [log appendFormat:@"    compile fail: %@\n", e]; [fm removeItemAtPath:td error:nil]; return NULL; }

    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 9, @{}, &e);
    if (!ok) { [log appendFormat:@"    load fail: %@\n", e]; [fm removeItemAtPath:td error:nil]; return NULL; }

    DynKernel *k = (DynKernel *)calloc(1, sizeof(DynKernel));
    k->model = (void *)CFBridgingRetain(mdl);
    k->tmpDir = (void *)CFBridgingRetain(td);
    return k;
}

static BOOL dt_eval(DynKernel *k, IOSurfaceRef ioIn, IOSurfaceRef ioOut) {
    id mdl = (__bridge id)k->model;
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(dt_IO, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(dt_IO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(dt_R,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    NSError *e = nil;
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
}

static void dt_free(DynKernel *k) {
    if (!k) return;
    id mdl = (__bridge id)k->model; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 9, &e);
    [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)k->tmpDir error:nil];
    CFRelease(k->model); CFRelease(k->tmpDir);
    free(k);
}

// ================================================================
// TEST A: Per-channel Scale Training (proven pattern from Phase 1.4)
// ================================================================
// Task: learn per-channel scale s[ch] such that y = s * x minimizes MSE vs target
// Target: target = s_true * x
// Input layout: [1, CH, 1, SP+1] where last spatial position = scale weight
// MIL: slice act = input[:,:,:,0:SP], w = input[:,:,:,SP:SP+1], y = act * w

static NSString *test_a_perchannel_scale(void) {
    NSMutableString *out = [NSMutableString string];
    [out appendString:@"\n  --- Test A: Per-Channel Scale (Dynamic Packing) ---\n"];

    int CH = 256, SP = 64;
    int TOTAL_SP = SP + 1; // 1 extra spatial for per-channel weight
    int nsteps = 20;
    float lr = 0.1f;

    [out appendFormat:@"  Config: ch=%d sp=%d steps=%d lr=%.2f\n", CH, SP, nsteps, lr];
    [out appendFormat:@"  Input: [1, %d, 1, %d] (act=%d + weight=1)\n", CH, TOTAL_SP, SP];

    // Generate MIL: compile ONCE
    NSString *mil = [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> input) {\n"
        // Slice activation: input[:, :, :, 0:SP]
        "        tensor<int32, [4]> ab = const()[name=string(\"ab\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [4]> as = const()[name=string(\"as\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=input, begin=ab, size=as)[name=string(\"act\")];\n"
        // Slice weight: input[:, :, :, SP:SP+1]
        "        tensor<int32, [4]> wb = const()[name=string(\"wb\"), val=tensor<int32, [4]>([0,0,0,%d])];\n"
        "        tensor<int32, [4]> ws = const()[name=string(\"ws\"), val=tensor<int32, [4]>([1,%d,1,1])];\n"
        "        tensor<fp16, [1,%d,1,1]> w = slice_by_size(x=input, begin=wb, size=ws)[name=string(\"wr\")];\n"
        // Output: act * w (broadcast weight over spatial dim)
        "        tensor<fp16, [1,%d,1,%d]> y = mul(x=act, y=w)[name=string(\"out\")];\n"
        "    } -> (y);\n}\n",
        CH, TOTAL_SP,
        CH, SP, CH, SP,
        SP, CH, CH,
        CH, SP];

    uint64_t tc = mach_absolute_time();
    DynKernel *k = dt_compile(mil, out);
    double compileMs = dt_ms(mach_absolute_time() - tc);
    if (!k) { [out appendString:@"    COMPILE FAILED\n"]; return out; }
    [out appendFormat:@"  Compiled ONCE: %.1f ms\n", compileMs];

    // IOSurfaces
    IOSurfaceRef ioIn  = dt_surface(CH * TOTAL_SP * 2);
    IOSurfaceRef ioOut = dt_surface(CH * SP * 2);

    // Ground truth: per-channel scale
    srand48(42);
    float *s_true  = (float *)malloc(CH * sizeof(float));
    float *s_learn = (float *)malloc(CH * sizeof(float));
    float *x_data  = (float *)malloc(CH * SP * sizeof(float));
    float *target  = (float *)malloc(CH * SP * sizeof(float));
    float *y_ane   = (float *)malloc(CH * SP * sizeof(float));

    for (int c = 0; c < CH; c++) s_true[c] = 0.5f + (float)drand48() * 2.0f; // [0.5, 2.5]
    for (int c = 0; c < CH; c++) s_learn[c] = 1.0f; // init to 1
    for (int i = 0; i < CH * SP; i++) x_data[i] = (float)(2*drand48()-1);

    // Compute target = s_true * x
    for (int c = 0; c < CH; c++)
        for (int s = 0; s < SP; s++)
            target[c*SP+s] = s_true[c] * x_data[c*SP+s];

    [out appendString:@"\n  Step   Loss        ms/step\n"];

    uint64_t t_total = mach_absolute_time();

    for (int step = 0; step < nsteps; step++) {
        uint64_t t_step = mach_absolute_time();

        // Pack input: activations + weights into IOSurface
        IOSurfaceLock(ioIn, 0, NULL);
        _Float16 *inp = (_Float16 *)IOSurfaceGetBaseAddress(ioIn);
        // Layout: channel c occupies positions [c*TOTAL_SP .. c*TOTAL_SP + TOTAL_SP-1]
        // First SP values = activation, last 1 value = weight
        for (int c = 0; c < CH; c++) {
            // Activation for channel c
            dt_f32_to_f16(inp + c * TOTAL_SP, x_data + c * SP, SP);
            // Weight for channel c (single value at position SP)
            inp[c * TOTAL_SP + SP] = (_Float16)s_learn[c];
        }
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Forward on ANE (NO recompile!)
        dt_eval(k, ioIn, ioOut);

        // Read output
        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        _Float16 *outp = (_Float16 *)IOSurfaceGetBaseAddress(ioOut);
        dt_f16_to_f32(y_ane, outp, CH * SP);
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

        double stepMs = dt_ms(mach_absolute_time() - t_step);

        // Loss: MSE = mean((y - target)^2)
        float loss = 0;
        int n = CH * SP;
        for (int i = 0; i < n; i++) {
            float diff = y_ane[i] - target[i];
            loss += diff * diff;
        }
        loss /= n;

        [out appendFormat:@"  %4d   %.6f   %.3f ms\n", step, loss, stepMs];

        // Backward: ds = sum_over_sp((y - target) * x) * 2/n
        // Update: s -= lr * ds
        float grad_scale = 2.0f / n;
        for (int c = 0; c < CH; c++) {
            float grad = 0;
            for (int s = 0; s < SP; s++) {
                float diff = y_ane[c*SP+s] - target[c*SP+s];
                grad += diff * x_data[c*SP+s];
            }
            s_learn[c] -= lr * grad_scale * grad;
        }
    }

    double totalMs = dt_ms(mach_absolute_time() - t_total);
    [out appendFormat:@"\n  Total: %.1f ms (%.3f ms/step avg)\n", totalMs, totalMs / nsteps];

    // Verify convergence
    float maxErr = 0, sumErr = 0;
    for (int c = 0; c < CH; c++) {
        float err = fabsf(s_learn[c] - s_true[c]);
        if (err > maxErr) maxErr = err;
        sumErr += err;
    }
    [out appendFormat:@"  s_learn vs s_true: max_err=%.6f avg_err=%.6f\n", maxErr, sumErr / CH];
    [out appendFormat:@"  Zero recompiles! Weights updated via IOSurface each step.\n"];

    dt_free(k);
    CFRelease(ioIn); CFRelease(ioOut);
    free(s_true); free(s_learn); free(x_data); free(target); free(y_ane);
    return out;
}

// ================================================================
// TEST B: Full Weight Matrix Training (Dynamic Packing)
// ================================================================
// Task: learn W[OUT_CH, IN_CH] such that y = W @ x minimizes MSE vs target
// Input layout: [1, OUT_CH, 1, SP + IN_CH]
//   - First SP spatial positions = x broadcasted (reshaped) OR separate input region
//   - Next IN_CH spatial positions = W[out_ch, :] (one row per output channel)
//
// Approach: We can't do a true conv with dynamic weights via slice_by_size alone,
// because conv requires weight as a const parameter. Instead, we use the spatial
// packing trick with mul + reduce_sum to emulate matmul:
//
// For each output channel o:
//   y[o, s] = sum_i(W[o, i] * x[i, s])
//
// Packing: input[1, IN_CH, 1, SP + OUT_CH]
//   - First SP cols = x[IN_CH, SP] (activations)
//   - Next OUT_CH cols = W^T[IN_CH, OUT_CH] (weight columns)
//
// MIL: Can't do general matmul this way with a single mul+reduce.
// Instead: pack x and W into spatial, use conv with weight=1 for accumulation.
//
// SIMPLER APPROACH for full matmul:
// Use the same pattern as per-channel but with IN_CH weights per output channel.
// Input: [1, IN_CH, 1, SP + OUT_CH]
//   - Spatial [0:SP] = x[IN_CH, SP]
//   - Spatial [SP:SP+OUT_CH] = W^T[IN_CH, OUT_CH] (transposed weights)
// Then: slice x, slice W^T, use a BAKED 1x1 conv with identity-like structure
//       to perform the matmul... but that defeats the purpose.
//
// ACTUAL WORKING APPROACH:
// We can implement y = W @ x as element-wise products + reduction.
// Pack W into extra channels of the input, then use reduce_sum.
//
// Input: [1, IN_CH * (1 + OUT_CH), 1, SP]
//   - Channels [0:IN_CH] = x (activations)
//   - Channels [IN_CH + o*IN_CH : IN_CH + (o+1)*IN_CH] = W[o,:] broadcast over SP
// But this requires OUT_CH * IN_CH extra channels, which is huge.
//
// MOST PRACTICAL APPROACH (what actually works on ANE):
// Use spatial packing with reshape + conv(weight=identity).
// Input: [1, OUT_CH, 1, SP + IN_CH]
//   positions [0:SP] per channel = replicated to fill via baked logic
//   positions [SP:SP+IN_CH] per channel o = W[o, :]
//
// For modest sizes (IN_CH=32, OUT_CH=16, SP=32), the full weight matrix
// fits in the spatial dimension. We use mul + reduce_sum via conv.
//
// CLEANEST WORKING APPROACH:
// Two-input approach: pack x AND W into one big spatial tensor.
// Input: [1, IN_CH, 1, TOTAL] where TOTAL = SP + OUT_CH
//   - [:, :, :, 0:SP]         = x[IN_CH, SP]
//   - [:, :, :, SP:SP+OUT_CH] = W^T[IN_CH, OUT_CH]
// MIL:
//   1. slice x = input[:,:,:,0:SP]
//   2. slice wt = input[:,:,:,SP:SP+OUT_CH]  -> [1, IN_CH, 1, OUT_CH]
//   3. For each output position s:
//      y[:, o, :, s] = sum_i(wt[:, i, :, o] * x[:, i, :, s])
//   This is just a matmul: y = wt^T @ x
//   We can implement it as: reshape wt to [OUT_CH, IN_CH, 1, 1] and use conv!
//   But MIL conv requires the weight to be a const or a BLOBFILE, not a runtime tensor.
//
// SOLUTION: Use matmul op directly! MIL has a matmul op.
// MIL matmul: tensor matmul(x, y) where x=[..., M, K], y=[..., K, N]
// We need x in [B, M, K] format and weights in [B, K, N] format.
//
// Actually the simplest thing that definitely works:
// Use element-wise mul with broadcasting + reduce_sum.
// For a small matmul (16x32 * 32xSP):
// 1. Slice x=[1, IN_CH, 1, SP] and W^T=[1, IN_CH, 1, OUT_CH]
// 2. Reshape x to [1, IN_CH, OUT_CH, SP] by tiling/broadcasting
// 3. Reshape W^T to [1, IN_CH, OUT_CH, 1] and broadcast over SP
// 4. mul -> [1, IN_CH, OUT_CH, SP]
// 5. reduce_sum(axis=1) -> [1, 1, OUT_CH, SP] = [1, OUT_CH, SP] = y
//
// But reshape/tile may not be available or efficient on ANE.
//
// PRAGMATIC APPROACH: Use MIL's built-in matmul op.
// Input: [1, IN_CH, 1, SP + OUT_CH]
// Slice x: [1, IN_CH, 1, SP]
// Slice wt: [1, IN_CH, 1, OUT_CH]
// Reshape x to [1, 1, IN_CH, SP] (treat as matrix [IN_CH, SP])
// Reshape wt to [1, 1, IN_CH, OUT_CH] then transpose -> [1, 1, OUT_CH, IN_CH]
// matmul(wt_t, x_reshaped) -> [1, 1, OUT_CH, SP]
// Reshape to [1, OUT_CH, 1, SP]

static NSString *gen_fullmatmul_mil(int in_ch, int out_ch, int sp) {
    int total_sp = sp + out_ch;
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> input) {\n"

        // Slice activation: input[:, :, :, 0:SP] -> [1, IN_CH, 1, SP]
        "        tensor<int32, [4]> ab = const()[name=string(\"ab\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [4]> as = const()[name=string(\"as\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "        tensor<fp16, [1,%d,1,%d]> x_raw = slice_by_size(x=input, begin=ab, size=as)[name=string(\"xr\")];\n"

        // Slice weight: input[:, :, :, SP:SP+OUT_CH] -> [1, IN_CH, 1, OUT_CH]
        "        tensor<int32, [4]> wb = const()[name=string(\"wb\"), val=tensor<int32, [4]>([0,0,0,%d])];\n"
        "        tensor<int32, [4]> ws = const()[name=string(\"ws\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "        tensor<fp16, [1,%d,1,%d]> wt_raw = slice_by_size(x=input, begin=wb, size=ws)[name=string(\"wr\")];\n"

        // Reshape x: [1, IN_CH, 1, SP] -> [1, IN_CH, SP] (squeeze dim 2)
        "        tensor<int32, [3]> x3s = const()[name=string(\"x3s\"), val=tensor<int32, [3]>([1,%d,%d])];\n"
        "        tensor<fp16, [1,%d,%d]> x3 = reshape(x=x_raw, shape=x3s)[name=string(\"x3\")];\n"

        // Reshape wt: [1, IN_CH, 1, OUT_CH] -> [1, IN_CH, OUT_CH] (squeeze dim 2)
        "        tensor<int32, [3]> w3s = const()[name=string(\"w3s\"), val=tensor<int32, [3]>([1,%d,%d])];\n"
        "        tensor<fp16, [1,%d,%d]> w3 = reshape(x=wt_raw, shape=w3s)[name=string(\"w3\")];\n"

        // Transpose wt: [1, IN_CH, OUT_CH] -> [1, OUT_CH, IN_CH]
        "        tensor<fp16, [1,%d,%d]> wt = transpose(x=w3, perm=[0,2,1])[name=string(\"wt\")];\n"

        // Matmul: [1, OUT_CH, IN_CH] @ [1, IN_CH, SP] -> [1, OUT_CH, SP]
        "        tensor<fp16, [1,%d,%d]> y3 = matmul(x=wt, y=x3, transpose_x=false, transpose_y=false)[name=string(\"y3\")];\n"

        // Reshape back: [1, OUT_CH, SP] -> [1, OUT_CH, 1, SP]
        "        tensor<int32, [4]> y4s = const()[name=string(\"y4s\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "        tensor<fp16, [1,%d,1,%d]> y = reshape(x=y3, shape=y4s)[name=string(\"out\")];\n"

        "    } -> (y);\n}\n",

        // func signature: [1, IN_CH, 1, TOTAL_SP]
        in_ch, total_sp,
        // slice x: size [1, IN_CH, 1, SP]
        in_ch, sp, in_ch, sp,
        // slice w: begin [0,0,0,SP], size [1, IN_CH, 1, OUT_CH]
        sp, in_ch, out_ch, in_ch, out_ch,
        // reshape x: [1, IN_CH, SP]
        in_ch, sp, in_ch, sp,
        // reshape w: [1, IN_CH, OUT_CH]
        in_ch, out_ch, in_ch, out_ch,
        // transpose w: [1, OUT_CH, IN_CH]
        out_ch, in_ch,
        // matmul: [1, OUT_CH, SP]
        out_ch, sp,
        // reshape y: [1, OUT_CH, 1, SP]
        out_ch, sp, out_ch, sp
    ];
}

static NSString *test_b_full_matmul(void) {
    NSMutableString *out = [NSMutableString string];
    [out appendString:@"\n  --- Test B: Full Weight Matrix (Dynamic Packing) ---\n"];

    // Small dimensions to keep weight matrix in spatial range
    int IN_CH = 32, OUT_CH = 16, SP = 32;
    int TOTAL_SP = SP + OUT_CH;
    int nsteps = 20;
    float lr = 0.01f;

    [out appendFormat:@"  Config: in=%d out=%d sp=%d steps=%d lr=%.3f\n", IN_CH, OUT_CH, SP, nsteps, lr];
    [out appendFormat:@"  Input: [1, %d, 1, %d] (act=%d + weight=%d)\n", IN_CH, TOTAL_SP, SP, OUT_CH];
    [out appendFormat:@"  Weight matrix: %d x %d = %d params packed in spatial dim\n", OUT_CH, IN_CH, OUT_CH * IN_CH];

    // Generate MIL and compile ONCE
    NSString *mil = gen_fullmatmul_mil(IN_CH, OUT_CH, SP);
    uint64_t tc = mach_absolute_time();
    DynKernel *k = dt_compile(mil, out);
    double compileMs = dt_ms(mach_absolute_time() - tc);
    if (!k) { [out appendString:@"    COMPILE FAILED\n"]; return out; }
    [out appendFormat:@"  Compiled ONCE: %.1f ms\n", compileMs];

    // IOSurfaces
    // Input: [1, IN_CH, 1, TOTAL_SP] -> IN_CH * TOTAL_SP fp16 values
    // Output: [1, OUT_CH, 1, SP] -> OUT_CH * SP fp16 values
    IOSurfaceRef ioIn  = dt_surface(IN_CH * TOTAL_SP * 2);
    IOSurfaceRef ioOut = dt_surface(OUT_CH * SP * 2);

    // Random data
    srand48(42);
    float scale = 1.0f / sqrtf((float)IN_CH);

    float *W_true  = (float *)malloc(OUT_CH * IN_CH * sizeof(float));
    float *W_learn = (float *)malloc(OUT_CH * IN_CH * sizeof(float));
    float *x_data  = (float *)malloc(IN_CH * SP * sizeof(float));
    float *target  = (float *)malloc(OUT_CH * SP * sizeof(float));
    float *y_ane   = (float *)malloc(OUT_CH * SP * sizeof(float));
    float *dW      = (float *)malloc(OUT_CH * IN_CH * sizeof(float));

    for (int i = 0; i < OUT_CH * IN_CH; i++) W_true[i] = (float)(2*drand48()-1) * scale;
    for (int i = 0; i < OUT_CH * IN_CH; i++) W_learn[i] = (float)(2*drand48()-1) * scale;
    for (int i = 0; i < IN_CH * SP; i++) x_data[i] = (float)(2*drand48()-1);

    // Target: y = W_true @ x
    for (int o = 0; o < OUT_CH; o++)
        for (int s = 0; s < SP; s++) {
            float sum = 0;
            for (int i = 0; i < IN_CH; i++) sum += W_true[o*IN_CH+i] * x_data[i*SP+s];
            target[o*SP+s] = sum;
        }

    [out appendString:@"\n  Step   Loss        ms/step\n"];

    uint64_t t_total = mach_absolute_time();

    for (int step = 0; step < nsteps; step++) {
        uint64_t t_step = mach_absolute_time();

        // Pack input IOSurface: [1, IN_CH, 1, TOTAL_SP]
        // Channel i occupies positions [i*TOTAL_SP .. i*TOTAL_SP + TOTAL_SP-1]
        // Positions [0:SP] = x[i, 0:SP]
        // Positions [SP:SP+OUT_CH] = W^T[i, 0:OUT_CH] = W[:, i] transposed
        //   i.e., position SP+o = W[o, i]
        IOSurfaceLock(ioIn, 0, NULL);
        _Float16 *inp = (_Float16 *)IOSurfaceGetBaseAddress(ioIn);
        // Clear (important for padding)
        memset(inp, 0, IN_CH * TOTAL_SP * 2);
        for (int i = 0; i < IN_CH; i++) {
            // Activation x[i, :] at spatial [0:SP]
            dt_f32_to_f16(inp + i * TOTAL_SP, x_data + i * SP, SP);
            // Weight column W[:, i] transposed into spatial [SP:SP+OUT_CH]
            // W^T[i, o] = W[o, i]
            for (int o = 0; o < OUT_CH; o++) {
                inp[i * TOTAL_SP + SP + o] = (_Float16)W_learn[o * IN_CH + i];
            }
        }
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Forward on ANE (NO recompile!)
        BOOL ok = dt_eval(k, ioIn, ioOut);
        if (!ok && step == 0) {
            [out appendString:@"    EVAL FAILED (matmul may not be supported)\n"];
            break;
        }

        // Read output: [1, OUT_CH, 1, SP]
        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        _Float16 *outp = (_Float16 *)IOSurfaceGetBaseAddress(ioOut);
        dt_f16_to_f32(y_ane, outp, OUT_CH * SP);
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

        double stepMs = dt_ms(mach_absolute_time() - t_step);

        // Loss: MSE
        float loss = 0;
        int n = OUT_CH * SP;
        for (int i = 0; i < n; i++) {
            float diff = y_ane[i] - target[i];
            loss += diff * diff;
        }
        loss /= n;

        [out appendFormat:@"  %4d   %.6f   %.3f ms\n", step, loss, stepMs];

        // Backward: dW[o,i] = (2/n) * sum_s((y[o,s] - target[o,s]) * x[i,s])
        float grad_scale = 2.0f / n;
        memset(dW, 0, OUT_CH * IN_CH * sizeof(float));
        for (int o = 0; o < OUT_CH; o++)
            for (int i = 0; i < IN_CH; i++) {
                float sum = 0;
                for (int s = 0; s < SP; s++)
                    sum += (y_ane[o*SP+s] - target[o*SP+s]) * x_data[i*SP+s];
                dW[o*IN_CH+i] = grad_scale * sum;
            }

        // SGD update
        for (int i = 0; i < OUT_CH * IN_CH; i++)
            W_learn[i] -= lr * dW[i];
    }

    double totalMs = dt_ms(mach_absolute_time() - t_total);
    [out appendFormat:@"\n  Total: %.1f ms (%.3f ms/step avg)\n", totalMs, totalMs / nsteps];

    // Verify convergence
    float wMaxErr = 0, wSumErr = 0;
    for (int i = 0; i < OUT_CH * IN_CH; i++) {
        float err = fabsf(W_learn[i] - W_true[i]);
        if (err > wMaxErr) wMaxErr = err;
        wSumErr += err;
    }
    [out appendFormat:@"  W_learn vs W_true: max_err=%.6f avg_err=%.6f\n",
        wMaxErr, wSumErr / (OUT_CH * IN_CH)];
    [out appendFormat:@"  Zero recompiles! Full %dx%d matmul with dynamic weights.\n", OUT_CH, IN_CH];

    dt_free(k);
    CFRelease(ioIn); CFRelease(ioOut);
    free(W_true); free(W_learn); free(x_data); free(target); free(y_ane); free(dW);
    return out;
}

// ================================================================
// TEST C: Throughput comparison (dynamic vs recompile)
// ================================================================

static NSString *test_c_throughput(void) {
    NSMutableString *out = [NSMutableString string];
    [out appendString:@"\n  --- Test C: Throughput Benchmark ---\n"];

    int CH = 256, SP = 64;
    int TOTAL_SP = SP + 1;
    int iters = 100;

    // Compile the dynamic kernel
    NSString *mil = [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> input) {\n"
        "        tensor<int32, [4]> ab = const()[name=string(\"ab\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [4]> as = const()[name=string(\"as\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=input, begin=ab, size=as)[name=string(\"act\")];\n"
        "        tensor<int32, [4]> wb = const()[name=string(\"wb\"), val=tensor<int32, [4]>([0,0,0,%d])];\n"
        "        tensor<int32, [4]> ws = const()[name=string(\"ws\"), val=tensor<int32, [4]>([1,%d,1,1])];\n"
        "        tensor<fp16, [1,%d,1,1]> w = slice_by_size(x=input, begin=wb, size=ws)[name=string(\"wr\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> y = mul(x=act, y=w)[name=string(\"out\")];\n"
        "    } -> (y);\n}\n",
        CH, TOTAL_SP,
        CH, SP, CH, SP,
        SP, CH, CH,
        CH, SP];

    DynKernel *k = dt_compile(mil, out);
    if (!k) { [out appendString:@"    COMPILE FAILED\n"]; return out; }

    IOSurfaceRef ioIn  = dt_surface(CH * TOTAL_SP * 2);
    IOSurfaceRef ioOut = dt_surface(CH * SP * 2);

    // Fill with data
    IOSurfaceLock(ioIn, 0, NULL);
    _Float16 *inp = (_Float16 *)IOSurfaceGetBaseAddress(ioIn);
    for (int c = 0; c < CH; c++) {
        for (int s = 0; s < SP; s++) inp[c * TOTAL_SP + s] = (_Float16)1.0f;
        inp[c * TOTAL_SP + SP] = (_Float16)2.0f;
    }
    IOSurfaceUnlock(ioIn, 0, NULL);

    // Warmup
    for (int i = 0; i < 5; i++) dt_eval(k, ioIn, ioOut);

    // Benchmark: weight update + eval (dynamic, no recompile)
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        // Simulate weight update (write new weight values)
        IOSurfaceLock(ioIn, 0, NULL);
        _Float16 *p = (_Float16 *)IOSurfaceGetBaseAddress(ioIn);
        _Float16 wval = (_Float16)(1.0f + 0.01f * i);
        for (int c = 0; c < CH; c++) p[c * TOTAL_SP + SP] = wval;
        IOSurfaceUnlock(ioIn, 0, NULL);
        // Eval
        dt_eval(k, ioIn, ioOut);
    }
    double dynMs = dt_ms(mach_absolute_time() - t0);

    [out appendFormat:@"  Dynamic (no recompile): %d iters, %.1f ms total, %.3f ms/iter\n",
        iters, dynMs, dynMs / iters];
    [out appendFormat:@"  (Compare to ~22.5 ms/iter with recompile from Phase 1.4)\n"];
    [out appendFormat:@"  Speedup: ~%.0fx faster per training step\n", 22.5 / (dynMs / iters)];

    dt_free(k);
    CFRelease(ioIn); CFRelease(ioOut);
    return out;
}

// ================================================================
// Public entry point
// ================================================================

NSString *ane_dynamic_train_test(void) {
    @autoreleasepool {
        dt_ensure();
        NSMutableString *out = [NSMutableString string];
        if (!dt_D || !dt_I || !dt_R || !dt_IO) {
            [out appendString:@"  ANE classes missing\n"];
            return out;
        }

        [out appendString:@"  === Dynamic Spatial Packing Training ===\n"];
        [out appendString:@"  Concept: pack weights into INPUT IOSurface, compile ONCE\n"];
        [out appendString:@"  Eliminates 20ms recompile overhead per training step\n"];

        // Test A: Per-channel scale (guaranteed to work, proven in Phase 1.4)
        [out appendString:[test_a_perchannel_scale() copy]];

        // Test B: Full weight matrix via matmul
        [out appendString:[test_b_full_matmul() copy]];

        // Test C: Throughput benchmark
        [out appendString:[test_c_throughput() copy]];

        return out;
    }
}
