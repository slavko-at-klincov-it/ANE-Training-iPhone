// ANERE.m — Phase 1.5: Training-critical ANE reverse engineering
// Probes: SRAM boundaries, MIL op coverage, perf stats, compile limits
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurfaceRef.h>
#import <mach/mach_time.h>

static Class re_Desc, re_Model, re_Request, re_IOObj, re_PerfStats;
static mach_timebase_info_data_t re_tb;

static void re_ensure(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_LAZY);
        re_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        re_Model = NSClassFromString(@"_ANEInMemoryModel");
        re_Request = NSClassFromString(@"_ANERequest");
        re_IOObj = NSClassFromString(@"_ANEIOSurfaceObject");
        re_PerfStats = NSClassFromString(@"_ANEPerformanceStatsIOSurface");
        mach_timebase_info(&re_tb);
    });
}

static double re_ms(uint64_t t) { return (double)t * re_tb.numer / re_tb.denom / 1e6; }

static IOSurfaceRef re_surface(size_t bytes) {
    // ANE requires minimum 16KB IOSurface for eval to succeed
    if (bytes < 16384) bytes = 16384;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Generic MIL for a single conv kernel
static NSString *re_conv_mil(int ch_in, int ch_out, int sp) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)"
        "[name=string(\"out\")];\n    } -> (y);\n}\n", ch_in, sp, ch_out, ch_in, ch_out, ch_in, ch_out, sp];
}

static NSData *re_weight_blob(int ch_in, int ch_out) {
    int ws = ch_out * ch_in * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t*)calloc(tot, 1);
    b[0]=1; b[4]=2;
    b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
    *(uint32_t*)(b+72) = ws; *(uint32_t*)(b+80) = 128;
    _Float16 *w = (_Float16*)(b+128);
    srand48(42);
    for (int i = 0; i < ch_out * ch_in; i++) w[i] = (_Float16)(0.01 * (2*drand48()-1));
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// Compile+load+bench helper, returns model or nil
static id re_compile_load(NSString *mil, NSData *wdata, NSMutableString *out) {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(re_Desc,
        @selector(modelWithMILText:weights:optionsPlist:),
        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
    if (!desc) { [out appendString:@"  desc=nil\n"]; return nil; }

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(re_Model, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) { [out appendString:@"  model=nil\n"]; return nil; }

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
    if (!ok) { [out appendFormat:@"  compile fail: %@\n", e.localizedDescription]; [fm removeItemAtPath:td error:nil]; return nil; }

    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 9, @{}, &e);
    if (!ok) { [out appendFormat:@"  load fail: %@\n", e.localizedDescription]; [fm removeItemAtPath:td error:nil]; return nil; }

    return mdl;
}

static void re_unload(id mdl) {
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 9, &e);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
}

// ================================================================
// TEST 1: SRAM Boundary Probing
// Find exact weight size where performance cliffs (= exceeds on-chip SRAM)
// ================================================================
NSString *ane_re_sram_probe(void) {
    @autoreleasepool {
        re_ensure();
        NSMutableString *out = [NSMutableString string];
        if (!re_Desc || !re_Model) { [out appendString:@"  Classes missing\n"]; return out; }

        [out appendString:@"  --- SRAM Boundary Probe ---\n"];
        [out appendString:@"  Finding exact weight size where perf drops (= SRAM overflow)\n"];
        [out appendString:@"  ch_in x ch_out  weight_KB  ms/eval  TFLOPS  note\n"];

        // Sweep: increase channel count to find the cliff
        // We know: 2048x2048 (8MB) = fast, 4096x4096 (32MB) = slow
        // Probe in between to find exact boundary
        int configs[][3] = {
            // ch_in, ch_out, sp
            {1024, 1024, 64},   // 2.0 MB
            {1536, 1536, 64},   // 4.5 MB
            {1792, 1792, 64},   // 6.1 MB
            {2048, 2048, 64},   // 8.0 MB
            {2304, 2304, 64},   // 10.1 MB
            {2560, 2560, 64},   // 12.5 MB
            {2816, 2816, 64},   // 15.1 MB
            {3072, 3072, 64},   // 18.0 MB
            {3328, 3328, 64},   // 21.1 MB
            {3584, 3584, 64},   // 24.5 MB
            {4096, 4096, 64},   // 32.0 MB
        };
        int nconfigs = sizeof(configs) / sizeof(configs[0]);

        for (int i = 0; i < nconfigs; i++) {
            int ci = configs[i][0], co = configs[i][1], sp = configs[i][2];
            double wMB = (double)(co * ci * 2) / (1024.0 * 1024.0);

            NSString *mil = re_conv_mil(ci, co, sp);
            NSData *wd = re_weight_blob(ci, co);
            id mdl = re_compile_load(mil, wd, out);
            if (!mdl) {
                [out appendFormat:@"  %4dx%-4d  %7.1f MB  FAILED\n", ci, co, wMB];
                continue;
            }

            IOSurfaceRef ioIn  = re_surface(ci * sp * 2);
            IOSurfaceRef ioOut = re_surface(co * sp * 2);
            id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(re_IOObj, @selector(objectWithIOSurface:), ioIn);
            id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(re_IOObj, @selector(objectWithIOSurface:), ioOut);
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(re_Request,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            NSError *e = nil;
            // Warmup
            for (int w = 0; w < 3; w++)
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);

            int iters = (wMB > 16) ? 10 : 30;
            uint64_t t0 = mach_absolute_time();
            for (int j = 0; j < iters; j++)
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
            double ms = re_ms(mach_absolute_time() - t0);
            double per = ms / iters;
            double flops = 2.0 * ci * co * sp;
            double tflops = flops / (per * 1e9);

            [out appendFormat:@"  %4dx%-4d  %7.1f MB  %6.3f ms  %5.2f TFLOPS\n",
                ci, co, wMB, per, tflops];

            re_unload(mdl);
            CFRelease(ioIn); CFRelease(ioOut);
        }

        return out;
    }
}

// ================================================================
// TEST 2: MIL Op Coverage — which ops compile to ANE?
// Critical for training: we need add, sub, mul, relu, tanh, exp, rsqrt, etc.
// ================================================================
static id re_try_op_mil(NSString *mil, NSMutableString *out, NSString *opName) {
    // Minimal weight blob (unused for most ops)
    int tot = 128 + 2;
    uint8_t *b = (uint8_t*)calloc(tot, 1);
    b[0]=1; b[4]=2;
    b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
    *(uint32_t*)(b+72) = 2; *(uint32_t*)(b+80) = 128;
    NSData *wdata = [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];

    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(re_Desc,
        @selector(modelWithMILText:weights:optionsPlist:),
        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
    if (!desc) return nil;

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(re_Model, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) return nil;

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
    if (!ok) {
        [fm removeItemAtPath:td error:nil];
        return nil;
    }

    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 9, @{}, &e);
    if (!ok) {
        [fm removeItemAtPath:td error:nil];
        return nil;
    }

    return mdl;
}

// MIL program header
#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

#define MIL_FTR @"    } -> (y);\n}\n"

NSString *ane_re_op_coverage(void) {
    @autoreleasepool {
        re_ensure();
        NSMutableString *out = [NSMutableString string];
        if (!re_Desc || !re_Model) { [out appendString:@"  Classes missing\n"]; return out; }

        [out appendString:@"  --- MIL Op Coverage (training-critical ops) ---\n"];

        // Each test: MIL program with a single op, check if it compiles+loads on ANE
        // Input: tensor<fp16, [1, 256, 1, 64]>
        int CH = 256, SP = 64;

        // Define ops to test: (name, MIL body)
        // All take input x and produce output y
        struct { const char *name; NSString *body; } ops[] = {
            // === Elementwise (critical for gradients) ===
            {"add (x+x)", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = add(x=x, y=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            {"sub (x-x)", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = sub(x=x, y=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            {"mul (x*x)", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = mul(x=x, y=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            {"real_div (x/x)", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = real_div(x=x, y=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            // === Activations (forward pass) ===
            {"relu", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = relu(x=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            {"tanh", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = tanh(x=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            {"sigmoid", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = sigmoid(x=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            {"silu (SwiGLU needs this)", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = silu(x=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            {"gelu", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = gelu(x=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            // === Math (needed for RMSNorm, softmax grads) ===
            {"exp", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = exp(x=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            {"log", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = log(x=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            {"rsqrt", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = rsqrt(x=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            {"sqrt", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = sqrt(x=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            {"pow (x^2)", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = pow(x=x, y=x)[name=string(\"out\")];\n", CH, SP, CH, SP]},

            // === Reduction (needed for norms, loss) ===
            {"reduce_mean (axis=-1)", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([-1])];\n"
                "        bool ka = const()[name=string(\"ka\"), val=bool(true)];\n"
                "        tensor<fp16, [1,%d,1,1]> y = reduce_mean(x=x, axes=ax, keep_dims=ka)[name=string(\"out\")];\n",
                CH, SP, CH]},

            {"reduce_sum (axis=-1)", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([-1])];\n"
                "        bool ka = const()[name=string(\"ka\"), val=bool(true)];\n"
                "        tensor<fp16, [1,%d,1,1]> y = reduce_sum(x=x, axes=ax, keep_dims=ka)[name=string(\"out\")];\n",
                CH, SP, CH]},

            {"reduce_sum_square", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([-1])];\n"
                "        bool ka = const()[name=string(\"ka\"), val=bool(true)];\n"
                "        tensor<fp16, [1,%d,1,1]> y = reduce_sum_square(x=x, axes=ax, keep_dims=ka)[name=string(\"out\")];\n",
                CH, SP, CH]},

            // === Reshape/Transpose (needed for attention) ===
            {"transpose (perm=[0,3,2,1])", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,3,2,1])];\n"
                "        tensor<fp16, [1,%d,1,%d]> y = transpose(x=x, perm=pm)[name=string(\"out\")];\n",
                CH, SP, SP, CH]},

            {"reshape", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [4]> sh = const()[name=string(\"sh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
                "        tensor<fp16, [1,%d,1,%d]> y = reshape(x=x, shape=sh)[name=string(\"out\")];\n",
                CH, SP, SP, CH, SP, CH]},

            // === Matmul (if supported natively — huge for training) ===
            {"matmul", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = matmul(x=x, y=x, transpose_x=false, transpose_y=true)"
                "[name=string(\"out\")];\n",
                CH, SP, CH, CH]},

            // === Softmax (attention) ===
            {"softmax (axis=-1)", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        int32 ax = const()[name=string(\"ax\"), val=int32(-1)];\n"
                "        tensor<fp16, [1,%d,1,%d]> y = softmax(x=x, axis=ax)[name=string(\"out\")];\n",
                CH, SP, CH, SP]},

            // === Concat (needed for SwiGLU gate+up split) ===
            {"concat (axis=1)", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        int32 ax = const()[name=string(\"ax\"), val=int32(1)];\n"
                "        tensor<fp16, [1,%d,1,%d]> y = concat(values=(x, x), axis=ax, interleave=false)"
                "[name=string(\"out\")];\n",
                CH, SP, CH*2, SP]},

            // === Slice (needed for SwiGLU split) ===
            {"slice_by_size", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [4]> bg = const()[name=string(\"bg\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
                "        tensor<fp16, [1,%d,1,%d]> y = slice_by_size(x=x, begin=bg, size=sz)[name=string(\"out\")];\n",
                CH, SP, CH/2, SP, CH/2, SP]},

            // === Negate (gradient computation) ===
            {"neg", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = sub(x=x, y=x)[name=string(\"z\")];\n"
                "        tensor<fp16, [1,%d,1,%d]> y2 = sub(x=y, y=x)[name=string(\"out\")];\n",
                CH, SP, CH, SP, CH, SP]},

            // === Square (for RMSNorm: mean(x^2)) ===
            {"square (x*x for RMSNorm)", [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1,%d,1,%d]> y = mul(x=x, y=x)[name=string(\"out\")];\n",
                CH, SP, CH, SP]},
        };
        int nops = sizeof(ops) / sizeof(ops[0]);

        for (int i = 0; i < nops; i++) {
            NSString *mil = [NSString stringWithFormat:@"%@%@%@", MIL_HDR, ops[i].body, MIL_FTR];
            id mdl = re_try_op_mil(mil, out, [NSString stringWithUTF8String:ops[i].name]);
            if (mdl) {
                // Quick bench: 50 evals
                int outCh = CH, outSp = SP;
                // Adjust output size for ops that change shape
                if (strstr(ops[i].name, "reduce_mean") || strstr(ops[i].name, "reduce_sum")) {
                    outSp = 1;
                }
                if (strstr(ops[i].name, "concat")) {
                    outCh = CH * 2;
                }
                if (strstr(ops[i].name, "slice")) {
                    outCh = CH / 2;
                }
                if (strstr(ops[i].name, "matmul")) {
                    outSp = CH; // [1,CH,1,CH]
                }
                if (strstr(ops[i].name, "transpose") || strstr(ops[i].name, "reshape")) {
                    outCh = SP; outSp = CH;
                }

                IOSurfaceRef ioIn  = re_surface(CH * SP * 2);
                IOSurfaceRef ioOut = re_surface(outCh * outSp * 2);
                id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(re_IOObj, @selector(objectWithIOSurface:), ioIn);
                id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(re_IOObj, @selector(objectWithIOSurface:), ioOut);
                id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(re_Request,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

                NSError *e = nil;
                for (int w = 0; w < 3; w++)
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);

                uint64_t t0 = mach_absolute_time();
                int iters = 50;
                BOOL evalOk = YES;
                for (int j = 0; j < iters; j++) {
                    BOOL r = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
                    if (!r) { evalOk = NO; break; }
                }
                double ms = re_ms(mach_absolute_time() - t0);

                if (evalOk) {
                    [out appendFormat:@"  %-35s  OK   %.3f ms/eval\n", ops[i].name, ms/iters];
                } else {
                    [out appendFormat:@"  %-35s  COMPILE OK, EVAL FAIL\n", ops[i].name];
                }

                re_unload(mdl);
                CFRelease(ioIn); CFRelease(ioOut);
            } else {
                [out appendFormat:@"  %-35s  FAIL (not supported on ANE)\n", ops[i].name];
            }
        }

        return out;
    }
}

// ================================================================
// TEST 3: Performance Stats Extraction
// What can _ANEPerformanceStats tell us?
// ================================================================
NSString *ane_re_perf_stats(void) {
    @autoreleasepool {
        re_ensure();
        NSMutableString *out = [NSMutableString string];
        if (!re_Desc || !re_Model) { [out appendString:@"  Classes missing\n"]; return out; }

        [out appendString:@"  --- Performance Stats Probe ---\n"];

        // Enumerate _ANEPerformanceStats and _ANEPerformanceStatsIOSurface methods
        Class perfCls = NSClassFromString(@"_ANEPerformanceStats");
        Class perfIOCls = NSClassFromString(@"_ANEPerformanceStatsIOSurface");

        if (perfCls) {
            [out appendString:@"\n  _ANEPerformanceStats methods:\n"];
            unsigned int mc = 0;
            Method *methods = class_copyMethodList(perfCls, &mc);
            for (unsigned int i = 0; i < mc; i++) {
                [out appendFormat:@"    - %@\n", NSStringFromSelector(method_getName(methods[i]))];
            }
            free(methods);
            // Class methods
            Class meta = object_getClass(perfCls);
            methods = class_copyMethodList(meta, &mc);
            for (unsigned int i = 0; i < mc; i++) {
                [out appendFormat:@"    + %@\n", NSStringFromSelector(method_getName(methods[i]))];
            }
            free(methods);
        }

        if (perfIOCls) {
            [out appendString:@"\n  _ANEPerformanceStatsIOSurface methods:\n"];
            unsigned int mc = 0;
            Method *methods = class_copyMethodList(perfIOCls, &mc);
            for (unsigned int i = 0; i < mc; i++) {
                [out appendFormat:@"    - %@\n", NSStringFromSelector(method_getName(methods[i]))];
            }
            free(methods);
            Class meta = object_getClass(perfIOCls);
            methods = class_copyMethodList(meta, &mc);
            for (unsigned int i = 0; i < mc; i++) {
                [out appendFormat:@"    + %@\n", NSStringFromSelector(method_getName(methods[i]))];
            }
            free(methods);
        }

        // Try to create a perf stats IOSurface and pass it through an eval
        [out appendString:@"\n  Attempting eval with perf stats capture...\n"];
        {
            int CH = 512, SP = 64;
            NSString *mil = re_conv_mil(CH, CH, SP);
            NSData *wd = re_weight_blob(CH, CH);
            id mdl = re_compile_load(mil, wd, out);
            if (mdl) {
                IOSurfaceRef ioIn  = re_surface(CH * SP * 2);
                IOSurfaceRef ioOut = re_surface(CH * SP * 2);

                // Try creating perf stats object
                id perfObj = nil;
                if (perfIOCls) {
                    // Try alloc/init
                    perfObj = ((id(*)(Class,SEL))objc_msgSend)(perfIOCls, @selector(alloc));
                    if (perfObj) {
                        perfObj = ((id(*)(id,SEL))objc_msgSend)(perfObj, @selector(init));
                    }
                    if (perfObj) {
                        [out appendFormat:@"  PerfStatsIOSurface created: %@\n", [perfObj description]];
                    } else {
                        [out appendString:@"  PerfStatsIOSurface alloc/init returned nil\n"];
                    }
                }

                id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(re_IOObj, @selector(objectWithIOSurface:), ioIn);
                id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(re_IOObj, @selector(objectWithIOSurface:), ioOut);

                // Pass perfObj in the request
                id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(re_Request,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, perfObj, @0);

                NSError *e = nil;
                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
                [out appendFormat:@"  Eval with perf stats: %@\n", ok ? @"OK" : @"FAIL"];

                if (ok && perfObj) {
                    // Try to read perf stats properties
                    unsigned int mc = 0;
                    Method *methods = class_copyMethodList([perfObj class], &mc);
                    for (unsigned int i = 0; i < mc; i++) {
                        SEL sel = method_getName(methods[i]);
                        NSString *selName = NSStringFromSelector(sel);
                        // Only call getters (no args)
                        if ([selName rangeOfString:@":"].location == NSNotFound &&
                            ![selName isEqualToString:@"description"] &&
                            ![selName isEqualToString:@"dealloc"] &&
                            ![selName isEqualToString:@"init"]) {
                            @try {
                                // Try as uint64
                                char retType[64];
                                method_getReturnType(methods[i], retType, sizeof(retType));
                                if (retType[0] == 'Q' || retType[0] == 'q' || retType[0] == 'I' || retType[0] == 'i') {
                                    // Integer return type
                                    uint64_t val = ((uint64_t(*)(id,SEL))objc_msgSend)(perfObj, sel);
                                    [out appendFormat:@"    %@ = %llu\n", selName, val];
                                } else if (retType[0] == 'd') {
                                    double val = ((double(*)(id,SEL))objc_msgSend)(perfObj, sel);
                                    [out appendFormat:@"    %@ = %f\n", selName, val];
                                } else if (retType[0] == '@') {
                                    id val = ((id(*)(id,SEL))objc_msgSend)(perfObj, sel);
                                    [out appendFormat:@"    %@ = %@\n", selName, val];
                                }
                            } @catch (NSException *ex) {
                                [out appendFormat:@"    %@ = (exception: %@)\n", selName, ex.reason];
                            }
                        }
                    }
                    free(methods);
                }

                re_unload(mdl);
                CFRelease(ioIn); CFRelease(ioOut);
            }
        }

        return out;
    }
}

// ================================================================
// TEST 4: Compile Limit Probe
// How many models can we compile before hitting the ~119 limit?
// Does purgeCompiledModel reclaim slots?
// ================================================================
NSString *ane_re_compile_limits(void) {
    @autoreleasepool {
        re_ensure();
        NSMutableString *out = [NSMutableString string];
        if (!re_Desc || !re_Model) { [out appendString:@"  Classes missing\n"]; return out; }

        [out appendString:@"  --- Compile Limit Probe ---\n"];
        [out appendString:@"  Compiling small models until failure...\n"];

        int CH = 64, SP = 16; // Tiny to minimize memory
        NSMutableArray *models = [NSMutableArray array];
        int maxCompile = 150; // Try up to 150
        int failAt = -1;

        for (int i = 0; i < maxCompile; i++) {
            // Vary weights slightly so each model is unique
            int ws = CH * CH * 2, tot = 128 + ws;
            uint8_t *b = (uint8_t*)calloc(tot, 1);
            b[0]=1; b[4]=2;
            b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
            *(uint32_t*)(b+72) = ws; *(uint32_t*)(b+80) = 128;
            _Float16 *w = (_Float16*)(b+128);
            for (int j = 0; j < CH*CH; j++) w[j] = (_Float16)(0.001 * (i + j));
            NSData *wdata = [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];

            NSString *mil = re_conv_mil(CH, CH, SP);
            NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(re_Desc,
                @selector(modelWithMILText:weights:optionsPlist:),
                md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
            if (!desc) { failAt = i; [out appendFormat:@"  desc=nil at model %d\n", i]; break; }

            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(re_Model, @selector(inMemoryModelWithDescriptor:), desc);
            if (!mdl) { failAt = i; [out appendFormat:@"  model=nil at %d\n", i]; break; }

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
            if (!ok) {
                failAt = i;
                [out appendFormat:@"  Compile failed at model %d: %@\n", i, e.localizedDescription];
                [fm removeItemAtPath:td error:nil];
                break;
            }

            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 9, @{}, &e);
            if (!ok) {
                failAt = i;
                [out appendFormat:@"  Load failed at model %d: %@\n", i, e.localizedDescription];
                [fm removeItemAtPath:td error:nil];
                break;
            }

            [models addObject:mdl];

            if ((i+1) % 20 == 0) {
                [out appendFormat:@"  ... compiled %d models OK\n", i+1];
            }
        }

        if (failAt < 0) {
            [out appendFormat:@"  Compiled all %d models without failure!\n", maxCompile];
        } else {
            [out appendFormat:@"  LIMIT HIT: %d models compiled before failure\n", failAt];
        }

        // Now test: unload some and try to compile more
        if (models.count > 0 && failAt > 0) {
            [out appendString:@"\n  Testing purge/unload reclaim...\n"];
            int unloadCount = MIN(10, (int)models.count);
            for (int i = 0; i < unloadCount; i++) {
                re_unload(models[i]);
            }
            [models removeObjectsInRange:NSMakeRange(0, unloadCount)];
            [out appendFormat:@"  Unloaded %d models\n", unloadCount];

            // Try to compile new ones
            int reclaimOk = 0;
            for (int i = 0; i < unloadCount; i++) {
                NSString *mil = re_conv_mil(CH, CH, SP);
                NSData *wd = re_weight_blob(CH, CH);
                id mdl = re_compile_load(mil, wd, out);
                if (mdl) {
                    reclaimOk++;
                    [models addObject:mdl];
                } else {
                    break;
                }
            }
            [out appendFormat:@"  Reclaimed %d/%d slots after unload: %@\n",
                reclaimOk, unloadCount, reclaimOk > 0 ? @"RECLAIM WORKS" : @"NO RECLAIM"];
        }

        // Cleanup
        for (id mdl in models) {
            re_unload(mdl);
        }

        return out;
    }
}
