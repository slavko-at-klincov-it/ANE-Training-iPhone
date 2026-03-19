// ANEWeightTest.m — Test weight update strategies on iOS
// Tests: (A) Recompile, (B) Dynamic spatial packing (weights-as-input)
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurfaceRef.h>
#import <mach/mach_time.h>

static Class g_D, g_M, g_R, g_IO;
static mach_timebase_info_data_t g_tb;

static void wt_ensure(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_LAZY);
        g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_M  = NSClassFromString(@"_ANEInMemoryModel");
        g_R  = NSClassFromString(@"_ANERequest");
        g_IO = NSClassFromString(@"_ANEIOSurfaceObject");
        mach_timebase_info(&g_tb);
    });
}

static double wt_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static IOSurfaceRef wt_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Build weight blob with scale factor
static NSData *make_weight_blob(int ch, float scale) {
    int ws = ch * ch * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0]=1; b[4]=2;
    b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
    *(uint32_t*)(b+72) = ws; *(uint32_t*)(b+80) = 128;
    _Float16 *w = (_Float16*)(b+128);
    for (int i = 0; i < ch; i++) w[i*ch+i] = (_Float16)scale;
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

static NSString *make_conv_mil(int ch, int sp) {
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
        "[name=string(\"out\")];\n    } -> (y);\n}\n", ch, sp, ch, ch, ch, ch, ch, sp];
}

// Read first output value as FP16
static uint16_t read_output(IOSurfaceRef io) {
    IOSurfaceLock(io, kIOSurfaceLockReadOnly, NULL);
    uint16_t v = *(uint16_t*)IOSurfaceGetBaseAddress(io);
    IOSurfaceUnlock(io, kIOSurfaceLockReadOnly, NULL);
    return v;
}

// Fill input with value
static void fill_input(IOSurfaceRef io, uint16_t val, int count) {
    IOSurfaceLock(io, 0, NULL);
    uint16_t *p = (uint16_t*)IOSurfaceGetBaseAddress(io);
    for (int i = 0; i < count; i++) p[i] = val;
    IOSurfaceUnlock(io, 0, NULL);
}

// Compile+load helper — returns model (caller must unload)
static id compile_and_load(NSString *mil, NSData *wdata, NSMutableString *out) {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
        @selector(modelWithMILText:weights:optionsPlist:),
        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
    if (!desc) { [out appendString:@"    desc=nil\n"]; return nil; }

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_M, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) { [out appendString:@"    model=nil\n"]; return nil; }

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
    if (!ok) { [out appendFormat:@"    compile fail: %@\n", e]; [fm removeItemAtPath:td error:nil]; return nil; }

    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 9, @{}, &e);
    if (!ok) { [out appendFormat:@"    load fail: %@\n", e]; [fm removeItemAtPath:td error:nil]; return nil; }

    return mdl;
}

static void unload_cleanup(id mdl) {
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 9, &e);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
}

NSString *ane_weight_test(void) {
    @autoreleasepool {
        wt_ensure();
        NSMutableString *out = [NSMutableString string];
        if (!g_D || !g_M || !g_R || !g_IO) {
            [out appendString:@"  Classes missing\n"];
            return out;
        }

        int CH = 256, SP = 64; // Must be large enough for ANE minimum IOSurface

        // ============================================================
        // TEST A: Recompile with different weights
        // ============================================================
        [out appendString:@"  --- Test A: Recompile with new weights ---\n"];
        {
            NSString *mil = make_conv_mil(CH, SP);

            // Step 1: Compile with identity weights (scale=1.0)
            NSData *w1 = make_weight_blob(CH, 1.0f);
            uint64_t t0 = mach_absolute_time();
            id mdl1 = compile_and_load(mil, w1, out);
            double ms1 = wt_ms(mach_absolute_time() - t0);
            if (!mdl1) { [out appendString:@"    FAILED\n"]; return out; }

            IOSurfaceRef ioIn = wt_surface(CH * SP * 2);
            IOSurfaceRef ioOut = wt_surface(CH * SP * 2);
            fill_input(ioIn, 0x3C00, CH * SP); // 1.0

            id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioIn);
            id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioOut);
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_R,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl1, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
            uint16_t v1 = read_output(ioOut);
            [out appendFormat:@"    W=1.0: output[0]=0x%04X (%.1f ms compile+load)\n", v1, ms1];

            unload_cleanup(mdl1);

            // Step 2: Recompile with 3x weights (scale=3.0)
            NSData *w2 = make_weight_blob(CH, 3.0f);
            t0 = mach_absolute_time();
            id mdl2 = compile_and_load(mil, w2, out);
            double ms2 = wt_ms(mach_absolute_time() - t0);
            if (!mdl2) { [out appendString:@"    FAILED\n"]; CFRelease(ioIn); CFRelease(ioOut); return out; }

            // Reuse IOSurfaces
            wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioIn);
            wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioOut);
            req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_R,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl2, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
            uint16_t v2 = read_output(ioOut);
            [out appendFormat:@"    W=3.0: output[0]=0x%04X (%.1f ms compile+load)\n", v2, ms2];

            BOOL changed = (v1 != v2);
            [out appendFormat:@"    Output changed: %@ (1.0→3.0 expected: 0x3C00→0x4200)\n",
                changed ? @"YES ✓" : @"NO ✗"];

            // Benchmark: repeated recompile cycles (simulating training)
            int cycles = 5;
            t0 = mach_absolute_time();
            for (int c = 0; c < cycles; c++) {
                unload_cleanup(mdl2);
                float scale = 1.0f + 0.5f * c;
                NSData *wc = make_weight_blob(CH, scale);
                mdl2 = compile_and_load(mil, wc, out);
                if (!mdl2) break;
                wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioIn);
                wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioOut);
                req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_R,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl2, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
            }
            double cycleMs = wt_ms(mach_absolute_time() - t0);
            [out appendFormat:@"    %d recompile cycles: %.1f ms total, %.1f ms/cycle\n",
                cycles, cycleMs, cycleMs / cycles];

            if (mdl2) unload_cleanup(mdl2);
            CFRelease(ioIn); CFRelease(ioOut);
        }

        // ============================================================
        // TEST B: Dynamic Spatial Packing (weights-as-input)
        // ============================================================
        [out appendString:@"\n  --- Test B: Dynamic Spatial Packing ---\n"];
        {
            // Pack weights into input: [1, CH, 1, SP + CH]
            // First SP elements = activation, next CH elements = weight diagonal
            // MIL slices input → (activation, weight_diag), then does element-wise mul
            int TOTAL_SP = SP + CH;

            // Simple test: input * weight_diag (element-wise per channel)
            // MIL: slice input into x[0:SP] and w[SP:SP+CH], reshape w, mul
            NSString *mil = [NSString stringWithFormat:
                @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"
                "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> input) {\n"
                // Slice activation: input[:, :, :, 0:SP]
                "        tensor<int32, [4]> act_begin = const()[name=string(\"ab\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "        tensor<int32, [4]> act_size = const()[name=string(\"as\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
                "        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=input, begin=act_begin, size=act_size)"
                "[name=string(\"act\")];\n"
                // Slice weight: input[:, :, :, SP:SP+1] (one weight per channel)
                "        tensor<int32, [4]> w_begin = const()[name=string(\"wb\"), val=tensor<int32, [4]>([0,0,0,%d])];\n"
                "        tensor<int32, [4]> w_size = const()[name=string(\"ws\"), val=tensor<int32, [4]>([1,%d,1,1])];\n"
                "        tensor<fp16, [1,%d,1,1]> w_raw = slice_by_size(x=input, begin=w_begin, size=w_size)"
                "[name=string(\"wr\")];\n"
                // Multiply: act * w (broadcast over spatial)
                "        tensor<fp16, [1,%d,1,%d]> y = mul(x=act, y=w_raw)[name=string(\"out\")];\n"
                "    } -> (y);\n}\n",
                CH, TOTAL_SP, CH, SP, CH, SP, SP, CH, CH, CH, SP];

            // Dummy weights (no baked weights needed — weights are in input!)
            int ws = 2; // minimal weight blob
            int tot = 128 + ws;
            uint8_t *blob = (uint8_t*)calloc(tot, 1);
            blob[0]=1; blob[4]=2;
            blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
            *(uint32_t*)(blob+72) = ws; *(uint32_t*)(blob+80) = 128;
            NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];

            // Compile ONCE
            uint64_t t0 = mach_absolute_time();
            id mdl = compile_and_load(mil, wdata, out);
            double compMs = wt_ms(mach_absolute_time() - t0);
            if (!mdl) {
                [out appendString:@"    Dynamic compile FAILED\n"];
                return out;
            }
            [out appendFormat:@"    Compiled once: %.1f ms\n", compMs];

            // Create IOSurfaces
            IOSurfaceRef ioIn  = wt_surface(CH * TOTAL_SP * 2);
            IOSurfaceRef ioOut = wt_surface(CH * SP * 2);

            // --- Eval with weight=1.0 ---
            IOSurfaceLock(ioIn, 0, NULL);
            uint16_t *inp = (uint16_t*)IOSurfaceGetBaseAddress(ioIn);
            // Activation region: all 1.0
            for (int c = 0; c < CH; c++)
                for (int s = 0; s < SP; s++)
                    inp[c * TOTAL_SP + s] = 0x3C00; // 1.0
            // Weight region: all 1.0
            for (int c = 0; c < CH; c++)
                inp[c * TOTAL_SP + SP] = 0x3C00; // weight = 1.0
            IOSurfaceUnlock(ioIn, 0, NULL);

            id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioIn);
            id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioOut);
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_R,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
            uint16_t v1 = read_output(ioOut);
            [out appendFormat:@"    w=1.0: output[0]=0x%04X\n", v1];

            // --- Change weight to 3.0 WITHOUT recompile ---
            IOSurfaceLock(ioIn, 0, NULL);
            inp = (uint16_t*)IOSurfaceGetBaseAddress(ioIn);
            for (int c = 0; c < CH; c++)
                inp[c * TOTAL_SP + SP] = 0x4200; // weight = 3.0
            IOSurfaceUnlock(ioIn, 0, NULL);

            // Must recreate request with same IOSurfaces (they point to updated data)
            wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioIn);
            wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IO, @selector(objectWithIOSurface:), ioOut);
            req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_R,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
            uint16_t v2 = read_output(ioOut);
            [out appendFormat:@"    w=3.0: output[0]=0x%04X (NO recompile!)\n", v2];

            BOOL changed = (v1 != v2);
            [out appendFormat:@"    Output changed: %@ (expect 0x3C00→0x4200)\n",
                changed ? @"YES ✓ DYNAMIC WEIGHTS WORK!" : @"NO ✗"];

            // Benchmark: weight update + eval (no recompile)
            int iters = 200;
            t0 = mach_absolute_time();
            for (int i = 0; i < iters; i++) {
                // Update weight in IOSurface
                IOSurfaceLock(ioIn, 0, NULL);
                uint16_t *p = (uint16_t*)IOSurfaceGetBaseAddress(ioIn);
                uint16_t wval = 0x3C00 + (i & 0xFF); // varying weight
                for (int c = 0; c < CH; c++) p[c * TOTAL_SP + SP] = wval;
                IOSurfaceUnlock(ioIn, 0, NULL);
                // Eval
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
            }
            double dynMs = wt_ms(mach_absolute_time() - t0);
            [out appendFormat:@"    %d update+eval cycles: %.1f ms total, %.3f ms/cycle\n",
                iters, dynMs, dynMs / iters];

            unload_cleanup(mdl);
            CFRelease(ioIn); CFRelease(ioOut);
        }

        return out;
    }
}
