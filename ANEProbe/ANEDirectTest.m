// ANEDirectTest.m — Direct ANE compilation & evaluation on iOS
// Ported from macOS bench_ane_peak.m
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurfaceRef.h>
#import <mach/mach_time.h>

static Class g_Desc, g_Model, g_Request, g_IOObj;
static mach_timebase_info_data_t g_tb;

static void ensure_classes(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_LAZY);
        g_Desc    = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_Model   = NSClassFromString(@"_ANEInMemoryModel");
        g_Request = NSClassFromString(@"_ANERequest");
        g_IOObj   = NSClassFromString(@"_ANEIOSurfaceObject");
        mach_timebase_info(&g_tb);
    });
}

static double tb_ms(uint64_t t) {
    return (double)t * g_tb.numer / g_tb.denom / 1e6;
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0
    });
}

// Helper: compile+load a single conv kernel, returns nil on failure
static id compile_conv(int ch_in, int ch_out, int sp, NSMutableString *out) {
    int ws = ch_out * ch_in * 2;
    int tot = 128 + ws;
    uint8_t *blob = (uint8_t *)calloc(tot, 1);
    blob[0]=1; blob[4]=2;
    blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
    *(uint32_t*)(blob+72) = ws;
    *(uint32_t*)(blob+80) = 128;
    _Float16 *wp = (_Float16*)(blob+128);
    srand48(42);
    for (int i = 0; i < ch_out*ch_in; i++) wp[i] = (_Float16)(0.01*(2*drand48()-1));
    NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];

    NSString *mil = [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)"
        "[name=string(\"out\")];\n"
        "    } -> (y);\n"
        "}\n", ch_in, sp, ch_out, ch_in, ch_out, ch_in, ch_out, sp];

    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_Desc,
        @selector(modelWithMILText:weights:optionsPlist:),
        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
    if (!desc) return nil;

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_Model, @selector(inMemoryModelWithDescriptor:), desc);
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
        [out appendFormat:@"  COMPILE FAIL %dx%d sp%d: %@\n", ch_in, ch_out, sp, e.localizedDescription];
        [fm removeItemAtPath:td error:nil];
        return nil;
    }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 9, @{}, &e);
    if (!ok) {
        [out appendFormat:@"  LOAD FAIL %dx%d sp%d: %@\n", ch_in, ch_out, sp, e.localizedDescription];
        [fm removeItemAtPath:td error:nil];
        return nil;
    }
    return mdl;
}

static void unload_model(id mdl) {
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
        mdl, @selector(unloadWithQoS:error:), 9, &e);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
}

static void bench_kernel(int ch_in, int ch_out, int sp, int iters, NSMutableString *out) {
    id mdl = compile_conv(ch_in, ch_out, sp, out);
    if (!mdl) return;

    IOSurfaceRef ioIn  = make_surface(ch_in * sp * 2);
    IOSurfaceRef ioOut = make_surface(ch_out * sp * 2);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOObj, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOObj, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_Request,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

    NSError *e = nil;
    // Warmup
    for (int i = 0; i < 5; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
    double ms = tb_ms(mach_absolute_time() - t0);
    double per = ms / iters;
    double flops = 2.0 * ch_in * ch_out * sp;
    double tflops = flops / (per * 1e9);
    double wMB = (double)(ch_out * ch_in * 2) / (1024.0 * 1024.0);

    [out appendFormat:@"  %4dx%-4d sp=%-3d  w=%.1fMB  %6.3f ms/eval  %6.2f TFLOPS\n",
        ch_in, ch_out, sp, wMB, per, tflops];

    unload_model(mdl);
    CFRelease(ioIn);
    CFRelease(ioOut);
}

NSString *ane_direct_test(void) {
    @autoreleasepool {
        ensure_classes();
        NSMutableString *out = [NSMutableString string];

        if (!g_Desc || !g_Model || !g_Request || !g_IOObj) {
            [out appendString:@"  FAILED: ANE classes not found\n"];
            return out;
        }

        // --- Phase 1: Proof of concept with identity conv ---
        [out appendString:@"  --- Phase 1: Identity Conv Proof ---\n"];
        {
            int CH = 256, SP = 64;
            int ws = CH*CH*2, tot = 128+ws;
            uint8_t *blob = (uint8_t*)calloc(tot, 1);
            blob[0]=1; blob[4]=2;
            blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
            *(uint32_t*)(blob+72) = ws; *(uint32_t*)(blob+80) = 128;
            _Float16 *wp = (_Float16*)(blob+128);
            for (int i = 0; i < CH; i++) wp[i*CH+i] = (_Float16)1.0f;
            NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];

            NSString *mil = [NSString stringWithFormat:
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
                "[name=string(\"out\")];\n    } -> (y);\n}\n", CH, SP, CH, CH, CH, CH, CH, SP];

            NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_Desc,
                @selector(modelWithMILText:weights:optionsPlist:),
                md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_Model, @selector(inMemoryModelWithDescriptor:), desc);
            id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
            NSFileManager *fm = [NSFileManager defaultManager];
            [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

            NSError *e = nil;
            uint64_t t0 = mach_absolute_time();
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 9, @{}, &e);
            [out appendFormat:@"  Compile: %@ (%.1f ms)\n", ok?@"OK":@"FAIL", tb_ms(mach_absolute_time()-t0)];
            if (!ok) { [fm removeItemAtPath:td error:nil]; return out; }

            t0 = mach_absolute_time();
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 9, @{}, &e);
            [out appendFormat:@"  Load: %@ (%.1f ms)\n", ok?@"OK":@"FAIL", tb_ms(mach_absolute_time()-t0)];

            IOSurfaceRef ioIn = make_surface(CH*SP*2), ioOut = make_surface(CH*SP*2);
            IOSurfaceLock(ioIn, 0, NULL);
            uint16_t *p = (uint16_t*)IOSurfaceGetBaseAddress(ioIn);
            for (int i = 0; i < CH*SP; i++) p[i] = 0x3C00;
            IOSurfaceUnlock(ioIn, 0, NULL);

            id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOObj, @selector(objectWithIOSurface:), ioIn);
            id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOObj, @selector(objectWithIOSurface:), ioOut);
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_Request,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
            IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
            uint16_t *op = (uint16_t*)IOSurfaceGetBaseAddress(ioOut);
            int correct = 0;
            for (int i = 0; i < CH*SP; i++) if (op[i]==0x3C00) correct++;
            IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
            [out appendFormat:@"  Eval: %@, Output: %d/%d correct\n", ok?@"OK":@"FAIL", correct, CH*SP];

            unload_model(mdl);
            CFRelease(ioIn); CFRelease(ioOut);
        }

        // --- Phase 2: Throughput benchmark suite ---
        [out appendString:@"\n  --- Phase 2: Throughput Benchmark ---\n"];
        [out appendString:@"  Config               Weight    ms/eval   TFLOPS\n"];

        // Single conv throughput
        bench_kernel(256,  256,  64, 200, out);
        bench_kernel(512,  512,  64, 200, out);
        bench_kernel(1024, 1024, 64, 100, out);
        bench_kernel(2048, 2048, 64, 50,  out);
        bench_kernel(4096, 4096, 64, 20,  out);

        // Spatial sweep at 512ch
        [out appendString:@"\n  --- Spatial sweep (512ch) ---\n"];
        bench_kernel(512, 512,  16, 200, out);
        bench_kernel(512, 512,  32, 200, out);
        bench_kernel(512, 512,  64, 200, out);
        bench_kernel(512, 512, 128, 200, out);
        bench_kernel(512, 512, 256, 100, out);

        // Stacked benchmark (16x sequential conv for peak — stay within compile budget)
        [out appendString:@"\n  --- Stacked peak (16x conv 512ch sp64) ---\n"];
        {
            int N = 16;
            id models[16];
            IOSurfaceRef ios_in[16], ios_out[16];
            id reqs[16];
            int compiled = 0;
            for (int i = 0; i < N; i++) {
                models[i] = compile_conv(512, 512, 64, out);
                if (!models[i]) break;
                ios_in[i]  = make_surface(512*64*2);
                ios_out[i] = make_surface(512*64*2);
                id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOObj, @selector(objectWithIOSurface:), ios_in[i]);
                id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_IOObj, @selector(objectWithIOSurface:), ios_out[i]);
                reqs[i] = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_Request,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
                compiled++;
            }
            [out appendFormat:@"  Compiled %d/%d kernels\n", compiled, N];

            if (compiled > 0) {
                NSError *e = nil;
                // Warmup
                for (int i = 0; i < compiled; i++)
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        models[i], @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqs[i], &e);

                int iters = 5;
                uint64_t t0 = mach_absolute_time();
                for (int iter = 0; iter < iters; iter++) {
                    for (int i = 0; i < compiled; i++)
                        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                            models[i], @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqs[i], &e);
                }
                double ms = tb_ms(mach_absolute_time() - t0);
                double per = ms / (iters * compiled);
                double gflop = 2.0*512*512*64 / 1e9;
                double total_gflop = gflop * compiled;
                double tflops = total_gflop / (ms/iters/1000.0);
                [out appendFormat:@"  %dx%d evals: %.1f ms/pass, %.3f ms/kernel, %.2f TFLOPS peak\n",
                    iters, compiled, ms/iters, per, tflops];
            }

            for (int i = 0; i < compiled; i++) {
                unload_model(models[i]);
                CFRelease(ios_in[i]); CFRelease(ios_out[i]);
            }
        }

        return out;
    }
}
