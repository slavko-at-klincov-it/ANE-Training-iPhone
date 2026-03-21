// ANEReduceDebug.m — Debug reduce_mean/sum eval failures
// Hypothesis: IOSurface output size mismatch or axis issue
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurfaceRef.h>
#import <mach/mach_time.h>

static Class rd_D, rd_M, rd_R, rd_IO;
static mach_timebase_info_data_t rd_tb;

static void rd_ensure(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_LAZY);
        rd_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        rd_M  = NSClassFromString(@"_ANEInMemoryModel");
        rd_R  = NSClassFromString(@"_ANERequest");
        rd_IO = NSClassFromString(@"_ANEIOSurfaceObject");
        mach_timebase_info(&rd_tb);
    });
}

static double rd_ms(uint64_t t) { return (double)t * rd_tb.numer / rd_tb.denom / 1e6; }

static IOSurfaceRef rd_surface(size_t bytes) {
    // Ensure minimum 16KB (ANE may have alignment requirements)
    if (bytes < 16384) bytes = 16384;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

#define MIL_FTR @"    } -> (y);\n}\n"

// Try compile+load+eval for a given MIL program, with specified IOSurface sizes
static NSString *rd_try(NSString *label, NSString *mil, size_t inBytes, size_t outBytes) {
    NSMutableString *out = [NSMutableString string];

    // Minimal weight blob
    int tot = 128 + 2;
    uint8_t *b = (uint8_t*)calloc(tot, 1);
    b[0]=1; b[4]=2;
    b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
    *(uint32_t*)(b+72) = 2; *(uint32_t*)(b+80) = 128;
    NSData *wdata = [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];

    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(rd_D,
        @selector(modelWithMILText:weights:optionsPlist:),
        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
    if (!desc) { [out appendFormat:@"    %-50s DESC=nil\n", label.UTF8String]; return out; }

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(rd_M, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) { [out appendFormat:@"    %-50s MODEL=nil\n", label.UTF8String]; return out; }

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
        [out appendFormat:@"    %-50s COMPILE FAIL: %@\n", label.UTF8String, e.localizedDescription];
        [fm removeItemAtPath:td error:nil];
        return out;
    }

    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 9, @{}, &e);
    if (!ok) {
        [out appendFormat:@"    %-50s LOAD FAIL: %@\n", label.UTF8String, e.localizedDescription];
        [fm removeItemAtPath:td error:nil];
        return out;
    }

    // Fill input with 1.0 (FP16 = 0x3C00)
    IOSurfaceRef ioIn = rd_surface(inBytes);
    IOSurfaceRef ioOut = rd_surface(outBytes);

    IOSurfaceLock(ioIn, 0, NULL);
    uint16_t *ip = (uint16_t*)IOSurfaceGetBaseAddress(ioIn);
    for (size_t i = 0; i < inBytes/2; i++) ip[i] = 0x3C00;
    IOSurfaceUnlock(ioIn, 0, NULL);

    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(rd_IO, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(rd_IO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(rd_R,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

    e = nil;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);

    if (ok) {
        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        uint16_t *op = (uint16_t*)IOSurfaceGetBaseAddress(ioOut);
        // Read first few output values
        [out appendFormat:@"    %-50s OK  out[0..3]=0x%04X 0x%04X 0x%04X 0x%04X\n",
            label.UTF8String, op[0], op[1], op[2], op[3]];
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
    } else {
        [out appendFormat:@"    %-50s EVAL FAIL: %@\n", label.UTF8String,
            e ? e.localizedDescription : @"(no error)"];
    }

    // Cleanup
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 9, &e);
    [fm removeItemAtPath:td error:nil];
    CFRelease(ioIn); CFRelease(ioOut);

    return out;
}

NSString *ane_reduce_debug(void) {
    @autoreleasepool {
        rd_ensure();
        NSMutableString *out = [NSMutableString string];
        if (!rd_D || !rd_M) { [out appendString:@"  Classes missing\n"]; return out; }

        [out appendString:@"  --- Reduce Op Debug ---\n\n"];

        int CH = 256, SP = 64;
        size_t inBytes = CH * SP * 2;

        // ============================================================
        // Test 1: reduce_mean axis=-1 (spatial), various output IOSurface sizes
        // ============================================================
        [out appendString:@"  == Test 1: reduce_mean axis=-1, vary output IOSurface size ==\n"];
        {
            NSString *body = [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([-1])];\n"
                "        bool ka = const()[name=string(\"ka\"), val=bool(true)];\n"
                "        tensor<fp16, [1,%d,1,1]> y = reduce_mean(x=x, axes=ax, keep_dims=ka)[name=string(\"out\")];\n",
                CH, SP, CH];
            NSString *mil = [NSString stringWithFormat:@"%@%@%@", MIL_HDR, body, MIL_FTR];

            // Try different output sizes
            size_t sizes[] = {
                CH * 1 * 2,          // exact: 512 bytes
                CH * 2,              // same
                16384,               // 16KB minimum
                32768,               // 32KB
                65536,               // 64KB
                CH * SP * 2,         // same as input size
            };
            NSString *labels[] = {
                @"out=CH*1*2 (512B)",
                @"out=CH*2 (512B)",
                @"out=16KB",
                @"out=32KB",
                @"out=64KB",
                @"out=same as input",
            };
            for (int i = 0; i < 6; i++) {
                [out appendString:rd_try(labels[i], mil, inBytes, sizes[i])];
            }
        }

        // ============================================================
        // Test 2: reduce_mean axis=1 (channel axis), output=[1,1,1,SP]
        // ============================================================
        [out appendString:@"\n  == Test 2: reduce_mean axis=1 (channel), vary output ==\n"];
        {
            NSString *body = [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\n"
                "        bool ka = const()[name=string(\"ka\"), val=bool(true)];\n"
                "        tensor<fp16, [1,1,1,%d]> y = reduce_mean(x=x, axes=ax, keep_dims=ka)[name=string(\"out\")];\n",
                CH, SP, SP];
            NSString *mil = [NSString stringWithFormat:@"%@%@%@", MIL_HDR, body, MIL_FTR];

            [out appendString:rd_try(@"axis=1 out=16KB", mil, inBytes, 16384)];
            [out appendString:rd_try(@"axis=1 out=same-as-input", mil, inBytes, inBytes)];
        }

        // ============================================================
        // Test 3: reduce_sum axis=-1
        // ============================================================
        [out appendString:@"\n  == Test 3: reduce_sum axis=-1, various output sizes ==\n"];
        {
            NSString *body = [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([-1])];\n"
                "        bool ka = const()[name=string(\"ka\"), val=bool(true)];\n"
                "        tensor<fp16, [1,%d,1,1]> y = reduce_sum(x=x, axes=ax, keep_dims=ka)[name=string(\"out\")];\n",
                CH, SP, CH];
            NSString *mil = [NSString stringWithFormat:@"%@%@%@", MIL_HDR, body, MIL_FTR];

            [out appendString:rd_try(@"reduce_sum out=16KB", mil, inBytes, 16384)];
            [out appendString:rd_try(@"reduce_sum out=same-as-input", mil, inBytes, inBytes)];
        }

        // ============================================================
        // Test 4: reduce_mean keep_dims=false
        // ============================================================
        [out appendString:@"\n  == Test 4: reduce_mean keep_dims=false ==\n"];
        {
            // Output shape: [1, CH, 1] (3D) — but ANE always wants 4D...
            // Try with 3D output type
            NSString *body3d = [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([-1])];\n"
                "        bool ka = const()[name=string(\"ka\"), val=bool(false)];\n"
                "        tensor<fp16, [1,%d,1]> y = reduce_mean(x=x, axes=ax, keep_dims=ka)[name=string(\"out\")];\n",
                CH, SP, CH];
            NSString *mil3d = [NSString stringWithFormat:@"%@%@%@", MIL_HDR, body3d, MIL_FTR];
            [out appendString:rd_try(@"keep_dims=false (3D output)", mil3d, inBytes, 16384)];
        }

        // ============================================================
        // Test 5: Smaller tensor — maybe alignment issue at 256ch
        // ============================================================
        [out appendString:@"\n  == Test 5: Small tensor reduce (16ch, sp=16) ==\n"];
        {
            int sCH = 16, sSP = 16;
            size_t sIn = sCH * sSP * 2;

            NSString *body = [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([-1])];\n"
                "        bool ka = const()[name=string(\"ka\"), val=bool(true)];\n"
                "        tensor<fp16, [1,%d,1,1]> y = reduce_mean(x=x, axes=ax, keep_dims=ka)[name=string(\"out\")];\n",
                sCH, sSP, sCH];
            NSString *mil = [NSString stringWithFormat:@"%@%@%@", MIL_HDR, body, MIL_FTR];

            [out appendString:rd_try(@"16ch reduce_mean out=16KB", mil, sIn, 16384)];
            [out appendString:rd_try(@"16ch reduce_mean out=same-as-input", mil, sIn, sIn)];
        }

        // ============================================================
        // Test 6: Manual reduce — mul by 1/N constant then sum via conv trick
        // This is the workaround if native reduce doesn't work
        // ============================================================
        [out appendString:@"\n  == Test 6: Manual mean via mul + conv(all-ones kernel) ==\n"];
        {
            // mean(x, axis=-1) = (1/SP) * sum(x, axis=-1)
            // sum via 1x1 conv: conv(x, ones_kernel) with ch_in=SP, ch_out=1
            // But wait — we need to reduce over spatial, not channels
            // Alternative: transpose [1,CH,1,SP] -> [1,SP,1,CH], conv 1x1 (SP->1), transpose back
            // Or simpler: just use mul(x, 1/SP) then test if that works
            // Actually simplest manual mean: x * (1/N) summed — but we need the sum first
            // Let's try: reshape to make spatial into channel, then conv to reduce
            // reshape [1,CH,1,SP] -> [1, CH*SP, 1, 1]... doesn't help

            // Simpler: test if reduce works over channel axis (axis=1) with spatial=1
            // This maps to "how many channels can be reduced"
            int sCH = 64, sSP = 1;
            size_t sIn = sCH * sSP * 2;

            NSString *body = [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([1])];\n"
                "        bool ka = const()[name=string(\"ka\"), val=bool(true)];\n"
                "        tensor<fp16, [1,1,1,%d]> y = reduce_mean(x=x, axes=ax, keep_dims=ka)[name=string(\"out\")];\n",
                sCH, sSP, sSP];
            NSString *mil = [NSString stringWithFormat:@"%@%@%@", MIL_HDR, body, MIL_FTR];

            [out appendString:rd_try(@"64ch sp=1 reduce_mean axis=1 out=16KB", mil, sIn, 16384)];
        }

        // ============================================================
        // Test 7: reduce_mean axis=3 (explicit instead of -1)
        // ============================================================
        [out appendString:@"\n  == Test 7: reduce_mean axis=3 (explicit) ==\n"];
        {
            NSString *body = [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([3])];\n"
                "        bool ka = const()[name=string(\"ka\"), val=bool(true)];\n"
                "        tensor<fp16, [1,%d,1,1]> y = reduce_mean(x=x, axes=ax, keep_dims=ka)[name=string(\"out\")];\n",
                CH, SP, CH];
            NSString *mil = [NSString stringWithFormat:@"%@%@%@", MIL_HDR, body, MIL_FTR];

            [out appendString:rd_try(@"axis=3 explicit out=16KB", mil, inBytes, 16384)];
            [out appendString:rd_try(@"axis=3 explicit out=same-as-input", mil, inBytes, inBytes)];
        }

        // ============================================================
        // Test 8: Use mapIOSurfaces approach (maybe we need to map first?)
        // ============================================================
        [out appendString:@"\n  == Test 8: reduce with mapped IOSurfaces ==\n"];
        {
            NSString *body = [NSString stringWithFormat:
                @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<int32, [1]> ax = const()[name=string(\"ax\"), val=tensor<int32, [1]>([-1])];\n"
                "        bool ka = const()[name=string(\"ka\"), val=bool(true)];\n"
                "        tensor<fp16, [1,%d,1,1]> y = reduce_mean(x=x, axes=ax, keep_dims=ka)[name=string(\"out\")];\n",
                CH, SP, CH];
            NSString *mil = [NSString stringWithFormat:@"%@%@%@", MIL_HDR, body, MIL_FTR];

            // Compile+load manually so we can try mapIOSurfaces
            int tot = 128 + 2;
            uint8_t *b = (uint8_t*)calloc(tot, 1);
            b[0]=1; b[4]=2;
            b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
            *(uint32_t*)(b+72) = 2; *(uint32_t*)(b+80) = 128;
            NSData *wdata = [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];

            NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(rd_D,
                @selector(modelWithMILText:weights:optionsPlist:),
                md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(rd_M, @selector(inMemoryModelWithDescriptor:), desc);

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
            if (!ok) { [out appendFormat:@"    compile fail: %@\n", e.localizedDescription]; }
            else {
                ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl, @selector(loadWithQoS:options:error:), 9, @{}, &e);
                if (!ok) { [out appendFormat:@"    load fail: %@\n", e.localizedDescription]; }
                else {
                    IOSurfaceRef ioIn = rd_surface(inBytes);
                    IOSurfaceRef ioOut = rd_surface(inBytes); // same size as input

                    IOSurfaceLock(ioIn, 0, NULL);
                    uint16_t *ip = (uint16_t*)IOSurfaceGetBaseAddress(ioIn);
                    for (size_t i = 0; i < inBytes/2; i++) ip[i] = 0x3C00;
                    IOSurfaceUnlock(ioIn, 0, NULL);

                    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(rd_IO, @selector(objectWithIOSurface:), ioIn);
                    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(rd_IO, @selector(objectWithIOSurface:), ioOut);
                    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(rd_R,
                        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

                    // Try mapIOSurfaces first
                    BOOL mapOk = ((BOOL(*)(id,SEL,id,BOOL,NSError**))objc_msgSend)(
                        mdl, @selector(mapIOSurfacesWithRequest:cacheInference:error:), req, NO, &e);
                    [out appendFormat:@"    mapIOSurfaces: %@ %@\n", mapOk ? @"OK" : @"FAIL",
                        e ? e.localizedDescription : @""];

                    e = nil;
                    ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
                    if (ok) {
                        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
                        uint16_t *op = (uint16_t*)IOSurfaceGetBaseAddress(ioOut);
                        [out appendFormat:@"    eval with map: OK  out[0..3]=0x%04X 0x%04X 0x%04X 0x%04X\n",
                            op[0], op[1], op[2], op[3]];
                        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
                    } else {
                        [out appendFormat:@"    eval with map: FAIL %@\n", e ? e.localizedDescription : @"(no error)"];
                    }

                    // Cleanup
                    ((void(*)(id,SEL,id))objc_msgSend)(mdl, @selector(unmapIOSurfacesWithRequest:), req);
                    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 9, &e);
                    [fm removeItemAtPath:td error:nil];
                    CFRelease(ioIn); CFRelease(ioOut);
                }
            }
        }

        return out;
    }
}
