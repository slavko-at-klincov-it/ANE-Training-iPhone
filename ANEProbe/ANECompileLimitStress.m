// ANECompileLimitStress.m — Find the real compile limit on iOS
// Strategy: compile+load tiny models until failure, report every 50
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurfaceRef.h>
#import <mach/mach_time.h>
#import <mach/mach.h>

static Class cl_D, cl_M;
static mach_timebase_info_data_t cl_tb;

static void cl_ensure(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_LAZY);
        cl_D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        cl_M = NSClassFromString(@"_ANEInMemoryModel");
        mach_timebase_info(&cl_tb);
    });
}

static double cl_ms(uint64_t t) { return (double)t * cl_tb.numer / cl_tb.denom / 1e6; }

// Get current app memory usage in MB
static double app_memory_mb(void) {
    struct task_basic_info info;
    mach_msg_type_number_t size = sizeof(info) / sizeof(natural_t);
    kern_return_t kr = task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &size);
    if (kr != KERN_SUCCESS) return -1;
    return (double)info.resident_size / (1024.0 * 1024.0);
}

NSString *ane_compile_limit_stress(void) {
    @autoreleasepool {
        cl_ensure();
        NSMutableString *out = [NSMutableString string];
        if (!cl_D || !cl_M) { [out appendString:@"  Classes missing\n"]; return out; }

        [out appendString:@"  --- Compile Limit Stress Test ---\n"];
        [out appendFormat:@"  Starting memory: %.1f MB\n", app_memory_mb()];

        int CH = 32, SP = 8; // Tiny models to minimize memory per model
        int maxCompile = 1000;
        NSMutableArray *models = [NSMutableArray array];
        NSMutableArray *tmpDirs = [NSMutableArray array];
        int failAt = -1;
        NSString *failReason = nil;

        uint64_t totalStart = mach_absolute_time();

        for (int i = 0; i < maxCompile; i++) {
            // Unique weights per model (different hash)
            int ws = CH * CH * 2, tot = 128 + ws;
            uint8_t *b = (uint8_t*)calloc(tot, 1);
            if (!b) { failAt = i; failReason = @"calloc failed (OOM)"; break; }
            b[0]=1; b[4]=2;
            b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
            *(uint32_t*)(b+72) = ws; *(uint32_t*)(b+80) = 128;
            _Float16 *w = (_Float16*)(b+128);
            // Use index to ensure unique weights → unique model hash
            for (int j = 0; j < CH*CH; j++) w[j] = (_Float16)(0.001 * (i * CH * CH + j));
            NSData *wdata = [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];

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
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(cl_D,
                @selector(modelWithMILText:weights:optionsPlist:),
                md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
            if (!desc) { failAt = i; failReason = @"desc=nil"; break; }

            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(cl_M, @selector(inMemoryModelWithDescriptor:), desc);
            if (!mdl) { failAt = i; failReason = @"model=nil"; break; }

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
                failReason = [NSString stringWithFormat:@"compile fail: %@", e.localizedDescription ?: @"(no error)"];
                [fm removeItemAtPath:td error:nil];
                break;
            }

            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 9, @{}, &e);
            if (!ok) {
                failAt = i;
                failReason = [NSString stringWithFormat:@"load fail: %@", e.localizedDescription ?: @"(no error)"];
                [fm removeItemAtPath:td error:nil];
                break;
            }

            [models addObject:mdl];
            [tmpDirs addObject:td];

            if ((i+1) % 50 == 0) {
                double elapsed = cl_ms(mach_absolute_time() - totalStart);
                [out appendFormat:@"  %4d models OK  |  mem=%.0f MB  |  %.0f ms total\n",
                    i+1, app_memory_mb(), elapsed];
            }
        }

        double totalMs = cl_ms(mach_absolute_time() - totalStart);

        if (failAt < 0) {
            [out appendFormat:@"\n  RESULT: ALL %d models compiled+loaded successfully!\n", maxCompile];
        } else {
            [out appendFormat:@"\n  RESULT: FAILED at model %d — %@\n", failAt, failReason];
        }
        [out appendFormat:@"  Total time: %.1f s\n", totalMs / 1000.0];
        [out appendFormat:@"  Final memory: %.1f MB\n", app_memory_mb()];
        [out appendFormat:@"  Per-model: %.1f ms compile+load, %.1f KB memory\n",
            totalMs / models.count,
            (app_memory_mb() * 1024.0) / models.count];

        // Test reclaim: unload 50, try to compile+load 50 more
        if (models.count >= 50 && failAt > 0) {
            [out appendString:@"\n  --- Reclaim Test: unload 50, compile 50 new ---\n"];
            NSError *e = nil;
            for (int i = 0; i < 50; i++) {
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                    models[i], @selector(unloadWithQoS:error:), 9, &e);
                [[NSFileManager defaultManager] removeItemAtPath:tmpDirs[i] error:nil];
            }
            [models removeObjectsInRange:NSMakeRange(0, 50)];
            [tmpDirs removeObjectsInRange:NSMakeRange(0, 50)];
            [out appendFormat:@"  Unloaded 50 → %d remaining, mem=%.0f MB\n",
                (int)models.count, app_memory_mb()];

            int reclaimOk = 0;
            for (int i = 0; i < 50; i++) {
                int idx = failAt + i;
                int ws = CH * CH * 2, tot = 128 + ws;
                uint8_t *b = (uint8_t*)calloc(tot, 1);
                if (!b) break;
                b[0]=1; b[4]=2;
                b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
                *(uint32_t*)(b+72) = ws; *(uint32_t*)(b+80) = 128;
                _Float16 *w = (_Float16*)(b+128);
                for (int j = 0; j < CH*CH; j++) w[j] = (_Float16)(0.002 * (idx * CH * CH + j));
                NSData *wdata = [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];

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
                id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(cl_D,
                    @selector(modelWithMILText:weights:optionsPlist:),
                    md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
                if (!desc) break;
                id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(cl_M, @selector(inMemoryModelWithDescriptor:), desc);
                if (!mdl) break;

                id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
                NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
                NSFileManager *fm = [NSFileManager defaultManager];
                [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                    withIntermediateDirectories:YES attributes:nil error:nil];
                [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
                [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

                e = nil;
                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl, @selector(compileWithQoS:options:error:), 9, @{}, &e);
                if (!ok) { [out appendFormat:@"  Reclaim compile fail at +%d\n", i]; break; }
                ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl, @selector(loadWithQoS:options:error:), 9, @{}, &e);
                if (!ok) { [out appendFormat:@"  Reclaim load fail at +%d: %@\n", i, e.localizedDescription]; break; }

                [models addObject:mdl];
                [tmpDirs addObject:td];
                reclaimOk++;
            }
            [out appendFormat:@"  Reclaimed %d/50 slots: %@\n",
                reclaimOk, reclaimOk == 50 ? @"FULL RECLAIM" : @"PARTIAL"];
        }

        // Cleanup
        for (int i = 0; i < (int)models.count; i++) {
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                models[i], @selector(unloadWithQoS:error:), 9, &e);
            [[NSFileManager defaultManager] removeItemAtPath:tmpDirs[i] error:nil];
        }

        [out appendFormat:@"  After cleanup: %.1f MB\n", app_memory_mb()];

        return out;
    }
}
