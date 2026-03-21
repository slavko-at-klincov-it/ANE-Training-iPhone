// ANERMSNorm.m — RMSNorm forward + backward on ANE
// Ported from macOS ANE-Training: stories_mil.h (fwd) + ane_rmsnorm_bwd.h (bwd)
//
// Forward: xn = x * rrms * w,  where rrms = 1/sqrt(mean(x²) + eps)
// Backward: dx = rrms * (w*dy - x * sum(dy*w*x) * invd * rrms²)
//
// Test: compare ANE output to CPU reference implementation
#import "ANETrainingConfig.h"

// ================================================================
// MIL Generators
// ================================================================

// RMSNorm forward MIL
// Input:  x [1, DIM, 1, SEQ]
// Baked:  rms_w [1, DIM, 1, 1]
// Output: xn [1, DIM, 1, SEQ]  (normalized + scaled)
static NSString *gen_rmsnorm_fwd(void) {
    float invd = 1.0f / (float)DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];
    // sq = x * x
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", DIM, SEQ];
    // ss = sum(sq, axis=1) → [1,1,1,SEQ]
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", SEQ];
    // ss2 = ss * (1/DIM)
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", SEQ];
    // ss3 = ss2 + eps
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", SEQ];
    // rrms = pow(ss3, -0.5) = 1/sqrt(mean(x²)+eps)
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", SEQ];
    // xr = x * rrms (broadcast over channels)
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", DIM, SEQ];
    // xn = xr * w (broadcast weight [DIM,1] over seq)
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), "
        "val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_w.bin\"), offset=uint64(64)))];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=string(\"out\")];\n", DIM, SEQ];
    [m appendString:@"    } -> (xn);\n}\n"];
    return m;
}

// RMSNorm backward MIL
// Input:  concat(dy, x) as [1, 2*DIM, 1, SEQ]
// Baked:  rms_w [1, DIM, 1, 1]
// Output: dx [1, DIM, 1, SEQ]
static NSString *gen_rmsnorm_bwd(void) {
    float invd = 1.0f / (float)DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    // Input: concat of dy and x along channel dim
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> inp) {\n", 2*DIM, SEQ];

    // Slice dy [0:DIM] and x [DIM:2*DIM]
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dy = slice_by_size(x=inp,begin=b0,size=sz)[name=string(\"sdy\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x = slice_by_size(x=inp,begin=b1,size=sz)[name=string(\"sx\")];\n", DIM, SEQ];

    // rrms = 1/sqrt(mean(x²) + eps)
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", SEQ];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", SEQ];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", SEQ];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", SEQ];

    // Load weights w [1, DIM, 1, 1]
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> w = const()[name=string(\"w\"), "
        "val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_w.bin\"), offset=uint64(64)))];\n", DIM, DIM];

    // dyw = dy * w
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dyw = mul(x=dy,y=w)[name=string(\"dyw\")];\n", DIM, SEQ];
    // dywx = dyw * x
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dywx = mul(x=dyw,y=x)[name=string(\"dywx\")];\n", DIM, SEQ];
    // dot_sum = sum(dywx, axis=1) → [1,1,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> dot_sum = reduce_sum(x=dywx,axes=rax,keep_dims=kd)[name=string(\"ds\")];\n", SEQ];
    // dot_sc = dot_sum * invd
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> dot_sc = mul(x=dot_sum,y=invd)[name=string(\"dsc\")];\n", SEQ];
    // rrms2 = rrms * rrms
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms2 = mul(x=rrms,y=rrms)[name=string(\"rr2\")];\n", SEQ];
    // coeff = dot_sc * rrms2
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> coeff = mul(x=dot_sc,y=rrms2)[name=string(\"cof\")];\n", SEQ];

    // dx = (dyw - x * coeff) * rrms
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xc = mul(x=x,y=coeff)[name=string(\"xc\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> diff = sub(x=dyw,y=xc)[name=string(\"dif\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = mul(x=diff,y=rrms)[name=string(\"out\")];\n", DIM, SEQ];

    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ================================================================
// CPU Reference Implementation
// ================================================================

static void cpu_rmsnorm_fwd(float *out, const float *x, const float *w, int dim, int seq) {
    for (int s = 0; s < seq; s++) {
        float ss = 0;
        for (int d = 0; d < dim; d++) ss += x[d*seq+s] * x[d*seq+s];
        float rrms = 1.0f / sqrtf(ss / dim + 1e-5f);
        for (int d = 0; d < dim; d++)
            out[d*seq+s] = x[d*seq+s] * rrms * w[d];
    }
}

static void cpu_rmsnorm_bwd(float *dx, const float *dy, const float *x, const float *w, int dim, int seq) {
    for (int s = 0; s < seq; s++) {
        float ss = 0;
        for (int d = 0; d < dim; d++) ss += x[d*seq+s] * x[d*seq+s];
        float rrms = 1.0f / sqrtf(ss / dim + 1e-5f);
        // dot = sum(dy * w * x) / dim * rrms²
        float dot = 0;
        for (int d = 0; d < dim; d++) dot += dy[d*seq+s] * w[d] * x[d*seq+s];
        dot = dot / dim * rrms * rrms;
        for (int d = 0; d < dim; d++)
            dx[d*seq+s] = (dy[d*seq+s] * w[d] - x[d*seq+s] * dot) * rrms;
    }
}

// ================================================================
// Test
// ================================================================

NSString *ane_rmsnorm_test(void) {
    @autoreleasepool {
        ane_init();
        NSMutableString *out = [NSMutableString string];
        if (!g_D || !g_I) { [out appendString:@"  ANE classes missing\n"]; return out; }

        [out appendString:@"  --- RMSNorm Forward + Backward Test ---\n"];
        [out appendFormat:@"  Config: DIM=%d SEQ=%d\n", DIM, SEQ];

        // Generate random input and weights
        srand48(42);
        int n = DIM * SEQ;
        float *x   = (float*)malloc(n * sizeof(float));
        float *w    = (float*)malloc(DIM * sizeof(float));
        float *cpu_fwd = (float*)malloc(n * sizeof(float));
        float *ane_fwd = (float*)malloc(n * sizeof(float));
        float *dy   = (float*)malloc(n * sizeof(float));
        float *cpu_bwd = (float*)malloc(n * sizeof(float));
        float *ane_bwd = (float*)malloc(n * sizeof(float));

        for (int i = 0; i < n; i++) x[i] = (float)(2*drand48()-1);
        for (int i = 0; i < DIM; i++) w[i] = (float)(0.5 + drand48());
        for (int i = 0; i < n; i++) dy[i] = (float)(2*drand48()-1) * 0.1f;

        // ============================================================
        // Forward Test
        // ============================================================
        [out appendString:@"\n  == Forward ==\n"];

        // CPU reference
        uint64_t t0 = mach_absolute_time();
        cpu_rmsnorm_fwd(cpu_fwd, x, w, DIM, SEQ);
        double cpuMs = tb_ms(mach_absolute_time() - t0);
        [out appendFormat:@"  CPU: %.3f ms\n", cpuMs];

        // ANE: compile kernel with baked weights
        NSString *fwdMil = gen_rmsnorm_fwd();
        NSData *wBlob = build_blob(w, 1, DIM);
        NSDictionary *wDict = @{@"@model_path/weights/rms_w.bin": @{@"offset":@0, @"data":wBlob}};
        Kern *fwdK = compile_kern(fwdMil, wDict, DIM*SEQ*2, DIM*SEQ*2);
        if (!fwdK) {
            [out appendString:@"  Forward compile FAILED\n"];
            free(x); free(w); free(cpu_fwd); free(ane_fwd);
            free(dy); free(cpu_bwd); free(ane_bwd);
            return out;
        }

        // Write input, eval, read output
        io_write_fp16(fwdK->ioIn, x, DIM, SEQ);
        t0 = mach_absolute_time();
        if (!ane_eval(fwdK)) {
            [out appendString:@"  Forward eval FAILED\n"]; free_kern(fwdK);
            free(x); free(w); free(cpu_fwd); free(ane_fwd);
            free(dy); free(cpu_bwd); free(ane_bwd);
            return out;
        }
        double aneMs = tb_ms(mach_absolute_time() - t0);
        io_read_fp16(fwdK->ioOut, ane_fwd, 0, DIM, SEQ);

        // Compare forward
        float fwdMaxErr = 0, fwdSumErr = 0;
        for (int i = 0; i < n; i++) {
            float err = fabsf(ane_fwd[i] - cpu_fwd[i]);
            if (err > fwdMaxErr) fwdMaxErr = err;
            fwdSumErr += err;
        }
        [out appendFormat:@"  ANE: %.3f ms  (%.1fx vs CPU)\n", aneMs, cpuMs/aneMs];
        [out appendFormat:@"  Max error: %.6f  Avg error: %.6f\n", fwdMaxErr, fwdSumErr/n];
        [out appendFormat:@"  Forward: %@\n", fwdMaxErr < 0.05f ? @"PASS" : @"FAIL"];

        // Benchmark: 100 evals
        t0 = mach_absolute_time();
        for (int i = 0; i < 100; i++) ane_eval(fwdK);
        double fwdBenchMs = tb_ms(mach_absolute_time() - t0);
        [out appendFormat:@"  Throughput: %.3f ms/eval (100 runs)\n", fwdBenchMs/100];

        free_kern(fwdK);

        // ============================================================
        // Backward Test
        // ============================================================
        [out appendString:@"\n  == Backward ==\n"];

        // CPU reference
        t0 = mach_absolute_time();
        cpu_rmsnorm_bwd(cpu_bwd, dy, x, w, DIM, SEQ);
        cpuMs = tb_ms(mach_absolute_time() - t0);
        [out appendFormat:@"  CPU: %.3f ms\n", cpuMs];

        // ANE: compile backward kernel
        NSString *bwdMil = gen_rmsnorm_bwd();
        Kern *bwdK = compile_kern(bwdMil, wDict, 2*DIM*SEQ*2, DIM*SEQ*2);
        if (!bwdK) { [out appendString:@"  Backward compile FAILED\n"]; }
        else {
            // Write input: concat(dy, x) along channel dim
            _Float16 *inp = (_Float16*)malloc(2*DIM*SEQ*sizeof(_Float16));
            cvt_f32_f16(inp, dy, DIM*SEQ);
            cvt_f32_f16(inp + DIM*SEQ, x, DIM*SEQ);
            IOSurfaceLock(bwdK->ioIn, 0, NULL);
            memcpy(IOSurfaceGetBaseAddress(bwdK->ioIn), inp, 2*DIM*SEQ*sizeof(_Float16));
            IOSurfaceUnlock(bwdK->ioIn, 0, NULL);
            free(inp);

            t0 = mach_absolute_time();
            if (!ane_eval(bwdK)) { [out appendString:@"  Backward eval FAILED\n"]; }
            else {
                aneMs = tb_ms(mach_absolute_time() - t0);
                io_read_fp16(bwdK->ioOut, ane_bwd, 0, DIM, SEQ);

                // Compare backward
                float bwdMaxErr = 0, bwdSumErr = 0;
                for (int i = 0; i < n; i++) {
                    float err = fabsf(ane_bwd[i] - cpu_bwd[i]);
                    if (err > bwdMaxErr) bwdMaxErr = err;
                    bwdSumErr += err;
                }
                [out appendFormat:@"  ANE: %.3f ms  (%.1fx vs CPU)\n", aneMs, cpuMs/aneMs];
                [out appendFormat:@"  Max error: %.6f  Avg error: %.6f\n", bwdMaxErr, bwdSumErr/n];
                [out appendFormat:@"  Backward: %@\n", bwdMaxErr < 0.1f ? @"PASS" : @"FAIL"];

                // Benchmark
                t0 = mach_absolute_time();
                for (int i = 0; i < 100; i++) ane_eval(bwdK);
                double bwdBenchMs = tb_ms(mach_absolute_time() - t0);
                [out appendFormat:@"  Throughput: %.3f ms/eval (100 runs)\n", bwdBenchMs/100];
            }
            free_kern(bwdK);
        }

        free(x); free(w); free(cpu_fwd); free(ane_fwd);
        free(dy); free(cpu_bwd); free(ane_bwd);

        return out;
    }
}
