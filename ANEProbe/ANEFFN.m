// ANEFFN.m — SwiGLU FFN on ANE
// Ported from macOS ANE-Training: stories_mil.h gen_ffn_fwd_taps
//
// SwiGLU: gate = silu(W1@x) * (W3@x), y = W2@gate
// silu(x) = x * sigmoid(x)
//
// Input: xn [1, DIM, 1, SEQ] (post-RMSNorm)
// Baked: W1 [HIDDEN,DIM], W3 [HIDDEN,DIM], W2 [DIM,HIDDEN]
// Output: y [1, DIM, 1, SEQ]
#import "ANETrainingConfig.h"

// ================================================================
// MIL Generator
// ================================================================

static NSString *gen_ffn_fwd(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> xn) {\n", DIM, SEQ];
    [m appendString:@CONV_CONST];

    // Baked weights
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n", HIDDEN,DIM,HIDDEN,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3 = const()[name=string(\"W3\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];\n", HIDDEN,DIM,HIDDEN,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2 = const()[name=string(\"W2\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n", DIM,HIDDEN,DIM,HIDDEN];

    // h1 = W1 @ xn, h3 = W3 @ xn  [1, HIDDEN, 1, SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=xn)[name=string(\"c1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=xn)[name=string(\"c3\")];\n", HIDDEN, SEQ];

    // SwiGLU: gate = silu(h1) * h3 = h1 * sigmoid(h1) * h3
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1,y=sig)[name=string(\"si\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];\n", HIDDEN, SEQ];

    // y = W2 @ gate  [1, DIM, 1, SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=string(\"out\")];\n", DIM, SEQ];

    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// ================================================================
// CPU Reference
// ================================================================

static float silu_f(float x) { return x / (1.0f + expf(-x)); }

static void cpu_ffn_fwd(float *y, const float *xn,
                         const float *W1, const float *W3, const float *W2,
                         int dim, int hidden, int seq) {
    float *h1   = (float*)malloc(hidden * seq * sizeof(float));
    float *h3   = (float*)malloc(hidden * seq * sizeof(float));
    float *gate = (float*)malloc(hidden * seq * sizeof(float));

    // h1 = W1 @ xn (channel-first)
    for (int o = 0; o < hidden; o++)
        for (int s = 0; s < seq; s++) {
            float sum = 0;
            for (int i = 0; i < dim; i++) sum += W1[o*dim+i] * xn[i*seq+s];
            h1[o*seq+s] = sum;
        }

    // h3 = W3 @ xn
    for (int o = 0; o < hidden; o++)
        for (int s = 0; s < seq; s++) {
            float sum = 0;
            for (int i = 0; i < dim; i++) sum += W3[o*dim+i] * xn[i*seq+s];
            h3[o*seq+s] = sum;
        }

    // gate = silu(h1) * h3
    for (int i = 0; i < hidden * seq; i++)
        gate[i] = silu_f(h1[i]) * h3[i];

    // y = W2 @ gate
    for (int o = 0; o < dim; o++)
        for (int s = 0; s < seq; s++) {
            float sum = 0;
            for (int i = 0; i < hidden; i++) sum += W2[o*hidden+i] * gate[i*seq+s];
            y[o*seq+s] = sum;
        }

    free(h1); free(h3); free(gate);
}

// ================================================================
// Test
// ================================================================

NSString *ane_ffn_test(void) {
    @autoreleasepool {
        ane_init();
        NSMutableString *out = [NSMutableString string];
        if (!g_D || !g_I) { [out appendString:@"  ANE classes missing\n"]; return out; }

        [out appendString:@"  --- FFN (SwiGLU) Forward Test ---\n"];
        [out appendFormat:@"  Config: DIM=%d HIDDEN=%d SEQ=%d\n", DIM, HIDDEN, SEQ];

        // Random data
        srand48(456);
        int n = DIM * SEQ;
        float scaleD = 1.0f / sqrtf((float)DIM);
        float scaleH = 1.0f / sqrtf((float)HIDDEN);
        float *xn = (float*)malloc(n * sizeof(float));
        float *W1 = (float*)malloc(HIDDEN*DIM * sizeof(float));
        float *W3 = (float*)malloc(HIDDEN*DIM * sizeof(float));
        float *W2 = (float*)malloc(DIM*HIDDEN * sizeof(float));
        float *cpu_y = (float*)malloc(n * sizeof(float));
        float *ane_y = (float*)malloc(n * sizeof(float));

        for (int i = 0; i < n; i++) xn[i] = (float)(2*drand48()-1);
        for (int i = 0; i < HIDDEN*DIM; i++) W1[i] = (float)(2*drand48()-1) * scaleD;
        for (int i = 0; i < HIDDEN*DIM; i++) W3[i] = (float)(2*drand48()-1) * scaleD;
        for (int i = 0; i < DIM*HIDDEN; i++) W2[i] = (float)(2*drand48()-1) * scaleH;

        // CPU reference
        [out appendString:@"  Computing CPU reference...\n"];
        uint64_t t0 = mach_absolute_time();
        cpu_ffn_fwd(cpu_y, xn, W1, W3, W2, DIM, HIDDEN, SEQ);
        double cpuMs = tb_ms(mach_absolute_time() - t0);
        [out appendFormat:@"  CPU: %.1f ms\n", cpuMs];

        // ANE
        [out appendString:@"  Compiling ANE kernel...\n"];
        NSString *mil = gen_ffn_fwd();
        NSData *w1Blob = build_blob(W1, HIDDEN, DIM);
        NSData *w3Blob = build_blob(W3, HIDDEN, DIM);
        NSData *w2Blob = build_blob(W2, DIM, HIDDEN);

        NSDictionary *weights = @{
            @"@model_path/weights/w1.bin": @{@"offset":@0, @"data":w1Blob},
            @"@model_path/weights/w3.bin": @{@"offset":@0, @"data":w3Blob},
            @"@model_path/weights/w2.bin": @{@"offset":@0, @"data":w2Blob},
        };

        // Weight sizes: W1=3MB, W3=3MB, W2=3MB ≈ 9MB total (within SRAM sweet spot)
        t0 = mach_absolute_time();
        Kern *k = compile_kern(mil, weights, DIM*SEQ*2, DIM*SEQ*2);
        double compileMs = tb_ms(mach_absolute_time() - t0);

        if (!k) {
            [out appendString:@"  FFN compile FAILED\n"];
            free(xn); free(W1); free(W3); free(W2); free(cpu_y); free(ane_y);
            return out;
        }
        [out appendFormat:@"  Compiled in %.1f ms\n", compileMs];

        // Eval
        io_write_fp16(k->ioIn, xn, DIM, SEQ);
        t0 = mach_absolute_time();
        BOOL ok = ane_eval(k);
        double aneMs = tb_ms(mach_absolute_time() - t0);

        if (!ok) {
            [out appendString:@"  FFN eval FAILED\n"];
            free_kern(k);
            free(xn); free(W1); free(W3); free(W2); free(cpu_y); free(ane_y);
            return out;
        }
        io_read_fp16(k->ioOut, ane_y, 0, DIM, SEQ);

        // Compare
        float maxErr = 0, sumErr = 0;
        int errCount = 0;
        for (int i = 0; i < n; i++) {
            float err = fabsf(ane_y[i] - cpu_y[i]);
            if (err > maxErr) maxErr = err;
            sumErr += err;
            if (err > 0.1f) errCount++;
        }
        [out appendFormat:@"  ANE: %.3f ms  (%.0fx vs CPU)\n", aneMs, cpuMs/aneMs];
        [out appendFormat:@"  Max error: %.6f  Avg error: %.6f  Outliers(>0.1): %d/%d\n",
            maxErr, sumErr/n, errCount, n];
        [out appendFormat:@"  FFN: %@\n", maxErr < 1.0f ? @"PASS" : @"FAIL"];

        // Benchmark
        t0 = mach_absolute_time();
        for (int i = 0; i < 50; i++) ane_eval(k);
        double benchMs = tb_ms(mach_absolute_time() - t0);
        [out appendFormat:@"  Throughput: %.3f ms/eval (50 runs)\n", benchMs/50];

        free_kern(k);
        free(xn); free(W1); free(W3); free(W2); free(cpu_y); free(ane_y);

        return out;
    }
}
