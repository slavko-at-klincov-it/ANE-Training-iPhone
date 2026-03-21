// ANELinear.m — Linear layer (y = Wx) on ANE via 1x1 conv
// Ported from macOS ANE-Training: stories_mil.h (conv-based matmul)
//
// Forward:  y = W @ x       (conv with baked weights)
// Backward: dx = W^T @ dy   (conv with transposed baked weights)
//           dW = dy @ x^T   (done on CPU — accumulate across seq positions)
//
// Layout: channel-first [1, channels, 1, seq] for ANE
//         Weight: [out_ch, in_ch, 1, 1] for conv
#import "ANETrainingConfig.h"

// ================================================================
// MIL Generators
// ================================================================

// Linear forward: y = W @ x
// Input:  x  [1, in_ch, 1, seq]
// Baked:  W  [out_ch, in_ch, 1, 1]
// Output: y  [1, out_ch, 1, seq]
static NSString *gen_linear_fwd(int in_ch, int out_ch, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", in_ch, seq];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n",
        out_ch, in_ch, out_ch, in_ch];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)"
        "[name=string(\"out\")];\n", out_ch, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// Linear backward: dx = W^T @ dy
// Same structure as forward but with transposed weights
// Input:  dy [1, out_ch, 1, seq]
// Baked:  W^T [in_ch, out_ch, 1, 1]
// Output: dx [1, in_ch, 1, seq]
static NSString *gen_linear_bwd(int in_ch, int out_ch, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> dy) {\n", out_ch, seq];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wt = const()[name=string(\"Wt\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wt.bin\"), offset=uint64(64)))];\n",
        in_ch, out_ch, in_ch, out_ch];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wt,x=dy)"
        "[name=string(\"out\")];\n", in_ch, seq];
    [m appendString:@"    } -> (dx);\n}\n"];
    return m;
}

// ================================================================
// CPU Reference
// ================================================================

// y = W @ x, channel-first layout: W[out,in], x[in,seq] → y[out,seq]
static void cpu_linear_fwd(float *y, const float *W, const float *x, int in_ch, int out_ch, int seq) {
    for (int o = 0; o < out_ch; o++) {
        for (int s = 0; s < seq; s++) {
            float sum = 0;
            for (int i = 0; i < in_ch; i++)
                sum += W[o*in_ch + i] * x[i*seq + s];
            y[o*seq + s] = sum;
        }
    }
}

// dx = W^T @ dy, channel-first: W[out,in], dy[out,seq] → dx[in,seq]
static void cpu_linear_bwd_x(float *dx, const float *W, const float *dy, int in_ch, int out_ch, int seq) {
    memset(dx, 0, in_ch * seq * sizeof(float));
    for (int i = 0; i < in_ch; i++) {
        for (int s = 0; s < seq; s++) {
            float sum = 0;
            for (int o = 0; o < out_ch; o++)
                sum += W[o*in_ch + i] * dy[o*seq + s];
            dx[i*seq + s] = sum;
        }
    }
}

// dW = dy @ x^T, channel-first: dy[out,seq], x[in,seq] → dW[out,in]
static void cpu_linear_bwd_w(float *dW, const float *dy, const float *x, int in_ch, int out_ch, int seq) {
    for (int o = 0; o < out_ch; o++) {
        for (int i = 0; i < in_ch; i++) {
            float sum = 0;
            for (int s = 0; s < seq; s++)
                sum += dy[o*seq + s] * x[i*seq + s];
            dW[o*in_ch + i] = sum;
        }
    }
}

// ================================================================
// Test
// ================================================================

NSString *ane_linear_test(void) {
    @autoreleasepool {
        ane_init();
        NSMutableString *out = [NSMutableString string];
        if (!g_D || !g_I) { [out appendString:@"  ANE classes missing\n"]; return out; }

        [out appendString:@"  --- Linear Layer (1x1 Conv) Test ---\n"];

        // Test multiple configs relevant for training
        struct { int in_ch; int out_ch; int seq; const char *name; } configs[] = {
            {DIM, DIM, SEQ, "Wq/Wk/Wv/Wo (768→768)"},
            {DIM, HIDDEN, SEQ, "W1/W3 (768→2048)"},
            {HIDDEN, DIM, SEQ, "W2 (2048→768)"},
        };
        int nconfigs = sizeof(configs) / sizeof(configs[0]);

        for (int c = 0; c < nconfigs; c++) {
            int ic = configs[c].in_ch, oc = configs[c].out_ch, sq = configs[c].seq;
            [out appendFormat:@"\n  == %s ==\n", configs[c].name];

            // Random data
            srand48(42 + c);
            int wn = oc * ic;
            float *W = (float*)malloc(wn * sizeof(float));
            float *x = (float*)malloc(ic * sq * sizeof(float));
            float *cpu_y = (float*)malloc(oc * sq * sizeof(float));
            float *ane_y = (float*)malloc(oc * sq * sizeof(float));
            float *dy = (float*)malloc(oc * sq * sizeof(float));
            float *cpu_dx = (float*)malloc(ic * sq * sizeof(float));
            float *ane_dx = (float*)malloc(ic * sq * sizeof(float));

            // Xavier-ish init
            float scale = 1.0f / sqrtf((float)ic);
            for (int i = 0; i < wn; i++) W[i] = (float)(2*drand48()-1) * scale;
            for (int i = 0; i < ic*sq; i++) x[i] = (float)(2*drand48()-1);
            for (int i = 0; i < oc*sq; i++) dy[i] = (float)(2*drand48()-1) * 0.01f;

            // --- Forward ---
            uint64_t t0 = mach_absolute_time();
            cpu_linear_fwd(cpu_y, W, x, ic, oc, sq);
            double cpuMs = tb_ms(mach_absolute_time() - t0);

            NSString *fwdMil = gen_linear_fwd(ic, oc, sq);
            NSData *wBlob = build_blob(W, oc, ic);
            NSDictionary *wDict = @{@"@model_path/weights/w.bin": @{@"offset":@0, @"data":wBlob}};
            Kern *fwdK = compile_kern(fwdMil, wDict, ic*sq*2, oc*sq*2);

            if (!fwdK) {
                [out appendString:@"  Forward compile FAILED\n"];
                free(W); free(x); free(cpu_y); free(ane_y); free(dy); free(cpu_dx); free(ane_dx);
                continue;
            }

            io_write_fp16(fwdK->ioIn, x, ic, sq);
            t0 = mach_absolute_time();
            BOOL ok = ane_eval(fwdK);
            double aneMs = tb_ms(mach_absolute_time() - t0);
            if (!ok) {
                [out appendString:@"  Forward eval FAILED\n"];
                free_kern(fwdK);
                free(W); free(x); free(cpu_y); free(ane_y); free(dy); free(cpu_dx); free(ane_dx);
                continue;
            }
            io_read_fp16(fwdK->ioOut, ane_y, 0, oc, sq);

            float fMaxErr = 0, fSumErr = 0;
            for (int i = 0; i < oc*sq; i++) {
                float err = fabsf(ane_y[i] - cpu_y[i]);
                if (err > fMaxErr) fMaxErr = err;
                fSumErr += err;
            }
            [out appendFormat:@"  Fwd CPU: %.2f ms  ANE: %.3f ms  (%.1fx)\n", cpuMs, aneMs, cpuMs/aneMs];
            [out appendFormat:@"  Fwd max_err=%.6f avg_err=%.6f  %@\n",
                fMaxErr, fSumErr/(oc*sq), fMaxErr < 0.5f ? @"PASS" : @"FAIL"];

            // Benchmark forward
            t0 = mach_absolute_time();
            for (int i = 0; i < 50; i++) ane_eval(fwdK);
            double fwdBench = tb_ms(mach_absolute_time() - t0);
            [out appendFormat:@"  Fwd throughput: %.3f ms/eval (50 runs)\n", fwdBench/50];

            free_kern(fwdK);

            // --- Backward (dx = W^T @ dy) ---
            t0 = mach_absolute_time();
            cpu_linear_bwd_x(cpu_dx, W, dy, ic, oc, sq);
            cpuMs = tb_ms(mach_absolute_time() - t0);

            NSString *bwdMil = gen_linear_bwd(ic, oc, sq);
            NSData *wtBlob = build_blob_t(W, oc, ic); // W transposed
            NSDictionary *wtDict = @{@"@model_path/weights/wt.bin": @{@"offset":@0, @"data":wtBlob}};
            Kern *bwdK = compile_kern(bwdMil, wtDict, oc*sq*2, ic*sq*2);

            if (!bwdK) {
                [out appendString:@"  Backward compile FAILED\n"];
                free(W); free(x); free(cpu_y); free(ane_y); free(dy); free(cpu_dx); free(ane_dx);
                continue;
            }

            io_write_fp16(bwdK->ioIn, dy, oc, sq);
            t0 = mach_absolute_time();
            ok = ane_eval(bwdK);
            aneMs = tb_ms(mach_absolute_time() - t0);
            if (!ok) {
                [out appendString:@"  Backward eval FAILED\n"];
                free_kern(bwdK);
                free(W); free(x); free(cpu_y); free(ane_y); free(dy); free(cpu_dx); free(ane_dx);
                continue;
            }
            io_read_fp16(bwdK->ioOut, ane_dx, 0, ic, sq);

            float bMaxErr = 0, bSumErr = 0;
            for (int i = 0; i < ic*sq; i++) {
                float err = fabsf(ane_dx[i] - cpu_dx[i]);
                if (err > bMaxErr) bMaxErr = err;
                bSumErr += err;
            }
            [out appendFormat:@"  Bwd CPU: %.2f ms  ANE: %.3f ms  (%.1fx)\n", cpuMs, aneMs, cpuMs/aneMs];
            [out appendFormat:@"  Bwd max_err=%.6f avg_err=%.6f  %@\n",
                bMaxErr, bSumErr/(ic*sq), bMaxErr < 0.5f ? @"PASS" : @"FAIL"];

            // Benchmark backward
            t0 = mach_absolute_time();
            for (int i = 0; i < 50; i++) ane_eval(bwdK);
            double bwdBench = tb_ms(mach_absolute_time() - t0);
            [out appendFormat:@"  Bwd throughput: %.3f ms/eval (50 runs)\n", bwdBench/50];

            free_kern(bwdK);
            free(W); free(x); free(cpu_y); free(ane_y); free(dy); free(cpu_dx); free(ane_dx);
        }

        // Note about dW
        [out appendString:@"\n  Note: dW = dy @ x^T stays on CPU (outer product, accumulates across steps)\n"];

        return out;
    }
}
