// ANETrainStep.m — Prove training works on iPhone ANE
// Test: train a linear layer to learn y = W_true @ x
// If loss → 0, the full compile→eval→gradient→update cycle works.
//
// Steps:
// 1. Generate W_true, random x, compute target = W_true @ x
// 2. Init W_learn randomly
// 3. For each step:
//    a) Compile W_learn as baked conv kernel
//    b) Forward: y = W_learn @ x (on ANE)
//    c) Loss: MSE(y, target)
//    d) Backward: dy = (y - target), dW = dy @ x^T (on CPU)
//    e) Update: W_learn -= lr * dW
// 4. Report loss per step — should decrease toward 0
#import "ANETrainingConfig.h"

NSString *ane_train_step_test(void) {
    @autoreleasepool {
        ane_init();
        NSMutableString *out = [NSMutableString string];
        if (!g_D || !g_I) { [out appendString:@"  ANE classes missing\n"]; return out; }

        [out appendString:@"  --- Training Step Test ---\n"];
        [out appendString:@"  Task: learn y = W_true @ x via SGD on ANE\n"];

        // Small dimensions for fast iteration
        int IN_CH = 128, OUT_CH = 64, SQ = 32;
        int nsteps = 10;
        float lr = 0.01f;

        [out appendFormat:@"  Config: in=%d out=%d seq=%d steps=%d lr=%.3f\n", IN_CH, OUT_CH, SQ, nsteps, lr];

        srand48(42);
        float scale = 1.0f / sqrtf((float)IN_CH);

        // Ground truth weight matrix
        float *W_true = (float*)malloc(OUT_CH * IN_CH * sizeof(float));
        for (int i = 0; i < OUT_CH * IN_CH; i++) W_true[i] = (float)(2*drand48()-1) * scale;

        // Fixed input
        float *x = (float*)malloc(IN_CH * SQ * sizeof(float));
        for (int i = 0; i < IN_CH * SQ; i++) x[i] = (float)(2*drand48()-1);

        // Target: y_true = W_true @ x (channel-first)
        float *target = (float*)malloc(OUT_CH * SQ * sizeof(float));
        for (int o = 0; o < OUT_CH; o++)
            for (int s = 0; s < SQ; s++) {
                float sum = 0;
                for (int i = 0; i < IN_CH; i++) sum += W_true[o*IN_CH+i] * x[i*SQ+s];
                target[o*SQ+s] = sum;
            }

        // Learnable weights (random init, different from W_true)
        float *W_learn = (float*)malloc(OUT_CH * IN_CH * sizeof(float));
        for (int i = 0; i < OUT_CH * IN_CH; i++) W_learn[i] = (float)(2*drand48()-1) * scale;

        float *y_ane = (float*)malloc(OUT_CH * SQ * sizeof(float));
        float *dW    = (float*)malloc(OUT_CH * IN_CH * sizeof(float));

        [out appendString:@"\n  Step   Loss        Compile   Eval\n"];

        uint64_t t_total = mach_absolute_time();

        for (int step = 0; step < nsteps; step++) {
            // 1) Compile new kernel with current weights
            NSString *mil = [NSString stringWithFormat:
                @"%@    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                @CONV_CONST
                "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
                "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n"
                "        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)"
                "[name=string(\"out\")];\n"
                "    } -> (y);\n}\n",
                MIL_HDR, IN_CH, SQ, OUT_CH, IN_CH, OUT_CH, IN_CH, OUT_CH, SQ];

            NSData *wBlob = build_blob(W_learn, OUT_CH, IN_CH);
            NSDictionary *wDict = @{@"@model_path/weights/w.bin": @{@"offset":@0, @"data":wBlob}};

            uint64_t tc = mach_absolute_time();
            Kern *k = compile_kern(mil, wDict, IN_CH*SQ*2, OUT_CH*SQ*2);
            double compMs = tb_ms(mach_absolute_time() - tc);

            if (!k) {
                [out appendFormat:@"  %4d   COMPILE FAIL\n", step];
                break;
            }

            // 2) Forward: y = W_learn @ x (on ANE)
            io_write_fp16(k->ioIn, x, IN_CH, SQ);
            uint64_t te = mach_absolute_time();
            ane_eval(k);
            double evalMs = tb_ms(mach_absolute_time() - te);
            io_read_fp16(k->ioOut, y_ane, 0, OUT_CH, SQ);

            free_kern(k);

            // 3) Loss: MSE = mean((y - target)^2)
            float loss = 0;
            int n = OUT_CH * SQ;
            for (int i = 0; i < n; i++) {
                float diff = y_ane[i] - target[i];
                loss += diff * diff;
            }
            loss /= n;

            [out appendFormat:@"  %4d   %.6f   %.1fms   %.3fms\n", step, loss, compMs, evalMs];

            // 4) Backward: dy = 2/n * (y - target), dW = dy @ x^T (on CPU)
            // dW[o, i] = sum_s(dy[o,s] * x[i,s])
            float grad_scale = 2.0f / n;
            memset(dW, 0, OUT_CH * IN_CH * sizeof(float));
            for (int o = 0; o < OUT_CH; o++)
                for (int i = 0; i < IN_CH; i++) {
                    float sum = 0;
                    for (int s = 0; s < SQ; s++)
                        sum += (y_ane[o*SQ+s] - target[o*SQ+s]) * x[i*SQ+s];
                    dW[o*IN_CH+i] = grad_scale * sum;
                }

            // 5) Update: W -= lr * dW
            for (int i = 0; i < OUT_CH * IN_CH; i++)
                W_learn[i] -= lr * dW[i];
        }

        double totalMs = tb_ms(mach_absolute_time() - t_total);
        [out appendFormat:@"\n  Total: %.1f ms (%.1f ms/step)\n", totalMs, totalMs/nsteps];

        // Final verification: how close is W_learn to W_true?
        float wMaxErr = 0, wSumErr = 0;
        for (int i = 0; i < OUT_CH * IN_CH; i++) {
            float err = fabsf(W_learn[i] - W_true[i]);
            if (err > wMaxErr) wMaxErr = err;
            wSumErr += err;
        }
        [out appendFormat:@"  W_learn vs W_true: max_err=%.6f avg_err=%.6f\n", wMaxErr, wSumErr/(OUT_CH*IN_CH)];

        free(W_true); free(x); free(target); free(W_learn); free(y_ane); free(dW);
        return out;
    }
}
