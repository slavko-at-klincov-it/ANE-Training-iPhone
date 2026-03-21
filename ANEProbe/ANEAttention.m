// ANEAttention.m — Scaled Dot-Product Attention on ANE
// Ported from macOS ANE-Training: stories_mil.h gen_sdpa_fwd_taps
//
// Pipeline: xn → QKV convs → reshape → transpose → Q@K^T → scale → mask → softmax → @V → Wo → output
// Input: xn [1, DIM, 1, SEQ] (post-RMSNorm)
// Baked: Wq, Wk, Wv, Wo [DIM,DIM,1,1], causal mask [1,1,SEQ,SEQ]
// Output: o_out [1, DIM, 1, SEQ]
#import "ANETrainingConfig.h"

// ================================================================
// MIL Generator
// ================================================================

// Build causal mask: lower-triangular 0, upper-triangular -inf (large negative)
static NSData *build_causal_mask(int seq) {
    int n = seq * seq;
    float *mask = (float*)calloc(n, sizeof(float));
    for (int i = 0; i < seq; i++)
        for (int j = i+1; j < seq; j++)
            mask[i*seq + j] = -1e4f; // large negative (not -inf to avoid NaN in fp16)
    NSData *blob = build_blob(mask, 1, n); // [1, SEQ*SEQ] → stored as flat, reinterpreted as [1,1,SEQ,SEQ]
    free(mask);
    return blob;
}

// Full attention forward
static NSString *gen_attention_fwd(void) {
    float sc = 1.0f / sqrtf((float)HD);
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> xn) {\n", DIM, SEQ];

    // QKV projections via 1x1 conv
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wk = const()[name=string(\"Wk\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wv = const()[name=string(\"Wv\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wo = const()[name=string(\"Wo\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];

    // Q = Wq @ xn, K = Wk @ xn, V = Wv @ xn  [1, DIM, 1, SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)[name=string(\"cq\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)[name=string(\"ck\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)[name=string(\"cv\")];\n", DIM, SEQ];

    // Reshape [1, DIM, 1, SEQ] → [1, HEADS, HD, SEQ]
    [m appendFormat:@"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=qsh,x=kf)[name=string(\"rk\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=qsh,x=vf)[name=string(\"rv\")];\n", HEADS, HD, SEQ];

    // Transpose [1, HEADS, HD, SEQ] → [1, HEADS, SEQ, HD]
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];\n", HEADS, SEQ, HD];

    // Scaled dot-product: scores = Q @ K^T / sqrt(HD)
    [m appendString:@"        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n"];
    [m appendString:@"        bool ty = const()[name=string(\"ty\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)[name=string(\"mm1\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS, SEQ, SEQ];

    // Causal mask + softmax
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), "
        "val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", SEQ, SEQ, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS, SEQ, SEQ];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS, SEQ, SEQ];

    // attn_out = softmax_scores @ V  [1, HEADS, SEQ, HD]
    [m appendString:@"        bool tf = const()[name=string(\"tf\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=tf,transpose_y=tf,x=aw,y=v)[name=string(\"mm2\")];\n", HEADS, SEQ, HD];

    // Transpose back [1, HEADS, SEQ, HD] → [1, HEADS, HD, SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];\n", HEADS, HD, SEQ];

    // Reshape [1, HEADS, HD, SEQ] → [1, DIM, 1, SEQ]
    [m appendFormat:@"        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=os,x=at)[name=string(\"ra\")];\n", DIM, SEQ];

    // Wo projection
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=af)[name=string(\"out\")];\n", DIM, SEQ];

    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ================================================================
// CPU Reference
// ================================================================

static void cpu_attention_fwd(float *out, const float *xn,
                               const float *Wq, const float *Wk, const float *Wv, const float *Wo,
                               int dim, int heads, int hd, int seq) {
    float sc = 1.0f / sqrtf((float)hd);
    int n = dim * seq;

    // QKV projections (channel-first)
    float *Q = (float*)malloc(n * sizeof(float));
    float *K = (float*)malloc(n * sizeof(float));
    float *V = (float*)malloc(n * sizeof(float));

    // Q = Wq @ xn (channel-first: W[dim,dim], x[dim,seq])
    for (int o = 0; o < dim; o++)
        for (int s = 0; s < seq; s++) {
            float sum = 0;
            for (int i = 0; i < dim; i++) sum += Wq[o*dim+i] * xn[i*seq+s];
            Q[o*seq+s] = sum;
        }
    for (int o = 0; o < dim; o++)
        for (int s = 0; s < seq; s++) {
            float sum = 0;
            for (int i = 0; i < dim; i++) sum += Wk[o*dim+i] * xn[i*seq+s];
            K[o*seq+s] = sum;
        }
    for (int o = 0; o < dim; o++)
        for (int s = 0; s < seq; s++) {
            float sum = 0;
            for (int i = 0; i < dim; i++) sum += Wv[o*dim+i] * xn[i*seq+s];
            V[o*seq+s] = sum;
        }

    // Per-head attention
    // Q,K,V layout: [dim, seq] = [heads*hd, seq]
    // For head h: Q_h[hd, seq] at offset h*hd*seq
    float *attn_flat = (float*)malloc(n * sizeof(float)); // [dim, seq]

    for (int h = 0; h < heads; h++) {
        float *Qh = Q + h * hd * seq;
        float *Kh = K + h * hd * seq;
        float *Vh = V + h * hd * seq;
        float *Oh = attn_flat + h * hd * seq;

        // scores[i,j] = sum_d(Q[d,i] * K[d,j]) * scale (channel-first)
        float *scores = (float*)calloc(seq * seq, sizeof(float));
        for (int i = 0; i < seq; i++)
            for (int j = 0; j < seq; j++) {
                float s = 0;
                for (int d = 0; d < hd; d++) s += Qh[d*seq+i] * Kh[d*seq+j];
                scores[i*seq+j] = s * sc;
            }

        // Causal mask
        for (int i = 0; i < seq; i++)
            for (int j = i+1; j < seq; j++)
                scores[i*seq+j] = -1e4f;

        // Softmax per row
        for (int i = 0; i < seq; i++) {
            float maxv = -1e30f;
            for (int j = 0; j < seq; j++) if (scores[i*seq+j] > maxv) maxv = scores[i*seq+j];
            float sum = 0;
            for (int j = 0; j < seq; j++) { scores[i*seq+j] = expf(scores[i*seq+j] - maxv); sum += scores[i*seq+j]; }
            for (int j = 0; j < seq; j++) scores[i*seq+j] /= sum;
        }

        // attn_out = scores @ V  → Oh[d, i] = sum_j(scores[i,j] * Vh[d, j])
        for (int d = 0; d < hd; d++)
            for (int i = 0; i < seq; i++) {
                float s = 0;
                for (int j = 0; j < seq; j++) s += scores[i*seq+j] * Vh[d*seq+j];
                Oh[d*seq+i] = s;
            }

        free(scores);
    }

    // Wo projection: out = Wo @ attn_flat
    for (int o = 0; o < dim; o++)
        for (int s = 0; s < seq; s++) {
            float sum = 0;
            for (int i = 0; i < dim; i++) sum += Wo[o*dim+i] * attn_flat[i*seq+s];
            out[o*seq+s] = sum;
        }

    free(Q); free(K); free(V); free(attn_flat);
}

// ================================================================
// Test
// ================================================================

NSString *ane_attention_test(void) {
    @autoreleasepool {
        ane_init();
        NSMutableString *out = [NSMutableString string];
        if (!g_D || !g_I) { [out appendString:@"  ANE classes missing\n"]; return out; }

        [out appendString:@"  --- Attention Forward Test ---\n"];
        [out appendFormat:@"  Config: DIM=%d HEADS=%d HD=%d SEQ=%d\n", DIM, HEADS, HD, SEQ];

        // Random data
        srand48(123);
        int n = DIM * SEQ;
        float scale = 1.0f / sqrtf((float)DIM);
        float *xn  = (float*)malloc(n * sizeof(float));
        float *Wq  = (float*)malloc(DIM*DIM * sizeof(float));
        float *Wk  = (float*)malloc(DIM*DIM * sizeof(float));
        float *Wv  = (float*)malloc(DIM*DIM * sizeof(float));
        float *Wo  = (float*)malloc(DIM*DIM * sizeof(float));
        float *cpu_out = (float*)malloc(n * sizeof(float));
        float *ane_out = (float*)malloc(n * sizeof(float));

        for (int i = 0; i < n; i++) xn[i] = (float)(2*drand48()-1);
        for (int i = 0; i < DIM*DIM; i++) Wq[i] = (float)(2*drand48()-1) * scale;
        for (int i = 0; i < DIM*DIM; i++) Wk[i] = (float)(2*drand48()-1) * scale;
        for (int i = 0; i < DIM*DIM; i++) Wv[i] = (float)(2*drand48()-1) * scale;
        for (int i = 0; i < DIM*DIM; i++) Wo[i] = (float)(2*drand48()-1) * scale;

        // CPU reference
        [out appendString:@"  Computing CPU reference...\n"];
        uint64_t t0 = mach_absolute_time();
        cpu_attention_fwd(cpu_out, xn, Wq, Wk, Wv, Wo, DIM, HEADS, HD, SEQ);
        double cpuMs = tb_ms(mach_absolute_time() - t0);
        [out appendFormat:@"  CPU: %.1f ms\n", cpuMs];

        // ANE: compile attention kernel
        [out appendString:@"  Compiling ANE kernel...\n"];
        NSString *mil = gen_attention_fwd();
        NSData *wqBlob = build_blob(Wq, DIM, DIM);
        NSData *wkBlob = build_blob(Wk, DIM, DIM);
        NSData *wvBlob = build_blob(Wv, DIM, DIM);
        NSData *woBlob = build_blob(Wo, DIM, DIM);
        NSData *maskBlob = build_causal_mask(SEQ);

        NSDictionary *weights = @{
            @"@model_path/weights/wq.bin":   @{@"offset":@0, @"data":wqBlob},
            @"@model_path/weights/wk.bin":   @{@"offset":@0, @"data":wkBlob},
            @"@model_path/weights/wv.bin":   @{@"offset":@0, @"data":wvBlob},
            @"@model_path/weights/wo.bin":   @{@"offset":@0, @"data":woBlob},
            @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":maskBlob},
        };

        // Total baked weight size: 4 * DIM*DIM*2 + SEQ*SEQ*2 = 4*1.125MB + 128KB ≈ 4.6MB
        int inBytes  = DIM * SEQ * 2;
        int outBytes = DIM * SEQ * 2;

        t0 = mach_absolute_time();
        Kern *k = compile_kern(mil, weights, inBytes, outBytes);
        double compileMs = tb_ms(mach_absolute_time() - t0);

        if (!k) {
            [out appendString:@"  Attention compile FAILED\n"];
            // Try to diagnose: print MIL size
            [out appendFormat:@"  MIL size: %lu bytes\n", (unsigned long)[mil lengthOfBytesUsingEncoding:NSUTF8StringEncoding]];
            free(xn); free(Wq); free(Wk); free(Wv); free(Wo); free(cpu_out); free(ane_out);
            return out;
        }
        [out appendFormat:@"  Compiled in %.1f ms\n", compileMs];

        // Eval
        io_write_fp16(k->ioIn, xn, DIM, SEQ);
        t0 = mach_absolute_time();
        BOOL ok = ane_eval(k);
        double aneMs = tb_ms(mach_absolute_time() - t0);

        if (!ok) {
            [out appendString:@"  Attention eval FAILED\n"];
            free_kern(k);
            free(xn); free(Wq); free(Wk); free(Wv); free(Wo); free(cpu_out); free(ane_out);
            return out;
        }

        io_read_fp16(k->ioOut, ane_out, 0, DIM, SEQ);

        // Compare
        float maxErr = 0, sumErr = 0;
        int errCount = 0;
        for (int i = 0; i < n; i++) {
            float err = fabsf(ane_out[i] - cpu_out[i]);
            if (err > maxErr) maxErr = err;
            sumErr += err;
            if (err > 0.1f) errCount++;
        }
        float avgErr = sumErr / n;
        [out appendFormat:@"  ANE: %.3f ms  (%.0fx vs CPU)\n", aneMs, cpuMs/aneMs];
        [out appendFormat:@"  Max error: %.6f  Avg error: %.6f  Outliers(>0.1): %d/%d\n",
            maxErr, avgErr, errCount, n];
        // FP16 attention has higher error due to softmax chain
        [out appendFormat:@"  Attention: %@\n", maxErr < 1.0f ? @"PASS" : @"FAIL"];

        // Benchmark
        t0 = mach_absolute_time();
        for (int i = 0; i < 50; i++) ane_eval(k);
        double benchMs = tb_ms(mach_absolute_time() - t0);
        [out appendFormat:@"  Throughput: %.3f ms/eval (50 runs)\n", benchMs/50];

        free_kern(k);
        free(xn); free(Wq); free(Wk); free(Wv); free(Wo); free(cpu_out); free(ane_out);

        return out;
    }
}
