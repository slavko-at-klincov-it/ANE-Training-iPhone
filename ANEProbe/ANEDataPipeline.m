// ANEDataPipeline.m — Tokenizer, data loading, and training CPU ops for iOS ANE training
// Ported from macOS ANE-Training/training/stories_cpu_ops.h
#import "ANEDataPipeline.h"
#import <Accelerate/Accelerate.h>
#import <sys/mman.h>
#import <sys/stat.h>
#import <fcntl.h>
#import <mach/mach_time.h>

// Import ANETrainingConfig.h if available, otherwise define model constants locally
#if __has_include("ANETrainingConfig.h")
#import "ANETrainingConfig.h"
#else
// Stories-110M model configuration
#ifndef DIM
#define DIM 768
#endif
#ifndef SEQ
#define SEQ 256
#endif
#ifndef VOCAB
#define VOCAB 32000
#endif
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - Data Loading (mmap-backed)
// ═══════════════════════════════════════════════════════════════════════════════

TokenData *load_token_data(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[data] open failed: %s\n", path);
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "[data] fstat failed: %s\n", path);
        close(fd);
        return NULL;
    }

    size_t file_size = (size_t)st.st_size;
    if (file_size < 2) {
        fprintf(stderr, "[data] file too small: %zu bytes\n", file_size);
        close(fd);
        return NULL;
    }

    void *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "[data] mmap failed: %s\n", path);
        close(fd);
        return NULL;
    }

    // Advise sequential access for training workload
    madvise(mapped, file_size, MADV_SEQUENTIAL);

    TokenData *td = (TokenData *)calloc(1, sizeof(TokenData));
    td->tokens   = (uint16_t *)mapped;
    td->count    = file_size / sizeof(uint16_t);
    td->mmap_len = file_size;
    td->fd       = fd;

    fprintf(stderr, "[data] loaded %zu tokens from %s (%.1f MB)\n",
            td->count, path, (double)file_size / (1024.0 * 1024.0));
    return td;
}

void free_token_data(TokenData *td) {
    if (!td) return;
    if (td->mmap_len > 0) {
        munmap(td->tokens, td->mmap_len);
        close(td->fd);
    } else {
        free(td->tokens);
    }
    free(td);
}

TokenData *load_token_data_from_bundle(NSString *name, NSString *ext) {
    NSString *path = [[NSBundle mainBundle] pathForResource:name ofType:ext];
    if (!path) {
        fprintf(stderr, "[data] resource not found in bundle: %s.%s\n",
                [name UTF8String], [ext UTF8String]);
        return NULL;
    }
    return load_token_data([path UTF8String]);
}

TokenData *load_token_data_from_documents(NSString *filename) {
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *docsDir = [paths firstObject];
    NSString *path = [docsDir stringByAppendingPathComponent:filename];
    return load_token_data([path UTF8String]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - Random Sequence Sampling
// ═══════════════════════════════════════════════════════════════════════════════

size_t sample_training_position(const TokenData *td, int seq_len) {
    // Need seq_len + 1 tokens (input[0..seq-1] + target[1..seq])
    size_t max_pos = td->count - (size_t)(seq_len + 1);
    return (size_t)(arc4random_uniform((uint32_t)max_pos));
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - Embedding Lookup (Forward + Backward)
// ═══════════════════════════════════════════════════════════════════════════════

// Forward: token_ids → x [DIM, SEQ] channel-first
// embed is [VOCAB, DIM] row-major (vocab_size rows, dim cols)
void embed_lookup(float *x, const float *embed, const uint16_t *tokens, int dim, int seq) {
    // Zero output first
    memset(x, 0, (size_t)dim * (size_t)seq * sizeof(float));
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        if (tok >= VOCAB) {
            fprintf(stderr, "WARN: token %d out of range [0,%d)\n", tok, VOCAB);
            continue;
        }
        // embed[tok*dim + d] → x[d*seq + t] (scatter into channel-first layout)
        const float *src = embed + tok * dim;
        for (int d = 0; d < dim; d++) {
            x[d * seq + t] = src[d];
        }
    }
}

// Backward: accumulate dE[tok] += dx[:,t] for each position
void embed_backward(float *d_embed, const float *dx, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        if (tok >= VOCAB) continue;
        // dx[d*seq + t] → d_embed[tok*dim + d] (gather from channel-first layout)
        float *dst = d_embed + tok * dim;
        for (int d = 0; d < dim; d++) {
            dst[d] += dx[d * seq + t];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - Cross-Entropy Loss + Gradient (Accelerate-optimized)
// ═══════════════════════════════════════════════════════════════════════════════

// Cross-entropy loss + gradient for logits (column-major: [VOCAB, SEQ])
// logits[v*SEQ+t] = logit for vocab v, position t
// targets[t] = target token id for position t
// Returns mean CE loss, writes dlogits = (softmax(logits) - one_hot(targets)) / SEQ
float cross_entropy_loss(float *dlogits, const float *logits, const uint16_t *targets, int V, int S) {
    // Work in transposed layout [S, V] where each row is one position's logits (contiguous)
    float *buf = (float *)malloc((size_t)S * (size_t)V * sizeof(float));

    // Transpose [V,S] → [S,V]: buf[t*V+v] = logits[v*S+t]
    vDSP_mtrans(logits, 1, buf, 1, (vDSP_Length)S, (vDSP_Length)V);

    float total_loss = 0;
    float invS = 1.0f / S;

    for (int t = 0; t < S; t++) {
        float *row = buf + t * V;

        // Numerically stable softmax: subtract max
        float maxv;
        vDSP_maxv(row, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv;
        vDSP_vsadd(row, 1, &neg_max, row, 1, (vDSP_Length)V);

        // exp in-place (vectorized via Accelerate)
        int n = V;
        vvexpf(row, row, &n);

        // sum of exponentials
        float sum;
        vDSP_sve(row, 1, &sum, (vDSP_Length)V);

        // normalize → softmax probabilities
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(row, 1, &inv_sum, row, 1, (vDSP_Length)V);

        // accumulate loss: -log(p[target])
        int tgt = targets[t];
        if (tgt >= V) {
            fprintf(stderr, "WARN: target token %d out of vocab range [0,%d)\n", tgt, V);
            continue;
        }
        total_loss -= logf(row[tgt] + 1e-10f);

        // gradient: softmax - one_hot, scaled by 1/S
        row[tgt] -= 1.0f;
        vDSP_vsmul(row, 1, &invS, row, 1, (vDSP_Length)V);
    }

    // Transpose back [S,V] → [V,S]
    vDSP_mtrans(buf, 1, dlogits, 1, (vDSP_Length)V, (vDSP_Length)S);
    free(buf);
    return total_loss / S;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - Batch Preparation
// ═══════════════════════════════════════════════════════════════════════════════

void prepare_batch(const uint16_t *tokens, size_t pos, const float *embed,
                   float *x_out, uint16_t *input_tokens_out, uint16_t *target_tokens_out,
                   int dim, int seq) {
    // Split: input = tokens[pos..pos+seq-1], target = tokens[pos+1..pos+seq]
    memcpy(input_tokens_out, tokens + pos, (size_t)seq * sizeof(uint16_t));
    memcpy(target_tokens_out, tokens + pos + 1, (size_t)seq * sizeof(uint16_t));

    // Embedding lookup: input tokens → x [dim, seq] channel-first
    embed_lookup(x_out, embed, input_tokens_out, dim, seq);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - End-to-End Test
// ═══════════════════════════════════════════════════════════════════════════════

NSString *ane_data_pipeline_test(void) {
    @autoreleasepool {
        NSMutableString *out = [NSMutableString string];
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);

        #define TB_MS(t) ((double)(t) * tb.numer / tb.denom / 1e6)

        [out appendString:@"  --- Data Pipeline Test ---\n"];

        // ── 1. Create dummy token data ────────────────────────────────────────
        const int test_seq = SEQ;
        const int test_dim = DIM;
        const int test_vocab = VOCAB;
        const int num_tokens = test_seq + 1;  // Need SEQ+1 for input/target split

        uint16_t *dummy_tokens = (uint16_t *)malloc((size_t)num_tokens * sizeof(uint16_t));
        for (int i = 0; i < num_tokens; i++) {
            dummy_tokens[i] = (uint16_t)(arc4random_uniform(test_vocab));
        }
        [out appendFormat:@"  Created %d dummy tokens (vocab=%d)\n", num_tokens, test_vocab];
        [out appendFormat:@"  First 8 tokens: [%d, %d, %d, %d, %d, %d, %d, %d]\n",
            dummy_tokens[0], dummy_tokens[1], dummy_tokens[2], dummy_tokens[3],
            dummy_tokens[4], dummy_tokens[5], dummy_tokens[6], dummy_tokens[7]];

        // ── 2. Create dummy embedding table ───────────────────────────────────
        [out appendFormat:@"  Allocating embedding table [%d, %d] (%.1f MB)...\n",
            test_vocab, test_dim,
            (double)test_vocab * test_dim * sizeof(float) / (1024.0 * 1024.0)];

        float *embed = (float *)calloc((size_t)test_vocab * test_dim, sizeof(float));
        // Initialize with small random values (Xavier-style)
        float scale = 1.0f / sqrtf((float)test_dim);
        for (int i = 0; i < test_vocab * test_dim; i++) {
            embed[i] = scale * (2.0f * (float)drand48() - 1.0f);
        }

        // ── 3. Batch preparation ──────────────────────────────────────────────
        float *x = (float *)calloc((size_t)test_dim * test_seq, sizeof(float));
        uint16_t *input_toks = (uint16_t *)malloc((size_t)test_seq * sizeof(uint16_t));
        uint16_t *target_toks = (uint16_t *)malloc((size_t)test_seq * sizeof(uint16_t));

        uint64_t t0 = mach_absolute_time();
        prepare_batch(dummy_tokens, 0, embed, x, input_toks, target_toks, test_dim, test_seq);
        double prep_ms = TB_MS(mach_absolute_time() - t0);

        [out appendFormat:@"  Batch prep: %.3f ms\n", prep_ms];
        [out appendFormat:@"  Input[0]=%d, Target[0]=%d (target should be input shifted by 1)\n",
            input_toks[0], target_toks[0]];

        // Verify shift: target[t] == dummy_tokens[t+1]
        int shift_ok = 1;
        for (int t = 0; t < test_seq; t++) {
            if (target_toks[t] != dummy_tokens[t + 1]) { shift_ok = 0; break; }
        }
        [out appendFormat:@"  Token shift verification: %s\n", shift_ok ? "PASS" : "FAIL"];

        // ── 4. Verify embedding lookup ────────────────────────────────────────
        // Check x[d*seq + 0] == embed[input_toks[0]*dim + d] for all d
        int embed_ok = 1;
        int tok0 = input_toks[0];
        for (int d = 0; d < test_dim; d++) {
            float expected = embed[tok0 * test_dim + d];
            float actual = x[d * test_seq + 0];
            if (fabsf(expected - actual) > 1e-6f) {
                [out appendFormat:@"  Embed mismatch at d=%d: expected %.6f, got %.6f\n",
                    d, expected, actual];
                embed_ok = 0;
                break;
            }
        }
        [out appendFormat:@"  Embedding lookup verification: %s\n", embed_ok ? "PASS" : "FAIL"];

        // Spot-check: print x stats
        float x_min, x_max, x_mean;
        vDSP_Length x_min_idx, x_max_idx;
        vDSP_Length x_n = (vDSP_Length)(test_dim * test_seq);
        vDSP_minvi(x, 1, &x_min, &x_min_idx, x_n);
        vDSP_maxvi(x, 1, &x_max, &x_max_idx, x_n);
        vDSP_meanv(x, 1, &x_mean, x_n);
        [out appendFormat:@"  x stats: min=%.4f max=%.4f mean=%.6f shape=[%d,%d]\n",
            x_min, x_max, x_mean, test_dim, test_seq];

        // ── 5. Cross-entropy loss with random logits ──────────────────────────
        size_t logits_size = (size_t)test_vocab * test_seq;
        float *logits = (float *)malloc(logits_size * sizeof(float));
        float *dlogits = (float *)malloc(logits_size * sizeof(float));

        // Random logits (small values to keep softmax stable)
        for (size_t i = 0; i < logits_size; i++) {
            logits[i] = 0.1f * (2.0f * (float)drand48() - 1.0f);
        }
        [out appendFormat:@"  Logits shape: [%d, %d] (%.1f MB)\n",
            test_vocab, test_seq,
            (double)logits_size * sizeof(float) / (1024.0 * 1024.0)];

        t0 = mach_absolute_time();
        float loss = cross_entropy_loss(dlogits, logits, target_toks, test_vocab, test_seq);
        double loss_ms = TB_MS(mach_absolute_time() - t0);

        [out appendFormat:@"  Cross-entropy loss: %.4f (expected ~%.1f for random logits)\n",
            loss, logf((float)test_vocab)];
        [out appendFormat:@"  Loss computation: %.3f ms\n", loss_ms];

        // Verify loss is reasonable: for random logits, CE ~ log(vocab) = log(32000) ~ 10.37
        float expected_loss = logf((float)test_vocab);
        int loss_ok = (loss > expected_loss * 0.8f && loss < expected_loss * 1.2f);
        [out appendFormat:@"  Loss sanity check (within 20%% of log(%d)=%.2f): %s\n",
            test_vocab, expected_loss, loss_ok ? "PASS" : "FAIL"];

        // ── 6. Verify gradient shape and properties ───────────────────────────
        // dlogits should sum to ~0 per position (softmax - one_hot sums to 0 before /S)
        // After /S, sum per position should be ~0/S ~ 0
        float grad_min, grad_max, grad_mean;
        vDSP_Length g_min_idx, g_max_idx;
        vDSP_minvi(dlogits, 1, &grad_min, &g_min_idx, (vDSP_Length)logits_size);
        vDSP_maxvi(dlogits, 1, &grad_max, &g_max_idx, (vDSP_Length)logits_size);
        vDSP_meanv(dlogits, 1, &grad_mean, (vDSP_Length)logits_size);
        [out appendFormat:@"  dlogits stats: min=%.6f max=%.6f mean=%.8f\n",
            grad_min, grad_max, grad_mean];

        // The gradient for each position should sum to 0 (softmax - one_hot) / S
        // Check column sums (sum over vocab for each position)
        int grad_sum_ok = 1;
        for (int t = 0; t < test_seq; t++) {
            float col_sum = 0;
            for (int v = 0; v < test_vocab; v++) {
                col_sum += dlogits[v * test_seq + t];
            }
            // Should be ~0 (the -1 for one-hot exactly cancels the +1 sum of softmax)
            // But scaled by 1/S, so tolerance is 1e-4
            if (fabsf(col_sum) > 1e-4f) {
                [out appendFormat:@"  Grad column %d sum = %.6f (expected ~0)\n", t, col_sum];
                grad_sum_ok = 0;
                break;
            }
        }
        [out appendFormat:@"  Gradient column-sum check (should be ~0): %s\n",
            grad_sum_ok ? "PASS" : "FAIL"];

        // Gradient at target position should be negative (softmax_prob - 1) / S < 0
        int grad_neg_ok = 1;
        for (int t = 0; t < MIN(8, test_seq); t++) {
            int tgt = target_toks[t];
            float g = dlogits[tgt * test_seq + t];
            if (g >= 0) {
                [out appendFormat:@"  Grad at target[%d]=%d: %.6f (expected < 0)\n", t, tgt, g];
                grad_neg_ok = 0;
                break;
            }
        }
        [out appendFormat:@"  Target gradient sign check (should be < 0): %s\n",
            grad_neg_ok ? "PASS" : "FAIL"];

        // ── 7. Embedding backward ─────────────────────────────────────────────
        // Use dlogits as a proxy for dx (just testing the backward function works)
        // Create a fake dx of the right shape [DIM, SEQ]
        float *dx = (float *)calloc((size_t)test_dim * test_seq, sizeof(float));
        for (size_t i = 0; i < (size_t)test_dim * test_seq; i++) {
            dx[i] = 0.01f * (2.0f * (float)drand48() - 1.0f);
        }

        float *d_embed = (float *)calloc((size_t)test_vocab * test_dim, sizeof(float));

        t0 = mach_absolute_time();
        embed_backward(d_embed, dx, input_toks, test_dim, test_seq);
        double bwd_ms = TB_MS(mach_absolute_time() - t0);

        [out appendFormat:@"  Embedding backward: %.3f ms\n", bwd_ms];

        // Verify: d_embed should be non-zero only at rows corresponding to tokens in input
        int bwd_ok = 1;
        // Check that d_embed[tok0] is non-zero
        float row_sum = 0;
        for (int d = 0; d < test_dim; d++) {
            row_sum += fabsf(d_embed[tok0 * test_dim + d]);
        }
        if (row_sum < 1e-10f) {
            [out appendFormat:@"  d_embed[tok=%d] is all zeros — FAIL\n", tok0];
            bwd_ok = 0;
        }
        // Check that a random token NOT in input has zero gradient (probabilistic)
        // Find a token not in input_toks (with high probability for vocab=32000)
        uint16_t absent_tok = 0;
        for (uint16_t cand = 0; cand < (uint16_t)test_vocab; cand++) {
            int found = 0;
            for (int t = 0; t < test_seq; t++) {
                if (input_toks[t] == cand) { found = 1; break; }
            }
            if (!found) { absent_tok = cand; break; }
        }
        float absent_sum = 0;
        for (int d = 0; d < test_dim; d++) {
            absent_sum += fabsf(d_embed[absent_tok * test_dim + d]);
        }
        if (absent_sum > 1e-10f) {
            [out appendFormat:@"  d_embed[tok=%d] (absent) is non-zero (%.6f) — FAIL\n",
                absent_tok, absent_sum];
            bwd_ok = 0;
        }
        [out appendFormat:@"  Embedding backward verification: %s\n", bwd_ok ? "PASS" : "FAIL"];

        // ── 8. Benchmark: repeated loss computation ───────────────────────────
        int bench_iters = 5;
        t0 = mach_absolute_time();
        for (int i = 0; i < bench_iters; i++) {
            cross_entropy_loss(dlogits, logits, target_toks, test_vocab, test_seq);
        }
        double bench_ms = TB_MS(mach_absolute_time() - t0);
        [out appendFormat:@"  Benchmark: %dx cross_entropy_loss = %.1f ms total, %.2f ms/iter\n",
            bench_iters, bench_ms, bench_ms / bench_iters];

        // ── 9. Benchmark: embedding lookup ────────────────────────────────────
        t0 = mach_absolute_time();
        for (int i = 0; i < bench_iters; i++) {
            embed_lookup(x, embed, input_toks, test_dim, test_seq);
        }
        double embed_ms = TB_MS(mach_absolute_time() - t0);
        [out appendFormat:@"  Benchmark: %dx embed_lookup = %.1f ms total, %.2f ms/iter\n",
            bench_iters, embed_ms, embed_ms / bench_iters];

        // ── 10. Summary ───────────────────────────────────────────────────────
        int all_pass = shift_ok && embed_ok && loss_ok && grad_sum_ok && grad_neg_ok && bwd_ok;
        [out appendFormat:@"\n  === DATA PIPELINE: %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED"];
        [out appendFormat:@"  Timings: batch_prep=%.2fms loss=%.2fms embed_bwd=%.2fms\n",
            prep_ms, loss_ms, bwd_ms];

        // Cleanup
        free(dummy_tokens);
        free(embed);
        free(x);
        free(input_toks);
        free(target_toks);
        free(logits);
        free(dlogits);
        free(dx);
        free(d_embed);

        #undef TB_MS
        return out;
    }
}
