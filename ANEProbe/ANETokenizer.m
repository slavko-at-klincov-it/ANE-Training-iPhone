// ANETokenizer.m — BPE tokenizer matching llama2.c for Stories-110M
#import "ANETokenizer.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <mach/mach_time.h>

// ===== Hash table for vocab lookup =====

static unsigned int _tok_hash(const char *s) {
    // FNV-1a hash
    unsigned int h = 2166136261u;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 16777619u;
    }
    return h;
}

static void _tok_hash_build(ANETokenizer *t) {
    // Open addressing hash table, 2x vocab size
    t->hash_size = t->vocab_size * 2;
    t->hash_keys = (int *)malloc(t->hash_size * sizeof(int));
    memset(t->hash_keys, -1, t->hash_size * sizeof(int));
    for (int i = 0; i < t->vocab_size; i++) {
        unsigned int idx = _tok_hash(t->vocab[i]) % t->hash_size;
        while (t->hash_keys[idx] != -1) {
            idx = (idx + 1) % t->hash_size;
        }
        t->hash_keys[idx] = i;
    }
}

// Lookup string in vocab, returns token ID or -1
static int _tok_lookup(ANETokenizer *t, const char *s) {
    unsigned int idx = _tok_hash(s) % t->hash_size;
    while (t->hash_keys[idx] != -1) {
        int tok = t->hash_keys[idx];
        if (strcmp(t->vocab[tok], s) == 0) return tok;
        idx = (idx + 1) % t->hash_size;
    }
    return -1;
}

// ===== Load / Free =====

ANETokenizer *ane_tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[tokenizer] cannot open %s\n", path);
        return NULL;
    }

    ANETokenizer *t = (ANETokenizer *)calloc(1, sizeof(ANETokenizer));
    t->vocab_size = 32000; // Stories-110M / LLaMA2 vocab

    fread(&t->max_token_length, sizeof(int), 1, f);

    t->vocab = (char **)malloc(t->vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(t->vocab_size * sizeof(float));

    for (int i = 0; i < t->vocab_size; i++) {
        fread(&t->vocab_scores[i], sizeof(float), 1, f);
        int len;
        fread(&len, sizeof(int), 1, f);
        t->vocab[i] = (char *)malloc(len + 1);
        fread(t->vocab[i], 1, len, f);
        t->vocab[i][len] = '\0';
    }

    fclose(f);

    // Build hash table for fast lookup
    _tok_hash_build(t);

    return t;
}

ANETokenizer *ane_tokenizer_load_from_bundle(void) {
    NSString *path = [[NSBundle mainBundle] pathForResource:@"tokenizer" ofType:@"bin"];
    if (!path) {
        fprintf(stderr, "[tokenizer] tokenizer.bin not found in bundle\n");
        return NULL;
    }
    return ane_tokenizer_load([path UTF8String]);
}

void ane_tokenizer_free(ANETokenizer *t) {
    if (!t) return;
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->vocab_scores);
    free(t->hash_keys);
    free(t);
}

// ===== BPE Encode =====

uint16_t *ane_tokenize(ANETokenizer *t, const char *text, int *out_len) {
    if (!text || !*text) { *out_len = 0; return NULL; }

    size_t text_len = strlen(text);

    // Allocate working buffer for token IDs (max = text_len tokens)
    int *tokens = (int *)malloc((text_len + 1) * sizeof(int));
    int n_tokens = 0;

    // Step 1: Encode each byte as its single-character token
    // For UTF-8 characters, try multi-byte lookup first, then fall back to byte tokens
    const char *p = text;
    while (*p) {
        // Determine UTF-8 character length
        int char_len = 1;
        unsigned char c = (unsigned char)*p;
        if (c >= 0xF0) char_len = 4;
        else if (c >= 0xE0) char_len = 3;
        else if (c >= 0xC0) char_len = 2;

        // Try to find the UTF-8 character as a single token
        char buf[8];
        if (char_len <= 4) {
            memcpy(buf, p, char_len);
            buf[char_len] = '\0';
            int id = _tok_lookup(t, buf);
            if (id != -1) {
                tokens[n_tokens++] = id;
                p += char_len;
                continue;
            }
        }

        // Fallback: use byte tokens <0xNN> for each byte
        for (int b = 0; b < char_len && *p; b++) {
            unsigned char byte = (unsigned char)*p;
            // Byte fallback tokens are at indices 3..258 for bytes 0x00..0xFF
            tokens[n_tokens++] = byte + 3;
            p++;
        }
    }

    // Step 2: BPE merge — repeatedly merge the highest-scoring adjacent pair
    // This matches llama2.c's greedy BPE algorithm exactly
    char merged[256]; // max_token_length is 27, so 256 is plenty

    while (1) {
        float best_score = -1e10f;
        int best_idx = -1;
        int best_id = -1;

        for (int i = 0; i < n_tokens - 1; i++) {
            snprintf(merged, sizeof(merged), "%s%s",
                     t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = _tok_lookup(t, merged);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_idx = i;
                best_id = id;
            }
        }

        if (best_idx == -1) break; // No more merges

        // Apply merge: replace tokens[best_idx] with merged token,
        // shift remaining tokens left by 1
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < n_tokens - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        n_tokens--;
    }

    // Convert to uint16_t output
    uint16_t *result = (uint16_t *)malloc(n_tokens * sizeof(uint16_t));
    for (int i = 0; i < n_tokens; i++) {
        result[i] = (uint16_t)tokens[i];
    }
    *out_len = n_tokens;

    free(tokens);
    return result;
}

// ===== Decode =====

char *ane_detokenize(ANETokenizer *t, const uint16_t *tokens, int len) {
    if (!tokens || len == 0) {
        char *empty = (char *)malloc(1);
        empty[0] = '\0';
        return empty;
    }

    // Estimate output size: max_token_length * len
    size_t buf_size = (size_t)t->max_token_length * len + 1;
    char *buf = (char *)malloc(buf_size);
    size_t pos = 0;

    for (int i = 0; i < len; i++) {
        int tok = tokens[i];
        if (tok < 0 || tok >= t->vocab_size) continue;

        const char *piece = t->vocab[tok];

        // Handle byte fallback tokens: <0xNN>
        if (tok >= 3 && tok <= 258) {
            // Byte token: index 3 = 0x00, index 258 = 0xFF
            unsigned char byte_val = (unsigned char)(tok - 3);
            if (pos < buf_size - 1) {
                buf[pos++] = (char)byte_val;
            }
        } else {
            size_t piece_len = strlen(piece);
            if (pos + piece_len < buf_size) {
                memcpy(buf + pos, piece, piece_len);
                pos += piece_len;
            }
        }
    }

    buf[pos] = '\0';
    return buf;
}

// ===== Test =====

NSString *ane_tokenizer_test(void) {
    NSMutableString *log = [NSMutableString string];

    // Load
    ANETokenizer *t = ane_tokenizer_load_from_bundle();
    if (!t) {
        return @"[FAIL] Cannot load tokenizer.bin from bundle";
    }
    [log appendFormat:@"Loaded tokenizer: vocab=%d max_token_len=%d\n",
     t->vocab_size, t->max_token_length];

    // Tokenize test string
    const char *test_str = "Once upon a time there was a little girl";
    int n_tokens = 0;
    uint16_t *tokens = ane_tokenize(t, test_str, &n_tokens);

    [log appendFormat:@"Input: \"%s\"\n", test_str];
    [log appendFormat:@"Tokens (%d): [", n_tokens];
    for (int i = 0; i < n_tokens; i++) {
        [log appendFormat:@"%s%d", i ? ", " : "", tokens[i]];
    }
    [log appendString:@"]\n"];

    // Show token strings
    [log appendString:@"Decoded tokens: "];
    for (int i = 0; i < n_tokens; i++) {
        [log appendFormat:@"'%s' ", t->vocab[tokens[i]]];
    }
    [log appendString:@"\n"];

    // Detokenize
    char *decoded = ane_detokenize(t, tokens, n_tokens);
    [log appendFormat:@"Roundtrip: \"%s\"\n", decoded];

    // Verify
    BOOL match = (strcmp(test_str, decoded) == 0);
    [log appendFormat:@"Roundtrip %s\n", match ? "OK" : "MISMATCH"];

    free(tokens);
    free(decoded);

    // Benchmark: tokenize a 1000-char string
    NSMutableString *bench_str = [NSMutableString string];
    while ([bench_str length] < 1000) {
        [bench_str appendString:@"Once upon a time there was a little girl who loved to read books. "];
    }
    const char *bench_cstr = [bench_str UTF8String];

    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);

    // Warm up
    int dummy_len;
    uint16_t *dummy = ane_tokenize(t, bench_cstr, &dummy_len);
    free(dummy);

    // Timed run
    int iters = 10;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        int blen;
        uint16_t *btok = ane_tokenize(t, bench_cstr, &blen);
        free(btok);
    }
    uint64_t t1 = mach_absolute_time();
    double ms = (double)(t1 - t0) * tb.numer / tb.denom / 1e6 / iters;

    [log appendFormat:@"\nBenchmark: %lu chars → %d tokens, %.2f ms/encode\n",
     strlen(bench_cstr), dummy_len, ms];

    ane_tokenizer_free(t);
    return log;
}
