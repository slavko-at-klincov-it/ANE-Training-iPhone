// ANETokenizer.h — BPE tokenizer for Stories-110M (VOCAB=32000)
// Loads llama2.c tokenizer.bin format, encodes/decodes text ↔ token IDs
#pragma once
#import <Foundation/Foundation.h>

typedef struct {
    char **vocab;        // vocab[i] = token string (null-terminated)
    float *vocab_scores; // vocab_scores[i] = BPE merge score
    int vocab_size;
    int max_token_length;
    // Hash table for O(1) string → token ID lookup
    int *hash_keys;      // token IDs stored at hash positions
    int hash_size;
} ANETokenizer;

// Load tokenizer from file path or app bundle
ANETokenizer *ane_tokenizer_load(const char *path);
ANETokenizer *ane_tokenizer_load_from_bundle(void);
void ane_tokenizer_free(ANETokenizer *t);

// Encode text → token IDs (BPE). Returns malloc'd array, sets *out_len.
// Caller must free() the returned array.
uint16_t *ane_tokenize(ANETokenizer *t, const char *text, int *out_len);

// Decode token IDs → text. Returns malloc'd string. Caller must free().
char *ane_detokenize(ANETokenizer *t, const uint16_t *tokens, int len);

// Test: load from bundle, tokenize/detokenize, verify roundtrip, benchmark
NSString *ane_tokenizer_test(void);
