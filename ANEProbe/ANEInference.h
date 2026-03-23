// ANEInference.h — Text generation (inference) engine for iPhone ANE
// Uses the same forward-pass MIL kernels as training, but no backward pass.
// Only 24 kernels needed (12 fwdAttn + 12 fwdFFN) vs 72 for training.
#pragma once
#import <Foundation/Foundation.h>

typedef struct ANEInferenceState ANEInferenceState;

// Initialize inference with trained weights
// weights_path: checkpoint file (BLZT) or pretrained llama2.c format, or NULL for random init
// tokenizer_path: tokenizer.bin path, or NULL to load from app bundle
ANEInferenceState *ane_inference_init(const char *weights_path, const char *tokenizer_path);

// Generate text from a prompt
// prompt: input text (will be tokenized)
// max_tokens: maximum number of tokens to generate
// temperature: sampling temperature (0.0 = greedy, 1.0 = normal)
// Returns malloc'd string (caller must free)
char *ane_generate(ANEInferenceState *state, const char *prompt, int max_tokens, float temperature);

// Free inference state
void ane_inference_free(ANEInferenceState *state);

// Test function: init with random weights, generate 20 tokens, report pipeline status
NSString *ane_inference_test(void);
