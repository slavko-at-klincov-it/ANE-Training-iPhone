// ANEDataPipeline.h — Tokenizer, data loading, and training ops for iOS ANE training
// Handles: pre-tokenized binary data, embedding lookup, cross-entropy loss, batch prep
#pragma once
#import <Foundation/Foundation.h>

// ── Data loading ──────────────────────────────────────────────────────────────

/// Token data handle (mmap-backed or malloc-backed)
typedef struct {
    uint16_t *tokens;   // Token ID array
    size_t    count;    // Number of tokens
    size_t    mmap_len; // If >0, this is mmap'd (use munmap to free)
    int       fd;       // File descriptor for mmap (-1 if malloc'd)
} TokenData;

/// Load pre-tokenized binary file (uint16_t token IDs, same format as macOS).
/// Supports mmap for large files. Path can be app bundle or Documents directory.
/// Returns NULL on failure.
TokenData *load_token_data(const char *path);

/// Free token data (handles both mmap and malloc)
void free_token_data(TokenData *td);

/// Load token data from the app bundle by resource name (e.g. "train_data", "bin")
TokenData *load_token_data_from_bundle(NSString *name, NSString *ext);

/// Load token data from Documents directory
TokenData *load_token_data_from_documents(NSString *filename);

// ── Random sequence sampling ─────────────────────────────────────────────────

/// Sample a random contiguous sequence of (SEQ+1) tokens for training.
/// Returns the starting position. Caller uses tokens[pos..pos+SEQ] as input
/// and tokens[pos+1..pos+SEQ+1] as targets.
size_t sample_training_position(const TokenData *td, int seq_len);

// ── Embedding lookup ─────────────────────────────────────────────────────────

/// Forward: token IDs → x [DIM, SEQ] channel-first
/// embed is [VOCAB, DIM] row-major
void embed_lookup(float *x, const float *embed, const uint16_t *tokens, int dim, int seq);

/// Backward: accumulate d_embed[tok] += dx[:,t]
void embed_backward(float *d_embed, const float *dx, const uint16_t *tokens, int dim, int seq);

// ── Cross-entropy loss ───────────────────────────────────────────────────────

/// Compute cross-entropy loss and write gradient into dlogits.
/// logits/dlogits: [VOCAB, SEQ] column-major (v*SEQ+t)
/// targets: [SEQ] token IDs
/// Returns mean CE loss over positions.
float cross_entropy_loss(float *dlogits, const float *logits, const uint16_t *targets, int V, int S);

// ── Batch preparation ────────────────────────────────────────────────────────

/// Prepare a training batch from token data.
/// - tokens: source token array (at least pos + seq + 1 elements)
/// - pos: starting position in token array
/// - embed: embedding table [VOCAB, DIM] row-major
/// - x_out: output activation [DIM, SEQ] channel-first (caller allocates)
/// - input_tokens_out: the SEQ input token IDs (caller allocates, uint16_t[SEQ])
/// - target_tokens_out: the SEQ target token IDs (caller allocates, uint16_t[SEQ])
/// - dim, seq: model dimensions
void prepare_batch(const uint16_t *tokens, size_t pos, const float *embed,
                   float *x_out, uint16_t *input_tokens_out, uint16_t *target_tokens_out,
                   int dim, int seq);

// ── Test ─────────────────────────────────────────────────────────────────────

/// End-to-end data pipeline test: dummy tokens → embed → loss → gradient check
NSString *ane_data_pipeline_test(void);
