// ANECheckpoint.h — Training state persistence for iOS ANE training
// Saves/loads weights + Adam optimizer state + training metadata
// Atomic writes for crash safety, checksum for integrity
#pragma once
#import <Foundation/Foundation.h>
#include <stdbool.h>
#include <stdint.h>

// ===== Model config (must match ANETrainingConfig.h) =====
#ifndef DIM
#define DIM 768
#define HIDDEN 2048
#define HEADS 12
#define HD (DIM/HEADS)
#define SEQ 256
#define NLAYERS 12
#define VOCAB 32000
#endif

// ===== Weight sizes per layer =====
#define CKPT_WQ_SZ (DIM*DIM)
#define CKPT_WO_SZ (DIM*DIM)
#define CKPT_W1_SZ (HIDDEN*DIM)
#define CKPT_W2_SZ (DIM*HIDDEN)
#define CKPT_W3_SZ (HIDDEN*DIM)
#define CKPT_LAYER_PARAMS (4*CKPT_WQ_SZ + CKPT_W1_SZ + CKPT_W2_SZ + CKPT_W3_SZ + 2*DIM)
#define CKPT_TOTAL_PARAMS (NLAYERS * CKPT_LAYER_PARAMS + DIM + VOCAB*DIM)

// Number of floats in all per-layer weights (no optimizer)
#define CKPT_LAYER_WEIGHT_FLOATS (4*CKPT_WQ_SZ + CKPT_W1_SZ + CKPT_W2_SZ + CKPT_W3_SZ + 2*DIM)
// Number of floats in per-layer Adam state (m + v for each weight)
#define CKPT_LAYER_ADAM_FLOATS (2 * CKPT_LAYER_WEIGHT_FLOATS)

// ===== Checkpoint header (compatible with macOS format) =====
#define CKPT_MAGIC 0x424C5A54  // "BLZT"
#define CKPT_VERSION 3         // iOS version

typedef struct {
    uint32_t magic;             // 0x424C5A54 "BLZT"
    uint32_t version;           // 3 (iOS version)
    int32_t step, total_steps;
    int32_t n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len;
    float lr, loss;
    double cum_compile_ms, cum_train_ms, cum_wall_ms;
    int32_t cum_steps, cum_batches;
    int32_t adam_t;
    int32_t pad[3];             // alignment to 128 bytes
} ANECkptHeader;

// ===== Per-layer weight struct (mirrors LayerWeights from stories_config.h) =====
typedef struct {
    float *Wq, *Wk, *Wv, *Wo;
    float *W1, *W2, *W3;
    float *rms_att, *rms_ffn;
} ANELayerWeights;

// ===== Per-layer Adam optimizer state =====
typedef struct {
    float *m, *v;
    size_t n;
} ANEAdamState;

typedef struct {
    ANEAdamState Wq, Wk, Wv, Wo;
    ANEAdamState W1, W2, W3;
    ANEAdamState rms_att, rms_ffn;
} ANELayerAdam;

// ===== llama2.c model file header =====
typedef struct {
    int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} ANELlama2Config;

// ===== Checkpoint slot management =====
#define CKPT_NUM_SLOTS 2
#define CKPT_DEFAULT_INTERVAL 100  // save every N steps

// ===== Core functions =====

// Save full training state to binary file (atomic: writes .tmp then renames)
bool ane_save_checkpoint(const char *path, ANECkptHeader *hdr,
                         ANELayerWeights *lw, ANELayerAdam *la,
                         float *rms_final, ANEAdamState *adam_rms_final,
                         float *embed, ANEAdamState *adam_embed);

// Load training state from checkpoint file
// Validates magic, version, config, and checksum
bool ane_load_checkpoint(const char *path, ANECkptHeader *hdr,
                         ANELayerWeights *lw, ANELayerAdam *la,
                         float *rms_final, ANEAdamState *adam_rms_final,
                         float *embed, ANEAdamState *adam_embed);

// Load pretrained weights from llama2.c format (no optimizer state)
bool ane_load_pretrained(const char *path,
                         ANELayerWeights *lw, float *rms_final, float *embed);

// ===== Auto-checkpoint manager =====

// Initialize checkpoint manager (registers for background/thermal notifications)
void ane_checkpoint_manager_init(void);

// Call every step — saves if current_step % interval == 0
void ane_checkpoint_maybe_save(int current_step, int interval);

// Set the state pointers for auto-save (must be called before ane_checkpoint_maybe_save)
void ane_checkpoint_set_state(ANECkptHeader *hdr,
                              ANELayerWeights *lw, ANELayerAdam *la,
                              float *rms_final, ANEAdamState *adam_rms_final,
                              float *embed, ANEAdamState *adam_embed);

// Force immediate save (called on background/thermal events)
void ane_checkpoint_force_save(void);

// ===== Path helpers =====

const char *ane_documents_path(void);            // app's Documents directory
const char *ane_checkpoint_path(int slot);        // Documents/ane_ckpt_0.bin or _1.bin
const char *ane_latest_checkpoint_path(void);     // most recent valid checkpoint, or NULL

// ===== Alloc/free helpers =====

ANELayerWeights ane_layer_weights_alloc(void);
void ane_layer_weights_free(ANELayerWeights *w);
ANEAdamState ane_adam_state_alloc(size_t n);
void ane_adam_state_free(ANEAdamState *s);
ANELayerAdam ane_layer_adam_alloc(void);
void ane_layer_adam_free(ANELayerAdam *a);

// ===== Test =====

void ane_checkpoint_test(void);
