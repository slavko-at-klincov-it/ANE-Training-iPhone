// ANETrainingEngine.h — Public C API for full Stories110M training on iPhone ANE
// Ported from macOS ANE-Training/training/train_large.m
#pragma once
#import <Foundation/Foundation.h>

typedef struct ANETrainState ANETrainState;

// Initialize training state. model_path can be NULL for random init.
// data_path points to pre-tokenized .bin file (uint16_t tokens, mmap'd)
ANETrainState *ane_train_init(const char *model_path, const char *data_path);

// Run one training step. Returns loss.
// Handles recompile automatically when weights are updated.
float ane_train_step(ANETrainState *state);

// Get current state
int ane_train_current_step(ANETrainState *state);
float ane_train_current_loss(ANETrainState *state);
bool ane_train_is_compiling(ANETrainState *state);

// Save checkpoint to app Documents directory
void ane_train_save(ANETrainState *state);

// Free all resources
void ane_train_free(ANETrainState *state);

// Quick test: init with random weights + dummy data, run 5 steps, verify loss decreases
NSString *ane_training_engine_test(void);
