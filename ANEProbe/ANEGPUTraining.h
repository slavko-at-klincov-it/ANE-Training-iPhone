// ANEGPUTraining.h — GPU-based transformer training via Metal Performance Shaders
// Mirrors ANETrainingEngine but uses MPS matmuls instead of ANE kernels.
// Key advantage: NO kernel recompilation needed after weight updates.
// Weights live in MTLBuffer (SharedMemory) — Adam writes directly, GPU reads next step.
#pragma once
#import <Foundation/Foundation.h>

typedef struct GPUTrainState GPUTrainState;

// Initialize GPU training state. model_path can be NULL for random init.
// data_path points to pre-tokenized .bin file (uint16_t tokens, mmap'd)
GPUTrainState *gpu_train_init(const char *model_path, const char *data_path);

// Run one training step. Returns loss.
float gpu_train_step(GPUTrainState *state);

// Get current state
int gpu_train_current_step(GPUTrainState *state);
float gpu_train_current_loss(GPUTrainState *state);

// Free all resources
void gpu_train_free(GPUTrainState *state);

// Quick test: 20 steps, verify loss decreases
NSString *gpu_training_test(void);

// Benchmark: run for `minutes`, report steps/s, power, thermal
NSString *gpu_training_benchmark(float minutes);
