// ANEUltimateBenchmark.h — Definitive ANE benchmark: inference (ANE/GPU/CPU) + training
// Measures: latency, throughput, power, efficiency, thermal, CPU utilization, memory
// Per-component timing breakdown, idle baseline subtraction, percentile statistics
#pragma once
#import <Foundation/Foundation.h>

// Run the ultimate benchmark. Phases:
//   1. System baseline (idle power measurement)
//   2. Inference comparison: ANE vs GPU(batched MPS) vs CPU, 3 backends
//   3. Training: ANE with component micro-benchmarks
//   4. Results: comprehensive report + CSV
//
// total_minutes: 0 = full ~50 min run. Non-zero scales phases proportionally.
// Returns human-readable report. CSV written to Documents/ane_ultimate_benchmark.csv
NSString *ane_ultimate_benchmark(float total_minutes);

// Standalone training benchmark. Runs ONLY training (no inference).
// Avoids memory conflicts. minutes: duration of training phase.
// Runs optimized: ACCUM_STEPS=8, no sdpaBwd2 recompile, concurrent dW.
NSString *ane_training_only_benchmark(float minutes);

// ANE-only power test. Runs ANE inference for `minutes` to measure power drain.
// Needs long runs (15+ min) since ANE draws <1.5W and battery has 1% granularity.
NSString *ane_power_test(float minutes);
