// ANEBenchmark.h — Performance, power, and efficiency benchmark for ANE training
// Measures: throughput (steps/s, tokens/s), power draw (Watts via battery drain),
// efficiency (tokens/Joule), thermal profile, and compile vs compute overhead.
#pragma once
#import <Foundation/Foundation.h>

// Run the full benchmark for `minutes` duration (recommended: 30 min minimum).
// Returns a human-readable report with all metrics.
// Phases:
//   1. Warmup (2 min) — establish baseline timing, let ANE reach steady state
//   2. Sustained (main duration) — detailed measurement with per-second logging
//   3. Summary — statistical analysis, power estimation, efficiency calculation
NSString *ane_benchmark_run(float minutes);
