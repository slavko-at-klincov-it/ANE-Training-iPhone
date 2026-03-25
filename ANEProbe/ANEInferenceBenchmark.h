// ANEInferenceBenchmark.h — Inference benchmark comparing ANE vs GPU (MPS) vs CPU
// Runs Stories-110M forward pass on all three backends, measures latency and throughput.
#pragma once
#import <Foundation/Foundation.h>

// Run inference benchmark: N forward passes on ANE, GPU (MPS), and CPU (Accelerate).
// iterations: number of forward passes per backend (recommended: 50-100)
// Returns human-readable comparison report.
NSString *ane_inference_benchmark(int iterations);

// Power benchmark: runs each backend for `minutes_per_backend` minutes,
// measures battery drain and computes Watts + Tokens/Joule for ANE vs GPU vs CPU.
// Requires device unplugged from charger. Recommended: 5-10 min per backend.
NSString *ane_power_benchmark(float minutes_per_backend);
