# Integration Guide — ANE Training for iOS Apps

How to integrate ANE training capabilities into your own iOS app.

---

## Overview

This project provides a C API for training transformers on the Apple Neural Engine. You can integrate it as a static library, or copy the source files directly into your project.

```
Your App
   │
   ▼
ANETrainingEngine.h  ←── Public C API (init, step, save, free)
   │
   ├── ANETrainingConfig.h     (ANE runtime + IOSurface I/O)
   ├── ANEStoriesMIL.h         (MIL kernel generators)
   ├── ANEStoriesCPUOps.h      (CPU ops: RMSNorm, Adam, loss)
   ├── ANEDataPipeline.h       (token loading + batch prep)
   └── ANECheckpoint.h         (save/load training state)
```

---

## Option 1: Copy Source Files (Simplest)

### Required Files

Copy these files into your Xcode project:

**Minimum for training (6 files):**
```
ANETrainingConfig.h      # ANE runtime layer
ANEStoriesMIL.h          # MIL kernel generators
ANEStoriesCPUOps.h       # CPU operations
ANETrainingEngine.h      # Public API header
ANETrainingEngine.m      # Training engine implementation
ANEDataPipeline.h/.m     # Data loading + loss
```

**For checkpoint support (+2 files):**
```
ANECheckpoint.h/.m       # Save/load training state
```

**For thermal management (+2 files):**
```
ANEThermal.h/.m          # Thermal monitoring + adaptive control
```

**For background training (+2 Swift files):**
```
ANEBackgroundTraining.swift    # BGProcessingTask
ANEThermalMonitor.swift        # SwiftUI thermal observer
```

### Build Settings

Add to your Xcode project or `project.yml`:

```yaml
settings:
  base:
    SWIFT_OBJC_BRIDGING_HEADER: YourApp-Bridging-Header.h
```

In your bridging header:
```objc
#import "ANETrainingEngine.h"
#import "ANECheckpoint.h"      // optional
#import "ANEDataPipeline.h"    // optional
#import "ANEThermal.h"         // optional
```

Link `Accelerate.framework` (system SDK framework):
```yaml
dependencies:
  - sdk: Accelerate.framework
```

### Info.plist (for background training)

```xml
<key>UIBackgroundModes</key>
<array>
    <string>processing</string>
</array>
<key>BGTaskSchedulerPermittedIdentifiers</key>
<array>
    <string>your.app.bundle.training</string>
</array>
```

---

## Option 2: Use the C API Directly

### API Reference

```c
#import "ANETrainingEngine.h"

// 1. Initialize
// model_path: path to llama2.c format weights (NULL for random init)
// data_path: path to pre-tokenized .bin file (uint16_t token IDs)
ANETrainState *state = ane_train_init(model_path, data_path);

// 2. Run training steps
float loss = ane_train_step(state);  // returns current loss

// 3. Query state
int step = ane_train_current_step(state);
float loss = ane_train_current_loss(state);
bool compiling = ane_train_is_compiling(state);

// 4. Save checkpoint
ane_train_save(state);

// 5. Cleanup
ane_train_free(state);
```

### Training Loop Example (ObjC)

```objc
#import "ANETrainingEngine.h"

- (void)trainInBackground {
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_BACKGROUND, 0), ^{
        // Init with bundled data
        NSString *dataPath = [[NSBundle mainBundle] pathForResource:@"training_data" ofType:@"bin"];
        ANETrainState *state = ane_train_init(NULL, [dataPath UTF8String]);

        if (!state) {
            NSLog(@"Training init failed");
            return;
        }

        // Train for 1000 steps
        for (int i = 0; i < 1000; i++) {
            float loss = ane_train_step(state);

            if (i % 100 == 0) {
                NSLog(@"Step %d: loss=%.4f", i, loss);
            }

            // Save checkpoint every 500 steps
            if (i % 500 == 0 && i > 0) {
                ane_train_save(state);
            }
        }

        ane_train_save(state);
        ane_train_free(state);
    });
}
```

### Training Loop Example (Swift)

```swift
import Foundation

func startTraining() {
    Task.detached(priority: .background) {
        guard let dataPath = Bundle.main.path(forResource: "training_data", ofType: "bin") else {
            print("No training data found")
            return
        }

        guard let state = ane_train_init(nil, dataPath) else {
            print("Training init failed")
            return
        }

        for step in 0..<1000 {
            let loss = ane_train_step(state)

            if step % 100 == 0 {
                print("Step \(step): loss=\(loss)")
            }
        }

        ane_train_save(state)
        ane_train_free(state)
    }
}
```

### Time-Based Training (Overnight)

```c
// Train for 8 hours with automatic LR adjustment
NSString *result = ane_timed_training(8.0);
// Result contains: steps, best loss, phases, duration
```

---

## Data Format

### Training Data

Pre-tokenized binary file: array of `uint16_t` token IDs.

```
[token0][token1][token2]...[tokenN]
  2 bytes each, little-endian
```

Same format as [llama2.c](https://github.com/karpathy/llama2.c) tokenized data.

To create your own:
```python
import struct

tokens = [42, 100, 2048, ...]  # your token IDs (0..32000)
with open("my_data.bin", "wb") as f:
    for t in tokens:
        f.write(struct.pack("<H", t))
```

Minimum size: `(SEQ + 1) * 2 bytes` = 514 bytes for SEQ=256.
Recommended: 100K+ tokens for meaningful training.

### Pretrained Weights (Optional)

[llama2.c](https://github.com/karpathy/llama2.c) binary format:
```
[header: 7 int32s] [embed: VOCAB*DIM float32s] [per-layer weights...]
```

If no pretrained weights are provided (`model_path = NULL`), random Xavier initialization is used.

---

## Model Configuration

The model architecture is defined at compile time in `ANETrainingConfig.h`:

```c
#define DIM 768       // embedding dimension
#define HIDDEN 2048   // FFN hidden dimension
#define HEADS 12      // attention heads
#define HD (DIM/HEADS) // head dimension (64)
#define SEQ 256       // sequence length
#define NLAYERS 12    // transformer layers
#define VOCAB 32000   // vocabulary size
```

To change the architecture, modify these defines and recompile. The MIL generators in `ANEStoriesMIL.h` automatically adapt to the new dimensions.

### Memory Requirements

| Config | Params | Weights | +Adam | Total |
|:--|:-:|:-:|:-:|:-:|
| Stories-110M (default) | 110M | 440MB | +880MB | ~1.4GB |
| Smaller (DIM=512, HIDDEN=1024, 6 layers) | ~25M | 100MB | +200MB | ~350MB |
| Tiny (DIM=256, HIDDEN=512, 4 layers) | ~5M | 20MB | +40MB | ~70MB |

iPhone RAM limit is ~2-3GB for foreground apps. The default 110M config fits with Adam.

---

## Customization

### Custom MIL Kernels

To add new layer types, create MIL generators following the pattern in `ANEStoriesMIL.h`:

```c
static NSString *gen_my_custom_kernel(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];  // standard MIL header
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];

    // Your MIL operations here
    // Supported: add, sub, mul, div, relu, tanh, sigmoid, silu, softmax,
    //            exp, sqrt, pow, reduce_mean, reduce_sum, transpose, reshape,
    //            slice_by_size, conv (1x1 for matmul)

    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}
```

### ANE Op Compatibility

| Works | Doesn't Work | Workaround |
|:--|:--|:--|
| conv (1x1) | matmul with singleton dims | reshape to remove singletons |
| softmax | gelu | sigmoid(1.702*x)*x |
| reduce_mean/sum | log | CPU fallback |
| transpose, reshape | rsqrt | div(1, sqrt(x)) |
| sigmoid, silu, tanh | concat (standalone) | works inside larger MIL programs |

**Critical:** Output IOSurfaces must be ≥16KB or eval silently fails.

---

## Limitations & Caveats

1. **Private API** — Uses `AppleNeuralEngine.framework`. Not App Store compatible. Use TestFlight/Ad-Hoc/Enterprise.

2. **Compile-time architecture** — DIM, HEADS, etc. are `#define`s. Changing requires recompile.

3. **FP16 precision** — ANE operates in FP16. Weight gradients accumulate in FP32 on CPU.

4. **239 model limit** — Max 239 concurrent loaded ANE models. The training engine uses 72 (well within limit). If your app also uses CoreML models, account for shared capacity.

5. **Recompile overhead** — Every weight update requires recompiling 60 ANE kernels (~5s). Dynamic spatial packing (0.12ms/iter) is implemented but not yet integrated into the full training loop.

6. **Single device tested** — Verified on iPhone 15 Pro (A17 Pro, h16). Other A-series chips should work but performance may vary.

---

## Troubleshooting

| Problem | Cause | Fix |
|:--|:--|:--|
| `ane_train_init` returns NULL | ANE classes not found | Ensure `AppleNeuralEngine.framework` is available (all modern iPhones) |
| Eval returns NO silently | Output IOSurface < 16KB | Use `make_surface()` which enforces 16KB minimum |
| `Program load failure (0x50004)` | Too many loaded models (>239) | Call `free_kern()` to unload unused models |
| NaN in loss | FP16 overflow | Reduce learning rate, check for large weight values |
| Slow first step (~8s) | Initial kernel compilation | Normal — 72 kernels being compiled. Subsequent steps are ~230ms |
| App killed in background | iOS memory pressure | Reduce model size or use SGD instead of Adam |
