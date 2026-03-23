# Integration Guide — ANE Training for iOS Apps

How to integrate ANE training capabilities into your own iOS app.

---

## Overview

Two integration levels:

**Level 1 — Swift API (recommended):** Use `ANETrainer` for a 5-line integration.
**Level 2 — C API:** Full control over the training loop.

```
Your App
   │
   ▼
ANETrainer.swift     ←── High-level Swift API (addText, train, scheduleOvernight)
   │
   ├── ANETokenizer.h/.m      (BPE tokenizer: text → token IDs)
   ├── ANETrainingEngine.h/.m  (C training loop: init, step, save, free)
   ├── ANEInference.h/.m       (text generation with trained model)
   ├── ANECheckpoint.h/.m      (save/load training state)
   │
   └── Internal:
       ├── ANETrainingConfig.h (ANE runtime + IOSurface I/O)
       ├── ANEStoriesMIL.h     (MIL kernel generators)
       ├── ANEStoriesCPUOps.h  (CPU ops: RMSNorm, Adam, loss)
       └── ANEDataPipeline.h   (token loading + batch prep)
```

---

## Quick Integration (Swift API)

```swift
import UIKit

class MyViewController: UIViewController {
    let trainer = ANETrainer.shared

    // During the day: collect training data
    func userDidWriteText(_ text: String) {
        trainer.addText(text)
    }

    // When user goes to sleep / app backgrounds
    func scheduleTraining() {
        trainer.scheduleOvernight(hours: 8)
    }

    // Next morning: generate text
    func generateResponse(prompt: String) -> String {
        let docsDir = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first!
        let ckptPath = "\(docsDir)/ane_ckpt_0.bin"

        guard let state = ane_inference_init(ckptPath, nil) else { return "No model" }
        defer { ane_inference_free(state) }

        guard let cStr = ane_generate(state, prompt, 100, 0.8) else { return "Generation failed" }
        defer { free(cStr) }
        return String(cString: cStr)
    }

    // SwiftUI: bind to trainer's published properties
    // trainer.status, trainer.currentStep, trainer.bestLoss, trainer.stepsPerSecond
}
```

### ANETrainer API Reference

```swift
// Singleton
let trainer = ANETrainer.shared

// Data collection
trainer.addText("raw text")           // tokenize + append to training file
trainer.addTokens([42, 100, 2048])    // append pre-tokenized IDs
trainer.collectedTokenCount            // how many tokens collected

// Training
trainer.train(steps: 1000) { result in ... }   // step-based
trainer.train(hours: 8.0) { result in ... }     // time-based
trainer.scheduleOvernight(hours: 8)              // BGProcessingTask
trainer.stop()                                    // graceful stop + checkpoint

// State (ObservableObject — bind to SwiftUI)
trainer.status          // .idle, .training, .completed, .error
trainer.currentStep     // Int
trainer.bestLoss        // Float
trainer.currentLoss     // Float
trainer.stepsPerSecond  // Double

// Checkpoint
trainer.saveCheckpoint()
trainer.loadCheckpoint() -> Bool
trainer.hasCheckpoint    // Bool

// Result (returned in completion handler)
result.steps            // total steps trained
result.duration         // TimeInterval
result.bestLoss         // Float
result.finalLoss        // Float
result.adamUpdates      // Int
```

### Tokenizer API

```c
// Load tokenizer (from app bundle)
ANETokenizer *tok = ane_tokenizer_load_from_bundle();

// Encode
int len;
uint16_t *tokens = ane_tokenize(tok, "Hello world", &len);
// tokens = [15043, 3186], len = 2

// Decode
char *text = ane_detokenize(tok, tokens, len);
// text = "Hello world"

free(tokens);
free(text);
ane_tokenizer_free(tok);
```

### Inference API

```c
// Init (from checkpoint or pretrained weights)
ANEInferenceState *state = ane_inference_init("/path/to/checkpoint.bin", NULL);

// Generate
char *output = ane_generate(state, "Once upon a time", 100, 0.8);
printf("%s\n", output);
free(output);

// Cleanup
ane_inference_free(state);
```

---

## Option 1: Copy Source Files (Simplest)

### Required Files

Copy these files into your Xcode project:

**Full integration with Swift API (recommended, 14 files + 2 data):**
```
ANETrainer.swift               # High-level Swift API
ANETokenizer.h/.m              # BPE tokenizer
ANEInference.h/.m              # Text generation
ANETrainingEngine.h/.m         # Training loop
ANETrainingConfig.h            # ANE runtime layer
ANEStoriesMIL.h                # MIL kernel generators
ANEStoriesCPUOps.h             # CPU operations
ANEDataPipeline.h/.m           # Data loading + loss
ANECheckpoint.h/.m             # Checkpoint save/load
tokenizer.bin                  # BPE vocabulary (bundle resource)
tinystories_data00.bin         # Training data (bundle resource, optional)
```

**For thermal management (optional, +4 files):**
```
ANEThermal.h/.m                # Thermal monitoring + adaptive control
ANEBackgroundTraining.swift    # BGProcessingTask scheduling
ANEThermalMonitor.swift        # SwiftUI thermal observer
```

**Minimum C-only integration (7 files, no Swift):**
```
ANETrainingConfig.h            # ANE runtime
ANEStoriesMIL.h                # MIL generators
ANEStoriesCPUOps.h             # CPU ops
ANETrainingEngine.h/.m         # Training loop
ANEDataPipeline.h/.m           # Data loading
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

## Real-World Example: Meus (Thought Journal App)

[Meus](https://github.com/slavko-at-klincov-it) is a thought journal app where users speak their thoughts throughout the day. ANE Training personalizes a language model to the user's thinking style — entirely on-device, no cloud.

### Architecture

```
User speaks thought
    ↓
Speech-to-Text (iOS)
    ↓
trainer.addText(thought)     // tokenize + append to training data
    ↓
trainer.train(steps: 4)      // ~1 second, runs in background
    ↓
User puts phone down          // model is already slightly more personalized
```

### Integration Code

```swift
import ANEProbe

class ThoughtRecorder {
    let trainer = ANETrainer.shared

    // Called after each thought is transcribed
    func didRecordThought(_ text: String) {
        // 1. Add to training data (tokenized + appended to .bin file)
        trainer.addText(text)

        // 2. Train immediately — 4 steps, ~1 second, user won't notice
        trainer.train(steps: 4) { _ in }
    }

    // Called when app enters background
    func appDidBackground() {
        // Save checkpoint in case phone gets turned off
        trainer.saveCheckpoint()

        // If charging: bonus training on all today's data
        if UIDevice.current.batteryState == .charging {
            trainer.scheduleOvernight(hours: 4)
        }
    }

    // Use the personalized model
    func suggestCompletion(for prompt: String) -> String {
        let docsDir = NSSearchPathForDirectoriesInDomains(
            .documentDirectory, .userDomainMask, true).first!
        let ckptPath = "\(docsDir)/ane_ckpt_0.bin"

        guard let state = ane_inference_init(ckptPath, nil) else { return "" }
        defer { ane_inference_free(state) }

        guard let cStr = ane_generate(state, prompt, 50, 0.7) else { return "" }
        defer { free(cStr) }
        return String(cString: cStr)
    }
}
```

### Training Pattern: Micro-Training

There is no difference between "micro-training" and "normal training" — each step is independent. The choice is purely about **when** to train:

| Trigger | Steps | Time | Use Case |
|:--|:-:|:-:|:--|
| After each thought | 4 | ~1s | Immediate learning, works even if phone turns off later |
| App goes to background | 20 | ~5s | Catch-up on recent thoughts |
| Overnight while charging | 60,000+ | 8h | Deep training on all accumulated data |

All three can coexist. The checkpoint system ensures seamless resume between sessions.

### Why This Works

- **Each thought** contains the user's vocabulary, sentence structure, and topics
- **4 steps per thought** nudges the model toward the user's style
- **After 30 days** of ~10 thoughts/day: ~1,200 training steps from personal data
- **The model never sees a server** — everything stays on the iPhone's ANE
- **No minimum session length** — even 1 step is meaningful

### Typical Day

```
08:15  "I need to remember to call the dentist today"
       → addText + 4 steps (1s)

10:30  "The meeting with Sarah went well, we decided to..."
       → addText + 4 steps (1s)

12:45  "Had an idea about the new feature — what if we..."
       → addText + 4 steps (1s)

18:00  "Today was productive, I feel good about the progress"
       → addText + 4 steps (1s)

22:00  Phone plugged in, user sleeps
       → Overnight: 4h training on all 4 thoughts + historical data
```

After weeks of this: the model understands how this specific person thinks, what topics they care about, and how they express themselves.

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
