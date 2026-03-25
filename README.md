# ANE-Training-iPhone

**Train neural networks directly on your iPhone's Neural Engine — no jailbreak, no cloud, no data leaves your device.**

The first open-source project that enables on-device transformer training on Apple's Neural Engine (ANE) via reverse-engineered private APIs. A 110M-parameter language model trains on an iPhone 15 Pro at 3.25 steps/second. Includes a production-ready app with training dashboard, data import, inference chat, and overnight scheduling.

---

## Why This Matters

| | Cloud Training | On-Device (CPU/GPU) | **This Project (ANE)** |
|:--|:-:|:-:|:-:|
| Privacy | Data leaves device | Private | **Private** |
| Cost | $/hour | Free | **Free** |
| Speed (110M model) | Fast | Very slow | **3.25 steps/s** |
| Offline | No | Yes | **Yes** |
| Requires jailbreak | N/A | No | **No** |
| Hardware utilization | GPU cluster | CPU/GPU only | **Dedicated ML chip** |

The Apple Neural Engine is a dedicated chip in every modern iPhone designed for ML inference. It's faster and more power-efficient than CPU/GPU for tensor operations. Apple once offered public ANE training APIs (MLCompute, 2020) but deprecated them without replacement. This project restores that capability via low-level private APIs.

---

## What You Can Do

### Today (Proven & Working)

- **Train a 110M-parameter transformer** (Stories-110M architecture) on iPhone ANE
- **Fine-tune on your own text data** — pre-tokenized binary format, loaded from app bundle or Documents
- **Background training** — train overnight while charging via BGProcessingTask
- **Checkpoint & resume** — save/load full training state (weights + optimizer) to survive app kills
- **Thermal-aware training** — automatically pause/slow down when iPhone gets warm

### Verified Results

```
Model:    Stories-110M (12 layers, 768 dim, 12 heads, 2048 hidden)
Data:     TinyStories (pre-tokenized)
Device:   iPhone 15 Pro (A17 Pro)

8-Hour Overnight Run:
  Steps:    64,040
  Duration: 8.0 hours
  Loss:     10.48 → 9.41 (best, -10.2%)
  Adam:     16,000 updates
  Speed:    2.2 steps/s — constant for 8 hours, zero thermal throttle
  LR:       Automatic plateau detection + adjustment (234 plateaus, 215 phases)
```

### Architecture Support

Any transformer with these building blocks works on ANE:

| Layer | ANE Support | Speed |
|:--|:-:|:-:|
| Linear (matmul via 1x1 conv) | Full | 0.73ms |
| RMSNorm | Full | 0.74ms |
| Multi-Head Attention (SDPA) | Full | 0.60ms |
| SwiGLU FFN | Full | 0.45ms |
| Softmax | Native | — |
| SiLU / Sigmoid / Tanh / ReLU | Native | — |

---

## Integration — 5 Lines to Train on ANE

```swift
let trainer = ANETrainer.shared

// Collect data during the day
trainer.addText("User wrote this today...")

// Train overnight (auto-schedules BGProcessingTask)
trainer.scheduleOvernight(hours: 8)

// Next morning: generate text with trained model
let state = ane_inference_init(checkpointPath, nil)
let text = String(cString: ane_generate(state, "Once upon a time", 100, 0.8))
```

`ANETrainer` is an `ObservableObject` — bind `status`, `currentStep`, `bestLoss` directly to SwiftUI.

See [INTEGRATION.md](INTEGRATION.md) for the full API reference, code examples, and data format.

---

## What You Need

### Hardware
- iPhone with A-series chip (tested: A17 Pro / iPhone 15 Pro)
- Developer Mode enabled on device
- Mac for building (Xcode)

### Software
- Xcode 16+
- [xcodegen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`)
- Apple Developer account (free is sufficient for device testing)

### Data
- Pre-tokenized training data as binary file (uint16_t token IDs)
- Same format as [llama2.c](https://github.com/karpathy/llama2.c) tokenized data

---

## Quick Start

```bash
# Clone
git clone https://github.com/slavko-at-klincov-it/ANE-Training-iPhone.git
cd ANE-Training-iPhone/ANEProbe

# Generate Xcode project
xcodegen generate

# Find your device ID
xcrun devicectl list devices

# Build and deploy
xcodebuild build \
  -project ANEProbe.xcodeproj \
  -scheme ANEProbe \
  -destination 'platform=iOS,id=YOUR_DEVICE_ID' \
  -allowProvisioningUpdates

# Install
xcrun devicectl device install app \
  --device YOUR_DEVICE_UUID \
  path/to/Build/Products/Debug-iphoneos/ANEProbe.app

# Launch and watch training
xcrun devicectl device process launch \
  --device YOUR_DEVICE_UUID \
  --console \
  com.klincov.aneprobe
```

The app starts training automatically on launch. Training progress is printed to the console and saved to `Documents/ane_training_log.txt` on the device.

---

## How It Works

```
Your Text Data (tokens)
        │
        ▼
┌─────────────────────────────┐
│      Embedding (CPU)        │  Token IDs → Vectors
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   12× Transformer Layer     │  Each layer:
│   ┌───────────────────────┐ │
│   │  ANE: Attention       │ │  RMSNorm → QKV → SDPA → Output Proj
│   │  (0.6ms per layer)    │ │  via compiled MIL → ANE hardware
│   └───────────────────────┘ │
│   + Residual (CPU)          │
│   ┌───────────────────────┐ │
│   │  ANE: SwiGLU FFN      │ │  RMSNorm → W1/W3 → SiLU → Gate → W2
│   │  (0.45ms per layer)   │ │
│   └───────────────────────┘ │
│   + Residual (CPU)          │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   Loss + Backward (CPU+ANE) │  Cross-entropy → Backprop through all layers
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   Adam Optimizer (CPU)      │  Update weights → Recompile ANE kernels
└─────────────────────────────┘
```

The ANE doesn't natively support training — only inference. This project makes training possible by:

1. **Compiling forward pass kernels** as MIL programs with baked weights → runs on ANE
2. **Running backward pass kernels** on ANE (custom MIL for each gradient computation)
3. **Computing weight gradients** on CPU (via Accelerate/BLAS — matrix outer products)
4. **Updating weights** on CPU (Adam optimizer), then recompiling ANE kernels with new weights

72 ANE kernels run simultaneously (12 layers × 6 kernels per layer).

---

## Advantages

- **100% Private** — All computation on-device. No data sent anywhere. Train on personal texts, notes, messages without privacy concerns.
- **Free** — No cloud GPU costs. Uses hardware you already own.
- **Offline** — Works in airplane mode. No internet required.
- **Power Efficient** — ANE uses significantly less power than CPU/GPU for the same operations. Idle power: 0mW (hard power gating).
- **No Jailbreak** — Standard developer-signed app. Works via TestFlight or Ad-Hoc distribution.
- **Background Training** — Train overnight while charging. Automatic checkpoint on interruption.
- **Thermal Aware** — Monitors device temperature, slows/pauses training to prevent overheating.

## Limitations

- **Speed** — 3.25 steps/s for 110M model (optimized). A cloud GPU does this 100-1000x faster. This is for personalization, not pre-training.
- **Model Size** — iPhone has ~2-3GB usable RAM. With Adam optimizer, ~110M parameters is the practical limit. Larger models need SGD (no momentum memory) or quantized optimizer states.
- **Recompile Overhead** — Weights are baked into MIL programs. Every weight update requires recompiling 60 ANE kernels (~1s). Optimized to 42% overhead (was 62%) via ACCUM_STEPS=8 and skipping weight-free kernel recompilation.
- **Batch Size 1** — Current implementation processes one sequence at a time. Gradient accumulation over 4 steps compensates.
- **No App Store** — Uses private Apple APIs. Distribution via TestFlight, Ad-Hoc, or enterprise signing only. An App Store variant would need a CoreML wrapper.
- **iPhone Only** — Tested on iPhone 15 Pro (A17 Pro). Other A-series chips likely work but are untested. iPad should work identically.
- **Inference UI included** — Chat interface for testing the trained model with temperature and token controls.
- **FP16 Precision** — ANE operates in FP16. Weight gradients accumulate in FP32 on CPU, but forward/backward passes have FP16 rounding. Sufficient for fine-tuning, may limit pre-training from scratch.

---

## Practical Use Cases

### 1. Personal Language Model
Train a small language model on your own writing style, notes, or messages. The model never sees a server — everything stays on your iPhone.

### 2. Domain Adaptation
Fine-tune a pre-trained model on domain-specific text (medical notes, legal documents, technical manuals) directly on the device that uses it.

### 3. Federated Learning Node
Each iPhone trains locally, only sharing model updates (not data). ANE makes this fast enough to be practical.

### 4. Research & Education
Experiment with transformer training on real hardware without cloud costs. Understand how ANE works at the lowest level.

### 5. Offline Learning
In environments without internet (flights, remote areas, secure facilities), the model can continue learning from local data.

---

## Performance

| Metric | Value |
|:--|:--|
| ANE Training speed | **3.25 steps/s** (optimized) |
| GPU Training speed | 2.17 steps/s (no recompilation) |
| ANE Inference | 2,480 tok/s (96.9 ms/pass) |
| CPU Inference | 3,215 tok/s (73.3 ms/pass) |
| ANE Power (inference) | **2.51 W** (3.4x more efficient than CPU) |
| ANE Tokens/Joule | **990** (vs CPU: 372) |
| Peak memory (training) | ~2.4 GB (with Adam) |
| ANE SRAM | ~32 MB |
| Max concurrent ANE models | 239 |

For detailed benchmarks, hardware specs, and memory layout see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Project Structure

```
ANEProbe/
├── Public API
│   ├── ANETrainer.swift             # High-level Swift API: addText, train, scheduleOvernight
│   ├── ANETokenizer.h/.m            # BPE tokenizer (32K vocab, llama2.c compatible)
│   └── ANEInference.h/.m            # Text generation with trained model
│
├── Core Infrastructure
│   ├── ANETrainingConfig.h          # ANE runtime, IOSurface I/O, structs, compile/eval
│   ├── ANEStoriesMIL.h              # 7 MIL kernel generators (attention, FFN, backward)
│   └── ANEStoriesCPUOps.h           # CPU ops: RMSNorm, Adam, cross-entropy, embedding
│
├── Training Engine
│   ├── ANETrainingEngine.m/.h       # Full 12-layer training loop + 8h overnight mode
│   ├── ANEDataPipeline.m/.h         # Token loading (mmap), batch prep, loss computation
│   └── ANECheckpoint.m/.h           # Checkpoint save/load, crash-safety, pretrained loading
│
├── System Integration
│   ├── ANEBackgroundTraining.swift   # BGProcessingTask scheduling + TrainingSession
│   ├── ANEThermal.m/.h              # Thermal monitoring, throttle detection, adaptive control
│   ├── ANEThermalMonitor.swift       # SwiftUI thermal state observer
│   └── ANEDynamicTrain.m/.h         # Dynamic spatial packing (0.12ms/iter, no recompile)
│
├── UI
│   ├── ANEProbe.swift               # Main app, overnight training launcher
│   ├── TrainingDashboardView.swift   # Training status: loss, steps, thermal, start/stop
│   └── ANEProbe-Bridging-Header.h   # Swift↔ObjC bridge (all test imports)
│
├── Research & Tests
│   ├── ANEDirectTest.m/.h           # Phase 1: MIL compile + eval proof
│   ├── ANEWeightTest.m/.h           # Phase 1: Recompile vs dynamic packing
│   ├── ANERE.m/.h                   # Phase 1.5: SRAM, op coverage, perf stats
│   ├── ANEReduceDebug.m/.h          # Phase 1.5: 16KB IOSurface bug investigation
│   ├── ANECompileLimitStress.m/.h   # Phase 1.5: 239 model load limit stress test
│   ├── ANERMSNorm.m/.h              # Phase 2: RMSNorm forward + backward correctness
│   ├── ANELinear.m/.h               # Phase 2: Linear (1x1 conv) correctness
│   ├── ANEAttention.m/.h            # Phase 2: Full SDPA attention correctness
│   ├── ANEFFN.m/.h                  # Phase 2: SwiGLU FFN correctness
│   ├── ANEBackward.m/.h             # Phase 2: FFN + attention backward correctness
│   └── ANETrainStep.m/.h            # Phase 2.3: Single-layer training proof
│
├── Config
│   ├── project.yml                  # Xcode project config (xcodegen)
│   └── Info.plist                   # App config + BGTask permissions
│
└── Data
    ├── tinystories_data00.bin       # Pre-tokenized TinyStories (977KB, ~488K tokens)
    ├── tokenizer.bin                # BPE tokenizer vocabulary (424KB, 32K tokens)
    └── IdentityConv.mlmodelc/       # CoreML identity conv test model
```

## Documentation

| Document | Content |
|:--|:--|
| [INTEGRATION.md](INTEGRATION.md) | **How to integrate ANE training into your own app** — API reference, code examples, data format |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, data flow, layer diagrams, memory layout, ANE hardware specs |
| [iOS_ANE_RESEARCH.md](iOS_ANE_RESEARCH.md) | Complete reverse engineering log (15 steps), all findings with data |
| [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) | **Comprehensive benchmark results** — ANE vs GPU vs CPU, power, efficiency, thermal |
| [ROADMAP_iOS.md](ROADMAP_iOS.md) | Phase 1-3 task status, what's done, what's open |

---

## Key Discoveries

During development, several low-level ANE behaviors were discovered that are not covered by Apple's public documentation. Highlights:

- **16 KB IOSurface Minimum** — silent eval failure below this size
- **239 Concurrent Model Limit** (vs 119 on macOS), full reclaim on unload
- **matmul shape dependency** — works in `[1,H,S,D]`, fails with singleton dims
- **In-process compiler** on iOS (46MB framework, no XPC like macOS)

Full details: [iOS_ANE_RESEARCH.md](iOS_ANE_RESEARCH.md) (14 research steps documented)

---

## Why Private APIs? Apple's ANE Training History

> [!NOTE]
> **Correction (March 2026):** Earlier versions of this README stated that Apple's ANE APIs were "undocumented" and that Apple "never exposed the ANE for training." This was incorrect. Apple provided a public, documented ANE training API via the [MLCompute framework](https://developer.apple.com/documentation/mlcompute) (2020) — discoverable with a simple search for "ANE Apple" on developer.apple.com. We initially missed this and have corrected all documentation. We apologize for the inaccuracy.

Apple once provided **public APIs for ANE training** — then removed them without replacement:

| Year | Framework | ANE Training Support | Status |
|:--|:--|:--|:--|
| 2020 | **MLCompute** | `MLCDevice.ane()` + `MLCTrainingGraph` — full ANE training | **Deprecated** (no replacement) |
| 2019+ | **CoreML** | `MLUpdateTask` — only last FC/Conv layers, no transformer support | Active but limited |
| 2024 | **MPSGraph** | GPU training with auto-diff, no ANE target | Active (GPU only) |
| 2025 | **MLX** | Training on Apple Silicon (GPU), no ANE target on iOS | Active (Mac-focused) |
| 2026 | **Core AI** (iOS 27) | Unknown — may restore ANE training access | Announced, not released |

**The gap:** Apple deprecated `MLCompute` (the only public ANE training API) without providing an alternative. CoreML's on-device training is limited to fine-tuning last layers and cannot backpropagate through full transformers. This project fills that gap by accessing the ANE directly via private APIs.

**Apple's deprecated MLCompute training API (was fully documented):**

| API | What it did | Documentation |
|:----|:------------|:-------------|
| `MLCDevice.ane()` | Select ANE as compute target | [Link](https://developer.apple.com/documentation/mlcompute/mlcdevice/ane()) |
| `MLCTrainingGraph` | Full training graph with auto-differentiation | [Link](https://developer.apple.com/documentation/mlcompute/mlctraininggraph/) |
| `MLCTrainingGraph.compile(options:device:)` | Compile graph for ANE/GPU/CPU | [Link](https://developer.apple.com/documentation/mlcompute/mlctraininggraph/compile(options:device:)) |
| `MLCTrainingGraph.optimizer` | Set optimizer (Adam, SGD) | [Link](https://developer.apple.com/documentation/mlcompute/mlctraininggraph/optimizer) |
| `MLCTrainingGraph.gradientData(forParameter:layer:)` | Access gradients per parameter | [Link](https://developer.apple.com/documentation/mlcompute/mlctraininggraph/gradientdata(forparameter:layer:)) |
| `MLCTrainingGraph.stopGradient(for:)` | Exclude tensors from gradient computation | [Link](https://developer.apple.com/documentation/mlcompute/mlctraininggraph/stopgradient(for:)) |
| `MLCSGDOptimizer` / `MLCAdamOptimizer` | Built-in optimizers | [Link](https://developer.apple.com/documentation/mlcompute/mlcsgdoptimizer) |
| `MLCDeviceType.ane` | ANE device type enum | [Link](https://developer.apple.com/documentation/mlcompute/mlcdevicetype/ane) |

This was a complete training framework — auto-differentiation, optimizer, gradient access, ANE support. All deprecated, no replacement.

**Other relevant Apple documentation:**
- [`com.apple.developer.coreml.neural-engine-access`](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.coreml.neural-engine-access) — entitlement for CoreML ANE access (inference only)
- [`MLNeuralEngineComputeDevice`](https://developer.apple.com/documentation/coreml/mlneuralenginecomputedevice) — read-only ANE device info
- [`Personalizing a Model with On-Device Updates`](https://developer.apple.com/documentation/coreml/personalizing-a-model-with-on-device-updates) — CoreML fine-tuning (last layers only)

**Legal basis:** Reverse engineering for interoperability under [DMCA §1201(f)](https://www.law.cornell.edu/uscode/text/17/1201) and fair use doctrine ([Sega v. Accolade, 1992](https://en.wikipedia.org/wiki/Sega_v._Accolade)). No Apple proprietary code is included in this repository.

---

## Related Projects

- [ANE-Training (macOS)](https://github.com/slavko-at-klincov-it/ANE-Training) — The macOS counterpart with full Stories-110M training
- [maderix/ANE](https://github.com/maderix/ANE) — ANE reverse engineering on M4 Mac
- [ANEMLL](https://github.com/Anemll/Anemll) — ANE inference for LLMs via CoreML (no training, public APIs)
- [Orion Paper](https://arxiv.org/html/2603.06728v1) — Academic paper on ANE programming
- [Metal FlashAttention](https://github.com/philipturner/metal-flash-attention) — GPU attention with backward pass (used by Draw Things)
- [llama2.c](https://github.com/karpathy/llama2.c) — The model format and tokenizer we use

---

## FAQ

**Q: Will this get my app rejected from the App Store?**
A: Yes — it uses private APIs. Apple's automated review detects private API usage and rejects the app. Distribution works via GitHub (source code), TestFlight, Ad-Hoc signing, or enterprise distribution. For App Store, the training backend would need to be rewritten using MLX or the upcoming Core AI framework (iOS 27).

**Q: Isn't there a public API for ANE training?**
A: There was — Apple's MLCompute framework had `MLCDevice.ane()` with full training graph support. Apple deprecated the entire framework without providing an alternative. CoreML's on-device training only supports fine-tuning the last fully-connected or convolution layers, which is insufficient for transformer training. Our project fills this gap.

**Q: What about Core AI (iOS 27)?**
A: Apple is replacing CoreML with a new "Core AI" framework at WWDC 2026 (June). It may restore public ANE training access — but this is unconfirmed. If it does, this project can be ported to use public APIs.

**Q: Does this work on older iPhones?**
A: Untested, but likely works on any iPhone with an ANE (A11 Bionic / iPhone 8 and later). Performance will vary. The A17 Pro has the most capable ANE (h16 architecture, 35 TOPS).

**Q: Can I train GPT-4 sized models?**
A: No. iPhone RAM limits practical model size to ~150M parameters (6 GB devices) or ~250M (8 GB Pro devices) with Adam optimizer. This is for small personalized models, not foundation models. For larger models, consider LoRA/QLoRA fine-tuning.

**Q: How does this compare to CoreML training?**
A: CoreML's `MLUpdateTask` supports on-device training but only for the last fully-connected and convolution layers. It cannot backpropagate through attention, normalization, or activation layers. This project enables full transformer training through all 12 layers with complete backpropagation on ANE.

**Q: Is the ANE really faster than the GPU for training?**
A: Yes and no. ANE computes individual steps 2.3x faster than GPU (200ms vs 462ms per step). But ANE requires kernel recompilation after weight updates (42% overhead). GPU training has no recompilation but slower per-step compute. Net result: ANE is 50% faster overall (3.25 vs 2.17 steps/s). For power efficiency, ANE inference uses 2.51W vs CPU's 8.65W (3.4x more efficient).

**Q: Can I use my own model architecture?**
A: Yes, if it can be expressed in MIL operations that the ANE supports. You'd need to write custom MIL generators. The existing generators cover the standard transformer architecture (RMSNorm + Multi-Head Attention + SwiGLU FFN).

---

## License

MIT
