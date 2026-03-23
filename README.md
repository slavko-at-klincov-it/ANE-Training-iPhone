# ANE-Training-iPhone

**Train neural networks directly on your iPhone's Neural Engine — no jailbreak, no cloud, no data leaves your device.**

The first open-source project that enables on-device transformer training on Apple's Neural Engine (ANE) via reverse-engineered private APIs. A 110M-parameter language model trains on an iPhone 15 Pro at 2.4 steps/second.

---

## Why This Matters

| | Cloud Training | On-Device (CPU/GPU) | **This Project (ANE)** |
|:--|:-:|:-:|:-:|
| Privacy | Data leaves device | Private | **Private** |
| Cost | $/hour | Free | **Free** |
| Speed (110M model) | Fast | Very slow | **2.4 steps/s** |
| Offline | No | Yes | **Yes** |
| Requires jailbreak | N/A | No | **No** |
| Hardware utilization | GPU cluster | CPU/GPU only | **Dedicated ML chip** |

The Apple Neural Engine is a dedicated chip in every modern iPhone designed for ML inference. It's faster and more power-efficient than CPU/GPU for tensor operations — but Apple never exposed it for training. This project changes that.

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

- **Speed** — 2.4 steps/s for 110M model. A cloud GPU does this 100-1000x faster. This is for personalization, not pre-training.
- **Model Size** — iPhone has ~2-3GB usable RAM. With Adam optimizer, ~110M parameters is the practical limit. Larger models need SGD (no momentum memory) or quantized optimizer states.
- **Recompile Overhead** — Weights are baked into MIL programs. Every weight update requires recompiling 60 ANE kernels (~5s). Dynamic spatial packing (proven at 0.12ms/iter) can eliminate this but isn't yet integrated into the full training loop.
- **Batch Size 1** — Current implementation processes one sequence at a time. Gradient accumulation over 4 steps compensates.
- **No App Store** — Uses private Apple APIs. Distribution via TestFlight, Ad-Hoc, or enterprise signing only. An App Store variant would need a CoreML wrapper.
- **iPhone Only** — Tested on iPhone 15 Pro (A17 Pro). Other A-series chips likely work but are untested. iPad should work identically.
- **No Inference UI** — Training only. A chat interface for the trained model is not yet built.
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
| Training speed | **2.4 steps/s** (~230ms/step) |
| Forward pass (12 layers) | ~24ms |
| Backward pass (12 layers) | ~30ms |
| Peak memory | ~1.4 GB (with Adam) |
| ANE SRAM | ~32 MB |
| Max concurrent ANE models | 239 |

For detailed benchmarks, hardware specs, and memory layout see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Project Structure

```
ANEProbe/
├── Core Infrastructure
│   ├── ANETrainingConfig.h        # ANE runtime, IOSurface I/O, structs
│   ├── ANEStoriesMIL.h            # 7 MIL kernel generators
│   └── ANEStoriesCPUOps.h         # CPU ops (RMSNorm, Adam, loss)
│
├── Training Engine
│   ├── ANETrainingEngine.m/.h     # Full 12-layer training loop
│   ├── ANEDataPipeline.m/.h       # Token loading, embedding, loss
│   └── ANECheckpoint.m/.h         # Save/load training state
│
├── System Integration
│   ├── ANEBackgroundTraining.swift # BGProcessingTask scheduling
│   ├── ANEThermal.m/.h            # Thermal monitoring + adaptation
│   ├── ANEThermalMonitor.swift     # SwiftUI thermal state
│   └── ANEDynamicTrain.m/.h       # Dynamic spatial packing (0.12ms)
│
├── UI
│   ├── ANEProbe.swift             # Main app + overnight training
│   └── TrainingDashboardView.swift # Training status dashboard
│
├── Research & Tests
│   ├── ANEDirectTest.m            # Phase 1: ANE access proof
│   ├── ANEWeightTest.m            # Phase 1: Weight update strategies
│   ├── ANERE.m                    # Phase 1.5: Hardware reverse engineering
│   ├── ANERMSNorm.m              # Phase 2: RMSNorm correctness
│   ├── ANELinear.m               # Phase 2: Linear layer correctness
│   ├── ANEAttention.m            # Phase 2: Attention correctness
│   ├── ANEFFN.m                  # Phase 2: FFN correctness
│   ├── ANEBackward.m             # Phase 2: Backward pass correctness
│   └── ANETrainStep.m            # Phase 2.3: Training proof
│
└── Data
    ├── tinystories_data00.bin     # Pre-tokenized training data (977KB)
    └── IdentityConv.mlmodelc/     # CoreML test model
```

## Documentation

| Document | Content |
|:--|:--|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, data flow, layer diagrams, memory layout, ANE hardware specs |
| [iOS_ANE_RESEARCH.md](iOS_ANE_RESEARCH.md) | Complete reverse engineering log (14 steps), all findings with data |
| [ROADMAP_iOS.md](ROADMAP_iOS.md) | Phase 1-3 task status, what's done, what's open |

---

## Key Discoveries

During reverse engineering, several undocumented ANE behaviors were found. Highlights:

- **16 KB IOSurface Minimum** — silent eval failure below this size
- **239 Concurrent Model Limit** (vs 119 on macOS), full reclaim on unload
- **matmul shape dependency** — works in `[1,H,S,D]`, fails with singleton dims
- **In-process compiler** on iOS (46MB framework, no XPC like macOS)

Full details: [iOS_ANE_RESEARCH.md](iOS_ANE_RESEARCH.md) (14 research steps documented)

---

## Related Projects

- [ANE-Training (macOS)](https://github.com/slavko-at-klincov-it/ANE-Training) — The macOS counterpart with full Stories-110M training
- [maderix/ANE](https://github.com/maderix/ANE) — Original ANE reverse engineering
- [Orion Paper](https://arxiv.org/html/2603.06728v1) — Academic paper on ANE programming
- [llama2.c](https://github.com/karpathy/llama2.c) — The model format and tokenizer we use

---

## FAQ

**Q: Will this get my app rejected from the App Store?**
A: Yes — it uses private APIs. For App Store distribution, you'd need to wrap everything in CoreML. The private API approach works for TestFlight, Ad-Hoc, and enterprise distribution.

**Q: Does this work on older iPhones?**
A: Untested, but likely works on any iPhone with an ANE (A11 Bionic / iPhone 8 and later). Performance will vary. The A17 Pro has the most capable ANE (h16 architecture).

**Q: Can I train GPT-4 sized models?**
A: No. iPhone RAM limits practical model size to ~110M parameters with Adam, or ~200M with SGD. This is for small personalized models, not foundation models.

**Q: How does this compare to CoreML training?**
A: CoreML supports on-device training for specific layer types but with significant limitations. This project bypasses CoreML entirely and talks directly to the ANE hardware, enabling full transformer training with any MIL-expressible architecture.

**Q: Is the ANE really faster than the GPU for training?**
A: For the specific operations in transformer training (conv, matmul, softmax, elementwise), the ANE is significantly faster than the iPhone GPU and dramatically faster than the CPU. The ANE is purpose-built for these operations.

**Q: Can I use my own model architecture?**
A: Yes, if it can be expressed in MIL operations that the ANE supports (see Op Coverage table). You'd need to write custom MIL generators. The existing generators cover the standard transformer architecture (RMSNorm + Multi-Head Attention + SwiGLU FFN).

---

## License

MIT
