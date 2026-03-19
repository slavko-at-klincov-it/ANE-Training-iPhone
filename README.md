# ANE-Training-iPhone

Direct Apple Neural Engine access on iOS — without jailbreak.

This project proves that the private ANE API (`AppleNeuralEngine.framework`) is fully accessible from a normal iOS app sandbox, with 100% API compatibility to macOS.

Built on top of the macOS ANE reverse engineering work at [ANE-Training](https://github.com/slavko-at-klincov-it/ANE-Training).

## Key Findings

| Finding | Detail |
|---------|--------|
| **42 ANE classes** available at runtime | All macOS classes + 7 iOS-exclusive |
| **100% method match** | Every selector identical to macOS |
| **A17 Pro = h16 architecture** | Same generation as M4 (not M3 Pro!) |
| **Direct hardware client** | `_ANEClient.sharedConnection` works from app sandbox |
| **No jailbreak needed** | Standard developer-signed app |
| **MIL compile on-device** | 21ms compile, 2.6ms load |
| **Dynamic weight updates** | 0.3ms/cycle without recompile |

## Architecture

```
MIL Text (in app) → _ANEInMemoryModelDescriptor → _ANEInMemoryModel
  → compileWithQoS: (21ms) → loadWithQoS: (2.6ms)
  → evaluateWithQoS: (0.3ms) → IOSurface Output
```

## Benchmark (iPhone 15 Pro, A17 Pro)

### Single Convolution Throughput

| Config | Weight | ms/eval | TFLOPS |
|--------|:------:|:-------:|:------:|
| 256x256 sp=64 | 0.1 MB | 0.319 | 0.03 |
| 512x512 sp=64 | 0.5 MB | 0.291 | 0.12 |
| 1024x1024 sp=64 | 2.0 MB | 0.300 | 0.45 |
| 2048x2048 sp=64 | 8.0 MB | 0.382 | **1.40** |
| 4096x4096 sp=64 | 32.0 MB | 2.971 | 0.72 |

### Weight Update Strategies

| Approach | ms/cycle | Compile budget |
|----------|:--------:|:--------------:|
| Recompile | 22.5 ms | ~119 per process |
| **Dynamic Spatial Packing** | **0.3 ms** | **Unlimited** |

Dynamic packing is **73x faster** — weights live in the input IOSurface, no recompile needed.

## A17 Pro Hardware Identity

```
Architecture:     h16 (same generation as M4!)
Cores:            16
Units:            1
Board Type:       208
Virtual Machine:  NO
Power Idle:       0 mW (hard power gating)
```

## iOS vs macOS Comparison

| Aspect | macOS | iOS |
|--------|-------|-----|
| Classes | 35 | 42 (35 + 7 iOS-exclusive) |
| Methods | 100% match | 100% match |
| Compiler | Via XPC service | **In-process** (46 MB framework) |
| QoS values | 0/9/17/21/25/33 | Identical |
| ANE access | Direct | Direct (from app sandbox!) |

## Project Structure

```
ANEProbe/                  # iOS app for on-device testing
  ANEProbe.swift           # SwiftUI app with runtime introspection
  ANEDirectTest.m          # Direct MIL compile + eval + benchmark
  ANEWeightTest.m          # Weight update tests (recompile vs dynamic)
  project.yml              # Xcode project config (xcodegen)
iOS_ANE_RESEARCH.md        # Complete research log with all findings
ROADMAP_iOS.md             # Phase 1-3 roadmap with status
create_test_model.py       # CoreML test model generator
```

## Building & Running

```bash
# Prerequisites: Xcode, xcodegen, iPhone with Developer Mode
brew install xcodegen

# Generate Xcode project
cd ANEProbe && xcodegen generate

# Build and deploy
xcodebuild build \
  -project ANEProbe.xcodeproj \
  -scheme ANEProbe \
  -destination 'id=YOUR_DEVICE_ID' \
  -allowProvisioningUpdates

# Install on device
xcrun devicectl device install app \
  --device YOUR_DEVICE_UUID \
  path/to/ANEProbe.app
```

## Roadmap

- [x] **Phase 1**: Direct API Proof of Concept (complete)
  - [x] MIL compile on iPhone
  - [x] IOSurface I/O
  - [x] A17 Pro benchmark
  - [x] Weight update (recompile + dynamic packing)
- [ ] **Phase 2**: Training Loop
  - [ ] Forward pass kernels (Linear, RMSNorm, Attention, FFN)
  - [ ] Backward pass on CPU
  - [ ] Full training step
  - [ ] Memory management for iOS
- [ ] **Phase 3**: Background Training & Personal AI
  - [ ] BGProcessingTask for overnight training
  - [ ] Thermal management
  - [ ] Data pipeline & tokenizer
  - [ ] Inference server

## Related Projects

- [ANE-Training (macOS)](https://github.com/slavko-at-klincov-it/ANE-Training) — The macOS counterpart with 76 discovered API classes
- [maderix/ANE](https://github.com/maderix/ANE) — Original reverse engineering project
- [Orion Paper](https://arxiv.org/html/2603.06728v1) — Academic paper on ANE programming

## License

MIT
