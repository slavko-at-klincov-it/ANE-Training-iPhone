# ANE-Training-iPhone — Architektur

## Übersicht

Training eines 110M-Parameter Transformers direkt auf dem Apple Neural Engine (ANE) eines iPhones, ohne Jailbreak. Das System nutzt Apples private `AppleNeuralEngine.framework` um MIL-Programme (Machine Learning Intermediate Language) zur Laufzeit zu kompilieren und auf der ANE auszuführen.

```
┌─────────────────────────────────────────────────────────────┐
│                    Public API (Swift)                         │
│  ANETrainer.swift — addText · train · scheduleOvernight      │
│  ANETokenizer — text ↔ tokens · ANEInference — generate      │
├─────────────────────────────────────────────────────────────┤
│                     SwiftUI App Layer                         │
│  ANEProbe.swift · TrainingDashboardView · BackgroundTraining  │
├─────────────────────────────────────────────────────────────┤
│                   Training Engine (ObjC)                      │
│              ANETrainingEngine.m (1191 Zeilen)                │
│   Forward → Loss → Backward → Adam → Recompile → Repeat      │
├──────────────────┬──────────────────┬───────────────────────┤
│  MIL Generators  │    CPU Ops       │   Infrastructure      │
│  ANEStoriesMIL.h │ ANEStoriesCPUOps │ Checkpoint · Thermal  │
│  7 Kernel-Typen  │ RMSNorm · Adam   │ DataPipeline · BGTask │
├──────────────────┴──────────────────┴───────────────────────┤
│                ANE Runtime Layer (ObjC)                       │
│              ANETrainingConfig.h (392 Zeilen)                 │
│  compile_kern · ane_eval · io_read/write · IOSurface I/O     │
├─────────────────────────────────────────────────────────────┤
│           Apple Private API (AppleNeuralEngine.framework)     │
│  _ANEInMemoryModel · _ANEClient · _ANERequest · _ANEIOSurface│
├─────────────────────────────────────────────────────────────┤
│                    A17 Pro ANE Hardware                        │
│              16 Cores · h16 Architektur · ~32MB SRAM           │
└─────────────────────────────────────────────────────────────┘
```

---

## Datenfluss: Ein Trainingsschritt

```
Token-IDs [256]
     │
     ▼ embed_lookup (CPU)
Aktivierungen x [768, 256] (FP32, channel-first)
     │
     ▼ cvt_f32_f16 + IOSurfaceLock/Write
IOSurface (FP16, shared memory CPU↔ANE)
     │
     ├─── Layer 0..11 (Forward) ──────────────────────────────┐
     │    ┌─────────────────────────────────────────────┐      │
     │    │ fwdAttn Kernel (ANE)                        │      │
     │    │ RMSNorm → Wq/Wk/Wv Conv → Reshape →        │      │
     │    │ Transpose → Q@K^T → Scale → Mask →          │      │
     │    │ Softmax → @V → Reshape → Wo Conv            │      │
     │    │ Output: [o_out, Q, K, V, attn_out, xnorm]  │      │
     │    └─────────────────────────────────────────────┘      │
     │    x2 = x + o_out (CPU, vDSP_vadd)                     │
     │    ┌─────────────────────────────────────────────┐      │
     │    │ fwdFFN Kernel (ANE)                         │      │
     │    │ RMSNorm → W1/W3 Conv → SiLU → Gate →       │      │
     │    │ W2 Conv                                     │      │
     │    │ Output: [ffn_out, h1, h3, silu, x2norm]    │      │
     │    └─────────────────────────────────────────────┘      │
     │    x = x2 + ffn_out (CPU)                               │
     └────────────────────────────────────────────────────────┘
     │
     ▼ rmsnorm (CPU, vDSP)
x_final [768, 256]
     │
     ▼ cblas_sgemm: logits = embed^T @ x_final
logits [32000, 256]
     │
     ▼ cross_entropy_loss (CPU, vDSP + vvexpf)
loss (float) + dlogits [32000, 256]
     │
     ▼ cblas_sgemm: dy = embed @ dlogits
     │
     ├─── Layer 11..0 (Backward) ─────────────────────────────┐
     │    ┌─────────────────────────────────────────────┐      │
     │    │ ffnBwd Kernel (ANE)                         │      │
     │    │ W2^T Conv → SiLU' → W1^T/W3^T Conv → dx    │      │
     │    └─────────────────────────────────────────────┘      │
     │    dW_ffn: cblas_sgemm (CPU, async dispatch)            │
     │    rmsnorm2_bwd (CPU)                                   │
     │    ┌─────────────────────────────────────────────┐      │
     │    │ sdpaBwd1 Kernel (ANE)                       │      │
     │    │ Wo^T Conv → Recompute Softmax → dV, dProbs  │      │
     │    └─────────────────────────────────────────────┘      │
     │    ┌─────────────────────────────────────────────┐      │
     │    │ sdpaBwd2 Kernel (ANE, weight-free)          │      │
     │    │ Softmax Gradient → dQ, dK via Matmul        │      │
     │    └─────────────────────────────────────────────┘      │
     │    dW_qkv: cblas_sgemm (CPU, async)                     │
     │    ┌─────────────────────────────────────────────┐      │
     │    │ qkvBwd Kernel (ANE)                         │      │
     │    │ Wq^T/Wk^T/Wv^T Conv → dx                   │      │
     │    └─────────────────────────────────────────────┘      │
     │    rmsnorm1_bwd (CPU)                                   │
     │    dy = dx_rms1 + dx2 (Residual)                        │
     └────────────────────────────────────────────────────────┘
     │
     ▼ embed_backward (CPU)
     │
     ▼ Alle ACCUM_STEPS Schritte:
       adam_update (CPU) → Weights aktualisiert
       free_kern → compile_kern (neue Weights baked in MIL)
```

---

## Datei-Architektur

### Schicht 1: ANE Runtime (Basis)

| Datei | Zeilen | Funktion |
|:--|:-:|:--|
| **ANETrainingConfig.h** | 392 | Zentrales Config-Header. Definiert DIM/HIDDEN/HEADS/SEQ/NLAYERS/VOCAB. Enthält ANE-Klassen-Init (`ane_init`), IOSurface-Helpers (`make_surface`, `io_write_fp16`, `io_read_fp16`, `io_copy`), Kernel-Compile (`compile_kern`, `compile_kern_mil_w`), Eval (`ane_eval`), Blob-Builder (`build_blob`, `build_blob_t`), NEON FP16↔FP32 Konvertierung. Alle Struct-Definitionen: `Kern`, `LayerWeights`, `LayerAdam`, `LayerActs`, `LayerGrads`, `LayerKernels`. |

**Wie ANE-Zugriff funktioniert:**
```
1. dlopen("AppleNeuralEngine.framework")          → Private Framework laden
2. NSClassFromString("_ANEInMemoryModelDescriptor") → Klassen per Runtime holen
3. modelWithMILText:weights:optionsPlist:           → MIL-Text → Modell-Descriptor
4. inMemoryModelWithDescriptor:                     → Descriptor → In-Memory-Modell
5. compileWithQoS:options:error:                    → MIL → ANE-Bytecode (on-device!)
6. loadWithQoS:options:error:                       → Bytecode → ANE-Hardware laden
7. evaluateWithQoS:options:request:error:           → Input IOSurface → ANE → Output IOSurface
```

**IOSurface** = Shared-Memory-Buffer zwischen CPU und ANE. Kein Kopieren nötig — beide sehen denselben physischen Speicher. Minimum 16KB (ANE DMA-Controller Alignment).

### Schicht 2: MIL-Generatoren

| Datei | Zeilen | Funktion |
|:--|:-:|:--|
| **ANEStoriesMIL.h** | 277 | 7 MIL-Programm-Generatoren. Erzeugen MIL-Text (Apples ML-IR) zur Laufzeit mit Modell-Dimensionen eingesetzt. |

**Die 7 Kernel-Typen:**

| Kernel | Input | Output | Baked Weights | ANE Ops |
|:--|:--|:--|:--|:--|
| `gen_sdpa_fwd_taps` | x [DIM,SEQ] | [6×DIM,SEQ] (mit Taps) | rms_w, Wq, Wk, Wv, Wo, Mask | RMSNorm + 4 Conv + Reshape + Transpose + 2 Matmul + Softmax + Concat |
| `gen_ffn_fwd_taps` | x [DIM,SEQ] | [2DIM+3HID,SEQ] (mit Taps) | rms_w, W1, W3, W2 | RMSNorm + 3 Conv + Sigmoid + Mul + Concat |
| `gen_ffn_bwd` | [DIM+2HID,SEQ] | [DIM+2HID,SEQ] | W2^T, W1^T, W3^T | 3 Conv + SiLU' + Slice + Concat |
| `gen_sdpa_bwd1` | [4DIM,SEQ] | [DIM+2SC,SEQ] | Wo^T, Mask | Conv + Reshape + 3 Matmul + Softmax + Concat |
| `gen_sdpa_bwd2` | [2SC+2DIM,SEQ] | [2DIM,SEQ] | (keine) | Reshape + 2 Matmul + Reduce_sum + Concat |
| `gen_qkvb` | [3DIM,SEQ] | [DIM,SEQ] | Wq^T, Wk^T, Wv^T | 3 Conv + 2 Add + Slice |
| `get_mask_blob` | — | — | Causal Mask [SEQ,SEQ] | (Daten, kein Kernel) |

**MIL-Format** (Machine Learning Intermediate Language):
```
program(1.3)
[buildInfo = dict<string, string>(...)]
{
    func main<ios18>(tensor<fp16, [1, 768, 1, 256]> x) {
        tensor<fp16, [768,768,1,1]> W = const()[val=...BLOBFILE(...)];
        tensor<fp16, [1,768,1,256]> y = conv(weight=W, x=x)[name="out"];
    } -> (y);
}
```

Weights werden als BLOBFILE in das MIL eingebettet ("baked weights"). Bei jedem Weight-Update muss der Kernel neu kompiliert werden (~20ms pro Kernel).

### Schicht 3: CPU-Operationen

| Datei | Zeilen | Funktion |
|:--|:-:|:--|
| **ANEStoriesCPUOps.h** | 133 | Operationen die auf der CPU laufen (vDSP/Accelerate). |

| Op | Warum CPU? |
|:--|:--|
| `rmsnorm` / `rmsnorm_bwd` | Reduce + Elementweise — könnte auf ANE, aber CPU ist schnell genug und spart Kernel-Slots |
| `adam_update` | Rein elementweise auf FP32 Weights — kein ANE-Vorteil |
| `cross_entropy_loss` | Softmax über 32K Vocab — zu groß für ANE IOSurface |
| `embed_lookup` / `embed_backward` | Scatter/Gather — nicht ANE-kompatibel |
| Classifier (cblas_sgemm) | 32K×768 Matmul — zu groß für ANE SRAM (~32MB) |

### Schicht 4: Training Engine

| Datei | Zeilen | Funktion |
|:--|:-:|:--|
| **ANETrainingEngine.m** | 1191 | Haupttrainingsschleife + 8h-Overnight-Training. Portiert von macOS `train_large.m`. |
| **ANETrainingEngine.h** | 30 | Public C API: init/step/save/free |

**ANETrainState** Struktur:
```c
struct ANETrainState {
    LayerWeights lw[12];      // 12 Layer × (Wq,Wk,Wv,Wo,W1,W2,W3,rms_att,rms_ffn)
    LayerAdam la[12];         // Adam m/v State pro Weight-Matrix
    LayerActs acts[12];       // Gespeicherte Aktivierungen für Backward
    LayerGrads grads[12];     // Gradient-Akkumulatoren
    LayerKernels kern[12];    // Kompilierte ANE-Kernel (je 5 weight-bearing)
    Kern *sdpaBwd2[12];       // Weight-freie Kernel (einmal kompiliert)
    float *embed, *rms_final; // Globale Weights
    uint16_t *token_data;     // mmap'd Trainingsdaten
    dispatch_queue_t dw_q;    // Async dW Berechnung (cblas)
    int step, adam_t;
    float lr, last_loss;
};
```

**Compile-Strategie:**
- 5 weight-bearing Kernel pro Layer × 12 Layer = 60 Kernel
- 12 sdpaBwd2 (weight-free, einmal kompiliert) = 12 Kernel
- Total: 72 gleichzeitig geladene Kernel (von max 239)
- Alle `ACCUM_STEPS=4` Steps: Gradients mitteln, Adam Update, alle 60 Kernel neu kompilieren (~5s)
- Kein `exec()` wie auf macOS — stattdessen `free_kern` + `compile_kern` Recycling

### Schicht 5: Public API

| Datei | Zeilen | Funktion |
|:--|:-:|:--|
| **ANETrainer.swift** | ~350 | High-Level Swift API. `addText()` tokenisiert + speichert, `train()` startet Training, `scheduleOvernight()` registriert BGProcessingTask. ObservableObject für SwiftUI. |
| **ANETokenizer.m/.h** | ~350 | BPE Tokenizer (llama2.c Format, 32K Vocab). `ane_tokenize()` Text→Tokens, `ane_detokenize()` Tokens→Text. FNV-1a Hash für O(1) Lookup. |
| **ANEInference.m/.h** | ~570 | Text-Generierung. 24 ANE-Kernel (Forward-only, kein Backward). Autoregressive Generation mit Temperature Sampling. Lädt BLZT Checkpoint oder llama2.c Weights. |

### Schicht 6: Infrastruktur

| Datei | Zeilen | Funktion |
|:--|:-:|:--|
| **ANECheckpoint.m/.h** | 780 | Checkpoint Save/Load. Atomic Write (.tmp→rename). FNV-1a Checksum. Slot-Rotation. Pretrained Weight Loading (llama2.c Format). Auto-Save bei Background/Thermal Events. |
| **ANEDataPipeline.m/.h** | 480 | Token-Daten laden (mmap), Batch-Prep, Embedding Lookup, Cross-Entropy Loss (vDSP), Gradient-Berechnung. |
| **ANEThermal.m/.h** | 320 | Thermal-State Monitoring (`ProcessInfo.thermalState`), ANE Throughput Monitor (Rolling Average), Adaptive Training Controller (Delay/Pause bei Throttle). |
| **ANEDynamicTrain.m/.h** | 530 | Dynamic Spatial Packing — Weights im IOSurface statt baked in MIL. 0.119ms/iter statt 22.5ms mit Recompile. Per-Channel Scale funktioniert, Full Matrix Matmul scheitert am ANE-Compiler. |

### Schicht 7: Swift UI / App Layer

| Datei | Zeilen | Funktion |
|:--|:-:|:--|
| **ANEProbe.swift** | 320 | Haupt-App. SwiftUI View mit Probe + Overnight Training. `isIdleTimerDisabled` für Nacht-Training. |
| **ANEBackgroundTraining.swift** | 280 | `BGProcessingTask` Manager. `TrainingSession` (ObservableObject). Scheduling, Expiration Handler, Checkpoint bei Kill. |
| **ANEThermalMonitor.swift** | 90 | SwiftUI-Observable Thermal Monitor. Maps `thermalState` → `TrainingPolicy`. |
| **TrainingDashboardView.swift** | 100 | SwiftUI Dashboard: Status, Loss, Steps, Thermal, Start/Stop. |

### Schicht 8: Forschung / Tests (nicht produktiv)

| Datei | Funktion |
|:--|:--|
| ANEDirectTest.m | Phase 1: MIL Compile + Eval Proof |
| ANEWeightTest.m | Phase 1: Recompile vs Dynamic Packing |
| ANERE.m | Phase 1.5: SRAM Probe, Op Coverage, Perf Stats, Compile Limits |
| ANEReduceDebug.m | Phase 1.5: Reduce-Op 16KB IOSurface Bug |
| ANECompileLimitStress.m | Phase 1.5: 239 Model Load Limit |
| ANERMSNorm.m | Phase 2: RMSNorm Forward+Backward Korrektheit |
| ANELinear.m | Phase 2: Linear (1x1 Conv) Forward+Backward |
| ANEAttention.m | Phase 2: Full SDPA Attention Forward |
| ANEFFN.m | Phase 2: SwiGLU FFN Forward |
| ANEBackward.m | Phase 2: FFN + SDPA Backward |
| ANETrainStep.m | Phase 2.3: Training-Step Proof (Loss sinkt) |

---

## Memory Layout

```
iPhone 15 Pro: 8GB RAM, ~2-3GB App-Limit

Weights (FP32):
  12 Layer × (4×768² + 2×768×2048 + 2048×768 + 2×768) = ~85M params
  + Embedding: 32000×768 = 24.6M params
  + rms_final: 768 params
  Total: ~110M params × 4B = 440MB

Adam State (FP32):
  m + v für jeden Parameter = 2 × 440MB = 880MB

Aktivierungen (FP32):
  12 Layer × ~12 Buffers × 768×256×4B ≈ 90MB

ANE Kernel IOSurfaces (FP16):
  72 Kernel × (Input + Output) × variable = ~50MB

Token Data (mmap, nicht im RAM-Budget):
  tinystories_data00.bin = 977KB (mmap'd)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Peak: ~1.4GB (mit Adam)
         ~560MB (ohne Adam, nur SGD)
```

---

## ANE Hardware-Eigenschaften (A17 Pro, h16)

| Eigenschaft | Wert | Quelle |
|:--|:--|:--|
| Architektur | h16 (gleiche Gen wie M4) | `_ANEDeviceInfo.aneArchitectureType` |
| Cores | 16 | `numANECores` |
| Units | 1 | `numANEs` |
| Board Type | 208 | `aneBoardType` |
| SRAM | ~32MB | Benchmark (Cliff bei 32MB Weights) |
| Optimal Weight Size | ≤8MB pro Kernel | Benchmark (Peak TFLOPS) |
| Max Loaded Models | 239 | Stress Test |
| Unload Reclaim | Vollständig | Stress Test |
| IOSurface Minimum | 16KB | Bug-Analyse |
| Idle Power | 0mW (Hard Power Gating) | `isExcessivePowerDrainWhenIdle` |
| QoS Background | 9 | `_ANEQoSMapper` |
| Compiler | In-Process (46MB) | Framework-Analyse |
| VM | Nein (direkte Hardware) | `isVirtualClient = false` |

---

## Unterstützte MIL-Operationen auf ANE

| Kategorie | Funktioniert | Scheitert |
|:--|:--|:--|
| **Elementweise** | add, sub, mul, real_div | — |
| **Aktivierungen** | relu, tanh, sigmoid, silu, softmax | gelu (→ sigmoid approx) |
| **Mathematik** | exp, sqrt, pow | log, rsqrt (→ div+sqrt) |
| **Reduktionen** | reduce_mean, reduce_sum, reduce_sum_square | — (braucht ≥16KB IOSurface) |
| **Tensor-Ops** | transpose, reshape, slice_by_size | — |
| **Matmul** | Funktioniert in [1,H,S,D] (kein Singleton) | Scheitert mit [1,C,1,S] |
| **Conv (1x1)** | Alle Konfigurationen | — |
| **Concat** | Innerhalb großer MIL-Programme | Standalone scheitert |

---

## Benchmarks (iPhone 15 Pro)

### Forward Pass (pro Layer)

| Kernel | ms/eval | Anmerkung |
|:--|:-:|:--|
| Attention (full SDPA) | 0.604 | QKV + Reshape + SDPA + Wo |
| FFN (SwiGLU) | 0.451 | W1/W3 + SiLU + Gate + W2 |
| RMSNorm | 0.739 | Reduce + Pow + Mul |
| **Gesamt pro Layer** | **~2ms** | Forward only |

### Backward Pass (pro Layer)

| Kernel | ms/eval |
|:--|:-:|
| FFN bwd | 0.734 |
| SDPA bwd1 | 0.451 |
| SDPA bwd2 | ~0.5 |
| QKV bwd | ~0.7 |
| **Gesamt pro Layer** | **~2.5ms** |

### Training Step (12 Layer)

| Phase | Zeit |
|:--|:--|
| Forward (12 Layer ANE) | ~24ms |
| Loss + Classifier (CPU cblas) | ~20ms |
| Backward (12 Layer ANE + CPU dW) | ~30ms |
| **Total pro Step** | **~165ms** |
| Adam Update + Recompile (alle 4 Steps) | ~900ms |
| **Amortisiert pro Step** | **~230ms** |

### Dynamic Spatial Packing

| Metrik | Recompile | Dynamic Packing |
|:--|:-:|:-:|
| ms/weight-update | 22.5 | **0.119** |
| Speedup | 1x | **189x** |
| Compile nötig | Ja (jedes Update) | Nein (einmal) |
