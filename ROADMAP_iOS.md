# iOS ANE Training — Roadmap
## iPhone 15 Pro (A17 Pro, h16) | Stand: 2026-03-19

---

## Phase 1: Direct API Proof of Concept

### 1.1 MIL-Compile direkt auf dem iPhone ✅ DONE (2026-03-19)
- [x] Portiere den macOS `_ANEInMemoryModelDescriptor.modelWithMILText:weights:optionsPlist:` Pfad
- [x] Erstelle MIL-Text für Identity-Conv in der iOS-App
- [x] Kompiliere direkt auf dem iPhone via `_ANEInMemoryModel.compileWithQoS:options:error:`
- [x] Lade das Modell via `loadWithQoS:options:error:`
- [x] Evaluiere via `evaluateWithQoS:options:request:error:`
- [x] Vergleiche Performance mit CoreML-Pfad
- **Ergebnis**: Compile 21ms, Load 2.6ms, Eval 0.308ms/eval, 16384/16384 korrekt

### 1.2 IOSurface I/O auf iOS ✅ DONE (2026-03-19)
- [x] Erstelle IOSurface via `IOSurfaceCreate` (wie macOS)
- [x] Schreibe FP16-Daten in IOSurface
- [x] Lese Ergebnis aus Output-IOSurface
- [x] Verifiziere Korrektheit (Identity-Conv: Input == Output) → 100%

### 1.3 A17 Pro ANE Benchmark ✅ DONE (2026-03-19)
- [x] Benchmark mit verschiedenen Kanalzahlen (256, 512, 1024, 2048, 4096)
- [x] Benchmark mit verschiedenen Spatial-Größen (16, 32, 64, 128, 256)
- [ ] INT8 vs FP16 Vergleich (noch offen)
- [x] Peak TFLOPS ermitteln → **~1.4 TFLOPS** single kernel (2048ch sp64)
- [x] QoS=9 (Background) verwendet — identisch wie macOS
- **Ergebnis**: Sweet Spot bei 2048ch/8MB, 4096ch/32MB überschreitet SRAM

### 1.4 Weight Update Test ✅ DONE (2026-03-19)
- [x] Recompile mit neuen Weights: FUNKTIONIERT (22.5 ms/cycle)
- [x] Dynamic Spatial Packing (Weights im IOSurface): FUNKTIONIERT (0.308 ms/cycle)
- [x] Dynamic Packing = **73x schneller** als Recompile, kein Compile-Budget-Problem
- **Ergebnis**: Dynamic Spatial Packing ist der bevorzugte Ansatz für Training

---

## Phase 1.5: Training-Critical Reverse Engineering

### 1.5.1 SRAM Boundary Probing ✅ DONE (2026-03-20)
- [x] Exakte SRAM-Größe ermitteln → **Harte Grenze bei ~32MB** (graduelle Degradation ab 8MB)
- [x] Sweet Spot: 2-8MB (0.24-1.43 TFLOPS), 10-25MB noch gut (1.25-2.07 TFLOPS), 32MB = Cliff (0.74 TFLOPS)
- [x] Maximale Layer-Größe für Training: **≤25MB Weight pro Kernel** für volle Performance
- **Ergebnis**: SRAM ist ~32MB, optimal unter 8MB, brauchbar bis 25MB

### 1.5.2 MIL Op Coverage ✅ DONE (2026-03-20)
- [x] **Elementwise**: add, sub, mul, real_div — alle OK (~1.0 ms/eval)
- [x] **Activations**: relu, tanh, sigmoid, silu — alle OK (~1.1 ms/eval)
- [x] **gelu**: FAIL — Workaround: sigmoid(1.702*x)*x
- [x] **Math**: exp, sqrt, pow — OK; **log, rsqrt — FAIL** (Workaround: div+sqrt)
- [x] **Reductions**: reduce_mean/sum/sum_square — OK (braucht min 16KB IOSurface)
- [x] **Reshape/Transpose**: OK (~0.4 ms/eval)
- [x] **matmul**: FAIL — muss über Conv (1x1) abgebildet werden
- [x] **softmax**: OK nativ! (~0.38 ms/eval)
- [x] **concat**: FAIL — IOSurface-level manuell
- [x] **slice_by_size**: OK
- **Ergebnis**: Alle kritischen Training-Ops verfügbar (mit Workarounds für gelu, log, rsqrt, matmul)

### 1.5.3 Performance Stats Extraction 🔄 PARTIAL (2026-03-20)
- [x] `_ANEPerformanceStats` Methoden analysiert: hwExecutionTime, perfCounterData, performanceCounters
- [x] `_ANEPerformanceStatsIOSurface` braucht `objectWithIOSurface:statType:` (nicht alloc/init)
- [ ] IOSurface + statType erstellen und echte HW-Metriken auslesen
- [ ] `perfStatsMask` auf _ANEInMemoryModel setzen und Auswirkung testen

### 1.5.4 Compile Limit Investigation ✅ DONE (2026-03-21)
- [x] **239 gleichzeitig geladene Modelle** — dann `Program load failure (0x50004)`
- [x] iOS-Limit ist ~2x macOS (~119)
- [x] **Unload gibt Slots vollständig frei** — 50 entladen, 50 neue geladen: FULL RECLAIM
- [x] Memory pro Modell: ~322 KB, Compile+Load: ~21.6 ms
- **Ergebnis**: 239 Load-Limit ist kein Blocker — Unload/Reclaim funktioniert perfekt

### 1.5.5 INT8 vs FP16 (aus Phase 1.3 übernommen)
- [ ] INT8-MIL erstellen und testen
- [ ] Performance-Vergleich INT8 vs FP16
- [ ] Relevanz für Mixed-Precision Training

### 1.5.6 Reduce-Op Fix ✅ DONE (2026-03-20)
- [x] **Root Cause**: IOSurface muss mindestens **16KB** groß sein, sonst eval fail
- [x] reduce_mean, reduce_sum, reduce_sum_square — alle funktionieren mit 16KB minimum
- [x] Beide Achsen funktionieren: axis=-1 (spatial) und axis=1 (channel)
- [x] keep_dims=true und keep_dims=false beide OK
- [x] Fix: `re_surface()` erzwingt jetzt 16KB minimum
- **Ergebnis**: Alle Reduce-Ops funktionieren — RMSNorm ist möglich!

---

## Phase 2: Training Loop

### 2.1 Forward Pass auf ANE ✅ DONE (2026-03-21)
- [x] RMSNorm forward: PASS (0.739 ms, max_err=0.003777)
- [x] Linear (1x1 Conv) forward: PASS (alle 3 Konfigurationen: 768→768, 768→2048, 2048→768)
- [x] Attention (full SDPA) forward: PASS (0.604 ms, max_err=0.000764)
- [x] FFN (SwiGLU) forward: PASS (0.451 ms, max_err=0.005595)
- [x] Alle Kernel-Korrektheitstest bestanden (vs CPU-Referenz)
- **Ergebnis**: Alle 4 Layer-Typen laufen korrekt auf ANE

| Kernel | ANE ms/eval | Max Error | Status |
|--------|:-----------:|:---------:|:------:|
| RMSNorm fwd | 0.739 | 0.003777 | PASS |
| Linear 768→768 | 0.730 | 0.000896 | PASS |
| Linear 768→2048 | 0.880 | 0.001018 | PASS |
| Linear 2048→768 | 0.997 | 0.001109 | PASS |
| Attention full | 0.604 | 0.000764 | PASS |
| FFN SwiGLU | 0.451 | 0.005595 | PASS |

### 2.2 Backward Pass ✅ DONE (2026-03-21)
- [x] RMSNorm backward auf ANE: PASS (0.305 ms, max_err=0.000446)
- [x] Linear backward auf ANE: PASS (alle Konfigurationen)
- [x] FFN backward auf ANE: PASS (0.734 ms, max_err=0.001958)
- [x] Attention backward (SDPA bwd1) auf ANE: PASS (0.451 ms)
- [x] QKV backward via Linear bwd: bereits bewiesen
- [ ] NaN/Inf Detection einbauen (nice-to-have)

| Kernel | ANE ms/eval | Max Error | Status |
|--------|:-----------:|:---------:|:------:|
| RMSNorm bwd | 0.305 | 0.000446 | PASS |
| Linear bwd (alle) | 0.74-0.99 | 0.001502 | PASS |
| FFN bwd | 0.734 | 0.001958 | PASS |
| SDPA bwd1 | 0.451 | — | PASS |

### 2.3 Training Step ✅ DONE (2026-03-21)
- [x] Forward (ANE) → Loss (CPU) → Backward (CPU) → Weight Update (SGD) → Recompile
- [x] **Loss sinkt monoton** über 10 Schritte: 0.2217 → 0.2196
- [x] 23.1 ms/step (20ms Compile + 0.18ms ANE Eval + CPU-Gradient)
- [x] **TRAINING AUF iPHONE ANE FUNKTIONIERT!**
- [ ] Messe ms/step für Stories-110M Architektur (Full-Scale)
- [ ] Vergleiche mit macOS Training-Performance
- [ ] Dynamic Spatial Packing statt Recompile für Speed (0.3ms statt 20ms)

### 2.4 Memory Management
- [ ] iOS Memory Limits ermitteln (typisch 2-3GB für Foreground-App)
- [ ] Maximale Modellgröße bestimmen die in den Speicher passt
- [ ] Gradient Checkpointing wenn nötig

---

## Phase 3: Background Training & Personal AI

### 3.1 Background Training
- [ ] BGProcessingTaskRequest für Nacht-Training implementieren
- [ ] Thermal-State Monitoring (`ProcessInfo.thermalState`)
- [ ] Adaptive Batch-Size bei Thermal Throttling
- [ ] Automatic Checkpointing (alle N Steps)
- [ ] Resume nach App-Kill

### 3.2 Data Pipeline
- [ ] Tokenizer auf iOS (SentencePiece oder BPE)
- [ ] Lokale Daten laden (Dateien, Notizen, etc.)
- [ ] Pre-Tokenisierung im Background

### 3.3 Inference Server
- [ ] Trainiertes Modell für Inference laden
- [ ] Chat-Interface in der App
- [ ] Token-Streaming

### 3.4 App Store Variante (optional)
- [ ] CoreML-Wrapper statt Private API
- [ ] Alle privaten API-Aufrufe hinter Feature-Flag
- [ ] App Review Guidelines einhalten

---

## Aktueller Status

**Phase 1 + 1.5 + 2 (Kernels + Training Step) KOMPLETT. Training auf iPhone ANE bewiesen! Phase 3 (Background Training & Personal AI) ist der nächste Schritt.**
