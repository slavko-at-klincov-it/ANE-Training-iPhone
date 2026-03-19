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

## Phase 2: Training Loop

### 2.1 Forward Pass auf ANE
- [ ] Portiere MIL-Generation für Linear Layer (1x1 Conv)
- [ ] Portiere MIL-Generation für RMSNorm
- [ ] Portiere MIL-Generation für Attention (mit CPU-Fallback für Masking)
- [ ] Portiere MIL-Generation für FFN (SwiGLU)
- [ ] Teste einzelne Kernel-Korrektheit

### 2.2 Backward Pass auf CPU
- [ ] Portiere Gradient-Berechnung für Linear
- [ ] Portiere Gradient-Berechnung für RMSNorm
- [ ] Portiere Gradient-Berechnung für Attention
- [ ] Portiere Gradient-Berechnung für FFN
- [ ] NaN/Inf Detection einbauen

### 2.3 Training Step
- [ ] Forward (ANE) → Loss (CPU) → Backward (CPU) → Weight Update (CPU) → Recompile
- [ ] Messe ms/step für Stories-110M Architektur
- [ ] Messe ms/step für kleineres Modell (angepasst an iPhone Speicher)
- [ ] Vergleiche mit macOS Training-Performance
- [ ] ~119 Compile-Limit Handling (purgeCompiledModel oder App-Restart)

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

**Phase 1 KOMPLETT. Phase 2 (Training Loop) ist der nächste Schritt.**
