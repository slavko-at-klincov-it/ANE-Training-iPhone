# ANE Benchmark-Ergebnisse & Analyse

Umfassende Performance-, Power- und Effizienz-Analyse des Stories-110M Transformer-Trainings und Inference auf dem iPhone 15 Pro (A17 Pro).

**Datum:** 24. Maerz 2026
**Geraet:** iPhone 15 Pro (iPhone16,1), A17 Pro, 8 GB RAM, 12.5 Wh Akku
**Modell:** Stories-110M (12 Layer, dim=768, hidden=2048, seq=256, vocab=32000, ~110M Parameter)

---

## 1. Inference-Vergleich: ANE vs CPU vs GPU

Jeder Backend lief 4.3 Minuten mit 256-Token Forward Passes. Geraet war vom Strom getrennt fuer Battery-Messung.

### Uebersichtstabelle

| Metrik | ANE | CPU (Accelerate) | GPU (MPS batched) |
|--------|-----|-------------------|-------------------|
| Latency (mean) | 96.9 ms | **73.3 ms** | 270.9 ms |
| Latency (p50) | 97.4 ms | 77.5 ms | 274.6 ms |
| Latency (p95) | 102.6 ms | 81.3 ms | 288.2 ms |
| Latency (p99) | 105.0 ms | 86.2 ms | 295.1 ms |
| Tokens/s | 2,480 | **3,215** | 923 |
| Battery drain (4.3 min) | **0%** | 5% | 5% |
| Power | **2.51 W** (15-min Test) | 8.65 W | 8.65 W |
| Tokens/Joule | **990** | 372 | 107 |
| CPU-Auslastung | **0%** | ~100% (multi-core) | 0.4% |
| Thermal (worst) | **nominal** | fair | fair |
| Peak RAM | 682 MB | 701 MB | 1,734 MB |
| Forward Passes | 2,528 | 3,277 | 941 |

### Per-Component Breakdown

#### ANE Forward Pass (96.9 ms)

```
IOSurface FP32<->FP16:  49.2 ms  (50.8%)  <-- GROESSTER BOTTLENECK
ANE Kernel Evals:       38.4 ms  (39.6%)
Classifier (cblas):      5.8 ms  ( 5.9%)
CPU Ops (residual etc):  2.0 ms  ( 2.0%)
Embedding Lookup:        1.6 ms  ( 1.6%)
```

**Erkenntnis:** Die ANE-Hardware selbst ist schnell (38.4 ms fuer 12 Layer = 3.2 ms/Layer). Der Bottleneck ist die Daten-Konvertierung zwischen CPU (FP32) und ANE (FP16) via IOSurface. Jeder der 24 Kernel-Aufrufe braucht Lock + FP32->FP16 Convert + Copy + Unlock fuer Input, und das Gleiche rueckwaerts fuer Output. Das sind 48 IOSurface-Operationen pro Forward Pass.

#### CPU Forward Pass (73.3 ms)

```
Matmuls (cblas_sgemm):  44.3 ms  (61.1%)
Attention (SDPA):       17.2 ms  (23.7%)
Norms + Activations:     8.0 ms  (11.1%)
Classifier:              2.5 ms  ( 3.4%)
Embedding:               0.5 ms  ( 0.7%)
```

**Erkenntnis:** Apple's Accelerate BLAS (AMX-beschleunigt) ist extrem schnell auf dem A17 Pro. Die Matmuls allein (44.3 ms) sind nur 6 ms langsamer als ANE eval + IOSurface zusammen, aber ohne den Transfer-Overhead. CPU hat keinen Konvertierungs-Overhead da alles in FP32 bleibt.

#### GPU Forward Pass (270.9 ms) — batched MPS, 48 Syncs

```
GPU Matmuls (MPS):     152.5 ms  (57.4%)
CPU Ops (Attention):   103.9 ms  (39.1%)
Classifier:              5.7 ms  ( 2.1%)
CPU<->GPU Transfer:      2.2 ms  ( 0.8%)
Embedding:               1.4 ms  ( 0.5%)
```

**Erkenntnis:** GPU-Matmuls selbst sind nicht schlecht (152 ms fuer 84 Matmuls), aber Attention muss auf der CPU laufen (kein MPS-Kernel dafuer) und kostet 104 ms. Die 48 Command-Buffer-Synchronisationen (4 pro Layer x 12 Layer) verursachen zusaetzlichen Overhead. Eine MPSGraph-Implementierung mit fused Attention wuerde das erheblich verbessern.

**Hinweis:** Urspruengliche GPU-Implementierung hatte 84 Sync-Punkte (291 ms). Batching auf 48 Syncs brachte ~8% Verbesserung auf 271 ms.

### ANE Power (dedizierter 15-Minuten-Test)

```
Dauer:          15 Minuten (900 Sekunden)
Forward Passes: 8,733
Tokens/s:       2,484
Battery:        85% -> 80% (5% drain)
Power:          2.51 W (inklusive Display + System-Overhead)
Tokens/Joule:   990
Thermal:        nominal (die gesamten 15 Minuten!)
CPU-Auslastung: 5%
```

**Erkenntnis:** ANE ist 3.4x effizienter als CPU (2.51W vs 8.65W) bei nur 23% weniger Durchsatz. Fuer sustained workloads (>10 Min) ist ANE klar ueberlegen: bleibt thermisch nominal waehrend CPU nach wenigen Minuten auf "fair" oder "serious" steigt.

---

## 2. Training-Benchmark

### Vor Optimierung (ACCUM_STEPS=4, sdpaBwd2 recompile, serial dW)

```
Steps/s:          2.44
Tokens/s:         624
Regular Step:     207.0 ms (p50=217, p95=243)
Compile Step:     994.7 ms (p50=1046, p95=1063)
Compile Overhead: 61.6% der Gesamtzeit
Compile Freq:     Alle 4 Steps (25% der Steps sind Compile-Steps)
Power:            7.52 W
J/Step:           3.08 J
Tokens/Joule:     83
CPU-Auslastung:   21.7%
Peak RAM:         2,667 MB
Thermal:          fair
```

### Nach Optimierung (ACCUM_STEPS=8, skip sdpaBwd2, concurrent dW)

```
Steps/s:          3.25  (+33%)
Tokens/s:         832   (+33%)
Regular Step:     200.4 ms (p50=215, p95=228)
Compile Step:     1007.3 ms (p50=1064, p95=1086)
Compile Overhead: 41.8% der Gesamtzeit  (-20 Prozentpunkte)
Compile Freq:     Alle 8 Steps (12.5% der Steps sind Compile-Steps)
Peak RAM:         2,367 MB  (-300 MB)
Thermal:          nominal
```

### Optimierungen im Detail

| Optimierung | Aenderung | Effekt |
|-------------|-----------|--------|
| ACCUM_STEPS 4 -> 8 | `ANETrainingConfig.h:21` | Halbiert Recompile-Frequenz, ~15% Speedup |
| sdpaBwd2 nicht recompilen | `ANETrainingEngine.m:366` | Spart 12 unnoetige Kernel-Compilations pro Zyklus |
| Concurrent dW Dispatch | `ANETrainingEngine.m:475` | Parallelisiert Gradient-Matmuls auf 6 CPU-Cores |
| MAX_COMPILES 200 -> 500 | `ANETrainingConfig.h:23` | Verhindert Budget-Exhaustion bei langen Runs |

### Warum Kernel-Recompilation noetig ist

Das ANE Training nutzt MIL (Machine Learning Intermediate Language) Programme, in denen die Gewichte als **BLOBFILE-Konstanten** eingebacken sind. Wenn der Adam-Optimizer die Gewichte aktualisiert, muessen alle 60 gewichtstragenden Kernels neu kompiliert werden (5 pro Layer x 12 Layer). Jede Compilation dauert ~20 ms, also ~1.2 Sekunden fuer alle 60 Kernels.

Die 12 sdpaBwd2-Kernels sind gewichtsfrei (berechnen dQ/dK aus Attention-Scores) und muessen nie recompiliert werden — das war ein Bug der behoben wurde.

---

## 3. Micro-Benchmarks (Einzeloperationen)

| Operation | Zeit | Beschreibung |
|-----------|------|--------------|
| `ane_eval` (1 Kernel) | 0.63 ms | Einzelner ANE-Kernel (fwdAttn, 12 pro Forward) |
| IO write+read Cycle | 0.22 ms | FP32->FP16 write + FP16->FP32 read via IOSurface |
| `rmsnorm` [768,256] | 0.05 ms | RMSNorm auf CPU via vDSP |
| `classifier` [32000,768] | 1.94 ms | Embedding-Projektion via cblas_sgemv |
| `compile_kern` (1 attn) | 179 ms | Einzelne Kernel-Compilation (MIL -> ANE) |

**Erkenntnis:** Ein einzelner `ane_eval` kostet nur 0.63 ms, aber der IO-Overhead pro Eval (0.22 ms write + 0.22 ms read) addiert sich bei 24 Evals pro Forward Pass zu ~10 ms. Der Rest der IOSurface-Zeit kommt von der FP32<->FP16-Konvertierung groesserer Tensoren (DIM x SEQ = 768 x 256 = 196K Floats).

---

## 4. Thermische Analyse

### 30-Minuten Training-Benchmark (frueherer Test)

```
Nominal:     0s   ( 0.0%)
Fair:       62s   ( 3.5%)
Serious:   896s   (49.8%)
Critical:  841s   (46.7%)
Erste Aenderung: 72s (fair -> serious)
```

**Erkenntnis:** Training erreicht "serious" nach nur 72 Sekunden und verbringt 97% der Zeit im gedrosselten Bereich. Trotzdem bleibt der Durchsatz bei 87% (2.1 -> 1.9 steps/s). Die ANE wird thermisch stark beansprucht, aber iOS drosselt sanft statt die App zu killen.

### ANE Inference (15 Minuten)

```
Thermal: nominal die gesamten 15 Minuten
```

**Erkenntnis:** Reine ANE-Inference bei 2.51W erzeugt nicht genug Waerme um das iPhone ueber "nominal" zu bringen. Das ist bemerkenswert — die ANE ist thermisch fast "unsichtbar".

### iOS CPU-Limit

```
CRASH: "90 seconds cpu time over 151 seconds (59% cpu average),
        exceeding limit of 50% cpu over 180 seconds"
```

**Erkenntnis:** iOS hat ein hartes Limit von 50% durchschnittlicher CPU-Auslastung ueber ein 180-Sekunden-Fenster. Apps die das ueberschreiten werden mit `cpu_resource` Kill terminiert (kein Crash-Dialog, App verschwindet einfach). Fix: 5ms `usleep()` zwischen Iterationen haelt CPU bei ~35-40%.

---

## 5. Speicheranalyse

| Zustand | RAM |
|---------|-----|
| App Start | ~12 MB |
| Nach Weight-Allokation (110M params FP32) | ~432 MB |
| ANE Inference (24 Kernels geladen) | 682 MB |
| CPU Inference (BLAS Buffers) | 701 MB |
| GPU Inference (Metal Buffers) | 1,734 MB |
| Training Init (Weights + Adam + Activations) | 1,891 MB |
| Training Peak (+ Gradients + Scratch) | 2,667 MB (vor Opt) / 2,367 MB (nach Opt) |

**Erkenntnis:** Training braucht ~2.4 GB, was auf dem iPhone 15 Pro (8 GB) knapp ist. iOS limitiert Apps auf ~3-4 GB je nach Systemzustand. Inference und Training koennen nicht gleichzeitig laufen — die Benchmark-App musste in separate Phasen aufgeteilt werden.

---

## 6. Ranking & Empfehlungen

### Inference

| Kriterium | Gewinner | Wert |
|-----------|----------|------|
| Geschwindigkeit | **CPU** | 3,215 tok/s |
| Energieeffizienz | **ANE** | 990 tok/J (2.7x besser als CPU) |
| Thermisch | **ANE** | Bleibt nominal, CPU wird fair/serious |
| RAM-sparend | **CPU** | 701 MB vs 682 MB (kaum Unterschied) |
| Sustained (>10 min) | **ANE** | Kein Throttling |

**Empfehlung:** ANE fuer batteriebetriene Inference (z.B. App laeuft den ganzen Tag). CPU fuer einmalige schnelle Anfragen wo Akku egal ist.

### Training

| Kriterium | ANE (einzig verfuegbar) |
|-----------|------------------------|
| Durchsatz | 3.25 steps/s (832 tok/s) |
| Groesster Bottleneck | Kernel-Recompilation (42% der Zeit) |
| Power | ~7.5 W |
| Thermisch | Erreicht serious/critical in <2 min |

### Optimierungspotential (noch nicht umgesetzt)

| Optimierung | Erwarteter Gewinn | Aufwand |
|-------------|-------------------|---------|
| **Batch-Size 1 -> 4** | 3-5x Speedup | Hoch (MIL-Generatoren umschreiben) |
| **Weight-Streaming via IOSurface** | Eliminiert Recompilation komplett | Sehr hoch (Architektur-Redesign) |
| **GPU-Training via MPSGraph** | Keine Recompilation, aber GPU langsamer | Hoch (~500-1000 Zeilen) |
| IOSurface Batching | 5-8% | Mittel |
| Adam Vektorisierung (vDSP) | <1% | Gering |

---

## 7. Technische Erkenntnisse

### ANE-Hardware (A17 Pro)

- 16 Cores, h16-Architektur (gleiche Generation wie M4)
- ~32 MB SRAM (Gewichts-Cliff bei 32 MB)
- Max 239 gleichzeitig geladene Modelle (von uns 72 genutzt: 60 weight-bearing + 12 weight-free)
- IOSurface-Minimum: 16 KB (DMA-Alignment)
- Idle Power: Hardware power-gated (0 mW wenn ungenutzt)
- Sustained Power: ~2.5 W bei voller Inference-Last
- ANE ist NICHT die GPU — es ist ein separater, dedizierter Beschleuniger

### iOS-Limitierungen

1. **CPU 50%-Limit:** Apps werden gekillt wenn CPU-Durchschnitt 50% ueber 180s uebersteigt
2. **RAM-Limit:** ~3-4 GB pro App, darüber hinaus stiller Kill (kein Crash-Log)
3. **Compile-Budget:** Max 200 (jetzt 500) ANE-Kernel-Kompilationen bevor Reset noetig
4. **Background-Kill:** iOS suspendiert/killt Apps die nicht im Vordergrund sind
5. **Battery-Granularitaet:** UIDevice.batteryLevel hat 1%-Schritte, braucht 15+ Min fuer ANE-Messung

### Datenfluss Training (pro Step)

```
Token-IDs (mmap)
  -> embed_lookup (CPU, FP32)
  -> 12x {
       io_write_fp16 -> ane_eval(fwdAttn) -> io_read_fp16 -> vDSP_vadd (residual)
       io_write_fp16 -> ane_eval(fwdFFN) -> io_read_fp16 -> vDSP_vadd (residual)
     }
  -> rmsnorm (CPU)
  -> cblas_sgemm (classifier, CPU)
  -> cross_entropy_loss (CPU)
  -> Backward (12x reverse, ANE + CPU)
  -> Alle 8 Steps: Adam Update (CPU) + Kernel Recompile (ANE)
```

---

## 8. Offene Fragen & naechste Schritte

1. **GPU-Training via MPSGraph:** Wuerde Recompilation eliminieren. Aber GPU ist 3.5x langsamer als ANE bei Inference — unklar ob der Wegfall der Recompilation das kompensiert. Muss gebaut und getestet werden.

2. **Weight-Streaming:** Statt Gewichte in MIL-BLOBFILEs einzubacken, als IOSurface-Inputs uebergeben. Wuerde Recompilation komplett eliminieren UND IOSurface-Overhead reduzieren. Aber erfordert Neugenerierung aller 7 MIL-Programme.

3. **Batching:** Aktuell Batch-Size 1. Mit Batch 4-8 wuerde der Compile-Overhead amortisiert (einmal kompilieren fuer 4-8 Sequenzen). Erfordert MIL-Generator-Aenderungen.

4. **Praezisions-Experimente:** FP16 Master-Weights mit FP32-Adam koennten RAM halbieren. Risiko: numerische Instabilitaet bei langen Runs.

---

## 9. GPU-Training vs ANE-Training — Direktvergleich

GPU-Training wurde implementiert mit MPS (Metal Performance Shaders) fuer Forward-Matmuls und CPU (Accelerate BLAS) fuer Backward + Attention + Norms. Keine Kernel-Recompilation noetig — Weights leben in MTLBuffer mit SharedMemory, Adam schreibt direkt.

### Vergleichstabelle (je 5 Minuten, gleiche Daten)

| Metrik | ANE Training | GPU Training | Gewinner |
|--------|-------------|-------------|----------|
| **Steps/s** | **3.25** | 2.17 | **ANE (+50%)** |
| **Tokens/s** | **832** | 556 | **ANE (+50%)** |
| **Loss best** | **9.57 (-8.2%)** | 9.97 (-4.4%) | **ANE** (mehr Steps) |
| **Compile Overhead** | 41.8% | **0%** | **GPU** |
| **Recompilation** | Alle 8 Steps | **Nie** | **GPU** |
| **Regular Step** | **200 ms** | ~462 ms | **ANE (2.3x schneller)** |
| **Power** | ~7.5 W | 7.5 W | Gleich |
| **Tokens/Joule** | **83** | 74 | **ANE** |
| **Thermal** | nominal | nominal | Gleich |
| **Init-Zeit** | ~15s (Kernel-Compile) | **<1s** | **GPU** |
| **Peak RAM** | 2,367 MB | ~1,800 MB | **GPU** |

### Analyse

**ANE ist trotz 42% Compile-Overhead 50% schneller als GPU.**

Das liegt daran:
1. **ANE-Hardware ist 2.3x schneller pro Step** (200 ms vs ~462 ms). Die Neural Engine ist fuer Matmul-lastige Workloads optimiert.
2. **GPU hat hohen CPU-GPU Sync-Overhead** — 48 Command Buffer Submissions pro Forward Pass, jeder mit `waitUntilCompleted`. Selbst bei batched Commands kostet jeder Sync ~2ms.
3. **Backward ist komplett auf CPU** (cblas) — die GPU wird nur fuer Forward genutzt. Ein vollstaendiger GPU-Backward wuerde Custom Metal Kernels erfordern (Attention Backward, RMSNorm Backward).

### GPU-Vorteil: Konsistenz

GPU hat **keine Compile-Spikes** — jeder Step dauert ~462 ms, konstant. ANE hat alle 8 Steps einen ~1007 ms Spike (5x langsamer). Fuer latenz-sensitive Anwendungen (Real-Time Micro-Training) ist GPU vorhersagbarer.

### GPU-Vorteil: Sofort startklar

GPU-Training startet in <1 Sekunde (Metal Device + Buffer-Allokation). ANE braucht ~15 Sekunden fuer die initiale Kernel-Compilation (72 Kernels). Fuer kurze Training-Bursts (z.B. 10 Steps nach User-Interaktion) waere GPU besser.

### Fazit

| Use Case | Bestes Backend | Warum |
|----------|---------------|-------|
| **Overnight Training** | **ANE** | 50% schneller, gleicher Stromverbrauch |
| **Micro-Training (10-50 Steps)** | **GPU** | Kein Init-Overhead, vorhersagbare Latenz |
| **Real-Time Fine-Tuning** | **GPU** | Keine Compile-Spikes |
| **Battery-Training** | **ANE** | 83 vs 74 Tokens/Joule |

---

## 10. Externe Validierung & Stand der Technik

Recherche vom 25. Maerz 2026 zum Vergleich unserer Ergebnisse mit dem aktuellen Stand.

### Existierende On-Device Training Loesungen

| Loesung | Typ | Training-Faehigkeit | Plattform |
|---------|-----|---------------------|-----------|
| **Apple MLX** (WWDC 2025) | Offizielles Framework | Full training + LoRA fine-tuning, auto-diff | Mac, iPhone, iPad, Vision Pro |
| **CoreML Updatable Models** | Offizielles API | Nur letzte Layer fine-tunen (FC + Conv) | iOS, macOS |
| **MPSGraph** | Offizielles API | Auto-differentiation, scaledDotProductAttention | iOS, macOS |
| **Metal FlashAttention** (Draw Things) | Open Source | Forward + Backward Attention auf GPU | iOS, macOS |
| **Dieses Projekt** | Forschung | Volles Transformer-Training auf ANE + GPU | iOS |

### Wie wir uns einordnen

**Wir sind nach unserer Recherche die Ersten, die einen vollstaendigen Transformer (110M Parameter) von Grund auf auf dem iPhone ANE trainieren.** Die existierenden Loesungen decken jeweils nur Teilaspekte ab:

- **CoreML**: Nur letzte Layer, nicht geeignet fuer volles Training
- **MLX**: Offiziell unterstuetzt (WWDC 2025), aber hauptsaechlich fuer Mac optimiert. iPhone-Support ist neu, kein oeffentliches Transformer-Training-Beispiel auf iPhone bekannt
- **Draw Things / Metal FlashAttention**: Fine-tuning von FLUX.1 (11B) mit LoRA auf iPhone — beeindruckend, aber kein Training from scratch. Nutzt GPU, nicht ANE
- **MPSGraph**: Hat alle APIs (matmul, attention, autograd), aber kein oeffentliches Beispiel fuer Transformer-Training auf iPhone

### Externe Power-Zahlen zum Vergleich

| Quelle | Messung | Unsere Messung |
|--------|---------|----------------|
| Forschung (iPhone 13 ANE) | 0.07 - 0.45 W fuer ANE allein | 2.51 W (inkl. Display + System) |
| Apple (Mac GPU vs ANE) | ANE ~2W vs GPU ~20W | ANE 2.51W vs CPU 8.65W (iPhone) |
| Apple (A17 Pro ANE) | 35 TOPS, 16 Cores | Bestaetigt durch unser Device-Info Probe |
| Apple (LLM Inference) | 30 tok/s (autoregressive, 3B Modell) | 2,480 tok/s (non-autoregressive, 110M) |

**Erklaerung Differenz:** Unsere 2.51W inkludieren Display (~1-1.5W) + System-Overhead (~0.3W). Der reine ANE-Verbrauch liegt vermutlich bei ~0.5-1W, was konsistent mit externen Messungen ist.

**Erklaerung Tokens/s:** Unsere 2,480 tok/s sind NICHT mit Apple's 30 tok/s vergleichbar. Apple misst autoregressive Generation (ein Token nach dem anderen, mit KV-Cache). Wir messen einen vollen Forward Pass ueber 256 Tokens gleichzeitig. Das ist ein fundamental anderer Workload.

### Validierung unserer Einzelkomponenten

| Unsere Messung | Extern validierbar? | Status |
|----------------|---------------------|--------|
| ANE 35 TOPS | Ja (Apple Spec) | Bestaetigt |
| ANE Power <1W (rein) | Ja (Research: 0.07-0.45W auf iPhone 13) | Konsistent |
| CPU schneller als ANE fuer Inference | Teilweise — bekannt dass AMX sehr schnell ist | Plausibel, IOSurface-Overhead erklaert es |
| 42% Compile-Overhead bei Training | Kein externer Vergleich (einzigartig) | Nicht validierbar, aber Mechanismus nachvollziehbar |
| GPU Training 2.17 steps/s | Kein externer Vergleich | Nicht validierbar |
| GPU keine Recompilation noetig | Architektonisch korrekt (MTLBuffer vs BLOBFILE) | Logisch begruendet |

### Limitierungen unserer Messungen

1. **Battery-Granularitaet**: UIDevice.batteryLevel hat 1% Schritte. Alle Power-Werte unter 3% Drain haben hohe Unsicherheit
2. **System-Overhead**: Alle Power-Messungen inkludieren Display + iOS System. Reine Compute-Power ist niedriger
3. **GPU Backward auf CPU**: Unser GPU-Training nutzt CPU fuer den Backward-Pass. Eine vollstaendige GPU-Implementation (mit Metal FlashAttention Backward) waere schneller
4. **Keine Kreuzvalidierung**: ANE und GPU Benchmarks liefen in verschiedenen Sessions mit unterschiedlichen Akku-Staenden. Nicht perfekt vergleichbar
5. **Kurze Laufzeiten**: 5-Minuten Tests. Laengere Runs (30+ Min) wuerden thermische Effekte besser zeigen
6. **Einzelgeraet**: Nur iPhone 15 Pro getestet. Ergebnisse auf anderen Geraeten (iPhone 16 Pro, iPad) koennen abweichen

### Quellen

- Apple: "Introducing Apple's On-Device and Server Foundation Models" — https://machinelearning.apple.com/research/introducing-apple-foundation-models
- Apple: "Training a Neural Network with MPS" — https://developer.apple.com/documentation/MetalPerformanceShaders/training-a-neural-network-with-metal-performance-shaders
- Apple: "Training a Neural Network using MPSGraph" — https://developer.apple.com/documentation/metalperformanceshadersgraph/training-a-neural-network-using-mps-graph
- Apple: "Deploying Transformers on the Apple Neural Engine" — https://machinelearning.apple.com/research/neural-engine-transformers
- Apple: "Train your ML and AI models on Apple GPUs" (WWDC24) — https://developer.apple.com/videos/play/wwdc2024/10160/
- Apple: "Get started with MLX for Apple silicon" (WWDC25) — https://developer.apple.com/videos/play/wwdc2025/315/
- Draw Things: "Metal FlashAttention 2.0" — https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c
- GitHub: metal-flash-attention — https://github.com/philipturner/metal-flash-attention
- GitHub: MLX Framework — https://github.com/ml-explore/mlx
- Swift.org: "On-device ML research with MLX and Swift" — https://www.swift.org/blog/mlx-swift/
- machinethink.net: "On-device training with Core ML" — https://machinethink.net/blog/coreml-training-part1/
- arXiv: "Benchmarking On-Device ML on Apple Silicon with MLX" — https://arxiv.org/abs/2510.18921
- GitHub: hollance/neural-engine (Supported Devices) — https://github.com/hollance/neural-engine/blob/master/docs/supported-devices.md
- Wikipedia: Apple A17 — https://en.wikipedia.org/wiki/Apple_A17

---

## 11. Modellgroesse & Geraete-Kompatibilitaet

### Warum das Modell wichtig ist

Die Wahl des Modells bestimmt, welches Backend (ANE vs GPU vs CPU) am besten ist:

| Modellgroesse | RAM (Training) | ANE vs GPU | Empfohlene Geraete |
|---------------|---------------|------------|---------------------|
| **<50M Params** | <1 GB | CPU am schnellsten | Alle iPhones ab iPhone 12 |
| **50-200M Params** (unser Stories-110M) | 1.5-2.5 GB | ANE schneller, CPU nah dran | iPhone 13 Pro+ (6 GB), iPhone 15+ (6-8 GB) |
| **200M-500M Params** | 3-5 GB | GPU koennte ANE schlagen (groessere Matmuls) | Nur iPhone Pro (8 GB) |
| **500M-1B Params** | 5-10 GB | GPU wahrscheinlich schneller | iPad Pro M-Chip (16 GB), nicht iPhone |
| **>1B Params** | >10 GB | Nur mit LoRA/QLoRA moeglich | iPad Pro/Mac, NICHT iPhone |

**Fuer normale iPhone-Nutzer (kein Pro, 6 GB RAM) sind Modelle bis ~150M Parameter realistisch.** iPhone Pro (8 GB) erlaubt bis ~250M. Alles darueber braucht iPad oder Mac.

### Warum ANE bei kleinen Modellen langsamer aussieht

Bei 110M Parametern sind die Matmuls relativ klein (768x768). Apple's CPU (AMX) berechnet solche Matmuls in Mikrosekunden. Der IOSurface-Transfer-Overhead (FP32<->FP16 Konvertierung + Lock/Unlock) dominiert die ANE-Laufzeit (51% der Inference-Zeit).

Bei groesseren Modellen (dim=2048+) waechst die Matmul-Rechenzeit quadratisch, waehrend der IO-Overhead linear bleibt. Ab einer bestimmten Groesse wird ANE deutlich schneller als CPU — genau dafuer ist die Neural Engine designed.

**Fazit fuer App-Entwickler:** Fuer Modelle unter ~200M auf iPhone ist CPU-Inference am schnellsten, ANE am effizientesten (Power). Fuer Training ist ANE immer besser wegen dem massiv niedrigeren Stromverbrauch.

### RAM-Management fuer Training

iPhone-Training braucht viel RAM (~2.4 GB fuer 110M). Tipps:

1. **Vor dem Training:** Alle anderen Apps manuell schliessen (iOS erlaubt kein programmatisches Schliessen fremder Apps)
2. **Low Power Mode deaktivieren:** Kann die ANE drosseln
3. **Am Ladegeraet:** Fuer Overnight-Training empfohlen — 7.5W Power Drain leert den Akku in ~1.5h
4. **Unsere App zeigt Memory-Warnung** wenn >2 GB belegt vor Training-Start

### Benchmark-Methodik

Die ANE vs GPU Vergleichszahlen wurden in **getrennten App-Sessions** gemessen (nicht gleichzeitig), weil:
- Beide zusammen in einer Session hat zu iOS Memory-Kills gefuehrt (~3.5 GB+ RAM)
- ANE Training alloziert ~2.4 GB, GPU Training ~1.8 GB — beides zusammen uebersteigt das iOS App-Limit
- Die Trennung ist methodisch korrekt: steps/s und Latenz sind unabhaengig vom Battery-Zustand
- Power-Vergleich ist eingeschraenkt da unterschiedliche Battery-Level (dokumentiert in Limitierungen)
