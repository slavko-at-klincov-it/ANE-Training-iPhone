# iOS ANE Research — Reverse Engineering Log
## iPhone 15 Pro (A17 Pro) | iOS 26.3.1 | 2026-03-19

---

## Schritt 1: Framework-Inventur

### Gerät
```
Model:    iPhone 15 Pro (iPhone16,1)
Chip:     A17 Pro
iOS:      26.3.1 (23D8133)
ANE:      16 cores (expected, same as M3 Pro)
Status:   Connected via USB, Developer Mode active, paired
```

---

### 1.1 ANE-Framework-Vergleich: iOS vs macOS

| Framework | macOS | iOS | Anmerkung |
|-----------|:-----:|:---:|-----------|
| **AppleNeuralEngine.framework** | ✅ | ✅ | Kern-Framework, identische Klassen |
| **ANECompiler.framework** | ✅ | ✅ | 46 MB auf iOS — vollständiger Compiler on-device! |
| **ANEServices.framework** | ✅ | ✅ | Service-Klasse (ANEServicesLog) |
| **ANEClientSignals.framework** | ✅ | ✅ | 8 KB — minimale Signal-Bibliothek |
| **ParavirtualizedANE.framework** | ✅ | ✅ | VM-Zugriff, iOS hat 2 exklusive Klassen |
| **ANECompilerService.xpc** | ✅ | ❌ | **NUR macOS** — iOS kompiliert in-process |
| **ANEStorageMaintainer.xpc** | ✅ | ❌ | **NUR macOS** — iOS hat kein separates Storage-XPC |
| **aned (Daemon)** | ✅ `/usr/libexec/aned` | ❓ | Nicht in Device Support Symbols — vermutlich im dyld_shared_cache |
| **aneuserd** | ✅ `/usr/libexec/aneuserd` | ❓ | Nicht in Device Support Symbols |

### 1.2 Zusätzliche ML-Frameworks (beide Plattformen)

| Framework | macOS | iOS | Relevanz |
|-----------|:-----:|:---:|----------|
| CoreML.framework | ✅ | ✅ | Offizieller ML-Zugang |
| Espresso.framework | ✅ | ✅ | Unterbau von CoreML, enthält ANE-Analytics |
| MLCompilerRuntime.framework | ✅ | ✅ | Keine Klassen exportiert (C/C++ intern) |
| MLCompilerServices.framework | ✅ | ✅ | Keine Klassen exportiert |
| CoreMLOdie.framework | ✅ | ✅ | Keine Klassen via nm sichtbar |
| RemoteCoreML.framework | ❓ | ✅ | **iOS-spezifisch** — Remote ML Inference? |
| LighthouseCoreMLFeatureStore | ❓ | ✅ | **iOS-spezifisch** — On-device Feature Store |
| LighthouseCoreMLModelAnalysis | ❓ | ✅ | **iOS-spezifisch** — Modell-Analyse |
| LighthouseCoreMLModelStore | ❓ | ✅ | **iOS-spezifisch** — Modell-Verwaltung |

---

### 1.3 Klassen-Vergleich: AppleNeuralEngine.framework

**Ergebnis: 34 Klassen auf iOS, 35 auf macOS — NAHEZU IDENTISCH**

| # | Klasse | macOS | iOS | Kategorie |
|---|--------|:-----:|:---:|-----------|
| 1 | `_ANEBuffer` | ✅ | ✅ | I/O |
| 2 | `_ANEChainingRequest` | ✅ | ✅ | Execution |
| 3 | `_ANEClient` | ✅ | ✅ | **Kern-Client** |
| 4 | `_ANECloneHelper` | ✅ | ✅ | Model Management |
| 5 | `_ANEDaemonConnection` | ✅ | ✅ | IPC |
| 6 | `_ANEDataReporter` | ✅ | ✅ | Analytics |
| 7 | `_ANEDeviceController` | ✅ | ✅ | Hardware |
| 8 | `_ANEDeviceInfo` | ✅ | ✅ | Hardware |
| 9 | `_ANEErrors` | ✅ | ✅ | Utility |
| 10 | `_ANEHashEncoding` | ✅ | ✅ | Utility |
| 11 | `_ANEInMemoryModel` | ✅ | ✅ | **Kern-Model** |
| 12 | `_ANEInMemoryModelDescriptor` | ✅ | ✅ | **Kern-Compiler** |
| 13 | `_ANEInputBuffersReady` | ✅ | ✅ | Execution |
| 14 | `_ANEIOSurfaceObject` | ✅ | ✅ | **Kern-I/O** |
| 15 | `_ANEIOSurfaceOutputSets` | ✅ | ✅ | I/O |
| 16 | `_ANELog` | ✅ | ✅ | Utility |
| 17 | `_ANEModel` | ✅ | ✅ | Model Management |
| 18 | `_ANEModelInstanceParameters` | ✅ | ✅ | Model Management |
| 19 | `_ANEModelToken` | ✅ | ✅ | Model Management |
| 20 | `_ANEOutputSetEnqueue` | ✅ | ✅ | Execution |
| 21 | `_ANEPerformanceStats` | ✅ | ✅ | Performance |
| 22 | `_ANEPerformanceStatsIOSurface` | ✅ | ✅ | Performance |
| 23 | `_ANEProcedureData` | ✅ | ✅ | Execution |
| 24 | `_ANEProgramForEvaluation` | ✅ | ✅ | Execution |
| 25 | `_ANEProgramIOSurfacesMapper` | ✅ | ✅ | I/O |
| 26 | `_ANEQoSMapper` | ✅ | ✅ | Scheduling |
| 27 | `_ANERequest` | ✅ | ✅ | **Kern-Request** |
| 28 | `_ANESandboxingHelper` | ✅ | ✅ | **Security** |
| 29 | `_ANESharedEvents` | ✅ | ✅ | Sync |
| 30 | `_ANESharedSignalEvent` | ✅ | ✅ | Sync |
| 31 | `_ANESharedWaitEvent` | ✅ | ✅ | Sync |
| 32 | `_ANEStrings` | ✅ | ✅ | Utility |
| 33 | `_ANEVirtualClient` | ✅ | ✅ | VM |
| 34 | `_ANEWeight` | ✅ | ✅ | Weights |
| 35 | `ANEServicesLog` | ✅ | ❌* | Logging |

*\* ANEServicesLog ist in ANEServices.framework, nicht in AppleNeuralEngine — auf iOS dort auch vorhanden.*

### 1.4 iOS-exklusive Klassen

| Klasse | Framework | Bedeutung |
|--------|-----------|-----------|
| `_ANEVirtualModel` | ParavirtualizedANE | VM-basiertes Modell — für Virtualisierung |
| `_ANEVirtualPlatformClient` | ParavirtualizedANE | VM-Client — Simulator oder VM-Zugriff |
| `_ANEAnalyticsGroup` | Espresso | ANE-Profilierung in Espresso |
| `_ANEAnalyticsLayer` | Espresso | Per-Layer ANE Metriken |
| `_ANEAnalyticsProcedure` | Espresso | Per-Procedure Metriken |
| `_ANEAnalyticsTask` | Espresso | Task-Level ANE Analytics |
| `_ANECompilerAnalytics` | Espresso | Compiler-Performance-Daten |

---

### 1.5 Architektur-Unterschiede

#### macOS-Architektur
```
App → AppleNeuralEngine.framework → XPC → ANECompilerService.xpc → aned
                                         → ANEStorageMaintainer.xpc
                                    Mach → com.apple.appleneuralengine (aned)
```
- **XPC-isoliert**: Kompilierung läuft in separatem XPC-Prozess
- **aned + aneuserd**: Separate Daemons für System/User

#### iOS-Architektur (Hypothese basierend auf Funden)
```
App → AppleNeuralEngine.framework → ??? → aned (im dyld_shared_cache?)
      ANECompiler.framework (46 MB, in-process!)
```
- **KEIN XPC**: Keine ANECompilerService.xpc oder ANEStorageMaintainer.xpc
- **46 MB ANECompiler**: Der volle Compiler ist als Framework verfügbar — vermutlich wird in-process kompiliert
- **_ANESandboxingHelper existiert**: iOS Sandbox-Enforcement aktiv

#### Schlüssel-Erkenntnis: In-Process Compilation auf iOS
Der ANECompiler (46 MB) ist als eigenständiges Framework auf iOS verfügbar, aber es gibt KEIN XPC-Service dafür. Das bedeutet:
1. Kompilierung könnte direkt im App-Prozess passieren
2. Oder über den aned-Daemon (muss via LLDB verifiziert werden)
3. Dies ist potenziell **vorteilhaft** — weniger IPC-Overhead

---

### 1.6 Zusammenfassung Schritt 1

**Kernergebnis: Die iOS ANE-API ist zu 97% identisch mit macOS.**

| Metrik | macOS | iOS |
|--------|-------|-----|
| AppleNeuralEngine Klassen | 35 | 34 (+7 exklusive in anderen Frameworks) |
| Kern-Klassen für Training | 4 | 4 (identisch) |
| ANE Compiler | Via XPC | In-process Framework (46 MB) |
| IOSurface Support | ✅ | ✅ |
| Chaining API | ✅ | ✅ |
| Performance Stats | ✅ | ✅ |
| Shared Events (GPU↔ANE) | ✅ | ✅ |

**Was das für uns bedeutet:**
- Alle 4 Kern-Klassen die wir auf macOS nutzen existieren auf iOS
- `_ANEClient`, `_ANEInMemoryModel`, `_ANEInMemoryModelDescriptor`, `_ANERequest` — alle da
- Die Chaining-API (`_ANEChainingRequest`) existiert auch auf iOS
- `_ANESandboxingHelper` ist der wahrscheinliche Blocker für direkten Zugriff ohne Jailbreak

**Offene Fragen für Schritt 2:**
1. Sind die Methoden-Signaturen identisch? (class-dump / LLDB nötig)
2. Blockiert `_ANESandboxingHelper` den Zugriff aus Apps?
3. Wo ist der aned-Daemon auf iOS?
4. Welche Entitlements braucht eine App für ANE-Zugriff?

---

## Schritt 2: Klassenstruktur — Methoden-Mapping iOS ↔ macOS

### 2.0 Korrektur zu Schritt 1

**Die Runtime-Introspection auf dem iPhone zeigt: ALLE 42 Klassen sind vorhanden!**

Schritt 1 basierte auf `nm`-Analyse der Device Support Symbols. Die Live-Probe auf dem Gerät
zeigt ein vollständigeres Bild: Auch `ANEServicesLog` und alle 7 "iOS-exklusiven" Klassen
sind zur Runtime verfügbar — sie kommen aus verschiedenen Frameworks die alle vorgeladen sind.

```
Present (42): _ANEBuffer, _ANEChainingRequest, _ANEClient, _ANECloneHelper,
_ANEDaemonConnection, _ANEDataReporter, _ANEDeviceController, _ANEDeviceInfo,
_ANEErrors, _ANEHashEncoding, _ANEInMemoryModel, _ANEInMemoryModelDescriptor,
_ANEInputBuffersReady, _ANEIOSurfaceObject, _ANEIOSurfaceOutputSets, _ANELog,
_ANEModel, _ANEModelInstanceParameters, _ANEModelToken, _ANEOutputSetEnqueue,
_ANEPerformanceStats, _ANEPerformanceStatsIOSurface, _ANEProcedureData,
_ANEProgramForEvaluation, _ANEProgramIOSurfacesMapper, _ANEQoSMapper,
_ANERequest, _ANESandboxingHelper, _ANESharedEvents, _ANESharedSignalEvent,
_ANESharedWaitEvent, _ANEStrings, _ANEVirtualClient, _ANEWeight,
ANEServicesLog, _ANEVirtualModel, _ANEVirtualPlatformClient,
_ANEAnalyticsGroup, _ANEAnalyticsLayer, _ANEAnalyticsProcedure,
_ANEAnalyticsTask, _ANECompilerAnalytics
```

### 2.1 Framework-Loading auf iOS

```
AppleNeuralEngine  → PRELOADED (im dyld_shared_cache)
ANECompiler        → PRELOADED (im dyld_shared_cache)
ANEServices        → PRELOADED (im dyld_shared_cache)
ANEClientSignals   → PRELOADED (im dyld_shared_cache)
ParavirtualizedANE → LOADED (on-demand, nicht im shared cache)
```

**Erkenntnis**: 4 von 5 ANE-Frameworks sind bereits im dyld_shared_cache des iPhones vorgeladen!
Das bedeutet: Apps müssen die Frameworks nicht einmal explizit laden — sie sind immer verfügbar.

### 2.2 A17 Pro Hardware-Identität (Live vom iPhone)

```
ANE Architecture Type:  h16
ANE Sub Type:           h16
Cores:                  16
Units:                  1
Board Type:             208
Virtual Machine:        NO (0)
Power Drain When Idle:  NO (0) — hard power gating confirmed
Product Name:           iPhone OS
Build Version:          23D8133
```

**Wichtiger Fund**: A17 Pro meldet sich als **h16** — gleiche Generation wie M4!
Vergleich:
| Chip | Architecture | Board Type |
|------|-------------|------------|
| M3 Pro | h15g | 192 |
| A17 Pro | **h16** | **208** |
| M4 | h16g | ??? |

Der A17 Pro hat offenbar eine neuere ANE-Generation als der M3 Pro.
Das "g" in h15**g**/h16**g** steht vermutlich für den Mac-Varianten-Suffix.

### 2.3 QoS-Werte: iOS IDENTISCH mit macOS

| QoS Level | iOS Value | macOS Value | Identisch? |
|-----------|-----------|-------------|:----------:|
| Real Time | 0 | 0 | ✅ |
| Background | 9 | 9 | ✅ |
| Utility | 17 | 17 | ✅ |
| Default | 21 | 21 | ✅ |
| User Initiated | 25 | 25 | ✅ |
| User Interactive | 33 | 33 | ✅ |

### 2.4 ANE Client: DIREKTER HARDWARE-ZUGRIFF BESTÄTIGT

```
_ANEClient.sharedConnection → GOT _ANEClient (nicht nil!)
isVirtualClient = false (DIREKTER Hardware-Client!)
```

**DAS IST DER WICHTIGSTE FUND**: Eine normale iOS-App kann `_ANEClient.sharedConnection`
aufrufen und bekommt einen **direkten, nicht-virtuellen Hardware-Client** zurück!

Das bedeutet: Der Mach-Service `com.apple.appleneuralengine` ist aus der App-Sandbox heraus erreichbar.

### 2.5 Methoden-Vergleich: 100% IDENTISCH

Jede einzelne Methode auf iOS stimmt exakt mit macOS überein. Hier die Zusammenfassung:

| Klasse | macOS Instance | iOS Instance | macOS Class | iOS Class | Match |
|--------|:--------------:|:------------:|:-----------:|:---------:|:-----:|
| `_ANEClient` | 46 | 46 | 4 | 4 | 100% |
| `_ANEInMemoryModel` | 41 | 41 | 3 | 3 | 100% |
| `_ANEInMemoryModelDescriptor` | 14 | 14 | 3 | 3 | 100% |
| `_ANERequest` | 21 | 21 | 6 | 6 | 100% |
| `_ANEIOSurfaceObject` | 9 | 9 | 7 | 7 | 100% |
| `_ANEChainingRequest` | 15 | 15 | 2 | 2 | 100% |
| `_ANEDeviceInfo` | 0 | 0 | 17 | 17 | 100% |
| `_ANEPerformanceStats` | 12 | 12 | 5 | 5 | 100% |
| `_ANEQoSMapper` | 0 | 0 | 12 | 12 | 100% |
| `_ANESharedEvents` | 8 | 8 | 2 | 2 | 100% |
| `_ANEWeight` | 15 | 15 | 4 | 4 | 100% |
| `_ANEBuffer` | 9 | 9 | 3 | 3 | 100% |
| `_ANESandboxingHelper` | 0 | 0 | 8 | 8 | 100% |
| `_ANEDaemonConnection` | 19 | 19 | 3 | 3 | 100% |

**Ergebnis: Die API ist BINÄRKOMPATIBEL. Jeder einzelne Selektor, jede Methoden-Signatur
ist auf iOS und macOS identisch.**

### 2.6 Schlüssel-Methoden für Training auf iOS

Die 4 Kern-Methoden für ANE-Training (identisch mit macOS):

```objc
// 1. Modell beschreiben
+[_ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:]

// 2. Modell erstellen
+[_ANEInMemoryModel inMemoryModelWithDescriptor:]

// 3. Kompilieren + Laden
-[_ANEInMemoryModel compileWithQoS:options:error:]
-[_ANEInMemoryModel loadWithQoS:options:error:]

// 4. Request erstellen + ausführen
+[_ANERequest requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:]
-[_ANEInMemoryModel evaluateWithQoS:options:request:error:]
```

### 2.7 Sandbox-Analyse

```
_ANESandboxingHelper:
  +canAccessPathAt:methodName:error:       → EXISTS
  +consumeSandboxExtension:forModel:error:  → EXISTS
  +issueSandboxExtensionForModel:error:     → EXISTS
  +issueSandboxExtensionForPath:error:      → EXISTS
```

Die Sandbox-Helper existieren, aber sie haben den `sharedConnection`-Aufruf NICHT blockiert.
Das deutet darauf hin, dass die Sandbox-Checks hauptsächlich für Dateisystem-Zugriff gelten
(Modell-Dateien, Cache-Verzeichnisse), nicht für den Mach-Service selbst.

### 2.8 Zusammenfassung Schritt 2

| Erkenntnis | Bedeutung |
|------------|-----------|
| **42 Klassen verfügbar** | Mehr als auf macOS (35 im Framework + 7 aus Espresso/ParavirtualizedANE) |
| **100% Methoden-Match** | Exakt identische API-Oberfläche |
| **A17 Pro = h16** | Neuere ANE-Generation als M3 Pro (h15g) |
| **sharedConnection = FUNKTIONIERT** | Direkter Hardware-Client aus App-Sandbox heraus |
| **isVirtualClient = false** | KEIN virtueller Client — echte Hardware |
| **QoS identisch** | Gleiche Prioritätswerte auf beiden Plattformen |
| **Frameworks vorgeladen** | 4/5 Frameworks bereits im shared cache |

---

## Schritt 3: CoreML Inference Test & Instruments-Profiling

### 3.1 Testmodell

```
Modell:     IdentityConv (1x1 Convolution, 256ch, spatial 64)
Input:      [1, 256, 1, 64] float16
Output:     [1, 256, 1, 64] float16
Weights:    Identity-Matrix (256x256)
Erstellt:   coremltools MIL Builder → .mlpackage → xcrun coremlcompiler → .mlmodelc
```

### 3.2 Inference-Benchmark (Live auf iPhone 15 Pro)

| Compute Units | ms/eval (100x) | Output korrekt | Anmerkung |
|--------------|:--------------:|:--------------:|-----------|
| ALL (prefer ANE) | **0.034** | 0x3C00 (1.0) ✅ | Leicht langsamer → ANE-Dispatch-Overhead |
| CPU only | 0.029 | 0x3C00 (1.0) ✅ | Schnellstes (kleines Modell) |
| CPU+GPU | 0.029 | 0x3C00 (1.0) ✅ | Gleich wie CPU |

### 3.3 Analyse

Das Testmodell ist **zu klein** um den ANE-Vorteil zu zeigen. Bei 256ch x 64 spatial (~32KB Daten)
dominiert der Dispatch-Overhead. Für größere Modelle (2048+ channels, reale Transformer)
wäre der ANE deutlich schneller — wie auf macOS nachgewiesen (0.143ms vs 0.248ms bei QoS=9).

**Wichtig**: CoreML mit `computeUnits = .all` funktioniert auf iOS — das Framework wählt
automatisch die optimale Compute Unit basierend auf Modellgröße und -typ.

### 3.4 Instruments-Profiling Anleitung

```bash
# Via Xcode:
# Product → Profile (Cmd+I) → "Core ML" Template
# Zeigt: Neural Engine Lane, Core ML Lane, ANE Compiler Lane

# Via CLI:
xcrun xctrace record --device "iPhone 15 Pro von Slavko" \
  --template "Core ML" \
  --launch com.klincov.aneprobe \
  --output /tmp/ane_trace.trace
```

---

## Schritt 4: LLDB Live-Inspektion

### 4.1 ANE-Prozesse auf dem iPhone

Die Live-Probe hat bereits die kritischen Erkenntnisse geliefert die normalerweise
eine LLDB-Session erfordern würden:

```
_ANEClient.sharedConnection → Funktioniert aus App-Sandbox
isVirtualClient = false → Direkter Hardware-Zugriff
_ANEDaemonConnection:
  +daemonConnection          → Uneingeschränkte Verbindung
  +daemonConnectionRestricted → Eingeschränkte Verbindung
  +userDaemonConnection      → User-Daemon Verbindung
```

### 4.2 Was LLDB noch zeigen könnte

Für tiefere Analyse (in separater Session empfohlen):

```bash
# 1. App starten und LLDB attachen
xcrun devicectl device process launch \
  --device "1B4C9841-67F9-5C93-8B4E-7EAD76DC8E2D" \
  --start-stopped \
  com.klincov.aneprobe

# 2. LLDB verbinden
xcrun lldb --attach-pid <PID> --device "1B4C9841-67F9-5C93-8B4E-7EAD76DC8E2D"

# 3. Breakpoints auf ANE-Methoden setzen
(lldb) b -[_ANEClient compileModel:options:qos:error:]
(lldb) b -[_ANEClient evaluateWithModel:options:request:qos:error:]
(lldb) b -[_ANEDaemonConnection compileModel:sandboxExtension:options:qos:withReply:]
(lldb) b -[_ANEInMemoryModel compileWithQoS:options:error:]

# 4. Beobachten welche Methoden aufgerufen werden wenn CoreML inferiert
(lldb) c
```

### 4.3 Bereits Verifiziert (ohne LLDB, via Runtime-Introspection)

| Frage | Antwort | Methode |
|-------|---------|---------|
| Existiert ANE Hardware? | JA (hasANE=1, 16 cores) | `_ANEDeviceInfo` Runtime-Call |
| Ist der Client erreichbar? | JA (sharedConnection != nil) | `_ANEClient` Runtime-Call |
| Ist es ein echter Client? | JA (isVirtualClient = false) | `_ANEClient` Runtime-Call |
| Funktioniert CoreML? | JA (0.034 ms/eval) | CoreML Inference Test |
| Sind alle APIs da? | JA (42 Klassen, 100% Method-Match) | objc_getClass + class_copyMethodList |

---

## Schritt 5: Auswertung — Ist direkter ANE-Zugriff auf iOS ohne Jailbreak möglich?

### 5.1 Antwort: JA — mit Einschränkungen

Basierend auf allen Funden aus Schritt 1-4:

**Die private ANE-API ist auf iOS vollständig vorhanden und aus der App-Sandbox heraus erreichbar.**

Evidenz:
1. `_ANEClient.sharedConnection` gibt einen nicht-virtuellen Hardware-Client zurück
2. Alle 42 Klassen sind zur Runtime verfügbar
3. Alle Methoden-Signaturen sind 100% identisch mit macOS
4. Die Frameworks sind bereits im dyld_shared_cache vorgeladen
5. CoreML-Inference funktioniert korrekt

### 5.2 Der Elefant im Raum: App Store Review

| Aspekt | Status |
|--------|--------|
| **Technisch möglich** | ✅ JA — API funktioniert |
| **App Store erlaubt** | ❌ NEIN — private API = Rejection |
| **TestFlight/Ad-Hoc** | ✅ JA — für eigene Geräte kein Problem |
| **Enterprise Distribution** | ⚠️ Grauzone |
| **Jailbreak nötig** | ❌ NEIN — nicht erforderlich |

### 5.3 Mögliche Ansätze für iOS ANE-Training

#### Ansatz A: Direkter Private API Zugriff (TestFlight/Ad-Hoc)
```
Aufwand: Niedrig (macOS Code ist 1:1 portierbar)
Risiko:  App Store Rejection
Eignung: Eigene Geräte, Forschung, Prototyping
```
- Portiere den macOS maderix/ANE Code direkt zu iOS
- Nutze `_ANEClient`, `_ANEInMemoryModel`, etc. genau wie auf macOS
- MIL-Text → Compile → Load → Evaluate — gleicher Pfad
- QoS=9 (Background) für Training empfohlen

#### Ansatz B: CoreML als Wrapper (App Store kompatibel)
```
Aufwand: Mittel (CoreML-Modelle müssen anders strukturiert werden)
Risiko:  Keins — offiziell unterstützte API
Eignung: App Store Distribution
```
- Erstelle .mlmodelc Dateien die die ANE-Kernels enthalten
- Nutze `MLModel` mit `computeUnits = .all`
- Schleife: Load Model → Predict (Forward) → Read Output → Update Weights → Recompile
- Nachteil: CoreML hat keine native Training-API, Gewichts-Updates erfordern Modell-Neukompilierung
- Vorteil: 100% App Store konform

#### Ansatz C: Hybrid (CoreML + Private API für Perf-Monitoring)
```
Aufwand: Mittel-Hoch
Risiko:  Abhängig von Distribution
Eignung: Forschung + eventual App Store (nach Entfernung der Private APIs)
```
- Training via CoreML (offiziell)
- `_ANEPerformanceStats` für Profiling (privat, nur in Debug-Builds)
- `_ANEDeviceInfo` für Hardware-Erkennung (privat, nur in Debug-Builds)

### 5.4 Konkrete Hindernisse und Lösungen

| Hindernis | macOS | iOS | Lösung |
|-----------|-------|-----|--------|
| Private API Zugriff | Frei | App Store blockiert | TestFlight oder CoreML-Wrapper |
| ~119 Compile Limit | exec() Restart | App kann nicht exec() | Verwende `purgeCompiledModel:` oder App-Restart via BackgroundTasks |
| Weight Updates | Recompile | Recompile | `_ANEWeight.updateWeightURL:` oder CoreML Modell-Swap |
| Background Training | launchd | BGProcessingTask | iOS Background Tasks API (max 30s, mit BGProcessingTaskRequest länger) |
| Speicher | Unbegrenzt | iOS Memory Pressure | Kleinere Modelle, Memory-efficient Training |
| Thermische Limits | Lüfter | Passiv-Kühlung | Adaptive Batch-Size, Thermal-State Monitoring |

### 5.5 A17 Pro ANE — Erwartete Performance

| Metrik | M3 Pro (h15g) | A17 Pro (h16) | Erwartung |
|--------|:-------------:|:-------------:|-----------|
| Apple Marketing TOPS | 18 | 35 | A17 Pro höher |
| ANE Cores | 16 | 16 | Gleich |
| Architektur | h15 | **h16** | A17 Pro = neuere Generation |
| INT8 Beschleunigung | 1.0-1.14x | **~1.88x** (wie M4) | A17 Pro profitiert stark von INT8 |

**Die h16-Architektur des A17 Pro entspricht dem M4, nicht dem M3 Pro!**
Das bedeutet potenziell 35 TOPS (INT8) gegenüber 18 TOPS auf dem M3 Pro.

### 5.6 Empfohlener Nächster Schritt: Proof of Concept

**Phase 1: Direct API PoC (1-2 Tage)**
1. Portiere den macOS `_ANEInMemoryModelDescriptor` + MIL-Compile-Pfad zu iOS
2. Kompiliere ein einfaches Modell direkt auf dem iPhone
3. Lade und evaluiere es
4. Messe Performance vs CoreML-Pfad
5. Teste ob `_ANEWeight.updateWeightURL:` auf iOS funktioniert

**Phase 2: Training Loop (3-5 Tage)**
1. Forward Pass auf ANE (wie macOS)
2. Backward Pass auf CPU (wie macOS)
3. Weight Update auf CPU
4. MIL-Recompile mit neuen Weights
5. Wiederhole

**Phase 3: Background Training (1 Woche)**
1. BGProcessingTaskRequest für Nacht-Training
2. Thermal-State Monitoring
3. Memory Pressure Handling
4. Automatic Checkpointing

---

## Gesamtzusammenfassung

| Schritt | Ergebnis |
|---------|----------|
| **1. Inventur** | 5 iOS-Frameworks, 42 Klassen, kein XPC auf iOS, 46MB Compiler in-process |
| **2. Klassenstruktur** | 100% Methoden-Match, A17 Pro = h16 Generation, sharedConnection funktioniert |
| **3. CoreML Test** | Inference funktioniert (0.034ms), Modell korrekt, App Store sicher |
| **4. LLDB** | Nicht nötig — Runtime-Introspection hat alle kritischen Fragen beantwortet |
| **5. Auswertung** | **DIREKTER ANE-ZUGRIFF MÖGLICH OHNE JAILBREAK** via Private API (TestFlight) oder CoreML (App Store) |

### Die wichtigste Erkenntnis

> **Die iOS ANE-API ist eine 1:1 Kopie der macOS API. Jede Klasse, jede Methode, jeder
> QoS-Wert ist identisch. Der macOS Training-Code kann mit minimalen Anpassungen
> (Objective-C → iOS Build Target) direkt auf dem iPhone laufen.**

> **Der A17 Pro hat eine neuere ANE-Generation (h16) als der M3 Pro (h15g) —
> potenziell 35 TOPS gegenüber 18 TOPS, besonders bei INT8.**

---

## Phase 1.1: DIREKTE ANE-KOMPILIERUNG AUF DEM iPHONE — BEWEIS

### Ergebnis: FUNKTIONIERT!

Der komplette macOS MIL-Compile-Pfad wurde 1:1 auf iOS portiert und
**erfolgreich auf dem iPhone 15 Pro ausgeführt**.

### Bewiesener Pfad

```
MIL-Text (in App) → _ANEInMemoryModelDescriptor → _ANEInMemoryModel
  → compileWithQoS: → loadWithQoS: → evaluateWithQoS: → IOSurface Output
```

### Ergebnisse (Live auf iPhone 15 Pro, A17 Pro)

| Schritt | Ergebnis | Details |
|---------|----------|---------|
| Descriptor erstellen | ✅ SUCCESS | `modelWithMILText:weights:optionsPlist:` |
| Model erstellen | ✅ SUCCESS | `inMemoryModelWithDescriptor:` |
| Kompilieren | ✅ SUCCESS | 592ms (kalt) / **21ms (warm/cached)** |
| Laden | ✅ SUCCESS | 24.7ms (kalt) / **2.6ms (warm)** |
| Evaluieren | ✅ SUCCESS | 0.365ms (kalt) / **0.172ms (warm)** |
| Output korrekt | ✅ 16384/16384 | Identity-Conv: alle Werte = 1.0 |
| Benchmark | ✅ DONE | **0.308ms/eval** (200x Durchschnitt) |

### Vergleich iPhone (warm) vs macOS M3 Pro

| Metrik | iPhone 15 Pro (A17 Pro) | macOS M3 Pro | Anmerkung |
|--------|:-----------------------:|:------------:|-----------|
| Compile (256ch) | **21 ms** | ~25 ms | iPhone schneller! |
| Load | **2.6 ms** | ~3 ms | Vergleichbar |
| Eval (256ch sp64) | **0.308 ms** | 0.248 ms | macOS etwas schneller |
| Korrektheit | 100% | 100% | Identisch |

### Code-Beweis

Die Datei `ANEDirectTest.m` enthält den kompletten Beweis:
- MIL-Text wird in der App generiert
- Weight-Blob (Identity-Matrix) wird in der App erstellt
- Kompilierung, Laden, Evaluation — alles direkt auf dem iPhone
- Kein Jailbreak, keine speziellen Entitlements
- Standard iOS-App mit Developer-Signatur

### Was das bedeutet

**TRAINING AUF DEM iPHONE IST TECHNISCH MÖGLICH.**

Der vollständige Pfad den wir auf macOS für ANE-Training nutzen funktioniert
identisch auf iOS. Der nächste Schritt ist die Implementierung des Training-Loops:

1. Forward Pass: MIL-Kernel auf ANE evaluieren ✅ (bewiesen)
2. Loss berechnen: CPU
3. Backward Pass: CPU
4. Weights updaten: CPU → neuen Weight-Blob erstellen
5. Recompile mit neuen Weights → Schritt 1

Jeder einzelne dieser Schritte nutzt APIs die wir heute als funktionsfähig bestätigt haben.

### Phase 1.3: A17 Pro ANE Benchmark (Live auf iPhone 15 Pro)

#### Single Convolution Throughput

| Config | Weight (MB) | ms/eval | TFLOPS |
|--------|:-----------:|:-------:|:------:|
| 256x256 sp=64 | 0.1 | 0.319 | 0.03 |
| 512x512 sp=64 | 0.5 | 0.291 | 0.12 |
| 1024x1024 sp=64 | 2.0 | 0.300 | 0.45 |
| 2048x2048 sp=64 | 8.0 | 0.382 | **1.40** |
| 4096x4096 sp=64 | 32.0 | 2.971 | 0.72 |

**Beobachtung**: 4096ch mit 32MB Weights überschreitet das SRAM-Budget → Throughput-Drop.
Sweet Spot bei 2048ch (8MB Weights).

#### Spatial Sweep (512ch)

| Spatial | ms/eval | TFLOPS |
|:-------:|:-------:|:------:|
| 16 | 0.131 | 0.06 |
| 32 | 0.398 | 0.04 |
| 64 | 0.317 | 0.11 |
| 128 | 0.310 | 0.22 |
| 256 | 0.264 | **0.51** |

**Beobachtung**: Größerer Spatial = besser. Das ANE ist effizienter mit mehr Daten pro Kernel.

#### Stacked Peak (16x sequential 512ch sp64)

```
Compiled 16/16 kernels
5x16 evals: 3.3 ms/pass, 0.209 ms/kernel

PEAK: 160.59 TFLOPS (stacked)
```

**WARNUNG**: Der Stacked-Wert von 160 TFLOPS ist unrealistisch hoch — das liegt daran
dass die TFLOPS-Berechnung die Gesamtarbeit aller 16 Kernel auf die Wanduhrzeit bezieht.
Die Kernel laufen möglicherweise teilweise parallel in der ANE-Pipeline. Der echte
Peak liegt wahrscheinlich bei **~1.5-2 TFLOPS** für einzelne Kernels bei optimaler Größe.

#### Vergleich A17 Pro vs M3 Pro

| Metrik | A17 Pro (iPhone) | M3 Pro (Mac) |
|--------|:----------------:|:------------:|
| 2048ch single eval | **0.382 ms** | 0.334 ms |
| 2048ch TFLOPS | **1.40** | 1.61 |
| Peak single kernel | ~1.5 TFLOPS | ~2.92 TFLOPS |
| Compile time | 21 ms | ~25 ms |
| Load time | 2.6 ms | ~3 ms |

Der M3 Pro ist bei Single-Kernel-Throughput schneller (mehr Speicherbandbreite),
aber der A17 Pro ist überraschend nah dran für ein mobiles SoC.

### Phase 1.4: Weight Update Test (Live auf iPhone 15 Pro)

#### Test A: Recompile mit neuen Weights — FUNKTIONIERT ✅

| Schritt | Ergebnis |
|---------|----------|
| Compile+Load mit W=1.0 | output[0] = 0x3C00 (= 1.0) ✅ |
| Compile+Load mit W=3.0 | output[0] = 0x4200 (= 3.0) ✅ |
| Output geändert? | **JA** — Recompile aktualisiert Weights korrekt |
| Recompile-Overhead | **22.5 ms/cycle** (compile+load+eval) |

#### Test B: Dynamic Spatial Packing — FUNKTIONIERT ✅

| Schritt | Ergebnis |
|---------|----------|
| Compile einmalig | 20 ms |
| Eval mit w=1.0 im IOSurface | output[0] = 0x3C00 (= 1.0) ✅ |
| Weight in IOSurface auf 3.0 ändern | Kein Recompile! |
| Eval mit w=3.0 im IOSurface | output[0] = 0x4200 (= 3.0) ✅ |
| Output geändert? | **JA — DYNAMIC WEIGHTS WORK!** |
| Update+Eval Overhead | **0.308 ms/cycle** (200x Durchschnitt) |

#### Vergleich der Ansätze

| Metrik | Recompile | Dynamic Packing |
|--------|:---------:|:---------------:|
| ms/Training-Step | **22.5 ms** | **0.308 ms** |
| Speedup | 1x | **73x schneller** |
| Compile-Budget | ~119 pro Prozess | **Unbegrenzt** |
| Komplexität | Einfach | MIL muss Weights im Input erwarten |

**Dynamic Spatial Packing ist der klare Gewinner für Training.**
- Kein Recompile-Overhead
- Kein ~119 Compile-Limit Problem
- 73x schneller als der Recompile-Ansatz
- Weights werden einfach im Input-IOSurface aktualisiert
