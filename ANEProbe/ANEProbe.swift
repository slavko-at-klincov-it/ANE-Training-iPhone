import SwiftUI
import CoreML
import Foundation
import os.log

private let logger = Logger(subsystem: "com.klincov.aneprobe", category: "probe")

@main
struct ANEProbeApp: App {
    var body: some Scene {
        WindowGroup {
            ANEProbeView()
        }
    }
}

struct ANEProbeView: View {
    @State private var output = "Starting overnight training..."
    @State private var isRunning = true

    var body: some View {
        VStack(spacing: 16) {
            Text("ANE Training").font(.title.bold())

            HStack {
                Button("Overnight Train (8 hours)") {
                    isRunning = true
                    runOvernightTraining()
                }
                .disabled(isRunning)
                .buttonStyle(.borderedProminent)

                Button("Quick Probe") {
                    isRunning = true
                    runProbe()
                }
                .disabled(isRunning)
                .buttonStyle(.bordered)
            }

            ScrollView {
                Text(output)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .textSelection(.enabled)
            }
            .background(Color(.systemGray6))
            .cornerRadius(8)
        }
        .padding()
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            runEndToEndTest()
        }
    }

    func runEndToEndTest() {
        Task.detached {
            var lines: [String] = []
            lines.append("=== END-TO-END TEST ===\n")

            // 1. Tokenizer test
            lines.append("--- 1. TOKENIZER ---")
            if let result = ane_tokenizer_test() {
                lines.append(result as String)
            }

            // 2. Quick training (20 steps)
            lines.append("\n--- 2. TRAINING (20 steps) ---")
            if let result = ane_training_engine_test() {
                lines.append(result as String)
            }

            // 3. Inference test
            lines.append("\n--- 3. INFERENCE ---")
            if let result = ane_inference_test() {
                lines.append(result as String)
            }

            lines.append("\n=== END-TO-END COMPLETE ===")

            let result = lines.joined(separator: "\n")
            for line in result.split(separator: "\n", omittingEmptySubsequences: false) {
                let s = String(line)
                logger.notice("\(s, privacy: .public)")
                fputs("ANEPROBE: \(s)\n", stderr)
            }
            fputs("ANEPROBE: === DONE ===\n", stderr)
            await MainActor.run {
                self.output = result
                self.isRunning = false
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }

    func runOvernightTraining() {
        Task.detached {
            // TIME-BASED training: 8 hours overnight
            let result = ane_timed_training(8.0)
            let resultStr = result as String? ?? "nil"
            for line in resultStr.split(separator: "\n", omittingEmptySubsequences: false) {
                let s = String(line)
                logger.notice("\(s, privacy: .public)")
                fputs("ANEPROBE: \(s)\n", stderr)
            }
            await MainActor.run {
                output = resultStr
                isRunning = false
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }

    func runProbe() {
        Task.detached {
            let result = ANEProber.probeAll()
            // Print to stderr which shows in console
            for line in result.split(separator: "\n", omittingEmptySubsequences: false) {
                let s = String(line)
                logger.notice("\(s, privacy: .public)")
                fputs("ANEPROBE: \(s)\n", stderr)
            }
            fputs("ANEPROBE: === COMPLETE ===\n", stderr)
            await MainActor.run {
                output = result
                isRunning = false
            }
        }
    }
}

class ANEProber {
    static func probeAll() -> String {
        var lines: [String] = []

        // 1. Load AppleNeuralEngine only - safest
        lines.append("=== LOADING FRAMEWORKS ===")
        let frameworks = [
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
            "/System/Library/PrivateFrameworks/ANEServices.framework/ANEServices",
            "/System/Library/PrivateFrameworks/ANEClientSignals.framework/ANEClientSignals",
            "/System/Library/PrivateFrameworks/ParavirtualizedANE.framework/ParavirtualizedANE"
        ]

        for fw in frameworks {
            let name = (fw as NSString).lastPathComponent
            if dlopen(fw, RTLD_LAZY | RTLD_NOLOAD) != nil {
                lines.append("  PRELOADED \(name)")
            } else if dlopen(fw, RTLD_LAZY) != nil {
                lines.append("  LOADED \(name)")
            } else {
                let err = String(cString: dlerror())
                lines.append("  FAILED \(name): \(err)")
            }
        }

        // 2. Check known ANE classes directly (avoid objc_copyClassList crash)
        lines.append("\n=== ANE CLASSES ===")
        let allKnown = [
            "_ANEBuffer", "_ANEChainingRequest", "_ANEClient", "_ANECloneHelper",
            "_ANEDaemonConnection", "_ANEDataReporter", "_ANEDeviceController",
            "_ANEDeviceInfo", "_ANEErrors", "_ANEHashEncoding",
            "_ANEInMemoryModel", "_ANEInMemoryModelDescriptor",
            "_ANEInputBuffersReady", "_ANEIOSurfaceObject", "_ANEIOSurfaceOutputSets",
            "_ANELog", "_ANEModel", "_ANEModelInstanceParameters", "_ANEModelToken",
            "_ANEOutputSetEnqueue", "_ANEPerformanceStats", "_ANEPerformanceStatsIOSurface",
            "_ANEProcedureData", "_ANEProgramForEvaluation", "_ANEProgramIOSurfacesMapper",
            "_ANEQoSMapper", "_ANERequest", "_ANESandboxingHelper",
            "_ANESharedEvents", "_ANESharedSignalEvent", "_ANESharedWaitEvent",
            "_ANEStrings", "_ANEVirtualClient", "_ANEWeight",
            "ANEServicesLog",
            // iOS-only candidates
            "_ANEVirtualModel", "_ANEVirtualPlatformClient",
            "_ANEAnalyticsGroup", "_ANEAnalyticsLayer",
            "_ANEAnalyticsProcedure", "_ANEAnalyticsTask", "_ANECompilerAnalytics"
        ]

        var found: [String] = []
        var missing: [String] = []
        for name in allKnown {
            if objc_getClass(name) != nil {
                found.append(name)
            } else {
                missing.append(name)
            }
        }
        lines.append("  Present (\(found.count)): \(found.joined(separator: ", "))")
        if !missing.isEmpty {
            lines.append("  Missing (\(missing.count)): \(missing.joined(separator: ", "))")
        }

        // 3. Dump methods for key classes
        let keyClasses = [
            "_ANEClient", "_ANEInMemoryModel", "_ANEInMemoryModelDescriptor",
            "_ANERequest", "_ANEIOSurfaceObject", "_ANEChainingRequest",
            "_ANEDeviceInfo", "_ANEPerformanceStats", "_ANEQoSMapper",
            "_ANESharedEvents", "_ANEWeight", "_ANEBuffer",
            "_ANESandboxingHelper", "_ANEDaemonConnection"
        ]

        for className in keyClasses {
            lines.append("\n=== \(className) ===")
            guard let cls = objc_getClass(className) as? AnyClass else {
                lines.append("  NOT FOUND")
                continue
            }

            // Instance methods
            var mCount: UInt32 = 0
            if let methods = class_copyMethodList(cls, &mCount) {
                var names: [String] = []
                for j in 0..<Int(mCount) {
                    names.append(NSStringFromSelector(method_getName(methods[j])))
                }
                free(methods)
                names.sort()
                lines.append("  Instance (\(names.count)):")
                for n in names { lines.append("    -\(n)") }
            }

            // Class methods
            let meta: AnyClass = object_getClass(cls)!
            if let methods = class_copyMethodList(meta, &mCount) {
                var names: [String] = []
                for j in 0..<Int(mCount) {
                    names.append(NSStringFromSelector(method_getName(methods[j])))
                }
                free(methods)
                names.sort()
                lines.append("  Class (\(names.count)):")
                for n in names { lines.append("    +\(n)") }
            }
        }

        // 4. Try _ANEDeviceInfo - safe integer calls
        lines.append("\n=== DEVICE INFO PROBE ===")
        if let deviceInfoClass = objc_getClass("_ANEDeviceInfo") as? AnyClass {
            // String-returning
            for name in ["aneArchitectureType", "aneSubType", "productName", "buildVersion"] {
                let sel = NSSelectorFromString(name)
                if let _ = class_getClassMethod(deviceInfoClass, sel) {
                    let result = (deviceInfoClass as AnyObject).perform(sel)
                    lines.append("  \(name) = \(result?.takeUnretainedValue() ?? "nil" as AnyObject)")
                }
            }
            // Int-returning
            for name in ["hasANE", "numANECores", "numANEs", "aneBoardType", "isVirtualMachine", "isExcessivePowerDrainWhenIdle"] {
                let sel = NSSelectorFromString(name)
                if let method = class_getClassMethod(deviceInfoClass, sel) {
                    typealias Fn = @convention(c) (AnyObject, Selector) -> Int
                    let f = unsafeBitCast(method_getImplementation(method), to: Fn.self)
                    lines.append("  \(name) = \(f(deviceInfoClass, sel))")
                }
            }
        }

        // 5. QoS values
        lines.append("\n=== QoS MAPPER PROBE ===")
        if let qosClass = objc_getClass("_ANEQoSMapper") as? AnyClass {
            for name in ["aneRealTimeTaskQoS", "aneUserInteractiveTaskQoS",
                         "aneUserInitiatedTaskQoS", "aneDefaultTaskQoS",
                         "aneUtilityTaskQoS", "aneBackgroundTaskQoS"] {
                let sel = NSSelectorFromString(name)
                if let method = class_getClassMethod(qosClass, sel) {
                    typealias Fn = @convention(c) (AnyObject, Selector) -> Int
                    let f = unsafeBitCast(method_getImplementation(method), to: Fn.self)
                    lines.append("  \(name) = \(f(qosClass, sel))")
                }
            }
        }

        // 6. ANE Client probe - this is the critical test
        lines.append("\n=== ANE CLIENT PROBE ===")
        if let clientClass = objc_getClass("_ANEClient") as? AnyClass {
            let sel = NSSelectorFromString("sharedConnection")
            if let _ = class_getClassMethod(clientClass, sel) {
                lines.append("  sharedConnection: METHOD EXISTS")
                // Actually try to get it
                let result = (clientClass as AnyObject).perform(sel)
                if let client = result?.takeUnretainedValue() {
                    lines.append("  sharedConnection: GOT \(type(of: client))")
                    // isVirtualClient
                    let vSel = NSSelectorFromString("isVirtualClient")
                    if let m = class_getInstanceMethod(type(of: client) as AnyClass, vSel) {
                        typealias Fn = @convention(c) (AnyObject, Selector) -> Bool
                        let f = unsafeBitCast(method_getImplementation(m), to: Fn.self)
                        lines.append("  isVirtualClient = \(f(client as AnyObject, vSel))")
                    }
                } else {
                    lines.append("  sharedConnection: RETURNED NIL")
                }
            } else {
                lines.append("  sharedConnection: NOT FOUND")
            }
        }

        // 7. Sandbox helper
        lines.append("\n=== SANDBOX CHECK ===")
        if let sbClass = objc_getClass("_ANESandboxingHelper") as? AnyClass {
            let sel = NSSelectorFromString("canAccessPathAt:methodName:error:")
            if let _ = class_getClassMethod(sbClass, sel) {
                lines.append("  canAccessPathAt:methodName:error: EXISTS")
            } else {
                lines.append("  canAccessPathAt:methodName:error: NOT FOUND")
            }
        }

        // 8. CoreML inference test
        lines.append("\n=== COREML INFERENCE TEST ===")
        lines.append(testCoreMLInference())

        // PHASE 2: Layer tests
        lines.append("\n=== PHASE 2: RMSNORM TEST ===")
        if let result = ane_rmsnorm_test() {
            lines.append(result as String)
        }

        lines.append("\n=== PHASE 2: LINEAR TEST ===")
        if let result = ane_linear_test() {
            lines.append(result as String)
        }

        lines.append("\n=== PHASE 2: ATTENTION TEST ===")
        if let result = ane_attention_test() {
            lines.append(result as String)
        }

        lines.append("\n=== PHASE 2: FFN (SwiGLU) TEST ===")
        if let result = ane_ffn_test() {
            lines.append(result as String)
        }

        lines.append("\n=== PHASE 2: FFN BACKWARD TEST ===")
        if let result = ane_ffn_bwd_test() {
            lines.append(result as String)
        }

        lines.append("\n=== PHASE 2: ATTENTION BACKWARD TEST ===")
        if let result = ane_attention_bwd_test() {
            lines.append(result as String)
        }

        lines.append("\n=== PHASE 2.3: TRAINING STEP TEST ===")
        if let result = ane_train_step_test() {
            lines.append(result as String)
        }

        // PHASE 3: New components
        lines.append("\n=== DYNAMIC SPATIAL PACKING TRAIN ===")
        if let result = ane_dynamic_train_test() {
            lines.append(result as String)
        }

        lines.append("\n=== CHECKPOINT TEST ===")
        ane_checkpoint_test()
        lines.append("  (Checkpoint test logs to stderr)")

        lines.append("\n=== DATA PIPELINE TEST ===")
        if let result = ane_data_pipeline_test() {
            lines.append(result as String)
        }

        lines.append("\n=== FULL TRAINING ENGINE TEST ===")
        if let result = ane_training_engine_test() {
            lines.append(result as String)
        }

        lines.append("\n=== THERMAL STRESS TEST ===")
        if let result = ane_thermal_test() { lines.append(result as String) }

        /* Phase 1/1.5 tests — disabled for speed, uncomment to re-run
        lines.append("\n=== DIRECT ANE COMPILE+EVAL TEST ===")
        if let result = ane_direct_test() { lines.append(result as String) }
        lines.append("\n=== WEIGHT UPDATE TEST ===")
        if let result = ane_weight_test() { lines.append(result as String) }
        lines.append("\n=== PHASE 1.5: SRAM BOUNDARY PROBE ===")
        if let result = ane_re_sram_probe() { lines.append(result as String) }
        lines.append("\n=== PHASE 1.5: MIL OP COVERAGE ===")
        if let result = ane_re_op_coverage() { lines.append(result as String) }
        lines.append("\n=== PHASE 1.5: PERFORMANCE STATS ===")
        if let result = ane_re_perf_stats() { lines.append(result as String) }
        lines.append("\n=== PHASE 1.5: COMPILE LIMITS ===")
        if let result = ane_re_compile_limits() { lines.append(result as String) }
        */

        lines.append("\n=== PROBE DONE ===")
        return lines.joined(separator: "\n")
    }

    static func testCoreMLInference() -> String {
        var lines: [String] = []

        guard let modelURL = Bundle.main.url(forResource: "IdentityConv", withExtension: "mlmodelc") else {
            lines.append("  IdentityConv.mlmodelc NOT FOUND in bundle")
            return lines.joined(separator: "\n")
        }
        lines.append("  Model found: \(modelURL.lastPathComponent)")

        // Test with different compute units
        let configs: [(String, MLComputeUnits)] = [
            ("ALL (prefer ANE)", .all),
            ("CPU only", .cpuOnly),
            ("CPU+GPU", .cpuAndGPU),
        ]

        for (label, units) in configs {
            let config = MLModelConfiguration()
            config.computeUnits = units

            do {
                let model = try MLModel(contentsOf: modelURL, configuration: config)
                lines.append("  [\(label)] Model loaded OK")

                // Create input: [1, 256, 1, 64] filled with 1.0
                let inputArray = try MLMultiArray(shape: [1, 256, 1, 64], dataType: .float16)
                let ptr = inputArray.dataPointer.bindMemory(to: UInt16.self, capacity: inputArray.count)
                // FP16 value 1.0 = 0x3C00
                for i in 0..<inputArray.count {
                    ptr[i] = 0x3C00
                }

                let provider = try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: inputArray)])

                // Warmup
                let _ = try model.prediction(from: provider)

                // Benchmark
                let iterations = 100
                let start = CFAbsoluteTimeGetCurrent()
                for _ in 0..<iterations {
                    let _ = try model.prediction(from: provider)
                }
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                let msPerEval = (elapsed / Double(iterations)) * 1000.0

                // Verify output
                let result = try model.prediction(from: provider)
                if let outArray = result.featureValue(for: "identity_conv")?.multiArrayValue {
                    let outPtr = outArray.dataPointer.bindMemory(to: UInt16.self, capacity: outArray.count)
                    let firstVal = outPtr[0]
                    lines.append("  [\(label)] \(iterations)x inference: \(String(format: "%.3f", msPerEval)) ms/eval, output[0]=0x\(String(firstVal, radix: 16))")
                } else {
                    lines.append("  [\(label)] \(iterations)x inference: \(String(format: "%.3f", msPerEval)) ms/eval")
                }
            } catch {
                lines.append("  [\(label)] ERROR: \(error.localizedDescription)")
            }
        }

        return lines.joined(separator: "\n")
    }
}
