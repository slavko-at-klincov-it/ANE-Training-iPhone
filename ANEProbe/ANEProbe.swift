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
    @State private var output = "Starting probe..."
    @State private var isRunning = true

    var body: some View {
        VStack(spacing: 16) {
            Text("ANE Probe").font(.title.bold())

            Button("Probe ANE") {
                isRunning = true
                runProbe()
            }
            .disabled(isRunning)
            .buttonStyle(.borderedProminent)

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
        .onAppear { runProbe() }
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

        // 9. DIRECT ANE TEST
        lines.append("\n=== DIRECT ANE COMPILE+EVAL TEST ===")
        if let result = ane_direct_test() {
            lines.append(result as String)
        }

        // 10. WEIGHT UPDATE TEST
        lines.append("\n=== WEIGHT UPDATE TEST ===")
        if let result = ane_weight_test() {
            lines.append(result as String)
        }

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
