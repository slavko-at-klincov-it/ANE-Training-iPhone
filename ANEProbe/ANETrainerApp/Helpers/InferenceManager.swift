// InferenceManager.swift — Swift wrapper for C inference API
import Foundation
import os.log

private let logger = Logger(subsystem: "com.klincov.anetrainer", category: "Inference")

@MainActor
class InferenceManager: ObservableObject {
    @Published var isLoaded = false
    @Published var isGenerating = false
    @Published var modelInfo = ""

    private var state: OpaquePointer?

    func loadModel() {
        guard !isLoaded else { return }
        Task.detached { [weak self] in
            // Try checkpoint first, then bundle
            let docs = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first!
            let ckptPath = (docs as NSString).appendingPathComponent("ane_ckpt_0.bin")
            let fm = FileManager.default

            var weightsPath: String? = nil
            if fm.fileExists(atPath: ckptPath) {
                weightsPath = ckptPath
            } else if let bundled = Bundle.main.path(forResource: "stories110M", ofType: "bin") {
                weightsPath = bundled
            }

            let s = ane_inference_init(weightsPath, nil)
            await MainActor.run {
                guard let self else { return }
                self.state = s
                self.isLoaded = s != nil
                if s != nil {
                    if weightsPath?.contains("ckpt") == true {
                        self.modelInfo = "Checkpoint geladen"
                    } else if weightsPath != nil {
                        self.modelInfo = "Pretrained Modell"
                    } else {
                        self.modelInfo = "Zufaellige Gewichte (Test)"
                    }
                }
            }
        }
    }

    func generate(prompt: String, maxTokens: Int, temperature: Float) -> String {
        guard let s = state else { return "[Kein Modell geladen]" }
        isGenerating = true
        defer { isGenerating = false }

        guard let result = ane_generate(s, prompt, Int32(maxTokens), temperature) else {
            return "[Fehler bei Generierung]"
        }
        let text = String(cString: result)
        free(result)
        return text
    }

    func unload() {
        if let s = state { ane_inference_free(s) }
        state = nil
        isLoaded = false
        modelInfo = ""
    }

    deinit {
        if let s = state { ane_inference_free(s) }
    }
}
