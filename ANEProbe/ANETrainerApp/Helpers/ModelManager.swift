// ModelManager.swift — Checkpoint scanning, export, import
import Foundation

struct SavedModel: Identifiable {
    let id = UUID()
    let path: String
    let filename: String
    let step: Int
    let loss: Float
    let date: Date
    let sizeBytes: UInt64
    var sizeMB: String { String(format: "%.1f MB", Double(sizeBytes) / 1e6) }
}

class ModelManager: ObservableObject {
    @Published var models: [SavedModel] = []

    private var docsDir: String {
        NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first!
    }

    func scan() {
        let fm = FileManager.default
        models = []
        guard let files = try? fm.contentsOfDirectory(atPath: docsDir) else { return }

        for file in files where file.hasSuffix(".bin") && file.contains("ckpt") {
            let path = (docsDir as NSString).appendingPathComponent(file)
            guard let data = fm.contents(atPath: path), data.count >= 128 else { continue }

            // Read BLZT header
            let header = data.withUnsafeBytes { buf -> (step: Int, loss: Float)? in
                guard let ptr = buf.baseAddress?.assumingMemoryBound(to: UInt32.self) else { return nil }
                let magic = ptr[0]
                guard magic == 0x424C5A54 else { return nil } // "BLZT"
                let step = Int(ptr.advanced(by: 2).pointee)
                let lossPtr = buf.baseAddress!.advanced(by: 44).assumingMemoryBound(to: Float.self)
                return (step: step, loss: lossPtr.pointee)
            }
            guard let h = header else { continue }

            let attrs = try? fm.attributesOfItem(atPath: path)
            let date = attrs?[.modificationDate] as? Date ?? Date()
            let size = attrs?[.size] as? UInt64 ?? 0

            models.append(SavedModel(path: path, filename: file, step: h.step, loss: h.loss, date: date, sizeBytes: size))
        }

        models.sort { $0.date > $1.date }
    }

    func delete(_ model: SavedModel) {
        try? FileManager.default.removeItem(atPath: model.path)
        scan()
    }

    func exportURL(_ model: SavedModel) -> URL {
        URL(fileURLWithPath: model.path)
    }
}
