// ANETrainer.swift — High-level public API for on-device transformer training on Apple Neural Engine
//
// Wraps the low-level C training engine (ANETrainingEngine.h) into a clean Swift interface
// for easy integration by app developers. Supports SwiftUI via ObservableObject.
//
// Usage:
//
//   let trainer = ANETrainer()
//
//   // Collect training data during normal app usage
//   trainer.addTokens([1, 234, 567, ...])   // pre-tokenized
//   trainer.addText("User wrote this today") // auto-tokenized (requires tokenizer.bin)
//
//   // Train immediately (foreground)
//   trainer.train(steps: 1000) { result in
//       print("Done: \(result.steps) steps, loss \(result.bestLoss)")
//   }
//
//   // Or schedule overnight training
//   trainer.scheduleOvernight(hours: 8)
//
//   // Check status next morning
//   print(trainer.status)     // .completed
//   print(trainer.bestLoss)   // 9.41
//   print(trainer.totalSteps) // 64000

import Foundation
import BackgroundTasks
import os.log

private let logger = Logger(subsystem: "com.klincov.aneprobe", category: "ANETrainer")

// MARK: - ANETrainer

/// High-level API for on-device transformer training on Apple Neural Engine.
///
/// `ANETrainer` manages the full training lifecycle: data collection, training execution,
/// checkpoint persistence, and background scheduling. It exposes `@Published` properties
/// for live SwiftUI binding and provides both step-based and time-based training modes.
///
/// All public methods are thread-safe. Training runs on a background queue; published
/// state updates are dispatched to the main thread automatically.
@objc public class ANETrainer: NSObject, ObservableObject {

    // MARK: - Training Status

    /// Current state of the training engine.
    public enum TrainingStatus: String, CustomStringConvertible {
        case idle            = "Idle"
        case initializing    = "Initializing..."
        case compiling       = "Compiling ANE kernels..."
        case training        = "Training"
        case savingCheckpoint = "Saving checkpoint..."
        case paused          = "Paused"
        case completed       = "Completed"
        case error           = "Error"

        public var description: String { rawValue }
    }

    // MARK: - Published State (SwiftUI-bindable)

    /// Current training status.
    @Published public private(set) var status: TrainingStatus = .idle

    /// Number of training steps completed in the current or most recent run.
    @Published public private(set) var currentStep: Int = 0

    /// Total steps completed across all runs (persisted via checkpoint).
    @Published public private(set) var totalSteps: Int = 0

    /// Best (lowest) loss observed during training.
    @Published public private(set) var bestLoss: Float = .infinity

    /// Loss from the most recent training step.
    @Published public private(set) var currentLoss: Float = 0

    /// Approximate training throughput.
    @Published public private(set) var stepsPerSecond: Double = 0

    /// Last error message, if `status == .error`.
    @Published public private(set) var lastError: String?

    // MARK: - Configuration

    /// Training hyperparameters and scheduling options.
    public struct Config {
        /// Adam learning rate (default: 3e-4).
        public var learningRate: Float = 3e-4

        /// Number of gradient accumulation steps before each Adam update (default: 4).
        public var accumulationSteps: Int = 4

        /// Whether background training requires the device to be charging (default: true).
        public var requiresCharging: Bool = true

        /// Save a checkpoint every N steps (default: 500).
        public var checkpointInterval: Int = 500

        /// Default configuration suitable for overnight training.
        public static let `default` = Config()

        public init(
            learningRate: Float = 3e-4,
            accumulationSteps: Int = 4,
            requiresCharging: Bool = true,
            checkpointInterval: Int = 500
        ) {
            self.learningRate = learningRate
            self.accumulationSteps = accumulationSteps
            self.requiresCharging = requiresCharging
            self.checkpointInterval = checkpointInterval
        }
    }

    /// Training result returned on completion.
    public struct TrainingResult {
        /// Number of training steps executed in this run.
        public let steps: Int

        /// Wall-clock duration of the run.
        public let duration: TimeInterval

        /// Best (lowest) loss observed.
        public let bestLoss: Float

        /// Loss at the final step.
        public let finalLoss: Float

        /// Number of Adam optimizer updates performed (steps / accumulationSteps).
        public let adamUpdates: Int
    }

    // MARK: - Properties

    /// Active configuration. Set before calling `train(...)` or `scheduleOvernight(...)`.
    public var config: Config = .default

    /// Background task identifier for BGProcessingTask registration.
    public static let backgroundTaskIdentifier = "com.klincov.aneprobe.anetrainer"

    // MARK: - Private State

    /// Serial queue for thread-safe data collection (token appending).
    private let dataQueue = DispatchQueue(label: "com.klincov.aneprobe.anetrainer.data")

    /// Background queue for training execution.
    private let trainQueue = DispatchQueue(label: "com.klincov.aneprobe.anetrainer.train", qos: .userInitiated)

    /// Opaque pointer to the C training state (`ANETrainState *`).
    private var trainState: OpaquePointer?

    /// Flag to request graceful stop.
    private var stopRequested = false

    /// Cached token count from the data file.
    private var _collectedTokenCount: Int = 0

    // MARK: - File Paths

    /// App Documents directory.
    private static var documentsDirectory: String {
        NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first!
    }

    /// Path to the pre-tokenized training data binary (uint16_t tokens, mmap-compatible).
    private var trainingDataPath: String {
        (Self.documentsDirectory as NSString).appendingPathComponent("ane_training_data.bin")
    }

    /// Path to the latest checkpoint file.
    private var checkpointPath: String {
        String(cString: ane_checkpoint_path(0))
    }

    /// Path to the CSV training log.
    private var trainingLogPath: String {
        (Self.documentsDirectory as NSString).appendingPathComponent("ane_training_log.csv")
    }

    // MARK: - Init

    public override init() {
        super.init()
        // Count existing tokens if the data file exists
        _collectedTokenCount = Self.tokenCountInFile(atPath: trainingDataPath)
        logger.info("ANETrainer initialized — \(self._collectedTokenCount) tokens on disk")
    }

    deinit {
        if let state = trainState {
            ane_train_free(state)
            trainState = nil
        }
    }

    // MARK: - Data Collection

    /// Add raw text for training. Text is tokenized and appended to the training data file.
    ///
    /// Call this during normal app usage to collect personalization data. The text is
    /// tokenized using the bundled `tokenizer.bin` and the resulting token IDs are
    /// appended to `Documents/ane_training_data.bin`.
    ///
    /// - Note: If no tokenizer is available yet, this method logs a warning and does nothing.
    ///   Use ``addTokens(_:)`` as a fallback with pre-tokenized data.
    ///
    /// - Parameter text: Raw text to tokenize and store for training.
    public func addText(_ text: String) {
        // TODO: Integrate tokenizer once ANETokenizer C API is available.
        //
        // Expected implementation:
        //   let tokenizer = ane_tokenizer_load_from_bundle()
        //   var len: Int32 = 0
        //   guard let tokPtr = ane_tokenize(tokenizer, text, &len) else { return }
        //   let tokens = Array(UnsafeBufferPointer(start: tokPtr, count: Int(len)))
        //   ane_tokenizer_free(tokenizer)
        //   addTokens(tokens)
        //
        // For now, fall back to a simple byte-level encoding so data collection works
        // without a tokenizer binary. Each UTF-8 byte maps to a token ID (offset by 3
        // to avoid special tokens 0-2). This is lossy but allows testing the pipeline.

        let utf8 = Array(text.utf8)
        if utf8.isEmpty { return }

        // Byte-level fallback: token = byte_value + 3 (reserving 0=pad, 1=bos, 2=eos)
        var tokens = [UInt16]()
        tokens.reserveCapacity(utf8.count + 2)
        tokens.append(1) // BOS
        for byte in utf8 {
            tokens.append(UInt16(byte) + 3)
        }
        tokens.append(2) // EOS

        logger.info("addText: \(utf8.count) bytes → \(tokens.count) fallback tokens (tokenizer not yet available)")
        addTokens(tokens)
    }

    /// Add pre-tokenized data (array of token IDs).
    ///
    /// Tokens are appended to `Documents/ane_training_data.bin` in the standard
    /// uint16_t binary format used by the C training engine. This file is memory-mapped
    /// during training for zero-copy access.
    ///
    /// - Parameter tokens: Array of uint16 token IDs (e.g., from SentencePiece or BPE).
    public func addTokens(_ tokens: [UInt16]) {
        guard !tokens.isEmpty else { return }

        dataQueue.async { [weak self] in
            guard let self = self else { return }

            let path = self.trainingDataPath
            let fm = FileManager.default

            // Create file if it doesn't exist
            if !fm.fileExists(atPath: path) {
                fm.createFile(atPath: path, contents: nil)
            }

            guard let handle = FileHandle(forWritingAtPath: path) else {
                logger.error("Failed to open training data file for writing: \(path, privacy: .public)")
                return
            }

            defer { handle.closeFile() }
            handle.seekToEndOfFile()

            // Write uint16_t tokens directly
            let data = tokens.withUnsafeBufferPointer { buffer -> Data in
                let raw = UnsafeRawBufferPointer(buffer)
                return Data(raw)
            }
            handle.write(data)

            self._collectedTokenCount += tokens.count
            let count = self._collectedTokenCount
            logger.debug("Appended \(tokens.count) tokens — total: \(count)")
        }
    }

    /// Number of training tokens collected so far in the data file.
    public var collectedTokenCount: Int {
        dataQueue.sync { _collectedTokenCount }
    }

    /// Remove all collected training data. Cannot be called while training.
    public func clearTrainingData() {
        guard status == .idle || status == .completed || status == .error else {
            logger.warning("Cannot clear training data while status is \(self.status.rawValue, privacy: .public)")
            return
        }
        dataQueue.async { [weak self] in
            guard let self = self else { return }
            try? FileManager.default.removeItem(atPath: self.trainingDataPath)
            self._collectedTokenCount = 0
            logger.info("Training data cleared")
        }
    }

    // MARK: - Training Control

    /// Start training for a fixed number of steps (foreground).
    ///
    /// Training runs on a background thread. Published properties update on the main
    /// thread for SwiftUI binding. The completion handler fires on the main thread.
    ///
    /// - Parameters:
    ///   - steps: Number of training steps to execute.
    ///   - completion: Called when training finishes, stops, or encounters an error.
    public func train(steps: Int, completion: @escaping (TrainingResult) -> Void) {
        guard status == .idle || status == .completed || status == .paused || status == .error else {
            logger.warning("Cannot start training — status is \(self.status.rawValue, privacy: .public)")
            return
        }

        stopRequested = false
        updateStatus(.initializing)

        trainQueue.async { [weak self] in
            guard let self = self else { return }

            let startTime = CFAbsoluteTimeGetCurrent()

            // Initialize the C training engine
            guard self.initializeEngine() else {
                self.failWithError("Failed to initialize training engine")
                DispatchQueue.main.async {
                    completion(TrainingResult(steps: 0, duration: 0, bestLoss: .infinity, finalLoss: .infinity, adamUpdates: 0))
                }
                return
            }

            self.updateStatus(.training)

            // Training loop
            var stepsCompleted = 0
            var runBestLoss: Float = .infinity
            var lastLoss: Float = 0
            var stepTimes: [Double] = []

            for _ in 0..<steps {
                if self.stopRequested { break }

                // Check if ANE is compiling new kernels
                if let state = self.trainState, ane_train_is_compiling(state) {
                    self.updateStatus(.compiling)
                }

                let t0 = CFAbsoluteTimeGetCurrent()

                // Execute one training step via C API
                guard let state = self.trainState else { break }
                let loss = ane_train_step(state)
                let step = Int(ane_train_current_step(state))

                let dt = CFAbsoluteTimeGetCurrent() - t0
                stepTimes.append(dt)
                if stepTimes.count > 100 { stepTimes.removeFirst() }

                lastLoss = loss
                if loss < runBestLoss { runBestLoss = loss }
                stepsCompleted += 1

                // Update published state on main thread
                let avgTime = stepTimes.reduce(0, +) / Double(stepTimes.count)
                let sps = avgTime > 0 ? 1.0 / avgTime : 0
                let best = runBestLoss

                DispatchQueue.main.async {
                    self.currentStep = step
                    self.currentLoss = loss
                    self.bestLoss = min(self.bestLoss, best)
                    self.stepsPerSecond = sps
                    self.totalSteps = step
                    if self.status != .compiling {
                        self.status = .training
                    }
                }

                // Periodic checkpoint
                if step > 0 && step % self.config.checkpointInterval == 0 {
                    self.updateStatus(.savingCheckpoint)
                    ane_train_save(state)
                    self.updateStatus(.training)
                    logger.info("Checkpoint at step \(step), loss=\(loss)")
                }
            }

            // Final save
            if let state = self.trainState {
                ane_train_save(state)
            }

            let duration = CFAbsoluteTimeGetCurrent() - startTime
            let adamUpdates = stepsCompleted / max(self.config.accumulationSteps, 1)

            let wasStopped = self.stopRequested
            self.updateStatus(wasStopped ? .paused : .completed)

            let result = TrainingResult(
                steps: stepsCompleted,
                duration: duration,
                bestLoss: runBestLoss,
                finalLoss: lastLoss,
                adamUpdates: adamUpdates
            )

            logger.info("Training \(wasStopped ? "paused" : "completed"): \(stepsCompleted) steps, \(String(format: "%.1f", duration))s, best_loss=\(runBestLoss)")

            DispatchQueue.main.async { completion(result) }
        }
    }

    /// Start time-based training (foreground). Runs for the specified number of hours.
    ///
    /// Uses the C engine's `ane_timed_training()` which handles LR scheduling,
    /// periodic checkpointing, and logging internally.
    ///
    /// - Parameters:
    ///   - hours: Duration to train in hours (e.g., 8.0 for overnight).
    ///   - completion: Called when training finishes. Provides summary result.
    public func train(hours: Float, completion: @escaping (TrainingResult) -> Void) {
        guard status == .idle || status == .completed || status == .paused || status == .error else {
            logger.warning("Cannot start training — status is \(self.status.rawValue, privacy: .public)")
            return
        }

        stopRequested = false
        updateStatus(.initializing)

        trainQueue.async { [weak self] in
            guard let self = self else { return }

            let startTime = CFAbsoluteTimeGetCurrent()
            self.updateStatus(.training)

            // ane_timed_training handles the full loop internally:
            // - Init from checkpoint or random weights
            // - LR warmup + cosine decay
            // - Periodic checkpointing (every 500 steps)
            // - Plateau-based LR adjustment
            // - CSV logging
            let summary = ane_timed_training(hours)
            let duration = CFAbsoluteTimeGetCurrent() - startTime

            // Parse summary string for result (format: "Timed training done: X steps ...")
            let summaryStr = summary as String? ?? ""
            logger.info("Timed training result: \(summaryStr, privacy: .public)")

            // Read final state from checkpoint
            let finalStep = self.readCheckpointStep()
            let finalLoss = self.readCheckpointLoss()

            self.updateStatus(.completed)

            DispatchQueue.main.async {
                self.totalSteps = finalStep
                self.currentStep = finalStep
                self.bestLoss = min(self.bestLoss, finalLoss)
                self.currentLoss = finalLoss
            }

            let result = TrainingResult(
                steps: finalStep,
                duration: duration,
                bestLoss: finalLoss,
                finalLoss: finalLoss,
                adamUpdates: finalStep / max(self.config.accumulationSteps, 1)
            )

            DispatchQueue.main.async { completion(result) }
        }
    }

    /// Schedule overnight training via BGProcessingTask.
    ///
    /// Training starts when the device is charging (if `config.requiresCharging` is true)
    /// and idle. The system allocates up to several minutes of background execution time;
    /// checkpoints ensure progress is saved between sessions.
    ///
    /// - Important: You must call ``registerBackgroundTask()`` once during app launch
    ///   (before `applicationDidFinishLaunching`) for this to work.
    ///
    /// - Parameter hours: Maximum training duration in hours (default: 8.0).
    ///   The actual time depends on system constraints.
    public func scheduleOvernight(hours: Float = 8.0) {
        let request = BGProcessingTaskRequest(identifier: Self.backgroundTaskIdentifier)
        request.requiresExternalPower = config.requiresCharging
        request.requiresNetworkConnectivity = false
        // Prefer late-night scheduling — earliest 30 minutes from now
        request.earliestBeginDate = Date(timeIntervalSinceNow: 30 * 60)

        do {
            try BGTaskScheduler.shared.submit(request)
            logger.info("Overnight training scheduled: \(hours)h, requiresCharging=\(self.config.requiresCharging)")
        } catch {
            logger.error("Failed to schedule overnight training: \(error.localizedDescription, privacy: .public)")
            updateStatus(.error)
            DispatchQueue.main.async { self.lastError = error.localizedDescription }
        }
    }

    /// Register the background processing task with the system.
    ///
    /// **Must be called once during app launch**, before `applicationDidFinishLaunching`
    /// returns. Typically called in your `App.init()` or `AppDelegate`.
    ///
    /// ```swift
    /// @main
    /// struct MyApp: App {
    ///     init() {
    ///         ANETrainer.shared.registerBackgroundTask()
    ///     }
    /// }
    /// ```
    public func registerBackgroundTask() {
        let success = BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.backgroundTaskIdentifier,
            using: nil
        ) { [weak self] task in
            guard let self = self, let bgTask = task as? BGProcessingTask else { return }
            self.handleBackgroundTask(bgTask)
        }

        if success {
            logger.info("BGProcessingTask registered: \(Self.backgroundTaskIdentifier, privacy: .public)")
        } else {
            logger.error("Failed to register BGProcessingTask — check Info.plist BGTaskSchedulerPermittedIdentifiers")
        }
    }

    /// Stop training gracefully. Saves a checkpoint before stopping.
    ///
    /// If training is not running, this method does nothing. After stopping,
    /// `status` transitions to `.paused` and training can be resumed.
    public func stop() {
        guard status == .training || status == .compiling else { return }
        logger.info("Stop requested")
        stopRequested = true
    }

    // MARK: - Checkpoint

    /// Save the current training state to disk.
    ///
    /// This is called automatically at `config.checkpointInterval` during training
    /// and when `stop()` is called. You can also call it manually.
    public func saveCheckpoint() {
        guard let state = trainState else {
            logger.warning("No active training state to save")
            return
        }
        ane_train_save(state)
        logger.info("Checkpoint saved manually at step \(self.currentStep)")
    }

    /// Load training state from the most recent checkpoint.
    ///
    /// Called automatically when starting a new training run. Can also be called
    /// manually to inspect the latest checkpoint state without training.
    ///
    /// - Returns: `true` if a valid checkpoint was loaded.
    @discardableResult
    public func loadCheckpoint() -> Bool {
        guard let path = ane_latest_checkpoint_path() else {
            logger.info("No checkpoint found")
            return false
        }

        let pathStr = String(cString: path)
        logger.info("Loading checkpoint from \(pathStr, privacy: .public)")

        // Read header to get metadata
        let step = readCheckpointStep()
        let loss = readCheckpointLoss()

        DispatchQueue.main.async {
            self.totalSteps = step
            self.currentStep = step
            self.bestLoss = min(self.bestLoss, loss)
            self.currentLoss = loss
        }

        logger.info("Checkpoint loaded: step \(step), loss \(loss)")
        return true
    }

    /// Whether a valid checkpoint exists on disk.
    public var hasCheckpoint: Bool {
        ane_latest_checkpoint_path() != nil
    }

    // MARK: - Shared Instance

    /// Shared singleton for convenience. App-wide access to the trainer.
    public static let shared = ANETrainer()

    // MARK: - Private: Engine Management

    /// Initialize the C training engine. Returns false on failure.
    private func initializeEngine() -> Bool {
        // Clean up previous state if any
        if let state = trainState {
            ane_train_free(state)
            trainState = nil
        }

        let dataPath = trainingDataPath

        // Verify training data exists
        guard FileManager.default.fileExists(atPath: dataPath) else {
            logger.error("No training data at \(dataPath, privacy: .public)")
            return false
        }

        // Check for pretrained model in bundle, otherwise use random init (NULL)
        let modelPath: String? = Bundle.main.path(forResource: "stories110M", ofType: "bin")

        let state = modelPath.map { path in
            ane_train_init(path, dataPath)
        } ?? ane_train_init(nil, dataPath)

        guard let state = state else {
            logger.error("ane_train_init returned NULL")
            return false
        }

        trainState = state
        logger.info("Training engine initialized, data=\(dataPath, privacy: .public)")
        return true
    }

    /// Handle a BGProcessingTask by running timed training.
    private func handleBackgroundTask(_ task: BGProcessingTask) {
        logger.info("Background training task started")
        updateStatus(.training)

        // Expiration: stop gracefully
        task.expirationHandler = { [weak self] in
            logger.warning("Background task expiring — requesting stop")
            self?.stopRequested = true
        }

        trainQueue.async { [weak self] in
            guard let self = self else {
                task.setTaskCompleted(success: false)
                return
            }

            // Use timed training with a conservative 0.5h per background session
            // (system may grant more or less time)
            let summary = ane_timed_training(0.5)
            let summaryStr = summary as String? ?? ""
            logger.info("Background training session done: \(summaryStr, privacy: .public)")

            // Update state from checkpoint
            let step = self.readCheckpointStep()
            let loss = self.readCheckpointLoss()

            DispatchQueue.main.async {
                self.totalSteps = step
                self.currentStep = step
                self.bestLoss = min(self.bestLoss, loss)
                self.currentLoss = loss
                self.status = .paused
            }

            // Re-schedule for next background window
            self.scheduleOvernight()
            task.setTaskCompleted(success: true)
        }
    }

    // MARK: - Private: Helpers

    /// Update status on the main thread.
    private func updateStatus(_ newStatus: TrainingStatus) {
        DispatchQueue.main.async { self.status = newStatus }
    }

    /// Record an error and update status.
    private func failWithError(_ message: String) {
        logger.error("\(message, privacy: .public)")
        DispatchQueue.main.async {
            self.lastError = message
            self.status = .error
        }
    }

    /// Count uint16_t tokens in a binary file.
    private static func tokenCountInFile(atPath path: String) -> Int {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? UInt64 else { return 0 }
        return Int(size) / MemoryLayout<UInt16>.size
    }

    /// Read the step number from the latest checkpoint header.
    private func readCheckpointStep() -> Int {
        guard let cPath = ane_latest_checkpoint_path() else { return 0 }
        let path = String(cString: cPath)

        guard let data = FileManager.default.contents(atPath: path),
              data.count >= MemoryLayout<ANECkptHeader>.size else { return 0 }

        return data.withUnsafeBytes { buf -> Int in
            guard let hdr = buf.baseAddress?.assumingMemoryBound(to: ANECkptHeader.self) else { return 0 }
            guard hdr.pointee.magic == 0x424C5A54 else { return 0 } // "BLZT"
            return Int(hdr.pointee.step)
        }
    }

    /// Read the loss from the latest checkpoint header.
    private func readCheckpointLoss() -> Float {
        guard let cPath = ane_latest_checkpoint_path() else { return .infinity }
        let path = String(cString: cPath)

        guard let data = FileManager.default.contents(atPath: path),
              data.count >= MemoryLayout<ANECkptHeader>.size else { return .infinity }

        return data.withUnsafeBytes { buf -> Float in
            guard let hdr = buf.baseAddress?.assumingMemoryBound(to: ANECkptHeader.self) else { return .infinity }
            guard hdr.pointee.magic == 0x424C5A54 else { return .infinity }
            return hdr.pointee.loss
        }
    }
}
