import Foundation
import BackgroundTasks
import Combine
import os.log

private let logger = Logger(subsystem: "com.klincov.aneprobe", category: "bgtraining")

// MARK: - Training Session

/// Manages a single training run — tracks progress, drives the step loop,
/// and bridges into ObjC training code.
final class TrainingSession: ObservableObject {

    // MARK: State

    enum Status: String, CustomStringConvertible {
        case idle
        case running
        case paused
        case stopped
        case completed

        var description: String { rawValue }
    }

    // MARK: Published properties (UI-bindable)

    @Published private(set) var status: Status = .idle
    @Published private(set) var currentStep: Int = 0
    @Published private(set) var totalSteps: Int = 0
    @Published private(set) var lastLoss: Float = .nan
    @Published private(set) var stepsPerSecond: Double = 0

    // MARK: Private

    private let thermalMonitor: ANEThermalMonitor
    private var trainTask: Task<Void, Never>?
    private var shouldStop = false
    private let checkpointURL: URL

    /// Directory for saving/loading checkpoints.
    private static var checkpointDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("ANETrainingCheckpoints", isDirectory: true)
    }

    // MARK: Init

    init(totalSteps: Int = 10_000, thermalMonitor: ANEThermalMonitor = ANEThermalMonitor()) {
        self.totalSteps = totalSteps
        self.thermalMonitor = thermalMonitor

        let dir = Self.checkpointDirectory
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        self.checkpointURL = dir.appendingPathComponent("latest.ckpt")

        // Restore from previous checkpoint if available
        restoreCheckpoint()
    }

    // MARK: - Public API

    func start() {
        guard status == .idle || status == .paused || status == .stopped else { return }

        logger.info("Training start requested — step \(self.currentStep)/\(self.totalSteps)")
        shouldStop = false
        status = .running

        trainTask = Task.detached(priority: .userInitiated) { [weak self] in
            await self?.runTrainingLoop()
        }
    }

    func pause() {
        guard status == .running else { return }
        logger.info("Training paused at step \(self.currentStep)")
        shouldStop = true
        status = .paused
    }

    func stop() {
        guard status == .running || status == .paused else { return }
        logger.info("Training stopped at step \(self.currentStep)")
        shouldStop = true
        status = .stopped
        saveCheckpoint()
    }

    /// Called from BGProcessingTask handler — runs until expiration signal.
    /// Returns the number of steps completed in this background session.
    @discardableResult
    func runUntilExpired(expirationFlag: UnsafeMutablePointer<Bool>) async -> Int {
        let startStep = currentStep
        await MainActor.run { status = .running }
        shouldStop = false

        logger.info("Background training session starting at step \(startStep)")

        while !expirationFlag.pointee && !shouldStop && currentStep < totalSteps {
            // Check thermal state
            let policy = thermalMonitor.policy
            switch policy {
            case .fullSpeed:
                break
            case .reduceBatch:
                try? await Task.sleep(nanoseconds: UInt64(thermalMonitor.interStepDelay * 1_000_000_000))
            case .pause:
                logger.notice("Background training paused due to thermal state (serious)")
                // In background, thermal pause means we should save and yield
                saveCheckpoint()
                await MainActor.run { status = .paused }
                return currentStep - startStep
            case .stop:
                logger.warning("Background training stopped due to critical thermal state")
                saveCheckpoint()
                await MainActor.run { status = .stopped }
                return currentStep - startStep
            }

            // Execute one training step via ObjC bridge
            let loss = executeTrainingStep()

            let step = currentStep + 1
            await MainActor.run {
                self.currentStep = step
                self.lastLoss = loss
            }

            if step >= totalSteps {
                await MainActor.run { status = .completed }
                logger.info("Training completed — all \(self.totalSteps) steps done")
                saveCheckpoint()
                return step - startStep
            }
        }

        // Expiring or stopped — save checkpoint
        saveCheckpoint()
        let stepsCompleted = currentStep - startStep
        logger.info("Background session ending — \(stepsCompleted) steps this session, total \(self.currentStep)/\(self.totalSteps)")

        if currentStep < totalSteps && !shouldStop {
            await MainActor.run { status = .paused }
        }

        return stepsCompleted
    }

    // MARK: - Training loop (foreground)

    private func runTrainingLoop() async {
        var stepTimes: [CFAbsoluteTime] = []

        while !shouldStop && currentStep < totalSteps {
            // Respect thermal policy
            let policy = thermalMonitor.policy
            if policy == .stop {
                logger.warning("Training stopped due to critical thermal state")
                saveCheckpoint()
                await MainActor.run { status = .stopped }
                return
            }
            if policy == .pause {
                logger.notice("Training paused due to serious thermal state")
                await MainActor.run { status = .paused }
                // Spin-wait until thermal recovers or we're stopped
                while thermalMonitor.policy == .pause && !shouldStop {
                    try? await Task.sleep(nanoseconds: 500_000_000) // 0.5s
                }
                if shouldStop { break }
                await MainActor.run { status = .running }
                continue
            }
            if policy == .reduceBatch {
                try? await Task.sleep(nanoseconds: UInt64(thermalMonitor.interStepDelay * 1_000_000_000))
            }

            let t0 = CFAbsoluteTimeGetCurrent()
            let loss = executeTrainingStep()
            let dt = CFAbsoluteTimeGetCurrent() - t0

            stepTimes.append(dt)
            if stepTimes.count > 100 { stepTimes.removeFirst() }
            let avgTime = stepTimes.reduce(0, +) / Double(stepTimes.count)

            let step = currentStep + 1
            let sps = avgTime > 0 ? 1.0 / avgTime : 0
            await MainActor.run {
                self.currentStep = step
                self.lastLoss = loss
                self.stepsPerSecond = sps
            }

            // Periodic checkpoint every 1000 steps
            if step % 1000 == 0 {
                saveCheckpoint()
                logger.info("Checkpoint saved at step \(step), loss=\(loss, format: .fixed(precision: 6))")
            }

            if step >= totalSteps {
                await MainActor.run { status = .completed }
                logger.info("Training completed — all \(self.totalSteps) steps done")
                saveCheckpoint()
                return
            }
        }

        if !shouldStop {
            await MainActor.run { status = .paused }
        }
    }

    // MARK: - ObjC Bridge

    /// Execute a single training step through the ObjC layer.
    /// Returns the loss for this step.
    private func executeTrainingStep() -> Float {
        // Bridge to the ObjC training implementation.
        // ane_weight_test() currently runs a weight update cycle — this is the hook
        // point where the actual per-step training call will go once ANETrainStep is ready.
        //
        // For now we call the weight test as a proxy to exercise the ANE weight path.
        // TODO: Replace with dedicated per-step API: ane_train_step(stepIndex, &loss)
        let _ = ane_weight_test()

        // Synthetic loss for now — will be replaced by actual loss from ObjC
        // Decaying curve to simulate convergence
        let step = Float(currentStep)
        let noise = Float.random(in: -0.01...0.01)
        let loss = 1.0 / (1.0 + step * 0.001) + noise
        return max(loss, 0)
    }

    // MARK: - Checkpoint Persistence

    private func saveCheckpoint() {
        let data: [String: Any] = [
            "currentStep": currentStep,
            "totalSteps": totalSteps,
            "lastLoss": lastLoss,
            "timestamp": Date().timeIntervalSince1970
        ]
        do {
            let encoded = try NSKeyedArchiver.archivedData(
                withRootObject: data,
                requiringSecureCoding: false
            )
            try encoded.write(to: checkpointURL, options: .atomic)
            logger.debug("Checkpoint saved: step \(self.currentStep)")
        } catch {
            logger.error("Failed to save checkpoint: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func restoreCheckpoint() {
        guard FileManager.default.fileExists(atPath: checkpointURL.path) else { return }
        do {
            let encoded = try Data(contentsOf: checkpointURL)
            if let data = try NSKeyedUnarchiver.unarchiveTopLevelObjectWithData(encoded) as? [String: Any] {
                currentStep = data["currentStep"] as? Int ?? 0
                totalSteps = data["totalSteps"] as? Int ?? totalSteps
                lastLoss = data["lastLoss"] as? Float ?? .nan
                let ts = data["timestamp"] as? TimeInterval ?? 0
                let date = Date(timeIntervalSince1970: ts)
                logger.info("Checkpoint restored: step \(self.currentStep)/\(self.totalSteps), saved \(date.description, privacy: .public)")
            }
        } catch {
            logger.error("Failed to restore checkpoint: \(error.localizedDescription, privacy: .public)")
        }
    }
}

// MARK: - Background Training Manager

/// Registers and manages BGProcessingTask for overnight ANE training.
final class ANEBackgroundTrainingManager {

    static let shared = ANEBackgroundTrainingManager()
    static let taskIdentifier = "com.klincov.aneprobe.training"

    private let session: TrainingSession
    private var expirationFlag = false

    private init() {
        self.session = TrainingSession()
    }

    /// Access the shared training session (for UI binding).
    var trainingSession: TrainingSession { session }

    // MARK: - Registration (call from app init, before app finishes launching)

    /// Register the background task with the system. Must be called before
    /// the app finishes launching (e.g., in App.init or application(_:didFinishLaunchingWithOptions:)).
    func registerBackgroundTask() {
        let success = BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.taskIdentifier,
            using: nil
        ) { [weak self] task in
            guard let self, let bgTask = task as? BGProcessingTask else { return }
            self.handleBackgroundTask(bgTask)
        }

        if success {
            logger.info("BGProcessingTask registered: \(Self.taskIdentifier, privacy: .public)")
        } else {
            logger.error("Failed to register BGProcessingTask — check Info.plist BGTaskSchedulerPermittedIdentifiers")
        }
    }

    // MARK: - Scheduling

    /// Schedule background training. Call when the app moves to background.
    func scheduleBackgroundTraining() {
        let request = BGProcessingTaskRequest(identifier: Self.taskIdentifier)
        request.requiresExternalPower = true
        request.requiresNetworkConnectivity = false
        // Prefer overnight — schedule no earlier than 30 minutes from now
        request.earliestBeginDate = Date(timeIntervalSinceNow: 30 * 60)

        do {
            try BGTaskScheduler.shared.submit(request)
            logger.info("Background training scheduled (earliest: +30min, requiresExternalPower=true)")
        } catch {
            logger.error("Failed to schedule background training: \(error.localizedDescription, privacy: .public)")
        }
    }

    /// Cancel any pending background training requests.
    func cancelScheduledTraining() {
        BGTaskScheduler.shared.cancel(taskRequestWithIdentifier: Self.taskIdentifier)
        logger.info("Cancelled pending background training requests")
    }

    // MARK: - Task Handling

    private func handleBackgroundTask(_ task: BGProcessingTask) {
        logger.info("Background training task started")

        // Reset expiration flag
        expirationFlag = false

        // Set up expiration handler — system is about to kill us
        task.expirationHandler = { [weak self] in
            logger.warning("Background task expiring — saving checkpoint")
            self?.expirationFlag = true
            // Give the training loop a moment to notice and save
        }

        // Run training in an async context
        Task {
            let stepsCompleted = await session.runUntilExpired(
                expirationFlag: withUnsafeMutablePointer(to: &expirationFlag) { $0 }
            )

            logger.info("Background session completed: \(stepsCompleted) steps")

            // Re-schedule if not finished
            if session.currentStep < session.totalSteps {
                scheduleBackgroundTraining()
                task.setTaskCompleted(success: true)
            } else {
                task.setTaskCompleted(success: true)
                logger.info("Training fully complete — not rescheduling")
            }
        }
    }
}
