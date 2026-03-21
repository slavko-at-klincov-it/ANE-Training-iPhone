import Foundation
import Combine
import os.log

private let logger = Logger(subsystem: "com.klincov.aneprobe", category: "thermal")

/// Monitors device thermal state and provides adaptive training rate guidance.
/// When thermal pressure rises, training should reduce intensity or stop entirely.
final class ANEThermalMonitor: ObservableObject {

    // MARK: - Training-oriented thermal policy

    enum TrainingPolicy: String, CustomStringConvertible {
        case fullSpeed   // .nominal — run at maximum step rate
        case reduceBatch // .fair — reduce batch size or insert short delays between steps
        case pause       // .serious — pause training, keep checkpoint in memory
        case stop        // .critical — save checkpoint to disk immediately and stop

        var description: String { rawValue }
    }

    // MARK: - Published state

    @Published private(set) var thermalState: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState
    @Published private(set) var policy: TrainingPolicy = .fullSpeed

    /// Suggested delay (seconds) to insert between training steps.
    /// Returns 0 for fullSpeed, a small delay for reduceBatch, and .infinity for pause/stop.
    var interStepDelay: TimeInterval {
        switch policy {
        case .fullSpeed:   return 0
        case .reduceBatch: return 0.05   // 50ms pause between steps
        case .pause:       return .infinity
        case .stop:        return .infinity
        }
    }

    /// Whether training should be running at all.
    var shouldTrain: Bool {
        policy == .fullSpeed || policy == .reduceBatch
    }

    // MARK: - Private

    private var cancellable: AnyCancellable?

    // MARK: - Init

    init() {
        update(ProcessInfo.processInfo.thermalState)

        cancellable = NotificationCenter.default.publisher(
            for: ProcessInfo.thermalStateDidChangeNotification
        )
        .compactMap { _ in ProcessInfo.processInfo.thermalState }
        .receive(on: DispatchQueue.main)
        .sink { [weak self] newState in
            self?.update(newState)
        }

        logger.info("Thermal monitor started — initial state: \(self.policy.rawValue, privacy: .public)")
    }

    deinit {
        cancellable?.cancel()
    }

    // MARK: - Internal

    private func update(_ state: ProcessInfo.ThermalState) {
        let oldPolicy = policy
        thermalState = state

        switch state {
        case .nominal:
            policy = .fullSpeed
        case .fair:
            policy = .reduceBatch
        case .serious:
            policy = .pause
        case .critical:
            policy = .stop
        @unknown default:
            policy = .pause
        }

        if policy != oldPolicy {
            logger.notice("Thermal transition: \(oldPolicy.rawValue, privacy: .public) → \(self.policy.rawValue, privacy: .public) (thermalState=\(state.rawValue))")
        }
    }
}

// MARK: - Convenience display helpers

extension ProcessInfo.ThermalState {
    var label: String {
        switch self {
        case .nominal:  return "Nominal"
        case .fair:     return "Fair"
        case .serious:  return "Serious"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
    }

    var symbolName: String {
        switch self {
        case .nominal:  return "thermometer.low"
        case .fair:     return "thermometer.medium"
        case .serious:  return "thermometer.high"
        case .critical: return "thermometer.sun.fill"
        @unknown default: return "thermometer"
        }
    }
}
