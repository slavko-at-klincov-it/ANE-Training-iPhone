import SwiftUI

/// Dashboard view showing training status, progress, loss, thermal state, and controls.
struct TrainingDashboardView: View {
    @StateObject private var session = ANEBackgroundTrainingManager.shared.trainingSession
    @StateObject private var thermal = ANEThermalMonitor()

    var body: some View {
        VStack(spacing: 20) {
            Text("ANE Training")
                .font(.title.bold())

            // MARK: - Status Badge
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 12, height: 12)
                Text(session.status.rawValue.capitalized)
                    .font(.headline)
            }

            // MARK: - Progress
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Step")
                    Spacer()
                    Text("\(session.currentStep) / \(session.totalSteps)")
                        .monospacedDigit()
                }

                ProgressView(value: progress)
                    .tint(statusColor)

                HStack {
                    Text("Loss")
                    Spacer()
                    if session.lastLoss.isNaN {
                        Text("--")
                            .monospacedDigit()
                    } else {
                        Text(String(format: "%.6f", session.lastLoss))
                            .monospacedDigit()
                    }
                }

                HStack {
                    Text("Speed")
                    Spacer()
                    Text(String(format: "%.1f steps/s", session.stepsPerSecond))
                        .monospacedDigit()
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)

            // MARK: - Thermal State
            HStack {
                Image(systemName: thermal.thermalState.symbolName)
                    .foregroundColor(thermalColor)
                    .font(.title2)
                VStack(alignment: .leading) {
                    Text("Thermal: \(thermal.thermalState.label)")
                        .font(.subheadline.bold())
                    Text("Policy: \(thermal.policy.rawValue)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
            }
            .padding()
            .background(thermalColor.opacity(0.1))
            .cornerRadius(12)

            Spacer()

            // MARK: - Controls
            HStack(spacing: 16) {
                Button(action: {
                    if session.status == .running {
                        session.pause()
                    } else {
                        session.start()
                    }
                }) {
                    Label(
                        session.status == .running ? "Pause" : "Start",
                        systemImage: session.status == .running ? "pause.fill" : "play.fill"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(session.status == .running ? .orange : .green)
                .disabled(session.status == .completed)

                Button(action: {
                    session.stop()
                }) {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .disabled(session.status == .idle || session.status == .stopped || session.status == .completed)
            }
        }
        .padding()
    }

    // MARK: - Helpers

    private var progress: Double {
        guard session.totalSteps > 0 else { return 0 }
        return Double(session.currentStep) / Double(session.totalSteps)
    }

    private var statusColor: Color {
        switch session.status {
        case .idle:      return .gray
        case .running:   return .green
        case .paused:    return .orange
        case .stopped:   return .red
        case .completed: return .blue
        }
    }

    private var thermalColor: Color {
        switch thermal.thermalState {
        case .nominal:  return .green
        case .fair:     return .yellow
        case .serious:  return .orange
        case .critical: return .red
        @unknown default: return .gray
        }
    }
}
