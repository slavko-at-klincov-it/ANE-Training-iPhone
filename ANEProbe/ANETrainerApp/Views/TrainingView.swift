// TrainingView.swift — Training dashboard with inline data input + loss curve
import SwiftUI
import Charts

struct TrainingView: View {
    @EnvironmentObject var trainer: ANETrainer
    @State private var lossHistory: [(step: Int, loss: Float)] = []
    @State private var trainingMode = 0
    @State private var elapsed: TimeInterval = 0
    @State private var timer: Timer?
    @State private var textInput = ""
    @State private var showFilePicker = false
    @State private var recentTexts: [String] = []

    private let modes = [
        ("Schnell", 100),
        ("Standard", 1000),
        ("Lang", 5000)
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Status + chart + stats
                    statusBadge
                    if !lossHistory.isEmpty { lossChart }
                    statsGrid
                    thermalRow

                    Divider().padding(.horizontal)

                    // Data input inline
                    VStack(spacing: 12) {
                        sectionHeader("Trainingsdaten")
                        tokenCounter
                        dataInputArea
                    }

                    Divider().padding(.horizontal)

                    // Immediate training
                    VStack(spacing: 12) {
                        sectionHeader("Sofort trainieren")
                        modePicker
                        controlButtons
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("Training")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    InfoButton(title: "Training", items: [
                        ("brain", "Was ist Training?",
                         "Das Modell lernt aus deinen Texten. Es passt seine 110 Millionen Parameter an, um Muster in deinen Daten zu erkennen."),
                        ("doc.text", "Trainingsdaten",
                         "Gib Texte direkt ein oder importiere .txt/.md Dateien. Jede Sprache, kein spezielles Format noetig. Mindestens ~200 Woerter."),
                        ("chart.line.downtrend.xyaxis", "Loss",
                         "Misst wie gut das Modell lernt. Niedrigerer Loss = besseres Verstaendnis. Startet bei ~10.4, gute Werte unter 9.0."),
                        ("gauge.with.dots.needle.33percent", "Steps & Tokens/s",
                         "Ein Step = ein Lernschritt mit 256 Tokens. Tokens/s zeigt die Geschwindigkeit. iPhone 15 Pro: ~800 Tokens/s."),
                        ("thermometer.medium", "Thermal",
                         "Training waermt das iPhone auf. Bei 'Hoch' wird automatisch gedrosselt. iPhone flach hinlegen, Huelle abnehmen.")
                    ])
                }
            }
            .sheet(isPresented: $showFilePicker) {
                DocumentPickerView { url in importFile(url) }
            }
            .onTapGesture {
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
            }
        }
    }

    // MARK: - Components

    private func sectionHeader(_ title: String) -> some View {
        HStack {
            Text(title)
                .font(.subheadline.bold())
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            Spacer()
        }
        .padding(.horizontal)
    }

    private var tokenCounter: some View {
        HStack {
            Image(systemName: "text.word.spacing")
                .foregroundStyle(.blue)
            Text("\(trainer.collectedTokenCount) Tokens")
                .font(.headline)
            Spacer()
            if trainer.collectedTokenCount >= 257 {
                Label("Bereit", systemImage: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else {
                Text("Min. 257")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .padding(.horizontal)
    }

    private var dataInputArea: some View {
        VStack(spacing: 8) {
            TextEditor(text: $textInput)
                .frame(height: 80)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color(.systemGray4), lineWidth: 1)
                )
                .overlay(
                    Group {
                        if textInput.isEmpty {
                            Text("Text hier eingeben...")
                                .foregroundStyle(.tertiary)
                                .padding(.top, 8)
                                .padding(.leading, 4)
                                .allowsHitTesting(false)
                        }
                    }, alignment: .topLeading
                )
                .padding(.horizontal)

            HStack(spacing: 8) {
                Button {
                    if let clip = UIPasteboard.general.string, !clip.isEmpty {
                        textInput = clip
                    }
                } label: {
                    Label("Einfuegen", systemImage: "doc.on.clipboard")
                        .font(.caption)
                }
                .buttonStyle(.bordered)

                Button {
                    showFilePicker = true
                } label: {
                    Label("Datei", systemImage: "doc.badge.plus")
                        .font(.caption)
                }
                .buttonStyle(.bordered)

                Spacer()

                Button {
                    guard !textInput.isEmpty else { return }
                    trainer.addText(textInput)
                    recentTexts.insert(String(textInput.prefix(60)), at: 0)
                    if recentTexts.count > 5 { recentTexts.removeLast() }
                    textInput = ""
                } label: {
                    Label("Hinzufuegen", systemImage: "plus.circle.fill")
                        .font(.caption)
                }
                .buttonStyle(.borderedProminent)
                .disabled(textInput.isEmpty)
            }
            .padding(.horizontal)

            // Recent additions (compact)
            if !recentTexts.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(recentTexts.prefix(3), id: \.self) { text in
                        Text(text)
                            .font(.caption2)
                            .lineLimit(1)
                            .foregroundStyle(.tertiary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)
            }
        }
    }

    private var statusBadge: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(statusColor)
                .frame(width: 10, height: 10)
                .overlay(Circle().stroke(statusColor.opacity(0.3), lineWidth: 3))
            Text(trainer.status.description)
                .font(.headline)
            Spacer()
            if trainer.status == .training {
                Text(formatTime(elapsed))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color(.systemGray6))
                    .cornerRadius(6)
            }
        }
        .padding(.horizontal)
    }

    private var statusColor: Color {
        switch trainer.status {
        case .idle: return .gray
        case .training: return .green
        case .compiling: return .orange
        case .completed: return .blue
        case .paused: return .yellow
        case .error: return .red
        default: return .gray
        }
    }

    private var lossChart: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Loss-Verlauf")
                .font(.caption.bold())
                .foregroundStyle(.secondary)
                .padding(.horizontal)
            Chart {
                ForEach(lossHistory.suffix(200), id: \.step) { point in
                    LineMark(x: .value("Step", point.step), y: .value("Loss", point.loss))
                        .foregroundStyle(.linearGradient(colors: [.blue, .cyan], startPoint: .leading, endPoint: .trailing))
                        .lineStyle(StrokeStyle(lineWidth: 2))
                }
            }
            .chartYScale(domain: .automatic(includesZero: false))
            .chartXAxis(.hidden)
            .frame(height: 180)
            .padding(.horizontal)
        }
    }

    private var statsGrid: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
            statCard("Steps", "\(trainer.totalSteps)", icon: "arrow.clockwise")
            statCard("Tokens/s", String(format: "%.0f", trainer.stepsPerSecond * 256), icon: "speedometer")
            statCard("Loss", String(format: "%.4f", trainer.currentLoss), icon: "chart.line.downtrend.xyaxis")
            statCard("Best Loss", trainer.bestLoss == .infinity ? "--" : String(format: "%.4f", trainer.bestLoss), icon: "star")
        }
        .padding(.horizontal)
    }

    private func statCard(_ title: String, _ value: String, icon: String) -> some View {
        VStack(spacing: 6) {
            HStack(spacing: 4) {
                Image(systemName: icon).font(.caption2).foregroundStyle(.secondary)
                Text(title).font(.caption2).foregroundStyle(.secondary)
            }
            Text(value).font(.title3.bold().monospacedDigit())
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private var thermalRow: some View {
        HStack {
            Image(systemName: thermalIcon).foregroundStyle(thermalColor)
            Text(thermalText).font(.caption)
            Spacer()
        }
        .padding(.horizontal)
    }

    private var thermalIcon: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "thermometer.low"; case .fair: return "thermometer.medium"
        case .serious: return "thermometer.high"; case .critical: return "flame.fill"
        @unknown default: return "thermometer"
        }
    }
    private var thermalColor: Color {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return .green; case .fair: return .yellow
        case .serious: return .orange; case .critical: return .red
        @unknown default: return .gray
        }
    }
    private var thermalText: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "Normal"; case .fair: return "Warm"
        case .serious: return "Hoch"; case .critical: return "Kritisch"
        @unknown default: return "?"
        }
    }

    private var modePicker: some View {
        Picker("Modus", selection: $trainingMode) {
            ForEach(0..<modes.count, id: \.self) { i in Text(modes[i].0).tag(i) }
        }
        .pickerStyle(.segmented)
        .padding(.horizontal)
    }

    private var controlButtons: some View {
        HStack(spacing: 12) {
            if trainer.status == .training || trainer.status == .compiling {
                Button { trainer.stop() } label: {
                    Label("Stop", systemImage: "stop.fill").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent).tint(.red)
            } else {
                Button { startTraining() } label: {
                    Label("Training starten", systemImage: "play.fill").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent).tint(.green)
                .disabled(trainer.collectedTokenCount < 257)
            }
        }
        .padding(.horizontal)
    }

    // MARK: - Actions

    private func importFile(_ url: URL) {
        guard url.startAccessingSecurityScopedResource() else { return }
        defer { url.stopAccessingSecurityScopedResource() }
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return }
        trainer.addText(text)
        recentTexts.insert("[\(url.lastPathComponent)]", at: 0)
        if recentTexts.count > 5 { recentTexts.removeLast() }
    }

    private func startTraining() {
        lossHistory = []; elapsed = 0
        UIApplication.shared.isIdleTimerDisabled = true
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            elapsed += 1
            if trainer.currentLoss > 0 {
                lossHistory.append((step: trainer.currentStep, loss: trainer.currentLoss))
            }
        }
        let steps = modes[trainingMode].1
        trainer.train(steps: steps) { _ in finishTraining() }
    }

    private func finishTraining() {
        timer?.invalidate(); timer = nil
        UIApplication.shared.isIdleTimerDisabled = false
    }

    private func formatTime(_ t: TimeInterval) -> String {
        String(format: "%d:%02d", Int(t)/60, Int(t)%60)
    }
}
