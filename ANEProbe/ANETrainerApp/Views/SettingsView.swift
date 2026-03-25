// SettingsView.swift — Configuration with per-setting info tooltips
import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var trainer: ANETrainer
    @State private var learningRate: Double = 3e-4
    @State private var accumSteps = 8
    @State private var checkpointInterval = 500

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    // Learning Rate
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Learning Rate")
                            Text(String(format: "%.1e", learningRate))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.blue)
                        }
                        Spacer()
                        SettingInfoButton(text: "Wie schnell das Modell lernt.\n\n1e-4 bis 3e-4: Standard (empfohlen)\n1e-5: Sehr langsam, feines Tuning\n1e-3: Schnell, aber riskant (kann divergieren)\n\nZu hoch = Modell 'vergisst', zu niedrig = lernt kaum.")
                    }
                    Slider(value: $learningRate, in: 1e-5...1e-3)
                        .onChange(of: learningRate) { _, v in trainer.config.learningRate = Float(v) }

                    // Accumulation Steps
                    HStack {
                        Text("Accumulation Steps: \(accumSteps)")
                        Spacer()
                        SettingInfoButton(text: "Wie viele Schritte zwischen Gewichts-Updates.\n\n4: Haeufige Updates, mehr Compile-Overhead\n8: Gute Balance (empfohlen)\n16: Weniger Compiles, aber groesserer effektiver Batch\n\nHoeher = weniger Kernel-Recompilation = schnelleres Training, aber braucht minimal mehr RAM.")
                    }
                    Stepper("", value: $accumSteps, in: 1...32)
                        .labelsHidden()
                        .onChange(of: accumSteps) { _, v in trainer.config.accumulationSteps = v }

                    // Checkpoint Interval
                    HStack {
                        Text("Checkpoint alle \(checkpointInterval) Steps")
                        Spacer()
                        SettingInfoButton(text: "Wie oft der Trainingsstand gespeichert wird.\n\n100: Sehr haeufig (sicher, aber 1.4 GB pro Save)\n500: Standard (empfohlen)\n2000: Selten (riskant bei Absturz)\n\nJeder Checkpoint schreibt ~1.4 GB auf die SSD. Zu haeufig = SSD-Verschleiss.")
                    }
                    Stepper("", value: $checkpointInterval, in: 50...2000, step: 50)
                        .labelsHidden()
                        .onChange(of: checkpointInterval) { _, v in trainer.config.checkpointInterval = v }
                } header: {
                    HStack {
                        Text("Training")
                        Spacer()
                        InfoButton(title: "Training-Einstellungen", items: [
                            ("slider.horizontal.3", "Was bewirken diese Einstellungen?",
                             "Diese Parameter steuern wie das Modell lernt. Die Standardwerte sind fuer die meisten Faelle optimal. Aendere sie nur wenn du weisst was du tust."),
                            ("exclamationmark.triangle", "Vorsicht",
                             "Falsche Einstellungen koennen dazu fuehren, dass das Modell nicht lernt (LR zu niedrig) oder divergiert (LR zu hoch). Im Zweifel: Standardwerte verwenden.")
                        ])
                    }
                }

                Section {
                    infoRow("Modell", "Stories-110M")
                    infoRow("Parameter", "~110 Millionen")
                    infoRow("Layer", "12 Transformer")
                    infoRow("Dimension", "768")
                    infoRow("Vocab", "32,000 Tokens")
                    infoRow("Sequenzlaenge", "256 Tokens")
                    infoRow("Tokens gesammelt", "\(trainer.collectedTokenCount)")
                    infoRow("Checkpoint", trainer.hasCheckpoint ? "Vorhanden" : "Keiner")
                } header: {
                    Text("Modell")
                }

                Section {
                    let mem = ProcessInfo.processInfo.physicalMemory / (1024*1024*1024)
                    HStack {
                        Text("Geraete-RAM")
                        Spacer()
                        Text("\(mem) GB")
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Text("Trainings-RAM")
                        Spacer()
                        Text("~2.4 GB")
                            .foregroundStyle(mem < 6 ? .red : .green)
                        SettingInfoButton(text: "Training braucht ~2.4 GB RAM fuer Gewichte + Optimizer + Aktivierungen.\n\n6 GB Geraet: Knapp, andere Apps schliessen!\n8 GB Geraet (Pro): Komfortabel\n\nWenn die App abstuerzt: Alle Hintergrund-Apps schliessen und erneut versuchen.")
                    }
                    infoRow("Thermal", thermalText)
                    infoRow("Engine", "ANE (Apple Neural Engine)")
                    infoRow("Praezision", "FP16 (ANE) + FP32 (CPU)")
                } header: {
                    Text("System")
                }

                Section {
                    infoRow("Version", "1.0")
                    Link(destination: URL(string: "https://github.com/slavko-at-klincov-it/ANE-Training-iPhone")!) {
                        HStack {
                            Text("GitHub Repository")
                            Spacer()
                            Image(systemName: "arrow.up.right.square")
                                .foregroundStyle(.secondary)
                        }
                    }
                } header: {
                    Text("Info")
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    InfoButton(title: "Einstellungen", items: [
                        ("gear", "Was kann ich hier einstellen?",
                         "Trainings-Hyperparameter die beeinflussen wie schnell und gut das Modell lernt. Die Standardwerte funktionieren fuer die meisten Faelle."),
                        ("exclamationmark.shield", "Auswirkungen auf das System",
                         "Einige Einstellungen beeinflussen RAM-Verbrauch und Akku. Hohe Accumulation Steps brauchen mehr RAM. Haeufige Checkpoints belasten den Speicher.")
                    ])
                }
            }
        }
    }

    private func infoRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary)
        }
    }

    private var thermalText: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "Normal"
        case .fair: return "Warm"
        case .serious: return "Hoch"
        case .critical: return "Kritisch"
        @unknown default: return "?"
        }
    }
}
