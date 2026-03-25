// InferenceView.swift — Chat interface with info tooltips for controls
import SwiftUI

struct InferenceView: View {
    @StateObject private var inference = InferenceManager()
    @State private var prompt = ""
    @State private var output = ""
    @State private var temperature: Float = 0.7
    @State private var maxTokens = 50

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                modelStatusBar

                ScrollView {
                    if output.isEmpty {
                        VStack(spacing: 12) {
                            Image(systemName: "bubble.left.and.bubble.right")
                                .font(.system(size: 44))
                                .foregroundStyle(.quaternary)
                            Text("Prompt eingeben und generieren")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.top, 80)
                    } else {
                        Text(output)
                            .font(.system(.body, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                            .textSelection(.enabled)
                    }
                }
                .frame(maxHeight: .infinity)

                Divider()

                VStack(spacing: 10) {
                    // Temperature row with info
                    HStack(spacing: 6) {
                        SettingInfoButton(text: "Temperatur steuert die Kreativitaet.\n\n0.0 = immer das wahrscheinlichste Wort (deterministisch)\n0.7 = gute Balance\n1.5 = sehr kreativ/zufaellig\n\nNiedriger = praeziser, hoeher = ueberraschender.")
                        Text("Temp \(String(format: "%.1f", temperature))")
                            .font(.caption.monospacedDigit())
                            .frame(width: 60)
                        Slider(value: $temperature, in: 0...1.5, step: 0.1)
                    }

                    // Token count row with info
                    HStack(spacing: 6) {
                        SettingInfoButton(text: "Maximale Anzahl Tokens die generiert werden.\n\nMehr Tokens = laengerer Output, aber auch laengere Wartezeit.\n\n10 = kurzes Wort/Satz\n50 = Absatz\n256 = maximale Laenge (ganzes Kontextfenster)")
                        Text("\(maxTokens) Tok")
                            .font(.caption.monospacedDigit())
                            .frame(width: 50)
                        Slider(value: Binding(
                            get: { Float(maxTokens) },
                            set: { maxTokens = Int($0) }
                        ), in: 10...256, step: 10)
                    }

                    // Prompt + send
                    HStack(spacing: 8) {
                        TextField("Prompt eingeben...", text: $prompt, axis: .vertical)
                            .textFieldStyle(.roundedBorder)
                            .lineLimit(1...3)

                        Button {
                            generateText()
                        } label: {
                            Image(systemName: "paperplane.fill")
                                .font(.body)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(prompt.isEmpty || !inference.isLoaded || inference.isGenerating)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 10)
                .background(Color(.systemBackground))
            }
            .navigationTitle("Chat")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 12) {
                        if !output.isEmpty {
                            Button {
                                UIPasteboard.general.string = output
                            } label: {
                                Image(systemName: "doc.on.doc")
                            }
                        }
                        InfoButton(title: "Chat & Inference", items: [
                            ("bubble.left.fill", "Was ist Inference?",
                             "Das trainierte Modell generiert Text basierend auf deinem Prompt. Je besser trainiert, desto kohaerenter der Output."),
                            ("thermometer.variable", "Temperatur",
                             "Steuert wie kreativ/zufaellig der Output ist. 0.0 = immer gleiches Ergebnis (deterministisch). 0.7 = gute Balance. 1.5 = sehr kreativ aber chaotisch."),
                            ("number", "Token-Anzahl",
                             "Wie viele Woerter/Teilwoerter generiert werden. 10 = kurz, 50 = Absatz, 256 = Maximum. Mehr Tokens = laengere Wartezeit."),
                            ("exclamationmark.triangle", "Zufaellige Gewichte",
                             "Wenn 'Zufaellige Gewichte (Test)' angezeigt wird, ist das Modell noch nicht trainiert. Der Output ist dann Unsinn — das ist normal. Trainiere zuerst!")
                        ])
                    }
                }
            }
            .onAppear { inference.loadModel() }
            .onTapGesture {
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
            }
        }
    }

    private var modelStatusBar: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(inference.isLoaded ? .green : .orange)
                .frame(width: 8, height: 8)
            Text(inference.isLoaded ? inference.modelInfo : "Modell wird geladen...")
                .font(.caption)
            Spacer()
            if inference.isGenerating {
                ProgressView().scaleEffect(0.7)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(Color(.systemGray6))
    }

    private func generateText() {
        let p = prompt
        output = "Generiere..."
        Task.detached {
            let result = await inference.generate(prompt: p, maxTokens: maxTokens, temperature: temperature)
            await MainActor.run { output = result }
        }
    }
}
