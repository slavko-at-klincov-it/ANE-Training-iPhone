// ModelsView.swift — Checkpoint management with info
import SwiftUI

struct ModelsView: View {
    @StateObject private var manager = ModelManager()
    @State private var showExport = false
    @State private var exportURL: URL?

    var body: some View {
        NavigationStack {
            Group {
                if manager.models.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "folder.badge.questionmark")
                            .font(.system(size: 44))
                            .foregroundStyle(.quaternary)
                        Text("Keine gespeicherten Modelle")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        Text("Starte ein Training um ein Modell zu erstellen")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                } else {
                    List {
                        ForEach(manager.models) { model in
                            modelRow(model)
                        }
                        .onDelete { indices in
                            for i in indices { manager.delete(manager.models[i]) }
                        }
                    }
                }
            }
            .navigationTitle("Modelle")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    InfoButton(title: "Modelle & Checkpoints", items: [
                        ("square.and.arrow.down", "Was sind Checkpoints?",
                         "Ein Checkpoint speichert den kompletten Zustand des Modells: alle 110M Gewichte + Optimizer-Status. Damit kann Training spaeter fortgesetzt werden."),
                        ("doc.zipper", "Dateigroesse",
                         "Ein Checkpoint ist ~1.4 GB gross (Gewichte + Adam Optimizer State). Stelle sicher, dass genug Speicher frei ist."),
                        ("square.and.arrow.up", "Exportieren",
                         "Exportiere Checkpoints per AirDrop oder in die Dateien-App. Format: .bin (BLZT), kompatibel mit dem llama2.c Oekosystem."),
                        ("arrow.clockwise", "Fortsetzen",
                         "Training wird automatisch vom letzten Checkpoint fortgesetzt. Der Step-Counter und Loss-Wert bleiben erhalten.")
                    ])
                }
            }
            .onAppear { manager.scan() }
            .refreshable { manager.scan() }
            .sheet(isPresented: $showExport) {
                if let url = exportURL { ActivityView(url: url) }
            }
        }
    }

    private func modelRow(_ model: SavedModel) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "brain.filled.head.profile")
                    .foregroundStyle(.indigo)
                Text(model.filename)
                    .font(.subheadline.bold().monospaced())
                Spacer()
                Text(model.sizeMB)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color(.systemGray5))
                    .cornerRadius(4)
            }
            HStack(spacing: 16) {
                Label("Step \(model.step)", systemImage: "arrow.clockwise")
                Label(String(format: "%.4f", model.loss), systemImage: "chart.line.downtrend.xyaxis")
                Spacer()
                Text(model.date, style: .relative)
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            Button {
                exportURL = manager.exportURL(model)
                showExport = true
            } label: {
                Label("Exportieren", systemImage: "square.and.arrow.up")
                    .font(.caption)
            }
            .buttonStyle(.bordered)
            .tint(.indigo)
        }
        .padding(.vertical, 4)
    }
}

struct ActivityView: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
