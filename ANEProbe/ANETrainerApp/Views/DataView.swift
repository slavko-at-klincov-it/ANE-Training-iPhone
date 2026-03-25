// DataView.swift — Renamed to "Planen": schedule overnight training + data management
import SwiftUI
import UniformTypeIdentifiers

// Scheduled training job model (shared)
struct TrainingJob: Identifiable {
    let id = UUID()
    let scheduledDate: Date
    let duration: Float
    let tokenCount: Int
    let type: JobType
    var status: JobStatus = .pending

    enum JobType: String {
        case manual = "Manuell"
        case overnight = "Overnight"
    }
    enum JobStatus: String {
        case pending = "Geplant"
        case running = "Laeuft"
        case completed = "Fertig"
        case failed = "Fehler"
    }
}

struct DataView: View {
    @EnvironmentObject var trainer: ANETrainer
    @State private var showScheduleSheet = false
    @State private var showManualScheduleSheet = false
    @State private var scheduledJobs: [TrainingJob] = []
    @State private var showClearConfirm = false

    var body: some View {
        NavigationStack {
            List {
                // Token overview
                Section {
                    HStack {
                        Image(systemName: "text.word.spacing")
                            .foregroundStyle(.blue)
                        Text("\(trainer.collectedTokenCount) Tokens gesammelt")
                            .font(.headline)
                        Spacer()
                        if trainer.collectedTokenCount >= 257 {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                        }
                    }
                }

                // Schedule training
                Section {
                    Button {
                        showManualScheduleSheet = true
                    } label: {
                        HStack {
                            Label("Manuelles Training", systemImage: "clock")
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                    }
                    .tint(.green)

                    Button {
                        showScheduleSheet = true
                    } label: {
                        HStack {
                            Label("Overnight Training", systemImage: "moon.stars")
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                    }
                    .tint(.indigo)
                } header: {
                    Text("Training planen")
                } footer: {
                    Text("Manuell: Trainiert zu einer bestimmten Uhrzeit, kein Ladegeraet noetig.\nOvernight: Startet automatisch wenn iPhone am Strom haengt und idle ist.")
                }

                // Scheduled jobs
                if !scheduledJobs.isEmpty {
                    Section {
                        ForEach(scheduledJobs) { job in
                            jobRow(job)
                        }
                        .onDelete { indices in
                            scheduledJobs.remove(atOffsets: indices)
                        }
                    } header: {
                        Text("Geplante Trainings")
                    }
                }

                // Data management
                Section {
                    Button(role: .destructive) {
                        showClearConfirm = true
                    } label: {
                        Label("Alle Trainingsdaten loeschen", systemImage: "trash")
                    }
                    .disabled(trainer.collectedTokenCount == 0)
                } header: {
                    Text("Daten verwalten")
                }
            }
            .navigationTitle("Planen")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    InfoButton(title: "Training planen", items: [
                        ("clock", "Manuelles Training",
                         "Plane ein Training fuer eine bestimmte Uhrzeit. Kein Ladegeraet noetig, aber Akku wird verbraucht (~7.5W). Ideal fuer tagsüber: z.B. von 13:00-15:00."),
                        ("moon.stars", "Overnight Training",
                         "Das iPhone trainiert automatisch ueber Nacht. Voraussetzung: Ladegeraet angeschlossen. iOS entscheidet den genauen Startzeitpunkt."),
                        ("battery.75", "Akku-Verbrauch",
                         "Training verbraucht ~7.5 Watt. 2h manuelles Training braucht ca. 30% Akku. Overnight am Ladekabel ist kostenlos."),
                        ("list.bullet", "Jobs verwalten",
                         "Geplante Trainings werden hier angezeigt. Wische nach links um einen Job zu loeschen.")
                    ])
                }
            }
            .sheet(isPresented: $showScheduleSheet) {
                scheduleSheet
            }
            .sheet(isPresented: $showManualScheduleSheet) {
                manualScheduleSheet
            }
            .alert("Daten loeschen?", isPresented: $showClearConfirm) {
                Button("Loeschen", role: .destructive) { trainer.clearTrainingData() }
                Button("Abbrechen", role: .cancel) {}
            } message: {
                Text("Alle \(trainer.collectedTokenCount) Tokens werden unwiderruflich geloescht.")
            }
        }
    }

    private func jobRow(_ job: TrainingJob) -> some View {
        HStack {
            Image(systemName: job.type == .overnight ? "moon.stars" : "clock")
                .foregroundStyle(job.type == .overnight ? .indigo : .green)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 2) {
                Text("\(job.duration, specifier: "%.0f")h \(job.type.rawValue)")
                    .font(.subheadline.bold())
                HStack(spacing: 8) {
                    Text(job.scheduledDate, style: .date)
                    Text(job.scheduledDate, style: .time)
                    Text("\(job.tokenCount) Tok")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer()
            Text(job.status.rawValue)
                .font(.caption.bold())
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(jobColor(job.status).opacity(0.15))
                .foregroundStyle(jobColor(job.status))
                .cornerRadius(6)
        }
    }

    private func jobColor(_ status: TrainingJob.JobStatus) -> Color {
        switch status {
        case .pending: return .indigo; case .running: return .green
        case .completed: return .blue; case .failed: return .red
        }
    }

    // MARK: - Schedule Sheet

    @State private var scheduleHours: Float = 8
    @State private var manualStartDate = Date()
    @State private var manualDurationMinutes: Float = 120
    @State private var manualTextInput = ""
    @State private var showManualFilePicker = false
    @State private var showManualNoDataAlert = false
    @State private var scheduleTextInput = ""
    @State private var showScheduleFilePicker = false
    @State private var showNoDataAlert = false

    private var scheduleSheet: some View {
        NavigationStack {
            Form {
                Section {
                    HStack {
                        Text("Dauer")
                        Spacer()
                        Text("\(scheduleHours, specifier: "%.0f") Stunden")
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: $scheduleHours, in: 1...12, step: 1)
                } header: {
                    Text("Training-Dauer")
                }

                Section {
                    HStack {
                        Image(systemName: "text.word.spacing")
                            .foregroundStyle(.blue)
                        Text("\(trainer.collectedTokenCount) Tokens vorhanden")
                        Spacer()
                        if trainer.collectedTokenCount >= 257 {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                        }
                    }

                    TextEditor(text: $scheduleTextInput)
                        .frame(minHeight: 80)
                        .overlay(
                            Group {
                                if scheduleTextInput.isEmpty {
                                    Text("Text hier eingeben...")
                                        .foregroundStyle(.tertiary)
                                        .padding(.top, 8)
                                        .padding(.leading, 4)
                                        .allowsHitTesting(false)
                                }
                            }, alignment: .topLeading
                        )

                    if !scheduleTextInput.isEmpty {
                        Button {
                            trainer.addText(scheduleTextInput)
                            scheduleTextInput = ""
                        } label: {
                            Label("Text hinzufuegen (\(scheduleTextInput.count) Zeichen)", systemImage: "plus.circle.fill")
                        }
                        .buttonStyle(.bordered).tint(.blue)
                    }

                    HStack {
                        Button {
                            if let clip = UIPasteboard.general.string, !clip.isEmpty {
                                scheduleTextInput = clip
                            }
                        } label: { Label("Einfuegen", systemImage: "doc.on.clipboard") }
                        .buttonStyle(.bordered)

                        Button {
                            showScheduleFilePicker = true
                        } label: { Label("Datei", systemImage: "doc.badge.plus") }
                        .buttonStyle(.bordered)
                    }
                } header: {
                    Text("Trainingsdaten")
                } footer: {
                    if trainer.collectedTokenCount < 257 && scheduleTextInput.isEmpty {
                        Text("Mindestens 257 Tokens noetig (~200 Woerter).")
                            .foregroundStyle(.orange)
                    }
                }

                Section {
                    HStack(spacing: 6) {
                        Image(systemName: "bolt.fill").foregroundStyle(.yellow)
                        Text("Ladegeraet wird benoetigt")
                    }
                    HStack(spacing: 6) {
                        Image(systemName: "moon.stars").foregroundStyle(.indigo)
                        Text("Startet wenn iPhone idle & am Strom")
                    }
                } header: {
                    Text("Voraussetzungen")
                }

                Section {
                    Button {
                        if !scheduleTextInput.isEmpty {
                            trainer.addText(scheduleTextInput)
                            scheduleTextInput = ""
                        }
                        if trainer.collectedTokenCount < 257 {
                            showNoDataAlert = true
                        } else {
                            let job = TrainingJob(scheduledDate: Date(), duration: scheduleHours, tokenCount: trainer.collectedTokenCount, type: .overnight)
                            scheduledJobs.append(job)
                            trainer.scheduleOvernight(hours: scheduleHours)
                            showScheduleSheet = false
                        }
                    } label: {
                        Label("Training planen", systemImage: "calendar.badge.plus")
                            .frame(maxWidth: .infinity)
                            .font(.headline)
                    }
                }
                .alert("Nicht genug Trainingsdaten", isPresented: $showNoDataAlert) {
                    Button("OK", role: .cancel) {}
                } message: {
                    Text("Bitte lade zuerst Texte oder Dateien hoch, damit das Modell etwas zum Lernen hat. Mindestens 257 Tokens (~200 Woerter).")
                }
            }
            .navigationTitle("Training planen")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Abbrechen") { showScheduleSheet = false }
                }
            }
        }
        .presentationDetents([.large])
        .sheet(isPresented: $showScheduleFilePicker) {
            DocumentPickerView { url in
                guard url.startAccessingSecurityScopedResource() else { return }
                defer { url.stopAccessingSecurityScopedResource() }
                if let text = try? String(contentsOf: url, encoding: .utf8) {
                    trainer.addText(text)
                }
            }
        }
    }
    private func formatDuration(_ minutes: Float) -> String {
        let h = Int(minutes) / 60
        let m = Int(minutes) % 60
        if h == 0 { return "\(m) Min" }
        if m == 0 { return "\(h) Std" }
        return "\(h) Std \(m) Min"
    }

    // MARK: - Manual Schedule Sheet

    private var manualScheduleSheet: some View {
        NavigationStack {
            Form {
                Section {
                    DatePicker("Startzeit", selection: $manualStartDate, in: Date()..., displayedComponents: [.date, .hourAndMinute])
                } header: {
                    Text("Wann")
                }

                Section {
                    HStack {
                        Text("Dauer")
                        Spacer()
                        Text(formatDuration(manualDurationMinutes))
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: $manualDurationMinutes, in: 15...480, step: 15)
                } header: {
                    Text("Wie lange")
                }

                Section {
                    HStack {
                        Image(systemName: "text.word.spacing")
                            .foregroundStyle(.blue)
                        Text("\(trainer.collectedTokenCount) Tokens vorhanden")
                        Spacer()
                        if trainer.collectedTokenCount >= 257 {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                        }
                    }

                    TextEditor(text: $manualTextInput)
                        .frame(minHeight: 80)
                        .overlay(
                            Group {
                                if manualTextInput.isEmpty {
                                    Text("Text hier eingeben...")
                                        .foregroundStyle(.tertiary)
                                        .padding(.top, 8)
                                        .padding(.leading, 4)
                                        .allowsHitTesting(false)
                                }
                            }, alignment: .topLeading
                        )

                    if !manualTextInput.isEmpty {
                        Button {
                            trainer.addText(manualTextInput)
                            manualTextInput = ""
                        } label: {
                            Label("Text hinzufuegen (\(manualTextInput.count) Zeichen)", systemImage: "plus.circle.fill")
                        }
                        .buttonStyle(.bordered).tint(.blue)
                    }

                    HStack {
                        Button {
                            if let clip = UIPasteboard.general.string, !clip.isEmpty {
                                manualTextInput = clip
                            }
                        } label: { Label("Einfuegen", systemImage: "doc.on.clipboard") }
                        .buttonStyle(.bordered)

                        Button { showManualFilePicker = true } label: {
                            Label("Datei", systemImage: "doc.badge.plus")
                        }
                        .buttonStyle(.bordered)
                    }
                } header: {
                    Text("Trainingsdaten")
                } footer: {
                    if trainer.collectedTokenCount < 257 && manualTextInput.isEmpty {
                        Text("Mindestens 257 Tokens noetig (~200 Woerter).")
                            .foregroundStyle(.orange)
                    }
                }

                Section {
                    HStack(spacing: 6) {
                        Image(systemName: "battery.100")
                            .foregroundStyle(.green)
                        Text("Kein Ladegeraet noetig")
                    }
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle")
                            .foregroundStyle(.orange)
                        Text("Akku-Verbrauch: ~7.5 W (\(manualDurationMinutes / 60, specifier: "%.0f")h = ~\(Int(manualDurationMinutes / 60 * 7.5 / 12.5 * 100))% Akku)")
                            .font(.caption)
                    }
                } header: {
                    Text("Hinweise")
                }

                Section {
                    Button {
                        if !manualTextInput.isEmpty {
                            trainer.addText(manualTextInput)
                            manualTextInput = ""
                        }
                        if trainer.collectedTokenCount < 257 {
                            showManualNoDataAlert = true
                        } else {
                            let job = TrainingJob(scheduledDate: manualStartDate, duration: manualDurationMinutes / 60, tokenCount: trainer.collectedTokenCount, type: .manual)
                            scheduledJobs.append(job)
                            // TODO: Implement actual timed training trigger at manualStartDate
                            showManualScheduleSheet = false
                        }
                    } label: {
                        Label("Training planen", systemImage: "calendar.badge.plus")
                            .frame(maxWidth: .infinity)
                            .font(.headline)
                    }
                }
                .alert("Nicht genug Trainingsdaten", isPresented: $showManualNoDataAlert) {
                    Button("OK", role: .cancel) {}
                } message: {
                    Text("Bitte lade zuerst Texte oder Dateien hoch. Mindestens 257 Tokens (~200 Woerter).")
                }
            }
            .navigationTitle("Manuelles Training")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Abbrechen") { showManualScheduleSheet = false }
                }
            }
        }
        .presentationDetents([.large])
        .sheet(isPresented: $showManualFilePicker) {
            DocumentPickerView { url in
                guard url.startAccessingSecurityScopedResource() else { return }
                defer { url.stopAccessingSecurityScopedResource() }
                if let text = try? String(contentsOf: url, encoding: .utf8) {
                    trainer.addText(text)
                }
            }
        }
    }
}

struct DocumentPickerView: UIViewControllerRepresentable {
    let onPick: (URL) -> Void
    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [UTType.plainText, UTType.text, UTType.utf8PlainText])
        picker.delegate = context.coordinator; return picker
    }
    func updateUIViewController(_ vc: UIDocumentPickerViewController, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(onPick: onPick) }
    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void
        init(onPick: @escaping (URL) -> Void) { self.onPick = onPick }
        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            if let url = urls.first { onPick(url) }
        }
    }
}
