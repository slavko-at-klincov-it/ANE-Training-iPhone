// InfoSheet.swift — Reusable info popover/sheet for contextual help
import SwiftUI

struct InfoButton: View {
    let title: String
    let items: [(icon: String, label: String, detail: String)]
    @State private var showInfo = false

    var body: some View {
        Button {
            showInfo = true
        } label: {
            Image(systemName: "info.circle")
                .font(.body)
                .foregroundStyle(.secondary)
        }
        .sheet(isPresented: $showInfo) {
            NavigationStack {
                List {
                    ForEach(items, id: \.label) { item in
                        HStack(alignment: .top, spacing: 12) {
                            Image(systemName: item.icon)
                                .font(.title3)
                                .foregroundStyle(.tint)
                                .frame(width: 28)
                            VStack(alignment: .leading, spacing: 4) {
                                Text(item.label)
                                    .font(.subheadline.bold())
                                Text(item.detail)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
                .navigationTitle(title)
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Fertig") { showInfo = false }
                    }
                }
            }
            .presentationDetents([.medium, .large])
        }
    }
}

struct SettingInfoButton: View {
    let text: String
    @State private var show = false

    var body: some View {
        Button {
            show = true
        } label: {
            Image(systemName: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .popover(isPresented: $show) {
            Text(text)
                .font(.caption)
                .padding()
                .frame(maxWidth: 280)
                .presentationCompactAdaptation(.popover)
        }
    }
}
