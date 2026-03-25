// ANETrainerApp.swift — Production app for on-device transformer training
import SwiftUI

struct ANETrainerAppView: View {
    @StateObject private var trainer = ANETrainer.shared

    var body: some View {
        TabView {
            TrainingView()
                .tabItem { Label("Training", systemImage: "brain") }

            DataView()
                .tabItem { Label("Planen", systemImage: "calendar") }

            InferenceView()
                .tabItem { Label("Chat", systemImage: "bubble.left.fill") }

            ModelsView()
                .tabItem { Label("Modelle", systemImage: "folder") }

            SettingsView()
                .tabItem { Label("Settings", systemImage: "gear") }
        }
        .environmentObject(trainer)
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = false
            trainer.registerBackgroundTask()
        }
    }
}
