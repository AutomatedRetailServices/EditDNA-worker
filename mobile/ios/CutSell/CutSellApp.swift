import SwiftUI

@main
struct CutSellApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(appState)
                .task { await appState.bootstrap() }
        }
    }
}

struct RootView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        Group {
            if appState.isBootstrapping {
                ProgressView("Starting CutSell…")
            } else if let error = appState.bootstrapError {
                ContentUnavailableView(
                    "Couldn’t start CutSell",
                    systemImage: "exclamationmark.triangle",
                    description: Text(error)
                )
            } else {
                ProjectsView()
            }
        }
    }
}
