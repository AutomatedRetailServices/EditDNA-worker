import SwiftUI
import UIKit

final class CutSellAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        guard identifier == BackgroundPartUploader.sessionIdentifier else {
            completionHandler()
            return
        }
        BackgroundPartUploader.shared.acceptBackgroundEventsCompletion(completionHandler)
    }
}

@main
struct CutSellApp: App {
    @UIApplicationDelegateAdaptor(CutSellAppDelegate.self) private var appDelegate
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var appState = AppState()
    @StateObject private var notificationCenter = NotificationCenterModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(appState)
                .environmentObject(notificationCenter)
                .task {
                    await notificationCenter.requestPermission()
                    await appState.bootstrap()
                    if let session = appState.session {
                        await notificationCenter.refresh(session: session, surfaceLocalAlerts: false)
                    }
                }
                .onChange(of: scenePhase) { _, phase in
                    guard phase == .active, let session = appState.session else { return }
                    Task {
                        try? await appState.refreshProjects()
                        await notificationCenter.refresh(session: session)
                    }
                }
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
