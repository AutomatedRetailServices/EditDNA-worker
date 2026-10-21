import SwiftUI
import UIKit

final class CutSellAppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        // Build 24: Background App Refresh checks cuts still being made.
        MainActor.assumeIsolated {
            BackgroundFollow.register()
            // Plan my week: tomorrow's videos exported at night (WiFi + charging); reminders open their Day.
            PostPrefetch.register()
            PostReminders.shared.install()
        }
        return true
    }

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
        // Build 22: iOS woke CutSell because parts finished with the app closed: keep some time to
        // complete the upload and start the cut.
        MainActor.assumeIsolated { PendingUploadFinisher.wokeForUploads() }
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
            #if DEBUG
            if let screen = DesignScreenshots.requestedScreen {
                DesignScreenshotRoot(screen: screen)
            } else {
                mainRoot
            }
            #else
            mainRoot
            #endif
        }
    }

    private var mainRoot: some View {
        // The AI notice is presented by AIConsentCenter itself (on top of any cover or sheet).
        RootView()
                .environmentObject(appState)
                .environmentObject(notificationCenter)
                .task {
                    EditorAudio.setUpAtLaunch()
                    await notificationCenter.requestPermission()
                    await appState.bootstrap()
                    if let session = appState.session {
                        await notificationCenter.refresh(session: session, surfaceLocalAlerts: false)
                    }
                    // Build 22: an upload whose parts are all on R2 is completed and its cut started.
                    if await PendingUploadFinisher.run(session: appState.session, reason: "launch", resumeUpload: { record in
                        await ResumeRunner.resume(record, appState: appState)
                    }) > 0 {
                        try? await appState.refreshProjects()
                    }
                }
                .onChange(of: scenePhase) { _, phase in
                    if phase == .background { BackgroundFollow.enteredBackground(); PostPrefetch.schedule() }
                    guard phase == .active, appState.session != nil else { return }
                    Task {
                        do {
                            try await appState.refreshProjects()
                        } catch {
                            await appState.recoverIfUnauthorized(error)
                        }
                        if let current = appState.session {
                            await notificationCenter.refresh(session: current)
                        }
                        if await PendingUploadFinisher.run(session: appState.session, reason: "reopen", resumeUpload: { record in
                        await ResumeRunner.resume(record, appState: appState)
                    }) > 0 {
                            try? await appState.refreshProjects()
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
                ZStack {
                    ScreenBackground()
                    VStack(spacing: 18) {
                        BrandLogo()
                        BrandRing(size: 72)
                    }
                }
            } else if let error = appState.bootstrapError {
                ContentUnavailableView {
                    Label("Couldn’t start CutSell", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(error)
                } actions: {
                    Button("Try Again") {
                        Task { await appState.bootstrap() }
                    }
                    .buttonStyle(.borderedProminent)
                }
            } else if appState.needsSignIn {
                WelcomeView()
            } else {
                MainTabView()
            }
        }
        .preferredColorScheme(.dark)
        .alert(
            "Signed in",
            isPresented: Binding(
                get: { appState.accountNotice != nil },
                set: { if !$0 { appState.accountNotice = nil } }
            )
        ) {
            Button("OK", role: .cancel) { appState.accountNotice = nil }
        } message: {
            Text(appState.accountNotice ?? "")
        }
    }
}
