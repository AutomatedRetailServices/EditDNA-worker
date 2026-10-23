import BackgroundTasks
import Foundation
import UIKit
import UserNotifications

/// Build 24: the cut is made on the server whatever CutSell does; this is how the iPhone finds out
/// it is ready while CutSell is not on screen.
/// - Leaving CutSell: extra time from iOS (~30 s) to keep following the cuts in progress.
/// - Background App Refresh: iOS wakes CutSell now and then (iOS decides when) to check them and
///   post "<name> is ready" for each one that finished.
/// A server push (APNs) would say it at once even with CutSell closed; that needs a push key and a
/// server change (not done here).
@MainActor
enum BackgroundFollow {
    static let taskID = "ai.cutsell.app.follow-cuts"
    private static var time: BackgroundTime?

    static func register() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: taskID, using: nil) { task in
            guard let refresh = task as? BGAppRefreshTask else { task.setTaskCompleted(success: false); return }
            let work = Task { @MainActor in
                let stillProcessing = await checkOnce()
                if stillProcessing { schedule() }
                refresh.setTaskCompleted(success: true)
            }
            refresh.expirationHandler = { work.cancel() }
        }
    }

    /// CutSell went to the background.
    static func enteredBackground() {
        guard CutTracker.shared.hasActiveCuts || MixCenter.shared.hasActive else { return }
        if time == nil {
            let background = BackgroundTime("cutsell-follow-cuts")
            background.begin()
            time = background
        }
        CutSellDiagnostics.log("follow_in_background", ["cuts": String(CutTracker.shared.activeCount)])
        schedule()
        Task { @MainActor in
            // Keep following while iOS allows it (the tracker polls every 3 s).
            var waited = 0
            while CutTracker.shared.hasActiveCuts && waited < 28 {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                waited += 1
            }
            time?.end()
            time = nil
        }
    }

    static func schedule() {
        let request = BGAppRefreshTaskRequest(identifier: taskID)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 60)
        try? BGTaskScheduler.shared.submit(request)
    }

    /// One check of the cuts still processing. Returns true while some are not finished.
    static func checkOnce() async -> Bool {
        if await APIClient.shared.session == nil, let saved = try? KeychainStore.load() {
            await APIClient.shared.setSession(saved)
        }
        guard let session = await APIClient.shared.session else { return false }
        guard let list: ProjectListResponse = try? await APIClient.shared.request(
            "/v1/projects", query: [URLQueryItem(name: "user_id", value: session.userID)], timeout: 20
        ) else { return true }
        let known = Set(UserDefaults.standard.stringArray(forKey: "cutsell.follow.processing") ?? [])
        var stillProcessing: [String] = []
        for project in list.projects {
            if Project.processingStates.contains(project.state) {
                stillProcessing.append(project.projectID)
            } else if GroupNotify.isGrouped(project.projectID) {
                // 23 Oct: a Bulk video: no notification of its own; the group's one when all of it is done.
                if let done = GroupNotify.finished(project.projectID, ok: project.state != "failed") {
                    GroupNotify.post(done.id, done.group)
                }
            } else if project.state == "draft_ready", known.contains(project.projectID) {
                let content = UNMutableNotificationContent()
                content.title = "\(project.title.isEmpty ? "Your cut" : project.title) is ready"
                content.body = "Open CutSell to review and export it."
                content.sound = .default
                try? await UNUserNotificationCenter.current().add(
                    UNNotificationRequest(identifier: "cut-ready-\(project.projectID)", content: content, trigger: nil))
                CutSellDiagnostics.log("cut_ready_background_refresh", [:])
            }
        }
        UserDefaults.standard.set(stillProcessing, forKey: "cutsell.follow.processing")
        // Mix: the server sends one notice for the whole Mix; shown here too while CutSell is not on screen.
        await NotificationCenterModel.surfaceMixNotices(session: session)
        return !stillProcessing.isEmpty || MixCenter.shared.hasActive
    }
}
