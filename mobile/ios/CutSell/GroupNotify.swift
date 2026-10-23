import Foundation
import UIKit
import UserNotifications

/// 23 Oct (PO): a Bulk send gets ONE notification when every video of it is done -- "Your 10 videos are ready ✨",
/// or "9 ready, 1 needs attention" -- instead of one per video. Single and Combine are unchanged (one per cut).
/// Kept in UserDefaults so the background refresh (CutSell not on screen) can tell too, and the notice is posted once.
@MainActor
enum GroupNotify {
    struct Group: Codable, Equatable {
        var total: Int
        var projects: [String] = []
        var ready: [String] = []
        var failed: [String] = []
        var notified = false
        var startedAt = Date()

        var done: Int { Set(ready).union(failed).count }
        var isComplete: Bool { projects.count >= total && done >= total }
    }

    private static let key = "cutsell.bulk.groups"

    static func load() -> [String: Group] {
        guard let data = UserDefaults.standard.data(forKey: key),
              let groups = try? JSONDecoder().decode([String: Group].self, from: data) else { return [:] }
        return groups
    }

    private static func save(_ groups: [String: Group]) {
        // Forget groups told about over 2 days ago (and any older than a week).
        let kept = groups.filter { _, g in
            let age = Date().timeIntervalSince(g.startedAt)
            return g.notified ? age < 2 * 86_400 : age < 7 * 86_400
        }
        if let data = try? JSONEncoder().encode(kept) { UserDefaults.standard.set(data, forKey: key) }
    }

    /// A Bulk send started with `total` videos.
    static func start(_ groupID: String, total: Int) {
        var groups = load()
        if groups[groupID] == nil { groups[groupID] = Group(total: total) }
        save(groups)
    }

    /// A video of the group now has its project (its cut was sent to the server).
    static func link(_ groupID: String, projectID: String) {
        var groups = load()
        guard var g = groups[groupID] else { return }
        if !g.projects.contains(projectID) { g.projects.append(projectID) }
        groups[groupID] = g
        save(groups)
    }

    static func group(of projectID: String) -> String? {
        load().first { $0.value.projects.contains(projectID) }?.key
    }

    static func isGrouped(_ projectID: String) -> Bool { group(of: projectID) != nil }

    /// A video of a group finished (ready or failed). Returns the group when that was the last one and nobody was
    /// told yet (the caller then tells: notification and/or the in-app notice).
    @discardableResult
    static func finished(_ projectID: String, ok: Bool) -> (id: String, group: Group)? {
        var groups = load()
        guard let id = groups.first(where: { $0.value.projects.contains(projectID) })?.key, var g = groups[id] else { return nil }
        if ok {
            if !g.ready.contains(projectID) { g.ready.append(projectID) }
            g.failed.removeAll { $0 == projectID }
        } else if !g.ready.contains(projectID), !g.failed.contains(projectID) {
            g.failed.append(projectID)
        }
        var announce: (id: String, group: Group)?
        if g.isComplete && !g.notified {
            g.notified = true
            announce = (id: id, group: g)
        }
        groups[id] = g
        save(groups)
        return announce
    }

    /// A video whose upload failed before it had a project counts as needing attention (it has its own Retry).
    @discardableResult
    static func uploadFailed(_ groupID: String, itemID: UUID) -> (id: String, group: Group)? {
        let placeholder = "upload-" + itemID.uuidString
        var groups = load()
        guard var g = groups[groupID] else { return nil }
        if !g.projects.contains(placeholder) { g.projects.append(placeholder) }
        groups[groupID] = g
        save(groups)
        return finished(placeholder, ok: false)
    }

    /// The retried upload got its project: the placeholder is replaced.
    static func replacePlaceholder(_ groupID: String, itemID: UUID) {
        var groups = load()
        guard var g = groups[groupID] else { return }
        let placeholder = "upload-" + itemID.uuidString
        g.projects.removeAll { $0 == placeholder }
        g.failed.removeAll { $0 == placeholder }
        groups[groupID] = g
        save(groups)
    }

    // MARK: The words (EN / ES through L10n)

    static func title(_ g: Group) -> String {
        let ready = Set(g.ready).count, failed = g.done - ready
        if failed == 0 { return L10n.tr(ready == 1 ? "Your video is ready ✨" : "Your \(ready) videos are ready ✨") }
        if ready == 0 { return L10n.tr(failed == 1 ? "1 video needs attention" : "\(failed) videos need attention") }
        // Whole sentences (Spanish singular / plural: "1 listo", "9 listos"; "1 necesita", "2 necesitan").
        switch (ready == 1, failed == 1) {
        case (true, true): return L10n.tr("1 ready, 1 needs attention")
        case (true, false): return L10n.tr("1 ready, \(failed) need attention")
        case (false, true): return L10n.tr("\(ready) ready, 1 needs attention")
        case (false, false): return L10n.tr("\(ready) ready, \(failed) need attention")
        }
    }

    /// The one notification of the group (only when CutSell is not on screen; on screen the in-app notice shows).
    static func post(_ id: String, _ g: Group) {
        guard AppPrefs.processingAlertsOn, UIApplication.shared.applicationState != .active else { return }
        let content = UNMutableNotificationContent()
        content.title = title(g)
        content.body = L10n.tr("Open CutSell to review and export them.")
        content.sound = .default
        content.userInfo = ["bulk_id": id]
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: "bulk-ready-\(id)", content: content, trigger: nil))
        CutSellDiagnostics.log("bulk_ready_notified", ["videos": String(g.total), "failed": String(g.done - Set(g.ready).count)])
    }

    /// Mix (the server sends one notice for the whole Mix): "Your 6 videos are ready ✨".
    static func mixTitle(videos: Int) -> String {
        L10n.tr(videos == 1 ? "Your video is ready ✨" : videos > 1 ? "Your \(videos) videos are ready ✨" : "Your videos are ready ✨")
    }
}
