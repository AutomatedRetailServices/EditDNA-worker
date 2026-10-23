import Foundation
import UserNotifications

struct CutSellNotification: Codable, Identifiable, Hashable {
    let notificationID: String
    let createdAt: String
    let projectID: String
    let kind: String
    let payload: JSONValue

    var id: String { notificationID }

    enum CodingKeys: String, CodingKey {
        case notificationID = "notification_id"
        case createdAt = "created_at"
        case projectID = "project_id"
        case kind, payload
    }
}

private struct NotificationListResponse: Codable {
    let notifications: [CutSellNotification]
}

@MainActor
final class NotificationCenterModel: ObservableObject {
    @Published private(set) var items: [CutSellNotification] = []
    @Published private(set) var unreadCount = 0
    @Published var errorMessage: String?

    private let api = APIClient.shared
    private let seenKey = "cutsell.notifications.seen"

    func requestPermission() async {
        do {
            _ = try await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge])
        } catch {
            // Notification permission is optional; never block CutSell startup.
        }
    }

    func refresh(session: CutSellSession, surfaceLocalAlerts: Bool = true) async {
        do {
            let response: NotificationListResponse = try await api.request(
                "/v1/notifications",
                query: [
                    URLQueryItem(name: "user_id", value: session.userID),
                    URLQueryItem(name: "limit", value: "50"),
                ]
            )
            let seen = Set(UserDefaults.standard.stringArray(forKey: seenKey) ?? [])
            let newItems = response.notifications.filter { !seen.contains($0.notificationID) }
            items = response.notifications
            unreadCount = newItems.count
            if surfaceLocalAlerts {
                for item in newItems.prefix(3) {
                    await scheduleLocalAlert(for: item)
                }
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func markAllRead() {
        let ids = items.map(\.notificationID)
        UserDefaults.standard.set(ids, forKey: seenKey)
        unreadCount = 0
        UNUserNotificationCenter.current().setBadgeCount(0)
    }

    /// Notices already shown as an iPhone notification (here or by the background refresh): never twice.
    private static let surfacedKey = "cutsell.notifications.surfaced"
    private static func alreadySurfaced(_ id: String) -> Bool {
        (UserDefaults.standard.stringArray(forKey: surfacedKey) ?? []).contains(id)
    }
    private static func markSurfaced(_ id: String) {
        var ids = UserDefaults.standard.stringArray(forKey: surfacedKey) ?? []
        ids.append(id)
        UserDefaults.standard.set(Array(ids.suffix(200)), forKey: surfacedKey)
    }

    /// Background refresh: the server's one notice for a finished Mix -> "Your 6 videos are ready ✨".
    static func surfaceMixNotices(session: CutSellSession) async {
        guard AppPrefs.processingAlertsOn,
              let response: NotificationListResponse = try? await APIClient.shared.request(
                "/v1/notifications", query: [URLQueryItem(name: "user_id", value: session.userID),
                                             URLQueryItem(name: "limit", value: "20")], timeout: 20) else { return }
        for item in response.notifications where item.kind == "draft_ready" && item.payload["mix_id"] != nil
            && !alreadySurfaced(item.notificationID) {
            let content = UNMutableNotificationContent()
            content.title = GroupNotify.mixTitle(videos: Self.videos(item))
            content.body = L10n.tr("Open CutSell to review and export them.")
            content.sound = .default
            content.userInfo = ["project_id": item.projectID, "kind": item.kind]
            try? await UNUserNotificationCenter.current().add(
                UNNotificationRequest(identifier: item.notificationID, content: content, trigger: nil))
            markSurfaced(item.notificationID)
        }
    }

    private static func videos(_ item: CutSellNotification) -> Int {
        if case .number(let n)? = item.payload["videos"] { return Int(n) }
        return 0
    }

    private func scheduleLocalAlert(for item: CutSellNotification) async {
        // You → Preferences → "Processing complete" off: no alert (the list still shows it).
        guard AppPrefs.processingAlertsOn, !Self.alreadySurfaced(item.notificationID) else { return }
        // 23 Oct: a Bulk video has no notification of its own (its group sends one when all are done).
        if (item.kind == "draft_ready" || item.kind == "processing_failed"), GroupNotify.isGrouped(item.projectID) { return }
        let content = UNMutableNotificationContent()
        content.title = "CutSell"
        switch item.kind {
        case "draft_ready" where item.payload["mix_id"] != nil:
            // Mix: one notice for all its videos.
            content.title = GroupNotify.mixTitle(videos: Self.videos(item))
            content.body = "Open CutSell to review and export them."
        case "draft_ready":
            content.body = "Your AI draft is ready to edit."
        case "render_finished":
            content.body = "Your finished CutSell video is ready."
        case "processing_failed":
            content.body = item.payload["error"]?.stringValue == "no_speech"
                ? "We couldn't hear anyone talking in this video" : "This cut needs another try."
        case "render_failed":
            content.body = "The export failed, but your draft is safe."
        default:
            return
        }
        content.body = L10n.tr(content.body)
        content.sound = .default
        content.userInfo = ["project_id": item.projectID, "kind": item.kind]
        let request = UNNotificationRequest(
            identifier: item.notificationID,
            content: content,
            trigger: nil
        )
        try? await UNUserNotificationCenter.current().add(request)
        Self.markSurfaced(item.notificationID)
        try? await UNUserNotificationCenter.current().setBadgeCount(unreadCount)
    }
}
