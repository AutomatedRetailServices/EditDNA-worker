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

    private func scheduleLocalAlert(for item: CutSellNotification) async {
        let content = UNMutableNotificationContent()
        content.title = "CutSell"
        switch item.kind {
        case "draft_ready":
            content.body = "Your AI draft is ready to edit."
        case "render_finished":
            content.body = "Your finished CutSell video is ready."
        case "processing_failed":
            content.body = "This cut needs another try."
        case "render_failed":
            content.body = "The export failed, but your draft is safe."
        default:
            return
        }
        content.sound = .default
        content.userInfo = ["project_id": item.projectID, "kind": item.kind]
        let request = UNNotificationRequest(
            identifier: item.notificationID,
            content: content,
            trigger: nil
        )
        try? await UNUserNotificationCenter.current().add(request)
        try? await UNUserNotificationCenter.current().setBadgeCount(unreadCount)
    }
}
