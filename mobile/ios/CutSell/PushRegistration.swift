import Foundation
import UIKit
import UserNotifications

/// 23 Oct (PO): Apple push notifications, so "your videos are ready" arrives even when CutSell was closed.
/// The iPhone's token goes to the server (POST /v1/push/register, with the app language); the server then sends
/// the "ready" notifications (Single / Combine one per cut, a Bulk ONE when all its videos are done, a Mix one).
/// While the server sends them (`active`), CutSell posts none of its own for those -- never twice.
@MainActor
enum PushRegistration {
    private static let tokenKey = "cutsell.push.token"
    private static let activeKey = "cutsell.push.active"

    /// The server sends the "ready" notifications for this iPhone.
    nonisolated static var active: Bool { UserDefaults.standard.bool(forKey: activeKey) }

    /// After the notification permission: ask iOS for this iPhone's push token (answers in the app delegate).
    static func start() async {
        let settings = await UNUserNotificationCenter.current().notificationSettings()
        guard settings.authorizationStatus == .authorized || settings.authorizationStatus == .provisional else {
            UserDefaults.standard.set(false, forKey: activeKey)
            return
        }
        UIApplication.shared.registerForRemoteNotifications()
        await sendSaved()
    }

    static func received(_ deviceToken: Data) {
        let hex = deviceToken.map { String(format: "%02x", $0) }.joined()
        UserDefaults.standard.set(hex, forKey: tokenKey)
        Task { await sendSaved() }
    }

    static func failed(_ error: Error) {
        UserDefaults.standard.set(false, forKey: activeKey)
        CutSellDiagnostics.log("push_register_failed", ["error": String(describing: error).prefix(80).description])
    }

    /// Sends the saved token (again at every launch: the server keeps it 120 days; the user may have signed in since).
    static func sendSaved() async {
        guard let token = UserDefaults.standard.string(forKey: tokenKey), !token.isEmpty,
              let session = await APIClient.shared.session else { return }
        struct Body: Encodable { let user_id: String; let token: String; let language: String; let environment: String }
        struct Answer: Decodable { let ok: Bool; let push_active: Bool }
        #if DEBUG
        let environment = "sandbox"
        #else
        let environment = "production"
        #endif
        do {
            let answer: Answer = try await APIClient.shared.request(
                "/v1/push/register", method: "POST",
                body: Body(user_id: session.userID, token: token, language: L10nState.code, environment: environment),
                timeout: 20)
            UserDefaults.standard.set(answer.push_active, forKey: activeKey)
            CutSellDiagnostics.log("push_registered", ["active": answer.push_active ? "1" : "0"])
        } catch {
            // An older server (404) or no network: CutSell keeps its own notifications.
            UserDefaults.standard.set(false, forKey: activeKey)
        }
    }

    /// A "ready" notice the server sends by push (shown by iOS when CutSell is not on screen).
    nonisolated static func isReadyPush(_ userInfo: [AnyHashable: Any]) -> Bool {
        guard let kind = userInfo["kind"] as? String else { return false }
        return ["draft_ready", "processing_failed", "bulk_ready", "mix_ready"].contains(kind) && userInfo["aps"] != nil
    }
}
