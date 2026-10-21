import BackgroundTasks
import Photos
import SwiftUI
import UIKit
import UserNotifications

// MARK: - Model (/v1/plan …)

struct PlanReminder: Codable, Hashable {
    var label: String
    var time: String          // "HH:mm"
    var on: Bool
}

struct PlanPrefs: Codable, Hashable {
    var perDay: Int
    var days: [Int]           // Monday = 0
    var reminders: [PlanReminder]
    var tz: String?
    // Goal v2 (flexible, Figma 1140:2773): her own posting times (one video each), how many videos, start day, reminders
    var times: [String]? = nil
    var count: Int? = nil
    var start: String? = nil
    var remind: Bool? = nil
    enum CodingKeys: String, CodingKey { case perDay = "per_day", days, reminders, tz, times, count, start, remind }

    /// The posting times: hers (Goal v2), else the old reminders that are on.
    var postingTimes: [String] {
        if let times, !times.isEmpty { return times.sorted() }
        let on = reminders.filter(\.on).map(\.time).sorted()
        return on.isEmpty ? ["09:00", "19:00"] : Array(on.prefix(max(1, perDay)))
    }
}

struct PlanSlot: Codable, Hashable, Identifiable {
    let date: String          // yyyy-MM-dd
    let time: String
    let projectID: String
    var pinned: Bool?
    let product: String?
    let brand: String?
    let productID: String?
    let mixCode: String?
    let source: String?
    var postedAt: String?
    var caption: String?
    var captionSource: String?
    let durationSec: Double?
    var id: String { projectID }
    enum CodingKeys: String, CodingKey {
        case date, time, projectID = "project_id", pinned, product, brand, productID = "product_id", mixCode = "mix_code"
        case source, postedAt = "posted_at", caption, captionSource = "caption_source", durationSec = "duration_sec"
    }
}

struct PlanMissing: Codable, Hashable {
    let days: [String]
    let videos: Int
    let coveredDays: Int
    let totalDays: Int
    enum CodingKeys: String, CodingKey { case days, videos, coveredDays = "covered_days", totalDays = "total_days" }
}

struct PlanWeek: Codable, Hashable {
    let start: String
    var slots: [PlanSlot]
    let missing: PlanMissing?
    // Goal v2: the plan's last day, how many videos she asked for and how many fit the rules
    var end: String? = nil
    var flex: Bool? = nil
    var asked: Int? = nil
    var placed: Int? = nil
    var endsEarly: Bool? = nil
    enum CodingKeys: String, CodingKey { case start, slots, missing, end, flex, asked, placed, endsEarly = "ends_early" }
}

struct PlanSummary: Codable, Hashable {
    let ready: Int
    let products: Int
    var noProduct: Int? = nil          // videos without a product: "no product", never counted as products
    let perDay: Int
    let coversDays: Int
    enum CodingKeys: String, CodingKey { case ready, products, noProduct = "no_product", perDay = "per_day", coversDays = "covers_days" }
}

struct PlanState: Codable, Hashable {
    var prefs: PlanPrefs
    let today: String
    let summary: PlanSummary
    var week: PlanWeek?
    /// Her personal videos (seen in Edits, never planned). Absent on an older server.
    var personal: [String]? = nil
}

enum PlanDay {
    static let iso: DateFormatter = {
        let f = DateFormatter(); f.calendar = Calendar(identifier: .gregorian); f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"; return f
    }()
    static func date(_ day: String) -> Date? { iso.date(from: day) }
    /// "Tue 21"
    static func short(_ day: String) -> String {
        guard let d = date(day) else { return day }
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US"); f.dateFormat = "EEE d"
        return f.string(from: d)
    }
    static func weekdayName(_ day: String) -> String {
        guard let d = date(day) else { return day }
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US"); f.dateFormat = "EEE"
        return f.string(from: d)
    }
    /// "09:00" -> "9:00 AM"
    static func clock(_ time: String) -> String {
        let parts = time.split(separator: ":").compactMap { Int($0) }
        guard parts.count >= 2 else { return time }
        let hour = parts[0] % 12 == 0 ? 12 : parts[0] % 12
        return String(format: "%d:%02d %@", hour, parts[1], parts[0] < 12 ? "AM" : "PM")
    }
    static let dots = ["PlanDotCyan", "PlanDotMagenta", "PlanDotGreen", "PlanDotAmber"]
}

// MARK: - Store

/// The creator's posting plan. `-CutSellPlanMock` (simulator probe) answers from memory.
@MainActor
final class PlanWeekStore: ObservableObject {
    static let shared = PlanWeekStore()

    @Published private(set) var state: PlanState?
    @Published var lastError: String?
    /// A reminder was tapped: the Plan tab opens this day.
    @Published var requestedDay: String?
    let mock = ProcessInfo.processInfo.arguments.contains("-CutSellPlanMock")
    private var mockSlots: [PlanSlot] = []
    private var mockPrefs = PlanPrefs(perDay: 2, days: [0, 1, 2, 3, 4, 5],
                                      reminders: [PlanReminder(label: "Morning", time: "09:00", on: true),
                                                  PlanReminder(label: "Evening", time: "19:00", on: true),
                                                  PlanReminder(label: "Night", time: "22:00", on: false)], tz: nil)
    private var mockBuilt = false

    var hasWeek: Bool { !(state?.week?.slots.isEmpty ?? true) }
    private var mockPersonal: Set<String> = []

    func isPersonal(_ projectID: String) -> Bool { state?.personal?.contains(projectID) ?? false }

    /// Edits ••• → "Personal · don't plan" / "Use in Plan" (POST /v1/plan/personal).
    func setPersonal(_ projectID: String, _ personal: Bool, session: CutSellSession?) async throws {
        if mock {
            if personal { mockPersonal.insert(projectID); mockSlots.removeAll { $0.projectID == projectID } }
            else { mockPersonal.remove(projectID) }
            state = mockState(); return
        }
        guard let session else { throw APIError.invalidResponse }
        struct Body: Encodable { let user_id: String; let project_id: String; let personal: Bool }
        state = try await APIClient.shared.request("/v1/plan/personal", method: "POST",
                                                   body: Body(user_id: session.userID, project_id: projectID, personal: personal))
        PostReminders.shared.schedule(state)
        CutSellDiagnostics.log("plan_personal", ["personal": personal ? "1" : "0"])
    }

    /// A stable color dot per product (the same product, the same color everywhere).
    func dot(for slot: PlanSlot) -> String {
        let products = Array(Set((state?.week?.slots ?? []).map { $0.productID ?? $0.projectID })).sorted()
        let index = products.firstIndex(of: slot.productID ?? slot.projectID) ?? 0
        return PlanDay.dots[index % PlanDay.dots.count]
    }

    /// Simulator only (-CutSellPlanMock + CUTSELL_PLAN_FIXTURE): her real plan computed read-only on the server,
    /// before and after "Build my plan" (base64 JSON {"before": PlanState, "after": PlanState}).
    private lazy var fixture: (before: PlanState, after: PlanState)? = {
        guard mock, let raw = ProcessInfo.processInfo.environment["CUTSELL_PLAN_FIXTURE"], let data = Data(base64Encoded: raw) else { return nil }
        struct Pair: Decodable { let before: PlanState; let after: PlanState }
        guard let pair = try? JSONDecoder().decode(Pair.self, from: data) else { return nil }
        return (pair.before, pair.after)
    }()
    private var fixtureBuilt = false

    func refresh(session: CutSellSession?) async {
        if let fixture { state = fixtureBuilt ? fixture.after : fixture.before; return }
        if mock { state = mockState(); return }
        guard let session else { return }
        do {
            state = try await APIClient.shared.request("/v1/plan", query: [URLQueryItem(name: "user_id", value: session.userID)],
                                                       timeout: 20)
            PostReminders.shared.schedule(state)
        } catch APIError.http(404, _) {
            state = nil
        } catch { lastError = error.localizedDescription }
    }

    func savePrefs(_ prefs: PlanPrefs, session: CutSellSession?) async throws {
        if fixture != nil { return }
        if mock { mockPrefs = prefs; state = mockState(); return }
        guard let session else { throw APIError.invalidResponse }
        struct Body: Encodable {
            let user_id: String; let days: [Int]; let tz: String
            let times: [String]; let count: Int?; let start: String?; let remind: Bool
        }
        state = try await APIClient.shared.request("/v1/plan/prefs", method: "PUT",
                                                   body: Body(user_id: session.userID, days: prefs.days, tz: TimeZone.current.identifier,
                                                              times: prefs.postingTimes, count: prefs.count, start: prefs.start,
                                                              remind: prefs.remind ?? true))
    }

    func build(session: CutSellSession?) async throws {
        if let fixture { fixtureBuilt = true; state = fixture.after; PostReminders.shared.schedule(state); return }
        if mock { mockBuild(); state = mockState(); PostReminders.shared.schedule(state); return }
        guard let session else { throw APIError.invalidResponse }
        struct Body: Encodable { let user_id: String }
        state = try await APIClient.shared.request("/v1/plan/week/build", method: "POST", body: Body(user_id: session.userID), timeout: 30)
        PostReminders.shared.schedule(state)
        CutSellDiagnostics.log("plan_week_built", ["videos": String(state?.week?.slots.count ?? 0)])
    }

    func move(_ slot: PlanSlot, to day: String, time: String, session: CutSellSession?) async throws {
        if mock {
            if let i = mockSlots.firstIndex(where: { $0.projectID == slot.projectID }) {
                let s = mockSlots[i]
                mockSlots[i] = PlanSlot(date: day, time: time, projectID: s.projectID, pinned: true, product: s.product, brand: s.brand,
                                        productID: s.productID, mixCode: s.mixCode, source: s.source, postedAt: s.postedAt,
                                        caption: s.caption, captionSource: s.captionSource, durationSec: s.durationSec)
                mockSlots.sort { ($0.date, $0.time) < ($1.date, $1.time) }
            }
            state = mockState(); return
        }
        guard let session else { throw APIError.invalidResponse }
        struct Body: Encodable { let user_id: String; let project_id: String; let date: String; let time: String }
        state = try await APIClient.shared.request("/v1/plan/slots/move", method: "POST",
                                                   body: Body(user_id: session.userID, project_id: slot.projectID, date: day, time: time))
        PostReminders.shared.schedule(state)
    }

    func remove(_ slot: PlanSlot, session: CutSellSession?) async throws {
        if mock { mockSlots.removeAll { $0.projectID == slot.projectID }; state = mockState(); return }
        guard let session else { throw APIError.invalidResponse }
        struct Body: Encodable { let user_id: String; let project_id: String }
        state = try await APIClient.shared.request("/v1/plan/slots/remove", method: "POST",
                                                   body: Body(user_id: session.userID, project_id: slot.projectID))
        PostReminders.shared.schedule(state)
    }

    func markPosted(_ slot: PlanSlot, session: CutSellSession?) async throws {
        if mock {
            if let i = mockSlots.firstIndex(where: { $0.projectID == slot.projectID }) { mockSlots[i].postedAt = "now" }
            state = mockState(); return
        }
        guard let session else { throw APIError.invalidResponse }
        struct Body: Encodable { let user_id: String; let project_id: String; let posted: Bool }
        state = try await APIClient.shared.request("/v1/plan/posted", method: "POST",
                                                   body: Body(user_id: session.userID, project_id: slot.projectID, posted: true))
        PostReminders.shared.schedule(state)
        CutSellDiagnostics.log("plan_marked_posted", [:])
    }

    func saveCaption(_ slot: PlanSlot, caption: String, session: CutSellSession?) async throws {
        if mock {
            if let i = mockSlots.firstIndex(where: { $0.projectID == slot.projectID }) {
                mockSlots[i].caption = caption; mockSlots[i].captionSource = "user"
            }
            state = mockState(); return
        }
        guard let session else { throw APIError.invalidResponse }
        struct Body: Encodable { let user_id: String; let project_id: String; let caption: String }
        struct Done: Decodable {}
        let _: Done = try await APIClient.shared.request("/v1/plan/caption", method: "PUT",
                                                         body: Body(user_id: session.userID, project_id: slot.projectID, caption: caption))
        await refresh(session: session)
    }

    // MARK: Probe-only plan (videos of the four origins, built like the server: no product twice in a row)

    private func mockState() -> PlanState {
        let today = PlanDay.iso.string(from: Date())
        let ready = 9 - mockSlots.filter { $0.postedAt != nil }.count
        let flex = mockPrefs.times != nil
        let week = mockBuilt ? PlanWeek(start: mockSlots.first?.date ?? today, slots: mockSlots, missing: flex ? nil : mockMissing(),
                                        end: mockSlots.last?.date, flex: flex, asked: mockSlots.count, placed: mockSlots.count,
                                        endsEarly: false) : nil
        return PlanState(prefs: mockPrefs, today: today,
                         summary: PlanSummary(ready: ready, products: 6, noProduct: 0, perDay: mockPrefs.postingTimes.count,
                                              coversDays: ready / max(1, mockPrefs.perDay)), week: week,
                         personal: mockPersonal.sorted())
    }

    private func mockBuild() {
        let library: [(String, String, String, String, String?)] = [
            ("CeraVe", "Hydrating Cleanser", "p1", "single", nil), ("Glow Recipe", "Watermelon Toner", "p5", "mix", "H1 · B1 · C1"),
            ("Dior", "Lip Glow", "p2", "combine", nil), ("Glow Recipe", "Watermelon Toner", "p5", "mix", "H2 · B2 · C2"),
            ("Old Navy", "Denim Jacket", "p3", "bulk", nil), ("Glow Recipe", "Watermelon Toner", "p5", "mix", "H1 · B2 · C1"),
            ("Rhino", "Power Bank", "p4", "bulk", nil), ("Glow Recipe", "Watermelon Toner", "p5", "mix", "H2 · B1 · C2"),
            ("CeraVe", "Moisturizing Cream", "p6", "single", nil)]
        let calendar = Calendar(identifier: .gregorian)
        let times = mockPrefs.postingTimes
        let wanted = min(mockPrefs.count ?? library.count, library.count)
        var day = calendar.startOfDay(for: mockPrefs.start.flatMap { PlanDay.date($0) } ?? Date())
        var out: [PlanSlot] = []
        var k = 0
        while k < wanted {
            let weekday = (calendar.component(.weekday, from: day) + 5) % 7
            if mockPrefs.days.contains(weekday) {
                for time in times where k < wanted {
                    let v = library[k]
                    out.append(PlanSlot(date: PlanDay.iso.string(from: day), time: time, projectID: "plan_\(k)", pinned: false,
                                        product: v.1, brand: v.0, productID: v.2, mixCode: v.4, source: v.3, postedAt: nil,
                                        caption: k == 8 ? nil : "So good on my skin ✨ #\(v.0.lowercased().filter(\.isLetter)) #tiktokshop",
                                        captionSource: k == 8 ? nil : "ai", durationSec: 31 + Double(k)))
                    k += 1
                }
            }
            day = calendar.date(byAdding: .day, value: 1, to: day) ?? day
            if out.count > 40 { break }
        }
        // Built again: what she already posted stays posted, her own captions stay hers.
        let old = Dictionary(uniqueKeysWithValues: mockSlots.map { ($0.projectID, $0) })
        mockSlots = out.map { slot in
            var slot = slot
            if let before = old[slot.projectID] {
                slot.postedAt = before.postedAt
                if before.captionSource == "user" { slot.caption = before.caption; slot.captionSource = "user" }
            }
            return slot
        }
        mockBuilt = true
    }

    private func mockMissing() -> PlanMissing {
        let used = Set(mockSlots.map(\.date))
        let calendar = Calendar(identifier: .gregorian)
        let today = calendar.startOfDay(for: Date())
        var short: [String] = []
        var videos = 0
        for offset in 0..<7 {
            guard let day = calendar.date(byAdding: .day, value: offset, to: today) else { continue }
            let weekday = (calendar.component(.weekday, from: day) + 5) % 7
            if weekday == 0 && offset > 0 { break }                    // this week only (Monday starts the next)
            guard mockPrefs.days.contains(weekday) else { continue }
            let key = PlanDay.iso.string(from: day)
            let count = mockSlots.filter { $0.date == key }.count
            if count < mockPrefs.perDay { short.append(key); videos += mockPrefs.perDay - count }
        }
        let total = used.union(short).count
        return PlanMissing(days: short, videos: videos, coveredDays: total - short.count, totalDays: total)
    }
}

// MARK: - The video file to post (exported once, kept on the iPhone)

@MainActor
final class PostExporter: ObservableObject {
    static let shared = PostExporter()
    @Published var progress: [String: Double] = [:]

    private var folder: URL {
        let url = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("PostVideos", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    func localFile(_ projectID: String) -> URL? {
        let url = folder.appendingPathComponent("\(projectID).mp4")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// The finished video of a planned cut: the file already exported (at night) or exported now, with its progress.
    func file(for projectID: String, session: CutSellSession?) async throws -> URL {
        if let ready = localFile(projectID) { return ready }
        let target = folder.appendingPathComponent("\(projectID).mp4")
        if PlanWeekStore.shared.mock {
            #if DEBUG
            for step in 1...5 { progress[projectID] = Double(step) / 5; try await Task.sleep(nanoseconds: 150_000_000) }
            if let made = await ProbeMedia.makeVideo(seconds: 3) {
                try? FileManager.default.removeItem(at: target)
                try FileManager.default.copyItem(at: made, to: target)
                progress[projectID] = nil
                return target
            }
            #endif
            throw APIError.invalidResponse
        }
        guard let session else { throw APIError.invalidResponse }
        progress[projectID] = 0.02
        defer { progress[projectID] = nil }
        let snap: DraftSnapshot = try await APIClient.shared.request("/v1/projects/\(projectID)/draft",
                                                                     query: [URLQueryItem(name: "user_id", value: session.userID)])
        let sources = snap.sources.compactMap { object -> JSONValue? in
            guard let sid = object["source_asset_id"], let uri = object["uri"] else { return nil }
            return .object(["source_asset_id": sid, "original_name": object["original_name"] ?? .string("video.mp4"), "uri": uri])
        }
        let response: ExportSubmitResponse = try await APIClient.shared.request(
            "/v1/exports/jobs", method: "POST",
            body: JSONValue.object(["project_id": .string(projectID), "user_id": .string(session.userID), "draft": snap.draft,
                                    "sources": .array(sources)]))
        var link: URL?
        for _ in 0..<800 {
            let job: JobStatus = try await APIClient.shared.request("/v1/jobs/\(response.jobID)")
            progress[projectID] = max(0.03, min(0.9, Double(job.progress ?? 0) / 100 * 0.9))
            if job.state == "finished" { link = job.result?["download_url"]?.stringValue.flatMap(URL.init(string:)); break }
            if ["failed", "canceled"].contains(job.state) { break }
            try await Task.sleep(nanoseconds: 1_500_000_000)
        }
        guard let link else { throw APIError.invalidResponse }
        let (temporary, _) = try await URLSession.shared.download(from: link)
        try? FileManager.default.removeItem(at: target)
        try FileManager.default.moveItem(at: temporary, to: target)
        CutSellDiagnostics.log("post_video_exported", ["mb": String(((try? FileManager.default.attributesOfItem(atPath: target.path)[.size] as? Int) ?? 0) / 1_048_576)])
        return target
    }

    /// Files of videos already posted or no longer planned are removed.
    func clean(keeping ids: Set<String>) {
        for url in (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [] {
            if !ids.contains(url.deletingPathExtension().lastPathComponent) { try? FileManager.default.removeItem(at: url) }
        }
    }
}

/// The night before: with WiFi and the iPhone charging, tomorrow's videos are exported and downloaded, so
/// "Share to TikTok" is instant. iOS decides the exact moment (BGProcessingTask).
@MainActor
enum PostPrefetch {
    static let taskID = "ai.cutsell.app.prefetch-posts"

    static func register() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: taskID, using: nil) { task in
            let work = Task { @MainActor in
                let done = await run()
                task.setTaskCompleted(success: done)
                schedule()
            }
            task.expirationHandler = { work.cancel() }
        }
    }

    static func schedule() {
        guard PlanWeekStore.shared.hasWeek else { return }
        let request = BGProcessingTaskRequest(identifier: taskID)
        request.requiresNetworkConnectivity = true
        request.requiresExternalPower = true
        var evening = Calendar.current.date(bySettingHour: 21, minute: 0, second: 0, of: Date()) ?? Date()
        if evening < Date() { evening = Date().addingTimeInterval(15 * 60) }
        request.earliestBeginDate = evening
        try? BGTaskScheduler.shared.submit(request)
    }

    /// Exports tomorrow's (and today's remaining) unposted videos. True when all of them are on the iPhone.
    static func run() async -> Bool {
        if await APIClient.shared.session == nil, let saved = try? KeychainStore.load() {
            await APIClient.shared.setSession(saved)
        }
        let session = await APIClient.shared.session
        await PlanWeekStore.shared.refresh(session: session)
        guard let state = PlanWeekStore.shared.state, let week = state.week else { return true }
        let tomorrow = PlanDay.iso.string(from: Date().addingTimeInterval(86_400))
        let wanted = week.slots.filter { $0.postedAt == nil && ($0.date == tomorrow || $0.date == state.today) }
        var ok = true
        for slot in wanted where PostExporter.shared.localFile(slot.projectID) == nil {
            do { _ = try await PostExporter.shared.file(for: slot.projectID, session: session) } catch { ok = false }
        }
        PostExporter.shared.clean(keeping: Set(week.slots.filter { $0.postedAt == nil }.map(\.projectID)))
        CutSellDiagnostics.log("post_prefetch", ["videos": String(wanted.count), "ok": ok ? "yes" : "no"])
        return ok
    }
}

/// Local reminders at the times she chose ("Time to post: <product>"); tapping one opens that day, ready to share.
@MainActor
final class PostReminders: NSObject, UNUserNotificationCenterDelegate {
    static let shared = PostReminders()
    static let prefix = "cutsell-post-"

    func install() { UNUserNotificationCenter.current().delegate = self }

    func schedule(_ state: PlanState?) {
        let center = UNUserNotificationCenter.current()
        center.getPendingNotificationRequests { requests in
            let old = requests.map(\.identifier).filter { $0.hasPrefix(Self.prefix) }
            center.removePendingNotificationRequests(withIdentifiers: old)
            guard let state, let week = state.week else { return }
            // Goal v2: "Remind me at each time" (one reminder per planned video); old Goal: the reminders that are on
            let on = state.prefs.times != nil ? ((state.prefs.remind ?? true) ? Set(week.slots.map(\.time)) : [])
                                              : Set(state.prefs.reminders.filter(\.on).map(\.time))
            for slot in week.slots where slot.postedAt == nil && on.contains(slot.time) {
                guard let day = PlanDay.date(slot.date) else { continue }
                let parts = slot.time.split(separator: ":").compactMap { Int($0) }
                guard parts.count == 2, let when = Calendar.current.date(bySettingHour: parts[0], minute: parts[1], second: 0, of: day),
                      when > Date() else { continue }
                let content = UNMutableNotificationContent()
                content.title = "Time to post: \(slot.product ?? "your video")"
                content.body = [slot.brand, "Caption and video are ready."].compactMap { $0 }.joined(separator: " · ")
                content.sound = .default
                content.userInfo = ["plan_day": slot.date, "project_id": slot.projectID]
                let trigger = UNCalendarNotificationTrigger(
                    dateMatching: Calendar.current.dateComponents([.year, .month, .day, .hour, .minute], from: when), repeats: false)
                center.add(UNNotificationRequest(identifier: Self.prefix + slot.projectID, content: content, trigger: trigger))
            }
        }
    }

    /// Probe / "Test reminder": one reminder in a few seconds.
    func test(_ slot: PlanSlot, seconds: TimeInterval = 4) {
        let content = UNMutableNotificationContent()
        content.title = "Time to post: \(slot.product ?? "your video")"
        content.body = "Caption and video are ready."
        content.userInfo = ["plan_day": slot.date, "project_id": slot.projectID]
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: Self.prefix + "test",
                                                                     content: content,
                                                                     trigger: UNTimeIntervalNotificationTrigger(timeInterval: seconds, repeats: false)))
    }

    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification,
                                            withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        completionHandler([.banner, .sound, .list])
    }

    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse,
                                            withCompletionHandler completionHandler: @escaping () -> Void) {
        let day = response.notification.request.content.userInfo["plan_day"] as? String
        Task { @MainActor in
            if let day {
                PlanWeekStore.shared.requestedDay = day
                CutSellDiagnostics.log("post_reminder_opened", [:])
            }
            completionHandler()
        }
    }
}

// MARK: - Shared look

enum PlanStyle {
    static let cta = LinearGradient(stops: [.init(color: Color(hex: 0x1969FF), location: 0), .init(color: Color(hex: 0x7C3AED), location: 0.55),
                                            .init(color: Color(hex: 0x19E6FF), location: 1)], startPoint: .leading, endPoint: .trailing)
    static let background = LinearGradient(stops: [.init(color: Color(hex: 0x03050D), location: 0), .init(color: Color(hex: 0x050B1C), location: 0.55),
                                                   .init(color: Color(hex: 0x030409), location: 1)], startPoint: .top, endPoint: .bottom)
    static let muted = Color(hex: 0x99A3BD)
    static let card = Color(hex: 0x0F1726)
    static let border = Color(hex: 0x33405C)
}

struct PlanBackButton: View {
    let title: String
    let action: () -> Void
    var body: some View {
        Button(action: action) { Text("‹ \(title)").font(CS.inter(16, .semibold)).foregroundStyle(CS.cyan) }
            .accessibilityIdentifier("planBack")
    }
}

// MARK: - Plan my week · Goal v2 (flexible, Figma 12_Prototype 1140:2773)

struct PlanGoalView: View {
    let onBuilt: () -> Void
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var store = PlanWeekStore.shared
    @State private var count = 1
    @State private var times: [String] = ["09:00", "19:00"]
    @State private var days: [Int] = [0, 1, 2, 3, 4]
    @State private var start = PlanGoalView.tomorrow()
    @State private var remind = true
    @State private var loaded = false
    @State private var building = false
    @State private var editing: TimeEdit?
    @State private var pickingStart = false
    @State private var errorMessage: String?
    @State private var endedEarly: String?

    static let maxTimes = 8
    private var ready: Int { store.state?.summary.ready ?? 0 }
    private var products: Int { store.state?.summary.products ?? 0 }
    private var noProduct: Int { store.state?.summary.noProduct ?? 0 }
    private var perDay: Int { max(1, times.count) }

    var body: some View {
        ZStack {
            PlanStyle.background.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    PlanBackButton(title: "Plan") { dismiss() }.padding(.top, 8)
                    Text("Plan my week").font(CS.inter(28, .bold)).foregroundStyle(.white).padding(.top, 14)
                    Text("Pick how many videos and when. We spread them so every day has variety.")
                        .font(CS.inter(13)).foregroundStyle(PlanStyle.muted).fixedSize(horizontal: false, vertical: true).padding(.top, 6)

                    label("HOW MANY VIDEOS").padding(.top, 20)
                    howMany.padding(.top, 8)

                    label("POSTING TIMES · ANY TIME YOU WANT").padding(.top, 18)
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 4), alignment: .leading, spacing: 6) {
                        ForEach(Array(times.enumerated()), id: \.element) { index, time in
                            Button { editing = TimeEdit(id: index) } label: {
                                Text(PlanDay.clock(time)).font(CS.inter(13, .semibold)).foregroundStyle(.white).lineLimit(1)
                                    .minimumScaleFactor(0.8)
                                    .frame(maxWidth: .infinity).frame(height: 44)
                                    .background(Capsule().fill(CS.brandFull))
                            }
                            .buttonStyle(.plain)
                            .accessibilityIdentifier("goalTime_\(index)")
                        }
                        if times.count < Self.maxTimes {
                            Button { editing = TimeEdit(id: -1) } label: {
                                Text("+ Time").font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
                                    .frame(maxWidth: .infinity).frame(height: 44)
                                    .overlay(Capsule().strokeBorder(Color(hex: 0x596685), style: StrokeStyle(lineWidth: 1, dash: [4, 3])))
                            }
                            .buttonStyle(.plain)
                            .accessibilityIdentifier("goalAddTime")
                        }
                    }
                    .padding(.top, 8)
                    Text("\(times.count) time\(times.count == 1 ? "" : "s") = \(times.count) video\(times.count == 1 ? "" : "s") per day. Tap a time to change it.")
                        .font(CS.inter(11)).foregroundStyle(PlanStyle.muted).padding(.top, 8)
                        .accessibilityIdentifier("goalTimesHelp")

                    label("DAYS").padding(.top, 22)
                    HStack(spacing: 6) {
                        ForEach(0..<7, id: \.self) { d in
                            let on = days.contains(d)
                            Button {
                                if on { if days.count > 1 { days.removeAll { $0 == d } } } else { days = (days + [d]).sorted() }
                            } label: {
                                Text(["M", "T", "W", "T", "F", "S", "S"][d]).font(CS.inter(14, .bold))
                                    .foregroundStyle(on ? .white : PlanStyle.muted)
                                    .frame(width: 44, height: 44)
                                    .background { if on { Circle().fill(CS.brandFull) } else { Circle().fill(PlanStyle.card) } }
                                    .overlay(Circle().stroke(on ? .clear : PlanStyle.border, lineWidth: 1))
                            }
                            .buttonStyle(.plain)
                            .frame(maxWidth: .infinity)
                            .accessibilityIdentifier("goalDay_\(d)")
                        }
                    }
                    .padding(.top, 8)

                    label("START").padding(.top, 18)
                    Button { pickingStart = true } label: {
                        HStack {
                            Text(startText).font(CS.inter(14, .semibold)).foregroundStyle(.white)
                            Spacer()
                            Text("Change").font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
                        }
                        .padding(.horizontal, 15).frame(height: 48)
                        .background(PlanStyle.card, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(PlanStyle.border, lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .padding(.top, 8)
                    .accessibilityIdentifier("goalStart")

                    HStack {
                        Text("Remind me at each time").font(CS.inter(14, .semibold)).foregroundStyle(.white)
                        Spacer()
                        Button { remind.toggle() } label: {
                            Image(remind ? "PlanToggleOn" : "PlanToggleOff").resizable().frame(width: 44, height: 26)
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("goalRemind")
                        .accessibilityValue(remind ? "on" : "off")
                    }
                    .padding(.horizontal, 15).frame(height: 52)
                    .background(PlanStyle.card, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(PlanStyle.border, lineWidth: 1))
                    .padding(.top, 12)

                    summaryBox.padding(.top, 14)

                    Button { Task { await build() } } label: {
                        Group {
                            if building { ProgressView().tint(.white) } else { Text("Build my plan").font(CS.sf(15, .semibold)) }
                        }
                        .foregroundStyle(.white).frame(maxWidth: .infinity).frame(height: 50)
                        .background(Capsule().fill(PlanStyle.cta))
                        .shadow(color: Color(hex: 0x1969FF).opacity(0.25), radius: 9, y: 6)
                    }
                    .disabled(building || ready == 0)
                    .padding(.top, 18)
                    .accessibilityIdentifier("goalBuild")
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 30)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .preferredColorScheme(.dark)
        .task {
            await store.refresh(session: appState.session)
            guard !loaded else { return }
            loaded = true
            if let prefs = store.state?.prefs {
                times = prefs.postingTimes
                days = prefs.days.isEmpty ? [0, 1, 2, 3, 4] : prefs.days
                remind = prefs.remind ?? true
                if let saved = prefs.start, saved > PlanDay.iso.string(from: Date()) { start = saved }
            }
            count = max(1, min(store.state?.prefs.count ?? ready, max(ready, 1)))
        }
        .onChange(of: ready) { _, value in if count > value || count == 0 { count = max(1, value) } }
        .sheet(item: $editing) { edit in
            TimePickerSheet(time: edit.id >= 0 ? times[edit.id] : (times.last.map(Self.later) ?? "09:00"),
                            title: edit.id >= 0 ? "Change time" : "Add a time",
                            onDelete: edit.id >= 0 && times.count > 1 ? { times.remove(at: edit.id) } : nil) { value in
                var next = times
                if edit.id >= 0 { next[edit.id] = value } else { next.append(value) }
                times = Array(Set(next)).sorted().prefix(Self.maxTimes).map { $0 }
            }
            .presentationDetents([.height(edit.id >= 0 ? 380 : 330)])
            .presentationBackground(Color(hex: 0x0D121F))
        }
        .sheet(isPresented: $pickingStart) {
            StartDateSheet(day: start) { start = $0 }
                .presentationDetents([.height(470)])
                .presentationBackground(Color(hex: 0x0D121F))
        }
        .alert("CutSell", isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(errorMessage ?? "") }
        .alert("Your plan ends early", isPresented: Binding(get: { endedEarly != nil }, set: { if !$0 { endedEarly = nil; onBuilt() } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(endedEarly ?? "") }
    }

    struct TimeEdit: Identifiable { let id: Int }

    private var howMany: some View {
        HStack {
            Button { count = max(1, count - 1) } label: {
                Text("−").font(CS.inter(20, .bold)).foregroundStyle(.white)
                    .frame(width: 40, height: 40).background(Circle().fill(Color(hex: 0x1F2638)))
            }
            .buttonStyle(.plain)
            .disabled(count <= 1)
            .accessibilityIdentifier("goalMinus")
            Spacer()
            VStack(spacing: 2) {
                Text("\(count)").font(CS.inter(28, .bold)).foregroundStyle(.white).accessibilityIdentifier("goalCount")
                Text("of \(ready) ready video\(ready == 1 ? "" : "s")").font(CS.inter(11)).foregroundStyle(PlanStyle.muted)
            }
            Spacer()
            Button { count = min(max(ready, 1), count + 1) } label: {
                Text("+").font(CS.inter(20, .bold)).foregroundStyle(.white)
                    .frame(width: 40, height: 40).background(Circle().fill(CS.brandFull))
            }
            .buttonStyle(.plain)
            .disabled(count >= ready)
            .accessibilityIdentifier("goalPlus")
        }
        .padding(.horizontal, 13)
        .frame(height: 74)
        .background(PlanStyle.card, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(PlanStyle.border, lineWidth: 1))
    }

    /// "✦  11 videos · 3 per day · 4 days" / "Thu 22 → Tue 27 · 6 products"
    private var summaryBox: some View {
        let dates = Self.planDays(count: count, perDay: perDay, days: days, start: start)
        let range = dates.isEmpty ? "" : "\(Self.dayLabel(dates.first!)) → \(Self.dayLabel(dates.last!))"
        var line2 = range + " · \(products) product\(products == 1 ? "" : "s")"
        if noProduct > 0 { line2 += " · \(noProduct) without a product" }
        return VStack(alignment: .leading, spacing: 4) {
            Text("✦  \(count) video\(count == 1 ? "" : "s") · \(perDay) per day · \(dates.count) day\(dates.count == 1 ? "" : "s")")
                .font(CS.inter(14, .semibold)).foregroundStyle(.white)
            Text(line2).font(CS.inter(12)).foregroundStyle(PlanStyle.muted).padding(.leading, 20)
        }
        .padding(.horizontal, 13).padding(.vertical, 9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(hex: 0x121A2B), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color(hex: 0x19E5FF), lineWidth: 1))
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("goalSummary")
    }

    private func label(_ text: String) -> some View {
        Text(text).font(CS.inter(10, .bold)).foregroundStyle(CS.cyan)
    }

    private var startText: String {
        let today = PlanDay.iso.string(from: Date())
        let prefix = start == today ? "Today · " : start == Self.tomorrow() ? "Tomorrow · " : ""
        return prefix + Self.dayLabel(start)
    }

    /// The days the plan uses: her days from `start`, `perDay` videos each, until `count` videos.
    static func planDays(count: Int, perDay: Int, days: [Int], start: String) -> [String] {
        guard count > 0, !days.isEmpty, var day = PlanDay.date(start) else { return [] }
        let calendar = Calendar(identifier: .gregorian)
        var out: [String] = []
        var left = count
        var guardDays = 0
        while left > 0 && guardDays < 400 {
            let weekday = (calendar.component(.weekday, from: day) + 5) % 7      // Monday = 0
            if days.contains(weekday) { out.append(PlanDay.iso.string(from: day)); left -= perDay }
            day = calendar.date(byAdding: .day, value: 1, to: day) ?? day
            guardDays += 1
        }
        return out
    }

    /// "Thu 22"
    static func dayLabel(_ day: String) -> String {
        guard let d = PlanDay.date(day) else { return day }
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US"); f.dateFormat = "EEE d"
        return f.string(from: d)
    }

    static func tomorrow() -> String {
        PlanDay.iso.string(from: Calendar.current.date(byAdding: .day, value: 1, to: Date()) ?? Date())
    }

    /// A new time: one hour after the last one (stays within the day).
    static func later(_ time: String) -> String {
        let parts = time.split(separator: ":").compactMap { Int($0) }
        return String(format: "%02d:%02d", min(23, (parts.first ?? 9) + 1), parts.count > 1 ? parts[1] : 0)
    }

    private func build() async {
        building = true
        defer { building = false }
        var prefs = store.state?.prefs ?? PlanPrefs(perDay: perDay, days: days, reminders: [], tz: nil)
        prefs.times = times
        prefs.perDay = perDay
        prefs.days = days
        prefs.count = count
        prefs.start = start
        prefs.remind = remind
        do {
            try await store.savePrefs(prefs, session: appState.session)
            try await store.build(session: appState.session)
            if remind {
                _ = try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge])
            }
            PostReminders.shared.schedule(store.state)
            PostPrefetch.schedule()
            if let week = store.state?.week, week.endsEarly == true {
                let placed = week.placed ?? week.slots.count
                let end = week.end.map(Self.dayLabel) ?? ""
                endedEarly = "\(placed) of \(week.asked ?? count) videos fit\(end.isEmpty ? "" : " — the plan ends \(end)"). The rest would put the same product back to back."
            } else {
                onBuilt()
            }
        } catch {
            errorMessage = "Your plan couldn’t be built. Check your connection and try again."
        }
    }
}

/// START · Change: a day from today on.
struct StartDateSheet: View {
    let day: String
    let onDone: (String) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var value = Date()

    var body: some View {
        VStack(spacing: 10) {
            DatePicker("", selection: $value, in: Calendar.current.startOfDay(for: Date())..., displayedComponents: .date)
                .datePickerStyle(.graphical).labelsHidden().tint(CS.cyan)
            Button("Save") { onDone(PlanDay.iso.string(from: value)); dismiss() }
                .font(CS.inter(15, .semibold)).foregroundStyle(CS.cyan)
                .accessibilityIdentifier("startSave")
        }
        .padding(16)
        .preferredColorScheme(.dark)
        .onAppear { value = PlanDay.date(day) ?? Date() }
    }
}

struct TimePickerSheet: View {
    let time: String
    var title: String? = nil
    /// "Delete this time" (Goal v2: tapping a time lets her change it or remove it).
    var onDelete: (() -> Void)? = nil
    let onDone: (String) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var value = Date()

    var body: some View {
        VStack(spacing: 12) {
            if let title { Text(title).font(CS.inter(15, .semibold)).foregroundStyle(.white) }
            DatePicker("", selection: $value, displayedComponents: .hourAndMinute).datePickerStyle(.wheel).labelsHidden()
            Button("Save") {
                let c = Calendar.current.dateComponents([.hour, .minute], from: value)
                onDone(String(format: "%02d:%02d", c.hour ?? 9, c.minute ?? 0))
                dismiss()
            }
            .font(CS.inter(15, .semibold)).foregroundStyle(CS.cyan)
            .accessibilityIdentifier("timeSave")
            if let onDelete {
                Button("Delete this time", role: .destructive) { onDelete(); dismiss() }
                    .font(CS.inter(14, .semibold))
                    .accessibilityIdentifier("timeDelete")
            }
        }
        .padding(.top, 20)
        .preferredColorScheme(.dark)
        .onAppear {
            let parts = time.split(separator: ":").compactMap { Int($0) }
            value = Calendar.current.date(bySettingHour: parts.first ?? 9, minute: parts.count > 1 ? parts[1] : 0, second: 0, of: Date()) ?? Date()
        }
    }
}

// MARK: - Plan my week · Week (Figma 1138:2798)

struct PlanWeekView: View {
    let onNewMix: () -> Void
    let onSettings: () -> Void
    /// One navigation path for the whole Plan tab (a reminder can open a day from anywhere).
    let onOpenDay: (String) -> Void
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var store = PlanWeekStore.shared
    @ObservedObject private var thumbnails = ThumbnailStore.shared
    @State private var moving: PlanSlot?

    private var days: [String] { Array(Set((store.state?.week?.slots ?? []).map(\.date))).sorted() }

    var body: some View {
        ZStack {
            PlanStyle.background.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    HStack {
                        PlanBackButton(title: "Plan") { dismiss() }
                        Spacer()
                        Button("Settings", action: onSettings).font(CS.inter(14, .semibold)).foregroundStyle(CS.cyan)
                            .accessibilityIdentifier("weekSettings")
                    }
                    .padding(.top, 8)
                    Text("This week").font(CS.inter(28, .bold)).foregroundStyle(.white).padding(.top, 18)
                    Text(subtitle).font(CS.inter(13)).foregroundStyle(PlanStyle.muted).padding(.top, 6)
                        .accessibilityIdentifier("weekSubtitle")
                    VStack(spacing: 8) {
                        ForEach(days, id: \.self) { day in dayCard(day) }
                    }
                    .padding(.top, 14)
                    if let missing = store.state?.week?.missing, missing.videos > 0 {
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Videos for \(missing.coveredDays) of \(missing.totalDays) days")
                                    .font(CS.inter(13, .semibold)).foregroundStyle(Color(hex: 0xFFB840))
                                Text("Make \(missing.videos) more to fill \(Self.list(missing.days.compactMap { PlanDay.date($0) == nil ? nil : PlanDay.weekdayName($0) })).")
                                    .font(CS.inter(12)).foregroundStyle(PlanStyle.muted).lineLimit(2)
                            }
                            Spacer()
                            Button("New Mix ›", action: onNewMix).font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
                                .accessibilityIdentifier("weekNewMix")
                        }
                        .padding(.horizontal, 13).padding(.vertical, 10)
                        .background(Color(hex: 0x1F170D), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color(hex: 0x73521F), lineWidth: 1))
                        .padding(.top, 10)
                        .accessibilityElement(children: .contain)
                        .accessibilityIdentifier("weekMissing")
                    }
                    if let week = store.state?.week, week.endsEarly == true {
                        Text("Plan ends early: \(week.placed ?? week.slots.count) of \(week.asked ?? 0) videos fit without the same product back to back.")
                            .font(CS.inter(12)).foregroundStyle(Color(hex: 0xFFB840)).padding(.top, 10)
                            .accessibilityIdentifier("weekEndsEarly")
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 24)
            }
            .refreshable { await store.refresh(session: appState.session) }
        }
        .toolbar(.hidden, for: .navigationBar)
        .preferredColorScheme(.dark)
        .sheet(item: $moving) { slot in
            MoveSlotSheet(slot: slot).environmentObject(appState)
                .presentationDetents([.height(420)])
                .presentationBackground(Color(hex: 0x0D121F))
        }
        .task { await store.refresh(session: appState.session) }
    }

    struct DayRoute: Identifiable, Hashable { let id: String }

    /// "Fri", "Fri and Sat", "Thu, Fri and Sat"
    static func list(_ names: [String]) -> String {
        guard let last = names.last else { return "the week" }
        return names.count == 1 ? last : names.dropLast().joined(separator: ", ") + " and " + last
    }

    private var subtitle: String {
        let slots = store.state?.week?.slots ?? []
        let products = Set(slots.compactMap(\.productID)).count            // a video without a product is not a product
        let perDay = store.state?.prefs.postingTimes.count ?? 1
        return "\(slots.count) video\(slots.count == 1 ? "" : "s") · \(products) product\(products == 1 ? "" : "s") · \(perDay) per day"
    }

    private func dayCard(_ day: String) -> some View {
        let slots = (store.state?.week?.slots ?? []).filter { $0.date == day }
        let isToday = day == store.state?.today
        let allPosted = !slots.isEmpty && slots.allSatisfy { $0.postedAt != nil }
        return Button { onOpenDay(day) } label: {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Text(PlanDay.short(day)).font(CS.inter(14, .bold)).foregroundStyle(.white)
                    if allPosted {
                        Text("POSTED").font(CS.inter(9, .bold)).foregroundStyle(Color(hex: 0x33D98C))
                            .frame(width: 58, height: 18).background(Color(hex: 0x143324), in: Capsule())
                    } else if isToday {
                        Text("TODAY").font(CS.inter(9, .bold)).foregroundStyle(.white)
                            .frame(width: 51, height: 18).background(CS.brandFull, in: Capsule())
                    }
                    Spacer()
                }
                if slots.count > 3 {                       // Goal v2: up to 8 times a day -> the day scrolls sideways
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(alignment: .top, spacing: 8) {
                            ForEach(slots) { slot in slotCell(slot, highlight: isToday || !allPosted && slot.postedAt == nil).frame(width: 96) }
                        }
                    }
                } else {
                    HStack(alignment: .top, spacing: 8) {
                        ForEach(slots) { slot in slotCell(slot, highlight: isToday || !allPosted && slot.postedAt == nil) }
                        if slots.count < 2 { Spacer(minLength: 0) }
                    }
                }
            }
            .padding(13)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(PlanStyle.card, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(isToday ? Color(hex: 0x19E5FF) : PlanStyle.border, lineWidth: isToday ? 1.5 : 1))
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("weekDay_\(day)")
    }

    private func slotCell(_ slot: PlanSlot, highlight: Bool) -> some View {
        let project = appState.projects.first { $0.projectID == slot.projectID }
        return HStack(alignment: .top, spacing: 8) {
            Color(hex: 0x1A2236).frame(width: 36, height: 58)
                .overlay { if let image = thumbnails.image(for: slot.projectID) { Image(uiImage: image).resizable().scaledToFill() } }
                .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 5) {
                    Image(store.dot(for: slot)).resizable().frame(width: 7, height: 7)
                    Text(slot.brand ?? "No product").font(CS.inter(11, .semibold)).foregroundStyle(.white).lineLimit(1)
                }
                Text(slot.product ?? slot.mixCode ?? "Video").font(CS.inter(11)).foregroundStyle(PlanStyle.muted).lineLimit(2)
                Text(PlanDay.clock(slot.time)).font(CS.inter(11, .semibold))
                    .foregroundStyle(slot.postedAt == nil && highlight ? CS.cyan : PlanStyle.muted)
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
        .contextMenu {
            Button("Move to another day or time") { moving = slot }
            Button("Remove from this week", role: .destructive) { Task { try? await store.remove(slot, session: appState.session) } }
        }
        .task { if let project { await thumbnails.load(project: project, session: appState.session) } }
        .accessibilityIdentifier("weekSlot")
    }
}

/// Move a planned video to another day / time (the server keeps it where she put it when the week is built again).
struct MoveSlotSheet: View {
    let slot: PlanSlot
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var store = PlanWeekStore.shared
    @State private var day = Date()
    @State private var time = Date()
    @State private var errorMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Move \(slot.product ?? "this video")").font(CS.inter(20, .bold)).foregroundStyle(.white).padding(.top, 24)
            DatePicker("Day", selection: $day, in: Date()..., displayedComponents: .date).accessibilityIdentifier("moveDay")
            DatePicker("Time", selection: $time, displayedComponents: .hourAndMinute).accessibilityIdentifier("moveTime")
            Button {
                Task {
                    let c = Calendar.current.dateComponents([.hour, .minute], from: time)
                    do {
                        try await store.move(slot, to: PlanDay.iso.string(from: day),
                                             time: String(format: "%02d:%02d", c.hour ?? 9, c.minute ?? 0), session: appState.session)
                        dismiss()
                    } catch { errorMessage = "Couldn’t move it. Please try again." }
                }
            } label: {
                Text("Move").font(CS.sf(15, .semibold)).foregroundStyle(.white).frame(maxWidth: .infinity).frame(height: 50)
                    .background(Capsule().fill(PlanStyle.cta))
            }
            .accessibilityIdentifier("moveSave")
            if let errorMessage { Text(errorMessage).font(CS.inter(12)).foregroundStyle(CS.error) }
            Spacer()
        }
        .padding(.horizontal, 23)
        .preferredColorScheme(.dark)
        .onAppear {
            day = PlanDay.date(slot.date) ?? Date()
            let parts = slot.time.split(separator: ":").compactMap { Int($0) }
            time = Calendar.current.date(bySettingHour: parts.first ?? 9, minute: parts.count > 1 ? parts[1] : 0, second: 0, of: Date()) ?? Date()
        }
    }
}

// MARK: - Plan my week · Day (Figma 1138:2872)

struct PlanDayView: View {
    let day: String
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var store = PlanWeekStore.shared
    @ObservedObject private var exporter = PostExporter.shared
    @ObservedObject private var thumbnails = ThumbnailStore.shared
    @State private var current: String?
    @State private var editing = false
    @State private var draftCaption = ""
    @State private var shareItems: [Any]?
    @State private var notice: String?
    @State private var busy = false

    private var slots: [PlanSlot] { (store.state?.week?.slots ?? []).filter { $0.date == day } }
    private var focus: PlanSlot? {
        slots.first { $0.projectID == current } ?? slots.first { $0.postedAt == nil } ?? slots.first
    }

    var body: some View {
        ZStack {
            PlanStyle.background.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    PlanBackButton(title: "This week") { dismiss() }.padding(.top, 8)
                    Text(day == store.state?.today ? "Today · \(PlanDay.short(day))" : PlanDay.short(day))
                        .font(CS.inter(28, .bold)).foregroundStyle(.white).padding(.top, 18)
                        .accessibilityIdentifier("dayTitle")
                    let left = slots.filter { $0.postedAt == nil }.count
                    Text(left == 0 ? "Everything is posted 🎉" : "\(left) video\(left == 1 ? "" : "s") to post")
                        .font(CS.inter(13)).foregroundStyle(PlanStyle.muted).padding(.top, 6)
                    if let slot = focus {
                        mainCard(slot).padding(.top, 14)
                        actions(slot).padding(.top, 14)
                        ForEach(slots.filter { $0.projectID != slot.projectID }) { other in
                            Text("\(Int(other.time.prefix(2)) ?? 0 >= 17 ? "Tonight" : "Later") · \(PlanDay.clock(other.time))")
                                .font(CS.inter(14, .semibold)).foregroundStyle(.white).padding(.top, 22)
                            Button { current = other.projectID } label: { otherCard(other) }.buttonStyle(.plain)
                                .padding(.top, 10)
                                .accessibilityIdentifier("dayOther")
                        }
                    }
                    if let notice {
                        Text(notice).font(CS.inter(12, .semibold)).foregroundStyle(CS.cyan).padding(.top, 14)
                            .accessibilityIdentifier("dayNotice")
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 24)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .preferredColorScheme(.dark)
        .sheet(isPresented: Binding(get: { shareItems != nil }, set: { if !$0 { shareItems = nil } })) {
            ShareSheet(items: shareItems ?? []).ignoresSafeArea()
        }
        // Simulator probe: one "Time to post" reminder a few seconds after CutSell goes to the background.
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)) { _ in
            if ProcessInfo.processInfo.arguments.contains("-CutSellPlanTestReminder"), let slot = focus {
                PostReminders.shared.test(slot, seconds: 4)
            }
        }
        .alert("Edit caption", isPresented: $editing) {
            TextField("Caption", text: $draftCaption, axis: .vertical).accessibilityIdentifier("captionField")
            Button("Save") { Task { await saveCaption() } }
            Button("Cancel", role: .cancel) {}
        }
    }

    private func mainCard(_ slot: PlanSlot) -> some View {
        HStack(alignment: .top, spacing: 14) {
            Color(hex: 0x1A2236).frame(width: 124, height: 222)
                .overlay { if let image = thumbnails.image(for: slot.projectID) { Image(uiImage: image).resizable().scaledToFill() } }
                .overlay(alignment: .topLeading) {
                    if slot.mixCode != nil {
                        Text("MIX").font(CS.inter(9, .bold)).foregroundStyle(.white).frame(width: 34, height: 18)
                            .background(CS.brandFull, in: Capsule()).padding(8)
                    }
                }
                .overlay { if let p = exporter.progress[slot.projectID] { ProgressView(value: p).tint(CS.cyan).padding(12) } }
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 6) {
                    Image(store.dot(for: slot)).resizable().frame(width: 8, height: 8)
                    Text(slot.brand ?? "No product").font(CS.inter(13, .semibold)).foregroundStyle(.white).lineLimit(1)
                }
                Text(slot.product ?? "Video").font(CS.inter(15, .bold)).foregroundStyle(.white).lineLimit(2).padding(.top, 3)
                    .accessibilityIdentifier("dayProduct")
                Text(PlanDay.clock(slot.time) + (reminderOn(slot) ? " · reminder on" : ""))
                    .font(CS.inter(12, .semibold)).foregroundStyle(CS.cyan).padding(.top, 6)
                Text([slot.mixCode, ProductDetailView.length(slot.durationSec)].compactMap { $0 }.joined(separator: " · "))
                    .font(CS.inter(11)).foregroundStyle(PlanStyle.muted).padding(.top, 6)
                VStack(alignment: .leading, spacing: 6) {
                    Text(slot.captionSource == "user" ? "✦ YOUR CAPTION" : "✦ SUGGESTED CAPTION")
                        .font(CS.inter(9, .bold)).foregroundStyle(CS.cyan)
                    Text(slot.caption ?? "Writing your caption…").font(CS.inter(11)).foregroundStyle(.white)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("dayCaption")
                    Spacer(minLength: 0)
                    HStack {
                        Button("Edit") { draftCaption = slot.caption ?? ""; editing = true }
                            .accessibilityIdentifier("captionEdit")
                        Spacer()
                        Button("Copy") {
                            UIPasteboard.general.string = slot.caption ?? ""
                            notice = "Caption copied"
                        }
                        .accessibilityIdentifier("captionCopy")
                    }
                    .font(CS.inter(11, .semibold)).foregroundStyle(CS.cyan)
                }
                .padding(9)
                .frame(maxWidth: .infinity, minHeight: 132, alignment: .topLeading)
                .background(Color(hex: 0x0A0F1A), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(PlanStyle.border, lineWidth: 1))
                .padding(.top, 12)
            }
        }
        .padding(13)
        .background(PlanStyle.card, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(PlanStyle.border, lineWidth: 1))
        .task {
            if let project = appState.projects.first(where: { $0.projectID == slot.projectID }) {
                await thumbnails.load(project: project, session: appState.session)
            }
        }
    }

    private func actions(_ slot: PlanSlot) -> some View {
        VStack(spacing: 10) {
            Button { Task { await share(slot) } } label: {
                Text(exporter.progress[slot.projectID].map { "Preparing… \(Int($0 * 100)) %" } ?? "Share to TikTok")
                    .font(CS.sf(15, .semibold)).foregroundStyle(.white).frame(maxWidth: .infinity).frame(height: 50)
                    .background(Capsule().fill(PlanStyle.cta))
                    .shadow(color: Color(hex: 0x1969FF).opacity(0.25), radius: 9, y: 6)
            }
            .disabled(busy)
            .accessibilityIdentifier("dayShare")
            HStack(spacing: 8) {
                Button { Task { await save(slot) } } label: { outline("Save to Photos") }.disabled(busy)
                    .accessibilityIdentifier("daySave")
                Button { Task { await posted(slot) } } label: { outline(slot.postedAt == nil ? "✓  Mark as posted" : "✓  Posted") }
                    .disabled(slot.postedAt != nil)
                    .accessibilityIdentifier("dayPosted")
            }
        }
    }

    private func outline(_ title: String) -> some View {
        Text(title).font(CS.inter(13, .semibold)).foregroundStyle(.white).frame(maxWidth: .infinity).frame(height: 46)
            .overlay(Capsule().stroke(Color(hex: 0x596685), lineWidth: 1))
    }

    private func otherCard(_ slot: PlanSlot) -> some View {
        HStack(spacing: 12) {
            Color(hex: 0x1A2236).frame(width: 40, height: 60)
                .overlay { if let image = thumbnails.image(for: slot.projectID) { Image(uiImage: image).resizable().scaledToFill() } }
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Image(store.dot(for: slot)).resizable().frame(width: 8, height: 8)
                    Text(slot.brand ?? "No product").font(CS.inter(12, .semibold)).foregroundStyle(.white)
                }
                Text(slot.product ?? "Video").font(CS.inter(14, .bold)).foregroundStyle(.white).lineLimit(1)
                Text(slot.postedAt != nil ? "Posted" : slot.caption == nil ? "Writing caption…" : "Caption ready")
                    .font(CS.inter(11)).foregroundStyle(PlanStyle.muted)
            }
            Spacer()
            Text("›").font(CS.inter(20, .medium)).foregroundStyle(PlanStyle.muted)
        }
        .padding(.horizontal, 13).padding(.vertical, 11)
        .background(PlanStyle.card, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(PlanStyle.border, lineWidth: 1))
    }

    private func reminderOn(_ slot: PlanSlot) -> Bool {
        guard let prefs = store.state?.prefs else { return false }
        if prefs.times != nil { return prefs.remind ?? true }
        return prefs.reminders.contains { $0.on && $0.time == slot.time }
    }

    /// The video file (exported at night, or now with its %), the caption copied: the iPhone's share sheet (TikTok…).
    private func share(_ slot: PlanSlot) async {
        busy = true
        defer { busy = false }
        UIPasteboard.general.string = slot.caption ?? ""
        do {
            let file = try await exporter.file(for: slot.projectID, session: appState.session)
            notice = "Caption copied · paste it in TikTok"
            shareItems = [file]
            CutSellDiagnostics.log("post_share_opened", ["prefetched": PostExporter.shared.localFile(slot.projectID) != nil ? "yes" : "no"])
        } catch {
            notice = "The video couldn’t be prepared. Check your connection and try again."
        }
    }

    private func save(_ slot: PlanSlot) async {
        busy = true
        defer { busy = false }
        do {
            let file = try await exporter.file(for: slot.projectID, session: appState.session)
            let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
            guard status == .authorized || status == .limited else { notice = "Allow CutSell to add videos to Photos."; return }
            try await PHPhotoLibrary.shared().performChanges { PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: file) }
            notice = "Saved to Photos"
        } catch {
            notice = "Couldn’t save the video. Please try again."
        }
    }

    private func posted(_ slot: PlanSlot) async {
        current = slot.projectID                    // stays on screen, marked Posted
        do {
            try await store.markPosted(slot, session: appState.session)
            notice = "Marked as posted"
        } catch { notice = "Couldn’t save that. Please try again." }
    }

    private func saveCaption() async {
        guard let slot = focus else { return }
        let text = draftCaption.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        do {
            try await store.saveCaption(slot, caption: text, session: appState.session)
            notice = "Caption saved"
        } catch { notice = "Couldn’t save the caption. Please try again." }
    }
}

/// The iPhone's share sheet (TikTok, Instagram, Messages, Files…).
struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
