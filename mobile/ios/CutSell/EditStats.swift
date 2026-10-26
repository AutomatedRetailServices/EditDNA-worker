import SwiftUI

/// Profile › Your results (PO 26 Oct): how many videos CutSell edited, their total length, and the hours saved
/// (20 minutes of hand editing per video). The length of each finished cut is measured on this iPhone from its draft
/// (the draft the editor already loads), kept here, and grows as new cuts finish -- no server change.
@MainActor
final class EditStats: ObservableObject {
    static let shared = EditStats()
    /// Minutes of hand editing one video saves (PO 26 Oct: 20).
    static let minutesSavedPerVideo = 20.0

    @Published private(set) var seconds: [String: Double] = UserDefaults.standard.dictionary(forKey: key) as? [String: Double] ?? [:]
    @Published private(set) var measuring = false
    private static let key = "cutsell.stats.seconds"

    /// A finished cut (Ready, Drafts or Posted): every project the server finished, never a Mix container or one still processing.
    static func edited(_ projects: [Project]) -> [Project] {
        projects.filter { !Project.processingStates.contains($0.state) && $0.state != Project.mixContainerState
            && $0.state != Project.uploadNotFinishedState && $0.state != "failed" }
    }

    func totalSeconds(_ projects: [Project]) -> Double {
        Self.edited(projects).reduce(0) { $0 + (seconds[$1.projectID] ?? 0) }
    }

    /// Measures the cuts not measured yet (cached draft first, else the server's draft), one after another.
    func measure(_ projects: [Project], session: CutSellSession?) async {
        guard !measuring else { return }
        let missing = Self.edited(projects).filter { seconds[$0.projectID] == nil }
        guard !missing.isEmpty else { return }
        measuring = true
        defer { measuring = false }
        for project in missing {
            var snapshot = DraftSnapshotCache.load(projectID: project.projectID)
            if snapshot == nil, let session {
                snapshot = try? await APIClient.shared.request("/v1/projects/\(project.projectID)/draft",
                                                               query: [URLQueryItem(name: "user_id", value: session.userID)])
                if let snapshot { DraftSnapshotCache.save(snapshot) }
            }
            guard let snapshot else { continue }
            seconds[project.projectID] = Self.length(of: snapshot)
            UserDefaults.standard.set(seconds, forKey: Self.key)
        }
    }

    /// The final cut's length: each kept clip (end − start) at its speed.
    static func length(of snapshot: DraftSnapshot) -> Double {
        let clips = snapshot.draft.objectValue?["selected"]?.arrayValue?.compactMap(\.objectValue) ?? []
        return clips.reduce(0.0) { sum, clip in
            let start = clip["start"]?.doubleValue ?? 0, end = clip["end"]?.doubleValue ?? start
            return sum + max(0, end - start) / ClipSpeed.of(clip)
        }
    }

    #if DEBUG
    /// Design screenshots only: sample lengths (no drafts in the simulator).
    func previewSeed(_ projects: [Project]) {
        for (i, p) in Self.edited(projects).enumerated() where seconds[p.projectID] == nil { seconds[p.projectID] = Double(32 + (i * 7) % 40) }
    }
    #endif

    /// "2 h 05 min", "12 min", "45 s"
    static func duration(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        if s < 60 { return "\(s) s" }
        let minutes = s / 60
        if minutes < 60 { return "\(minutes) min" }
        return String(format: "%d h %02d min", minutes / 60, minutes % 60)
    }

    /// "That's 1.5 workdays (8 h)" · "That's 40 min of your day"
    static func workdays(_ minutes: Double) -> String {
        if minutes < 8 * 60 { return minutes < 60 ? "That's \(Int(minutes)) min of your day" : "That's \(hours(minutes)) of an 8-hour workday" }
        let days = minutes / (8 * 60)
        let text = days < 10 ? String(format: "%.1f", days) : "\(Int(days.rounded()))"
        return "That's \(text) workdays of 8 hours"
    }

    /// The server's dates ("2026-10-12T22:28:05.952628+00:00").
    nonisolated static func date(_ raw: String) -> Date? {
        let f = DateFormatter(); f.calendar = Calendar(identifier: .gregorian); f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"
        return f.date(from: String(raw.prefix(10)))
    }

    /// "6.7 h"
    static func hours(_ minutes: Double) -> String {
        let h = minutes / 60
        return h < 10 ? String(format: "%.1f h", h) : "\(Int(h.rounded())) h"
    }
}

/// The periods of Your results: this week (since Monday), this month, lifetime.
enum StatsPeriod: String, CaseIterable, Identifiable {
    case week = "This week", month = "This month", lifetime = "Lifetime"
    var id: String { rawValue }

    func contains(_ project: Project, now: Date = Date()) -> Bool {
        guard self != .lifetime else { return true }
        guard let raw = project.createdAt ?? project.updatedAt, let date = EditStats.date(raw) else { return false }
        var cal = Calendar(identifier: .gregorian)
        cal.firstWeekday = 2                                   // Monday
        return cal.isDate(date, equalTo: now, toGranularity: self == .week ? .weekOfYear : .month)
    }
}

struct EditStatsPage: View {
    let back: () -> Void
    @EnvironmentObject private var appState: AppState
    @ObservedObject private var stats = EditStats.shared

    private func videos(_ period: StatsPeriod) -> [Project] { EditStats.edited(appState.projects).filter { period.contains($0) } }

    var body: some View {
        YouScreen(title: "Your results", subtitle: "Everything CutSell edited for you", back: back) {
            VStack(spacing: 12) {
                card(.lifetime, blue: true)
                card(.month)
                card(.week)
                Text(L10n.tr("Hours saved = \(Int(EditStats.minutesSavedPerVideo)) min of editing by hand per video. A workday is 8 hours."))
                    .font(CS.inter(12, .medium)).foregroundStyle(CS.wiSecondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier("statsRule")
            }
            .padding(.top, 20)
        }
        .task {
            try? await appState.refreshProjects()
            await stats.measure(appState.projects, session: appState.session)
        }
    }

    /// One period: hours saved (and the 8-hour workdays they make), videos edited, video time.
    private func card(_ period: StatsPeriod, blue: Bool = false) -> some View {
        let list = videos(period)
        let minutes = Double(list.count) * EditStats.minutesSavedPerVideo
        let white = Color(fixed: 0xFFFFFF)
        let ink = blue ? white : CS.wiInk
        let soft = blue ? white.opacity(0.85) : CS.wiSecondary
        return VStack(alignment: .leading, spacing: 10) {
            Text(L10n.tr(period.rawValue).uppercased()).font(CS.inter(11, .bold)).tracking(0.8).foregroundStyle(soft)
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(verbatim: EditStats.hours(minutes)).font(CS.inter(blue ? 40 : 30, .heavy)).foregroundStyle(ink)
                    .contentTransition(.numericText())
                    .accessibilityIdentifier("statsSaved_\(period)")
                Text(L10n.tr("saved")).font(CS.inter(15, .semibold)).foregroundStyle(ink)
            }
            Text(L10n.tr(EditStats.workdays(minutes))).font(CS.inter(13, .semibold)).foregroundStyle(soft)
            Rectangle().fill(blue ? white.opacity(0.25) : CS.wiBorder).frame(height: 1)
            HStack {
                stat("\(list.count)", "Videos edited", ink: ink, soft: soft, id: "statsVideos_\(period)")
                Spacer()
                stat(EditStats.duration(stats.totalSeconds(list)), stats.measuring ? "Adding up…" : "Video time",
                     ink: ink, soft: soft, id: "statsTime_\(period)")
                Spacer()
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(blue ? CS.accent : CS.wiCard, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(blue ? .clear : CS.wiBorder, lineWidth: 1))
        .animation(.easeOut, value: list.count)
    }

    private func stat(_ value: String, _ label: String, ink: Color, soft: Color, id: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(verbatim: value).font(CS.inter(20, .bold)).foregroundStyle(ink).accessibilityIdentifier(id)
            Text(L10n.tr(label)).font(CS.inter(12, .medium)).foregroundStyle(soft)
        }
    }
}
