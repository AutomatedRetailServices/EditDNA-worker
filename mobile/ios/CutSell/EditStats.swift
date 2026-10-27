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
        if minutes < 60 { return "\(Int(minutes)) min" }          // "20 min saved", never "0.3 h"
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
    @State private var period: StatsPeriod = .lifetime
    /// 0 → 1 when the page (or another period) appears: numbers count up, the ring fills, the bars grow.
    @State private var progress: Double = 0

    private var all: [Project] { EditStats.edited(appState.projects) }
    private var list: [Project] { all.filter { period.contains($0) } }
    private var minutesSaved: Double { Double(list.count) * EditStats.minutesSavedPerVideo }

    var body: some View {
        YouScreen(title: "Your results", subtitle: "Everything CutSell edited for you", back: back) {
            VStack(spacing: 14) {
                hero
                HStack(spacing: 10) {
                    tile(icon: "film.stack", value: Double(list.count), text: { "\(Int($0))" }, label: "Videos edited", id: "statsVideos")
                    tile(icon: "clock.arrow.circlepath", value: stats.totalSeconds(list),
                         text: { EditStats.duration($0) }, label: stats.measuring ? "Adding up…" : "Video time", id: "statsTime")
                }
                weeksChart
                milestone
                Text(L10n.tr("Hours saved = \(Int(EditStats.minutesSavedPerVideo)) min of editing by hand per video. A workday is 8 hours."))
                    .font(CS.inter(11, .medium)).foregroundStyle(CS.wiSecondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityIdentifier("statsRule")
            }
            .padding(.top, 18)
        }
        .task {
            try? await appState.refreshProjects()
            await stats.measure(appState.projects, session: appState.session)
        }
        .onAppear { replay() }
        .onChange(of: period) { _, _ in replay() }
    }

    private func replay() {
        progress = 0
        withAnimation(.spring(response: 1.1, dampingFraction: 0.85)) { progress = 1 }
    }

    // MARK: Hero: the period switch, the hours saved counting up, a 24-hour ring filling with them.

    private var hero: some View {
        let white = Color(fixed: 0xFFFFFF)
        return VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 4) {
                ForEach(StatsPeriod.allCases.reversed()) { p in
                    Button { withAnimation(.easeOut(duration: 0.2)) { period = p } } label: {
                        Text(L10n.tr(p.rawValue)).font(CS.inter(13, .semibold))
                            .foregroundStyle(period == p ? CS.accent : white)
                            .frame(maxWidth: .infinity).frame(height: 32)
                            .background(period == p ? white : .clear, in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("statsPeriod_\(p)")
                }
            }
            .padding(3)
            .background(white.opacity(0.18), in: Capsule())

            HStack(alignment: .center, spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(L10n.tr("TIME SAVED")).font(CS.inter(11, .bold)).tracking(1).foregroundStyle(white.opacity(0.8))
                    CountingText(value: minutesSaved * progress) { EditStats.hours($0) }
                        .font(CS.inter(44, .heavy)).foregroundStyle(white)
                        .lineLimit(1).minimumScaleFactor(0.6)
                        .accessibilityIdentifier("statsSaved")
                    Text(L10n.tr(EditStats.workdays(minutesSaved))).font(CS.inter(13, .semibold))
                        .foregroundStyle(white.opacity(0.9)).fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                dayRing
            }
        }
        .padding(18)
        .background {
            ZStack {
                LinearGradient(colors: [Color(fixed: 0x1769FF), Color(fixed: 0x5B3BFF), Color(fixed: 0xB13CF2)],
                               startPoint: .topLeading, endPoint: .bottomTrailing)
                Circle().fill(Color(fixed: 0x00E5FF, opacity: 0.35)).frame(width: 180).blur(radius: 40).offset(x: 120, y: -60)
                Circle().fill(Color(fixed: 0xFF1F3D, opacity: 0.25)).frame(width: 160).blur(radius: 50).offset(x: -110, y: 80)
            }
            .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
        }
        .shadow(color: Color(fixed: 0x1769FF, opacity: 0.35), radius: 16, y: 8)
    }

    /// A 24-hour day; the hours saved fill it (more than a day: full, "×N days").
    private var dayRing: some View {
        let white = Color(fixed: 0xFFFFFF)
        let days = minutesSaved / (24 * 60)
        let fill = min(1, days) * progress
        return ZStack {
            Circle().stroke(white.opacity(0.22), lineWidth: 10)
            Circle().trim(from: 0, to: fill)
                .stroke(AngularGradient(colors: [Color(fixed: 0x00E5FF), white, Color(fixed: 0xFFD166)], center: .center),
                        style: StrokeStyle(lineWidth: 10, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .shadow(color: Color(fixed: 0x00E5FF, opacity: 0.8), radius: 6)
            VStack(spacing: 0) {
                Text(verbatim: days >= 1 ? String(format: days < 10 ? "%.1f" : "%.0f", days) : "\(Int((minutesSaved / (24 * 60) * 100).rounded()))%")
                    .font(CS.inter(17, .heavy)).foregroundStyle(white)
                Text(L10n.tr(days >= 1 ? "days" : "of 24 h")).font(CS.inter(10, .semibold)).foregroundStyle(white.opacity(0.85))
            }
        }
        .frame(width: 92, height: 92)
        .accessibilityIdentifier("statsDayRing")
    }

    private func tile(icon: String, value: Double, text: @escaping (Double) -> String, label: String, id: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: icon).font(.system(size: 16, weight: .semibold)).foregroundStyle(CS.accent)
                .frame(width: 34, height: 34).background(CS.accentSoft, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            CountingText(value: value * progress, format: text).font(CS.inter(26, .heavy)).foregroundStyle(CS.wiInk)
                .lineLimit(1).minimumScaleFactor(0.6)
                .accessibilityIdentifier(id)
            Text(L10n.tr(label)).font(CS.inter(12, .semibold)).foregroundStyle(CS.wiSecondary)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .wiCard(radius: 18)
    }

    // MARK: Last 8 weeks: videos edited per week, bars growing, this week in blue.

    private var weeks: [(label: String, count: Int)] {
        var cal = Calendar(identifier: .gregorian); cal.firstWeekday = 2
        let now = Date()
        let start = cal.dateInterval(of: .weekOfYear, for: now)?.start ?? now
        let f = DateFormatter(); f.locale = L10n.locale; f.dateFormat = "d MMM"
        return (0..<8).reversed().map { back in
            let week = cal.date(byAdding: .weekOfYear, value: -back, to: start) ?? start
            let count = all.filter {
                guard let raw = $0.createdAt ?? $0.updatedAt, let d = EditStats.date(raw) else { return false }
                return cal.isDate(d, equalTo: week, toGranularity: .weekOfYear)
            }.count
            return (back == 0 ? L10n.tr("This week") : f.string(from: week), count)
        }
    }

    private var weeksChart: some View {
        let data = weeks
        let top = max(1, data.map(\.count).max() ?? 1)
        return VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(L10n.tr("Last 8 weeks")).font(CS.inter(15, .bold)).foregroundStyle(CS.wiInk)
                Spacer()
                Text(L10n.tr("\(data.map(\.count).reduce(0, +)) videos · \(EditStats.hours(Double(data.map(\.count).reduce(0, +)) * EditStats.minutesSavedPerVideo)) saved"))
                    .font(CS.inter(12, .semibold)).foregroundStyle(CS.accent)
            }
            HStack(alignment: .bottom, spacing: 8) {
                ForEach(Array(data.enumerated()), id: \.offset) { index, week in
                    let current = index == data.count - 1
                    VStack(spacing: 6) {
                        Text(verbatim: week.count > 0 ? "\(week.count)" : "").font(CS.inter(10, .bold))
                            .foregroundStyle(current ? CS.accent : CS.wiSecondary)
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .fill(current ? AnyShapeStyle(LinearGradient(colors: [Color(fixed: 0x5B3BFF), CS.accent], startPoint: .top, endPoint: .bottom))
                                          : AnyShapeStyle(CS.accentSoft))
                            .frame(height: max(6, 96 * CGFloat(week.count) / CGFloat(top) * progress))
                    }
                    .frame(maxWidth: .infinity)
                }
            }
            .frame(height: 118, alignment: .bottom)
            HStack {
                Text(verbatim: data.first?.label ?? "").font(CS.inter(10, .medium)).foregroundStyle(CS.wiSecondary)
                Spacer()
                Text(verbatim: data.last?.label ?? "").font(CS.inter(10, .semibold)).foregroundStyle(CS.accent)
            }
        }
        .padding(16)
        .wiCard(radius: 18)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("statsWeeks")
    }

    // MARK: The next goal (10 · 25 · 50 · 100 · 250 · 500 · 1000 videos).

    private var milestone: some View {
        let goals = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
        let done = all.count
        let goal = goals.first { $0 > done } ?? (done + 1000)
        let previous = goals.last { $0 <= done } ?? 0
        let fraction = Double(done - previous) / Double(max(1, goal - previous))
        return HStack(spacing: 14) {
            ZStack {
                Circle().fill(LinearGradient(colors: [Color(fixed: 0xFFD166), Color(fixed: 0xFF8A3D)], startPoint: .top, endPoint: .bottom))
                Image(systemName: "trophy.fill").font(.system(size: 20, weight: .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
            }
            .frame(width: 48, height: 48)
            .shadow(color: Color(fixed: 0xFF8A3D, opacity: 0.4), radius: 8, y: 3)
            VStack(alignment: .leading, spacing: 6) {
                Text(L10n.tr(previous > 0 ? "\(previous) videos reached!" : "Your first goal")).font(CS.inter(14, .bold)).foregroundStyle(CS.wiInk)
                GeometryReader { box in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color(fixed: 0xF1E7D6))
                        Capsule().fill(LinearGradient(colors: [Color(fixed: 0xFFD166), Color(fixed: 0xFF8A3D)], startPoint: .leading, endPoint: .trailing))
                            .frame(width: max(8, box.size.width * fraction * progress))
                    }
                }
                .frame(height: 8)
                Text(L10n.tr("\(goal - done) more to \(goal) videos")).font(CS.inter(12, .medium)).foregroundStyle(CS.wiSecondary)
                    .accessibilityIdentifier("statsGoal")
            }
        }
        .padding(16)
        .wiCard(radius: 18)
    }
}

/// A number that counts up (animatable), shown with its own format.
struct CountingText: View, Animatable {
    var value: Double
    let format: (Double) -> String
    var animatableData: Double { get { value } set { value = newValue } }
    var body: some View { Text(verbatim: format(value)).monospacedDigit() }
}
