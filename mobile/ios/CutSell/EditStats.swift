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

    /// "6.7 h"
    static func hours(_ minutes: Double) -> String {
        let h = minutes / 60
        return h < 10 ? String(format: "%.1f h", h) : "\(Int(h.rounded())) h"
    }
}

struct EditStatsPage: View {
    let back: () -> Void
    @EnvironmentObject private var appState: AppState
    @ObservedObject private var stats = EditStats.shared

    private var edited: [Project] { EditStats.edited(appState.projects) }
    private var thisMonth: [Project] {
        let month = String(ISO8601DateFormatter().string(from: Date()).prefix(7))
        return edited.filter { ($0.createdAt ?? $0.updatedAt ?? "").hasPrefix(month) }
    }

    var body: some View {
        YouScreen(title: "Your results", subtitle: "Everything CutSell edited for you", back: back) {
            VStack(spacing: 10) {
                big(value: EditStats.hours(Double(edited.count) * EditStats.minutesSavedPerVideo), label: "Hours saved",
                    detail: "\(Int(EditStats.minutesSavedPerVideo)) min of editing by hand per video", id: "statsSaved", blue: true)
                HStack(spacing: 10) {
                    big(value: "\(edited.count)", label: "Videos edited", detail: nil, id: "statsVideos")
                    big(value: EditStats.duration(stats.totalSeconds(appState.projects)), label: "Video time",
                        detail: stats.measuring ? "Adding up…" : nil, id: "statsTime")
                }
            }
            .padding(.top, 20)
            YouSection(label: "THIS MONTH") {
                YouRow(title: "Videos edited", value: "\(thisMonth.count)", chevron: false, divider: true, id: "statsMonthVideos")
                YouRow(title: "Video time", value: EditStats.duration(stats.totalSeconds(thisMonth)), chevron: false,
                       divider: true, id: "statsMonthTime")
                YouRow(title: "Hours saved", value: EditStats.hours(Double(thisMonth.count) * EditStats.minutesSavedPerVideo),
                       chevron: false, divider: false, id: "statsMonthSaved")
            }
        }
        .task {
            try? await appState.refreshProjects()
            await stats.measure(appState.projects, session: appState.session)
        }
    }

    private func big(value: String, label: String, detail: String?, id: String, blue: Bool = false) -> some View {
        let white = Color(fixed: 0xFFFFFF)
        return VStack(alignment: .leading, spacing: 4) {
            Text(verbatim: value).font(CS.inter(blue ? 40 : 30, .heavy)).foregroundStyle(blue ? white : CS.wiInk)
                .contentTransition(.numericText()).lineLimit(1).minimumScaleFactor(0.6)
                .accessibilityIdentifier(id)
            Text(L10n.tr(label)).font(CS.inter(14, .semibold)).foregroundStyle(blue ? white : CS.wiInk)
            if let detail {
                Text(L10n.tr(detail)).font(CS.inter(12, .medium)).foregroundStyle(blue ? white.opacity(0.85) : CS.wiSecondary)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            if blue { RoundedRectangle(cornerRadius: 20, style: .continuous).fill(CS.accent) }
            else { RoundedRectangle(cornerRadius: 20, style: .continuous).fill(CS.wiCard) }
        }
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(blue ? .clear : CS.wiBorder, lineWidth: 1))
        .animation(.easeOut, value: value)
    }
}
