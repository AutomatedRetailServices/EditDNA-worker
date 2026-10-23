import SwiftUI
import UIKit

/// Build 24: every cut being made, in order, for the Edits "Processing" filter (Figma 1134:2683)
/// and the Bulk "being edited" sheet (Figma 1134:4635). A cut is here from the moment its videos
/// start uploading until the server says it is ready (then it moves to Ready at once, with the
/// "<name> is ready · View" notice) -- whether or not the Processing screen is open, and after
/// CutSell is reopened (server projects still processing are listed too).
@MainActor
final class ProcessingQueue: ObservableObject {
    static let shared = ProcessingQueue()

    enum State: Equatable {
        case queued
        case uploading(Int)
        case processing(Int)
        case failed(String)
        /// 23 Oct: a Bulk video already done; it stays in its group until the whole group is done.
        case ready
    }

    struct Item: Identifiable, Equatable {
        let id: UUID
        var name: String
        var duration: Double
        var thumbnail: UIImage?
        var projectID: String?
        var state: State
        let addedAt: Date
        /// Plan · In progress: the Bulk send this video belongs to (nil for a single cut).
        var group: String? = nil

        static func == (a: Item, b: Item) -> Bool {
            a.id == b.id && a.name == b.name && a.duration == b.duration && a.projectID == b.projectID
                && a.state == b.state && (a.thumbnail == nil) == (b.thumbnail == nil)
        }
    }

    @Published private(set) var items: [Item] = []
    /// "<name> is ready · View" (bottom notice, Figma 1134:3027).
    @Published var readyToast: (projectID: String, name: String)?
    /// 23 Oct: "Your 10 videos are ready ✨" / "9 ready, 1 needs attention" -- one notice for a whole Bulk.
    @Published var groupToast: String?
    /// "processing": Edits opens on its Processing filter ("Go to Edits").
    @Published var requestedEditsFilter: String?
    /// Mix: the Mix just started (Edits opens on Processing and shows its group).
    @Published var requestedMix: MixSummary?
    /// Plan · "In progress": Edits · Processing shows only this group (a Bulk id or a Mix id); nil = everything.
    @Published var focus: String?
    /// Plan · "Bulk": the Create flow opens the gallery in Bulk (the same upload pipeline as before).
    @Published var requestedCreateMode: CreateMode?

    /// A Bulk send followed in Plan · In progress: how many videos it had (the ones still here are not ready).
    struct BulkGroup: Identifiable, Equatable {
        let id: String
        let total: Int
        let startedAt: Date
    }
    @Published private(set) var bulkGroups: [BulkGroup] = []

    /// Bulk sends with videos still being made, newest first: (group, ready, total, first thumbnail).
    var activeBulks: [(group: BulkGroup, ready: Int, thumbnail: UIImage?)] {
        bulkGroups.compactMap { group in
            let left = items.filter { $0.group == group.id }
            guard left.contains(where: { $0.state != .ready }) else { return nil }
            let notReady = left.filter { $0.state != .ready }.count
            return (group, max(0, group.total - notReady), left.first { $0.thumbnail != nil }?.thumbnail)
        }.sorted { $0.group.startedAt > $1.group.startedAt }
    }

    /// The Bulk send a row belongs to ("3 of 10 ready" above its videos).
    func bulk(of itemID: UUID?) -> (group: BulkGroup, ready: Int)? {
        guard let itemID, let groupID = items.first(where: { $0.id == itemID })?.group,
              let group = bulkGroups.first(where: { $0.id == groupID }) else { return nil }
        let notReady = items.filter { $0.group == groupID && $0.state != .ready }.count
        return (group, max(0, group.total - notReady))
    }
    /// Retry for a failed item (resume the upload, or open the cut).
    private var retries: [UUID: () -> Void] = [:]

    /// Adds the videos of a new job, in order; returns their ids.
    @discardableResult
    func add(_ entries: [(name: String, duration: Double)], group: String? = nil) -> [UUID] {
        let now = Date()
        let new = entries.map { Item(id: UUID(), name: $0.name, duration: $0.duration, thumbnail: nil, projectID: nil,
                                     state: .queued, addedAt: now, group: group) }
        items.append(contentsOf: new)
        if let group, !new.isEmpty {
            bulkGroups.append(BulkGroup(id: group, total: new.count, startedAt: now))
            GroupNotify.start(group, total: new.count)
        }
        return new.map(\.id)
    }

    func update(_ id: UUID?, name: String? = nil, duration: Double? = nil, thumbnail: UIImage? = nil,
                projectID: String? = nil, state: State? = nil) {
        guard let id, let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        if let name, !name.isEmpty { item.name = name }
        if let duration, duration > 0 { item.duration = duration }
        if let thumbnail { item.thumbnail = thumbnail }
        if let projectID {
            if item.projectID != projectID, let group = item.group {
                GroupNotify.replacePlaceholder(group, itemID: item.id)
                GroupNotify.link(group, projectID: projectID)
            }
            item.projectID = projectID
        }
        if let state { item.state = state }
        if item != items[index] || thumbnail != nil { items[index] = item }
    }

    func fail(_ id: UUID?, detail: String, retry: (() -> Void)?) {
        guard let id else { return }
        update(id, state: .failed(detail))
        retries[id] = retry
        // A Bulk video whose upload stopped counts as "needs attention" for the group's one notice.
        if let item = items.first(where: { $0.id == id }), let group = item.group, item.projectID == nil,
           let done = GroupNotify.uploadFailed(group, itemID: id) {
            groupDone(done.id, done.group)
        }
    }

    func retry(_ id: UUID) {
        guard let action = retries.removeValue(forKey: id) else { return }
        if let group = items.first(where: { $0.id == id })?.group { GroupNotify.replacePlaceholder(group, itemID: id) }
        update(id, state: .uploading(0))
        action()
    }

    func remove(_ id: UUID) {
        items.removeAll { $0.id == id }
        retries[id] = nil
    }

    /// The server finished this cut: it leaves Processing and the notice shows.
    func ready(projectID: String, title: String?) {
        // 23 Oct: a Bulk video stays in its group as Ready (no notice of its own); the group tells once, when all are done.
        if let index = items.firstIndex(where: { $0.projectID == projectID }), items[index].group != nil {
            items[index].state = .ready
            if let done = GroupNotify.finished(projectID, ok: true) { groupDone(done.id, done.group) }
            return
        }
        if GroupNotify.isGrouped(projectID) {
            // A Bulk video of a send this iPhone no longer lists (CutSell was reopened): only the group tells.
            if let done = GroupNotify.finished(projectID, ok: true) { groupDone(done.id, done.group) }
            return
        }
        let name = items.first { $0.projectID == projectID }?.name ?? title ?? "Your cut"
        items.removeAll { $0.projectID == projectID }
        readyToast = (projectID, name)
    }

    /// The server could not make this cut (a Bulk video: it counts as "needs attention" in the group's notice).
    func serverFailed(projectID: String) {
        if let done = GroupNotify.finished(projectID, ok: false) { groupDone(done.id, done.group) }
    }

    /// Every video of a Bulk send is done: one notification (CutSell not on screen) or one notice (on screen); the
    /// ready videos leave Processing (they are in Ready), the ones that need attention stay with their Retry.
    private func groupDone(_ id: String, _ g: GroupNotify.Group) {
        GroupNotify.post(id, g)
        groupToast = GroupNotify.title(g)
        items.removeAll { $0.group == id && $0.state == .ready }
        if !items.contains(where: { $0.group == id }) { bulkGroups.removeAll { $0.id == id } }
    }

    /// The rows of the Processing filter, in order: this iPhone's items, then cuts still processing on
    /// the server that this iPhone is not uploading (e.g. after CutSell was reopened).
    func rows(projects: [Project], tracker: CutTracker) -> [Row] {
        var rows: [Row] = []
        var listed = Set<String>()
        for item in items {
            var state = item.state
            if let id = item.projectID {
                listed.insert(id)
                if let cut = tracker.cut(id), state != .ready {
                    if cut.isReady { if item.group == nil { continue }; state = .ready }
                    else if cut.isFailed { state = .failed(Self.failure(cut)) }
                    else if case .failed = state {} else {
                        state = cut.jobState == "queued" ? .queued : .processing(max(0, CutTracker.serverStage(cut).percent))
                    }
                }
            }
            // 23 Oct: a Bulk video shows only Queued / Processing / Ready, no times ("Taking a little longer" is Single only).
            rows.append(Row(id: item.id.uuidString, itemID: item.id, projectID: item.projectID, name: item.name,
                            duration: item.duration, thumbnail: item.thumbnail, state: state,
                            slow: item.group == nil && Self.slow(item.projectID.flatMap { tracker.cut($0) }, state),
                            group: item.group))
        }
        for project in projects where Project.processingStates.contains(project.state) && !listed.contains(project.projectID) {
            var state = State.processing(0)
            if let cut = tracker.cut(project.projectID) {
                if cut.isReady { continue }
                state = cut.isFailed ? .failed(Self.failure(cut))
                    : cut.jobState == "queued" ? .queued : .processing(max(0, CutTracker.serverStage(cut).percent))
            }
            let grouped = GroupNotify.group(of: project.projectID)
            rows.append(Row(id: project.projectID, itemID: nil, projectID: project.projectID, name: project.title,
                            duration: 0, thumbnail: ThumbnailStore.shared.image(for: project.projectID), state: state,
                            slow: grouped == nil && Self.slow(tracker.cut(project.projectID), state), group: grouped))
        }
        return rows
    }

    /// 22 Oct: a video where nobody can be heard says so (and can go to Mix) instead of "needs another try".
    static let noSpeech = "We couldn't hear anyone talking in this video"
    static func failure(_ cut: CutTracker.Cut) -> String {
        cut.error == "no_speech" ? noSpeech : "The cut needs another try"
    }

    struct Row: Identifiable {
        let id: String
        let itemID: UUID?
        let projectID: String?
        let name: String
        let duration: Double
        let thumbnail: UIImage?
        let state: State
        /// 23 Oct: being cut for longer than usual (Single / Combine only; Bulk and Mix never show times).
        var slow = false
        /// The Bulk send of this video (its row shows only Queued / Processing / Ready).
        var group: String? = nil
    }

    /// 23 Oct (PO): Bulk and Mix, from the first moment (no times, no "Taking a little longer").
    static let groupMessage = "We'll notify you when your videos are ready. You can keep using CutSell."

    static let slowAfter: TimeInterval = 8 * 60
    static let slowText = "Taking a little longer, we'll notify you"
    static func slow(_ cut: CutTracker.Cut?, _ state: State) -> Bool {
        guard let cut, case .processing = state else { return false }
        return Date().timeIntervalSince(cut.trackedAt) > slowAfter
    }
}

// MARK: - Queue row (Figma Plan/QueueRow 462:720: Processing / Queued / Failed; 350 x 84)

struct QueueRowView: View {
    let order: Int
    let row: ProcessingQueue.Row
    var onRetry: () -> Void = {}

    private var failed: Bool { if case .failed = row.state { return true }; return false }

    /// White Indigo row (Figma 1177:113 / 125 / 145): number, thumbnail, name ("…" when long), duration · state.
    var body: some View {
        HStack(spacing: 0) {
            Text(verbatim: String(format: "%02d", order))
                .font(CS.inter(11, .bold))
                .foregroundStyle(failed ? CS.wiError : CS.accent)
                .frame(width: 28, height: 28)
                .background(failed ? CS.wiErrorBg2 : CS.accentSoft, in: Circle())
            thumbnail
                .frame(width: 46, height: 60)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .padding(.leading, 10)
            VStack(alignment: .leading, spacing: 0) {
                Text(row.name)
                    .font(CS.inter(14, .bold))
                    .foregroundStyle(CS.wiInk)
                    .lineLimit(1)
                    .truncationMode(.tail)
                HStack(spacing: 0) {
                    let time = isNoSpeech || row.slow ? "" : Self.clock(row.duration)
                    if isNoSpeech || row.slow {
                        Text(L10n.tr(isNoSpeech ? ProcessingQueue.noSpeech : ProcessingQueue.slowText))
                            .foregroundStyle(isNoSpeech ? CS.wiError : CS.wiSecondary)
                            .font(CS.inter(12, .medium))
                            .accessibilityIdentifier(isNoSpeech ? "queueRowNoSpeech" : "queueRowSlow")
                    } else {
                        Text(verbatim: time)
                            .font(CS.inter(12, .medium))
                            .foregroundStyle(CS.wiSecondary)
                            .accessibilityIdentifier("queueRowDuration")
                        Text(verbatim: (time.isEmpty ? "" : " · ") + L10n.tr(statusText))
                            .font(CS.inter(12, .semibold))
                            .foregroundStyle(statusColor)
                            .accessibilityIdentifier("queueRowStatus")
                    }
                }
                .lineLimit(1).minimumScaleFactor(0.75)
                .padding(.top, 4)
                if let fraction = progress {
                    GeometryReader { proxy in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color(fixed: 0xE6EAF2))
                            Capsule().fill(CS.accent).frame(width: max(6, proxy.size.width * fraction))
                        }
                    }
                    .frame(height: 6)
                    .frame(maxWidth: 180)
                    .padding(.top, 10)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.leading, 12)
            trailing.padding(.leading, 8)
        }
        .padding(.leading, 13)
        .padding(.trailing, 13)
        .frame(height: 84)
        .wiCard(fill: failed ? CS.wiErrorBg : CS.wiCard, border: failed ? Color(fixed: 0xF6C9CB) : CS.wiBorder, shadow: false)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("queueRow")
    }

    @ViewBuilder private var thumbnail: some View {
        if let image = row.thumbnail {
            Image(uiImage: image).resizable().scaledToFill()
        } else {
            CS.wiSurface2
        }
    }

    private var isNoSpeech: Bool { if case .failed(let why) = row.state { return why == ProcessingQueue.noSpeech }; return false }

    /// Figma 1177:152 "Retry": white, 1.5 pt red border, 32 high, radius 16, the turn icon and Bold 12.
    private func outlined(_ title: String, color: Color, icon: String?) -> some View {
        HStack(spacing: 4) {
            if let icon { Image(icon).renderingMode(.template).resizable().frame(width: 14, height: 14) }
            Text(L10n.tr(title)).font(CS.inter(12, .bold)).lineLimit(1).fixedSize()
        }
        .foregroundStyle(color)
        .padding(.leading, icon == nil ? 12 : 10).padding(.trailing, 12)
        .frame(height: 32)
        .background(CS.wiPage, in: Capsule())
        .overlay(Capsule().stroke(color, lineWidth: 1.5))
    }

    @ViewBuilder private var trailing: some View {
        switch row.state {
        case .failed where isNoSpeech:
            // Nobody talks in it: Retry would fail again; it can still be a clip of a Mix.
            Button { AppRouter.shared.pending = .mix } label: { outlined("Use in Sales Mix", color: CS.accent, icon: nil) }
                .buttonStyle(.plain)
                .accessibilityIdentifier("queueRowUseInMix")
        case .failed:
            Button(action: onRetry) { outlined("Retry", color: CS.wiError, icon: "WIRetry") }
                .buttonStyle(.plain)
                .accessibilityIdentifier("queueRowRetry")
        case .queued:
            Image("WIClock").renderingMode(.template).resizable().frame(width: 24, height: 24)
                .foregroundStyle(CS.wiSecondary)
                .frame(width: 28)
        case .uploading(let p), .processing(let p):
            ProgressRing(fraction: row.group != nil ? nil : Double(min(100, max(0, p))) / 100)
                .frame(width: 28, height: 28)
        case .ready:
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 24, weight: .semibold))
                .foregroundStyle(CS.wiReady)
                .frame(width: 28)
                .accessibilityIdentifier("queueRowReady")
        }
    }

    private var progress: Double? {
        if row.group != nil { return nil }        // Bulk: the group's "3 of 10 ready" bar, no bar per video
        switch row.state {
        case .uploading(let p), .processing(let p): return Double(min(100, max(0, p))) / 100
        default: return nil
        }
    }

    private var statusText: String {
        switch row.state {
        case .queued: return "Queued"
        case .ready: return "Done"
        case .uploading(let p): return row.group != nil ? "Processing" : "Uploading \(p)%"
        case .processing(let p): return row.group != nil ? "Processing" : "Processing \(p)%"
        case .failed(let why): return why == ProcessingQueue.noSpeech ? "No voice" : "Failed"
        }
    }

    private var statusColor: Color {
        switch row.state {
        case .queued: return CS.wiSecondary
        case .failed: return CS.wiError
        case .ready: return CS.wiSuccess
        default: return CS.accent
        }
    }

    static func clock(_ seconds: Double) -> String {
        guard seconds > 0 else { return "" }
        let total = Int(seconds.rounded())
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}

/// Figma 1177:122 "Ring": a 28 pt progress ring (#E6EAF2 track, #1769FF arc). No fraction = an arc that turns.
struct ProgressRing: View {
    let fraction: Double?
    @State private var turning = false

    var body: some View {
        ZStack {
            Circle().stroke(Color(fixed: 0xE6EAF2), lineWidth: 3)
            if let fraction {
                Circle()
                    .trim(from: 0, to: max(0.04, fraction))
                    .stroke(CS.accent, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .animation(.easeOut(duration: 0.3), value: fraction)
            } else {
                Circle()
                    .trim(from: 0, to: 0.28)
                    .stroke(CS.accent, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    .rotationEffect(.degrees(turning ? 270 : -90))
                    .animation(.linear(duration: 1.1).repeatForever(autoreverses: false), value: turning)
            }
        }
        .padding(1.5)
        .onAppear { turning = true }
    }
}

/// The Processing icon (Figma Queue/Icon/Processing), turning while the cut is being made.
private struct SpinningIcon: View {
    @State private var turning = false
    var body: some View {
        Image("QueueIconProcessing").resizable().frame(width: 26, height: 26)
            .rotationEffect(.degrees(turning ? 360 : 0))
            .animation(.linear(duration: 1.2).repeatForever(autoreverses: false), value: turning)
            .onAppear { turning = true }
    }
}

// MARK: - Bulk group header: "Bulk · 10 videos" · "3 of 10 ready" and one bar for the group (no times)

struct BulkGroupHeader: View {
    let total: Int
    let ready: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(L10n.tr("Bulk · \(total) videos"))
                    .font(CS.cardTitle).foregroundStyle(CS.wiInk)
                Spacer()
                Text(L10n.tr("\(ready) of \(total) ready"))
                    .font(CS.inter(12, .semibold)).foregroundStyle(CS.accent)
                    .accessibilityIdentifier("bulkGroupReady")
            }
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color(fixed: 0xE6EAF2))
                    Capsule().fill(CS.accent).frame(width: max(6, proxy.size.width * Double(ready) / Double(max(1, total))))
                }
            }
            .frame(height: 6)
        }
        .padding(14)
        .wiCard(shadow: false)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("bulkGroupHeader")
    }
}

// MARK: - Bulk · Started sheet (Figma 1134:4654)

struct BulkStartedSheet: View {
    let count: Int
    let onGoToEdits: () -> Void
    let onStay: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Capsule().fill(Color(hex: 0x4D5770)).frame(width: 40, height: 5)
                .frame(maxWidth: .infinity).padding(.top, 9)
            Text(L10n.tr("Your \(count) videos are being edited"))
                .font(CS.inter(22, .bold))
                .foregroundStyle(Color.ink)
                .padding(.top, 21)
            Text(L10n.tr(ProcessingQueue.groupMessage))
                .font(CS.inter(14))
                .foregroundStyle(Color(hex: 0x99A3BD))
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 10)
            HStack(spacing: 0) {
                card(title: "Processing", subtitle: "In Edits, in order", highlighted: true)
                Text(L10n.tr("→")).font(CS.inter(20, .bold)).foregroundStyle(Color(hex: 0x99A3BD)).frame(width: 42)
                card(title: "Ready", subtitle: "Each one, as soon as it’s done", highlighted: false)
            }
            .padding(.top, 16)
            Text(L10n.tr("One notification when the whole group is ready."))
                .font(CS.inter(12))
                .foregroundStyle(Color(hex: 0x99A3BD))
                .padding(.top, 16)
            Button(action: onGoToEdits) {
                Text(L10n.tr("Go to Edits"))
                    .font(CS.inter(16, .bold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 52)
                    .background(CS.accentGradient, in: Capsule())
            }
            .buttonStyle(.plain)
            .padding(.top, 18)
            .accessibilityIdentifier("bulkGoToEdits")
            Button(action: onStay) {
                Text(L10n.tr("Stay here"))
                    .font(CS.inter(15, .semibold))
                    .foregroundStyle(Color.ink)
                    .frame(maxWidth: .infinity)
                    .frame(height: 44)
            }
            .buttonStyle(.plain)
            .padding(.top, 6)
            .accessibilityIdentifier("bulkStayHere")
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 23)
        .frame(maxWidth: .infinity)
        .frame(height: 400, alignment: .top)
        .background(Color(hex: 0x0D121F))
        .clipShape(UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28, style: .continuous))
        .overlay(UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28, style: .continuous)
            .stroke(Color(hex: 0x33405C), lineWidth: 1))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("bulkStartedSheet")
    }

    private func card(title: String, subtitle: String, highlighted: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(L10n.tr(title))
                .font(CS.inter(15, .semibold))
                .foregroundStyle(highlighted ? CS.cyan : .white)
            Text(L10n.tr(subtitle))
                .font(CS.inter(11))
                .foregroundStyle(Color(hex: 0x99A3BD))
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(12.5)
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(height: 76)
        .background(Color(hex: 0x141C2E), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay {
            if highlighted {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(CS.accent, lineWidth: 1.5)
            } else {
                RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color(hex: 0x33405C), lineWidth: 1)
            }
        }
    }
}

// MARK: - "<name> is ready · View" (Figma Edits/Toast/NewReady 1134:3027)

struct ReadyToast: View {
    let name: String
    let onView: () -> Void

    var body: some View {
        // White Indigo (Figma 1213:2): white card, green check, "View" in blue.
        HStack(spacing: 10) {
            Image(systemName: "checkmark").font(.system(size: 11, weight: .bold)).foregroundStyle(CS.wiSuccess)
                .frame(width: 22, height: 22).background(CS.wiSuccessBg, in: Circle())
            Text(L10n.tr("\(Self.short(name)) is ready"))
                .font(CS.inter(14, .semibold))
                .foregroundStyle(CS.wiInk)
                .lineLimit(1)
            Spacer(minLength: 8)
            Button(L10n.tr("View"), action: onView)
                .font(CS.inter(14, .bold))
                .foregroundStyle(CS.accent)
                .accessibilityIdentifier("readyToastView")
        }
        .padding(.horizontal, 14)
        .frame(height: 46)
        .wiCard(radius: 14, fill: CS.wiPage)
        .accessibilityIdentifier("readyToast")
    }

    /// "cutsell01001902849.mp4" → "cutsell…849" (Figma).
    static func short(_ name: String) -> String {
        let base = (name as NSString).deletingPathExtension
        guard base.count > 18 else { return base }
        return "\(base.prefix(7))…\(base.suffix(3))"
    }
}


// MARK: - A whole Bulk is done (one notice for the group)

struct GroupReadyToast: View {
    let text: String
    let onView: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "checkmark").font(.system(size: 11, weight: .bold)).foregroundStyle(CS.wiSuccess)
                .frame(width: 22, height: 22).background(CS.wiSuccessBg, in: Circle())
            Text(text)
                .font(CS.inter(14, .semibold))
                .foregroundStyle(CS.wiInk)
                .lineLimit(1).minimumScaleFactor(0.8)
            Spacer(minLength: 8)
            Button(L10n.tr("View"), action: onView)
                .font(CS.inter(14, .bold))
                .foregroundStyle(CS.accent)
                .accessibilityIdentifier("groupToastView")
        }
        .padding(.horizontal, 14)
        .frame(height: 46)
        .wiCard(radius: 14, fill: CS.wiPage)
        .accessibilityIdentifier("groupToast")
    }
}
