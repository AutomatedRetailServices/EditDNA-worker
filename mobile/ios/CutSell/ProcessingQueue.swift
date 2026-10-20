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
    }

    struct Item: Identifiable, Equatable {
        let id: UUID
        var name: String
        var duration: Double
        var thumbnail: UIImage?
        var projectID: String?
        var state: State
        let addedAt: Date

        static func == (a: Item, b: Item) -> Bool {
            a.id == b.id && a.name == b.name && a.duration == b.duration && a.projectID == b.projectID
                && a.state == b.state && (a.thumbnail == nil) == (b.thumbnail == nil)
        }
    }

    @Published private(set) var items: [Item] = []
    /// "<name> is ready · View" (bottom notice, Figma 1134:3027).
    @Published var readyToast: (projectID: String, name: String)?
    /// "processing": Edits opens on its Processing filter ("Go to Edits").
    @Published var requestedEditsFilter: String?
    /// Mix: the Mix just started (Edits opens on Processing and shows its group).
    @Published var requestedMix: MixSummary?
    /// Retry for a failed item (resume the upload, or open the cut).
    private var retries: [UUID: () -> Void] = [:]

    /// Adds the videos of a new job, in order; returns their ids.
    @discardableResult
    func add(_ entries: [(name: String, duration: Double)]) -> [UUID] {
        let now = Date()
        let new = entries.map { Item(id: UUID(), name: $0.name, duration: $0.duration, thumbnail: nil, projectID: nil,
                                     state: .queued, addedAt: now) }
        items.append(contentsOf: new)
        return new.map(\.id)
    }

    func update(_ id: UUID?, name: String? = nil, duration: Double? = nil, thumbnail: UIImage? = nil,
                projectID: String? = nil, state: State? = nil) {
        guard let id, let index = items.firstIndex(where: { $0.id == id }) else { return }
        var item = items[index]
        if let name, !name.isEmpty { item.name = name }
        if let duration, duration > 0 { item.duration = duration }
        if let thumbnail { item.thumbnail = thumbnail }
        if let projectID { item.projectID = projectID }
        if let state { item.state = state }
        if item != items[index] || thumbnail != nil { items[index] = item }
    }

    func fail(_ id: UUID?, detail: String, retry: (() -> Void)?) {
        guard let id else { return }
        update(id, state: .failed(detail))
        retries[id] = retry
    }

    func retry(_ id: UUID) {
        guard let action = retries.removeValue(forKey: id) else { return }
        update(id, state: .uploading(0))
        action()
    }

    func remove(_ id: UUID) {
        items.removeAll { $0.id == id }
        retries[id] = nil
    }

    /// The server finished this cut: it leaves Processing and the notice shows.
    func ready(projectID: String, title: String?) {
        let name = items.first { $0.projectID == projectID }?.name ?? title ?? "Your cut"
        items.removeAll { $0.projectID == projectID }
        readyToast = (projectID, name)
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
                if let cut = tracker.cut(id) {
                    if cut.isReady { continue }
                    if cut.isFailed { state = .failed("The cut needs another try") }
                    else if case .failed = state {} else {
                        state = cut.jobState == "queued" ? .queued : .processing(max(0, CutTracker.serverStage(cut).percent))
                    }
                }
            }
            rows.append(Row(id: item.id.uuidString, itemID: item.id, projectID: item.projectID, name: item.name,
                            duration: item.duration, thumbnail: item.thumbnail, state: state))
        }
        for project in projects where Project.processingStates.contains(project.state) && !listed.contains(project.projectID) {
            var state = State.processing(0)
            if let cut = tracker.cut(project.projectID) {
                if cut.isReady { continue }
                state = cut.isFailed ? .failed("The cut needs another try")
                    : cut.jobState == "queued" ? .queued : .processing(max(0, CutTracker.serverStage(cut).percent))
            }
            rows.append(Row(id: project.projectID, itemID: nil, projectID: project.projectID, name: project.title,
                            duration: 0, thumbnail: ThumbnailStore.shared.image(for: project.projectID), state: state))
        }
        return rows
    }

    struct Row: Identifiable {
        let id: String
        let itemID: UUID?
        let projectID: String?
        let name: String
        let duration: Double
        let thumbnail: UIImage?
        let state: State
    }
}

// MARK: - Queue row (Figma Plan/QueueRow 462:720: Processing / Queued / Failed; 350 x 84)

struct QueueRowView: View {
    let order: Int
    let row: ProcessingQueue.Row
    var onRetry: () -> Void = {}

    private static let error = Color(hex: 0xFF4D67)
    private static let track = Color(hex: 0x081226)

    var body: some View {
        HStack(spacing: 10) {
            Text(String(format: "%02d", order))
                .font(CS.inter(11, .bold))
                .foregroundStyle(orderColor)
                .frame(width: 28, height: 28)
                .background(Self.track, in: Circle())
            thumbnail
                .frame(width: 46, height: 60)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            VStack(alignment: .leading, spacing: 5) {
                Text(row.name)
                    .font(CS.inter(12, .semibold))
                    .foregroundStyle(CS.textPrimaryF)
                    .lineLimit(1)
                    .truncationMode(.middle)
                HStack {
                    Text(Self.clock(row.duration))
                        .font(CS.inter(11, .medium))
                        .foregroundStyle(CS.textSecondaryF)
                    Spacer(minLength: 6)
                    Text(statusText)
                        .font(CS.inter(10, .semibold))
                        .foregroundStyle(statusColor)
                        .accessibilityIdentifier("queueRowStatus")
                }
                .frame(height: 16)
                if let fraction = progress {
                    GeometryReader { proxy in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Self.track)
                            Capsule()
                                .fill(LinearGradient(stops: [.init(color: Color(hex: 0x19E6FF), location: 0),
                                                             .init(color: Color(hex: 0x1769FF), location: 0.55),
                                                             .init(color: Color(hex: 0xFF21D7), location: 1)],
                                                     startPoint: .leading, endPoint: .trailing))
                                .frame(width: max(5, proxy.size.width * fraction))
                        }
                    }
                    .frame(height: 5)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .frame(height: 60)
            trailing
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(height: 84)
        .background(Color(hex: 0x0C1730), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color(hex: 0x19345F), lineWidth: 1))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("queueRow")
    }

    @ViewBuilder private var thumbnail: some View {
        if let image = row.thumbnail {
            Image(uiImage: image).resizable().scaledToFill()
        } else {
            Color(hex: 0x060B17)
        }
    }

    @ViewBuilder private var trailing: some View {
        switch row.state {
        case .failed:
            Button(action: onRetry) {
                Text("Retry")
                    .font(CS.inter(10, .semibold))
                    .foregroundStyle(Self.error)
                    .frame(width: 48, height: 28)
                    .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(Self.error, lineWidth: 1))
            }
            .buttonStyle(.plain)
            .frame(width: 48, height: 60)
            .accessibilityIdentifier("queueRowRetry")
        case .queued:
            Image("QueueIconQueued").resizable().frame(width: 26, height: 26).frame(width: 32, height: 60)
        case .uploading, .processing:
            SpinningIcon().frame(width: 32, height: 60)
        }
    }

    private var progress: Double? {
        switch row.state {
        case .uploading(let p), .processing(let p): return Double(min(100, max(0, p))) / 100
        default: return nil
        }
    }

    private var statusText: String {
        switch row.state {
        case .queued: return "QUEUED"
        case .uploading(let p): return "UPLOADING \(p)%"
        case .processing(let p): return "PROCESSING \(p)%"
        case .failed: return "FAILED"
        }
    }

    private var statusColor: Color {
        switch row.state {
        case .queued: return CS.textSecondaryF
        case .failed: return Self.error
        default: return CS.cyan
        }
    }

    private var orderColor: Color {
        switch row.state {
        case .queued: return CS.textSecondaryF
        case .failed: return Self.error
        default: return CS.cyan
        }
    }

    static func clock(_ seconds: Double) -> String {
        guard seconds > 0 else { return "" }
        let total = Int(seconds.rounded())
        return String(format: "%d:%02d", total / 60, total % 60)
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

// MARK: - Bulk · Started sheet (Figma 1134:4654)

struct BulkStartedSheet: View {
    let count: Int
    let onGoToEdits: () -> Void
    let onStay: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Capsule().fill(Color(hex: 0x4D5770)).frame(width: 40, height: 5)
                .frame(maxWidth: .infinity).padding(.top, 9)
            Text("Your \(count) videos are being edited")
                .font(CS.inter(22, .bold))
                .foregroundStyle(.white)
                .padding(.top, 21)
            Text("You can close this screen or the app. Editing keeps going, one video at a time.")
                .font(CS.inter(14))
                .foregroundStyle(Color(hex: 0x99A3BD))
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 10)
            HStack(spacing: 0) {
                card(title: "Processing", subtitle: "In Edits, in order", highlighted: true)
                Text("→").font(CS.inter(20, .bold)).foregroundStyle(Color(hex: 0x99A3BD)).frame(width: 42)
                card(title: "Ready", subtitle: "Each one, as soon as it’s done", highlighted: false)
            }
            .padding(.top, 16)
            Text("We’ll notify you as each cut is ready.")
                .font(CS.inter(12))
                .foregroundStyle(Color(hex: 0x99A3BD))
                .padding(.top, 16)
            Button(action: onGoToEdits) {
                Text("Go to Edits")
                    .font(CS.sf(15, .semibold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
                    .background(LinearGradient(stops: [.init(color: Color(hex: 0x1969FF), location: 0),
                                                       .init(color: Color(hex: 0x7C3AED), location: 0.55),
                                                       .init(color: Color(hex: 0x19E6FF), location: 1)],
                                               startPoint: .leading, endPoint: .trailing), in: Capsule())
                    .shadow(color: Color(hex: 0x1969FF, opacity: 0.25), radius: 9, y: 6)
            }
            .buttonStyle(.plain)
            .padding(.top, 18)
            .accessibilityIdentifier("bulkGoToEdits")
            Button(action: onStay) {
                Text("Stay here")
                    .font(CS.inter(15, .semibold))
                    .foregroundStyle(.white)
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
            Text(title)
                .font(CS.inter(15, .semibold))
                .foregroundStyle(highlighted ? CS.cyan : .white)
            Text(subtitle)
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
                    .stroke(LinearGradient(colors: [Color(hex: 0x19E5FF), Color(hex: 0xFF21D7)],
                                           startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 1.5)
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
        HStack(spacing: 0) {
            Text("✓  \(Self.short(name)) is ready")
                .font(CS.inter(13, .semibold))
                .foregroundStyle(.white)
                .lineLimit(1)
            Spacer(minLength: 8)
            Button("View", action: onView)
                .font(CS.inter(13, .semibold))
                .foregroundStyle(CS.cyan)
                .accessibilityIdentifier("readyToastView")
        }
        .padding(.horizontal, 14.5)
        .frame(height: 44)
        .background(Color(hex: 0x141F33), in: Capsule())
        .overlay(Capsule().stroke(LinearGradient(colors: [Color(hex: 0x19E5FF), Color(hex: 0xFF21D7)],
                                                  startPoint: .leading, endPoint: .trailing), lineWidth: 1.5))
        .accessibilityIdentifier("readyToast")
    }

    /// "cutsell01001902849.mp4" → "cutsell…849" (Figma).
    static func short(_ name: String) -> String {
        let base = (name as NSString).deletingPathExtension
        guard base.count > 18 else { return base }
        return "\(base.prefix(7))…\(base.suffix(3))"
    }
}
