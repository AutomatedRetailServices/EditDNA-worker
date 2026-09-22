import SwiftUI

/// The canonical, three-track CutSell Editor base (Mobile V1 Timeline UI
/// gate). Main Video stays on the pre-existing `draft`/`draft-edits`
/// system (`DraftEditorViewModel.selectedClips`/`split`/`remove`/`undo`);
/// Voice-over and Overlay are the D-279/D-282A `TimelineComposition`
/// placements (`timelineAssetLibrary.readyVoiceOvers`/`.readyBroll`
/// only -- an asset that is not `READY` never appears here).
///
/// This is a NEW, additive surface: the pre-existing `VisualTimelineView`
/// (single-row clip inspector -- swap take, trim, per-clip caption/audio)
/// is left completely unmodified and still reachable below this view in
/// `DraftEditorView`, since rebuilding that functionality is out of this
/// gate's scope.
enum TimelineTrackKind: String, CaseIterable, Identifiable, Equatable {
    case mainVideo, voiceOver, overlay
    var id: String { rawValue }

    var title: String {
        switch self {
        case .mainVideo: return "Main Video"
        case .voiceOver: return "Voice-over"
        case .overlay: return "Overlay"
        }
    }

    var systemImage: String {
        switch self {
        case .mainVideo: return "video"
        case .voiceOver: return "mic"
        case .overlay: return "photo.on.rectangle"
        }
    }
}

/// One active track/element at a time -- an enum-keyed struct, never a
/// pair of parallel row/column indices.
struct TimelineSelection: Equatable {
    var track: TimelineTrackKind
    var itemID: String
}

/// One item any track row can render, generalized over Main Video clips
/// (the existing flat `[String: JSONValue]` draft representation) and
/// Overlay/Voice-over placements (D-282A's own typed models) -- a single
/// rendering/selection surface without collapsing their different real
/// backends into one fake shared model.
private struct TimelineRowItem: Identifiable {
    let id: String
    let track: TimelineTrackKind
    let startSec: Double
    let endSec: Double
    let label: String
    /// Main Video only -- reuses the SAME real preview catalog
    /// `VisualTimelineView` already builds from the draft snapshot
    /// (`SourcePreviewAssetCatalog`); never a second/invented source.
    var previewFrames: [TimelineFrame] = []
}

struct TimelineEditorView: View {
    @ObservedObject var model: DraftEditorViewModel

    @State private var selection: TimelineSelection?
    @State private var playheadTime: Double = 0
    @State private var isExpanded = false
    @State private var isPlaying = false
    @State private var showBrollPicker = false
    @State private var showVoiceOverPicker = false
    @State private var justMutatedMainVideo = false
    @State private var canRedoMainVideo = false

    private let pixelsPerSecond: CGFloat = 42
    private let collapsedHeight: CGFloat = 200
    private let expandedHeight: CGFloat = 320

    private var totalDuration: Double {
        max(1, model.timelineDuration, model.timelineComposition?.timelineDurationSec ?? 0)
    }

    /// Reuses `VisualTimelineView`'s own real preview catalog verbatim --
    /// never a second filmstrip/waveform source.
    private var assetCatalog: [String: SourceTimelineAssets] {
        SourcePreviewAssetCatalog.build(from: model.snapshot)
    }

    private var mainVideoItems: [TimelineRowItem] {
        model.selectedClips.map { clip in
            let start = clip["start"]?.doubleValue ?? 0
            let end = clip["end"]?.doubleValue ?? start + 1
            let label = clip["caption_text"]?.stringValue ?? clip["text"]?.stringValue ?? "Clip"
            let sourceAssets = assetCatalog[clip["source_asset_id"]?.stringValue ?? ""]
            let frames = (sourceAssets?.frames ?? []).filter { $0.time >= start && $0.time <= end }
            return TimelineRowItem(
                id: clip["clip_id"]?.stringValue ?? UUID().uuidString, track: .mainVideo,
                startSec: start, endSec: end, label: label, previewFrames: frames
            )
        }
    }

    private var voiceOverItems: [TimelineRowItem] {
        (model.timelineComposition?.voiceOverPlacements ?? []).map { placement in
            let asset = model.timelineAssetLibrary?.readyVoiceOvers.first { $0.assetID == placement.assetID }
            return TimelineRowItem(
                id: placement.placementID, track: .voiceOver,
                startSec: placement.timelineStartSec, endSec: placement.timelineEndSec,
                label: asset != nil ? "Voice-over" : "Voice-over (unavailable)"
            )
        }
    }

    private var overlayItems: [TimelineRowItem] {
        (model.timelineComposition?.brollPlacements ?? []).map { placement in
            let asset = model.timelineAssetLibrary?.readyBroll.first { $0.assetID == placement.assetID }
            return TimelineRowItem(
                id: placement.placementID, track: .overlay,
                startSec: placement.timelineStartSec, endSec: placement.timelineEndSec,
                label: asset != nil ? "B-roll" : "B-roll (unavailable)"
            )
        }
    }

    private func items(for track: TimelineTrackKind) -> [TimelineRowItem] {
        switch track {
        case .mainVideo: return mainVideoItems
        case .voiceOver: return voiceOverItems
        case .overlay: return overlayItems
        }
    }

    private var selectedItem: TimelineRowItem? {
        guard let selection else { return nil }
        return items(for: selection.track).first { $0.id == selection.itemID }
    }

    /// Split/Delete require a real selected element whose span actually
    /// contains the playhead with a safety margin -- never an index that
    /// has scrolled out from under a stale selection.
    private var canSplitAtPlayhead: Bool {
        guard let item = selectedItem else { return false }
        return playheadTime > item.startSec + 0.05 && playheadTime < item.endSec - 0.05
    }

    private var canDeleteSelection: Bool { selectedItem != nil }

    private var canAddOverlayOrVoiceOver: Bool { model.baseEditAssetID != nil }

    private var canUndo: Bool {
        model.canUndoTimelineMutation || justMutatedMainVideo
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            transportRow
            timelineTracks
            if !canAddOverlayOrVoiceOver {
                Text("Overlay and Voice-over placement need this project's primary source registered first.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("timeline.noBaseEditAsset")
            }
            actionBar
        }
        .padding(.horizontal)
        .confirmationDialog("Add Voice-over", isPresented: $showVoiceOverPicker, titleVisibility: .visible) {
            ForEach(model.timelineAssetLibrary?.readyVoiceOvers ?? []) { asset in
                Button("\(Int(asset.durationSec))s voice-over") {
                    Task {
                        await model.addVoiceOverPlacement(
                            assetID: asset.assetID, start: totalDuration, end: totalDuration + asset.durationSec,
                            sourceIn: 0, sourceOut: asset.durationSec
                        )
                    }
                }
            }
            if (model.timelineAssetLibrary?.readyVoiceOvers ?? []).isEmpty {
                Button("No ready voice-overs available", role: .cancel) {}
            }
        }
        .confirmationDialog("Add Overlay", isPresented: $showBrollPicker, titleVisibility: .visible) {
            ForEach(model.timelineAssetLibrary?.readyBroll ?? []) { asset in
                Button("\(Int(asset.durationSec))s B-roll") {
                    Task {
                        await model.addBrollPlacement(
                            assetID: asset.assetID, start: totalDuration, end: totalDuration + asset.durationSec,
                            sourceIn: 0, sourceOut: asset.durationSec
                        )
                    }
                }
            }
            if (model.timelineAssetLibrary?.readyBroll ?? []).isEmpty {
                Button("No ready B-roll available", role: .cancel) {}
            }
        }
    }

    // MARK: - Header / transport

    private var header: some View {
        HStack {
            Text("Editor").font(.headline)
            Spacer()
            Button {
                withAnimation { isExpanded.toggle() }
            } label: {
                Label(isExpanded ? "Collapse" : "Expand", systemImage: isExpanded ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right")
                    .labelStyle(.iconOnly)
            }
            .accessibilityLabel(isExpanded ? "Collapse timeline" : "Expand timeline")
        }
    }

    private var transportRow: some View {
        HStack(spacing: 16) {
            Button {
                isPlaying.toggle()
            } label: {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .font(.title3)
            }
            .accessibilityLabel(isPlaying ? "Pause" : "Play")

            Text(String(format: "%.1fs / %.1fs", playheadTime, totalDuration))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)

            Slider(value: $playheadTime, in: 0...totalDuration)
                .accessibilityLabel("Playhead")
                .accessibilityValue("\(String(format: "%.1f", playheadTime)) seconds of \(String(format: "%.1f", totalDuration))")

            if model.isSavingTimeline {
                ProgressView().controlSize(.small)
            }
        }
    }

    // MARK: - Tracks + playhead

    private var timelineTracks: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            ZStack(alignment: .topLeading) {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(TimelineTrackKind.allCases) { track in
                        trackRow(track)
                    }
                }
                playheadLine
            }
            .frame(height: isExpanded ? expandedHeight : collapsedHeight)
            .padding(.top, 2)
        }
        .scrollIndicators(.hidden)
    }

    private var playheadLine: some View {
        Rectangle()
            .fill(Color.accentColor)
            .frame(width: 2, height: isExpanded ? expandedHeight : collapsedHeight)
            .offset(x: CGFloat(playheadTime) * pixelsPerSecond)
            .accessibilityHidden(true)
    }

    private func trackRow(_ track: TimelineTrackKind) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: track.systemImage).font(.caption2)
                Text(track.title).font(.caption).fontWeight(.semibold)
            }
            .foregroundStyle(.secondary)

            HStack(spacing: 3) {
                ForEach(items(for: track)) { item in
                    TimelineRowCell(
                        item: item,
                        pixelsPerSecond: pixelsPerSecond,
                        isSelected: selection == TimelineSelection(track: track, itemID: item.id)
                    )
                    .onTapGesture {
                        selection = TimelineSelection(track: track, itemID: item.id)
                        justMutatedMainVideo = false
                        canRedoMainVideo = false
                    }
                }

                if track != .mainVideo {
                    Button {
                        if track == .voiceOver { showVoiceOverPicker = true } else { showBrollPicker = true }
                    } label: {
                        Image(systemName: "plus")
                            .frame(width: 44, height: 56)
                            .background(.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
                    }
                    .disabled(!canAddOverlayOrVoiceOver)
                    .accessibilityLabel("Add \(track.title.lowercased())")
                }
            }
        }
    }

    // MARK: - Actions

    private var actionBar: some View {
        HStack(spacing: 14) {
            Button {
                Task { await performSplit() }
            } label: {
                Label("Split", systemImage: "scissors")
            }
            .disabled(!canSplitAtPlayhead)

            Button(role: .destructive) {
                Task { await performDelete() }
            } label: {
                Label("Delete", systemImage: "trash")
            }
            .disabled(!canDeleteSelection)

            Button {
                Task { await performUndo() }
            } label: {
                Label("Undo", systemImage: "arrow.uturn.backward")
            }
            .disabled(!canUndo)

            // Redo only ever has real backend authority for Main Video
            // (`/v1/projects/{id}/draft/redo`, reused verbatim); there is
            // no equivalent redo authority for Overlay/Voice-over
            // placements, so this control is never enabled for them --
            // never a simulated redo.
            Button {
                Task { await performRedo() }
            } label: {
                Label("Redo", systemImage: "arrow.uturn.forward")
            }
            .disabled(!canRedoMainVideo)

            Spacer()

            // D-129 Overlay-pacing entry point: this gate only exposes the
            // control surface (add Overlay/Voice-over material); the
            // dialogue-overlap pacing engine itself is a separate,
            // unauthorized-in-this-gate track.
            Menu {
                Button("Keyframe at playhead") {}
                    .disabled(true)
                Text("Full keyframe editing is not part of this base gate.")
            } label: {
                Label("Keyframe", systemImage: "diamond")
            }
        }
        .buttonStyle(.bordered)
        .labelStyle(.iconOnly)
        .frame(minHeight: 44)
    }

    private func performSplit() async {
        guard let selection, canSplitAtPlayhead else { return }
        switch selection.track {
        case .mainVideo:
            await model.split(clipID: selection.itemID, at: playheadTime)
            justMutatedMainVideo = true
            canRedoMainVideo = false
        case .voiceOver:
            await model.splitVoiceOverPlacement(id: selection.itemID, at: playheadTime)
        case .overlay:
            await model.splitBrollPlacement(id: selection.itemID, at: playheadTime)
        }
    }

    private func performDelete() async {
        guard let selection else { return }
        switch selection.track {
        case .mainVideo:
            await model.remove(clipID: selection.itemID)
            justMutatedMainVideo = true
            canRedoMainVideo = false
        case .voiceOver:
            await model.removeVoiceOverPlacement(id: selection.itemID)
        case .overlay:
            await model.removeBrollPlacement(id: selection.itemID)
        }
        self.selection = nil
    }

    private func performUndo() async {
        if model.canUndoTimelineMutation {
            await model.undoLastTimelineMutation()
        } else if justMutatedMainVideo {
            await model.undo()
            justMutatedMainVideo = false
            canRedoMainVideo = true
        }
    }

    /// Real authority only: reverses the Main Video undo just performed
    /// from this view via the SAME existing `/draft/redo` endpoint --
    /// never available for Overlay/Voice-over, which have no redo
    /// authority at all.
    private func performRedo() async {
        guard canRedoMainVideo else { return }
        await model.redo()
        canRedoMainVideo = false
    }
}

private struct TimelineRowCell: View {
    let item: TimelineRowItem
    let pixelsPerSecond: CGFloat
    let isSelected: Bool

    private var width: CGFloat { max(28, CGFloat(item.endSec - item.startSec) * pixelsPerSecond) }

    /// Downsamples the same way `VisualTimelineView`'s `TimelineClipCell`
    /// does -- never renders more frames than the cell has room for.
    private var displayFrames: [TimelineFrame] {
        guard !item.previewFrames.isEmpty else { return [] }
        let maxFrames = max(2, min(8, Int(width / 22)))
        guard item.previewFrames.count > maxFrames else { return item.previewFrames }
        let stride = max(1, item.previewFrames.count / maxFrames)
        return Swift.stride(from: 0, to: item.previewFrames.count, by: stride).prefix(maxFrames).map { item.previewFrames[$0] }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            ZStack {
                RoundedRectangle(cornerRadius: 7)
                    .fill(.secondary.opacity(0.15))
                if !displayFrames.isEmpty {
                    HStack(spacing: 1) {
                        ForEach(displayFrames) { frame in
                            AsyncImage(url: frame.url) { image in
                                image.resizable().scaledToFill()
                            } placeholder: {
                                Rectangle().fill(.secondary.opacity(0.15))
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .clipped()
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 7))
                }
            }
            .frame(width: width, height: 40)
            .overlay {
                RoundedRectangle(cornerRadius: 7)
                    .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 3)
            }
            Text(item.label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(width: width, alignment: .leading)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(item.label), \(String(format: "%.1f", item.endSec - item.startSec)) seconds")
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }
}
