import SwiftUI
import UniformTypeIdentifiers

/// Mobile V1 Overlay UI gate. Connects ONLY to real, already-existing
/// backend authority -- never an invented route, a simulated position/
/// scale/rotation/opacity/keyframe control, or a fabricated capture
/// capability.
///
/// Real authority used here:
/// - `TimelineAssetRegistryClient.list` (`GET /timeline-assets`) for the
///   asset library; `readyBroll` is D-279's own real READY/
///   SUPPLEMENTAL_BROLL/VIDEO filter.
/// - `OverlayImportManager.importVideo` (`POST /timeline-uploads` with
///   media_class="video" then `POST /timeline-assets` with
///   role="SUPPLEMENTAL_BROLL", media_kind="VIDEO") to import an EXISTING
///   local video file as a new asset. Ingestion is synchronous
///   (`create_video_timeline_asset` probes real source-media profile and
///   returns READY/REJECTED/FAILED immediately) -- never a fabricated
///   polling/QUALIFYING step. Only VIDEO is supported -- `TimelineMediaKind`
///   has no IMAGE case, so a still-image overlay has no real backend
///   authority and is never offered here.
/// - `model.addBrollPlacement` / `.removeBrollPlacement` /
///   `.splitBrollPlacement` / `.setBrollAudioMode` (all real D-282A
///   `PUT /timeline` mutations).
/// - `model.canUndoTimelineMutation` / `.undoLastTimelineMutation()` (the
///   same real captured-pre-mutation-snapshot Undo already used for
///   Voice-over placements -- Overlay shares the identical authority).
///
/// PENDING / honestly unsupported (documented here, never simulated):
/// - Live capture/recording of overlay video: not implemented anywhere in
///   this codebase. Import (an already-existing local video file) is the
///   only real "add new overlay media" capability today.
/// - Position, scale, rotation, opacity, keyframes: `BrollPlacementModel`
///   (`cutsell_app/timeline_routes.py`) carries exactly `placement_id,
///   asset_id, timeline_start_sec, timeline_end_sec, source_in_sec,
///   source_out_sec, audio_mode` -- no position/scale/rotation/opacity/
///   keyframe field at all. `timeline_composition.py`'s own
///   `BrollPlacement` dataclass confirms this: it is a full-frame visual
///   replacement over `[timeline_start_sec, timeline_end_sec)`, never a
///   positioned/scaled/rotated layer. (A DIFFERENT, older draft-based
///   "media overlay" system -- `cutsell_app/overlay_routes.py`'s
///   `add_media_overlay`, with real x/y/width fields -- does exist in this
///   codebase, but it operates on the separate `draft` dict, not this
///   D-279/D-282A `TimelineComposition` Overlay/B-roll track; mixing the
///   two would misrepresent which system this control surface actually
///   edits.) Every control below for these is shown, honestly disabled,
///   never wired to a fake mutation.
/// - `audio_mode` IS real and mutable (`setBrollAudioMode`): one of three
///   backend-accepted values. Note found during audit:
///   `timeline_composition_executor.py`'s own comment states
///   "KEEP_PRIMARY_VOICE and MUTE_BROLL_AUDIO both leave the primary voice
///   as-is" in the current renderer -- only USE_BROLL_AUDIO currently
///   changes the rendered audio outcome. All three are still shown (the
///   backend validates and stores all three), never narrowed to two on
///   this view's own authority.
struct OverlayView: View {
    @ObservedObject var model: DraftEditorViewModel
    let initialPlacementID: String?

    @Environment(\.dismiss) private var dismiss
    @State private var selectedPlacementID: String?
    @State private var splitTime: Double = 0
    @State private var isImporting = false
    @State private var justMutated = false

    private enum OverlayStage { case empty, importing, ready, error }

    private var placements: [BrollPlacement] {
        model.timelineComposition?.brollPlacements ?? []
    }

    private var readyAssets: [TimelineMediaAsset] {
        model.timelineAssetLibrary?.readyBroll ?? []
    }

    private var stage: OverlayStage {
        if model.errorMessage != nil { return .error }
        if model.isSavingTimeline { return .importing }
        if placements.isEmpty && readyAssets.isEmpty { return .empty }
        return .ready
    }

    private var selectedPlacement: BrollPlacement? {
        guard let selectedPlacementID else { return nil }
        return placements.first { $0.placementID == selectedPlacementID }
    }

    private var canUndo: Bool { model.canUndoTimelineMutation || justMutated }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    stageRow
                }

                Section("Import") {
                    // Real capability, independent of any timeline
                    // placement precondition: registering a new asset via
                    // /timeline-uploads + /timeline-assets never requires a
                    // base edit asset -- only PLACING one on the timeline
                    // does (see "Add existing to timeline" below). Never a
                    // live capture -- see the file doc for why recording
                    // has no real authority yet.
                    Button {
                        isImporting = true
                    } label: {
                        Label("Import overlay video", systemImage: "video.badge.plus")
                    }
                    .accessibilityIdentifier("overlay.importButton")

                    Text("Only video is supported -- there is no real backend authority for a still-image overlay on this track.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Add existing to timeline") {
                    if readyAssets.isEmpty {
                        Text("No ready overlay video yet.")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(readyAssets) { asset in
                            Button("\(Int(asset.durationSec))s overlay") {
                                Task { await addToTimeline(asset) }
                            }
                            .disabled(model.baseEditAssetID == nil)
                        }
                        if model.baseEditAssetID == nil {
                            Text("Placing overlay on the timeline needs this project's primary source registered first.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section("Overlay track") {
                    if placements.isEmpty {
                        Text("No overlay placed on the timeline yet.")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(placements) { placement in
                            Button {
                                selectedPlacementID = placement.placementID
                                syncSplitTime()
                            } label: {
                                HStack {
                                    Text(assetLabel(for: placement))
                                    Spacer()
                                    Text(String(format: "%.1fs–%.1fs", placement.timelineStartSec, placement.timelineEndSec))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    if placement.placementID == selectedPlacementID {
                                        Image(systemName: "checkmark.circle.fill").foregroundStyle(Color.accentColor)
                                    }
                                }
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }

                if let selectedPlacement {
                    Section("Split selected placement") {
                        Slider(
                            value: $splitTime,
                            in: selectedPlacement.timelineStartSec...selectedPlacement.timelineEndSec
                        )
                        .accessibilityLabel("Split point")
                        .accessibilityValue("\(String(format: "%.1f", splitTime)) seconds")

                        HStack(spacing: 14) {
                            Button {
                                Task { await performSplit() }
                            } label: {
                                Label("Split", systemImage: "scissors")
                            }
                            .disabled(!canSplitAtSplitTime(selectedPlacement))

                            Button(role: .destructive) {
                                Task { await performDelete() }
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                            .accessibilityIdentifier("overlay.deleteButton")
                        }
                        .buttonStyle(.bordered)
                        .frame(minHeight: 44)
                    }

                    // Real, backend-accepted mutation -- the only editable
                    // field on a B-roll placement besides timing.
                    Section("Audio mode") {
                        Picker("Audio mode", selection: audioModeBinding(for: selectedPlacement)) {
                            Text("Keep primary voice").tag(TimelineAudioMode.keepPrimaryVoice)
                            Text("Use overlay's own audio").tag(TimelineAudioMode.useBrollAudio)
                            Text("Mute overlay audio").tag(TimelineAudioMode.muteBrollAudio)
                        }
                        .pickerStyle(.menu)
                        .accessibilityIdentifier("overlay.audioModePicker")
                    }
                }

                // Position/scale/rotation/opacity/keyframes -- honestly
                // disabled: no real backend field exists for any of these
                // on a B-roll placement (see the file doc).
                Section("Position, scale & effects (not yet supported)") {
                    HStack {
                        Text("Position")
                        Spacer()
                        Text("Full-frame").foregroundStyle(.secondary)
                    }
                    HStack {
                        Text("Scale")
                        Spacer()
                        Slider(value: .constant(1.0), in: 0.1...1).disabled(true)
                    }
                    HStack {
                        Text("Rotation")
                        Spacer()
                        Slider(value: .constant(0.0), in: -180...180).disabled(true)
                    }
                    HStack {
                        Text("Opacity")
                        Spacer()
                        Slider(value: .constant(1.0), in: 0...1).disabled(true)
                    }
                    HStack {
                        Text("Keyframes")
                        Spacer()
                        Button("Add keyframe") {}
                            .disabled(true)
                    }
                    Text("The current backend contract carries no position, scale, rotation, opacity, or keyframe field for overlay placements -- these controls are shown disabled rather than simulated.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .accessibilityIdentifier("overlay.transformControlsUnsupported")
                }

                Section {
                    Button {
                        Task { await performUndo() }
                    } label: {
                        Label("Undo", systemImage: "arrow.uturn.backward")
                    }
                    .disabled(!canUndo)
                    .frame(minHeight: 44)
                }
            }
            .navigationTitle("Overlay")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
            }
            .onAppear {
                selectedPlacementID = initialPlacementID ?? placements.first?.placementID
                syncSplitTime()
            }
            .fileImporter(isPresented: $isImporting, allowedContentTypes: [.movie]) { result in
                switch result {
                case .success(let url):
                    Task { await performImport(url) }
                case .failure(let error):
                    model.errorMessage = error.localizedDescription
                }
            }
            .alert("CutSell", isPresented: Binding(
                get: { model.errorMessage != nil },
                set: { if !$0 { model.errorMessage = nil } }
            )) { Button("OK", role: .cancel) {} } message: { Text(model.errorMessage ?? "") }
        }
    }

    @ViewBuilder
    private var stageRow: some View {
        switch stage {
        case .empty:
            Label("No overlay yet", systemImage: "photo.on.rectangle")
                .foregroundStyle(.secondary)
        case .importing:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Importing…")
            }
        case .ready:
            Label("Ready", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .error:
            Label("Error", systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
        }
    }

    private func assetLabel(for placement: BrollPlacement) -> String {
        readyAssets.first { $0.assetID == placement.assetID } != nil ? "Overlay" : "Overlay (unavailable)"
    }

    private func syncSplitTime() {
        guard let selectedPlacement else { return }
        splitTime = (selectedPlacement.timelineStartSec + selectedPlacement.timelineEndSec) / 2
    }

    private func canSplitAtSplitTime(_ placement: BrollPlacement) -> Bool {
        splitTime > placement.timelineStartSec + 0.05 && splitTime < placement.timelineEndSec - 0.05
    }

    private func audioModeBinding(for placement: BrollPlacement) -> Binding<TimelineAudioMode> {
        Binding(
            get: { placement.audioMode },
            set: { newMode in Task { await performSetAudioMode(id: placement.placementID, mode: newMode) } }
        )
    }

    private func addToTimeline(_ asset: TimelineMediaAsset) async {
        let end = (model.timelineComposition?.timelineDurationSec ?? 0) + asset.durationSec
        let start = model.timelineComposition?.timelineDurationSec ?? 0
        justMutated = false
        await model.addBrollPlacement(
            assetID: asset.assetID, start: start, end: end, sourceIn: 0, sourceOut: asset.durationSec
        )
        justMutated = true
    }

    private func performSplit() async {
        guard let id = selectedPlacementID, let placement = selectedPlacement, canSplitAtSplitTime(placement) else { return }
        justMutated = false
        await model.splitBrollPlacement(id: id, at: splitTime)
        justMutated = true
    }

    private func performDelete() async {
        guard let id = selectedPlacementID else { return }
        justMutated = false
        await model.removeBrollPlacement(id: id)
        justMutated = true
        selectedPlacementID = nil
    }

    private func performSetAudioMode(id: String, mode: TimelineAudioMode) async {
        justMutated = false
        await model.setBrollAudioMode(id: id, mode: mode)
        justMutated = true
    }

    private func performImport(_ url: URL) async {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        await model.importOverlayVideo(fileURL: url)
    }

    /// Same honest, scoped Undo pattern already established for Main Video,
    /// Captions, and Voice-over: `undoLastTimelineMutation()` only clears
    /// its captured snapshot on a confirmed successful re-save -- a failed
    /// revert leaves the snapshot intact so the user can retry, never
    /// inferred from `await` alone.
    private func performUndo() async {
        if model.canUndoTimelineMutation {
            await model.undoLastTimelineMutation()
            justMutated = false
        }
    }
}
