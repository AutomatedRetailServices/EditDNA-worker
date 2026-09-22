import SwiftUI
import UniformTypeIdentifiers

/// Mobile V1 B-roll UI gate. Connects ONLY to real, already-existing D-279/
/// D-282A `TimelineComposition` backend authority -- never an invented
/// route, a simulated position/scale/rotation/opacity/keyframe control
/// (those belong to a DIFFERENT feature -- see the corrective note below),
/// or a fabricated capture capability.
///
/// CORRECTIVE NOTE (this file was previously named OverlayView.swift):
/// `BrollPlacement` is a full-frame visual REPLACEMENT clip rendered at
/// x=0,y=0, 1080x1920 by `timeline_composition_executor.py` -- it has no
/// position, scale, rotation, or opacity. Naming this screen "Overlay" was
/// a mislabeling: the real, positioned/scaled Overlay control surface
/// (`/v1/overlays/*`, `cutsell_worker/overlay_edits.py`, genuinely wired
/// into the active export renderer via `media_overlay_render.py`'s ffmpeg
/// `overlay=x=...:y=...` filter) is a SEPARATE, real feature -- see
/// `OverlayView.swift`. This screen is renamed to `BrollView` to match
/// what it actually controls, and no longer shows any position/scale/
/// rotation/opacity/keyframe control at all (not even disabled) -- those
/// never belonged to B-roll's real contract.
///
/// Real authority used here:
/// - `TimelineAssetRegistryClient.list` (`GET /timeline-assets`) for the
///   asset library; `readyBroll` is D-279's own real READY/
///   SUPPLEMENTAL_BROLL/VIDEO filter.
/// - `BrollImportManager.importVideo` (`POST /timeline-uploads` with
///   media_class="video" then `POST /timeline-assets` with
///   role="SUPPLEMENTAL_BROLL", media_kind="VIDEO") to import an EXISTING
///   local video file as a new asset. Ingestion is synchronous
///   (`create_video_timeline_asset` probes real source-media profile and
///   returns READY/REJECTED/FAILED immediately) -- never a fabricated
///   polling/QUALIFYING step. Only VIDEO is supported -- `TimelineMediaKind`
///   has no IMAGE case, so a still-image B-roll clip has no real backend
///   authority and is never offered here.
/// - `model.addBrollPlacement` / `.removeBrollPlacement` /
///   `.splitBrollPlacement` / `.setBrollAudioMode` (all real D-282A
///   `PUT /timeline` mutations).
/// - `model.canUndoTimelineMutation` / `.undoLastTimelineMutation()` (the
///   same real captured-pre-mutation-snapshot Undo already used for
///   Voice-over placements -- B-roll shares the identical authority).
///
/// PENDING / honestly unsupported (documented here, never simulated):
/// - Live capture/recording of B-roll video: not implemented anywhere in
///   this codebase. Import (an already-existing local video file) is the
///   only real "add new B-roll media" capability today.
/// - `audio_mode` IS real and mutable (`setBrollAudioMode`): one of three
///   backend-accepted values. Note found during audit:
///   `timeline_composition_executor.py`'s own comment states
///   "KEEP_PRIMARY_VOICE and MUTE_BROLL_AUDIO both leave the primary voice
///   as-is" in the current renderer -- only USE_BROLL_AUDIO currently
///   changes the rendered audio outcome. All three are still shown (the
///   backend validates and stores all three), never narrowed to two on
///   this view's own authority.
struct BrollView: View {
    @ObservedObject var model: DraftEditorViewModel
    let initialPlacementID: String?

    @Environment(\.dismiss) private var dismiss
    @State private var selectedPlacementID: String?
    @State private var splitTime: Double = 0
    @State private var isImporting = false
    @State private var justMutated = false

    private enum BrollStage { case empty, importing, ready, error }

    private var placements: [BrollPlacement] {
        model.timelineComposition?.brollPlacements ?? []
    }

    private var readyAssets: [TimelineMediaAsset] {
        model.timelineAssetLibrary?.readyBroll ?? []
    }

    private var stage: BrollStage {
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
                        Label("Import B-roll video", systemImage: "video.badge.plus")
                    }
                    .accessibilityIdentifier("broll.importButton")

                    Text("Only video is supported -- there is no real backend authority for a still-image B-roll clip on this track.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Add existing to timeline") {
                    if readyAssets.isEmpty {
                        Text("No ready B-roll video yet.")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(readyAssets) { asset in
                            Button("\(Int(asset.durationSec))s B-roll") {
                                Task { await addToTimeline(asset) }
                            }
                            .disabled(model.baseEditAssetID == nil)
                        }
                        if model.baseEditAssetID == nil {
                            Text("Placing B-roll on the timeline needs this project's primary source registered first.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section("B-roll track") {
                    if placements.isEmpty {
                        Text("No B-roll placed on the timeline yet.")
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
                            .accessibilityIdentifier("broll.deleteButton")
                        }
                        .buttonStyle(.bordered)
                        .frame(minHeight: 44)
                    }

                    // Real, backend-accepted mutation -- the only editable
                    // field on a B-roll placement besides timing. B-roll is
                    // a full-frame visual replacement (never positioned or
                    // scaled), so this is the ONLY per-placement control
                    // this screen offers beyond Split/Delete.
                    Section("Audio mode") {
                        Picker("Audio mode", selection: audioModeBinding(for: selectedPlacement)) {
                            Text("Keep primary voice").tag(TimelineAudioMode.keepPrimaryVoice)
                            Text("Use B-roll's own audio").tag(TimelineAudioMode.useBrollAudio)
                            Text("Mute B-roll audio").tag(TimelineAudioMode.muteBrollAudio)
                        }
                        .pickerStyle(.menu)
                        .accessibilityIdentifier("broll.audioModePicker")
                    }
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
            .navigationTitle("B-roll")
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
            Label("No B-roll yet", systemImage: "film.stack")
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
        readyAssets.first { $0.assetID == placement.assetID } != nil ? "B-roll" : "B-roll (unavailable)"
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
        await model.importBrollVideo(fileURL: url)
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
