import SwiftUI
import UniformTypeIdentifiers

/// Mobile V1 Voice-over UI gate. Connects ONLY to real, already-existing
/// backend authority -- never an invented route, a simulated volume/fade/
/// pan control, or a fabricated recording capability.
///
/// Real authority used here:
/// - `TimelineAssetRegistryClient.list` (`GET /timeline-assets`) for the
///   asset library; `readyVoiceOvers` is D-279's own real READY/VOICE_OVER/
///   AUDIO filter.
/// - `VoiceOverImportManager.importAudio` (`POST /timeline-uploads` then
///   `POST /timeline-assets`) to import an EXISTING local audio file as a
///   new asset. Ingestion is synchronous: the backend returns the asset
///   already READY or FAILED (`create_voice_over_asset` probes real audio
///   presence/duration immediately) -- never a fabricated polling/QUALIFYING
///   step.
/// - `model.addVoiceOverPlacement` / `.removeVoiceOverPlacement` /
///   `.splitVoiceOverPlacement` (all real D-282A `PUT /timeline` mutations).
/// - `model.canUndoTimelineMutation` / `.undoLastTimelineMutation()` (the
///   same real captured-pre-mutation-snapshot Undo already used for
///   Overlay placements -- Voice-over shares the identical authority).
///
/// PENDING / honestly unsupported (documented here, never simulated):
/// - Microphone RECORDING: not implemented anywhere in this codebase.
///   `cutsell_worker/timeline_asset_registry_store.py`'s own
///   `create_voice_over_asset` docstring states this directly -- Import
///   (an already-recorded/existing audio file) is the only real "add new
///   audio" capability today.
/// - Volume / fade-in / fade-out / pan: `VoiceOverPlacementModel`
///   (`cutsell_app/timeline_routes.py`) carries exactly `placement_id,
///   asset_id, timeline_start_sec, timeline_end_sec, source_in_sec,
///   source_out_sec, transcript_reference` -- no audio-shaping field at
///   all. `timeline_composition.py`'s own `VoiceOverPlacement` docstring
///   confirms the V1 audio behavior (muting the primary voice under the
///   placement) is fixed and not configurable. Every control below for
///   these is shown, honestly disabled, never wired to a fake mutation.
struct VoiceOverView: View {
    @ObservedObject var model: DraftEditorViewModel
    let initialPlacementID: String?

    @Environment(\.dismiss) private var dismiss
    @State private var selectedPlacementID: String?
    @State private var splitTime: Double = 0
    @State private var isImporting = false
    @State private var justMutated = false

    private enum VoiceOverStage { case empty, saving, ready, error }

    private var placements: [VoiceOverPlacement] {
        model.timelineComposition?.voiceOverPlacements ?? []
    }

    private var readyAssets: [TimelineMediaAsset] {
        model.timelineAssetLibrary?.readyVoiceOvers ?? []
    }

    private var stage: VoiceOverStage {
        if model.errorMessage != nil { return .error }
        if model.isSavingTimeline { return .saving }
        if placements.isEmpty && readyAssets.isEmpty { return .empty }
        return .ready
    }

    private var selectedPlacement: VoiceOverPlacement? {
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
                    // live microphone capture -- see the file doc for why
                    // recording has no real authority yet.
                    Button {
                        isImporting = true
                    } label: {
                        Label("Import audio file", systemImage: "waveform.badge.plus")
                    }
                    .accessibilityIdentifier("voiceover.importButton")

                    Label("Record (not available)", systemImage: "mic.slash")
                        .foregroundStyle(.secondary)
                        .accessibilityIdentifier("voiceover.recordUnavailable")
                }

                Section("Add existing to timeline") {
                    if readyAssets.isEmpty {
                        Text("No ready voice-over audio yet.")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(readyAssets) { asset in
                            Button("\(Int(asset.durationSec))s voice-over") {
                                Task { await addToTimeline(asset) }
                            }
                            .disabled(model.baseEditAssetID == nil)
                        }
                        if model.baseEditAssetID == nil {
                            Text("Placing voice-over on the timeline needs this project's primary source registered first.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section("Voice-over track") {
                    if placements.isEmpty {
                        Text("No voice-over placed on the timeline yet.")
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
                            .accessibilityIdentifier("voiceover.deleteButton")
                        }
                        .buttonStyle(.bordered)
                        .frame(minHeight: 44)
                    }
                }

                // Audio-shaping controls -- honestly disabled: no real
                // backend field exists for any of these on a voice-over
                // placement (see the file doc).
                Section("Audio (not yet supported)") {
                    HStack {
                        Text("Volume")
                        Spacer()
                        Slider(value: .constant(1.0), in: 0...1).disabled(true)
                    }
                    HStack {
                        Text("Fade in")
                        Spacer()
                        Slider(value: .constant(0.0), in: 0...5).disabled(true)
                    }
                    HStack {
                        Text("Fade out")
                        Spacer()
                        Slider(value: .constant(0.0), in: 0...5).disabled(true)
                    }
                    HStack {
                        Text("Pan")
                        Spacer()
                        Slider(value: .constant(0.0), in: -1...1).disabled(true)
                    }
                    Text("The current backend contract carries no volume, fade, or pan field for voice-over placements -- these controls are shown disabled rather than simulated.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .accessibilityIdentifier("voiceover.audioControlsUnsupported")
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
            .navigationTitle("Voice-over")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
            }
            .onAppear {
                selectedPlacementID = initialPlacementID ?? placements.first?.placementID
                syncSplitTime()
            }
            .fileImporter(isPresented: $isImporting, allowedContentTypes: [.audio]) { result in
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
            Label("No voice-over yet", systemImage: "waveform")
                .foregroundStyle(.secondary)
        case .saving:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Saving…")
            }
        case .ready:
            Label("Ready", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .error:
            Label("Error", systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
        }
    }

    private func assetLabel(for placement: VoiceOverPlacement) -> String {
        readyAssets.first { $0.assetID == placement.assetID } != nil ? "Voice-over" : "Voice-over (unavailable)"
    }

    private func syncSplitTime() {
        guard let selectedPlacement else { return }
        splitTime = (selectedPlacement.timelineStartSec + selectedPlacement.timelineEndSec) / 2
    }

    private func canSplitAtSplitTime(_ placement: VoiceOverPlacement) -> Bool {
        splitTime > placement.timelineStartSec + 0.05 && splitTime < placement.timelineEndSec - 0.05
    }

    private func addToTimeline(_ asset: TimelineMediaAsset) async {
        let end = (model.timelineComposition?.timelineDurationSec ?? 0) + asset.durationSec
        let start = model.timelineComposition?.timelineDurationSec ?? 0
        justMutated = false
        await model.addVoiceOverPlacement(
            assetID: asset.assetID, start: start, end: end, sourceIn: 0, sourceOut: asset.durationSec
        )
        justMutated = true
    }

    private func performSplit() async {
        guard let id = selectedPlacementID, let placement = selectedPlacement, canSplitAtSplitTime(placement) else { return }
        justMutated = false
        await model.splitVoiceOverPlacement(id: id, at: splitTime)
        justMutated = true
    }

    private func performDelete() async {
        guard let id = selectedPlacementID else { return }
        justMutated = false
        await model.removeVoiceOverPlacement(id: id)
        justMutated = true
        selectedPlacementID = nil
    }

    private func performImport(_ url: URL) async {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        await model.importVoiceOverAudio(fileURL: url)
    }

    /// Same honest, scoped Undo pattern already established for Main Video
    /// and Captions: `undoLastTimelineMutation()` only clears its captured
    /// snapshot on a confirmed successful re-save (a real typed-Bool result
    /// internally) -- a failed revert leaves the snapshot intact so the
    /// user can retry, never inferred from `await` alone.
    private func performUndo() async {
        if model.canUndoTimelineMutation {
            await model.undoLastTimelineMutation()
            justMutated = false
        }
    }
}
