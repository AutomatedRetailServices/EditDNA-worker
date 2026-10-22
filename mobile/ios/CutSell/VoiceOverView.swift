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
/// - Microphone RECORDING (`VoiceRecorder`, .m4a/AAC): the take is uploaded by
///   the SAME import path above and placed on the voice-over track at the
///   timeline cursor (`model.addRecordedVoiceOver`).
///
/// PENDING / honestly unsupported (documented here, never simulated):
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
    /// The timeline cursor when the sheet opened: a recorded take starts here.
    var playheadTime: Double = 0

    @Environment(\.dismiss) private var dismiss
    @State private var selectedPlacementID: String?
    @State private var splitTime: Double = 0
    @State private var isImporting = false
    @State private var justMutated = false
    @StateObject private var recorder = VoiceRecorder()
    @State private var isUsingTake = false
    @State private var recordNotice: String?

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

                Section(L10n.tr("Import")) {
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
                        Label(L10n.tr("Import audio file"), systemImage: "waveform.badge.plus")
                    }
                    .accessibilityIdentifier("voiceover.importButton")

                }

                Section(L10n.tr("Record")) {
                    recordSection
                }

                Section(L10n.tr("Add existing to timeline")) {
                    if readyAssets.isEmpty {
                        Text(L10n.tr("No ready voice-over audio yet."))
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(readyAssets) { asset in
                            Button(L10n.tr("\(Int(asset.durationSec))s voice-over")) {
                                Task { await addToTimeline(asset) }
                            }
                            .disabled(model.baseEditAssetID == nil)
                        }
                        if model.baseEditAssetID == nil {
                            Text(L10n.tr("This cut is still getting ready. Try again in a moment."))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section(L10n.tr("Voice-over track")) {
                    if placements.isEmpty {
                        Text(L10n.tr("No voice-over placed on the timeline yet."))
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(placements) { placement in
                            Button {
                                selectedPlacementID = placement.placementID
                                syncSplitTime()
                            } label: {
                                HStack {
                                    Text(L10n.tr(assetLabel(for: placement)))
                                    Spacer()
                                    Text(L10n.tr(String(format: "%.1fs–%.1fs", placement.timelineStartSec, placement.timelineEndSec)))
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
                    Section(L10n.tr("Split selected placement")) {
                        Slider(
                            value: $splitTime,
                            in: selectedPlacement.timelineStartSec...selectedPlacement.timelineEndSec
                        )
                        .accessibilityLabel(L10n.tr("Split point"))
                        .accessibilityValue("\(String(format: "%.1f", splitTime)) seconds")

                        HStack(spacing: 14) {
                            Button {
                                Task { await performSplit() }
                            } label: {
                                Label(L10n.tr("Split"), systemImage: "scissors")
                            }
                            .disabled(!canSplitAtSplitTime(selectedPlacement))

                            Button(role: .destructive) {
                                Task { await performDelete() }
                            } label: {
                                Label(L10n.tr("Delete"), systemImage: "trash")
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
                Section(L10n.tr("Audio (not yet supported)")) {
                    HStack {
                        Text(L10n.tr("Volume"))
                        Spacer()
                        Slider(value: .constant(1.0), in: 0...1).disabled(true)
                    }
                    HStack {
                        Text(L10n.tr("Fade in"))
                        Spacer()
                        Slider(value: .constant(0.0), in: 0...5).disabled(true)
                    }
                    HStack {
                        Text(L10n.tr("Fade out"))
                        Spacer()
                        Slider(value: .constant(0.0), in: 0...5).disabled(true)
                    }
                    HStack {
                        Text(L10n.tr("Pan"))
                        Spacer()
                        Slider(value: .constant(0.0), in: -1...1).disabled(true)
                    }
                    Text(L10n.tr("Change a voice-over's volume from its panel in the editor."))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .accessibilityIdentifier("voiceover.audioControlsUnsupported")
                }

                Section {
                    Button {
                        Task { await performUndo() }
                    } label: {
                        Label(L10n.tr("Undo"), systemImage: "arrow.uturn.backward")
                    }
                    .disabled(!canUndo)
                    .frame(minHeight: 44)
                }
            }
            .navigationTitle(L10n.tr("Voice-over"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button(L10n.tr("Done")) { dismiss() } }
            }
            .onAppear {
                selectedPlacementID = initialPlacementID ?? placements.first?.placementID
                syncSplitTime()
            }
            .onDisappear { recorder.discardTake() }
            .fileImporter(isPresented: $isImporting, allowedContentTypes: [.audio]) { result in
                switch result {
                case .success(let url):
                    Task { await performImport(url) }
                case .failure(let error):
                    model.errorMessage = error.localizedDescription
                }
            }
            .alert(L10n.tr("CutSell"), isPresented: Binding(
                get: { model.errorMessage != nil },
                set: { if !$0 { model.errorMessage = nil } }
            )) { Button(L10n.tr("OK"), role: .cancel) {} } message: { Text(L10n.tr(model.errorMessage ?? "")) }
        }
    }

    @ViewBuilder
    private var recordSection: some View {
        switch recorder.state {
        case .idle, .failed, .permissionDenied:
            Button {
                recordNotice = nil
                Task { await recorder.start() }
            } label: {
                Label(L10n.tr("Record"), systemImage: "mic.fill")
            }
            .accessibilityIdentifier("voiceover.recordButton")
            Text(L10n.tr("Starts at \(String(format: "%.1f", playheadTime))s on the timeline."))
                .font(.caption)
                .foregroundStyle(.secondary)
            if case .failed(let message) = recorder.state {
                Text(L10n.tr(message)).font(.caption).foregroundStyle(.red)
            }
            if recorder.state == .permissionDenied {
                Text(L10n.tr("CutSell can't use the microphone. Turn it on in Settings > CutSell > Microphone."))
                    .font(.caption)
                    .foregroundStyle(.red)
                    .accessibilityIdentifier("voiceover.micDenied")
            }
        case .recording:
            HStack {
                Image(systemName: "record.circle.fill").foregroundStyle(.red)
                Text(VoiceRecorder.formatted(recorder.elapsed))
                    .monospacedDigit()
                    .accessibilityIdentifier("voiceover.recordingTime")
                Spacer()
                Button(L10n.tr("Stop")) { recorder.stop() }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .accessibilityIdentifier("voiceover.stopButton")
            }
        case .recorded:
            HStack {
                Image(systemName: "waveform")
                Text(L10n.tr("Take: \(VoiceRecorder.formatted(recorder.elapsed))")).monospacedDigit()
            }
            HStack(spacing: 14) {
                Button {
                    Task {
                        recorder.discardTake()
                        await recorder.start()
                    }
                } label: {
                    Label(L10n.tr("Record again"), systemImage: "arrow.counterclockwise")
                }
                .accessibilityIdentifier("voiceover.recordAgainButton")
                Button {
                    Task { await useTake() }
                } label: {
                    if isUsingTake { ProgressView() } else { Label(L10n.tr("Use"), systemImage: "checkmark") }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isUsingTake)
                .accessibilityIdentifier("voiceover.useTakeButton")
            }
            .buttonStyle(.bordered)
        }
        if let recordNotice {
            Text(L10n.tr(recordNotice)).font(.caption).foregroundStyle(.secondary)
        }
    }

    private func useTake() async {
        guard let url = recorder.recordedURL else { return }
        isUsingTake = true
        defer { isUsingTake = false }
        justMutated = false
        switch await model.addRecordedVoiceOver(fileURL: url, at: playheadTime) {
        case .placed:
            recorder.finishTake()
            justMutated = true
            recordNotice = "Added to the voice-over track."
            selectedPlacementID = placements.last?.placementID
        case .savedNotPlaced:
            recorder.finishTake()
            recordNotice = "Saved. It can be placed on the timeline once this project's timeline is ready."
        case .failed:
            break   // the take is kept so "Use" can be tried again; the alert shows why
        }
    }

    @ViewBuilder
    private var stageRow: some View {
        switch stage {
        case .empty:
            Label(L10n.tr("No voice-over yet"), systemImage: "waveform")
                .foregroundStyle(.secondary)
        case .saving:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(L10n.tr("Saving…"))
            }
        case .ready:
            Label(L10n.tr("Ready"), systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .error:
            Label(L10n.tr("Error"), systemImage: "exclamationmark.triangle.fill")
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
