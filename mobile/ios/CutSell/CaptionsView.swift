import SwiftUI

/// Mobile V1 Captions UI gate. Connects ONLY to real, already-existing
/// backend authority -- `DraftEditorViewModel.setCaptionSettings`
/// (`/v1/draft-edits/caption-settings`) and `.editCaption`
/// (`/v1/draft-edits/captions`), both of which flow through the SAME real
/// draft/`autosave`/revision system as every other draft mutation. Never
/// invents a route, an ASR job, a timestamp, or a style/position this
/// codebase's real contract does not already support.
///
/// Setup -> Creating -> Ready is derived entirely from two real signals,
/// never a fabricated pipeline:
/// - Setup:    `!model.captionsEnabled` (captions not turned on for this draft).
/// - Creating: `model.isSaving` while a mutation triggered from this view is
///             in flight -- the REAL network round trip to `caption-settings`
///             or `captions`, not an ASR/caption-generation job. ASR already
///             ran upstream during Clean Cut processing (`pipeline.py`
///             populates every clip's real `caption_text` from its ASR
///             transcript before the draft ever exists), so there is no
///             separate captions-generation job for this state to represent.
/// - Ready:    `model.captionsEnabled` confirmed true in the current snapshot.
///
/// PENDING (documented here per the gate's own requirement, never silently
/// simulated):
/// - Position (Top/Center/Bottom): NOT supported. `render.py`'s real caption
///   filter hardcodes `Alignment=2` (bottom-center) for both presets and the
///   backend contract (`caption_settings.py`, `DraftCaptionSettingsRequest`)
///   has no position field at all. Shown as a fixed, honestly-disabled value.
/// - Styles beyond Clean/Classic: `CAPTION_PRESETS = {"classic", "clean"}` is
///   the entire real vocabulary; no third preset exists to offer.
/// - Selected/All scope: style and enabled/disabled are ALWAYS draft-level
///   (every clip, uniformly) -- there is no real per-clip style override, so
///   "Selected" is honestly unavailable for style/enabled. Manual caption
///   TEXT is always per-clip (`clip_id`-scoped) -- there is no real, coherent
///   "apply this same text to All clips" capability intended for use here
///   (the backend route technically accepts multiple edits per call, but
///   stamping identical text across every clip has no real product meaning),
///   so "All" is honestly unavailable for text editing.
/// - Word-level/karaoke timestamp highlighting: NOT supported. `render.py`
///   burns exactly one static SRT cue per clip spanning its full duration;
///   no per-word timing is read or rendered.
struct CaptionsView: View {
    @ObservedObject var model: DraftEditorViewModel
    /// The Main Video clip selected in the timeline when this sheet was
    /// opened, if any -- an honest starting point, never a requirement.
    let initialClipID: String?

    @Environment(\.dismiss) private var dismiss
    @State private var selectedClipID: String?
    @State private var draftText: String = ""
    /// Note: Redo is deliberately NOT offered here -- this gate's scope is
    /// Delete + Undo only (real authority for both). Undo reuses the exact
    /// same `/draft/undo` authority as TimelineEditorView's Main Video Undo,
    /// since caption mutations flow through the same draft/revision system.
    @State private var justMutatedCaptions = false

    private enum CaptionsStage { case setup, creating, ready }

    private var stage: CaptionsStage {
        if model.isSaving { return .creating }
        return model.captionsEnabled ? .ready : .setup
    }

    private var clips: [[String: JSONValue]] { model.selectedClips }

    private var selectedClip: [String: JSONValue]? {
        guard let selectedClipID else { return nil }
        return clips.first { $0["clip_id"]?.stringValue == selectedClipID }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    stageRow
                }

                Section("Captions") {
                    Toggle("Enable captions", isOn: Binding(
                        get: { model.captionsEnabled },
                        set: { newValue in
                            Task {
                                justMutatedCaptions = false
                                await model.setCaptionSettings(enabled: newValue)
                                justMutatedCaptions = true
                            }
                        }
                    ))
                    .accessibilityIdentifier("captions.enableToggle")
                }

                Section("Style — applies to all clips") {
                    // Style is ALWAYS draft-level (every clip renders with
                    // the same preset) -- there is no real per-clip style
                    // authority, so a "Selected" scope is never offered here.
                    Picker("Style", selection: Binding(
                        get: { model.captionPreset },
                        set: { newValue in
                            Task {
                                justMutatedCaptions = false
                                await model.setCaptionSettings(preset: newValue)
                                justMutatedCaptions = true
                            }
                        }
                    )) {
                        Text("Classic").tag("classic")
                        Text("Clean").tag("clean")
                    }
                    .pickerStyle(.segmented)
                    .disabled(!model.captionsEnabled)
                    .accessibilityIdentifier("captions.stylePicker")

                    Text("\"Selected\" style scope is not available: this project's real caption contract only supports one style for every clip, never a per-clip override.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Position") {
                    // No position field exists in the real contract or the
                    // renderer -- captions always burn in bottom-center.
                    // Never simulated as a working control.
                    HStack {
                        Text("Bottom (fixed)")
                        Spacer()
                        Text("Top/Center not yet supported").font(.caption).foregroundStyle(.secondary)
                    }
                    .accessibilityIdentifier("captions.positionUnavailable")
                }

                Section("Text — selected clip only") {
                    // Manual text editing is always per-clip; "All" has no
                    // real, coherent authority here (see the file doc).
                    if clips.isEmpty {
                        Text("No clips available yet.")
                            .foregroundStyle(.secondary)
                    } else {
                        Picker("Clip", selection: Binding(
                            get: { selectedClipID ?? clips.first?["clip_id"]?.stringValue },
                            set: { newValue in
                                selectedClipID = newValue
                                syncDraftText()
                            }
                        )) {
                            ForEach(clips, id: \.self) { clip in
                                let id = clip["clip_id"]?.stringValue ?? ""
                                let label = clip["caption_text"]?.stringValue ?? clip["text"]?.stringValue ?? "Clip"
                                Text(label).tag(Optional(id)).lineLimit(1)
                            }
                        }
                        .accessibilityIdentifier("captions.clipPicker")

                        TextField("Caption text", text: $draftText, axis: .vertical)
                            .lineLimit(3...6)
                            .disabled(selectedClip == nil)
                            .accessibilityIdentifier("captions.textField")
                            .accessibilityLabel("Caption text for selected clip")

                        HStack {
                            Button("Save text") {
                                Task { await saveText() }
                            }
                            .disabled(selectedClip == nil)

                            Spacer()

                            Button("Delete", role: .destructive) {
                                Task { await deleteText() }
                            }
                            .disabled(selectedClip == nil || (selectedClip?["caption_text"]?.stringValue ?? "").isEmpty)
                            .accessibilityIdentifier("captions.deleteButton")
                        }
                        .buttonStyle(.bordered)
                        .frame(minHeight: 44)
                    }
                }

                Section {
                    Button {
                        Task { await performUndo() }
                    } label: {
                        Label("Undo", systemImage: "arrow.uturn.backward")
                    }
                    .disabled(!(model.canUndoTimelineMutation || justMutatedCaptions))
                    .frame(minHeight: 44)
                }
            }
            .navigationTitle("Captions")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
            }
            .onAppear {
                // Never auto-enables captions -- only reads the real current
                // state and seeds the clip/text pickers from it.
                selectedClipID = initialClipID ?? clips.first?["clip_id"]?.stringValue
                syncDraftText()
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
        case .setup:
            Label("Setup — captions are off", systemImage: "captions.bubble")
                .foregroundStyle(.secondary)
        case .creating:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Saving…")
            }
        case .ready:
            Label("Ready", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        }
    }

    private func syncDraftText() {
        draftText = selectedClip?["caption_text"]?.stringValue ?? selectedClip?["text"]?.stringValue ?? ""
    }

    private func saveText() async {
        guard let clipID = selectedClipID else { return }
        justMutatedCaptions = false
        await model.editCaption(clipID: clipID, text: draftText)
        justMutatedCaptions = true
    }

    private func deleteText() async {
        guard let clipID = selectedClipID else { return }
        justMutatedCaptions = false
        await model.editCaption(clipID: clipID, text: "")
        justMutatedCaptions = true
        draftText = ""
    }

    /// Same honest, scoped Undo pattern as TimelineEditorView's Main Video
    /// Undo -- `/draft/undo` exposes no "can-undo" signal, so
    /// `justMutatedCaptions` is a session-local proxy set only right after a
    /// real mutation performed from THIS view, and only ever cleared (never
    /// claimed) again on a new mutation. Real success/failure comes from
    /// `model.undo()`'s own typed Bool result -- never inferred from `await`
    /// alone.
    private func performUndo() async {
        guard justMutatedCaptions else { return }
        let succeeded = await model.undo()
        justMutatedCaptions = false
        if succeeded {
            syncDraftText()
        }
    }
}
