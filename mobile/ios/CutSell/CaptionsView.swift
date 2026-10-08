import SwiftUI

/// "Fix words": edits the text of one clip's caption. Opened by tapping the
/// caption on the video while the Editor v2 Captions panel is open
/// (CaptionsPanelView sets the whole-video look, typeface, position and size).
///
/// Connects ONLY to real, already-existing backend authority --
/// `DraftEditorViewModel.editCaption` (`/v1/draft-edits/captions`), which
/// flows through the SAME real draft/`autosave`/revision system as every
/// other draft mutation, and `/draft/undo`. ASR already ran upstream during
/// processing, so there is no separate captions-generation job here.
///
/// Text is always per-clip (`clip_id`-scoped): there is no real, coherent
/// "apply this same text to All clips" capability, so "All" is honestly
/// unavailable for text editing. A clip whose text is edited by hand is
/// exported as one caption for the whole clip (the server can no longer time
/// it word by word), which the preview shows the same way.
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

    private var clips: [[String: JSONValue]] { model.selectedClips }

    private var selectedClip: [String: JSONValue]? {
        guard let selectedClipID else { return nil }
        return clips.first { $0["clip_id"]?.stringValue == selectedClipID }
    }

    var body: some View {
        NavigationStack {
            Form {
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
            .navigationTitle("Fix words")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
            }
            .onAppear {
                // Only reads the real current state and seeds the clip/text
                // pickers from it; never changes a caption setting.
                selectedClipID = initialClipID ?? clips.first?["clip_id"]?.stringValue
                syncDraftText()
            }
            .alert("CutSell", isPresented: Binding(
                get: { model.errorMessage != nil },
                set: { if !$0 { model.errorMessage = nil } }
            )) { Button("OK", role: .cancel) {} } message: { Text(model.errorMessage ?? "") }
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
