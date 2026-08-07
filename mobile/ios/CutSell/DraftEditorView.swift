import SwiftUI

struct DraftEditorView: View {
    @EnvironmentObject private var appState: AppState
    let project: Project
    @StateObject private var holder = Holder()

    var body: some View {
        Group {
            if let model = holder.model {
                editor(model)
            } else {
                ProgressView("Opening draft…")
                    .task {
                        guard let session = appState.session else { return }
                        let model = DraftEditorViewModel(project: project, session: session)
                        holder.model = model
                        await model.load()
                    }
            }
        }
    }

    @ViewBuilder
    private func editor(_ model: DraftEditorViewModel) -> some View {
        VStack(spacing: 0) {
            if model.isSaving {
                ProgressView().controlSize(.small).padding(.top, 6)
            }

            if model.isLoading {
                Spacer(); ProgressView("Loading timeline…"); Spacer()
            } else {
                List {
                    Section {
                        ForEach(Array(model.selectedClips.enumerated()), id: \.offset) { index, clip in
                            ClipRow(index: index, clip: clip) {
                                if let clipID = clip["clip_id"]?.stringValue {
                                    Task { await model.remove(clipID: clipID) }
                                }
                            } audioAction: { muted, volume in
                                if let clipID = clip["clip_id"]?.stringValue {
                                    Task { await model.setAudio(clipID: clipID, muted: muted, volume: volume) }
                                }
                            }
                        }
                        .onMove { source, destination in
                            var ids = model.selectedClips.compactMap { $0["clip_id"]?.stringValue }
                            ids.move(fromOffsets: source, toOffset: destination)
                            Task { await model.reorder(clipIDs: ids) }
                        }
                    } header: {
                        HStack {
                            Text("Timeline")
                            Spacer()
                            Text("\(model.selectedClips.count) clips")
                        }
                    }

                    Section("Captions") {
                        Toggle("Show captions", isOn: Binding(
                            get: { model.captionsEnabled },
                            set: { value in Task { await model.setCaptionSettings(enabled: value) } }
                        ))
                        Picker("Style", selection: Binding(
                            get: { model.captionPreset },
                            set: { value in Task { await model.setCaptionSettings(preset: value) } }
                        )) {
                            Text("Classic").tag("classic")
                            Text("Clean").tag("clean")
                        }
                    }

                    Section {
                        Button {
                            Task { await model.export() }
                        } label: {
                            Label("Export 1080p", systemImage: "square.and.arrow.up")
                                .frame(maxWidth: .infinity)
                        }
                        .disabled(model.exportJob?.state == "rendering")

                        if let export = model.exportJob {
                            HStack {
                                Text(export.state.replacingOccurrences(of: "_", with: " ").capitalized)
                                Spacer()
                                if let progress = export.progress { Text("\(progress)%").monospacedDigit() }
                            }
                            if let urlString = export.result?["download_url"]?.stringValue,
                               let url = URL(string: urlString) {
                                Link("Open finished video", destination: url)
                            }
                        }
                    }
                }
                .environment(\.editMode, .constant(.active))
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .bottomBar) {
                Button { Task { await model.undo() } } label: { Label("Undo", systemImage: "arrow.uturn.backward") }
                Spacer()
                Button { Task { await model.redo() } } label: { Label("Redo", systemImage: "arrow.uturn.forward") }
            }
        }
        .alert("CutSell", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) { Button("OK", role: .cancel) {} } message: { Text(model.errorMessage ?? "") }
    }
}

private struct ClipRow: View {
    let index: Int
    let clip: [String: JSONValue]
    let deleteAction: () -> Void
    let audioAction: (Bool?, Double?) -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "line.3.horizontal").foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 4) {
                Text(clip["caption_text"]?.stringValue ?? clip["text"]?.stringValue ?? "Clip")
                    .lineLimit(3)
                HStack(spacing: 8) {
                    Text("#\(index + 1)")
                    if let role = clip["semantic_role"]?.stringValue { Text(role.capitalized) }
                    if let start = clip["start"]?.doubleValue, let end = clip["end"]?.doubleValue {
                        Text(String(format: "%.1fs", max(0, end - start)))
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer()
            Menu {
                Button("Mute") { audioAction(true, nil) }
                Button("Unmute") { audioAction(false, nil) }
                Button("Volume 50%") { audioAction(nil, 0.5) }
                Button("Volume 100%") { audioAction(nil, 1.0) }
                Divider()
                Button("Delete clip", role: .destructive, action: deleteAction)
            } label: {
                Image(systemName: "ellipsis.circle")
            }
        }
        .padding(.vertical, 4)
    }
}

@MainActor
private final class Holder: ObservableObject {
    @Published var model: DraftEditorViewModel?
}
