import SwiftUI

struct DraftEditorView: View {
    @EnvironmentObject private var appState: AppState
    let project: Project
    @StateObject private var holder = Holder()
    @State private var showCaptionsPanel = false
    /// Clip whose caption words are being fixed (tapped on the video).
    @State private var captionTextClipID: String?
    private static let previewAnchor = "editor.preview"

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
        .navigationTitle(project.title)
        .navigationBarTitleDisplayMode(.inline)
    }

    @ViewBuilder
    private func editor(_ model: DraftEditorViewModel) -> some View {
        VStack(spacing: 0) {
            if model.isSaving {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Saving…").font(.caption).foregroundStyle(.secondary)
                }
                .padding(.vertical, 6)
            }

            if model.isLoading {
                Spacer(); ProgressView("Loading timeline…"); Spacer()
            } else {
                ScrollViewReader { scroller in
                ScrollView {
                    VStack(spacing: 20) {
                        DraftPlaybackView(
                            model: model,
                            onCaptionTap: showCaptionsPanel ? { (clipID: String) in captionTextClipID = clipID } : nil
                        )
                        .id(Self.previewAnchor)

                        if showCaptionsPanel {
                            CaptionsPanelView(model: model) {
                                withAnimation(.easeOut(duration: 0.2)) { showCaptionsPanel = false }
                            }
                            .padding(.horizontal)
                            .transition(.opacity)
                        }

                        TimelineEditorView(model: model) {
                            withAnimation(.easeOut(duration: 0.2)) { showCaptionsPanel = true }
                        }

                        VisualTimelineView(model: model)

                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text("Clip order").font(.headline)
                                Spacer()
                                Text("Reorder").font(.caption).foregroundStyle(.secondary)
                            }
                            ReorderList(model: model)
                        }
                        .padding(.horizontal)

                        EditorExtrasView(model: model)

                        VStack(spacing: 10) {
                            Button {
                                Task { await model.export() }
                            } label: {
                                Label("Export 1080p", systemImage: "square.and.arrow.up")
                                    .font(.headline)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 8)
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(model.exportJob?.state == "rendering")

                            if let export = model.exportJob {
                                HStack {
                                    Text(export.state.replacingOccurrences(of: "_", with: " ").capitalized)
                                    Spacer()
                                    if let progress = export.progress { Text("\(progress)%").monospacedDigit() }
                                }
                                .font(.subheadline)

                                if let urlString = export.result?["download_url"]?.stringValue,
                                   let url = URL(string: urlString) {
                                    FinishedExportActionsView(remoteURL: url)
                                }
                            }
                        }
                        .padding()
                    }
                    .padding(.vertical)
                }
                // While the panel is open, a finger on the video moves the
                // caption instead of scrolling the page (Done closes it).
                .scrollDisabled(showCaptionsPanel)
                .onChange(of: showCaptionsPanel) { _, isOpen in
                    // The Captions button sits below; bring the video and the
                    // panel into view so every change is seen.
                    guard isOpen else { return }
                    withAnimation { scroller.scrollTo(Self.previewAnchor, anchor: .top) }
                }
                }
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .bottomBar) {
                Button { Task { await model.undo() } } label: { Label("Undo", systemImage: "arrow.uturn.backward") }
                Spacer()
                Button { Task { await model.redo() } } label: { Label("Redo", systemImage: "arrow.uturn.forward") }
            }
        }
        .sheet(isPresented: Binding(
            get: { captionTextClipID != nil },
            set: { if !$0 { captionTextClipID = nil } }
        )) {
            CaptionsView(model: model, initialClipID: captionTextClipID)
        }
        .alert("CutSell", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) { Button("OK", role: .cancel) {} } message: { Text(model.errorMessage ?? "") }
    }
}

private struct ReorderList: View {
    @ObservedObject var model: DraftEditorViewModel
    @State private var localIDs: [String] = []

    var body: some View {
        VStack(spacing: 6) {
            ForEach(Array(localIDs.enumerated()), id: \.element) { index, clipID in
                let clip = model.selectedClips.first(where: { $0["clip_id"]?.stringValue == clipID })
                HStack(spacing: 10) {
                    Image(systemName: "line.3.horizontal")
                        .foregroundStyle(.secondary)
                    Text("\(index + 1)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 22)
                    Text(clip?["caption_text"]?.stringValue ?? clip?["text"]?.stringValue ?? "Clip")
                        .lineLimit(1)
                    Spacer()
                    HStack(spacing: 4) {
                        Button { move(index, by: -1) } label: { Image(systemName: "arrow.up") }
                            .disabled(index == 0)
                        Button { move(index, by: 1) } label: { Image(systemName: "arrow.down") }
                            .disabled(index == localIDs.count - 1)
                    }
                    .buttonStyle(.borderless)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 9)
                .background(.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
            }
        }
        .onAppear { sync() }
        .onChange(of: model.selectedClips.count) { _, _ in sync() }
    }

    private func sync() {
        localIDs = model.selectedClips.compactMap { $0["clip_id"]?.stringValue }
    }

    private func move(_ index: Int, by delta: Int) {
        let destination = index + delta
        guard localIDs.indices.contains(index), localIDs.indices.contains(destination) else { return }
        localIDs.swapAt(index, destination)
        Task { await model.reorder(clipIDs: localIDs) }
    }
}

@MainActor
private final class Holder: ObservableObject {
    @Published var model: DraftEditorViewModel?
}
