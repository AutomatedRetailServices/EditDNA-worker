import Combine
import SwiftUI

struct DraftEditorView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    let project: Project
    @StateObject private var holder = Holder()
    @State private var showCaptionsPanel = false
    /// Clip whose caption words are being fixed (tapped on the video).
    @State private var captionTextClipID: String?
    private static let previewAnchor = "editor.preview"

    /// Debug builds only: the editor probe opens a real cut in one Editor v4 state
    /// (`CUTSELL_PROBE_STATE`), so each Figma screen can be shown on a real cut. Release: nil.
    static var probeState: EditorPreviewState? {
        #if DEBUG
        guard let name = ProcessInfo.processInfo.environment["CUTSELL_PROBE_STATE"] else { return nil }
        switch name {
        case "caption_edit": return EditorPreviewState(panel: .captions("first"), probe: name)
        case "vo_record": return EditorPreviewState(panel: .voiceRecord, probe: name)
        case "vo_review": return EditorPreviewState(panel: .voiceReview, probe: name)
        case "cap_highlight", "cap_box", "cap_fonts": return EditorPreviewState(panel: .captions(nil), probe: name)
        case "overlay_gallery": return EditorPreviewState(sheet: .overlay, probe: name)
        case "overlay_selected": return EditorPreviewState(selection: .overlay("first_media", isText: false), probe: name)
        case "mic": return EditorPreviewState(panel: .micPermission, probe: name)
        case "style": return EditorPreviewState(panel: .style, probe: name)
        case "stress", "flow": return EditorPreviewState(probe: name)
        default: return EditorPreviewState(probe: name)
        }
        #else
        return nil
        #endif
    }

    var body: some View {
        Group {
            if let model = holder.model, !model.isLoading, model.snapshot != nil {
                // Editor v4 (Figma 04_Editor "Editor v4", approved 13 Oct 2026).
                EditorV4View(model: model, project: project, previewState: Self.probeState)
            } else if let model = holder.model, !model.isLoading, model.snapshot == nil, model.errorMessage != nil {
                VStack(spacing: 14) {
                    Text(L10n.tr("Couldn’t open this cut")).font(CS.inter(18, .semibold)).foregroundStyle(.white)
                    Text(L10n.tr(model.errorMessage.map { DraftEditorViewModel.plainWords($0) } ?? "Please try again.")).font(CS.inter(13)).foregroundStyle(CS.textSecondaryF)
                        .multilineTextAlignment(.center)
                    PrimaryButton(title: "Try again", height: 48, cornerRadius: 14, fontSize: 15) {
                        Task { await model.load() }
                    }
                    .frame(width: 200)
                }
                .padding(24)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(hex: 0x02040A).ignoresSafeArea())
            } else {
                ZStack {
                    Color(hex: 0x02040A).ignoresSafeArea()
                    VStack(spacing: 14) {
                        BrandRing(size: 84)
                        Text(L10n.tr("Opening your cut…")).font(CS.inter(14, .medium)).foregroundStyle(CS.textSecondaryF)
                    }
                }
            }
        }
        // The load belongs to the whole screen. It used to hang off the "Opening draft…" spinner:
        // setting the model removed the spinner, iOS cancelled its task, and the draft request
        // was cut off (Render log: 499 on /draft and /timeline-assets) -- the editor never loaded.
        .task(id: project.projectID) {
            guard holder.model == nil, let session = appState.session else { return }
            let model = DraftEditorViewModel(project: project, session: session)
            holder.model = model
            await model.load()
            // 22 Oct: a copy the server removed (3 days unused) is made again from the original in Photos.
            await SourceRestore.shared.restoreIfNeeded(model: model)
        }
        .overlay { RestoreBanner(projectID: project.projectID) }
        // The editor has its own header (back · name · 1080p HD · Export): never the system bar
        // on top of it (build 14 showed a second, large "‹ Cut 14 Oct" bar).
        .toolbar(.hidden, for: .navigationBar)
        .overlay(alignment: .topLeading) {
            // While it opens (or if it can't), a plain way back.
            if !(holder.model.map { !$0.isLoading && $0.snapshot != nil } ?? false) {
                Button { dismiss() } label: {
                    Image(systemName: "chevron.left").font(.system(size: 17, weight: .semibold)).foregroundStyle(.white)
                        .frame(width: 44, height: 44)
                }
                .accessibilityLabel(L10n.tr("Back"))
                .padding(.leading, 8)
            }
        }
    }

    @ViewBuilder
    private func editor(_ model: DraftEditorViewModel) -> some View {
        VStack(spacing: 0) {
            if model.isSaving {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(L10n.tr("Saving…")).font(.caption).foregroundStyle(.secondary)
                }
                .padding(.vertical, 6)
            }

            if model.isLoading {
                Spacer(); ProgressView(L10n.tr("Loading timeline…")); Spacer()
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
                                Text(L10n.tr("Clip order")).font(.headline)
                                Spacer()
                                Text(L10n.tr("Reorder")).font(.caption).foregroundStyle(.secondary)
                            }
                            ReorderList(model: model)
                        }
                        .padding(.horizontal)

                        EditorExtrasView(model: model)

                        VStack(spacing: 10) {
                            Button {
                                Task { await model.export() }
                            } label: {
                                Label(L10n.tr("Export 1080p"), systemImage: "square.and.arrow.up")
                                    .font(.headline)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 8)
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(model.exportJob?.state == "rendering")

                            if let export = model.exportJob {
                                HStack {
                                    Text(L10n.tr(export.state.replacingOccurrences(of: "_", with: " ").capitalized))
                                    Spacer()
                                    if let progress = export.progress { Text(L10n.tr("\(progress)%")).monospacedDigit() }
                                }
                                .font(.subheadline)

                                if let urlString = export.result?["download_url"]?.stringValue,
                                   let url = URL(string: urlString) {
                                    FinishedExportActionsView(remoteURL: url, projectID: model.project.projectID, session: model.session)
                                }
                            }
                        }
                        .padding()
                    }
                    .padding(.vertical)
                    // Exactly the screen width: nothing may push the editor past the edges.
                    .containerRelativeFrame(.horizontal)
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
        // Undo / Redo live once, in the timeline's tool bar (no floating copies over the content).
        .navigationTitle(model.videoTitle ?? project.title)
        .sheet(isPresented: Binding(
            get: { captionTextClipID != nil },
            set: { if !$0 { captionTextClipID = nil } }
        )) {
            CaptionsView(model: model, initialClipID: captionTextClipID)
        }
        .alert(L10n.tr("CutSell"), isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) { Button(L10n.tr("OK"), role: .cancel) {} } message: { Text(L10n.tr(model.errorMessage ?? "")) }
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
                    Text(L10n.tr("\(index + 1)"))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 22)
                    Text(L10n.tr(clip?["caption_text"]?.stringValue ?? clip?["text"]?.stringValue ?? "Clip"))
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
/// Keeps the editor's model and passes ITS changes on to the editor screen. The screen only
/// watches this holder; without forwarding, "Loading timeline…" stayed on screen forever even
/// after the draft had loaded (the screen never heard that isLoading/snapshot changed).
private final class Holder: ObservableObject {
    @Published var model: DraftEditorViewModel? {
        didSet {
            forward = model?.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        }
    }
    private var forward: AnyCancellable?
}

/// "Restoring video… 40%" over the editor while a removed copy is made again; the reason when it can't be.
private struct RestoreBanner: View {
    let projectID: String
    @ObservedObject private var restore = SourceRestore.shared

    var body: some View {
        if let value = restore.progress[projectID] {
            pill {
                ProgressView().controlSize(.small).tint(.white)
                Text(L10n.tr("Restoring video…")).font(CS.inter(13, .medium)).foregroundStyle(.white)
                Text(verbatim: "\(Int((value * 100).rounded()))%").font(CS.inter(13, .semibold)).monospacedDigit().foregroundStyle(CS.cyan)
            }
            .accessibilityIdentifier("restoringVideo")
        } else if let problem = restore.problem[projectID] {
            pill {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(Color(hex: 0xFFD60A))
                Text(L10n.tr(problem)).font(CS.inter(12, .medium)).foregroundStyle(.white).lineLimit(2)
            }
            .accessibilityIdentifier("restoreProblem")
        }
    }

    private func pill<C: View>(@ViewBuilder _ content: () -> C) -> some View {
        HStack(spacing: 8) { content() }
            .padding(.horizontal, 14).padding(.vertical, 9)
            .background(Color(hex: 0x040812, opacity: 0.92), in: Capsule())
            .overlay(Capsule().stroke(CS.borderDefault, lineWidth: 1))
            .frame(maxHeight: .infinity, alignment: .top)
            .padding(.top, 64)
            .padding(.horizontal, 16)
    }
}
