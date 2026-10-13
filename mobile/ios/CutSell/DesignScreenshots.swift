import SwiftUI

/// Debug builds only: `-CutSellScreen <name>` opens one redesigned screen with sample content,
/// without network, camera or photo library, so CI can take the screenshots placed next to Figma.
/// Release (TestFlight / App Store) builds never read this argument.
enum DesignScreenshots {
    static var requestedScreen: String? {
        #if DEBUG
        let args = ProcessInfo.processInfo.arguments
        if let index = args.firstIndex(of: "-CutSellScreen"), index + 1 < args.count {
            return args[index + 1]
        }
        #endif
        return nil
    }

    static let all = [
        "editor", "editor_sel_video", "editor_sel_voice", "editor_sel_overlay", "editor_captions",
        "editor_caption_deleted", "editor_voiceover", "editor_split",
        "hub", "processing", "ready", "camera", "gallery_single", "gallery_bulk", "gallery_combine", "style", "gallery_access",
        "perm_camera", "perm_mic", "edits", "edits_empty", "you", "plan",
        "err_unsupported", "err_too_large", "err_too_long", "err_clip_limit", "err_could_not_add", "err_formats",
    ]
}

#if DEBUG
struct DesignScreenshotRoot: View {
    let screen: String
    @StateObject private var appState = AppState()
    @StateObject private var notificationCenter = NotificationCenterModel()

    private static let durations: [Double] = [5, 12, 10, 8, 15, 7, 14, 11, 9, 20, 6, 13]

    var body: some View {
        content
            .environmentObject(appState)
            .environmentObject(notificationCenter)
            .preferredColorScheme(.dark)
            .onAppear {
                appState.isBootstrapping = false
                if screen == "edits" || screen == "hub" { appState.projects = Self.sampleProjects }
            }
    }

    @ViewBuilder private var content: some View {
        switch screen {
        case "hub":
            HomeHubView(onUpload: { _ in }, onRecord: {}, onOpenProject: { _ in }, selectTab: { _ in })
        case "processing":
            ProcessingContent(
                stage: CutTracker.Stage(step: 6, title: "Structuring your cut", percent: 68),
                fileName: "IMG_4821.MOV", thumbnail: nil, onClose: {}, onOpenEdits: {}
            )
            .background(CS.base.ignoresSafeArea())
        case "ready":
            ReadyContent(
                cut: CutTracker.Cut(id: "sample", title: "Sample edit", fileName: "IMG_4821.MOV", footageSeconds: 366),
                project: Self.sampleProjects[0], onBack: {}, onReview: {}
            )
            .background(CS.base.ignoresSafeArea())
        case "editor_probe":
            EditorProbe()
        case "editor":
            SampleEditor(state: EditorPreviewState())
        case "editor_sel_video":
            SampleEditor(state: EditorPreviewState(selection: .video("c2")))
        case "editor_sel_voice":
            SampleEditor(state: EditorPreviewState(selection: .voiceOver("v1")))
        case "editor_sel_overlay":
            SampleEditor(state: EditorPreviewState(selection: .overlay("t1", isText: true)))
        case "editor_captions":
            SampleEditor(state: EditorPreviewState(sheet: .captions))
        case "editor_caption_deleted":
            SampleEditor(state: EditorPreviewState(toast: .captionDeleted), captions: true)
        case "editor_voiceover":
            SampleEditor(state: EditorPreviewState(sheet: .voiceOver))
        case "editor_split":
            SampleEditor(state: EditorPreviewState(selection: .video("c2"), toast: .split("00:06.0")))
        case "camera":
            CreateCameraView(onBack: {}, onUpload: {}, onRecorded: { _ in }, previewMode: true)
        case "gallery_single":
            gallery(GalleryPreview(durations: Self.durations, mode: .single, selected: [2]))
        case "gallery_bulk":
            gallery(GalleryPreview(durations: Self.durations, mode: .bulk, selected: [0, 2, 4, 7]))
        case "gallery_combine":
            gallery(GalleryPreview(durations: Self.durations, mode: .combine, selected: [0, 2, 4, 7]))
        case "style":
            gallery(GalleryPreview(durations: Self.durations, mode: .single, selected: [2], showStyle: true))
        case "gallery_access":
            gallery(GalleryPreview(durations: Self.durations, mode: .single, selected: [], showLimitedCard: true))
        case "perm_camera":
            PermissionScreen(kind: .camera) {}
        case "perm_mic":
            PermissionScreen(kind: .microphone) {}
        case "edits", "edits_empty":
            ProjectsView()
        case "you":
            YouView(selectTab: { _ in })
        case "plan":
            PlanView(selectTab: { _ in })
        case "err_unsupported":
            CreateProblemView(problem: .unsupportedMedia, onBack: {}, onShowFormats: {})
        case "err_too_large":
            CreateProblemView(problem: .tooLarge(detail: "2:31 • 1.4 GB / 1 GB limit", sizeText: "1.4 GB"), onBack: {})
        case "err_too_long":
            CreateProblemView(problem: .tooLong(detail: "12:18"), onBack: {})
        case "err_clip_limit":
            CreateProblemView(problem: .clipLimit, onBack: {})
        case "err_could_not_add":
            CreateProblemView(problem: .couldNotAdd, onBack: {})
        case "err_formats":
            CreateProblemView(problem: .supportedFormats, onBack: {})
        default:
            Text("Unknown screen \(screen)").foregroundStyle(.white)
        }
    }

    private func gallery(_ preview: GalleryPreview) -> some View {
        GalleryPickerView(onCancel: {}, onCreate: { _, _, _ in }, preview: preview)
    }

    private static let sampleProjects: [Project] = (0..<15).map { index in
        Project(
            projectID: "sample-\(index)",
            userID: "sample",
            title: "Sample edit \(index + 1)",
            state: index % 3 == 1 ? "draft" : "draft_ready",
            latestJobID: nil,
            sourceCount: 1,
            updatedAt: nil
        )
    }
}

/// CI only (Debug): opens the real editor on a real cut, to prove the draft loads.
/// The session comes from environment variables set by the workflow (never in arguments or logs).
private struct EditorProbe: View {
    @EnvironmentObject private var appState: AppState
    @State private var project: Project?

    var body: some View {
        NavigationStack {
            if let project {
                DraftEditorView(project: project)
            } else {
                ProgressView("Probe…")
            }
        }
        .task {
            let env = ProcessInfo.processInfo.environment
            guard let token = env["CUTSELL_PROBE_TOKEN"], let user = env["CUTSELL_PROBE_USER"],
                  let projectID = env["CUTSELL_PROBE_PROJECT"], let api = env["CUTSELL_PROBE_API"] else { return }
            UserDefaults.standard.set(api, forKey: "cutsell.api.baseURL")
            let session = CutSellSession(userID: user, accessToken: token)
            await APIClient.shared.setSession(session)
            appState.session = session
            project = Project(projectID: projectID, userID: user, title: "Editor probe", state: "draft_ready",
                              latestJobID: nil, sourceCount: 1, updatedAt: nil)
        }
    }
}

/// Editor v2 with a sample cut (no server): 4 clips, a voice-over, two overlays.
private struct SampleEditor: View {
    let state: EditorPreviewState
    var captions = false
    @StateObject private var holder = SampleEditorHolder()

    var body: some View {
        NavigationStack {
            if let model = holder.model {
                EditorV2View(model: model, project: holder.project, previewState: state)
            }
        }
        .onAppear { holder.build(captions: captions) }
    }
}

@MainActor
private final class SampleEditorHolder: ObservableObject {
    @Published var model: DraftEditorViewModel?
    let project = Project(projectID: "sample", userID: "sample", title: "Untitled Cut", state: "draft_ready",
                          latestJobID: nil, sourceCount: 1, updatedAt: nil, createdAt: "2026-10-13T08:00:00+00:00")

    func build(captions: Bool) {
        guard model == nil else { return }
        let model = DraftEditorViewModel(project: project, session: CutSellSession(userID: "sample", accessToken: "sample"))
        let clips: [JSONValue] = [
            (0.0, 8.0, "This deal"), (8.0, 15.0, "saves you"), (15.0, 24.0, "hours."), (24.0, 32.0, "Grab it today."),
        ].enumerated().map { index, item in
            .object(["clip_id": .string("c\(index + 1)"), "source_asset_id": .string("s1"),
                     "start": .number(item.0), "end": .number(item.1), "caption_text": .string(item.2)])
        }
        let draft: JSONValue = .object([
            "selected": .array(clips),
            "captions_enabled": .bool(captions),
            "text_overlays": .array([.object(["overlay_id": .string("t1"), "text": .string("COSTCO Deals! 🔥"),
                                              "start": .number(0), "end": .number(15)])]),
            "media_overlays": .array([.object(["overlay_id": .string("m1"), "kind": .string("costco"),
                                               "start": .number(15.5), "end": .number(31)])]),
        ])
        model.snapshot = DraftSnapshot(projectID: "sample", userID: "sample", revision: 1, draft: draft, sources: [])
        model.setPreviewComposition(TimelineComposition(
            contractVersion: 1, baseEditIdentity: "sample", timelineDurationSec: 32, timelineRevisionIdentity: "r1",
            brollPlacements: [],
            voiceOverPlacements: [VoiceOverPlacement(placementID: "v1", assetID: "a1", timelineStartSec: 0, timelineEndSec: 32,
                                                     sourceInSec: 0, sourceOutSec: 32, transcriptReference: nil)]
        ))
        model.playheadTime = 6
        self.model = model
    }
}
#endif
