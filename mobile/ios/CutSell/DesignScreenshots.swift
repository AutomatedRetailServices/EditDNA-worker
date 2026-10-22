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
        "editor_v4_base", "editor_v4_caption_edit", "editor_v4_vo_record", "editor_v4_vo_review",
        "editor_v4_cap_highlight", "editor_v4_cap_box", "editor_v4_cap_fonts", "editor_v4_overlay_gallery",
        "editor_v4_overlay_selected", "editor_v4_mic", "editor_v4_style", "editor_v4_sel_video",
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
            .environmentObject(L10n.shared)
            .environment(\.locale, L10n.locale)
            .appColorScheme()
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
                fileName: CutName.display("Cut 14 Oct"), thumbnail: nil, onClose: {}, onOpenEdits: {}
            )
            .background(CS.base.ignoresSafeArea())
        case "ready":
            ReadyContent(
                cut: CutTracker.Cut(id: "sample", title: "Sample edit", fileName: "IMG_4821.MOV", footageSeconds: 366),
                project: Self.sampleProjects[0], onBack: {}, onReview: {}
            )
            .background(CS.base.ignoresSafeArea())
        case "create_probe":
            CreateProbe()
        case "app_probe":
            AppProbe()
        case "editor_probe":
            EditorProbe()
        case "perf_probe":
            PerfProbeScreen()
        case "editor_v4_base":
            SampleEditor(state: EditorPreviewState())
        case "editor_v4_caption_edit":
            SampleEditor(state: EditorPreviewState(panel: .captions("c1")), preset: "highlight_green")
        case "editor_v4_vo_record":
            SampleEditor(state: EditorPreviewState(panel: .voiceRecord))
        case "editor_v4_vo_review":
            SampleEditor(state: EditorPreviewState(panel: .voiceReview))
        case "editor_v4_cap_highlight":
            SampleEditor(state: EditorPreviewState(panel: .captions(nil)), preset: "highlight_green")
        case "editor_v4_cap_box":
            SampleEditor(state: EditorPreviewState(panel: .captions(nil)), preset: "box")
        case "editor_v4_cap_fonts":
            SampleEditor(state: EditorPreviewState(panel: .captions(nil)), preset: "classic", font: "bebas_neue")
        case "editor_v4_overlay_gallery":
            SampleEditor(state: EditorPreviewState(sheet: .overlay))
        case "editor_v4_overlay_selected":
            SampleEditor(state: EditorPreviewState(selection: .overlay("m1", isText: false)), overlays: true)
        case "editor_v4_mic":
            SampleEditor(state: EditorPreviewState(panel: .micPermission), overlays: true)
        case "editor_v4_style":
            SampleEditor(state: EditorPreviewState(panel: .style))
        case "editor_v4_sel_video":
            SampleEditor(state: EditorPreviewState(selection: .video("c2")))
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

/// Build 19 probe (Debug): the real Create flow (gallery → style → AI notice → upload → processing
/// → editor) signed in with the probe's throwaway session, against the live server.
private struct CreateProbe: View {
    @EnvironmentObject private var appState: AppState
    @State private var ready = false

    var body: some View {
        Group {
            if ready {
                CreateFlowView(selectTab: { _ in })
            } else {
                ProgressView("Probe…")
            }
        }
        .task {
            let env = ProcessInfo.processInfo.environment
            guard let token = env["CUTSELL_PROBE_TOKEN"], let user = env["CUTSELL_PROBE_USER"],
                  let api = env["CUTSELL_PROBE_API"] else { return }
            UserDefaults.standard.set(api, forKey: "cutsell.api.baseURL")
            let session = CutSellSession(userID: user, accessToken: token)
            await APIClient.shared.setSession(session)
            appState.session = session
            ready = true
        }
    }
}

/// CI only (Debug): the whole app (all four tabs) with the probe's session, and the same start-up
/// steps as a real launch (uploads to finish or resume). Build 24 probe.
private struct AppProbe: View {
    @EnvironmentObject private var appState: AppState
    @State private var ready = false

    @ObservedObject private var language = L10n.shared

    var body: some View {
        Group {
            if ready { MainTabView(initialTab: L10n.takeReopenTab() ?? .create).id(language.code) } else { ProgressView("Probe…") }
        }
        .task {
            let env = ProcessInfo.processInfo.environment
            guard let token = env["CUTSELL_PROBE_TOKEN"], let user = env["CUTSELL_PROBE_USER"],
                  let api = env["CUTSELL_PROBE_API"] else { return }
            UserDefaults.standard.set(api, forKey: "cutsell.api.baseURL")
            let session = CutSellSession(userID: user, accessToken: token)
            await APIClient.shared.setSession(session)
            appState.session = session
            try? await appState.refreshProjects()
            ready = true
            if await PendingUploadFinisher.run(session: session, reason: "launch", resumeUpload: { record in
                await ResumeRunner.resume(record, appState: appState)
            }) > 0 {
                try? await appState.refreshProjects()
            }
        }
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

/// Editor v4 with a sample cut (no server): 5 clips with captions, a voice-over, optional overlays.
private struct SampleEditor: View {
    let state: EditorPreviewState
    var preset = "classic"
    var font = "montserrat"
    var overlays = false
    @StateObject private var holder = SampleEditorHolder()

    var body: some View {
        NavigationStack {
            if let model = holder.model {
                EditorV4View(model: model, project: holder.project, previewState: state)
            }
        }
        .onAppear { holder.build(preset: preset, font: font, overlays: overlays) }
    }
}

@MainActor
private final class SampleEditorHolder: ObservableObject {
    @Published var model: DraftEditorViewModel?
    let project = Project(projectID: "sample", userID: "sample", title: "Untitled Cut", state: "draft_ready",
                          latestJobID: nil, sourceCount: 1, updatedAt: nil, createdAt: "2026-10-13T08:00:00+00:00")

    func build(preset: String, font: String, overlays: Bool) {
        guard model == nil else { return }
        let model = DraftEditorViewModel(project: project, session: CutSellSession(userID: "sample", accessToken: "sample"))
        let clips: [JSONValue] = [
            (0.0, 6.2, "This deal saves you money every week"), (6.2, 13.9, "Every time I walk into the store"),
            (13.9, 22.9, "It smells amazing and lasts all day"), (22.9, 31.2, "I didn't want to spend, but I bought it"),
            (31.2, 38.0, "Grab yours today"),
        ].enumerated().map { index, item in
            .object(["clip_id": .string("c\(index + 1)"), "source_asset_id": .string("s1"),
                     "start": .number(item.0), "end": .number(item.1), "caption_text": .string(item.2)])
        }
        var object: [String: JSONValue] = [
            "selected": .array(clips),
            "captions_enabled": .bool(true),
            "caption_preset": .string(preset),
            "caption_font": .string(font),
            "text_overlays": .array([]),
            "media_overlays": .array([]),
        ]
        if overlays {
            object["text_overlays"] = .array([.object(["overlay_id": .string("t1"), "text": .string("50% OFF 🔥"),
                                                       "start": .number(0), "end": .number(12)])])
            object["media_overlays"] = .array([.object(["overlay_id": .string("m1"), "kind": .string("photo"),
                                                        "x": .number(0.5), "y": .number(0.3), "width": .number(0.55),
                                                        "start": .number(0), "end": .number(30)])])
        }
        model.snapshot = DraftSnapshot(projectID: "sample", userID: "sample", revision: 1, draft: .object(object), sources: [])
        model.setPreviewComposition(TimelineComposition(
            contractVersion: 1, baseEditIdentity: "sample", timelineDurationSec: 38, timelineRevisionIdentity: "r1",
            brollPlacements: [],
            voiceOverPlacements: [VoiceOverPlacement(placementID: "v1", assetID: "a1", timelineStartSec: 3, timelineEndSec: 11,
                                                     sourceInSec: 0, sourceOutSec: 8, transcriptReference: nil)]
        ))
        model.playheadTime = 0
        self.model = model
    }
}
#endif
