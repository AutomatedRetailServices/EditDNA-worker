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

    /// White Indigo review set (Figma 12_Prototype): the screens the Product Owner asked to see.
    static let whiteIndigo = [
        "editor_v4_base", "editor_v4_caption_edit", "editor_v4_cap_highlight", "editor_v4_cap_box", "editor_v4_cap_fonts",
        "editor_v4_overlay_gallery", "editor_v4_overlay_selected", "editor_v4_vo_record", "editor_v4_vo_review",
        "you", "you_stats", "plan_built",
    ]
    /// The full White Indigo review set (main screens).
    static let whiteIndigoMain = [
        "camera_style", "mix_setup", "you_account", "you_prefs", "you_plan", "you_help", "ai_consent",
        "hub", "promoting_sheet", "promoting_bulk", "edits_ready", "product_detail", "assign", "plan_goal", "plan_week", "plan_today", "edits", "edits_products", "edits_processing", "plan", "you", "processing", "processing_failed", "ready",
        "err_too_long", "err_unsupported", "err_could_not_add", "perm_camera", "gallery_single", "gallery_access", "style",
        "camera", "editor_v4_base", "editor_v4_vo_record", "welcome",
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
                if ["edits", "hub", "edits_products", "plan", "you", "edits_ready", "assign", "promoting_sheet", "promoting_bulk", "you_stats",
                    "product_detail"].contains(screen) { appState.projects = Self.sampleProjects }
            }
    }

    @ViewBuilder private var content: some View {
        switch screen {
        case "hub":
            HomeHubView(onUpload: { _ in }, onRecord: {}, onOpenProject: { _ in }, selectTab: { _ in })
        case "processing":
            ProcessingContent(
                stage: CutTracker.Stage(step: 3, title: "Cutting your video…", percent: 42),
                fileName: CutName.display("Cut 14 Oct"), thumbnail: nil, onClose: {}, onOpenEdits: {}
            )
            .background(CS.wiPage.ignoresSafeArea())
        case "queued":
            ProcessingContent(
                stage: CutTracker.Stage(step: 3, title: "Cutting your video…", percent: 0),
                fileName: CutName.display("Cut 14 Oct"), thumbnail: nil, queued: true, onClose: {}, onOpenEdits: {}
            )
            .background(CS.wiPage.ignoresSafeArea())
        case "opening":
            NavigationStack {
                DraftEditorView(project: Project(projectID: "sample", userID: "sample", title: "Cut 14 Oct", state: "draft_ready",
                                                 latestJobID: nil, sourceCount: 1, updatedAt: nil))
            }
        case "export_failed":
            ExportFailedPreview()
        case "export_where", "export_saved", "export_schedule":
            ExportPreview(screen: screen)
        case "plan_my_week":
            NavigationStack { PlanMyWeekView(onBuilt: {}) }
        case "plan_my_week_between":
            NavigationStack { PlanMyWeekView(onBuilt: {}) }
        case "plan_week_sheet":
            NavigationStack { PlanWeekView(onNewMix: {}, onOpenDay: { _ in }) }
                .task { try? await PlanWeekStore.shared.build(session: nil) }
        case "plan_attention_bulk", "plan_attention_plan":
            PlanAttentionPreview(detailed: screen == "plan_attention_plan")
        case "processing_failed":
            ProcessingContent(
                stage: CutTracker.Stage(step: 6, title: "Structuring your cut", percent: 68),
                fileName: CutName.display("Cut 14 Oct"), thumbnail: nil, failed: true, onClose: {}, onOpenEdits: {}
            )
            .background(CS.wiPage.ignoresSafeArea())
        case "edits_processing":
            EditsProcessingPreview()
        case "promoting_sheet", "promoting_bulk":
            ZStack {
                HomeHubView(onUpload: { _ in }, onRecord: {}, onOpenProject: { _ in }, selectTab: { _ in })
                PromotingSheet(purpose: screen == "promoting_bulk" ? .upload(.bulk) : .upload(.single), onContinue: { _ in }, onClose: {})
            }
        case "edits_ready":
            ProjectsView(initialTab: .ready)
        case "product_detail":
            ProductDetailPreview()
        case "assign":
            ZStack(alignment: .bottom) {
                ProjectsView(initialTab: .products)
                Color(fixed: 0x0B1020, opacity: 0.45).ignoresSafeArea()
                AssignProductSheet(request: AssignRequest(projectIDs: ["a", "b", "c", "d"],
                                                          suggestion: CatalogSuggestion(brand: "CeraVe", product: "Foaming Cleanser",
                                                                                        category: nil, confidence: 0.8)))
                    .frame(height: 520)
                    .clipShape(UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28, style: .continuous))
                    .ignoresSafeArea(edges: .bottom)
            }
        case "plan_goal":
            NavigationStack { PlanGoalView(onBuilt: {}) }
        case "plan_week":
            NavigationStack { PlanWeekView(onNewMix: {}, onOpenDay: { _ in }) }
                .task { try? await PlanWeekStore.shared.build(session: nil) }
        case "plan_today":
            NavigationStack { PlanDayView(day: PlanDay.iso.string(from: Date())) }
                .task { try? await PlanWeekStore.shared.build(session: nil) }
        case "camera_style":
            ZStack {
                CreateCameraView(onBack: {}, onUpload: {}, onRecorded: { _ in }, previewMode: true)
                BottomSheetOverlay(isPresented: .constant(true)) {
                    StyleSheetView(ctaTitle: CreateMode.single.ctaTitle(1)) { _ in }
                }
            }
            .preferredColorScheme(.dark)
        case "mix_setup":
            MixSetupView(onClose: {}, onStarted: { _ in })
        case "you_stats":
            YouPagesView(start: .stats)
                .task { try? await Task.sleep(nanoseconds: 600_000_000); EditStats.shared.previewSeed(appState.projects) }
        case "you_account", "you_prefs", "you_plan", "you_help":
            YouPagesView(start: ["you_account": YouPage.account, "you_prefs": .preferences, "you_plan": .plan,
                                 "you_help": .help][screen] ?? .account)
        case "ai_consent":
            ZStack(alignment: .bottom) {
                Color(fixed: 0x0B1020, opacity: 0.45).ignoresSafeArea()
                AIConsentSheet().padding(20).background(CS.wiPage, in: UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28))
            }
        case "welcome":
            WelcomeView()
        case "edits_products":
            ProjectsView(initialTab: .products)
        case "ready":
            ReadyContent(
                cut: CutTracker.Cut(id: "sample", title: "Sample edit", fileName: "IMG_4821.MOV", footageSeconds: 366),
                project: Self.sampleProjects[0], onBack: {}, onReview: {}
            )
            .background(CS.wiPage.ignoresSafeArea())
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
        case "editor_v4_vo_track":
            SampleEditor(state: EditorPreviewState(), voice: true)
        case "editor_v4_vo_selected":
            SampleEditor(state: EditorPreviewState(selection: .voiceOver("vo1")), voice: true)
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
        case "plan_built":
            PlanView(selectTab: { _ in })
                .task { try? await PlanWeekStore.shared.build(session: nil) }
        case "plan_built_thu":
            // Option B on a Thursday: the row starts on that day.
            PlanView(selectTab: { _ in })
                .onAppear {
                    let cal = Calendar(identifier: .gregorian)
                    let wd = (cal.component(.weekday, from: Date()) + 5) % 7
                    PlanView.previewToday = cal.date(byAdding: .day, value: (3 - wd + 7) % 7, to: cal.startOfDay(for: Date()))
                }
                .task { try? await PlanWeekStore.shared.build(session: nil) }
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

    /// The first product of the in-memory catalog (-CutSellCatalogMock).
    private static var mockProductID: String { CatalogStore.shared.allProducts.first?.product.productID ?? "prod_cerave_hydrating" }

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

/// Product detail (Figma 1213:191) on the in-memory catalog (-CutSellCatalogMock).
private struct ProductDetailPreview: View {
    @EnvironmentObject private var appState: AppState
    @State private var id: String?
    var body: some View {
        NavigationStack {
            if let id { ProductDetailView(route: ProductRoute(productID: id)) } else { CS.wiPage }
        }
        .task {
            await CatalogStore.shared.refresh(session: nil, projects: appState.projects)
            id = CatalogStore.shared.allProducts.first?.product.productID
        }
    }
}

/// Figma 1259:3101 / 1259:3199: the Export sheet (WHERE TO?), "Saved to Photos", and the scheduling sheet, which goes on
/// to This week when "Schedule" is tapped (the probe walks it).
private struct ExportPreview: View {
    let screen: String
    @StateObject private var holder = SampleEditorHolder()
    @StateObject private var exporter = DeviceExport()
    @State private var settings = ExportSettings()
    @State private var scheduled = false
    @ObservedObject private var router = AppRouter.shared

    var body: some View {
        ZStack {
            CS.wiPage.ignoresSafeArea()
            if scheduled {
                NavigationStack { PlanWeekView(onNewMix: {}, onOpenDay: { _ in }) }
            } else if let model = holder.model {
                ExportSettingsView(model: model, exporter: exporter, settings: $settings, projectID: "sample", session: nil,
                                   onServerExport: {},
                                   previewDestination: screen == "export_schedule" ? .plan : .photos)
                    .onAppear { if screen != "export_where" { exporter.previewDone() } }
                    .onChange(of: router.pending) { _, value in if value == .thisWeek { router.pending = nil; scheduled = true } }
            }
        }
        .onAppear { holder.build(preset: "classic", font: "montserrat", overlays: false) }
    }
}

/// Figma 1233:3199 "Export failed" (stopped while encoding at 72 %).
private struct ExportFailedPreview: View {
    @StateObject private var exporter = DeviceExport()
    var body: some View {
        ExportFailedView(exporter: exporter, onBack: {}, onTryAgain: {}, onChangeSettings: {}, onServerExport: {})
            .onAppear { exporter.previewFailed(step: 1, progress: 0.72) }
    }
}

/// Figma 1233:3243 (Bulk) and 1233:3354 (Plan, with the counts line): a Bulk send where one video failed.
private struct PlanAttentionPreview: View {
    let detailed: Bool
    private var rows: [ProcessingQueue.Row] {
        [("1", 42.0, ProcessingQueue.State.ready), ("2", 33, .ready), ("3", 51, detailed ? .processing(40) : .ready),
         ("4", 26, .failed("The cut needs another try"))].map { id, duration, state in
            .init(id: id, itemID: nil, projectID: nil, name: "cutsell0100190284\(Int(id)! + 7).mp4", duration: duration,
                  thumbnail: nil, state: state, group: "g")
        }
    }
    var body: some View {
        ZStack {
            CS.wiPage.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    WIHeader { EmptyView() }
                    WIScreenTitle(title: "Plan", subtitle: "Plan needs attention").padding(.top, 2)
                    NeedsAttentionSection(summary: detailed ? NeedsAttentionSummary(ready: 8, active: 1, queued: 0, failed: 1)
                                                            : NeedsAttentionSummary(ready: 9, active: 0, queued: 0, failed: 1),
                                          rows: rows, onRetry: { _ in })
                        .padding(.top, 16)
                }
                .padding(.horizontal, 20)
            }
        }
        .wiStatusBarCover()
        .safeAreaInset(edge: .bottom) { CSTabBar(current: .plan, select: { _ in }) }
    }
}

/// White Indigo · Edits › Processing (Figma 1177:2) with sample rows: processing, queued, failed (no network).
private struct EditsProcessingPreview: View {
    @State private var tab: EditsTab = .processing
    private let rows: [ProcessingQueue.Row] = [
        .init(id: "1", itemID: nil, projectID: nil, name: "Cut 14 Oct", duration: 33, thumbnail: nil, state: .processing(68)),
        .init(id: "2", itemID: nil, projectID: nil, name: "Cut 14 Oct (2)", duration: 51, thumbnail: nil, state: .queued),
        .init(id: "3", itemID: nil, projectID: nil, name: "Product clip 14 Oct", duration: 12, thumbnail: nil,
              state: .failed(ProcessingQueue.noSpeech)),
        .init(id: "4", itemID: nil, projectID: nil, name: "A very long name for this cut of the day", duration: 26,
              thumbnail: nil, state: .failed("Failed")),
    ]

    var body: some View {
        ZStack {
            CS.wiPage.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    WIHeader { WIHeaderIcon(image: "WISearch"); WIHeaderIcon(image: "WIMore") }
                    WIScreenTitle(title: "Edits", subtitle: "Cuts being edited now. You can leave the app.").padding(.top, 2)
                    WISegmented(items: [.init(id: EditsTab.processing, title: "Processing", count: 3),
                                        .init(id: .ready, title: "Ready", count: 28), .init(id: .products, title: "Products"),
                                        .init(id: .drafts, title: "Drafts", count: 12)], selection: $tab)
                        .padding(.top, 13)
                    WISectionTitle(title: "In order", note: "Each one moves to Ready when it’s done").padding(.top, 20)
                    VStack(spacing: 8) {
                        ForEach(Array(rows.enumerated()), id: \.element.id) { index, row in QueueRowView(order: index + 1, row: row) }
                    }
                    .padding(.top, 12)
                    WINote(title: "You can close the app", text: "Editing keeps going on our servers. We’ll notify you as each cut is ready.")
                        .padding(.top, 16)
                }
                .padding(.horizontal, 20)
            }
        }
        .safeAreaInset(edge: .bottom) { CSTabBar(current: .edits, select: { _ in }) }
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
    var voice = false
    @StateObject private var holder = SampleEditorHolder()

    var body: some View {
        NavigationStack {
            if let model = holder.model {
                EditorV4View(model: model, project: holder.project, previewState: state)
            }
        }
        .onAppear { holder.build(preset: preset, font: font, overlays: overlays, voice: voice) }
    }
}

@MainActor
private final class SampleEditorHolder: ObservableObject {
    @Published var model: DraftEditorViewModel?
    let project = Project(projectID: "sample", userID: "sample", title: "Untitled Cut", state: "draft_ready",
                          latestJobID: nil, sourceCount: 1, updatedAt: nil, createdAt: "2026-10-13T08:00:00+00:00")

    func build(preset: String, font: String, overlays: Bool, voice: Bool = false) {
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
        if voice {
            object["audio_clips"] = .array([.object(["audio_id": .string("vo1"), "kind": .string("voice"), "asset_id": .string("a1"),
                                                     "start": .number(0.4), "source_in": .number(0), "source_out": .number(8)])])
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
