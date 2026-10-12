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
#endif
