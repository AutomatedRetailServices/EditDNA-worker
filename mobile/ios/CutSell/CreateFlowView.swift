import Photos
import SwiftUI

/// The Create tab: Home Hub (Figma 1068:22) → Upload (gallery) or Record (camera) → "Choose your
/// style" → Processing (upload %, then the cut's real progress on the server) → Ready → editor.
/// Uploading and starting the cut use `CutSubmitter`, the same pipeline as before.
struct CreateFlowView: View {
    let selectTab: (MainTab) -> Void

    @EnvironmentObject private var appState: AppState
    @StateObject private var submitter = CutSubmitter()
    @State private var showingCamera = false
    @State private var afterCamera: AfterCamera?
    @State private var showingGallery = false
    @State private var galleryMode: CreateMode = .single
    @State private var recorded: LocalVideo?
    @State private var job: CreateJob?
    /// Chosen in the gallery; started once the gallery has finished closing.
    @State private var jobAfterGallery: CreateJob?
    @State private var reopenGalleryAfterJob = false
    @State private var problem: CreateProblem?
    @State private var failure: SubmitFailure?
    /// Bulk: cuts already started before a failure (not repeated on "Try again").
    @State private var startedInJob = 0
    /// The cut shown on the Processing screen once the server created it.
    @State private var processingProjectID: String?
    @State private var processingFileName = ""
    @State private var processingThumbnail: UIImage?
    @State private var path: [Project] = []

    enum AfterCamera { case gallery, job(CreateJob) }

    enum SubmitFailure: Identifiable {
        /// The upload record is saved on the phone: Edits shows "Resume upload".
        case savedForResume
        /// Nothing was saved yet (e.g. no connection): the same videos can be sent again.
        case notSaved
        var id: Int { self == .savedForResume ? 1 : 2 }
    }

    var body: some View {
        NavigationStack(path: $path) {
            HomeHubView(
                onUpload: { mode in galleryMode = mode; showingGallery = true },
                onRecord: { showingCamera = true },
                onOpenProject: { path.append($0) },
                selectTab: selectTab
            )
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(for: Project.self) { project in
                ProjectDetailView(project: project, onOpenEdits: { selectTab(.edits) })
            }
        }
        .tint(CS.cyan)
        .fullScreenCover(isPresented: $showingCamera, onDismiss: {
            switch afterCamera {
            case .gallery: showingGallery = true
            case .job(let next): launch(next)
            case nil: break
            }
            afterCamera = nil
        }) {
            ZStack {
                CreateCameraView(
                    onBack: { showingCamera = false },
                    onUpload: { afterCamera = .gallery; showingCamera = false },
                    onRecorded: { recorded = $0 }
                )
                BottomSheetOverlay(isPresented: Binding(
                    get: { recorded != nil },
                    set: { if !$0, let video = recorded { recorded = nil; discardRecording(video) } }
                )) {
                    StyleSheetView(ctaTitle: CreateMode.single.ctaTitle(1)) { style in
                        guard let video = recorded else { return }
                        recorded = nil
                        afterCamera = .job(CreateJob(mode: .single, source: .recorded(video), style: style))
                        showingCamera = false
                    }
                }
            }
        }
        .fullScreenCover(isPresented: $showingGallery, onDismiss: {
            if let next = jobAfterGallery {
                jobAfterGallery = nil
                launch(next)
            }
        }) {
            GalleryPickerView(
                onCancel: { showingGallery = false },
                onCreate: { mode, assets, style in
                    jobAfterGallery = CreateJob(mode: mode, source: .library(assets), style: style)
                    showingGallery = false
                },
                initialMode: galleryMode
            )
        }
        .fullScreenCover(item: $job, onDismiss: {
            if reopenGalleryAfterJob {
                reopenGalleryAfterJob = false
                showingGallery = true
            }
        }) { current in
            NavigationStack {
                CutProgressView(
                    submitter: submitter,
                    projectID: processingProjectID,
                    fileName: processingFileName,
                    thumbnail: processingThumbnail,
                    onClose: { job = nil },
                    onOpenEdits: { job = nil; selectTab(.edits) }
                )
            }
            .environmentObject(appState)
            .fullScreenCover(item: $problem) { shown in
                CreateProblemView(
                    problem: shown,
                    onBack: {
                        reopenGalleryAfterJob = true
                        problem = nil
                        job = nil
                    },
                    onShowFormats: { problem = .supportedFormats }
                )
            }
            .fullScreenCover(item: $failure) { shown in
                switch shown {
                case .savedForResume:
                    ProblemScreen(
                        screenTitle: "Upload paused",
                        heading: "Something went wrong",
                        message: "Your video is saved on this iPhone. Open Edits to resume the upload.",
                        primaryTitle: "Go to Edits",
                        primaryAction: { failure = nil; job = nil; selectTab(.edits) }
                    )
                case .notSaved:
                    ProblemScreen(
                        screenTitle: "Couldn’t start the cut",
                        heading: "Something went wrong",
                        message: "Check your connection and try again. Your video is still here.",
                        primaryTitle: "Try again",
                        primaryAction: {
                            failure = nil
                            Task { await run(current) }
                        },
                        secondaryTitle: "Cancel",
                        secondaryAction: {
                            failure = nil
                            job = nil
                            if case .recorded(let video) = current.source { discardRecording(video) }
                        }
                    )
                }
            }
        }
    }

    private func launch(_ newJob: CreateJob) {
        // One cut at a time from this screen: the upload screen cannot be left until it is done.
        guard !submitter.isSubmitting else { return }
        submitter.reset()
        problem = nil
        failure = nil
        CutSellDiagnostics.log("create_style_chosen", ["overlap_audio": String(newJob.style.overlapAudio), "auto_zoom": String(newJob.style.autoZoom), "mode": newJob.mode.rawValue])
        startedInJob = 0
        processingProjectID = nil
        processingThumbnail = nil
        switch newJob.source {
        case .recorded(let video): processingFileName = video.url.lastPathComponent
        case .library: processingFileName = ""
        }
        job = newJob
        Task { await run(newJob) }
    }

    /// A recording the person closed without making a cut: remove it from the phone.
    private func discardRecording(_ video: LocalVideo) {
        try? FileManager.default.removeItem(at: video.url)
    }

    /// Shows the video on the Processing screen and, once the cut started, follows it on the server.
    @MainActor
    private func startTracking(projectID: String, videos: [LocalVideo], thumbnail: UIImage?) {
        let footage = videos.reduce(0) { $0 + $1.duration }
        CutTracker.shared.track(
            projectID: projectID, jobID: submitter.lastJobID,
            title: appState.projects.first { $0.projectID == projectID }?.title ?? "Untitled Cut",
            fileName: videos.count == 1 ? videos[0].name : "\(videos.count) clips",
            thumbnail: thumbnail, footageSeconds: footage
        )
        if processingProjectID == nil { processingProjectID = projectID }
        if let thumbnail { ThumbnailStore.shared.save(thumbnail, for: projectID) }
    }

    @MainActor
    private func prepareScreen(for videos: [LocalVideo]) async -> UIImage? {
        guard let first = videos.first else { return nil }
        if processingFileName.isEmpty || videos.count > 1 {
            processingFileName = videos.count == 1 ? first.name : "\(videos.count) clips"
        }
        let image = await CutTracker.firstFrame(of: first.url)
        if processingProjectID == nil { processingThumbnail = image }
        return image
    }

    @MainActor
    private func run(_ current: CreateJob) async {
        do {
            switch current.source {
            case .recorded(let video):
                let thumbnail = await prepareScreen(for: [video])
                let projectID = try await submitter.submit(
                    videos: [video], title: Self.cutTitle([video]), mode: CutSubmitter.singleMode,
                    audioOverlap: current.style.overlapAudio, autoZoom: current.style.autoZoom, resumeProjectID: nil, appState: appState
                )
                startTracking(projectID: projectID, videos: [video], thumbnail: thumbnail)
            case .library(let assets):
                if current.mode == .combine {
                    var videos: [LocalVideo] = []
                    for (index, asset) in assets.enumerated() {
                        videos.append(try await export(asset, index: index, count: assets.count))
                    }
                    let thumbnail = await prepareScreen(for: videos)
                    let projectID = try await submitter.submit(
                        videos: videos, title: Self.cutTitle(videos), mode: CutSubmitter.multipleMode,
                        audioOverlap: current.style.overlapAudio, autoZoom: current.style.autoZoom, resumeProjectID: nil, appState: appState
                    )
                    startTracking(projectID: projectID, videos: videos, thumbnail: thumbnail)
                } else {
                    // Single: one asset. Bulk: one separate cut per video, one after another.
                    let count = assets.count
                    for (index, asset) in assets.enumerated() where index >= startedInJob {
                        let video = try await export(asset, index: index, count: count)
                        let thumbnail = await prepareScreen(for: [video])
                        let lower = Double(index) / Double(count)
                        let projectID = try await submitter.submit(
                            videos: [video], title: Self.cutTitle([video]), mode: CutSubmitter.singleMode,
                            audioOverlap: current.style.overlapAudio, autoZoom: current.style.autoZoom, resumeProjectID: nil, appState: appState,
                            progressRange: lower...(lower + 1 / Double(count)),
                            label: count > 1 ? "Cut \(index + 1) of \(count)" : ""
                        )
                        startTracking(projectID: projectID, videos: [video], thumbnail: thumbnail)
                        startedInJob = index + 1
                    }
                }
            }
            // The Processing screen stays open and now follows the cut on the server.
        } catch let exportFailure as GalleryExport.Failure {
            switch exportFailure {
            case .unsupported: problem = .unsupportedMedia
            case .tooLarge(let bytes):
                problem = .tooLarge(
                    detail: "\(CreateLimits.sizeText(bytes)) / 1 GB limit",
                    sizeText: CreateLimits.sizeText(bytes)
                )
            case .unavailable: problem = .couldNotAdd
            }
        } catch let submitError as CutSubmitError {
            failure = submitError.savedForResume ? .savedForResume : .notSaved
        } catch {
            failure = .notSaved
        }
    }

    /// The project is named after the video (not "Untitled Cut").
    static func cutTitle(_ videos: [LocalVideo]) -> String? {
        guard let first = videos.first else { return nil }
        let base = (first.name as NSString).deletingPathExtension
        // Internal file names ("media-B3E6…", "Recorded clip") become "Cut 13 Oct".
        guard !CutNaming.isInternal(base) else { return CutNaming.dated() }
        return videos.count > 1 ? "\(base) +\(videos.count - 1)" : base
    }

    private func export(_ asset: PHAsset, index: Int, count: Int) async throws -> LocalVideo {
        submitter.announce(count > 1 ? "Getting video \(index + 1) of \(count) from Photos…" : "Getting your video from Photos…")
        return try await GalleryExport.export(asset, name: nil)
    }
}

struct CreateJob: Identifiable {
    enum Source {
        case recorded(LocalVideo)
        case library([PHAsset])
    }

    let id = UUID()
    let mode: CreateMode
    let source: Source
    let style: StyleChoice
}
