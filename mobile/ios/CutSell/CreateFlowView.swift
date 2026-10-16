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
        /// The upload record is saved on the phone: "Try again" resumes it (Edits too).
        case savedForResume(projectID: String?, detail: String)
        /// Nothing was saved yet (e.g. no connection): the same videos can be sent again.
        case notSaved(detail: String)
        var id: Int { if case .savedForResume = self { return 1 } else { return 2 } }
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
                case .savedForResume(let projectID, let detail):
                    ProblemScreen(
                        screenTitle: "Upload paused",
                        heading: "The upload stopped",
                        message: "\(detail)\n\nYour video is saved on this iPhone. Try again to continue where it stopped.",
                        primaryTitle: "Try again",
                        primaryAction: {
                            failure = nil
                            Task { await resume(projectID: projectID, current: current) }
                        },
                        secondaryTitle: "Go to Edits",
                        secondaryAction: { failure = nil; job = nil; selectTab(.edits) }
                    )
                case .notSaved(let detail):
                    ProblemScreen(
                        screenTitle: "Couldn’t start the cut",
                        heading: "The upload didn’t start",
                        message: "\(detail)\n\nCheck your connection and try again. Your video is still here.",
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
        guard !submitter.isSubmitting, job == nil else { return }
        // Build 19: the AI notice is answered BEFORE the upload screen covers everything (it used to
        // be asked under it, invisible, and the cut waited at 0 % forever).
        Task { @MainActor in
            guard await AIConsentCenter.shared.ensure() else { return }
            start(newJob)
        }
    }

    private func start(_ newJob: CreateJob) {
        guard !submitter.isSubmitting, job == nil else { return }
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

    /// "Try again" after a paused upload: the same project, from the parts already sent.
    @MainActor
    private func resume(projectID: String?, current: CreateJob) async {
        guard let projectID, let record = await PendingCutStore.shared.record(projectID: projectID) else {
            await run(current)
            return
        }
        submitter.reset()
        do {
            try await StallWatch.run(submitter) {
                let videos = record.videos.map { LocalVideo(url: $0.fileURL, name: $0.name, duration: $0.duration, contentType: $0.contentType) }
                let id = try await submitter.submit(
                    videos: videos, title: record.title, mode: record.mode, audioOverlap: record.audioOverlap,
                    autoZoom: record.autoZoom ?? true, resumeProjectID: projectID, appState: appState
                )
                startTracking(projectID: id, videos: videos, thumbnail: processingThumbnail)
            }
        } catch {
            show(error)
        }
    }

    /// The real error, in plain words, on the "Upload paused" / "Couldn't start" screen.
    @MainActor
    private func show(_ error: Error) {
        if let submitError = error as? CutSubmitError {
            let detail = Self.plain(submitError.underlying)
            failure = submitError.savedForResume ? .savedForResume(projectID: submitError.projectID, detail: detail) : .notSaved(detail: detail)
        } else if let stalled = error as? StallWatch.Stalled {
            failure = processingProjectID != nil || submitter.stageTitle.hasPrefix("Uploading")
                ? .savedForResume(projectID: lastProjectID, detail: stalled.localizedDescription)
                : .notSaved(detail: stalled.localizedDescription)
        } else {
            failure = .notSaved(detail: Self.plain(error))
        }
    }

    static func plain(_ error: Error) -> String {
        if let stalled = error as? StallWatch.Stalled { return stalled.localizedDescription }
        if error is AIConsentError { return AIConsentError.declined.localizedDescription }
        return "Error: \(Retry.short(error))."
    }

    /// The project the last submit created (a stall leaves it saved for resuming).
    @MainActor
    private var lastProjectID: String? { submitter.currentProjectID }

    @MainActor
    private func run(_ current: CreateJob) async {
        do {
            try await StallWatch.run(submitter) { try await work(current) }
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
        } catch {
            show(error)
        }
    }

    @MainActor
    private func work(_ current: CreateJob) async throws {
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
        let title = count > 1 ? "Downloading video \(index + 1) of \(count) from iCloud" : "Downloading from iCloud"
        let started = Date()
        let video = try await GalleryExport.export(asset, name: nil) { fraction in
            // Only called while the video comes down from iCloud (not when it is on the iPhone).
            Task { @MainActor in submitter.stage(title, fraction) }
        }
        CutSellDiagnostics.log("photos_copied", ["seconds": String(format: "%.1f", Date().timeIntervalSince(started))])
        return video
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
