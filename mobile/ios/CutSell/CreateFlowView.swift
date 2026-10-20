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
    /// Build 24: jobs chosen while another one is still uploading (they start one after another,
    /// shown as "Queued" in Edits · Processing).
    @State private var waiting: [(job: CreateJob, items: [UUID])] = []
    /// The Processing-queue items of the job running now (one per cut).
    @State private var jobItems: [UUID] = []
    /// Bulk: "Your N videos are being edited" (Figma 1134:4635).
    @State private var bulkSheetCount: Int?

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
            .overlay(alignment: .bottom) {
                if let count = bulkSheetCount {
                    ZStack(alignment: .bottom) {
                        Color.black.opacity(0.6).ignoresSafeArea()
                            .onTapGesture { bulkSheetCount = nil }
                        BulkStartedSheet(count: count, onGoToEdits: {
                            bulkSheetCount = nil
                            openEditsProcessing()
                        }, onStay: { bulkSheetCount = nil })
                    }
                    .ignoresSafeArea(edges: .bottom)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .animation(.easeOut(duration: 0.25), value: bulkSheetCount)
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
        // Build 24: the gallery is presented from an item that carries the mode, so the Bulk / Combine chip
        // that opened it always wins (with isPresented the cover could read a stale galleryMode → Single).
        .fullScreenCover(item: Binding(
            get: { showingGallery ? GalleryRoute(mode: galleryMode) : nil },
            set: { if $0 == nil { showingGallery = false } }
        ), onDismiss: {
            if let next = jobAfterGallery {
                jobAfterGallery = nil
                launch(next)
            }
        }) { route in
            GalleryPickerView(
                onCancel: { showingGallery = false },
                onCreate: { mode, assets, style in
                    jobAfterGallery = CreateJob(mode: mode, source: .library(assets), style: style)
                    showingGallery = false
                },
                initialMode: route.mode
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
                    // Build 24: "Continue in background" works at every step; the work goes on here.
                    onClose: { closeProcessing() },
                    onOpenEdits: { closeProcessing(); openEditsProcessing() }
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
        // Build 24: a job chosen while another one is still uploading waits its turn ("Queued").
        if submitter.isSubmitting || job != nil {
            Task { @MainActor in
                guard await AIConsentCenter.shared.ensure() else { return }
                let items = ProcessingQueue.shared.add(Self.entries(for: newJob))
                waiting.append((newJob, items))
                CutSellDiagnostics.log("job_queued", ["videos": String(items.count), "waiting": String(waiting.count)])
                if newJob.mode == .bulk && items.count > 1 { openEditsProcessing() }
            }
            return
        }
        // Build 19: the AI notice is answered BEFORE the upload screen covers everything (it used to
        // be asked under it, invisible, and the cut waited at 0 % forever).
        Task { @MainActor in
            guard await AIConsentCenter.shared.ensure() else { return }
            start(newJob)
        }
    }

    private func start(_ newJob: CreateJob, items: [UUID]? = nil, present: Bool = true) {
        guard !submitter.isSubmitting else { return }
        submitter.reset()
        jobItems = items ?? ProcessingQueue.shared.add(Self.entries(for: newJob))
        submitter.queueItem = jobItems.first
        loadThumbnails(for: newJob, items: jobItems)
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
        if present {
            // Bulk: no single-video Processing screen ("Step 1 of 8"); only the "being edited" sheet
            // (Figma 1134:4635), and the videos are followed in Edits · Processing.
            if newJob.mode == .bulk, case .library(let assets) = newJob.source, assets.count > 1 {
                bulkSheetCount = assets.count
            } else {
                job = newJob
            }
        }
        Task {
            await run(newJob)
            startNextWaiting()
        }
    }

    /// Closes the Processing screen; the upload and the cut go on (this view stays alive).
    private func closeProcessing() {
        bulkSheetCount = nil
        job = nil
        CutSellDiagnostics.log("continue_in_background", ["step": submitter.isSubmitting ? submitter.stageTitle : "server"])
    }

    private func openEditsProcessing() {
        ProcessingQueue.shared.requestedEditsFilter = "processing"
        selectTab(.edits)
    }

    /// The next job chosen while this one ran (no screen: it shows in Edits · Processing).
    private func startNextWaiting() {
        guard !submitter.isSubmitting, !waiting.isEmpty else { return }
        let next = waiting.removeFirst()
        start(next.job, items: next.items, present: false)
    }

    /// One Processing row per cut: Bulk = one per video; Combine = one; a recording = one.
    static func entries(for job: CreateJob) -> [(name: String, duration: Double)] {
        switch job.source {
        case .recorded(let video):
            return [(video.name, video.duration)]
        case .library(let assets):
            let names = assets.map { asset -> String in
                PHAssetResource.assetResources(for: asset).first { $0.type == .video }?.originalFilename ?? "Video"
            }
            if job.mode == .combine {
                return [("\(assets.count) clips", assets.reduce(0) { $0 + $1.duration })]
            }
            return zip(names, assets).map { ($0, $1.duration) }
        }
    }

    /// Small first-frame thumbnails for the Processing rows (from Photos, no download).
    private func loadThumbnails(for job: CreateJob, items: [UUID]) {
        guard case .library(let assets) = job.source else { return }
        let targets = job.mode == .combine ? Array(assets.prefix(1)) : assets
        for (asset, id) in zip(targets, items) {
            let options = PHImageRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .opportunistic
            PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: 138, height: 180),
                                                  contentMode: .aspectFill, options: options) { image, _ in
                guard let image else { return }
                Task { @MainActor in ProcessingQueue.shared.update(id, thumbnail: image) }
            }
        }
    }

    /// A recording the person closed without making a cut: remove it from the phone.
    private func discardRecording(_ video: LocalVideo) {
        try? FileManager.default.removeItem(at: video.url)
    }

    /// Shows the video on the Processing screen and, once the cut started, follows it on the server.
    @MainActor
    private func startTracking(projectID: String, videos: [LocalVideo], thumbnail: UIImage?) {
        let footage = videos.reduce(0) { $0 + $1.duration }
        ProcessingQueue.shared.update(submitter.queueItem, thumbnail: thumbnail, projectID: projectID, state: .processing(0))
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
        let image = await CutTracker.firstFrame(of: first.previewURL ?? first.url)
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
            // No Processing screen (Bulk, or left with Continue in background): the row shows FAILED.
            if job == nil { markFailed(submitter.queueItem, error: exportFailure, retryJob: current) }
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
            markFailed(submitter.queueItem, error: error, retryJob: current)
            show(error)
        }
    }

    /// Build 24: a failed cut stays in Edits · Processing as FAILED with Retry: the upload continues
    /// where it stopped (a saved project), or the video is sent again (nothing was saved yet).
    @MainActor
    private func markFailed(_ item: UUID?, error: Error, retryJob: CreateJob) {
        guard let item else { return }
        let projectID = (error as? CutSubmitError)?.projectID ?? submitter.currentProjectID
        let detail = Self.plain((error as? CutSubmitError)?.underlying ?? error)
        ProcessingQueue.shared.fail(item, detail: detail) {
            Task { @MainActor in await retryInBackground(item: item, projectID: projectID, job: retryJob) }
        }
    }

    @MainActor
    private func retryInBackground(item: UUID, projectID: String?, job retryJob: CreateJob) async {
        CutSellDiagnostics.log("queue_retry", ["saved": projectID == nil ? "no" : "yes"])
        if let projectID, let record = await PendingCutStore.shared.record(projectID: projectID) {
            // Its own submitter: a retry never waits behind (or disturbs) the upload running now.
            let retrying = CutSubmitter()
            retrying.queueItem = item
            let videos = record.videos.map { LocalVideo(url: $0.fileURL, name: $0.name, duration: $0.duration, contentType: $0.contentType) }
            do {
                let id = try await retrying.submit(
                    videos: videos, title: record.title, mode: record.mode, audioOverlap: record.audioOverlap,
                    autoZoom: record.autoZoom ?? true, resumeProjectID: projectID, appState: appState
                )
                CutTracker.shared.track(projectID: id, jobID: retrying.lastJobID, title: record.title,
                                        fileName: videos.count == 1 ? videos[0].name : "\(videos.count) clips",
                                        thumbnail: nil, footageSeconds: videos.reduce(0) { $0 + $1.duration })
                ProcessingQueue.shared.update(item, projectID: id, state: .processing(0))
            } catch {
                ProcessingQueue.shared.fail(item, detail: Self.plain((error as? CutSubmitError)?.underlying ?? error)) {
                    Task { @MainActor in await retryInBackground(item: item, projectID: projectID, job: retryJob) }
                }
            }
        } else if submitter.isSubmitting {
            ProcessingQueue.shared.update(item, state: .queued)
            waiting.append((retryJob, [item]))
        } else {
            start(retryJob, items: [item], present: false)
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
                    // Build 22: a clip that fails does NOT stop the others; it stays saved to resume
                    // ("Try again" sends only its missing parts) and the failure is shown at the end.
                    let count = assets.count
                    var firstFailure: Error?
                    var failedClips: [Int] = []
                    // Build 26: the videos of one Bulk send carry the same tag (server-side prompt caching).
                    submitter.bulkTag = current.mode == .bulk && count > 1
                        ? ["bulk_size": String(count), "bulk_id": UUID().uuidString] : nil
                    defer { submitter.bulkTag = nil }
                    for (index, asset) in assets.enumerated() where index >= startedInJob {
                        // Build 24: this video's row in Edits · Processing.
                        let item = index < jobItems.count ? jobItems[index] : nil
                        submitter.queueItem = item
                        ProcessingQueue.shared.update(item, state: .uploading(0))
                        do {
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
                        } catch {
                            if count == 1 || error is AIConsentError { throw error }
                            failedClips.append(index + 1)
                            firstFailure = firstFailure ?? error
                            CutSellDiagnostics.log("clip_cut_failed", Retry.fields(error, ["source_order": String(index), "of": String(count)]))
                            markFailed(item, error: error,
                                       retryJob: CreateJob(mode: .single, source: .library([asset]), style: current.style))
                        }
                        startedInJob = index + 1
                    }
                    if let firstFailure {
                        CutSellDiagnostics.log("multiple_finished", ["failed_clips": failedClips.map(String.init).joined(separator: ","),
                                                                     "of": String(count)])
                        throw firstFailure
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
        var video = try await exportFromPhotos(asset, index: index, count: count)
        video.albumHint = GalleryExport.albumName(of: asset)
        return video
    }

    private func exportFromPhotos(_ asset: PHAsset, index: Int, count: Int) async throws -> LocalVideo {
        submitter.announce(count > 1 ? "Getting video \(index + 1) of \(count) from Photos…" : "Getting your video from Photos…")
        let title = count > 1 ? "Downloading video \(index + 1) of \(count) from iCloud" : "Downloading from iCloud"
        let copying = count > 1 ? "Getting video \(index + 1) of \(count) from Photos" : "Getting your video from Photos"
        let started = Date()
        CutSellDiagnostics.log("asset_export_started", ["index": String(index + 1), "of": String(count),
                                                        "seconds_long": String(format: "%.0f", asset.duration)])
        // Build 23 ("iCloud light"): a video only in iCloud starts from a small fast copy; its original
        // downloads and uploads in the background (needs the server's audio-first flow).
        if await ServerCapabilities.audioFirst() {
            let fast = count > 1 ? "Getting video \(index + 1) of \(count) from iCloud" : "Getting your video from iCloud"
            if let light = try await GalleryExport.light(asset, progress: { fraction in
                Task { @MainActor in submitter.stage(fast, fraction) }
            }) {
                CutSellDiagnostics.log("asset_export_done", ["seconds": String(format: "%.1f", Date().timeIntervalSince(started)),
                                                             "mode": "icloud_light"])
                return light
            }
        }
        let video = try await GalleryExport.export(asset, name: nil, progress: { fraction in
            // Only called while the video comes down from iCloud (not when it is on the iPhone).
            Task { @MainActor in
                submitter.stage(title, fraction)
                submitter.logStep("asset_export_progress", fraction)
            }
        }, copied: { bytes in
            Task { @MainActor in
                if !submitter.stageTitle.hasPrefix("Downloading") {
                    submitter.stage("\(copying) · \(bytes / 1_048_576) MB")
                } else {
                    submitter.touch()
                }
            }
        })
        let mb = ((try? FileManager.default.attributesOfItem(atPath: video.url.path))?[.size] as? NSNumber)?.int64Value ?? 0
        CutSellDiagnostics.log("asset_export_done", ["seconds": String(format: "%.1f", Date().timeIntervalSince(started)),
                                                     "mb": String(mb / 1_048_576)])
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

/// The gallery sheet and the mode it opens in (Single / Bulk / Combine).
struct GalleryRoute: Identifiable {
    let mode: CreateMode
    var id: String { mode.rawValue }
}
