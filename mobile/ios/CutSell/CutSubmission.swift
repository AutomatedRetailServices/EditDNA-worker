import Foundation
import UIKit

/// Uploads the chosen videos and starts the AI cut. This is the same pipeline the app always used
/// (prepare → create project → save a resumable record → multipart upload → /v1/flow-b/jobs);
/// it only moved here so the camera, the gallery and "resume upload" all share one copy.
@MainActor
final class CutSubmitter: ObservableObject {
    @Published private(set) var progress = 0.0
    @Published private(set) var statusText = ""
    @Published private(set) var isSubmitting = false
    /// The server job of the last cut started (for the Processing screen).
    @Published private(set) var lastJobID: String?
    /// Build 19: the step shown on the Processing screen with ITS real % ("Downloading from iCloud",
    /// "Preparing your video", "Uploading your video"…); nil percent = no % for that step.
    @Published private(set) var stageTitle = "Getting your video ready"
    @Published private(set) var stagePercent: Int?
    /// Last time anything moved (a byte, a step, a retry): the 90 s stall watch reads it.
    private(set) var lastActivity = Date()
    /// The last error a retry saw, in plain words (shown if the upload stalls).
    private(set) var lastError: String?
    /// The project of the cut being sent (a stalled upload is resumed into it).
    private(set) var currentProjectID: String?

    /// Clears the previous cut's progress before a new one starts.
    func reset() {
        progress = 0
        statusText = ""
        lastJobID = nil
        stageTitle = "Getting your video ready"
        stagePercent = nil
        lastError = nil
        currentProjectID = nil
        loggedTenths = [:]
        touch()
    }

    func touch() {
        lastActivity = Date()
        if stalled != nil {
            stalled = nil
            CutSellDiagnostics.log("upload_moving_again", [:])
        }
    }

    /// Build 21: "No data sent for 90 seconds…" while the upload keeps trying (nothing is
    /// cancelled); nil again as soon as a byte moves.
    @Published private(set) var stalled: String?
    func markStalled(_ detail: String) { stalled = detail }
    func clearStalled() { stalled = nil }

    /// "Try again" on a paused upload: restarts only the parts in flight.
    func restartStuckParts() {
        CutSellDiagnostics.log("upload_try_again", ["step": stageTitle])
        MultipartUploadManager.restartStuckParts()
        touch()
    }

    /// One step and its real % (0...1), shown as it is.
    func stage(_ title: String, _ fraction: Double? = nil) {
        let percent = fraction.map { Int((min(1, max(0, $0)) * 100).rounded(.down)) }
        if title != stageTitle || percent != stagePercent {
            stageTitle = title
            stagePercent = percent
            touch()
        }
    }

    /// Diagnostics: one event per 10 % of a step (the report keeps the last 50 events).
    private var loggedTenths: [String: Int] = [:]
    func logStep(_ event: String, _ fraction: Double) {
        let tenth = Int((min(1, max(0, fraction)) * 10).rounded(.down))
        guard tenth > (loggedTenths[event] ?? -1) else { return }
        loggedTenths[event] = tenth
        CutSellDiagnostics.log(event, ["percent": String(tenth * 10)])
    }

    /// A retry happened (the work is alive; the screen keeps the step).
    func retried(_ error: String) {
        lastError = error
        touch()
    }

    /// Shows a step that happens before uploading (e.g. copying from Photos).
    func announce(_ text: String) {
        statusText = text
        stage(text.replacingOccurrences(of: "…", with: ""))
    }

    /// Mode names stored in resumable uploads (`PendingCutRecord.mode`); unchanged.
    static let singleMode = "Single clip"
    static let multipleMode = "Multiple clips"

    /// Starts one cut from `videos` (one video, or several combined in order).
    /// Returns the project id. `progressRange` lets several cuts in a row share one progress bar.
    @discardableResult
    func submit(
        videos selected: [LocalVideo],
        title: String?,
        mode: String,
        audioOverlap: Bool,
        autoZoom: Bool = true,
        resumeProjectID: String?,
        appState: AppState,
        progressRange: ClosedRange<Double> = 0...1,
        label: String = ""
    ) async throws -> String {
        guard let session = appState.session else {
            CutSellDiagnostics.log("upload_failed", Retry.fields(UploadFlowError.notSignedIn, ["step": "start"]))
            throw UploadFlowError.notSignedIn
        }
        // Nothing is uploaded or processed before the AI notice is agreed to (once).
        guard await AIConsentCenter.shared.ensure() else { throw AIConsentError.declined }
        isSubmitting = true
        // Build 21: extra time from iOS when the creator leaves CutSell mid-upload, so every part is
        // handed to the background upload service (which then keeps sending with the app closed).
        let backgroundTime = BackgroundTime()
        backgroundTime.begin()
        // Build 22: fresh time from iOS once every part is on R2, for "complete" and the start of the
        // processing (on 18 Oct iOS suspended CutSell right after "complete" was sent: the cut never started).
        let finishing = BackgroundTime("cutsell-finish")
        if let resumeProjectID { ActiveUploads.shared.insert(resumeProjectID) }
        defer { isSubmitting = false; backgroundTime.end(); finishing.end() }
        let span = progressRange.upperBound - progressRange.lowerBound
        func report(_ value: Double) { progress = progressRange.lowerBound + span * min(1, max(0, value)) }
        func say(_ text: String) { statusText = label.isEmpty ? text : "\(label) · \(text)" }
        report(0)
        var transientPreparedFiles: [URL] = []
        var pendingSaved = resumeProjectID != nil
        var createdProjectID = resumeProjectID
        defer { if let id = createdProjectID { ActiveUploads.shared.remove(id) } }

        do {
            let uploadVideos: [LocalVideo]
            if resumeProjectID != nil {
                uploadVideos = selected
                say("Resuming upload…")
            } else {
                var preparedVideos: [LocalVideo] = []
                for (index, video) in selected.enumerated() {
                    // iCloud light: the original is still downloading; it is uploaded as it is.
                    if video.cloudAssetID != nil { preparedVideos.append(video); continue }
                    say("Preparing \(index + 1) of \(selected.count)…")
                    let title = selected.count > 1 ? "Preparing video \(index + 1) of \(selected.count)" : "Preparing your video"
                    stage(title)
                    let prepared = try await VideoPreparation.shared.prepare(
                        fileURL: video.url,
                        contentType: video.contentType
                    ) { [weak self] fraction in
                        Task { @MainActor in
                            self?.stage(title, fraction)
                            self?.logStep("prepare_progress", fraction)
                        }
                    }
                    if prepared.wasTranscoded { transientPreparedFiles.append(prepared.url) }
                    preparedVideos.append(LocalVideo(
                        url: prepared.url,
                        name: video.name,
                        duration: video.duration,
                        contentType: prepared.contentType
                    ))
                }
                uploadVideos = preparedVideos
            }

            let projectID: String
            if let resumeProjectID {
                projectID = resumeProjectID
            } else {
                say("Creating project…")
                stage("Creating your project")
                let createStarted = Date()
                let cleanTitle = title?.trimmingCharacters(in: .whitespacesAndNewlines)
                let project = try await Retry.run("project", onRetry: { [weak self] _, error in
                    self?.retried(Retry.short(error))
                }) {
                    try await appState.createProject(title: (cleanTitle?.isEmpty ?? true) ? nil : cleanTitle)
                }
                CutSellDiagnostics.log("project_created", ["seconds": String(format: "%.1f", Date().timeIntervalSince(createStarted))])
                createdProjectID = project.projectID
                ActiveUploads.shared.insert(project.projectID)
                currentProjectID = project.projectID
                projectID = project.projectID
                let pending = PendingCutRecord(
                    projectID: projectID,
                    title: project.title,
                    mode: mode,
                    audioOverlap: audioOverlap,
                    autoZoom: autoZoom,
                    videos: uploadVideos.map {
                        PendingCutVideo(
                            filePath: $0.url.path,
                            name: $0.name,
                            duration: $0.duration,
                            contentType: $0.contentType,
                            cloudAssetID: $0.cloudAssetID,
                            expectedBytes: $0.expectedBytes
                        )
                    },
                    createdAt: Date()
                )
                await PendingCutStore.shared.save(pending)
                pendingSaved = true
                transientPreparedFiles.removeAll()
            }

            currentProjectID = projectID
            // Build 23 (audio first): the cut starts from the voice (a few hundred KB) while the full
            // video uploads in the background; an older server keeps the whole-video flow below.
            if resumeProjectID == nil, await ServerCapabilities.audioFirst(),
               let jobID = try await startFromVoice(projectID: projectID, session: session, videos: uploadVideos,
                                                    audioOverlap: audioOverlap, autoZoom: autoZoom) {
                lastJobID = jobID
                report(1)
                try? await appState.refreshProjects()
                return projectID
            }
            // The whole-video flow needs every original on the iPhone (iCloud light falls back here).
            for (index, video) in uploadVideos.enumerated() where video.cloudAssetID != nil
                && !FileManager.default.fileExists(atPath: video.url.path) {
                let title = uploadVideos.count > 1 ? "Clip \(index + 1) of \(uploadVideos.count) · Downloading from iCloud" : "Downloading from iCloud"
                stage(title, 0)
                try await GalleryExport.downloadOriginal(assetID: video.cloudAssetID ?? "", to: video.url) { [weak self] fraction in
                    Task { @MainActor in self?.stage(title, fraction) }
                }
            }
            var sources: [SourceInput] = []
            for (index, video) in uploadVideos.enumerated() {
                say("Uploading \(index + 1) of \(uploadVideos.count)…")
                // Build 22: "Clip 2 of 3 · Uploading 40 %" (each clip its own upload).
                let uploadTitle = uploadVideos.count > 1 ? "Clip \(index + 1) of \(uploadVideos.count) · Uploading" : "Uploading your video"
                stage(uploadTitle, 0)
                let base = Double(index) / Double(uploadVideos.count)
                // D-133: bounded, diagnostics-only upload-timing log around
                // the EXISTING multipart upload call -- no networking change.
                let uploadStartedAt = Date()
                let fileMB = ((try? FileManager.default.attributesOfItem(atPath: video.url.path))?[.size] as? NSNumber)?.int64Value ?? 0
                CutSellDiagnostics.log("upload_started", [
                    "source_order": String(index),
                    "of": String(uploadVideos.count),
                    "content_type": video.contentType,
                    "mb": String(fileMB / 1_048_576),
                ])
                // Build 22: each clip is its own upload; when one fails it is tried again on its own (up to
                // 3 times, sending only the parts R2 does not have yet) -- the other clips are not touched.
                var clipAttempt = 1
                var clipResult: MultipartCompleteResponse?
                while clipResult == nil {
                    do {
                        clipResult = try await MultipartUploadManager.shared.upload(
                            fileURL: video.url,
                            projectID: projectID,
                            session: session,
                            contentType: video.contentType,
                            onRetry: { [weak self] error in Task { @MainActor in self?.retried(error) } },
                            onPartsUploaded: { Task { @MainActor in finishing.begin() } },
                            clipOrder: index,
                            progress: { [weak self] partProgress in
                                let overall = base + partProgress / Double(uploadVideos.count)
                                let lower = progressRange.lowerBound
                                Task { @MainActor in
                                    self?.progress = lower + span * min(1, max(0, overall))
                                    // Called only when more bytes went out: that is activity for the 90 s watch.
                                    self?.touch()
                                    self?.stage(uploadTitle, partProgress)
                                    self?.logStep(uploadVideos.count > 1 ? "upload_progress_clip\(index + 1)" : "upload_progress", partProgress)
                                }
                            }
                        )
                    } catch {
                        if error is CancellationError || Task.isCancelled || clipAttempt >= 3 || !Self.clipRetriable(error) {
                            CutSellDiagnostics.log("clip_upload_failed", Retry.fields(error, ["source_order": String(index),
                                                                                              "attempts": String(clipAttempt)]))
                            throw error
                        }
                        CutSellDiagnostics.log("clip_upload_retry", Retry.fields(error, ["source_order": String(index),
                                                                                         "attempt": String(clipAttempt)]))
                        retried(Retry.short(error))
                        clipAttempt += 1
                        try await Task.sleep(nanoseconds: 3_000_000_000)
                    }
                }
                guard let result = clipResult else { throw UploadError.incompleteUpload }
                // Keep this file on the phone for the editor's preview (no download later).
                await SourceVideoCache.shared.store(fileURL: video.url, uri: result.sourceURI)
                CutSellDiagnostics.log("upload_completed", [
                    "source_order": String(index),
                    "elapsed_s": String(format: "%.2f", Date().timeIntervalSince(uploadStartedAt)),
                ])
                sources.append(SourceInput(
                    originalName: video.url.lastPathComponent,
                    uri: result.sourceURI,
                    sourceOrder: index,
                    durationSec: video.duration
                ))
            }

            say("Starting AI edit…")
            stage("Starting your cut")
            let jobID = try await Self.startJob(projectID: projectID, session: session, sources: sources,
                                                audioOverlap: audioOverlap, autoZoom: autoZoom,
                                                onRetry: { [weak self] error in self?.retried(error) })
            lastJobID = jobID
            for video in uploadVideos { await MultipartUploadManager.shared.finished(fileURL: video.url, projectID: projectID) }
            await PendingCutStore.shared.remove(projectID: projectID, deleteLocalFiles: true)
            report(1)
            try? await appState.refreshProjects()
            return projectID
        } catch {
            CutSellDiagnostics.log("upload_failed", Retry.fields(error, ["reason": Retry.short(error), "step": stageTitle]))
            if !pendingSaved {
                for url in transientPreparedFiles { try? FileManager.default.removeItem(at: url) }
            }
            throw CutSubmitError(underlying: error, savedForResume: pendingSaved, projectID: pendingSaved ? createdProjectID : nil)
        }
    }
}

extension CutSubmitter {
    /// Build 23: voice of every clip → upload (small) → the cut starts → the videos go on uploading in
    /// the background (BackgroundVideoUploads). nil = a clip has no sound track (the whole-video flow
    /// is used instead). Every step is timed in Diagnostics.
    func startFromVoice(projectID: String, session: CutSellSession, videos: [LocalVideo], audioOverlap: Bool,
                        autoZoom: Bool) async throws -> String? {
        let count = videos.count
        func clip(_ index: Int, _ text: String) -> String { count > 1 ? "Clip \(index + 1) of \(count) · \(text)" : text }
        var voices: [URL] = []
        let extractStarted = Date()
        for (index, video) in videos.enumerated() {
            stage(clip(index, "Preparing…"), 0)
            do {
                voices.append(try await AudioExtractor.extract(from: video.previewURL ?? video.url) { [weak self] fraction in
                    Task { @MainActor in self?.stage(clip(index, "Preparing…"), fraction) }
                })
            } catch AudioExtractor.Failure.noAudioTrack {
                CutSellDiagnostics.log("audio_first_skipped", ["reason": "no_audio_track", "source_order": String(index)])
                return nil
            }
        }
        CutSellDiagnostics.log("stage_timing", ["stage": "audio_extract", "seconds": String(format: "%.1f", Date().timeIntervalSince(extractStarted))])
        var starts: [MultipartStartResponse] = []
        for video in videos {
            let start = try await MultipartUploadManager.shared.begin(fileURL: video.url, projectID: projectID, session: session,
                                                                      contentType: video.contentType, knownSize: video.expectedBytes)
            // The editor plays the iPhone's own copy (the small iCloud copy until the original is here):
            // no download, even before the upload ends.
            await SourceVideoCache.shared.store(fileURL: video.previewURL ?? video.url, uri: start.sourceURI)
            starts.append(start)
        }
        let voiceStarted = Date()
        var sources: [SourceInput] = []
        for (index, video) in videos.enumerated() {
            stage(clip(index, "Uploading audio…"), Double(index) / Double(count))
            let voiceURI = try await VoiceUpload.upload(fileURL: voices[index], projectID: projectID, session: session, order: index)
            try? FileManager.default.removeItem(at: voices[index])
            sources.append(SourceInput(originalName: video.url.lastPathComponent, uri: starts[index].sourceURI,
                                       sourceOrder: index, durationSec: video.duration, metadata: ["audio_uri": voiceURI]))
        }
        CutSellDiagnostics.log("stage_timing", ["stage": "audio_upload", "seconds": String(format: "%.1f", Date().timeIntervalSince(voiceStarted))])
        stage("Cutting your video…")
        let jobID = try await Self.startJob(projectID: projectID, session: session, sources: sources,
                                            audioOverlap: audioOverlap, autoZoom: autoZoom,
                                            onRetry: { [weak self] error in self?.retried(error) })
        await PendingCutStore.shared.markStarted(projectID: projectID, jobID: jobID)
        if let record = await PendingCutStore.shared.record(projectID: projectID) {
            BackgroundVideoUploads.shared.start(record: record, session: session)
        }
        CutSellDiagnostics.log("audio_first_started", ["clips": String(count), "job_id": jobID])
        return jobID
    }

    /// A clip upload worth trying again on its own (a refused request -- 4xx -- would only fail again).
    nonisolated static func clipRetriable(_ error: Error) -> Bool {
        if error is AIConsentError || error is UploadFlowError { return false }
        if case UploadError.emptyFile = error { return false }
        return Retry.retriable(error)
    }

    /// Starts the AI cut on the server for sources already on R2. Logs processing_started, then
    /// job_submitted or job_submit_failed with the real code and message.
    static func startJob(projectID: String, session: CutSellSession, sources: [SourceInput], audioOverlap: Bool,
                         autoZoom: Bool, onRetry: ((String) -> Void)? = nil) async throws -> String {
        struct Body: Encodable {
            let project_id: String
            let user_id: String
            let sources: [SourceInput]
            let language_hint: String?
            let audio_overlap: Bool
            /// Same choice under its canonical name (the API reads either).
            let dialogue_overlap_enabled: Bool
            let auto_zoom_enabled: Bool
        }
        CutSellDiagnostics.log("processing_started", ["sources": String(sources.count)])
        do {
            let response: FlowBSubmitResponse = try await Retry.run("start cut", onRetry: { _, error in
                onRetry?(Retry.short(error))
            }) { try await APIClient.shared.request(
                "/v1/flow-b/jobs",
                method: "POST",
                body: Body(project_id: projectID, user_id: session.userID, sources: sources, language_hint: nil,
                           audio_overlap: audioOverlap, dialogue_overlap_enabled: audioOverlap, auto_zoom_enabled: autoZoom),
                timeout: 30
            ) }
            CutSellDiagnostics.log("job_submitted", ["job_id": response.jobID])
            return response.jobID
        } catch {
            CutSellDiagnostics.log("job_submit_failed", Retry.fields(error))
            throw error
        }
    }
}

/// Build 22: projects whose upload is running in this process right now (the finisher never starts a
/// second copy of the same upload).
@MainActor
final class ActiveUploads {
    static let shared = ActiveUploads()
    private(set) var ids: Set<String> = []
    func insert(_ id: String) { ids.insert(id) }
    func remove(_ id: String) { ids.remove(id) }
    func contains(_ id: String) -> Bool { ids.contains(id) }
}

/// A failed submission. `savedForResume` tells whether the upload record was already saved
/// (Edits then offers "Resume upload"); the message is the original error's.
struct CutSubmitError: LocalizedError {
    let underlying: Error
    let savedForResume: Bool
    /// The project created before the failure (its upload can be resumed with "Try again").
    var projectID: String? = nil

    var errorDescription: String? { underlying.localizedDescription }
}
