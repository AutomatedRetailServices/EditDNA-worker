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
        guard let session = appState.session else { throw APIError.invalidResponse }
        // Nothing is uploaded or processed before the AI notice is agreed to (once).
        guard await AIConsentCenter.shared.ensure() else { throw AIConsentError.declined }
        isSubmitting = true
        // Build 21: extra time from iOS when the creator leaves CutSell mid-upload, so every part is
        // handed to the background upload service (which then keeps sending with the app closed).
        let backgroundTime = BackgroundTime()
        backgroundTime.begin()
        defer { isSubmitting = false; backgroundTime.end() }
        let span = progressRange.upperBound - progressRange.lowerBound
        func report(_ value: Double) { progress = progressRange.lowerBound + span * min(1, max(0, value)) }
        func say(_ text: String) { statusText = label.isEmpty ? text : "\(label) · \(text)" }
        report(0)
        var transientPreparedFiles: [URL] = []
        var pendingSaved = resumeProjectID != nil
        var createdProjectID = resumeProjectID

        do {
            let uploadVideos: [LocalVideo]
            if resumeProjectID != nil {
                uploadVideos = selected
                say("Resuming upload…")
            } else {
                var preparedVideos: [LocalVideo] = []
                for (index, video) in selected.enumerated() {
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
                            contentType: $0.contentType
                        )
                    },
                    createdAt: Date()
                )
                await PendingCutStore.shared.save(pending)
                pendingSaved = true
                transientPreparedFiles.removeAll()
            }

            currentProjectID = projectID
            var sources: [SourceInput] = []
            for (index, video) in uploadVideos.enumerated() {
                say("Uploading \(index + 1) of \(uploadVideos.count)…")
                let uploadTitle = uploadVideos.count > 1 ? "Uploading video \(index + 1) of \(uploadVideos.count)" : "Uploading your video"
                stage(uploadTitle, 0)
                let base = Double(index) / Double(uploadVideos.count)
                // D-133: bounded, diagnostics-only upload-timing log around
                // the EXISTING multipart upload call -- no networking change.
                let uploadStartedAt = Date()
                CutSellDiagnostics.log("upload_started", [
                    "source_order": String(index),
                    "content_type": video.contentType,
                ])
                let result = try await MultipartUploadManager.shared.upload(
                    fileURL: video.url,
                    projectID: projectID,
                    session: session,
                    contentType: video.contentType,
                    onRetry: { [weak self] error in Task { @MainActor in self?.retried(error) } }
                ) { [weak self] partProgress in
                    let overall = base + partProgress / Double(uploadVideos.count)
                    let lower = progressRange.lowerBound
                    Task { @MainActor in
                        self?.progress = lower + span * min(1, max(0, overall))
                        // Called only when more bytes went out: that is activity for the 90 s watch.
                        self?.touch()
                        self?.stage(uploadTitle, partProgress)
                        self?.logStep("upload_progress", partProgress)
                    }
                }
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
            let submitResponse: FlowBSubmitResponse = try await Retry.run("start cut", onRetry: { [weak self] _, error in
                self?.retried(Retry.short(error))
            }) { try await APIClient.shared.request(
                "/v1/flow-b/jobs",
                method: "POST",
                body: Body(
                    project_id: projectID,
                    user_id: session.userID,
                    sources: sources,
                    language_hint: nil,
                    audio_overlap: audioOverlap,
                    dialogue_overlap_enabled: audioOverlap,
                    auto_zoom_enabled: autoZoom
                ),
                timeout: 20
            ) }
            // D-133: bounded diagnostics only -- logs the backend job id.
            CutSellDiagnostics.log("job_submitted", ["job_id": submitResponse.jobID])
            lastJobID = submitResponse.jobID
            await PendingCutStore.shared.remove(projectID: projectID, deleteLocalFiles: true)
            report(1)
            try? await appState.refreshProjects()
            return projectID
        } catch {
            CutSellDiagnostics.log("upload_failed", ["reason": Retry.short(error), "type": String(describing: type(of: error)),
                                                     "step": stageTitle])
            if !pendingSaved {
                for url in transientPreparedFiles { try? FileManager.default.removeItem(at: url) }
            }
            throw CutSubmitError(underlying: error, savedForResume: pendingSaved, projectID: pendingSaved ? createdProjectID : nil)
        }
    }
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
