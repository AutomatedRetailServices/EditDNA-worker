import Foundation

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

    /// Shows a step that happens before uploading (e.g. copying from Photos).
    func announce(_ text: String) {
        statusText = text
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
        resumeProjectID: String?,
        appState: AppState,
        progressRange: ClosedRange<Double> = 0...1,
        label: String = ""
    ) async throws -> String {
        guard let session = appState.session else { throw APIError.invalidResponse }
        isSubmitting = true
        defer { isSubmitting = false }
        let span = progressRange.upperBound - progressRange.lowerBound
        func report(_ value: Double) { progress = progressRange.lowerBound + span * min(1, max(0, value)) }
        func say(_ text: String) { statusText = label.isEmpty ? text : "\(label) · \(text)" }
        report(0)
        var transientPreparedFiles: [URL] = []
        var pendingSaved = resumeProjectID != nil

        do {
            let uploadVideos: [LocalVideo]
            if resumeProjectID != nil {
                uploadVideos = selected
                say("Resuming upload…")
            } else {
                var preparedVideos: [LocalVideo] = []
                for (index, video) in selected.enumerated() {
                    say("Preparing \(index + 1) of \(selected.count)…")
                    let prepared = try await VideoPreparation.shared.prepare(
                        fileURL: video.url,
                        contentType: video.contentType
                    )
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
                let cleanTitle = title?.trimmingCharacters(in: .whitespacesAndNewlines)
                let project = try await appState.createProject(
                    title: (cleanTitle?.isEmpty ?? true) ? nil : cleanTitle
                )
                projectID = project.projectID
                let pending = PendingCutRecord(
                    projectID: projectID,
                    title: project.title,
                    mode: mode,
                    audioOverlap: audioOverlap,
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

            var sources: [SourceInput] = []
            for (index, video) in uploadVideos.enumerated() {
                say("Uploading \(index + 1) of \(uploadVideos.count)…")
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
                    contentType: video.contentType
                ) { [weak self] partProgress in
                    let overall = base + partProgress / Double(uploadVideos.count)
                    let lower = progressRange.lowerBound
                    Task { @MainActor in self?.progress = lower + span * min(1, max(0, overall)) }
                }
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
            struct Body: Encodable {
                let project_id: String
                let user_id: String
                let sources: [SourceInput]
                let language_hint: String?
                let audio_overlap: Bool
            }
            let submitResponse: FlowBSubmitResponse = try await APIClient.shared.request(
                "/v1/flow-b/jobs",
                method: "POST",
                body: Body(
                    project_id: projectID,
                    user_id: session.userID,
                    sources: sources,
                    language_hint: nil,
                    audio_overlap: audioOverlap
                )
            )
            // D-133: bounded diagnostics only -- logs the backend job id.
            CutSellDiagnostics.log("job_submitted", ["job_id": submitResponse.jobID])
            lastJobID = submitResponse.jobID
            await PendingCutStore.shared.remove(projectID: projectID, deleteLocalFiles: true)
            report(1)
            try? await appState.refreshProjects()
            return projectID
        } catch {
            CutSellDiagnostics.log("cut_submit_failed", ["reason": String(describing: type(of: error))])
            if !pendingSaved {
                for url in transientPreparedFiles { try? FileManager.default.removeItem(at: url) }
            }
            throw CutSubmitError(underlying: error, savedForResume: pendingSaved)
        }
    }
}

/// A failed submission. `savedForResume` tells whether the upload record was already saved
/// (Edits then offers "Resume upload"); the message is the original error's.
struct CutSubmitError: LocalizedError {
    let underlying: Error
    let savedForResume: Bool

    var errorDescription: String? { underlying.localizedDescription }
}
