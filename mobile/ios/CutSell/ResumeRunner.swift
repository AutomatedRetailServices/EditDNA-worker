import Foundation

/// Build 24: an upload stopped when CutSell was closed (parts still missing) continues by itself when
/// CutSell opens again -- only the missing parts are sent -- and its row in Edits · Processing shows it.
@MainActor
enum ResumeRunner {
    static func resume(_ record: PendingCutRecord, appState: AppState) async {
        guard !ActiveUploads.shared.contains(record.projectID) else { return }
        let footage = record.videos.reduce(0) { $0 + $1.duration }
        let item = ProcessingQueue.shared.add([(record.title.isEmpty ? "Interrupted Cut" : record.title, footage)]).first
        ProcessingQueue.shared.update(item, thumbnail: ThumbnailStore.shared.image(for: record.projectID),
                                      projectID: record.projectID, state: .uploading(0))
        let submitter = CutSubmitter()
        submitter.queueItem = item
        let videos = record.videos.map { LocalVideo(url: $0.fileURL, name: $0.name, duration: $0.duration, contentType: $0.contentType) }
        do {
            let id = try await submitter.submit(
                videos: videos, title: record.title, mode: record.mode, audioOverlap: record.audioOverlap,
                autoZoom: record.autoZoom ?? true, resumeProjectID: record.projectID, appState: appState
            )
            CutTracker.shared.track(projectID: id, jobID: submitter.lastJobID, title: record.title,
                                    fileName: videos.count == 1 ? videos[0].name : "\(videos.count) clips",
                                    thumbnail: nil, footageSeconds: footage)
            ProcessingQueue.shared.update(item, projectID: id, state: .processing(0))
        } catch {
            let detail = CreateFlowView.plain((error as? CutSubmitError)?.underlying ?? error)
            ProcessingQueue.shared.fail(item, detail: detail) {
                if let item { ProcessingQueue.shared.remove(item) }
                Task { @MainActor in await resume(record, appState: appState) }
            }
        }
    }
}
