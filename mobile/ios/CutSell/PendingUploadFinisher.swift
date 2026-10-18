import Foundation

/// Build 22: finishes an upload whose parts are all on R2 but whose cut never started, without the
/// creator doing anything:
/// - when CutSell opens or comes back to the screen;
/// - when iOS wakes CutSell in the background because the last parts finished (the creator left the
///   app mid-upload), with extra time asked from iOS for "complete" and the start of the processing.
///
/// On Swanny's iPhone (18 Oct, build 21) all three uploads reached R2, but iOS suspended CutSell right
/// after "complete" was sent; nothing started the cut afterwards and "Try again" uploaded everything
/// again from 0. A project already uploading in this process (ActiveUploads) is never touched.
@MainActor
enum PendingUploadFinisher {
    private static var running = false

    /// `session` nil = the one saved in the Keychain (a background wake can come before the screen
    /// has loaded the account).
    @discardableResult
    static func run(session: CutSellSession?, reason: String,
                    resumeUpload: ((PendingCutRecord) async -> Void)? = nil) async -> Int {
        guard !running else { return 0 }
        running = true
        defer { running = false }
        var current = session
        if current == nil, let saved = try? KeychainStore.load() {
            current = saved
            await APIClient.shared.setSession(saved)
        }
        guard let current else { return 0 }
        UploadLedger.shared.prune()
        var finished = 0

        for record in await PendingCutStore.shared.list() where !ActiveUploads.shared.contains(record.projectID) {
            if BackgroundVideoUploads.shared.isRunning(record.projectID) { continue }
            // Build 23: the cut already started from the voice; only the video is left: it goes on
            // uploading by itself (only the parts R2 does not have yet).
            if record.jobID != nil {
                CutSellDiagnostics.log("video_background_resumed", ["reason": reason])
                BackgroundVideoUploads.shared.start(record: record, session: current)
                continue
            }
            var ready = true
            for video in record.videos {
                let state = await MultipartUploadManager.shared.readiness(fileURL: video.fileURL, projectID: record.projectID,
                                                                         userID: current.userID)
                if state != .allPartsUploaded && state != .completed { ready = false }
            }
            // Build 24: parts still missing -- the upload goes on by itself on reopen (only the missing
            // parts are sent); without a resumer (background wake) it waits for CutSell to open.
            guard ready else {
                if let resumeUpload {
                    CutSellDiagnostics.log("upload_auto_resumed", ["reason": reason, "videos": String(record.videos.count)])
                    await resumeUpload(record)
                }
                continue
            }
            ActiveUploads.shared.insert(record.projectID)
            let time = BackgroundTime("cutsell-auto-finish")
            time.begin()
            CutSellDiagnostics.log("upload_auto_finish_started", ["reason": reason, "videos": String(record.videos.count)])
            do {
                var sources: [SourceInput] = []
                for (index, video) in record.videos.enumerated() {
                    let result = try await MultipartUploadManager.shared.upload(
                        fileURL: video.fileURL, projectID: record.projectID, session: current,
                        contentType: video.contentType
                    ) { _ in }
                    sources.append(SourceInput(originalName: video.fileURL.lastPathComponent, uri: result.sourceURI,
                                               sourceOrder: index, durationSec: video.duration))
                }
                let jobID = try await CutSubmitter.startJob(projectID: record.projectID, session: current, sources: sources,
                                                            audioOverlap: record.audioOverlap, autoZoom: record.autoZoom ?? true)
                for (index, video) in record.videos.enumerated() {
                    await SourceVideoCache.shared.store(fileURL: video.fileURL, uri: sources[index].uri)
                    await MultipartUploadManager.shared.finished(fileURL: video.fileURL, projectID: record.projectID)
                }
                await PendingCutStore.shared.remove(projectID: record.projectID, deleteLocalFiles: true)
                CutSellDiagnostics.log("upload_auto_finish_done", ["reason": reason, "job_id": jobID])
                finished += 1
            } catch {
                CutSellDiagnostics.log("upload_auto_finish_failed", Retry.fields(error, ["reason": reason]))
            }
            ActiveUploads.shared.remove(record.projectID)
            time.end()
        }
        return finished
    }

    // MARK: iOS woke CutSell in the background (parts finished in its upload service).

    private static var wakeTime: BackgroundTime?

    /// Called from the app delegate: iOS gives a little time; it is kept until the finisher is done.
    static func wokeForUploads() {
        if wakeTime == nil {
            let time = BackgroundTime("cutsell-wake")
            time.begin()
            wakeTime = time
        }
        CutSellDiagnostics.log("upload_background_wake", [:])
    }

    /// Called when iOS has delivered every finished part (urlSessionDidFinishEvents).
    static func uploadEventsDelivered() {
        Task { @MainActor in
            // An upload still running in this process continues by itself (its parts just resumed);
            // a relaunched CutSell has no such upload: the finisher completes it.
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            await run(session: nil, reason: "background_wake")
            // Give the running upload time to send "complete" and start the cut before iOS suspends.
            var waited = 0
            while !ActiveUploads.shared.ids.isEmpty && waited < 25 {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                waited += 1
            }
            wakeTime?.end()
            wakeTime = nil
        }
    }
}
