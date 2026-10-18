import Foundation

/// Build 23 ("audio first"): the cut starts from the voice while the full video uploads in the
/// background. Used only when the server says it can (GET /v1/capabilities); an older server keeps
/// the build 22 flow (whole video first).
@MainActor
enum ServerCapabilities {
    private static var cached: (value: Bool, at: Date)?

    static func audioFirst() async -> Bool {
        if UserDefaults.standard.bool(forKey: "cutsell.audioFirst.off") { return false }
        if let cached, Date().timeIntervalSince(cached.at) < 600 { return cached.value }
        struct Answer: Decodable { let audio_first: Bool? }
        let value: Bool
        do {
            let answer: Answer = try await APIClient.shared.request("/v1/capabilities", timeout: 6)
            value = answer.audio_first == true
        } catch {
            value = false
        }
        cached = (value, Date())
        CutSellDiagnostics.log("server_capabilities", ["audio_first": value ? "yes" : "no"])
        return value
    }
}

/// The small voice file of one clip, uploaded with one signed PUT (it is ~0.5 MB per minute).
enum VoiceUpload {
    struct PresignBody: Encodable {
        let project_id: String
        let user_id: String
        let original_name: String
        let content_type: String
        let size_bytes: Int64
    }

    struct Presign: Decodable {
        let method: String
        let upload_url: String
        let headers: [String: String]?
        let source_uri: String
        let content_type: String
    }

    /// Returns the voice file's source URI.
    static func upload(fileURL: URL, projectID: String, session: CutSellSession, order: Int) async throws -> String {
        let size = ((try? FileManager.default.attributesOfItem(atPath: fileURL.path))?[.size] as? NSNumber)?.int64Value ?? 0
        guard size > 0 else { throw UploadError.emptyFile }
        let started = Date()
        let uri: String = try await Retry.run("voice \(order)") {
            // An audio-only MP4 (AAC): the server's video allowlist accepts it as video/mp4.
            let presign: Presign = try await APIClient.shared.request(
                "/v1/uploads/presign", method: "POST",
                body: PresignBody(project_id: projectID, user_id: session.userID, original_name: "voice-\(order + 1).mp4",
                                  content_type: "video/mp4", size_bytes: size),
                timeout: 20
            )
            guard DirectPutUpload.isPut(presign.method), let url = URL(string: presign.upload_url) else {
                throw APIError.http(-1, "storage does not take a direct PUT")
            }
            let status = try await DirectPutUpload.send(fileURL: fileURL, to: url, contentType: presign.content_type,
                                                        headers: presign.headers ?? [:])
            guard (200..<300).contains(status) else { throw BackgroundUploadError.http(status) }
            return presign.source_uri
        }
        CutSellDiagnostics.log("audio_uploaded", ["source_order": String(order), "kb": String(size / 1024),
                                                  "seconds": String(format: "%.1f", Date().timeIntervalSince(started))])
        return uri
    }
}

/// Build 23: the full videos of cuts that already started from their voice, uploading in the
/// background (each clip its own resumable multipart upload). Export waits for them; the editor plays
/// the iPhone's own copy meanwhile.
@MainActor
final class BackgroundVideoUploads: ObservableObject {
    static let shared = BackgroundVideoUploads()

    struct Status: Equatable {
        var fraction: Double = 0
        var clip = 1
        var clips = 1
        var failed: String?
        var startedAt = Date()
        var bytesPerSecond: Double = 0
        var totalBytes: Int64 = 0

        /// "Uploading video in background · 40 % · about 1 min left"
        var text: String {
            if let failed { return "Video upload paused (\(failed)). It continues when you open CutSell." }
            let clipPart = clips > 1 ? "Clip \(clip) of \(clips) · " : ""
            var line = "\(clipPart)Uploading video in background · \(Int(fraction * 100)) %"
            if bytesPerSecond > 0, totalBytes > 0 {
                let left = Double(totalBytes) * (1 - fraction) / bytesPerSecond
                line += " · " + Self.eta(left)
            }
            return line
        }

        static func eta(_ seconds: Double) -> String {
            if seconds < 60 { return "under a minute left" }
            let minutes = Int((seconds / 60).rounded(.up))
            return "about \(minutes) min left"
        }
    }

    @Published private(set) var statuses: [String: Status] = [:]
    private var tasks: [String: Task<Void, Never>] = [:]

    func status(_ projectID: String) -> Status? { statuses[projectID] }
    func isUploading(_ projectID: String) -> Bool { statuses[projectID].map { $0.failed == nil } ?? false }
    func isRunning(_ projectID: String) -> Bool { tasks[projectID] != nil }

    /// Starts (or resumes) the background upload of a cut's videos. Safe to call twice.
    func start(record: PendingCutRecord, session: CutSellSession) {
        let projectID = record.projectID
        if tasks[projectID] != nil { return }
        let total = record.videos.reduce(Int64(0)) { sum, video in
            sum + ((((try? FileManager.default.attributesOfItem(atPath: video.fileURL.path))?[.size]) as? NSNumber)?.int64Value ?? 0)
        }
        statuses[projectID] = Status(clips: record.videos.count, totalBytes: total)
        ActiveUploads.shared.insert(projectID)
        tasks[projectID] = Task { @MainActor [weak self] in
            let time = BackgroundTime("cutsell-video-upload")
            time.begin()
            defer { time.end(); ActiveUploads.shared.remove(projectID); self?.tasks[projectID] = nil }
            let started = Date()
            CutSellDiagnostics.log("video_background_started", ["clips": String(record.videos.count), "mb": String(total / 1_048_576)])
            do {
                for (index, video) in record.videos.enumerated() {
                    self?.statuses[projectID]?.clip = index + 1
                    let base = Double(index) / Double(record.videos.count)
                    var attempt = 1
                    var result: MultipartCompleteResponse?
                    while result == nil {
                        do {
                            result = try await MultipartUploadManager.shared.upload(
                                fileURL: video.fileURL, projectID: projectID, session: session, contentType: video.contentType,
                                clipOrder: index
                            ) { fraction in
                                Task { @MainActor in
                                    guard var status = self?.statuses[projectID] else { return }
                                    status.fraction = base + fraction / Double(record.videos.count)
                                    let elapsed = Date().timeIntervalSince(started)
                                    if elapsed > 3 { status.bytesPerSecond = Double(total) * status.fraction / elapsed }
                                    self?.statuses[projectID] = status
                                }
                            }
                        } catch {
                            if attempt >= 3 || !CutSubmitter.clipRetriable(error) { throw error }
                            CutSellDiagnostics.log("clip_upload_retry", Retry.fields(error, ["source_order": String(index),
                                                                                             "attempt": String(attempt)]))
                            attempt += 1
                            try await Task.sleep(nanoseconds: 3_000_000_000)
                        }
                    }
                    guard let result else { continue }
                    try await Self.videoReady(projectID: projectID, session: session, uri: result.sourceURI, order: index)
                }
                for video in record.videos {
                    await MultipartUploadManager.shared.finished(fileURL: video.fileURL, projectID: projectID)
                }
                await PendingCutStore.shared.remove(projectID: projectID, deleteLocalFiles: true)
                CutSellDiagnostics.log("video_background_done", ["seconds": String(format: "%.0f", Date().timeIntervalSince(started))])
                self?.statuses[projectID] = nil
            } catch {
                CutSellDiagnostics.log("video_background_failed", Retry.fields(error))
                self?.statuses[projectID]?.failed = Retry.short(error)
            }
        }
    }

    private static func videoReady(projectID: String, session: CutSellSession, uri: String, order: Int) async throws {
        struct Body: Encodable { let user_id: String; let uri: String; let source_order: Int }
        struct Answer: Decodable { let job_id: String? }
        let _: Answer = try await Retry.run("video ready") {
            try await APIClient.shared.request("/v1/projects/\(projectID)/video-ready", method: "POST",
                                               body: Body(user_id: session.userID, uri: uri, source_order: order), timeout: 20)
        }
        CutSellDiagnostics.log("video_ready_sent", ["source_order": String(order)])
    }
}
