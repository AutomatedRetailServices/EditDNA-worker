import CryptoKit
import Foundation

struct MultipartStartResponse: Codable {
    let uploadID: String
    let projectID: String
    let userID: String
    let sourceURI: String
    let contentType: String
    let sizeBytes: Int64
    let partSize: Int
    let partCount: Int

    enum CodingKeys: String, CodingKey {
        case uploadID = "upload_id"
        case projectID = "project_id"
        case userID = "user_id"
        case sourceURI = "source_uri"
        case contentType = "content_type"
        case sizeBytes = "size_bytes"
        case partSize = "part_size"
        case partCount = "part_count"
    }
}

struct MultipartPartSignResponse: Codable {
    let uploadID: String
    let partNumber: Int
    let uploadURL: URL

    enum CodingKeys: String, CodingKey {
        case uploadID = "upload_id"
        case partNumber = "part_number"
        case uploadURL = "upload_url"
    }
}

struct MultipartUploadedPart: Codable, Hashable {
    let partNumber: Int
    let etag: String

    enum CodingKeys: String, CodingKey {
        case partNumber = "part_number"
        case etag
    }
}

struct MultipartStatusResponse: Codable {
    let uploadID: String
    let sourceURI: String
    let partCount: Int
    let uploadedParts: [MultipartUploadedPart]
    let uploadedPartNumbers: [Int]
    /// Build 22 server: "completed" when the file is already whole on R2 (absent on older servers).
    var state: String? = nil

    enum CodingKeys: String, CodingKey {
        case state
        case uploadID = "upload_id"
        case sourceURI = "source_uri"
        case partCount = "part_count"
        case uploadedParts = "uploaded_parts"
        case uploadedPartNumbers = "uploaded_part_numbers"
    }
}

struct MultipartCompleteResponse: Codable {
    let uploadID: String
    let state: String
    let sourceURI: String
    let sizeBytes: Int64

    enum CodingKeys: String, CodingKey {
        case uploadID = "upload_id"
        case state
        case sourceURI = "source_uri"
        case sizeBytes = "size_bytes"
    }
}

actor MultipartUploadManager {
    static let shared = MultipartUploadManager()
    private let api = APIClient.shared
    private let resumeStore = UploadResumeStore.shared
    static let requestedPartSize = 8 * 1024 * 1024

    func upload(
        fileURL: URL,
        projectID: String,
        session: CutSellSession,
        contentType: String,
        existingUploadID: String? = nil,
        /// Short uploads made while the editor is open (a B-roll video): sent directly, not
        /// through the background session that keeps a long cut upload going when CutSell closes.
        foreground: Bool = false,
        /// Build 19: each retry (what failed, in plain words), so the screen and Diagnostics show it.
        onRetry: (@Sendable (String) -> Void)? = nil,
        /// Build 22: every part is on R2 and "complete" is about to be sent (the caller asks iOS for
        /// time so "complete" and the start of the processing finish even if CutSell is left now).
        onPartsUploaded: (@Sendable () -> Void)? = nil,
        /// Build 22: the clip's place in a Combine / Multiple cut (0-based), written on every
        /// Diagnostics event of this upload as source_order.
        clipOrder: Int? = nil,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> MultipartCompleteResponse {
        let clip: [String: String] = clipOrder.map { ["source_order": String($0)] } ?? [:]
        func tagged(_ fields: [String: String])) -> [String: String] { fields.merging(clip) { a, _ in a } }
        let retried: @Sendable (Int, Error) -> Void = { _, error in onRetry?(Retry.short(error)) }
        let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        guard size > 0 else { throw UploadError.emptyFile }

        let start: MultipartStartResponse
        var completedParts: [Int: String] = [:]

        let saved: UploadResumeRecord?
        if let existingUploadID {
            saved = await resumeStore.record(uploadID: existingUploadID)
        } else {
            saved = await resumeStore.record(fileURL: fileURL, projectID: projectID, size: size)
        }

        if let saved,
           saved.start.projectID == projectID,
           saved.start.userID == session.userID,
           saved.start.sizeBytes == size,
           let done = UploadLedger.shared.entry(saved.start.uploadID), done.completedAt != nil {
            // Build 22: R2 already has the whole file (the app was closed before it started the cut).
            CutSellDiagnostics.log("upload_already_complete", tagged(["parts": String(done.partCount)]))
            progress(1)
            return Self.completed(saved.start, sourceURI: done.sourceURI)
        }

        if let saved,
           saved.start.projectID == projectID,
           saved.start.userID == session.userID,
           saved.start.sizeBytes == size {
            do {
                let status: MultipartStatusResponse = try await Retry.run("status", onRetry: retried) {
                    try await api.request(
                        "/v1/uploads/multipart/\(saved.start.uploadID)",
                        query: [
                            URLQueryItem(name: "project_id", value: projectID),
                            URLQueryItem(name: "user_id", value: session.userID)
                        ],
                        timeout: 20
                    )
                }
                if status.state == "completed" {
                    UploadLedger.shared.markCompleted(saved.start.uploadID, sourceURI: status.sourceURI)
                    CutSellDiagnostics.log("upload_complete_done", tagged(["how": "server_says_completed"]))
                    progress(1)
                    return Self.completed(saved.start, sourceURI: status.sourceURI)
                }
                guard status.partCount == saved.start.partCount else {
                    throw UploadError.resumeMetadataMismatch
                }
                let known = UploadLedger.shared.entry(saved.start.uploadID)?.etags ?? [:]
                for part in status.uploadedParts {
                    // R2's own list (ListParts) decides which parts are there; the ETag saved on the
                    // iPhone fills in only if the list gave none.
                    completedParts[part.partNumber] = part.etag.isEmpty ? (known[part.partNumber] ?? "") : part.etag
                }
                completedParts = completedParts.filter { !$0.value.isEmpty }
                start = saved.start
                // D-133: bounded, diagnostics-only resume-detection log --
                // no change to the existing reconciliation logic above.
                CutSellDiagnostics.log("upload_resumed", tagged([
                    "resumed_part_count": String(completedParts.count),
                    "total_part_count": String(start.partCount),
                    "etags_on_iphone": String(known.count),
                ]))
            } catch APIError.http(let code, let message) where code == 404 {
                // Build 22: the server closes the session when "complete" succeeds. If this iPhone had
                // already sent "complete", the file IS on R2 (only the answer was lost): never upload it again.
                if UploadLedger.shared.entry(saved.start.uploadID)?.completeSentAt != nil {
                    UploadLedger.shared.markCompleted(saved.start.uploadID, sourceURI: saved.start.sourceURI)
                    CutSellDiagnostics.log("upload_complete_done", tagged(["how": "already_completed_on_server",
                                                                    "detail": Retry.clip(message)]))
                    progress(1)
                    return Self.completed(saved.start, sourceURI: saved.start.sourceURI)
                }
                CutSellDiagnostics.log("upload_session_gone", tagged(["detail": Retry.clip(message),
                                                               "parts_lost": String(saved.start.partCount)]))
                await resumeStore.remove(uploadID: saved.start.uploadID)
                start = try await Retry.run("start", onRetry: retried) {
                    try await createStart(fileURL: fileURL, projectID: projectID, session: session,
                                          contentType: contentType, size: size)
                }
            }
        } else {
            start = try await Retry.run("start", onRetry: retried) {
                try await createStart(fileURL: fileURL, projectID: projectID, session: session,
                                      contentType: contentType, size: size)
            }
        }
        CutSellDiagnostics.log("upload_parts", tagged(["parts": String(start.partCount), "part_mb": String(start.partSize / 1_048_576),
                                                "mb": String(size / 1_048_576), "done": String(completedParts.count)]))

        await resumeStore.save(start: start, fileURL: fileURL, projectID: projectID, size: size)
        UploadLedger.shared.begin(start)
        for (number, etag) in completedParts { UploadLedger.shared.recordPart(uploadID: start.uploadID, part: number, etag: etag) }
        // The real %: bytes of finished parts + bytes the parts in flight have sent.
        let already = completedParts.keys.reduce(Int64(0)) { sum, number in
            sum + Int64(min(start.partSize, max(0, Int(size) - (number - 1) * start.partSize)))
        }
        let tally = ByteTally(total: size, alreadyUploaded: already, report: progress)
        progress(Double(already) / Double(max(1, size)))

        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }

        // Build 21: every missing part is written to its own small file and queued at once in the
        // background URLSession (iOS sends 3 at a time and keeps going with CutSell in the background
        // or the screen locked). Each part is retried on its own, up to 3 times; parts already
        // uploaded are never sent again. Nothing here cancels a part that is sending.
        let missing = (1...start.partCount).filter { completedParts[$0] == nil }
        let folder = try partsFolder(uploadID: start.uploadID)
        let uploadID = start.uploadID
        let userID = session.userID
        let useForeground = foreground
        try await withThrowingTaskGroup(of: (Int, String).self) { group in
            for partNumber in missing {
                let offset = Int64(partNumber - 1) * Int64(start.partSize)
                let readCount = Int(min(Int64(start.partSize), size - offset))
                try handle.seek(toOffset: UInt64(offset))
                let data = try handle.read(upToCount: readCount) ?? Data()
                guard !data.isEmpty else { throw UploadError.missingPart(partNumber) }
                let partFile = folder.appendingPathComponent(String(format: "part-%05d.bin", partNumber))
                try data.write(to: partFile, options: .atomic)
                let partBytes = Int64(data.count)
                group.addTask {
                    struct SignBody: Encodable { let project_id: String; let user_id: String; let expires_in: Int }
                    defer { try? FileManager.default.removeItem(at: partFile) }
                    // A fresh link and the PUT, tried 3 times (a link can expire while a part waits).
                    let etagValue: String = try await Retry.run("part \(partNumber)", onRetry: { attempt, error in
                        tally.reset(part: partNumber)
                        retried(attempt, error)
                    }) {
                        let signed: MultipartPartSignResponse = try await self.api.request(
                            "/v1/uploads/multipart/\(uploadID)/parts/\(partNumber)/presign",
                            method: "POST",
                            // expires_in: 1 h once the server reads it (the current server keeps 15 min).
                            body: SignBody(project_id: projectID, user_id: userID, expires_in: 3600),
                            timeout: 20
                        )
                        let response = try await Self.uploadPart(
                            file: partFile, bytes: partBytes, partNumber: partNumber, signedURL: signed.uploadURL,
                            uploadID: uploadID, foreground: useForeground, onBytes: { tally.sent(part: partNumber, bytes: $0) }
                        )
                        guard let etag = response.value(forHTTPHeaderField: "ETag"), !etag.isEmpty else {
                            throw UploadError.missingETag(partNumber)
                        }
                        UploadLedger.shared.recordPart(uploadID: uploadID, part: partNumber, etag: etag)
                        return etag
                    }
                    tally.done(part: partNumber, size: partBytes)
                    CutSellDiagnostics.log("upload_part_done", ["part": String(partNumber), "of": String(start.partCount)]
                        .merging(clip) { a, _ in a })
                    return (partNumber, etagValue)
                }
            }
            while let (partNumber, etag) = try await group.next() {
                completedParts[partNumber] = etag
            }
        }
        try? FileManager.default.removeItem(at: folder)

        struct CompleteBody: Encodable {
            let project_id: String
            let user_id: String
            let parts: [MultipartUploadedPart]
        }
        let parts = completedParts
            .map { MultipartUploadedPart(partNumber: $0.key, etag: $0.value) }
            .sorted { $0.partNumber < $1.partNumber }
        guard parts.count == start.partCount else { throw UploadError.incompleteUpload }

        // Build 22: the final step is visible in Diagnostics, with the real code and message.
        onPartsUploaded?()
        let sentBefore = UploadLedger.shared.entry(start.uploadID)?.completeSentAt != nil
        UploadLedger.shared.markCompleteSent(start.uploadID)
        CutSellDiagnostics.log("upload_complete_started", tagged(["parts": String(parts.count), "mb": String(size / 1_048_576),
                                                          "part_mb": String(start.partSize / 1_048_576)]))
        let completeStarted = Date()
        let sends = Counter()
        let result: MultipartCompleteResponse
        do {
            result = try await Retry.run("complete", onRetry: retried) {
                sends.add()
                return try await api.request(
                    "/v1/uploads/multipart/\(start.uploadID)/complete",
                    method: "POST",
                    body: CompleteBody(project_id: projectID, user_id: session.userID, parts: parts),
                    timeout: 45
                )
            }
            CutSellDiagnostics.log("upload_complete_done", tagged(["state": result.state,
                                                           "seconds": String(format: "%.1f", Date().timeIntervalSince(completeStarted))]))
        } catch APIError.http(let code, let message) where code == 404 && (sentBefore || sends.value > 1) {
            // A "complete" sent earlier succeeded on R2 and the server closed the session; its answer
            // never reached the iPhone (CutSell was suspended). The file is there.
            CutSellDiagnostics.log("upload_complete_done", tagged(["how": "already_completed_on_server", "detail": Retry.clip(message)]))
            result = Self.completed(start, sourceURI: start.sourceURI)
        } catch {
            CutSellDiagnostics.log("upload_complete_failed", tagged(Retry.fields(error)))
            throw error
        }
        UploadLedger.shared.markCompleted(start.uploadID, sourceURI: result.sourceURI)
        progress(1)
        return result
    }

    /// The answer "complete" gives, for an upload R2 already finished.
    static func completed(_ start: MultipartStartResponse, sourceURI: String) -> MultipartCompleteResponse {
        MultipartCompleteResponse(uploadID: start.uploadID, state: "uploaded",
                                  sourceURI: sourceURI.isEmpty ? start.sourceURI : sourceURI, sizeBytes: start.sizeBytes)
    }

    enum Readiness: Equatable { case notStarted, partsMissing, allPartsUploaded, completed }

    /// Build 22: what this iPhone knows about a video's upload (no network): the finisher completes
    /// only uploads whose parts are all on R2.
    func readiness(fileURL: URL, projectID: String, userID: String) async -> Readiness {
        let size = ((try? FileManager.default.attributesOfItem(atPath: fileURL.path))?[.size] as? NSNumber)?.int64Value ?? 0
        guard size > 0, let record = await resumeStore.record(fileURL: fileURL, projectID: projectID, size: size),
              record.start.userID == userID else { return .notStarted }
        guard let entry = UploadLedger.shared.entry(record.start.uploadID) else { return .partsMissing }
        if entry.completedAt != nil { return .completed }
        if entry.completeSentAt != nil || entry.etags.count >= entry.partCount { return .allPartsUploaded }
        return .partsMissing
    }

    /// Build 22: called after the cut started on the server (until then the upload stays resumable).
    func finished(fileURL: URL, projectID: String) async {
        let size = ((try? FileManager.default.attributesOfItem(atPath: fileURL.path))?[.size] as? NSNumber)?.int64Value ?? 0
        if let record = await resumeStore.record(fileURL: fileURL, projectID: projectID, size: size) {
            UploadLedger.shared.remove(record.start.uploadID)
        }
        await resumeStore.remove(fileURL: fileURL, projectID: projectID, size: size)
    }

    private func createStart(
        fileURL: URL,
        projectID: String,
        session: CutSellSession,
        contentType: String,
        size: Int64
    ) async throws -> MultipartStartResponse {
        struct StartBody: Encodable {
            let project_id: String
            let user_id: String
            let original_name: String
            let content_type: String
            let size_bytes: Int64
            /// Build 21: 8 MiB parts (all equal except the last, as R2 requires). The current server
            /// ignores it and answers 16 MiB; the app always uses the part size the server answers.
            let part_size: Int
        }
        return try await api.request(
            "/v1/uploads/multipart/start",
            method: "POST",
            body: StartBody(
                project_id: projectID,
                user_id: session.userID,
                original_name: fileURL.lastPathComponent,
                content_type: contentType,
                size_bytes: size,
                part_size: Self.requestedPartSize
            ),
            timeout: 20
        )
    }

    /// Set after iOS's background upload service fails a part with NSURLErrorUnknown (-1; the
    /// Simulator always does): the rest of the parts go through the normal session.
    private static let backgroundBroken = Flag()

    static func uploadPart(
        file: URL,
        bytes: Int64,
        partNumber: Int,
        signedURL: URL,
        uploadID: String? = nil,
        foreground: Bool,
        onBytes: (@Sendable (Int64) -> Void)?
    ) async throws -> HTTPURLResponse {
        if !foreground && !backgroundBroken.value {
            do {
                let response = try await BackgroundPartUploader.shared.upload(fileURL: file, to: signedURL, uploadID: uploadID,
                                                                             part: partNumber, onBytes: onBytes)
                onBytes?(bytes)
                return response
            } catch let error as NSError where error.domain == NSURLErrorDomain && error.code == NSURLErrorUnknown {
                backgroundBroken.set()
                CutSellDiagnostics.log("upload_background_unavailable", ["part": String(partNumber)])
            }
        }
        var request = URLRequest(url: signedURL)
        request.httpMethod = "PUT"
        // Same empty Content-Type the server signs (see BackgroundPartUploader).
        request.setValue(BackgroundPartUploader.partContentType, forHTTPHeaderField: "Content-Type")
        let delegate = PartProgressDelegate(onBytes: onBytes)
        let (_, response) = try await partSession.upload(for: request, fromFile: file, delegate: delegate)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw BackgroundUploadError.http((response as? HTTPURLResponse)?.statusCode ?? -1)
        }
        onBytes?(bytes)
        return http
    }

    /// The normal session (short B-roll uploads, and the fallback). A request fails only after
    /// 90 s with no byte moving; three parts at a time.
    static let partSession: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 90
        configuration.timeoutIntervalForResource = 60 * 60
        configuration.waitsForConnectivity = false
        configuration.httpMaximumConnectionsPerHost = 3
        return URLSession(configuration: configuration)
    }()

    /// "Try again" after 90 s with no byte sent: restarts the parts that are sending (each is
    /// retried on its own; finished parts stay finished).
    static func restartStuckParts() {
        BackgroundPartUploader.shared.restartStuckParts()
        partSession.getAllTasks { tasks in
            for task in tasks where task.state == .running || task.state == .suspended { task.cancel() }
        }
        CutSellDiagnostics.log("upload_restart_stuck_parts", [:])
    }

    /// A short, stable folder name for an upload id of any length (SHA-256, first 32 hex digits).
    static func folderName(for uploadID: String) -> String {
        SHA256.hash(data: Data(uploadID.utf8)).prefix(16).map { String(format: "%02x", $0) }.joined()
    }

    private func partsFolder(uploadID: String) throws -> URL {
        let root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell", isDirectory: true)
            .appendingPathComponent("UploadParts", isDirectory: true)
            // Build 19: a short folder name made from the upload id. Cloudflare R2's upload ids are
            // ~350 characters; used as a folder name they broke iOS's 255-character limit
            // (NSCocoaErrorDomain 514) and no part ever left the iPhone after the move to R2.
            .appendingPathComponent(Self.folderName(for: uploadID), isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}

final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    var value: Int { lock.lock(); defer { lock.unlock() }; return count }
    func add() { lock.lock(); count += 1; lock.unlock() }
}

final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var on = false
    var value: Bool { lock.lock(); defer { lock.unlock() }; return on }
    func set() { lock.lock(); on = true; lock.unlock() }
}

enum UploadError: LocalizedError {
    case emptyFile
    case missingPart(Int)
    case missingETag(Int)
    case resumeMetadataMismatch
    case incompleteUpload

    var errorDescription: String? {
        switch self {
        case .emptyFile: return "The selected video is empty."
        case .missingPart(let part): return "Couldn’t read upload part \(part)."
        case .missingETag(let part): return "The storage did not confirm upload part \(part)."
        case .resumeMetadataMismatch: return "The saved upload does not match the server session."
        case .incompleteUpload: return "The upload is missing one or more parts."
        }
    }
}


/// Bytes sent by one part (the upload screen's real %).
final class PartProgressDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private let onBytes: (@Sendable (Int64) -> Void)?
    init(onBytes: (@Sendable (Int64) -> Void)?) { self.onBytes = onBytes }

    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64,
                    totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        onBytes?(totalBytesSent)
    }
}
