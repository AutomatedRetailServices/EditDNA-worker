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

    enum CodingKeys: String, CodingKey {
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
    private let backgroundUploader = BackgroundPartUploader.shared

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
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> MultipartCompleteResponse {
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
                guard status.partCount == saved.start.partCount else {
                    throw UploadError.resumeMetadataMismatch
                }
                for part in status.uploadedParts { completedParts[part.partNumber] = part.etag }
                start = saved.start
                // D-133: bounded, diagnostics-only resume-detection log --
                // no change to the existing reconciliation logic above.
                CutSellDiagnostics.log("upload_resumed", [
                    "resumed_part_count": String(completedParts.count),
                    "total_part_count": String(start.partCount),
                ])
            } catch APIError.http(let code, _) where code == 404 {
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
        CutSellDiagnostics.log("upload_parts", ["parts": String(start.partCount), "part_mb": String(start.partSize / 1_048_576),
                                                "mb": String(size / 1_048_576), "done": String(completedParts.count)])

        await resumeStore.save(start: start, fileURL: fileURL, projectID: projectID, size: size)
        // The real %: bytes of finished parts + bytes the parts in flight have sent.
        let already = completedParts.keys.reduce(Int64(0)) { sum, number in
            sum + Int64(min(start.partSize, max(0, Int(size) - (number - 1) * start.partSize)))
        }
        let tally = ByteTally(total: size, alreadyUploaded: already, report: progress)
        progress(Double(already) / Double(max(1, size)))

        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }

        // Up to three parts at a time (they used to go one after another, each waiting for its own
        // link first), so the phone's connection stays busy instead of idle between parts.
        let missing = (1...start.partCount).filter { completedParts[$0] == nil }
        let maxInFlight = 3
        try await withThrowingTaskGroup(of: (Int, String).self) { group in
            var next = 0
            func launch(_ partNumber: Int) throws {
                let offset = UInt64((partNumber - 1) * start.partSize)
                try handle.seek(toOffset: offset)
                let remaining = max(0, Int(size) - Int(offset))
                let readCount = min(start.partSize, remaining)
                let data = try handle.read(upToCount: readCount) ?? Data()
                guard !data.isEmpty else { throw UploadError.missingPart(partNumber) }
                let uploadID = start.uploadID
                let userID = session.userID
                group.addTask {
                    struct OwnerBody: Encodable { let project_id: String; let user_id: String }
                    // A fresh link and the PUT, tried 3 times (a link can expire while a part waits).
                    let etagValue: String = try await Retry.run("part \(partNumber)", onRetry: { attempt, error in
                        tally.reset(part: partNumber)
                        retried(attempt, error)
                    }) {
                        let signed: MultipartPartSignResponse = try await self.api.request(
                            "/v1/uploads/multipart/\(uploadID)/parts/\(partNumber)/presign",
                            method: "POST",
                            body: OwnerBody(project_id: projectID, user_id: userID),
                            timeout: 20
                        )
                        let response = try await self.uploadPart(
                            data, uploadID: uploadID, partNumber: partNumber, signedURL: signed.uploadURL,
                            foreground: foreground, onBytes: { tally.sent(part: partNumber, bytes: $0) }
                        )
                        guard let etag = response.value(forHTTPHeaderField: "ETag"), !etag.isEmpty else {
                            throw UploadError.missingETag(partNumber)
                        }
                        return etag
                    }
                    tally.done(part: partNumber, size: Int64(data.count))
                    return (partNumber, etagValue)
                }
            }
            while next < missing.count && next < maxInFlight { try launch(missing[next]); next += 1 }
            while let (partNumber, etag) = try await group.next() {
                completedParts[partNumber] = etag
                if next < missing.count { try launch(missing[next]); next += 1 }
            }
        }

        struct CompleteBody: Encodable {
            let project_id: String
            let user_id: String
            let parts: [MultipartUploadedPart]
        }
        let parts = completedParts
            .map { MultipartUploadedPart(partNumber: $0.key, etag: $0.value) }
            .sorted { $0.partNumber < $1.partNumber }
        guard parts.count == start.partCount else { throw UploadError.incompleteUpload }

        let result: MultipartCompleteResponse = try await Retry.run("complete", onRetry: retried) {
            try await api.request(
                "/v1/uploads/multipart/\(start.uploadID)/complete",
                method: "POST",
                body: CompleteBody(project_id: projectID, user_id: session.userID, parts: parts),
                timeout: 60
            )
        }
        await resumeStore.remove(fileURL: fileURL, projectID: projectID, size: size)
        progress(1)
        return result
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
        }
        return try await api.request(
            "/v1/uploads/multipart/start",
            method: "POST",
            body: StartBody(
                project_id: projectID,
                user_id: session.userID,
                original_name: fileURL.lastPathComponent,
                content_type: contentType,
                size_bytes: size
            ),
            timeout: 20
        )
    }

    private func uploadPart(
        _ data: Data,
        uploadID: String,
        partNumber: Int,
        signedURL: URL,
        foreground: Bool = false,
        onBytes: (@Sendable (Int64) -> Void)? = nil
    ) async throws -> HTTPURLResponse {
        // Build 19: every part goes through a normal (foreground) URLSession, like overlays and
        // voice-overs. Through iOS's background upload service every part to Cloudflare R2 ended in
        // NSURLErrorUnknown (-1) and the upload never moved past 0 %. CutSell stays open while a
        // video uploads (the screen says so); a stopped upload resumes from the parts already sent.
        _ = foreground
        var request = URLRequest(url: signedURL)
        request.httpMethod = "PUT"
        // Same empty Content-Type the server signs (see BackgroundPartUploader).
        request.setValue(BackgroundPartUploader.partContentType, forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30
        let delegate = PartProgressDelegate(onBytes: onBytes)
        let (_, response) = try await Self.partSession.upload(for: request, from: data, delegate: delegate)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw BackgroundUploadError.http((response as? HTTPURLResponse)?.statusCode ?? -1)
        }
        onBytes?(Int64(data.count))
        return http
    }

    /// One session for the parts (a request without progress for 30 s is retried).
    private static let partSession: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 30 * 60
        configuration.waitsForConnectivity = false
        configuration.httpMaximumConnectionsPerHost = 4
        return URLSession(configuration: configuration)
    }()

    /// A short, stable folder name for an upload id of any length (SHA-256, first 32 hex digits).
    static func folderName(for uploadID: String) -> String {
        SHA256.hash(data: Data(uploadID.utf8)).prefix(16).map { String(format: "%02x", $0) }.joined()
    }

    private func writePartFile(_ data: Data, uploadID: String, partNumber: Int) throws -> URL {
        let root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell", isDirectory: true)
            .appendingPathComponent("UploadParts", isDirectory: true)
            // Build 19: a short folder name made from the upload id. Cloudflare R2's upload ids are
            // ~350 characters; used as a folder name they broke iOS's 255-character limit
            // (NSCocoaErrorDomain 514) and no part ever left the iPhone after the move to R2.
            .appendingPathComponent(Self.folderName(for: uploadID), isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let destination = root.appendingPathComponent(String(format: "part-%05d.bin", partNumber))
        try data.write(to: destination, options: .atomic)
        return destination
    }
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
