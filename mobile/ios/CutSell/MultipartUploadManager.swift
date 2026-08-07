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
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> MultipartCompleteResponse {
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
                let status: MultipartStatusResponse = try await api.request(
                    "/v1/uploads/multipart/\(saved.start.uploadID)",
                    query: [
                        URLQueryItem(name: "project_id", value: projectID),
                        URLQueryItem(name: "user_id", value: session.userID)
                    ]
                )
                guard status.partCount == saved.start.partCount else {
                    throw UploadError.resumeMetadataMismatch
                }
                for part in status.uploadedParts { completedParts[part.partNumber] = part.etag }
                start = saved.start
            } catch APIError.http(let code, _) where code == 404 {
                await resumeStore.remove(uploadID: saved.start.uploadID)
                start = try await createStart(
                    fileURL: fileURL,
                    projectID: projectID,
                    session: session,
                    contentType: contentType,
                    size: size
                )
            }
        } else {
            start = try await createStart(
                fileURL: fileURL,
                projectID: projectID,
                session: session,
                contentType: contentType,
                size: size
            )
        }

        await resumeStore.save(start: start, fileURL: fileURL, projectID: projectID, size: size)
        progress(Double(completedParts.count) / Double(max(1, start.partCount)))

        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }

        for partNumber in 1...start.partCount {
            if completedParts[partNumber] != nil { continue }
            let offset = UInt64((partNumber - 1) * start.partSize)
            try handle.seek(toOffset: offset)
            let remaining = max(0, Int(size) - Int(offset))
            let readCount = min(start.partSize, remaining)
            let data = try handle.read(upToCount: readCount) ?? Data()
            guard !data.isEmpty else { throw UploadError.missingPart(partNumber) }

            struct OwnerBody: Encodable { let project_id: String; let user_id: String }
            let signed: MultipartPartSignResponse = try await api.request(
                "/v1/uploads/multipart/\(start.uploadID)/parts/\(partNumber)/presign",
                method: "POST",
                body: OwnerBody(project_id: projectID, user_id: session.userID)
            )

            let partFile = try writePartFile(
                data,
                uploadID: start.uploadID,
                partNumber: partNumber
            )
            defer { try? FileManager.default.removeItem(at: partFile) }
            let response = try await backgroundUploader.upload(fileURL: partFile, to: signed.uploadURL)
            guard let etagValue = response.value(forHTTPHeaderField: "ETag"), !etagValue.isEmpty else {
                throw UploadError.missingETag(partNumber)
            }
            completedParts[partNumber] = etagValue
            progress(Double(completedParts.count) / Double(start.partCount))
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

        let result: MultipartCompleteResponse = try await api.request(
            "/v1/uploads/multipart/\(start.uploadID)/complete",
            method: "POST",
            body: CompleteBody(project_id: projectID, user_id: session.userID, parts: parts)
        )
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
            )
        )
    }

    private func writePartFile(_ data: Data, uploadID: String, partNumber: Int) throws -> URL {
        let root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell", isDirectory: true)
            .appendingPathComponent("UploadParts", isDirectory: true)
            .appendingPathComponent(uploadID, isDirectory: true)
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
        case .missingETag(let part): return "S3 did not confirm upload part \(part)."
        case .resumeMetadataMismatch: return "The saved upload does not match the server session."
        case .incompleteUpload: return "The upload is missing one or more parts."
        }
    }
}
