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
        if let existingUploadID {
            let status: MultipartStatusResponse = try await api.request(
                "/v1/uploads/multipart/\(existingUploadID)",
                query: [
                    URLQueryItem(name: "project_id", value: projectID),
                    URLQueryItem(name: "user_id", value: session.userID)
                ]
            )
            for part in status.uploadedParts { completedParts[part.partNumber] = part.etag }
            // Resume metadata does not return part size, so restart safely if local metadata was lost.
            // The app persists upload IDs only alongside MultipartStartResponse in its project state.
            throw UploadError.resumeRequiresMetadata
        } else {
            struct StartBody: Encodable {
                let project_id: String
                let user_id: String
                let original_name: String
                let content_type: String
                let size_bytes: Int64
            }
            start = try await api.request(
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
            let response = try await api.upload(data, to: signed.uploadURL)
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
        let result: MultipartCompleteResponse = try await api.request(
            "/v1/uploads/multipart/\(start.uploadID)/complete",
            method: "POST",
            body: CompleteBody(project_id: projectID, user_id: session.userID, parts: parts)
        )
        progress(1)
        return result
    }
}

enum UploadError: LocalizedError {
    case emptyFile
    case missingPart(Int)
    case missingETag(Int)
    case resumeRequiresMetadata

    var errorDescription: String? {
        switch self {
        case .emptyFile: "The selected video is empty."
        case .missingPart(let part): "Couldn’t read upload part \(part)."
        case .missingETag(let part): "S3 did not confirm upload part \(part)."
        case .resumeRequiresMetadata: "Upload metadata needs to be restored before resuming."
        }
    }
}
