import Foundation
import UniformTypeIdentifiers

/// Real, already-registered D-282A routes for importing a NEW voice-over
/// asset from an existing audio file -- `POST /timeline-uploads` (real S3
/// presigned-POST via `cutsell_worker/uploads.py`'s
/// `create_presigned_voice_over_upload`, a dedicated audio-scoped upload
/// target, never the video-only `create_presigned_upload` path) then
/// `POST /timeline-assets` (`role: VOICE_OVER`, `media_kind: AUDIO`), which
/// synchronously probes real audio presence/duration and returns the asset
/// already `READY` or `FAILED` -- no polling/qualifying step exists for
/// voice-over ingest.
///
/// Microphone RECORDING is explicitly NOT implemented anywhere in this
/// codebase -- `cutsell_worker/timeline_asset_registry_store.py`'s own
/// `create_voice_over_asset` docstring states this directly ("NO microphone
/// recording happens here or anywhere in this module... the future mobile
/// client records audio and uploads the completed file"). This manager only
/// ever imports an ALREADY-EXISTING audio file the user picks; it never
/// captures live audio.
private struct TimelineUploadAuthorization: Decodable {
    let uploadID: String
    let method: String
    let uploadURL: String
    let fields: [String: String]
    let contentType: String?
    let maxBytes: Int?
    let expiresIn: Int

    enum CodingKeys: String, CodingKey {
        case uploadID = "upload_id"
        case method
        case uploadURL = "upload_url"
        case fields
        case contentType = "content_type"
        case maxBytes = "max_bytes"
        case expiresIn = "expires_in"
    }
}

enum VoiceOverImportError: LocalizedError {
    case invalidFile
    case invalidUploadURL
    case uploadFailed(Int)

    var errorDescription: String? {
        switch self {
        case .invalidFile: return "The selected audio file is unavailable."
        case .invalidUploadURL: return "CutSell returned an invalid voice-over upload URL."
        case .uploadFailed(let code): return "Voice-over upload failed (\(code))."
        }
    }
}

actor VoiceOverImportManager {
    static let shared = VoiceOverImportManager()

    /// Imports an already-recorded/selected local audio file as a new
    /// VOICE_OVER `TimelineMediaAsset`: real presign -> real S3 upload ->
    /// real asset registration, in that order. Returns the asset the
    /// backend actually created (READY or FAILED per its own real
    /// technical-media probe) -- never assumed ready client-side.
    func importAudio(
        fileURL: URL,
        projectID: String,
        session: CutSellSession,
        api: APIClient = .shared
    ) async throws -> TimelineMediaAsset {
        let values = try fileURL.resourceValues(forKeys: [.fileSizeKey])
        guard let size = values.fileSize, size > 0 else { throw VoiceOverImportError.invalidFile }
        let contentType = Self.contentType(for: fileURL)

        struct UploadAuthorizationBody: Encodable {
            let user_id: String
            let media_class: String
            let original_name: String
            let content_type: String
            let size_bytes: Int
        }
        let authorization: TimelineUploadAuthorization = try await api.request(
            "/v1/projects/\(projectID)/timeline-uploads",
            method: "POST",
            body: UploadAuthorizationBody(
                user_id: session.userID,
                media_class: "voice_over",
                original_name: fileURL.lastPathComponent,
                content_type: contentType,
                size_bytes: size
            )
        )

        try await Self.uploadBytes(fileURL: fileURL, contentType: contentType, authorization: authorization)

        struct AssetCreateBody: Encodable {
            let user_id: String
            let role: String
            let media_kind: String
            let upload_id: String
        }
        let asset: TimelineMediaAsset = try await api.request(
            "/v1/projects/\(projectID)/timeline-assets",
            method: "POST",
            body: AssetCreateBody(
                user_id: session.userID, role: "VOICE_OVER", media_kind: "AUDIO",
                upload_id: authorization.uploadID
            )
        )
        return asset
    }

    /// The real S3 presigned-POST upload step -- same multipart/form-data
    /// mechanics `OverlayUploadManager` already uses successfully for its
    /// own (different) presign route, applied here to the voice-over one.
    private static func uploadBytes(
        fileURL: URL, contentType: String, authorization: TimelineUploadAuthorization
    ) async throws {
        guard let uploadURL = URL(string: authorization.uploadURL) else {
            throw VoiceOverImportError.invalidUploadURL
        }
        let boundary = "CutSellVoiceOver-\(UUID().uuidString)"
        var request = URLRequest(url: uploadURL)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        for key in authorization.fields.keys.sorted() {
            guard let value = authorization.fields[key] else { continue }
            body.appendFormField(name: key, value: value, boundary: boundary)
        }
        let fileData = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(fileURL.lastPathComponent)\"\r\n")
        body.append("Content-Type: \(contentType)\r\n\r\n")
        body.append(fileData)
        body.append("\r\n--\(boundary)--\r\n")

        let (_, response) = try await URLSession.shared.upload(for: request, from: body)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw VoiceOverImportError.uploadFailed((response as? HTTPURLResponse)?.statusCode ?? -1)
        }
    }

    private static func contentType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "m4a": return "audio/mp4"
        case "mp3": return "audio/mpeg"
        case "wav": return "audio/wav"
        case "aac": return "audio/aac"
        case "caf": return "audio/x-caf"
        default:
            if let type = UTType(filenameExtension: url.pathExtension),
               let mime = type.preferredMIMEType {
                return mime
            }
            return "application/octet-stream"
        }
    }
}

private extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) { append(data) }
    }

    mutating func appendFormField(name: String, value: String, boundary: String) {
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
        append("\(value)\r\n")
    }
}
