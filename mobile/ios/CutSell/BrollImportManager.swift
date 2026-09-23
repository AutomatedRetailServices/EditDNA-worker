import Foundation
import UniformTypeIdentifiers

/// Mobile V1 B-roll UI gate -- real Import authority for B-roll (D-279/
/// D-282A `TimelineComposition` `SUPPLEMENTAL_BROLL`) video assets, via the
/// SAME D-282A upload+registration pipeline `VoiceOverImportManager`
/// already proved for voice-over audio: `POST /{project_id}/timeline-
/// uploads` (media_class="video", the same class PRIMARY_SOURCE uses --
/// `cutsell_app/timeline_routes.py::_media_class_for` picks "video" for
/// any non-AUDIO/non-VOICE_OVER role) -> real S3 multipart-form-data
/// upload -> `POST /{project_id}/timeline-assets` (role="SUPPLEMENTAL_
/// BROLL", media_kind="VIDEO") -> `ingest_broll_asset_from_upload` ->
/// `create_video_timeline_asset` (`cutsell_worker/timeline_asset_registry_
/// store.py`), which is fully synchronous: probes real source-media
/// profile, runs the SAME format policy/normalization decision PRIMARY_
/// SOURCE ingest uses, and returns READY/REJECTED/FAILED directly -- never
/// a fabricated QUALIFYING/polling step.
///
/// Deliberately distinct from `OverlayUploadManager.swift`/`OverlayView`
/// (`POST /v1/overlays/uploads/presign`, `cutsell_app/overlay_routes.py`'s
/// `add_media_overlay`/`update_media_overlay`, with real x/y/width
/// position fields operating on the `draft` dict directly): that is a
/// genuinely different, real, ACTIVE feature (confirmed wired through
/// `export_job.py` into the live export renderer's `overlay=x=...:y=...`
/// ffmpeg filter) -- a positioned/scaled Overlay layer, never to be
/// confused with this file's B-roll (a full-frame visual REPLACEMENT clip,
/// no position/scale/rotation/opacity field anywhere in its real
/// contract). This file was previously misnamed `OverlayImportManager`;
/// renamed to match what it actually imports. Only VIDEO media is
/// supported here -- `TimelineMediaKind` has no IMAGE case, and
/// `create_video_timeline_asset` requires `media_kind: VIDEO` -- a still-
/// image B-roll clip has no real backend authority in this system and is
/// never offered.
enum BrollImportError: LocalizedError {
    case invalidFile
    case invalidUploadURL
    case uploadFailed(Int)

    var errorDescription: String? {
        switch self {
        case .invalidFile: return "The selected B-roll video is unavailable."
        case .invalidUploadURL: return "CutSell returned an invalid B-roll upload URL."
        case .uploadFailed(let code): return "B-roll upload failed (\(code))."
        }
    }
}

private struct TimelineUploadAuthorization: Decodable {
    let uploadID: String
    let method: String
    let uploadURL: String
    let fields: [String: String]
    let contentType: String?
    let maxBytes: Int?
    let expiresIn: Int

    enum CodingKeys: String, CodingKey {
        case uploadID = "upload_id"; case method; case uploadURL = "upload_url"
        case fields; case contentType = "content_type"; case maxBytes = "max_bytes"
        case expiresIn = "expires_in"
    }
}

actor BrollImportManager {
    static let shared = BrollImportManager()

    func importVideo(fileURL: URL, projectID: String, session: CutSellSession, api: APIClient = .shared) async throws -> TimelineMediaAsset {
        let values = try fileURL.resourceValues(forKeys: [.fileSizeKey])
        guard let size = values.fileSize, size > 0 else { throw BrollImportError.invalidFile }
        let contentType = Self.contentType(for: fileURL)

        struct UploadAuthorizationBody: Encodable {
            let user_id: String; let media_class: String; let original_name: String
            let content_type: String; let size_bytes: Int
        }
        let authorization: TimelineUploadAuthorization = try await api.request(
            "/v1/projects/\(projectID)/timeline-uploads", method: "POST",
            body: UploadAuthorizationBody(
                user_id: session.userID, media_class: "video",
                original_name: fileURL.lastPathComponent, content_type: contentType, size_bytes: size
            )
        )
        try await Self.uploadBytes(fileURL: fileURL, contentType: contentType, authorization: authorization)

        struct AssetCreateBody: Encodable {
            let user_id: String; let role: String; let media_kind: String; let upload_id: String
        }
        let asset: TimelineMediaAsset = try await api.request(
            "/v1/projects/\(projectID)/timeline-assets", method: "POST",
            body: AssetCreateBody(
                user_id: session.userID, role: "SUPPLEMENTAL_BROLL", media_kind: "VIDEO",
                upload_id: authorization.uploadID
            )
        )
        return asset
    }

    private static func uploadBytes(fileURL: URL, contentType: String, authorization: TimelineUploadAuthorization) async throws {
        guard let uploadURL = URL(string: authorization.uploadURL) else { throw BrollImportError.invalidUploadURL }
        let boundary = "CutSellBroll-\(UUID().uuidString)"
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
            throw BrollImportError.uploadFailed((response as? HTTPURLResponse)?.statusCode ?? -1)
        }
    }

    private static func contentType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "mp4": return "video/mp4"
        case "mov": return "video/quicktime"
        case "m4v": return "video/x-m4v"
        case "webm": return "video/webm"
        default:
            if let type = UTType(filenameExtension: url.pathExtension), let mime = type.preferredMIMEType {
                return mime
            }
            return "application/octet-stream"
        }
    }
}

private extension Data {
    mutating func append(_ string: String) { if let data = string.data(using: .utf8) { append(data) } }
    mutating func appendFormField(name: String, value: String, boundary: String) {
        append("--\(boundary)\r\n"); append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n"); append("\(value)\r\n")
    }
}
