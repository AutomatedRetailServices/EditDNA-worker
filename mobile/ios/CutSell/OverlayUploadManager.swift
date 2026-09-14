import Foundation
import UniformTypeIdentifiers

struct OverlayPresignResponse: Decodable {
    let method: String
    let uploadURL: String
    let fields: [String: String]
    let uri: String
    let kind: String
    let contentType: String

    enum CodingKeys: String, CodingKey {
        case method
        case uploadURL = "upload_url"
        case fields, uri, kind
        case contentType = "content_type"
    }
}

enum OverlayUploadError: LocalizedError {
    case invalidFile
    case invalidUploadURL
    case uploadFailed(Int)

    var errorDescription: String? {
        switch self {
        case .invalidFile: return "The selected overlay file is unavailable."
        case .invalidUploadURL: return "CutSell returned an invalid overlay upload URL."
        case .uploadFailed(let code): return "Overlay upload failed (\(code))."
        }
    }
}

actor OverlayUploadManager {
    static let shared = OverlayUploadManager()

    func upload(fileURL: URL, projectID: String, session: CutSellSession) async throws -> OverlayPresignResponse {
        let values = try fileURL.resourceValues(forKeys: [.fileSizeKey])
        guard let size = values.fileSize, size > 0 else { throw OverlayUploadError.invalidFile }
        let contentType = Self.contentType(for: fileURL)

        struct Body: Encodable {
            let project_id: String
            let user_id: String
            let original_name: String
            let content_type: String
            let size_bytes: Int
        }

        let presign: OverlayPresignResponse = try await APIClient.shared.request(
            "/v1/overlays/uploads/presign",
            method: "POST",
            body: Body(
                project_id: projectID,
                user_id: session.userID,
                original_name: fileURL.lastPathComponent,
                content_type: contentType,
                size_bytes: size
            )
        )

        guard let uploadURL = URL(string: presign.uploadURL) else {
            throw OverlayUploadError.invalidUploadURL
        }

        let boundary = "CutSellOverlay-\(UUID().uuidString)"
        var request = URLRequest(url: uploadURL)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        for key in presign.fields.keys.sorted() {
            guard let value = presign.fields[key] else { continue }
            body.appendFormField(name: key, value: value, boundary: boundary)
        }
        let fileData = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(fileURL.lastPathComponent)\"\r\n")
        body.append("Content-Type: \(presign.contentType)\r\n\r\n")
        body.append(fileData)
        body.append("\r\n--\(boundary)--\r\n")

        let (_, response) = try await URLSession.shared.upload(for: request, from: body)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw OverlayUploadError.uploadFailed((response as? HTTPURLResponse)?.statusCode ?? -1)
        }
        return presign
    }

    private static func contentType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "jpg", "jpeg": return "image/jpeg"
        case "png": return "image/png"
        case "mp4": return "video/mp4"
        case "mov": return "video/quicktime"
        case "m4v": return "video/x-m4v"
        case "webm": return "video/webm"
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
