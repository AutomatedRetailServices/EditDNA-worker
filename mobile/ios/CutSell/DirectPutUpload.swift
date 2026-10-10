import Foundation

/// Signed single-PUT direct upload, used when the server's presign answer says
/// `method: "PUT"` (Cloudflare R2, which does not implement S3 POST form uploads).
/// On Amazon S3 the server keeps answering `method: "POST"` and each manager keeps
/// its existing multipart/form-data upload, so the app works with both storages.
enum DirectPutUpload {
    static func isPut(_ method: String) -> Bool {
        method.uppercased() == "PUT"
    }

    /// Streams the file as the request body. The server signs the exact
    /// Content-Type and Content-Length; `headers` (from the server) win over the
    /// locally guessed content type. Returns the HTTP status code (-1 if none).
    static func send(fileURL: URL, to url: URL, contentType: String, headers: [String: String]) async throws -> Int {
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        for (key, value) in headers {
            request.setValue(value, forHTTPHeaderField: key)
        }
        let (_, response) = try await URLSession.shared.upload(for: request, fromFile: fileURL)
        return (response as? HTTPURLResponse)?.statusCode ?? -1
    }
}
