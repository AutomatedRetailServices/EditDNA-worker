import Foundation

/// Uploads one already-presigned S3 multipart part using an iOS background URLSession.
///
/// The durable source of truth remains S3's multipart upload. If iOS suspends or
/// relaunches the app after a background PUT finishes, MultipartUploadManager asks
/// the backend which part numbers already exist and continues from there.
final class BackgroundPartUploader: NSObject, URLSessionTaskDelegate, URLSessionDataDelegate, @unchecked Sendable {
    static let shared = BackgroundPartUploader()
    static let sessionIdentifier = "ai.cutsell.background.multipart"
    /// Must stay empty: it is part of the part link's signature (see `upload`).
    static let partContentType = ""

    private struct Pending {
        let continuation: CheckedContinuation<HTTPURLResponse, Error>
        var response: HTTPURLResponse?
    }

    private let lock = NSLock()
    private var pending: [Int: Pending] = [:]
    private var backgroundEventsCompletion: (() -> Void)?

    private lazy var session: URLSession = {
        let configuration = URLSessionConfiguration.background(withIdentifier: Self.sessionIdentifier)
        configuration.sessionSendsLaunchEvents = true
        configuration.isDiscretionary = false
        configuration.allowsCellularAccess = true
        configuration.waitsForConnectivity = true
        configuration.timeoutIntervalForRequest = 15 * 60
        configuration.timeoutIntervalForResource = 60 * 60
        return URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }()

    func upload(fileURL: URL, to signedURL: URL, headers: [String: String] = [:]) async throws -> HTTPURLResponse {
        var request = URLRequest(url: signedURL)
        request.httpMethod = "PUT"
        // The server signs each part link with an EMPTY Content-Type. Left alone, URLSession adds
        // "Content-Type: application/octet-stream" to a file upload, the signature no longer
        // matches and Amazon S3 answers 403 SignatureDoesNotMatch -- no part ever uploaded from an
        // iPhone (TestFlight builds 3-4). An empty value makes URLSession send no Content-Type.
        request.setValue(Self.partContentType, forHTTPHeaderField: "Content-Type")
        for (key, value) in headers { request.setValue(value, forHTTPHeaderField: key) }

        return try await withCheckedThrowingContinuation { continuation in
            let task = session.uploadTask(with: request, fromFile: fileURL)
            lock.lock()
            pending[task.taskIdentifier] = Pending(continuation: continuation, response: nil)
            lock.unlock()
            task.resume()
        }
    }

    func acceptBackgroundEventsCompletion(_ completion: @escaping () -> Void) {
        lock.lock()
        backgroundEventsCompletion = completion
        lock.unlock()
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        if let http = response as? HTTPURLResponse {
            lock.lock()
            if var item = pending[dataTask.taskIdentifier] {
                item.response = http
                pending[dataTask.taskIdentifier] = item
            }
            lock.unlock()
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock()
        let item = pending.removeValue(forKey: task.taskIdentifier)
        lock.unlock()
        guard let item else { return }

        if let error {
            item.continuation.resume(throwing: error)
            return
        }
        guard let response = item.response ?? task.response as? HTTPURLResponse,
              (200..<300).contains(response.statusCode) else {
            let code = (item.response ?? task.response as? HTTPURLResponse)?.statusCode ?? -1
            item.continuation.resume(throwing: BackgroundUploadError.http(code))
            return
        }
        item.continuation.resume(returning: response)
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        lock.lock()
        let completion = backgroundEventsCompletion
        backgroundEventsCompletion = nil
        lock.unlock()
        DispatchQueue.main.async { completion?() }
    }
}

enum BackgroundUploadError: LocalizedError {
    case http(Int)

    var errorDescription: String? {
        switch self {
        case .http(let code): return "Background upload failed (HTTP \(code))."
        }
    }
}
