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
        /// Build 19: bytes sent so far (the upload screen shows the real %, not 0 % until a whole part ends).
        var onBytes: (@Sendable (Int64) -> Void)?
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
        // Build 21: every part is queued at once (the upload goes on with CutSell in the background
        // or the screen locked); iOS sends three at a time.
        configuration.httpMaximumConnectionsPerHost = 3
        return URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }()

    /// Build 22: `uploadID` + `part` travel with the task (taskDescription), so the part's ETag is saved
    /// to disk (UploadLedger) even when it finishes with CutSell suspended or relaunched by iOS.
    func upload(fileURL: URL, to signedURL: URL, headers: [String: String] = [:],
                uploadID: String? = nil, part: Int? = nil,
                onBytes: (@Sendable (Int64) -> Void)? = nil) async throws -> HTTPURLResponse {
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
            if let uploadID, let part { task.taskDescription = Self.tag(uploadID: uploadID, part: part) }
            lock.lock()
            pending[task.taskIdentifier] = Pending(continuation: continuation, response: nil, onBytes: onBytes)
            lock.unlock()
            task.resume()
            startPolling()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64,
                    totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        lock.lock()
        let callback = pending[task.taskIdentifier]?.onBytes
        lock.unlock()
        callback?(totalBytesSent)
    }

    // MARK: Build 21: bytes polled every 2 s (iOS delivers didSendBodyData from its background
    // upload service in bursts: on Swanny's iPhone nothing came for 90 s while parts were sending).

    private var poller: DispatchSourceTimer?

    private func startPolling() {
        lock.lock(); defer { lock.unlock() }
        guard poller == nil else { return }
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: .now() + 2, repeating: 2)
        timer.setEventHandler { [weak self] in self?.poll() }
        poller = timer
        timer.resume()
    }

    private func poll() {
        session.getAllTasks { [weak self] tasks in
            guard let self else { return }
            self.lock.lock()
            let callbacks = tasks.compactMap { task -> (@Sendable (Int64) -> Void, Int64)? in
                guard let callback = self.pending[task.taskIdentifier]?.onBytes, task.countOfBytesSent > 0 else { return nil }
                return (callback, task.countOfBytesSent)
            }
            let idle = self.pending.isEmpty
            if idle { self.poller?.cancel(); self.poller = nil }
            self.lock.unlock()
            for (callback, bytes) in callbacks { callback(bytes) }
        }
    }

    /// "Try again" after 90 s without a byte: the parts in flight are restarted (each part is
    /// retried on its own; the parts already uploaded stay uploaded).
    func restartStuckParts() {
        session.getAllTasks { tasks in
            for task in tasks where task.state == .running || task.state == .suspended { task.cancel() }
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

    static func tag(uploadID: String, part: Int) -> String { "cutsell-part|\(part)|\(uploadID)" }

    static func untag(_ description: String?) -> (uploadID: String, part: Int)? {
        let pieces = (description ?? "").split(separator: "|", maxSplits: 2).map(String.init)
        guard pieces.count == 3, pieces[0] == "cutsell-part", let part = Int(pieces[1]), !pieces[2].isEmpty else { return nil }
        return (pieces[2], part)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock()
        let item = pending.removeValue(forKey: task.taskIdentifier)
        lock.unlock()
        // Build 22: the ETag goes to disk first, whoever is (or is not) waiting for this part.
        let http = item?.response ?? task.response as? HTTPURLResponse
        if error == nil, let http, (200..<300).contains(http.statusCode), let tag = Self.untag(task.taskDescription),
           let etag = http.value(forHTTPHeaderField: "ETag"), !etag.isEmpty {
            UploadLedger.shared.recordPart(uploadID: tag.uploadID, part: tag.part, etag: etag)
            if item == nil {
                CutSellDiagnostics.log("upload_part_done_background", ["part": String(tag.part), "etag_saved": "yes"])
            }
        } else if item == nil, let tag = Self.untag(task.taskDescription) {
            CutSellDiagnostics.log("upload_part_failed_background", ["part": String(tag.part),
                                                                     "status": String(http?.statusCode ?? -1),
                                                                     "error": error.map { Retry.short($0) } ?? "none"])
        }
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
        // Build 22: finish ("complete" + start the cut) what these parts were for, before iOS suspends.
        Task { @MainActor in PendingUploadFinisher.uploadEventsDelivered() }
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
