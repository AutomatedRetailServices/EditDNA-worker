import Foundation
import UIKit

/// Build 19: a new cut must never sit at "Uploading 0 %" forever (Swanny's iPhone, builds 16-18).
/// - `Retry`: every request and every part is tried 3 times before it fails.
/// - `ByteTally`: the real % of a multipart upload, byte by byte, across parts in flight.
/// - `StallWatch`: when nothing moves for 30 s the work stops with the real error and the screen
///   offers "Try again".

enum Retry {
    /// Waits between attempts (seconds).
    static let waits: [Double] = [2, 5]

    /// `body` up to `attempts` times. A 4xx answer (except 408 / 429) is not retried: it would only
    /// fail again. `onRetry` hears about each retry (the screen shows it, Diagnostics logs it).
    static func run<T>(_ what: String, attempts: Int = 3, onRetry: ((Int, Error) -> Void)? = nil,
                       _ body: () async throws -> T) async throws -> T {
        var attempt = 1
        while true {
            do {
                return try await body()
            } catch {
                if error is CancellationError || Task.isCancelled || !retriable(error) || attempt >= attempts { throw error }
                CutSellDiagnostics.log("upload_retry", ["what": what, "attempt": String(attempt), "error": short(error)])
                onRetry?(attempt, error)
                try await Task.sleep(nanoseconds: UInt64(waits[min(attempt - 1, waits.count - 1)] * 1_000_000_000))
                attempt += 1
            }
        }
    }

    static func retriable(_ error: Error) -> Bool {
        if case APIError.http(let code, _) = error { return code >= 500 || code == 408 || code == 429 }
        if case BackgroundUploadError.http(let code) = error { return code >= 500 || code == 408 || code == 429 || code == 403 }
        return true
    }

    /// A short, plain description of an error (no links, no ids).
    static func short(_ error: Error) -> String {
        if case APIError.http(let code, _) = error { return "HTTP \(code)" }
        if case BackgroundUploadError.http(let code) = error { return "HTTP \(code)" }
        let ns = error as NSError
        if ns.domain == NSURLErrorDomain {
            switch ns.code {
            case NSURLErrorTimedOut: return "timed out"
            case NSURLErrorNotConnectedToInternet: return "no internet connection"
            case NSURLErrorNetworkConnectionLost: return "connection lost"
            case NSURLErrorCannotFindHost, NSURLErrorCannotConnectToHost: return "server not reachable"
            default: return "network error \(ns.code)"
            }
        }
        return "\(ns.domain) \(ns.code)"
    }
}

/// The real % of one multipart upload: bytes of finished parts + bytes sent by parts in flight.
final class ByteTally: @unchecked Sendable {
    private let lock = NSLock()
    private let total: Int64
    private var finished: Int64
    private var inFlight: [Int: Int64] = [:]
    private let report: @Sendable (Double) -> Void

    init(total: Int64, alreadyUploaded: Int64, report: @escaping @Sendable (Double) -> Void) {
        self.total = max(1, total)
        finished = alreadyUploaded
        self.report = report
    }

    /// Reports only when a part has sent MORE bytes (the 2 s poll repeats the same count; a
    /// repeat is not activity for the 90 s watch).
    func sent(part: Int, bytes: Int64) {
        lock.lock()
        guard bytes > (inFlight[part] ?? 0) else { lock.unlock(); return }
        inFlight[part] = bytes
        let value = Double(finished + inFlight.values.reduce(0, +)) / Double(total)
        lock.unlock()
        report(min(0.999, value))
    }

    /// A part failed and will be sent again: its bytes no longer count.
    func reset(part: Int) {
        lock.lock(); inFlight[part] = nil; lock.unlock()
    }

    func done(part: Int, size: Int64) {
        lock.lock()
        inFlight[part] = nil
        finished += size
        let value = Double(finished) / Double(total)
        lock.unlock()
        report(min(1, value))
    }
}

/// Build 21: watches `work` and NEVER cancels it. When no byte (and no other step) moves for
/// `limit` seconds (90), the screen says the upload is paused and offers "Try again", which restarts
/// only the parts in flight; as soon as a byte moves again the notice goes away by itself.
/// (Build 19 cancelled the whole upload after 30 s: on Swanny's iPhone that killed parts that were
/// still sending -- NSURLError -999.) The work itself ends only when it finishes or a part has
/// failed 3 times.
enum StallWatch {
    static let limit: TimeInterval = 90

    struct Stalled: LocalizedError {
        let step: String
        let lastError: String?
        var errorDescription: String? {
            let base = "No data sent for 90 seconds while \(step.isEmpty ? "uploading" : step.lowercased())."
            return lastError.map { "\(base) Last error: \($0)." } ?? base
        }
    }

    @MainActor
    static func run<T>(_ submitter: CutSubmitter, limit: TimeInterval = StallWatch.limit,
                       _ work: @escaping @MainActor () async throws -> T) async throws -> T {
        submitter.touch()
        let watcher = Task { @MainActor in
            var lastTick = Date()
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                if Task.isCancelled { return }
                // Back from the background (or the screen was locked): the parts kept going in iOS's
                // upload service; the 2 s poll reports them now. Not a stall.
                if Date().timeIntervalSince(lastTick) > 10 { submitter.touch() }
                lastTick = Date()
                if AIConsentCenter.shared.asking { submitter.touch(); continue }
                if Date().timeIntervalSince(submitter.lastActivity) > limit, submitter.stalled == nil {
                    let stalled = Stalled(step: submitter.stageTitle, lastError: submitter.lastError)
                    CutSellDiagnostics.log("upload_stalled", ["reason": "no bytes for 90 s", "step": submitter.stageTitle,
                                                              "last_error": submitter.lastError ?? "none"])
                    submitter.markStalled(stalled.localizedDescription)
                }
            }
        }
        defer { watcher.cancel(); submitter.clearStalled() }
        return try await work()
    }

    /// Resumes a continuation exactly once.
    final class Once<T>: @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: CheckedContinuation<T, Error>?
        init(_ continuation: CheckedContinuation<T, Error>) { self.continuation = continuation }
        var isDone: Bool { lock.lock(); defer { lock.unlock() }; return continuation == nil }
        func finish(_ result: Result<T, Error>) {
            lock.lock()
            let current = continuation
            continuation = nil
            lock.unlock()
            current?.resume(with: result)
        }
    }
}

/// Asks iOS for time to finish handing the parts to the background upload service when the
/// creator leaves CutSell (or locks the screen) mid-upload.
@MainActor
final class BackgroundTime {
    private var id = UIBackgroundTaskIdentifier.invalid

    func begin() {
        guard id == .invalid else { return }
        id = UIApplication.shared.beginBackgroundTask(withName: "cutsell-upload") { [weak self] in
            CutSellDiagnostics.log("upload_app_suspended", [:])
            self?.end()
        }
    }

    func end() {
        guard id != .invalid else { return }
        UIApplication.shared.endBackgroundTask(id)
        id = .invalid
    }
}
