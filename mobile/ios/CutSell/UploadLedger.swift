import Foundation

/// Build 22: what the iPhone knows about each multipart upload, on disk (Application Support), so it
/// survives CutSell being suspended or closed by iOS:
/// - the ETag of every part as soon as R2 answers it (also when the part finished in iOS's background
///   upload service with CutSell suspended or relaunched);
/// - that "complete" was sent, and that it succeeded (with the file's source URI).
///
/// Why: on Swanny's iPhone (build 21, 18 Oct) R2 accepted all three uploads, but the answer to
/// "complete" never reached the app (iOS had suspended it). The app sent "complete" again, the server
/// had already closed the session (404), and the app started the whole upload again from 0.
/// With the ledger, a 404 after "complete" was sent means the upload is already done.
struct UploadLedgerEntry: Codable, Equatable {
    var uploadID: String
    var projectID: String
    var sourceURI: String
    var partCount: Int
    var partSize: Int
    var etags: [Int: String] = [:]
    var completeSentAt: Date?
    var completedAt: Date?
    var updatedAt: Date = Date()
}

final class UploadLedger: @unchecked Sendable {
    static let shared = UploadLedger()

    private let lock = NSLock()
    private var cache: [String: UploadLedgerEntry]?
    private let fileURL: URL

    init(fileURL: URL? = nil) {
        if let fileURL {
            self.fileURL = fileURL
        } else {
            let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("CutSell", isDirectory: true)
            try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            self.fileURL = root.appendingPathComponent("upload-ledger.json")
        }
    }

    func entry(_ uploadID: String) -> UploadLedgerEntry? {
        lock.lock(); defer { lock.unlock() }
        return load()[uploadID]
    }

    /// Called when an upload starts (or resumes): keeps what the ledger already knows.
    func begin(_ start: MultipartStartResponse) {
        update(start.uploadID) { entry in
            entry = entry ?? UploadLedgerEntry(uploadID: start.uploadID, projectID: start.projectID,
                                               sourceURI: start.sourceURI, partCount: start.partCount,
                                               partSize: start.partSize)
        }
    }

    /// A part R2 accepted. Safe to call from any thread (the background session's delegate).
    func recordPart(uploadID: String, part: Int, etag: String) {
        guard !etag.isEmpty else { return }
        update(uploadID) { entry in entry?.etags[part] = etag }
    }

    func markCompleteSent(_ uploadID: String) {
        update(uploadID) { entry in if entry?.completeSentAt == nil { entry?.completeSentAt = Date() } }
    }

    func markCompleted(_ uploadID: String, sourceURI: String) {
        update(uploadID) { entry in
            entry?.completedAt = Date()
            if !sourceURI.isEmpty { entry?.sourceURI = sourceURI }
        }
    }

    func remove(_ uploadID: String) {
        update(uploadID) { entry in entry = nil }
    }

    /// Entries older than 3 days are dropped (the server keeps an upload session for 24 h).
    func prune(now: Date = Date()) {
        lock.lock(); defer { lock.unlock() }
        var all = load()
        all = all.filter { now.timeIntervalSince($0.value.updatedAt) < 3 * 24 * 3600 }
        save(all)
    }

    private func update(_ uploadID: String, _ change: (inout UploadLedgerEntry?) -> Void) {
        lock.lock(); defer { lock.unlock() }
        var all = load()
        var entry = all[uploadID]
        change(&entry)
        entry?.updatedAt = Date()
        all[uploadID] = entry
        save(all)
    }

    private func load() -> [String: UploadLedgerEntry] {
        if let cache { return cache }
        let value = (try? Data(contentsOf: fileURL)).flatMap { try? JSONDecoder().decode([String: UploadLedgerEntry].self, from: $0) } ?? [:]
        cache = value
        return value
    }

    private func save(_ value: [String: UploadLedgerEntry]) {
        cache = value
        guard let data = try? JSONEncoder().encode(value) else { return }
        // Readable while the iPhone is locked (a part can finish then), never synced to iCloud.
        try? data.write(to: fileURL, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        var url = fileURL
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? url.setResourceValues(values)
    }
}

/// Build 22: files saved with an absolute path (pending cuts, resumable uploads) are found again after
/// iOS moves the app's folder (an app update changes the container path).
enum ContainerPath {
    static func resolve(_ path: String) -> URL {
        let url = URL(fileURLWithPath: path)
        if FileManager.default.fileExists(atPath: path) { return url }
        let home = NSHomeDirectory()
        for marker in ["/Library/", "/Documents/", "/tmp/"] {
            if let range = path.range(of: marker) {
                let candidate = home + path[range.lowerBound...]
                if FileManager.default.fileExists(atPath: candidate) { return URL(fileURLWithPath: candidate) }
            }
        }
        return url
    }
}
