import CryptoKit
import Foundation

/// What the editor keeps on this iPhone so it opens and plays without waiting for the server:
/// - the cut's source video (the file uploaded from this phone, or downloaded once in the background),
/// - the last draft of each cut (shown at once, then refreshed),
/// - the Edits list (shown at once on launch, then refreshed).
/// Everything lives in Caches (iOS may clear it when space is low; it is then fetched again).
enum LocalCaches {
    static var root: URL {
        let url = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func folder(_ name: String) -> URL {
        let url = root.appendingPathComponent(name, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func key(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined().prefix(32).description
    }
}

/// Source videos by their storage URI (`s3://…`, the same value the draft's `sources[].uri` has).
actor SourceVideoCache {
    static let shared = SourceVideoCache()

    /// Total size kept; the oldest files go first.
    static let maxBytes: Int64 = 3 * 1024 * 1024 * 1024

    private var downloading: Set<String> = []

    private static var folder: URL { LocalCaches.folder("Sources") }

    private static func file(forURI uri: String) -> URL {
        folder.appendingPathComponent(LocalCaches.key(uri) + ".mp4")
    }

    /// The local copy, if this iPhone has it (also marks it as recently used).
    nonisolated static func localURL(forURI uri: String) -> URL? {
        let url = file(forURI: uri)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: url.path)
        return url
    }

    /// Keeps the file this iPhone just uploaded (no download later). A hard link when possible.
    func store(fileURL: URL, uri: String) {
        let target = Self.file(forURI: uri)
        guard !FileManager.default.fileExists(atPath: target.path) else { return }
        if (try? FileManager.default.linkItem(at: fileURL, to: target)) == nil {
            try? FileManager.default.copyItem(at: fileURL, to: target)
        }
        trim()
    }

    private static let wifiOnly: URLSession = {
        let config = URLSessionConfiguration.default
        config.allowsExpensiveNetworkAccess = false
        config.allowsConstrainedNetworkAccess = false
        return URLSession(configuration: config)
    }()

    /// Downloads a cut's source once, in the background, for the next time it plays.
    /// `onlyOnWiFi` for cuts that are merely listed (never spend cellular data on those).
    func prefetch(uri: String, remote: URL, onlyOnWiFi: Bool = false) async {
        guard Self.localURL(forURI: uri) == nil, !downloading.contains(uri) else { return }
        downloading.insert(uri)
        defer { downloading.remove(uri) }
        let session = onlyOnWiFi ? Self.wifiOnly : URLSession.shared
        guard let (temp, response) = try? await session.download(from: remote),
              let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else { return }
        let target = Self.file(forURI: uri)
        try? FileManager.default.removeItem(at: target)
        try? FileManager.default.moveItem(at: temp, to: target)
        trim()
    }

    private func trim() {
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey]
        guard let files = try? FileManager.default.contentsOfDirectory(at: Self.folder, includingPropertiesForKeys: keys) else { return }
        var entries = files.compactMap { url -> (URL, Int64, Date)? in
            guard let values = try? url.resourceValues(forKeys: Set(keys)) else { return nil }
            return (url, Int64(values.fileSize ?? 0), values.contentModificationDate ?? .distantPast)
        }
        var total = entries.reduce(Int64(0)) { $0 + $1.1 }
        entries.sort { $0.2 < $1.2 }
        for (url, size, _) in entries where total > Self.maxBytes {
            try? FileManager.default.removeItem(at: url)
            total -= size
        }
    }
}

/// The last draft of each cut, so the editor opens at once (the server copy replaces it right after).
enum DraftSnapshotCache {
    private static func file(_ projectID: String) -> URL {
        LocalCaches.folder("Drafts").appendingPathComponent(LocalCaches.key(projectID) + ".json")
    }

    static func load(projectID: String) -> DraftSnapshot? {
        guard let data = try? Data(contentsOf: file(projectID)) else { return nil }
        return try? JSONDecoder().decode(DraftSnapshot.self, from: data)
    }

    static func save(_ snapshot: DraftSnapshot) {
        guard let data = try? JSONEncoder().encode(snapshot) else { return }
        try? data.write(to: file(snapshot.projectID), options: .atomic)
    }
}

/// The Edits list, so the app shows it at once on launch.
enum ProjectListCache {
    private static func file(_ userID: String) -> URL {
        LocalCaches.folder("Projects").appendingPathComponent(LocalCaches.key(userID) + ".json")
    }

    static func load(userID: String) -> [Project]? {
        guard let data = try? Data(contentsOf: file(userID)) else { return nil }
        return try? JSONDecoder().decode([Project].self, from: data)
    }

    static func save(_ projects: [Project], userID: String) {
        guard let data = try? JSONEncoder().encode(projects) else { return }
        try? data.write(to: file(userID), options: .atomic)
    }
}

/// Gets a ready cut ready to open at once: its draft kept on the phone, and its video downloaded
/// (on Wi-Fi) for the preview. Called for cuts shown in Edits and right after a cut finishes.
enum EditorPrewarm {
    @MainActor private static var done: Set<String> = []

    @MainActor
    static func prewarm(projectID: String, session: CutSellSession?, snapshot known: DraftSnapshot? = nil) async {
        guard !done.contains(projectID), let session else { return }
        done.insert(projectID)
        var snapshot = known
        if let known { DraftSnapshotCache.save(known) }
        if snapshot == nil, DraftSnapshotCache.load(projectID: projectID) == nil {
            snapshot = try? await APIClient.shared.request(
                "/v1/projects/\(projectID)/draft",
                query: [URLQueryItem(name: "user_id", value: session.userID)]
            )
            if let snapshot { DraftSnapshotCache.save(snapshot) }
        }
        for source in snapshot?.sources ?? [] {
            guard let object = source.objectValue, let uri = object["uri"]?.stringValue,
                  let raw = object["playback_url"]?.stringValue, let url = URL(string: raw) else { continue }
            Task.detached(priority: .background) {
                await SourceVideoCache.shared.prefetch(uri: uri, remote: url, onlyOnWiFi: true)
            }
        }
    }
}
