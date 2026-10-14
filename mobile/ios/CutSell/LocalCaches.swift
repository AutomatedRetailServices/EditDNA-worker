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

    /// Downloads a cut's source once, in the background, for the next time it plays.
    func prefetch(uri: String, remote: URL) async {
        guard Self.localURL(forURI: uri) == nil, !downloading.contains(uri) else { return }
        downloading.insert(uri)
        defer { downloading.remove(uri) }
        guard let (temp, response) = try? await URLSession.shared.download(from: remote),
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
