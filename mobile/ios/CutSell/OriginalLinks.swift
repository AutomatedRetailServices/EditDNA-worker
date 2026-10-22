import Foundation
import Photos

/// 22 Oct (export from the original): which Photos video each uploaded source came from.
/// The server only has the light 1080p copy; the export on this iPhone reads the ORIGINAL from Photos.
/// - a file made from a Photos video is noted (`note(file:photosID:)`), and so is its lighter copy (`carry`);
/// - when a file is uploaded (`SourceVideoCache.store / replace`), its storage URI gets the same Photos id.
/// Kept in UserDefaults (ids only, never the media).
enum OriginalLinks {
    private static let filesKey = "cutsell.originals.files.v1"
    private static let urisKey = "cutsell.originals.uris.v1"
    private static let lock = NSLock()

    private static func load(_ key: String) -> [String: String] {
        (UserDefaults.standard.dictionary(forKey: key) as? [String: String]) ?? [:]
    }

    private static func save(_ map: [String: String], _ key: String, keep: Int = 2000) {
        var map = map
        if map.count > keep { for k in map.keys.prefix(map.count - keep) { map[k] = nil } }
        UserDefaults.standard.set(map, forKey: key)
    }

    /// The file at `file` is (a copy of) the Photos video `photosID`.
    static func note(file: URL, photosID: String?) {
        guard let photosID, !photosID.isEmpty else { return }
        lock.lock(); defer { lock.unlock() }
        var map = load(filesKey)
        map[file.standardizedFileURL.path] = photosID
        save(map, filesKey)
    }

    /// `to` was made from `from` (a lighter copy): same original.
    static func carry(from: URL, to: URL) {
        note(file: to, photosID: photosID(forFile: from))
    }

    static func photosID(forFile file: URL) -> String? {
        lock.lock(); defer { lock.unlock() }
        return load(filesKey)[file.standardizedFileURL.path]
    }

    /// Called when `file` was uploaded as `uri`.
    static func link(uri: String, file: URL) {
        guard let id = photosID(forFile: file) else { return }
        lock.lock(); defer { lock.unlock() }
        var map = load(urisKey)
        map[uri] = id
        save(map, urisKey)
    }

    static func photosID(forURI uri: String) -> String? {
        lock.lock(); defer { lock.unlock() }
        return load(urisKey)[uri]
    }

    /// The Photos video still in the library (nil when it was deleted or never came from Photos).
    static func asset(forURI uri: String) -> PHAsset? {
        guard let id = photosID(forURI: uri) else { return nil }
        return PHAsset.fetchAssets(withLocalIdentifiers: [id], options: nil).firstObject
    }
}
