import AVFoundation
import Foundation

/// A voice-over the creator saved that is not on the server yet. It shows on the mic row at once
/// (and plays in the preview from the file kept on this iPhone) while it uploads in the background.
/// If the upload fails, the recording stays on the phone and the upload is tried again (a few times
/// now, then every time the editor opens, or when the creator taps the block).
struct PendingVoiceOver: Codable, Identifiable, Equatable {
    let id: String
    let projectID: String
    let start: Double
    let duration: Double
    let fileName: String
    var attempts: Int = 0
    var failed: Bool = false
    let createdAt: Date

    var end: Double { start + duration }
}

/// Recordings waiting to upload, kept in Application Support (not a temporary folder) so they
/// survive the app closing, with one small list per project.
enum PendingVoiceOverStore {
    static var folder: URL {
        let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell/PendingVoiceOvers", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    static func fileURL(_ item: PendingVoiceOver) -> URL { folder.appendingPathComponent(item.fileName) }

    /// Copies the take into the pending folder; `nil` if the phone can't keep it.
    static func keep(_ file: URL, id: String) -> String? {
        let ext = file.pathExtension.isEmpty ? "m4a" : file.pathExtension
        let name = "\(id).\(ext)"
        let destination = folder.appendingPathComponent(name)
        try? FileManager.default.removeItem(at: destination)
        do { try FileManager.default.copyItem(at: file, to: destination) } catch { return nil }
        return name
    }

    private static func listURL(projectID: String) -> URL {
        let safe = String(projectID.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) || $0 == "_" || $0 == "-" })
        return folder.appendingPathComponent("list_\(safe).json")
    }

    static func load(projectID: String) -> [PendingVoiceOver] {
        guard let data = try? Data(contentsOf: listURL(projectID: projectID)),
              let items = try? JSONDecoder().decode([PendingVoiceOver].self, from: data) else { return [] }
        // A list entry whose recording is gone can never upload.
        return items.filter { FileManager.default.fileExists(atPath: fileURL($0).path) }
    }

    static func save(_ items: [PendingVoiceOver], projectID: String) {
        let url = listURL(projectID: projectID)
        if items.isEmpty { try? FileManager.default.removeItem(at: url); return }
        if let data = try? JSONEncoder().encode(items) { try? data.write(to: url, options: .atomic) }
    }

    static func discardFile(_ item: PendingVoiceOver) {
        try? FileManager.default.removeItem(at: fileURL(item))
    }

    /// Length of a recording in seconds (0 when it can't be read).
    static func duration(of url: URL) async -> Double {
        guard let time = try? await AVURLAsset(url: url).load(.duration) else { return 0 }
        let seconds = CMTimeGetSeconds(time)
        return seconds.isFinite ? max(0, seconds) : 0
    }

    /// Waits before each retry: 2 s, 6 s, then 15 s.
    static func retryDelay(afterAttempt attempt: Int) -> Double {
        switch attempt {
        case ..<1: return 0
        case 1: return 2
        case 2: return 6
        default: return 15
        }
    }

    /// First start at or after `start` where a voice of `duration` overlaps none of `taken`
    /// (timeline seconds). Voice-overs never overlap on the mic row (the server refuses it).
    static func freeStart(_ start: Double, duration: Double, taken: [(start: Double, end: Double)]) -> Double {
        let ranges = taken.sorted { $0.start < $1.start }
        var at = max(0, start)
        var moved = true
        while moved {
            moved = false
            for range in ranges where at < range.end - 0.001 && at + duration > range.start + 0.001 {
                at = ((range.end + 0.01) * 1000).rounded() / 1000
                moved = true
            }
        }
        return at
    }

    /// Tries in a row before the block is marked "tap to retry" (it is tried again on the next open).
    static let attemptsPerRound = 4
}

/// One voice the preview plays: a placed voice-over (recording kept on this iPhone) or one still uploading.
struct PreviewVoice: Equatable {
    let key: String
    let url: URL
    let start: Double
    let sourceIn: Double
    let sourceOut: Double
}
