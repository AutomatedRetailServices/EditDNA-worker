import AVFoundation
import SwiftUI
import UIKit

// MARK: - Trim by the edges (Editor v4)
//
// A clip is a window on its original video: `source_asset_id` + `start` / `end` (seconds of the
// original; the names never change). Selecting a clip shows white handles on its edges: dragging
// a handle outward brings back material the cut had hidden, inward shortens the clip. The clips
// after it follow (ripple, no gaps). One save when the finger lifts (the editor's one save queue,
// which also heals a 409), so Undo / Redo work as for any other edit.

enum TrimEdge: String { case start, end }

enum TrimMath {
    /// Shortest clip a trim can leave.
    static let minLength = 0.1
    /// How far past the neighbour clip's material the finger must push before the edge goes on.
    static let passThrough = 0.6

    struct Limits: Equatable {
        /// The original's own start / end (and the minimum length on the other side).
        var lower: Double
        var upper: Double
        /// Where the material of the clip next to it (same original) begins / ends, if any.
        var neighbor: Double?
    }

    struct Resolved: Equatable {
        var value: Double
        /// Held at the neighbour clip's material (vibrates once when it gets there).
        var atNeighbor: Bool
        /// Pushed 0.6 s further: the edge went past the neighbour's material.
        var passed: Bool
        /// Stopped at the start or the end of the original.
        var atSourceEdge: Bool
    }

    /// Limits for dragging `edge` of `clips[index]`. `sourceDuration` nil = unknown (no upper end).
    static func limits(clips: [[String: JSONValue]], index: Int, edge: TrimEdge, sourceDuration: Double?) -> Limits {
        let clip = clips[index]
        let source = clip["source_asset_id"]?.stringValue
        let start = clip["start"]?.doubleValue ?? 0
        let end = clip["end"]?.doubleValue ?? start
        switch edge {
        case .end:
            var neighbor: Double?
            if index + 1 < clips.count, clips[index + 1]["source_asset_id"]?.stringValue == source,
               let next = clips[index + 1]["start"]?.doubleValue, next >= end - 0.001 {
                neighbor = next
            }
            return Limits(lower: start + minLength, upper: sourceDuration ?? .greatestFiniteMagnitude, neighbor: neighbor)
        case .start:
            var neighbor: Double?
            if index > 0, clips[index - 1]["source_asset_id"]?.stringValue == source,
               let previous = clips[index - 1]["end"]?.doubleValue, previous <= start + 0.001 {
                neighbor = previous
            }
            return Limits(lower: 0, upper: end - minLength, neighbor: neighbor)
        }
    }

    /// Where the edge goes for a finger at `proposed` (seconds of the original).
    static func resolve(proposed: Double, edge: TrimEdge, limits: Limits) -> Resolved {
        var value = proposed
        var atNeighbor = false
        var passed = false
        if let neighbor = limits.neighbor {
            switch edge {
            case .end where proposed > neighbor:
                if proposed - neighbor < passThrough { value = neighbor; atNeighbor = true } else { passed = true }
            case .start where proposed < neighbor:
                if neighbor - proposed < passThrough { value = neighbor; atNeighbor = true } else { passed = true }
            default: break
            }
        }
        let clamped = min(max(value, limits.lower), limits.upper)
        let atSourceEdge = clamped != value && ((edge == .end && value > limits.upper) || (edge == .start && value < limits.lower))
        return Resolved(value: (clamped * 1000).rounded() / 1000, atNeighbor: atNeighbor && clamped == value,
                        passed: passed, atSourceEdge: atSourceEdge)
    }

    // MARK: Captions follow the window

    /// Every word of the draft (shown, hidden or discarded clips) heard in `sourceID`, once each.
    static func wordPool(_ draft: [String: JSONValue], sourceID: String?) -> [[String: JSONValue]] {
        var seen = Set<String>()
        var pool: [[String: JSONValue]] = []
        for key in ["selected", "alternates", "discarded"] {
            for clip in draft[key]?.arrayValue?.compactMap(\.objectValue) ?? [] {
                guard clip["source_asset_id"]?.stringValue == sourceID else { continue }
                for word in clip["words"]?.arrayValue?.compactMap(\.objectValue) ?? [] {
                    guard let start = word["start"]?.doubleValue, let end = word["end"]?.doubleValue, end >= start else { continue }
                    let id = "\(Int((start * 1000).rounded()))|\(Int((end * 1000).rounded()))|\(word["text"]?.stringValue ?? word["word"]?.stringValue ?? "")"
                    if seen.insert(id).inserted { pool.append(word) }
                }
            }
        }
        return pool.sorted { ($0["start"]?.doubleValue ?? 0) < ($1["start"]?.doubleValue ?? 0) }
    }

    /// The words heard inside [start, end] (a word belongs where its middle is, like the engine).
    static func words(in pool: [[String: JSONValue]], start: Double, end: Double) -> [[String: JSONValue]] {
        pool.filter { word in
            let middle = ((word["start"]?.doubleValue ?? 0) + (word["end"]?.doubleValue ?? 0)) / 2
            return middle >= start && middle <= end
        }
    }

    static func text(of words: [[String: JSONValue]]) -> String {
        words.compactMap { ($0["text"]?.stringValue ?? $0["word"]?.stringValue)?.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }.joined(separator: " ")
    }

    private static func normalized(_ text: String) -> String {
        text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    /// The creator rewrote this clip's caption by hand (it no longer matches what was said).
    static func captionEditedByHand(_ clip: [String: JSONValue]) -> Bool {
        guard let caption = clip["caption_text"]?.stringValue else { return false }
        return normalized(caption) != normalized(clip["text"]?.stringValue ?? "")
    }

    /// The clip with its new window: words that come in appear in its caption, words that go out
    /// are hidden; a caption written by hand stays as written. Mirror, volume, zooms and every
    /// other field are kept. An end dragged by hand is marked so the export keeps it.
    static func trimmed(_ clip: [String: JSONValue], start: Double, end: Double, edge: TrimEdge,
                        pool: [[String: JSONValue]]) -> [String: JSONValue] {
        var out = clip
        out["start"] = .number((start * 1000).rounded() / 1000)
        out["end"] = .number((end * 1000).rounded() / 1000)
        let handWritten = captionEditedByHand(clip)
        let inside = words(in: pool, start: start, end: end)
        out["words"] = .array(inside.map(JSONValue.object))
        let spoken = text(of: inside)
        out["text"] = .string(spoken)
        if !handWritten { out["caption_text"] = .string(spoken) }
        if edge == .end { out["manual_end"] = .bool(true) }
        return out
    }

    /// "+0,4 s" / "−0,2 s": how much material the drag adds (positive) or removes, in the phone's language.
    static func deltaLabel(_ seconds: Double, locale: Locale = .current) -> String {
        let formatter = NumberFormatter()
        formatter.locale = locale
        formatter.minimumFractionDigits = 1
        formatter.maximumFractionDigits = 1
        formatter.minimumIntegerDigits = 1
        let rounded = (seconds * 10).rounded() / 10
        let sign = rounded > 0 ? "+" : (rounded < 0 ? "−" : "")
        return sign + (formatter.string(from: NSNumber(value: abs(rounded))) ?? "\(abs(rounded))") + " s"
    }
}

// MARK: - The exact frame under the handle, shown on the video while dragging

/// What the video shows while a handle is dragged: the original's exact frame at the handle and
/// how much material the drag adds.
struct TrimScrub: Equatable {
    let sourceID: String?
    let sourceTime: Double
    let delta: Double
    let mirrored: Bool
}

actor TrimFrameGenerator {
    static let shared = TrimFrameGenerator()
    private var generators: [String: AVAssetImageGenerator] = [:]

    /// The frame at exactly `seconds` (no tolerance: the frame the cut will start or end on).
    func frame(url: URL, at seconds: Double) async -> UIImage? {
        let generator: AVAssetImageGenerator
        if let existing = generators[url.path] {
            generator = existing
        } else {
            generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: 720, height: 1280)
            generator.requestedTimeToleranceBefore = .zero
            generator.requestedTimeToleranceAfter = .zero
            generators[url.path] = generator
        }
        guard let cg = try? await generator.image(at: CMTime(seconds: max(0, seconds), preferredTimescale: 600)).image else {
            generators[url.path] = nil
            return nil
        }
        return UIImage(cgImage: cg)
    }
}

struct TrimFrameOverlay: View {
    let scrub: TrimScrub
    let url: URL?
    @State private var image: UIImage?

    var body: some View {
        ZStack(alignment: .top) {
            Color.black
            if let image {
                Image(uiImage: image).resizable().scaledToFit()
                    .scaleEffect(x: scrub.mirrored ? -1 : 1, y: 1)
            }
            Text(TrimMath.deltaLabel(scrub.delta))
                .font(CS.inter(15, .bold)).monospacedDigit()
                .foregroundStyle(.white)
                .padding(.horizontal, 12).padding(.vertical, 6)
                .background(Color.black.opacity(0.6), in: Capsule())
                .overlay(Capsule().stroke(Color.white.opacity(0.8), lineWidth: 1))
                .padding(.top, 12)
                .accessibilityIdentifier("trimDelta")
        }
        .allowsHitTesting(false)
        // One frame per 1/30 s of the original: a new request only when the frame really changes.
        .task(id: "\(url?.path ?? "")|\(Int((scrub.sourceTime * 30).rounded()))") {
            guard let url else { return }
            if let frame = await TrimFrameGenerator.shared.frame(url: url, at: scrub.sourceTime) { image = frame }
        }
        .accessibilityIdentifier("trimFrame")
    }
}

// MARK: - Haptics

enum TrimHaptics {
    /// Reached the neighbour clip's material: a firm knock.
    @MainActor static func neighbor() { UIImpactFeedbackGenerator(style: .rigid).impactOccurred() }
    /// Went past it (pushed 0.6 s more).
    @MainActor static func passed() { UIImpactFeedbackGenerator(style: .medium).impactOccurred() }
    /// Start or end of the original.
    @MainActor static func sourceEdge() { UIImpactFeedbackGenerator(style: .light).impactOccurred() }
}

// MARK: - 720p copy of the whole original, for the preview

/// A 720p copy of each original kept on this iPhone (Caches), made once in the background from the
/// local original. The preview, the clip frames and the trim frames use it: lighter to decode, and
/// it has the WHOLE original, so a clip can be stretched over material the cut had hidden.
/// The export always uses the high-quality original on the server.
actor PreviewProxyCache {
    static let shared = PreviewProxyCache()
    private var building: Set<String> = []

    private static func file(forURI uri: String) -> URL {
        LocalCaches.folder("Preview720").appendingPathComponent(LocalCaches.key(uri) + ".mp4")
    }

    nonisolated static func localURL(forURI uri: String) -> URL? {
        let url = file(forURI: uri)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Makes the 720p copy from the local original if it isn't there yet.
    func ensure(uri: String, original: URL) async {
        guard Self.localURL(forURI: uri) == nil, !building.contains(uri) else { return }
        building.insert(uri)
        defer { building.remove(uri) }
        let asset = AVURLAsset(url: original)
        guard let session = AVAssetExportSession(asset: asset, presetName: AVAssetExportPreset1280x720) else { return }
        let target = Self.file(forURI: uri)
        let temp = target.deletingLastPathComponent().appendingPathComponent(UUID().uuidString + ".mp4")
        session.outputURL = temp
        session.outputFileType = .mp4
        session.shouldOptimizeForNetworkUse = true
        await session.export()
        guard session.status == .completed else { try? FileManager.default.removeItem(at: temp); return }
        // The copy must cover the whole original, or a stretched clip would play past its end.
        let full = CMTimeGetSeconds((try? await asset.load(.duration)) ?? .zero)
        let copy = CMTimeGetSeconds((try? await AVURLAsset(url: temp).load(.duration)) ?? .zero)
        guard full > 0, copy >= full - 0.1 else { try? FileManager.default.removeItem(at: temp); return }
        try? FileManager.default.removeItem(at: target)
        try? FileManager.default.moveItem(at: temp, to: target)
    }
}

extension DraftEditorViewModel {
    /// The video used on the phone for one source: the 720p copy, else the original kept on this
    /// iPhone (and the 720p copy is made from it), else the signed link.
    func previewVideoURL(forSource sourceID: String?) -> URL? {
        let sources = allSources
        let source = sources.first { $0["source_asset_id"]?.stringValue == sourceID }
            ?? (sourceID == nil ? sources.first : nil)
        if let uri = source?["uri"]?.stringValue {
            if let proxy = PreviewProxyCache.localURL(forURI: uri) { return proxy }
            if let local = SourceVideoCache.localURL(forURI: uri) {
                Task.detached(priority: .utility) { await PreviewProxyCache.shared.ensure(uri: uri, original: local) }
                return local
            }
        }
        return playbackURL(forSource: sourceID)
    }

    /// Length of a source's original (seconds), when the server or the phone knows it.
    func sourceDuration(forSource sourceID: String?) -> Double? {
        let source = allSources.first { $0["source_asset_id"]?.stringValue == sourceID }
        if let seconds = source?["duration_sec"]?.doubleValue, seconds > 0 { return seconds }
        return knownSourceDurations[sourceID ?? ""]
    }
}
