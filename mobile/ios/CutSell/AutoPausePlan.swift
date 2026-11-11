import AVFoundation
import Foundation

/// The two Record modes (PO 9 Nov 2026): the user says what she is recording, the app never guesses (no face or
/// person detection anywhere).
/// - Talk (default): auto-pause by voice; silences longer than 0.7 s are not saved.
/// - B-roll: everything is recorded, nothing is cut, no 1-minute auto-stop; the clip is marked B-roll.
enum CaptureMode: String, CaseIterable, Identifiable {
    case talk, broll
    var id: String { rawValue }
    var title: String { self == .talk ? "Talk" : "B-roll" }
}

/// Auto-pause by voice (Talk mode), the rules only (unit-tested with synthetic timelines).
///
/// What is kept: every moment with voice, plus 0.3 s before it (the first word is never cut) and 0.3 s after it,
/// and pauses of 0.7 s or less, whole (natural breathing between phrases). Everything when no voice was heard.
/// What is dropped: silence longer than 0.7 s (minus the 0.3 s margins).
enum AutoPausePlan {
    static let silence: Double = 0.7
    static let margin: Double = 0.3
    static let autoStopAfter: Double = 60

    struct Range: Equatable {
        var start: Double
        var end: Double
        var length: Double { max(0, end - start) }
    }

    /// Whether this recording uses auto-pause: Talk mode with the Auto-pause switch on. Never in B-roll.
    static func applies(mode: CaptureMode, autoPauseOn: Bool) -> Bool { mode == .talk && autoPauseOn }

    /// Merges sorted / unsorted ranges that touch or overlap.
    static func merged(_ ranges: [Range]) -> [Range] {
        let sorted = ranges.filter { $0.end > $0.start }.sorted { $0.start < $1.start }
        var out: [Range] = []
        for r in sorted {
            if let last = out.last, r.start <= last.end + 1e-6 { out[out.count - 1].end = max(last.end, r.end) }
            else { out.append(r) }
        }
        return out
    }

    /// The parts of the recording to keep. B-roll: all of it.
    static func keep(mode: CaptureMode = .talk, voice: [Range], duration: Double) -> [Range] {
        guard duration > 0 else { return [] }
        let whole = [Range(start: 0, end: duration)]
        let voiced = merged(voice).map { Range(start: max(0, $0.start), end: min(duration, $0.end)) }
        if mode == .broll || voiced.isEmpty { return whole }
        // Silent gaps (before the first voice, between voices, after the last voice).
        var gaps: [Range] = []
        var cursor = 0.0
        for v in voiced {
            if v.start > cursor { gaps.append(Range(start: cursor, end: v.start)) }
            cursor = max(cursor, v.end)
        }
        if cursor < duration { gaps.append(Range(start: cursor, end: duration)) }
        // A gap is dropped only if it is long; its first / last 0.3 s stay (the margins around the voice).
        var dropped: [Range] = []
        for g in gaps {
            let leadsVoice = g.end < duration - 1e-9          // a voice follows this gap
            let followsVoice = g.start > 1e-9                 // a voice comes before it
            // Pauses of 0.7 s or less between two voices are kept whole.
            if leadsVoice && followsVoice && g.length <= silence + 1e-6 { continue }
            let start = g.start + (followsVoice ? margin : 0)
            let end = g.end - (leadsVoice ? margin : 0)
            if end > start { dropped.append(Range(start: start, end: end)) }
        }
        return subtract(whole, merged(dropped))
    }

    static func subtract(_ base: [Range], _ remove: [Range]) -> [Range] {
        var out: [Range] = []
        for b in base {
            var pieces = [b]
            for r in remove {
                pieces = pieces.flatMap { p -> [Range] in
                    if r.end <= p.start || r.start >= p.end { return [p] }
                    var parts: [Range] = []
                    if r.start > p.start { parts.append(Range(start: p.start, end: r.start)) }
                    if r.end < p.end { parts.append(Range(start: r.end, end: p.end)) }
                    return parts
                }
            }
            out.append(contentsOf: pieces)
        }
        return out.filter { $0.length > 0.01 }
    }

    static func savedSeconds(_ ranges: [Range]) -> Double { ranges.reduce(0) { $0 + $1.length } }

    /// Auto-stop: 1 minute without voice (Talk only; never in B-roll).
    static func shouldAutoStop(mode: CaptureMode = .talk, now: Double, lastVoice: Double?,
                               recordingStart: Double = 0) -> Bool {
        guard mode == .talk else { return false }
        return now - (lastVoice ?? recordingStart) >= autoStopAfter
    }

    /// The badge while recording: voice now (or within the 0.7 s pause) -> "REC", otherwise "Waiting for voice…".
    static func isSavingNow(mode: CaptureMode = .talk, now: Double, lastVoice: Double?) -> Bool {
        if mode == .broll { return true }
        guard let lastVoice else { return false }
        return now - lastVoice <= silence
    }
}

/// How a clip is marked as B-roll: in the video file itself (QuickTime metadata) and in the source metadata sent
/// with the cut (`clip_type: "broll"`), for clips recorded in B-roll mode and gallery clips the user marked.
enum ClipMark {
    static let fileKey = "ai.cutsell.cliptype"
    static let sourceKey = "clip_type"
    static let broll = "broll"
    static let primary = "primary"
    static let kindKey = "broll_kind"
    static let orderKey = "broll_order"
    static let keepOrderKey = "broll_keep_order"
    static let brollKeys = [sourceKey, kindKey, orderKey, keepOrderKey]

    /// The source metadata of a clip: its base metadata plus its tags (the tags win).
    static func sourceMetadata(_ base: [String: String]?, tags: [String: String]) -> [String: String]? {
        guard !tags.isEmpty else { return base }
        return (base ?? [:]).merging(tags) { _, tag in tag }
    }

    static func fileItem() -> AVMetadataItem {
        let item = AVMutableMetadataItem()
        item.keySpace = .quickTimeMetadata
        item.key = fileKey as NSString
        item.value = broll as NSString
        return item
    }

    /// Reads the mark back from a video file.
    static func isBroll(file url: URL) async -> Bool {
        let items = (try? await AVURLAsset(url: url).load(.metadata)) ?? []
        for item in items where (item.key as? String) == fileKey || item.identifier?.rawValue.hasSuffix(fileKey) == true {
            if (try? await item.load(.stringValue)) == broll { return true }
        }
        return false
    }

    /// Adds `clip_type: "broll"` to a source's metadata when the clip is B-roll.
    static func sourceMetadata(_ base: [String: String]?, broll isBroll: Bool, primary isPrimary: Bool = false)
        -> [String: String]? {
        guard isBroll || isPrimary else { return base }
        var out = base ?? [:]
        out[sourceKey] = isBroll ? broll : primary
        return out
    }

    /// Gallery: the clips the user marked B-roll (by Photos id) carry the mark.
    static func marked(_ video: LocalVideo, assetID: String, brollIDs: Set<String>) -> LocalVideo {
        var out = video
        out.isBroll = brollIDs.contains(assetID)
        return out
    }
}
