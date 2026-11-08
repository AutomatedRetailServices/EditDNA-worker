import Foundation

/// Auto-pause by voice + B-roll (PO 8 Nov 2026), the rules only (unit-tested with synthetic timelines).
///
/// What is kept:
/// - every moment with voice, plus 0.3 s before it (the first word is never cut) and 0.3 s after it;
/// - pauses of 0.7 s or less, whole (natural breathing between phrases);
/// - every moment with NO person on camera (B-roll: the product, the room), voice or not;
/// - everything, when B-roll mode is on or when no voice was heard at all.
/// What is dropped: silence longer than 0.7 s while a person is on camera (minus the 0.3 s margins).
enum AutoPausePlan {
    static let silence: Double = 0.7
    static let margin: Double = 0.3
    static let autoStopAfter: Double = 60

    struct Range: Equatable {
        var start: Double
        var end: Double
        var length: Double { max(0, end - start) }
    }

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

    /// Ranges with a person / face on camera, from samples taken ~5 times a second (each sample covers the time
    /// until the next one).
    static func presence(samples: [(time: Double, present: Bool)], duration: Double) -> [Range] {
        var out: [Range] = []
        for (i, s) in samples.enumerated() where s.present {
            let end = i + 1 < samples.count ? samples[i + 1].time : duration
            out.append(Range(start: s.time, end: end))
        }
        return merged(out)
    }

    /// The parts of the recording to keep.
    static func keep(voice: [Range], presence: [Range], duration: Double, brollMode: Bool) -> [Range] {
        guard duration > 0 else { return [] }
        let whole = [Range(start: 0, end: duration)]
        let voiced = merged(voice).map { Range(start: max(0, $0.start), end: min(duration, $0.end)) }
        if brollMode || voiced.isEmpty { return whole }
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
        // Silence is dropped only where a person is on camera (no person = B-roll = kept).
        let present = merged(presence)
        var cut: [Range] = []
        for d in dropped {
            for p in present {
                let s = max(d.start, p.start), e = min(d.end, p.end)
                if e > s { cut.append(Range(start: s, end: e)) }
            }
        }
        return subtract(whole, merged(cut))
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

    /// Auto-stop: 1 minute without voice while a person is on camera (B-roll moments do not count).
    static func shouldAutoStop(now: Double, lastVoice: Double?, recordingStart: Double = 0,
                               presence: [Range], brollMode: Bool) -> Bool {
        guard !brollMode else { return false }
        let since = lastVoice ?? recordingStart
        guard now - since >= autoStopAfter else { return false }
        // The whole silent minute had a person on camera.
        let window = Range(start: now - autoStopAfter, end: now)
        let covered = merged(presence).reduce(0.0) { acc, p in
            acc + max(0, min(p.end, window.end) - max(p.start, window.start))
        }
        return covered >= autoStopAfter - 0.5
    }

    /// The badge while recording: voice now (or within the 0.3 s tail), or no person on camera -> "REC".
    static func isSavingNow(now: Double, lastVoice: Double?, personOnCamera: Bool, brollMode: Bool) -> Bool {
        if brollMode || !personOnCamera { return true }
        guard let lastVoice else { return false }
        return now - lastVoice <= silence
    }
}
