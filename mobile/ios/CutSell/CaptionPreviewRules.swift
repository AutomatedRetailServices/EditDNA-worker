import Foundation

/// Caption rules for the in-editor preview.
///
/// This file is a line-for-line port of what the server does when it burns
/// captions into the exported video, so the editor shows the same phrases, at
/// the same moments, in the same place:
/// - phrase grouping  -> `cutsell_worker/render_plan.py` (`timed_caption_word_groups`)
/// - cue clamping and the highlighted word -> `cutsell_worker/caption_render.py`
///   (`build_caption_ass`, `_usable_words`, `clean_caption_text`)
/// - placement and size -> `cutsell_worker/caption_render.py` (`_placement`,
///   `_fit_size`, `caption_layout`)
///
/// Nothing here talks to the network or changes the draft: it only reads the
/// draft the server already returned. `tests/test_cutsell_ios_captions_v2_preview.py`
/// checks that the numbers below stay equal to the server's.

struct CaptionHighlightStep: Equatable {
    let start: Double
    let end: Double
    let wordIndex: Int
}

/// One caption as it appears on the preview. Times are seconds on the
/// preview timeline (clips played back to back).
struct CaptionPreviewCue: Equatable, Identifiable {
    let id: String
    let clipID: String
    let start: Double
    let end: Double
    let text: String
    /// The cue's words, only when the word being spoken can be highlighted.
    let words: [String]
    let highlightSteps: [CaptionHighlightStep]
    /// `true`: a short timed phrase (Editor v2 look, movable and resizable).
    /// `false`: one caption for the whole clip (clips without word timings,
    /// or text the creator edited by hand), drawn the way the server draws it.
    let isTimed: Bool

    func activeWordIndex(at time: Double) -> Int? {
        highlightSteps.first { time >= $0.start && time < $0.end }?.wordIndex
    }
}

/// Where one selected clip sits on the preview timeline.
struct CaptionClipWindow: Equatable {
    let clipIndex: Int
    let clipID: String
    let timelineStart: Double
    /// Seconds the clip plays (its speed applied).
    let duration: Double
    /// The clip's speed: word times of the original are divided by it (the export does the same).
    var speed: Double = 1
}

enum CaptionPreviewRules {
    // render_plan.py
    static let cueMaxWords = 3
    static let cueMaxGapSec = 0.45
    static let cueTailHoldSec = 0.30
    // caption_render.py / render.py
    static let cueTextLimit = 120
    static let wordTextLimit = 60
    static let wholeClipTextLimit = 500
    static let minimumCueSec = 0.05
    static let minimumHighlightStepSec = 0.02
    static let timedEngine = "simple"

    private struct TimedWord {
        let start: Double
        let end: Double
        let text: String
    }

    private struct TimedGroup {
        let start: Double
        let end: Double
        let words: [TimedWord]
    }

    static func collapseWhitespace(_ raw: String) -> String {
        raw.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }

    /// `clean_caption_text`: one line of plain text, no styling characters.
    static func cleanText(_ raw: String, limit: Int = cueTextLimit) -> String {
        var text = raw.replacingOccurrences(of: "\u{0}", with: "")
        for forbidden in ["{", "}", "\\"] {
            text = text.replacingOccurrences(of: forbidden, with: "")
        }
        return String(collapseWhitespace(text).prefix(limit))
    }

    /// Every caption of the preview, in timeline order.
    static func cues(
        clips: [[String: JSONValue]],
        windows: [CaptionClipWindow],
        engine: String?
    ) -> [CaptionPreviewCue] {
        var output: [CaptionPreviewCue] = []
        for window in windows {
            guard clips.indices.contains(window.clipIndex) else { continue }
            let clip = clips[window.clipIndex]
            // render.py: a clip with no caption text gets no caption at all.
            if collapseWhitespace(captionText(clip)).isEmpty { continue }
            var clipCues: [CaptionPreviewCue] = []
            if engine == timedEngine {
                clipCues = timedCues(clip: clip, window: window)
            }
            if clipCues.isEmpty, let whole = wholeClipCue(clip: clip, window: window) {
                clipCues = [whole]
            }
            output.append(contentsOf: clipCues)
        }
        return output
    }

    private static func spokenText(_ clip: [String: JSONValue]) -> String {
        clip["text"]?.stringValue ?? ""
    }

    private static func captionText(_ clip: [String: JSONValue]) -> String {
        // serde.py: a clip without its own caption text captions its spoken text.
        clip["caption_text"]?.stringValue ?? spokenText(clip)
    }

    private static func round3(_ value: Double) -> Double {
        (value * 1000).rounded() / 1000
    }

    private static func endsPhrase(_ word: String) -> Bool {
        var trimmed = Substring(word)
        while let last = trimmed.last, last.isWhitespace { trimmed = trimmed.dropLast() }
        guard let last = trimmed.last else { return false }
        return last == "." || last == "?" || last == "!" || last == ","
    }

    /// `timed_caption_word_groups`: groups of up to three spoken words, times
    /// relative to the clip start. Empty when the clip has no word timings or
    /// its caption was edited by hand.
    private static func timedGroups(clip: [String: JSONValue]) -> [TimedGroup] {
        let words: [TimedWord] = (clip["words"]?.arrayValue ?? []).compactMap { item in
            guard let object = item.objectValue,
                  let start = object["start"]?.doubleValue,
                  let end = object["end"]?.doubleValue,
                  end > start else { return nil }
            return TimedWord(start: start, end: end, text: object["text"]?.stringValue ?? "")
        }
        guard !words.isEmpty else { return [] }
        guard collapseWhitespace(captionText(clip)) == collapseWhitespace(spokenText(clip)) else { return [] }

        let clipStart = clip["start"]?.doubleValue ?? 0
        let clipEnd = clip["end"]?.doubleValue ?? clipStart

        var groups: [[TimedWord]] = []
        var current: [TimedWord] = []
        for word in words {
            if word.end <= clipStart || word.start >= clipEnd { continue }
            if let previous = current.last, word.start - previous.end > cueMaxGapSec {
                groups.append(current)
                current = []
            }
            current.append(word)
            if current.count >= cueMaxWords || endsPhrase(word.text) {
                groups.append(current)
                current = []
            }
        }
        if !current.isEmpty { groups.append(current) }

        var output: [TimedGroup] = []
        for (index, group) in groups.enumerated() {
            guard let first = group.first, let last = group.last else { continue }
            let cueStart = max(0, first.start - clipStart)
            let lastEnd = last.end - clipStart
            var cueEnd: Double
            if index + 1 < groups.count, let nextFirst = groups[index + 1].first {
                cueEnd = min(nextFirst.start - clipStart, lastEnd + cueTailHoldSec * 2)
            } else {
                cueEnd = lastEnd + cueTailHoldSec
            }
            cueEnd = min(cueEnd, clipEnd - clipStart)
            let text = group.map(\.text).joined(separator: " ")
            guard cueEnd - cueStart >= minimumCueSec,
                  !collapseWhitespace(text).isEmpty else { continue }
            output.append(TimedGroup(
                start: round3(cueStart),
                end: round3(cueEnd),
                words: group.map {
                    TimedWord(
                        start: round3(max(0, $0.start - clipStart)),
                        end: round3($0.end - clipStart),
                        text: $0.text
                    )
                }
            ))
        }
        return output
    }

    /// `build_caption_ass`: cues clamped to the clip, never overlapping, with
    /// the moment each word lights up (used by the Highlight looks only).
    private static func timedCues(clip: [String: JSONValue], window: CaptionClipWindow) -> [CaptionPreviewCue] {
        var output: [CaptionPreviewCue] = []
        var previousEnd = 0.0
        let rate = max(0.1, window.speed)
        let groups = timedGroups(clip: clip).map { group in
            TimedGroup(start: group.start / rate, end: group.end / rate,
                       words: group.words.map { TimedWord(start: $0.start / rate, end: $0.end / rate, text: $0.text) })
        }
        for (index, group) in groups.enumerated() {
            let text = cleanText(group.words.map(\.text).joined(separator: " "))
            let start = max(group.start, previousEnd)
            let end = min(group.end, window.duration)
            if text.isEmpty || end - start < minimumCueSec { continue }
            previousEnd = end

            // `_usable_words`: the word list must spell exactly the cue text.
            var usable = group.words
                .map { TimedWord(start: $0.start, end: $0.end, text: cleanText($0.text, limit: wordTextLimit)) }
                .filter { !$0.text.isEmpty }
            if usable.map(\.text).joined(separator: " ") != text { usable = [] }

            var steps: [CaptionHighlightStep] = []
            var cursor = start
            for (position, word) in usable.enumerated() {
                let stepStart = position == 0 ? cursor : max(cursor, min(word.start, end))
                let stepEnd: Double
                if position + 1 < usable.count {
                    stepEnd = max(stepStart, min(usable[position + 1].start, end))
                } else {
                    stepEnd = end
                }
                if stepEnd - stepStart < minimumHighlightStepSec { continue }
                steps.append(CaptionHighlightStep(
                    start: window.timelineStart + stepStart,
                    end: window.timelineStart + stepEnd,
                    wordIndex: position
                ))
                cursor = stepEnd
            }

            output.append(CaptionPreviewCue(
                id: "\(window.clipIndex)-\(window.clipID)-\(index)",
                clipID: window.clipID,
                start: window.timelineStart + start,
                end: window.timelineStart + end,
                text: text,
                words: usable.map(\.text),
                highlightSteps: steps,
                isTimed: true
            ))
        }
        return output
    }

    /// `render.py` `_caption_filter`: one caption for the whole clip.
    private static func wholeClipCue(clip: [String: JSONValue], window: CaptionClipWindow) -> CaptionPreviewCue? {
        let raw = captionText(clip).replacingOccurrences(of: "\u{0}", with: "")
        let text = String(collapseWhitespace(raw).prefix(wholeClipTextLimit))
        guard !text.isEmpty else { return nil }
        return CaptionPreviewCue(
            id: "\(window.clipIndex)-\(window.clipID)-whole",
            clipID: window.clipID,
            start: window.timelineStart,
            end: window.timelineStart + window.duration,
            text: text,
            words: [],
            highlightSteps: [],
            isTimed: false
        )
    }
}

/// One of the nine caption typefaces. The files are the server's own
/// (`cutsell_worker/fonts`, bundled into the app by `project.yml`), so the
/// editor and the exported video use the very same letters.
struct CaptionFontSpec: Equatable, Identifiable {
    /// The value sent to the server as `caption_font`.
    let key: String
    let displayName: String
    let fileName: String
    let fileExtension: String
    let postScriptName: String
    /// Size in pixels on a 1080x1920 frame, as the server's renderer counts it.
    let renderSize: Double
    /// Rough width of one character as a fraction of the size (keeps long words inside the frame).
    let charWidth: Double
    // Metrics read from the font file itself. The server's renderer sizes a
    // face by its tallest box (`winAscent + winDescent`), not by its em.
    let unitsPerEm: Double
    let ascent: Double
    let descent: Double
    let lineGap: Double
    let winAscent: Double
    let winDescent: Double

    var id: String { key }

    /// Point size of the face for a given renderer size.
    func emSize(forRenderSize size: Double) -> Double {
        size * unitsPerEm / (winAscent + winDescent)
    }

    /// Height of one line as the system lays this face out, for a given em.
    func naturalLineHeight(em: Double) -> Double {
        (ascent + descent + lineGap) / unitsPerEm * em
    }

    /// How far down the text must move so its baseline sits where the
    /// server's renderer puts it (the renderer centres the tallest box).
    func baselineShift(renderSize size: Double) -> Double {
        let em = emSize(forRenderSize: size)
        let wanted = size * (winAscent / (winAscent + winDescent) - 0.5)
        let natural = ascent / unitsPerEm * em - naturalLineHeight(em: em) / 2
        return wanted - natural
    }
}

enum CaptionFontCatalog {
    static let defaultKey = "montserrat"

    static let all: [CaptionFontSpec] = [
        CaptionFontSpec(key: "montserrat", displayName: "Montserrat", fileName: "Montserrat-ExtraBold", fileExtension: "ttf",
                        postScriptName: "Montserrat-ExtraBold", renderSize: 94, charWidth: 0.68,
                        unitsPerEm: 1000, ascent: 968, descent: 251, lineGap: 0, winAscent: 1109, winDescent: 453),
        CaptionFontSpec(key: "poppins", displayName: "Poppins", fileName: "Poppins-Bold", fileExtension: "ttf",
                        postScriptName: "Poppins-Bold", renderSize: 96, charWidth: 0.64,
                        unitsPerEm: 1000, ascent: 1050, descent: 350, lineGap: 100, winAscent: 1135, winDescent: 627),
        CaptionFontSpec(key: "roboto", displayName: "Roboto", fileName: "Roboto-Bold", fileExtension: "ttf",
                        postScriptName: "Roboto-Bold", renderSize: 86, charWidth: 0.58,
                        unitsPerEm: 2048, ascent: 1900, descent: 500, lineGap: 0, winAscent: 1946, winDescent: 512),
        CaptionFontSpec(key: "oswald", displayName: "Oswald", fileName: "Oswald-Bold", fileExtension: "ttf",
                        postScriptName: "Oswald-Bold", renderSize: 118, charWidth: 0.47,
                        unitsPerEm: 1000, ascent: 1193, descent: 289, lineGap: 0, winAscent: 1325, winDescent: 377),
        CaptionFontSpec(key: "anton", displayName: "Anton", fileName: "Anton-Regular", fileExtension: "ttf",
                        postScriptName: "Anton-Regular", renderSize: 116, charWidth: 0.45,
                        unitsPerEm: 2048, ascent: 2409, descent: 674, lineGap: 0, winAscent: 2876, winDescent: 674),
        CaptionFontSpec(key: "luckiest_guy", displayName: "Luckiest Guy", fileName: "LuckiestGuy-Regular", fileExtension: "ttf",
                        postScriptName: "LuckiestGuy-Regular", renderSize: 80, charWidth: 0.66,
                        unitsPerEm: 2048, ascent: 1440, descent: 608, lineGap: 0, winAscent: 2006, winDescent: 504),
        CaptionFontSpec(key: "bebas_neue", displayName: "Bebas Neue", fileName: "BebasNeue-Regular", fileExtension: "ttf",
                        postScriptName: "BebasNeue-Regular", renderSize: 112, charWidth: 0.42,
                        unitsPerEm: 1000, ascent: 900, descent: 300, lineGap: 0, winAscent: 950, winDescent: 350),
        CaptionFontSpec(key: "inter", displayName: "Inter", fileName: "Inter-Bold", fileExtension: "otf",
                        postScriptName: "Inter-Bold", renderSize: 84, charWidth: 0.62,
                        unitsPerEm: 2048, ascent: 1984, descent: 494, lineGap: 0, winAscent: 1984, winDescent: 494),
        CaptionFontSpec(key: "bangers", displayName: "Bangers", fileName: "Bangers-Regular", fileExtension: "ttf",
                        postScriptName: "Bangers-Regular", renderSize: 124, charWidth: 0.44,
                        unitsPerEm: 1000, ascent: 883, descent: 181, lineGap: 0, winAscent: 1401, winDescent: 356),
        CaptionFontSpec(key: "playfair", displayName: "Playfair", fileName: "PlayfairDisplay_800ExtraBold", fileExtension: "ttf",
                        postScriptName: "PlayfairDisplay-ExtraBold", renderSize: 101, charWidth: 0.6,
                        unitsPerEm: 1000, ascent: 1082, descent: 251, lineGap: 0, winAscent: 1159, winDescent: 251),
        CaptionFontSpec(key: "pacifico", displayName: "Pacifico", fileName: "Pacifico_400Regular", fileExtension: "ttf",
                        postScriptName: "Pacifico-Regular", renderSize: 117, charWidth: 0.54,
                        unitsPerEm: 1000, ascent: 1303, descent: 453, lineGap: 0, winAscent: 1478, winDescent: 457),
        CaptionFontSpec(key: "lobster", displayName: "Lobster", fileName: "Lobster_400Regular", fileExtension: "ttf",
                        postScriptName: "Lobster-Regular", renderSize: 101, charWidth: 0.49,
                        unitsPerEm: 1000, ascent: 1000, descent: 250, lineGap: 0, winAscent: 1000, winDescent: 490),
        CaptionFontSpec(key: "caveat", displayName: "Caveat", fileName: "Caveat_700Bold", fileExtension: "ttf",
                        postScriptName: "Caveat-Bold", renderSize: 108, charWidth: 0.45,
                        unitsPerEm: 1000, ascent: 960, descent: 300, lineGap: 0, winAscent: 974, winDescent: 315),
        CaptionFontSpec(key: "marker", displayName: "Marker", fileName: "PermanentMarker_400Regular", fileExtension: "ttf",
                        postScriptName: "PermanentMarker-Regular", renderSize: 98, charWidth: 0.65,
                        unitsPerEm: 1024, ascent: 1136, descent: 325, lineGap: 31, winAscent: 1136, winDescent: 325),
        CaptionFontSpec(key: "courier", displayName: "Courier", fileName: "CourierPrime_700Bold", fileExtension: "ttf",
                        postScriptName: "CourierPrime-Bold", renderSize: 116, charWidth: 0.74,
                        unitsPerEm: 2048, ascent: 1600, descent: 700, lineGap: 0, winAscent: 1900, winDescent: 800),
        CaptionFontSpec(key: "raleway", displayName: "Raleway", fileName: "Raleway_800ExtraBold", fileExtension: "ttf",
                        postScriptName: "Raleway-ExtraBold", renderSize: 100, charWidth: 0.64,
                        unitsPerEm: 1000, ascent: 940, descent: 234, lineGap: 0, winAscent: 1154, winDescent: 234),
        CaptionFontSpec(key: "righteous", displayName: "Righteous", fileName: "Righteous_400Regular", fileExtension: "ttf",
                        postScriptName: "Righteous-Regular", renderSize: 90, charWidth: 0.62,
                        unitsPerEm: 2048, ascent: 2017, descent: 526, lineGap: 0, winAscent: 2017, winDescent: 526),
        CaptionFontSpec(key: "fredoka", displayName: "Fredoka", fileName: "Fredoka_700Bold", fileExtension: "ttf",
                        postScriptName: "Fredoka-Bold", renderSize: 91, charWidth: 0.6,
                        unitsPerEm: 1000, ascent: 974, descent: 236, lineGap: 0, winAscent: 999, winDescent: 252),
        CaptionFontSpec(key: "bungee", displayName: "Bungee", fileName: "Bungee_400Regular", fileExtension: "ttf",
                        postScriptName: "Bungee-Regular", renderSize: 112, charWidth: 0.77,
                        unitsPerEm: 1000, ascent: 1020, descent: 300, lineGap: 0, winAscent: 1647, winDescent: 927),
        CaptionFontSpec(key: "lilita", displayName: "Lilita", fileName: "LilitaOne_400Regular", fileExtension: "ttf",
                        postScriptName: "LilitaOne", renderSize: 83, charWidth: 0.55,
                        unitsPerEm: 1000, ascent: 923, descent: 220, lineGap: 0, winAscent: 923, winDescent: 220),
        CaptionFontSpec(key: "archivo", displayName: "Archivo", fileName: "ArchivoBlack_400Regular", fileExtension: "ttf",
                        postScriptName: "ArchivoBlack-Regular", renderSize: 100, charWidth: 0.71,
                        unitsPerEm: 1000, ascent: 878, descent: 210, lineGap: 0, winAscent: 1035, winDescent: 312),
        CaptionFontSpec(key: "abril", displayName: "Abril", fileName: "AbrilFatface_400Regular", fileExtension: "ttf",
                        postScriptName: "AbrilFatface-Regular", renderSize: 98, charWidth: 0.59,
                        unitsPerEm: 1000, ascent: 1058, descent: 291, lineGap: 0, winAscent: 1058, winDescent: 291),
        CaptionFontSpec(key: "shrikhand", displayName: "Shrikhand", fileName: "Shrikhand_400Regular", fileExtension: "ttf",
                        postScriptName: "Shrikhand-Regular", renderSize: 111, charWidth: 0.67,
                        unitsPerEm: 1000, ascent: 1026, descent: 432, lineGap: 0, winAscent: 1026, winDescent: 432),
        CaptionFontSpec(key: "dancing", displayName: "Dancing", fileName: "DancingScript_700Bold", fileExtension: "ttf",
                        postScriptName: "DancingScript-Bold", renderSize: 99, charWidth: 0.47,
                        unitsPerEm: 1000, ascent: 920, descent: 280, lineGap: 0, winAscent: 1096, winDescent: 306),
        CaptionFontSpec(key: "satisfy", displayName: "Satisfy", fileName: "Satisfy_400Regular", fileExtension: "ttf",
                        postScriptName: "Satisfy-Regular", renderSize: 97, charWidth: 0.49,
                        unitsPerEm: 1024, ascent: 957, descent: 501, lineGap: 17, winAscent: 957, winDescent: 501),
        CaptionFontSpec(key: "vibes", displayName: "Vibes", fileName: "GreatVibes_400Regular", fileExtension: "ttf",
                        postScriptName: "GreatVibes-Regular", renderSize: 119, charWidth: 0.37,
                        unitsPerEm: 1000, ascent: 851, descent: 401, lineGap: 0, winAscent: 1160, winDescent: 590),
        CaptionFontSpec(key: "kaushan", displayName: "Kaushan", fileName: "KaushanScript_400Regular", fileExtension: "ttf",
                        postScriptName: "KaushanScript-Regular", renderSize: 103, charWidth: 0.51,
                        unitsPerEm: 1000, ascent: 1084, descent: 367, lineGap: 0, winAscent: 1084, winDescent: 367),
        CaptionFontSpec(key: "amatic", displayName: "Amatic", fileName: "AmaticSC_700Bold", fileExtension: "ttf",
                        postScriptName: "AmaticSC-Bold", renderSize: 94, charWidth: 0.37,
                        unitsPerEm: 1000, ascent: 1016, descent: 245, lineGap: 0, winAscent: 1139, winDescent: 265),
        CaptionFontSpec(key: "chewy", displayName: "Chewy", fileName: "Chewy_400Regular", fileExtension: "ttf",
                        postScriptName: "Chewy-Regular", renderSize: 87, charWidth: 0.52,
                        unitsPerEm: 1024, ascent: 1003, descent: 310, lineGap: 25, winAscent: 1003, winDescent: 310),
        CaptionFontSpec(key: "cinzel", displayName: "Cinzel", fileName: "Cinzel_700Bold", fileExtension: "ttf",
                        postScriptName: "Cinzel-Bold", renderSize: 98, charWidth: 0.74,
                        unitsPerEm: 1000, ascent: 976, descent: 372, lineGap: 0, winAscent: 976, winDescent: 372),
        CaptionFontSpec(key: "monoton", displayName: "Monoton", fileName: "Monoton_400Regular", fileExtension: "ttf",
                        postScriptName: "Monoton-Regular", renderSize: 98, charWidth: 0.8,
                        unitsPerEm: 2048, ascent: 2366, descent: 822, lineGap: 0, winAscent: 2366, winDescent: 822),
        CaptionFontSpec(key: "nunito", displayName: "Nunito", fileName: "Nunito_900Black", fileExtension: "ttf",
                        postScriptName: "Nunito-Black", renderSize: 99, charWidth: 0.64,
                        unitsPerEm: 1000, ascent: 1011, descent: 353, lineGap: 0, winAscent: 1077, winDescent: 300),
        CaptionFontSpec(key: "rubik", displayName: "Rubik", fileName: "Rubik_800ExtraBold", fileExtension: "ttf",
                        postScriptName: "Rubik-ExtraBold", renderSize: 111, charWidth: 0.67,
                        unitsPerEm: 1000, ascent: 935, descent: 250, lineGap: 0, winAscent: 1066, winDescent: 466),
    ]

    static func spec(for key: String) -> CaptionFontSpec {
        all.first { $0.key == key } ?? all[0]
    }
}

/// Placement of a timed caption, in the server's own 1080x1920 units.
struct CaptionPlacement: Equatable {
    let centerX: Double
    let centerY: Double
    /// Widest the caption may be; longer phrases wrap inside it.
    let column: Double
    let fontSize: Double
}

enum CaptionLayout {
    static let frameWidth = 1080.0
    static let frameHeight = 1920.0
    static let defaultX = 0.5
    static let defaultY = 0.758
    static let defaultScale = 1.0
    static let minimumScale = 0.5
    static let maximumScale = 2.0
    static let edgePad = 36.0
    static let minimumColumn = 520.0
    static let minimumFittedSize = 12.0

    static func clampedScale(_ scale: Double) -> Double {
        guard scale.isFinite else { return defaultScale }
        return min(max(scale, minimumScale), maximumScale)
    }

    static func clampedUnit(_ value: Double, fallback: Double) -> Double {
        guard value.isFinite else { return fallback }
        return min(max(value, 0), 1)
    }

    /// `_placement`: the caption is centred on the chosen point inside a
    /// column that never crosses a frame edge.
    static func placement(x: Double, y: Double, scale: Double, fontKey: String) -> CaptionPlacement {
        let spec = CaptionFontCatalog.spec(for: fontKey)
        let safeX = clampedUnit(x, fallback: defaultX)
        let safeY = clampedUnit(y, fallback: defaultY)
        let size = max(1, (spec.renderSize * clampedScale(scale)).rounded())
        let halfMinimum = (minimumColumn / 2).rounded(.down) + edgePad
        let centerX = min(max(safeX * frameWidth, halfMinimum), frameWidth - halfMinimum).rounded()
        let halfColumn = min(centerX, frameWidth - centerX) - edgePad
        let margin = max(0, ((frameWidth - 2 * halfColumn) / 2).rounded(.down))
        let halfHeight = (size * 1.3).rounded(.down) + edgePad
        let centerY = min(max(safeY * frameHeight, halfHeight), frameHeight - halfHeight).rounded()
        return CaptionPlacement(
            centerX: centerX,
            centerY: centerY,
            column: frameWidth - 2 * margin,
            fontSize: size
        )
    }

    /// `_fit_size`: a single word cannot wrap, so a cue whose longest word is
    /// wider than the column is drawn smaller.
    static func fittedSize(text: String, size: Double, column: Double, fontKey: String) -> Double {
        let spec = CaptionFontCatalog.spec(for: fontKey)
        let longest = text.split(whereSeparator: { $0.isWhitespace }).map(\.count).max() ?? 0
        guard longest > 0 else { return size }
        let widest = Double(longest) * size * spec.charWidth
        if widest <= column { return size }
        return max(minimumFittedSize, (size * column / widest).rounded(.down))
    }
}

/// The looks the server knows (`_PRESET_LOOKS`).
struct CaptionLook: Equatable {
    enum Kind: Equatable { case plain, highlight, box }

    let kind: Kind
    /// RGB of the words.
    let textHex: UInt32
    /// RGB of the word being spoken (Highlight only).
    let highlightHex: UInt32?
    /// RGB of the box behind the words (Box only).
    let boxHex: UInt32?

    static let white: UInt32 = 0xFFFFFF
    static let ink: UInt32 = 0x0B1020
    static let yellow: UInt32 = 0xFFD60A
    static let highlightGreen: UInt32 = 0x39FF78
    static let highlightRed: UInt32 = 0xFF453A
    static let highlightBlue: UInt32 = 0x1AA6FF

    // Sizes in 1080x1920 units, multiplied by the caption scale.
    static let outline = 3.0
    static let shadow = 3.0
    static let boxPadding = 16.0

    static func resolve(preset: String) -> CaptionLook {
        switch preset {
        case "yellow":
            return CaptionLook(kind: .plain, textHex: yellow, highlightHex: nil, boxHex: nil)
        case "highlight", "highlight_green":
            return CaptionLook(kind: .highlight, textHex: white, highlightHex: highlightGreen, boxHex: nil)
        case "highlight_red":
            return CaptionLook(kind: .highlight, textHex: white, highlightHex: highlightRed, boxHex: nil)
        case "highlight_blue":
            return CaptionLook(kind: .highlight, textHex: white, highlightHex: highlightBlue, boxHex: nil)
        case "box", "clean":
            return CaptionLook(kind: .box, textHex: white, highlightHex: nil, boxHex: ink)
        case "box_light":
            return CaptionLook(kind: .box, textHex: ink, highlightHex: nil, boxHex: white)
        default:
            return CaptionLook(kind: .plain, textHex: white, highlightHex: nil, boxHex: nil)
        }
    }
}
