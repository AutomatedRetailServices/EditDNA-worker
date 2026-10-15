import AVFoundation
import Foundation

// MARK: - Editor v5 data on the draft
//
// Every editor option lives on the draft (one save path, one Undo / Redo history, and the server's
// export reads the same fields):
//   clip:     speed (0.5 1 1.5 2 3) · rotation (0 90 180 270) · audio_volume (0–2) · audio_muted · mirror
//   overlay:  layer · layout · edge · fill · rotation · volume · mute_audio · source_start / source_end
//   audio_clips[]: audio_id · kind (voice | music | sfx) · asset_id | sfx_id · start · source_in ·
//             source_out · volume · duck_video · clean · enhanced_at_record
//   draft:    video_volume (0–2) · video_clean_voice · zoom_removed

/// One voice-over, music or CutSell sound on the timeline (draft `audio_clips`).
struct AudioClipItem: Identifiable, Equatable {
    enum Kind: String { case voice, music, sfx }

    let id: String
    var kind: Kind
    var assetID: String?
    var sfxID: String?
    var start: Double
    var sourceIn: Double
    var sourceOut: Double
    var volume: Double = 1
    var duckVideo = false
    var clean = false
    var enhancedAtRecord = false

    var length: Double { max(0, sourceOut - sourceIn) }
    var end: Double { start + length }

    init(id: String, kind: Kind, assetID: String? = nil, sfxID: String? = nil, start: Double,
         sourceIn: Double, sourceOut: Double, volume: Double = 1, duckVideo: Bool = false,
         clean: Bool = false, enhancedAtRecord: Bool = false) {
        self.id = id; self.kind = kind; self.assetID = assetID; self.sfxID = sfxID; self.start = start
        self.sourceIn = sourceIn; self.sourceOut = sourceOut; self.volume = volume; self.duckVideo = duckVideo
        self.clean = clean; self.enhancedAtRecord = enhancedAtRecord
    }

    init?(_ object: [String: JSONValue]) {
        guard let id = object["audio_id"]?.stringValue,
              let kind = Kind(rawValue: object["kind"]?.stringValue ?? ""),
              let start = object["start"]?.doubleValue,
              let sourceOut = object["source_out"]?.doubleValue else { return nil }
        self.id = id
        self.kind = kind
        assetID = object["asset_id"]?.stringValue
        sfxID = object["sfx_id"]?.stringValue
        self.start = start
        sourceIn = object["source_in"]?.doubleValue ?? 0
        self.sourceOut = sourceOut
        volume = object["volume"]?.doubleValue ?? 1
        duckVideo = object["duck_video"]?.boolValue ?? false
        clean = object["clean"]?.boolValue ?? false
        enhancedAtRecord = object["enhanced_at_record"]?.boolValue ?? false
    }

    var json: [String: JSONValue] {
        func ms(_ value: Double) -> JSONValue { .number((max(0, value) * 1000).rounded() / 1000) }
        var object: [String: JSONValue] = [
            "audio_id": .string(id), "kind": .string(kind.rawValue),
            "start": ms(start), "source_in": ms(sourceIn), "source_out": ms(sourceOut),
            "volume": .number((min(2, max(0, volume)) * 100).rounded() / 100),
        ]
        if let assetID { object["asset_id"] = .string(assetID) }
        if let sfxID { object["sfx_id"] = .string(sfxID) }
        if duckVideo { object["duck_video"] = .bool(true) }
        if clean { object["clean"] = .bool(true) }
        if enhancedAtRecord { object["enhanced_at_record"] = .bool(true) }
        return object
    }
}

/// How a photo / video overlay is drawn (Figma 1124:1191, 1124:1590).
enum OverlayLayout: String, CaseIterable, Identifiable {
    case float, side_by_side, stacked, left, right, top, bottom, full
    var id: String { rawValue }

    var title: String {
        switch self {
        case .float: return "Float"
        case .side_by_side: return "Side by side"
        case .stacked: return "Stacked"
        case .left: return "Left"
        case .right: return "Right"
        case .top: return "Top"
        case .bottom: return "Bottom"
        case .full: return "Full"
        }
    }

    /// Where the overlay goes in the frame (0...1), nil for Float (its own x, y and width).
    var region: CGRect? {
        switch self {
        case .float: return nil
        case .full: return CGRect(x: 0, y: 0, width: 1, height: 1)
        case .top: return CGRect(x: 0, y: 0, width: 1, height: 0.5)
        case .bottom, .stacked: return CGRect(x: 0, y: 0.5, width: 1, height: 0.5)
        case .left: return CGRect(x: 0, y: 0, width: 0.5, height: 1)
        case .right, .side_by_side: return CGRect(x: 0.5, y: 0, width: 0.5, height: 1)
        }
    }

    /// Stacked / Side by side also move the main video into the other half while the overlay shows.
    var baseRegion: CGRect? {
        switch self {
        case .stacked: return CGRect(x: 0, y: 0, width: 1, height: 0.5)
        case .side_by_side: return CGRect(x: 0, y: 0, width: 0.5, height: 1)
        default: return nil
        }
    }

    /// Fit leaves the rest of its region see-through (the video shows) for these; black for the others.
    var transparentPad: Bool { [.left, .right, .top, .bottom].contains(self) }
}

enum OverlayEdge: String, CaseIterable, Identifiable {
    case sharp, soft, extra_soft
    var id: String { rawValue }
    var title: String {
        switch self {
        case .sharp: return "Sharp"
        case .soft: return "Soft"
        case .extra_soft: return "Extra soft"
        }
    }
    /// Feather in points of a 1080-wide frame (the export's 12 px and 28 px).
    var featherOf1080: CGFloat {
        switch self {
        case .sharp: return 0
        case .soft: return 12
        case .extra_soft: return 28
        }
    }
}

/// Look of one overlay (saved on the draft overlay; kept on a pending one while it uploads).
struct OverlayStyle: Equatable {
    var layer = 0
    var layout: OverlayLayout = .float
    var edge: OverlayEdge = .sharp
    var fill = false
    var rotation: Double = 0
    var muted = true
    var volume: Double = 1

    init() {}

    init(_ object: [String: JSONValue]) {
        layer = Int(object["layer"]?.doubleValue ?? 0)
        layout = OverlayLayout(rawValue: object["layout"]?.stringValue ?? "") ?? .float
        edge = OverlayEdge(rawValue: object["edge"]?.stringValue ?? "") ?? .sharp
        fill = object["fill"]?.boolValue ?? false
        rotation = object["rotation"]?.doubleValue ?? 0
        muted = object["mute_audio"]?.boolValue ?? true
        volume = object["volume"]?.doubleValue ?? 1
    }

    /// Writes only what differs from the defaults (the server leaves defaults out too).
    func write(into object: inout [String: JSONValue]) {
        object["layer"] = layer == 0 ? nil : .number(Double(min(50, max(0, layer))))
        object["layout"] = layout == .float ? nil : .string(layout.rawValue)
        object["edge"] = edge == .sharp ? nil : .string(edge.rawValue)
        object["fill"] = fill ? .bool(true) : nil
        let turn = (rotation.truncatingRemainder(dividingBy: 360) * 10).rounded() / 10
        object["rotation"] = abs(turn) < 0.05 ? nil : .number(turn)
        object["mute_audio"] = .bool(muted)
        object["volume"] = abs(volume - 1) < 0.005 ? nil : .number((min(2, max(0, volume)) * 100).rounded() / 100)
    }
}

/// Speeds of the Speed panel (Figma 1119:1315).
enum ClipSpeed {
    static let choices: [Double] = [0.5, 1, 1.5, 2, 3]
    static func label(_ speed: Double) -> String {
        speed == speed.rounded() ? "\(Int(speed))x" : String(format: "%.1fx", speed)
    }
    static func of(_ clip: [String: JSONValue]?) -> Double {
        let value = clip?["speed"]?.doubleValue ?? 1
        return choices.contains(value) ? value : 1
    }
}

/// Where each overlay goes: every new overlay that meets another in time goes on a new row (layer);
/// the higher layer is drawn on top.
enum OverlayLayers {
    /// The first row where nothing overlaps [start, end); a new row when every row is taken.
    static func layer(for start: Double, end: Double, existing: [(layer: Int, start: Double, end: Double)]) -> (layer: Int, isNew: Bool) {
        let used = Set(existing.map(\.layer))
        let top = used.max() ?? -1
        if existing.isEmpty { return (0, false) }
        for row in 0...max(0, top) where used.contains(row) || row == 0 {
            let clash = existing.contains { $0.layer == row && $0.start < end - 0.01 && start < $0.end - 0.01 }
            if !clash { return (row, false) }
        }
        return (top + 1, true)
    }
}

// MARK: - The view model side

extension DraftEditorViewModel {
    // MARK: Speed, per-clip volume and rotation

    func clipSpeed(_ clipID: String) -> Double {
        ClipSpeed.of(selectedClips.first { $0["clip_id"]?.stringValue == clipID })
    }

    @discardableResult
    func setClipSpeed(clipID: String, speed: Double) -> Task<Void, Never> {
        let value = ClipSpeed.choices.contains(speed) ? speed : 1
        return applyDraftChange { object in
            Self.mapClips(&object) { clips in clips.map { clip in
                guard clip["clip_id"]?.stringValue == clipID else { return clip }
                var clip = clip
                clip["speed"] = value == 1 ? nil : .number(value)
                return clip
            } }
        }
    }

    @discardableResult
    func setClipVolume(clipID: String, volume: Double) -> Task<Void, Never> {
        let value = (min(2, max(0, volume)) * 100).rounded() / 100
        return applyDraftChange { object in
            Self.mapClips(&object) { clips in clips.map { clip in
                guard clip["clip_id"]?.stringValue == clipID else { return clip }
                var clip = clip
                clip["audio_volume"] = .number(value)
                clip["audio_muted"] = .bool(value <= 0.001)
                return clip
            } }
        }
    }

    func clipVolume(_ clipID: String) -> Double {
        guard let clip = selectedClips.first(where: { $0["clip_id"]?.stringValue == clipID }) else { return 1 }
        if clip["audio_muted"]?.boolValue == true { return 0 }
        return clip["audio_volume"]?.doubleValue ?? 1
    }

    func clipRotation(_ clipID: String) -> Int {
        Int(selectedClips.first { $0["clip_id"]?.stringValue == clipID }?["rotation"]?.doubleValue ?? 0) % 360
    }

    /// Rotation shown for the clip playing at `time` (whole video + that clip).
    func rotation(at time: Double) -> Int {
        let windows = EditorTracks.clipWindows(self)
        let window = windows.first { time >= $0.start && time < $0.end } ?? windows.last
        return (rotationDegrees + (window.map { clipRotation($0.clipID) } ?? 0)) % 360
    }

    @discardableResult
    func rotateClip(clipID: String) -> Task<Void, Never> {
        applyDraftChange { object in
            Self.mapClips(&object) { clips in clips.map { clip in
                guard clip["clip_id"]?.stringValue == clipID else { return clip }
                var clip = clip
                let next = (Int(clip["rotation"]?.doubleValue ?? 0) + 90) % 360
                clip["rotation"] = next == 0 ? nil : .number(Double(next))
                return clip
            } }
        }
    }

    // MARK: Whole-video volume and Clean up voice (Volume panel, Figma 1119:990)

    /// The video's own sound for the whole cut (0...2; 0 = muted).
    var videoVolume: Double { displayDraft?["video_volume"]?.doubleValue ?? 1 }
    var videoCleanVoice: Bool { displayDraft?["video_clean_voice"]?.boolValue ?? false }

    @discardableResult
    func setVideoVolume(_ volume: Double) -> Task<Void, Never> {
        let value = (min(2, max(0, volume)) * 100).rounded() / 100
        return applyDraftChange { $0["video_volume"] = abs(value - 1) < 0.005 ? nil : .number(value) }
    }

    @discardableResult
    func setVideoCleanVoice(_ on: Bool) -> Task<Void, Never> {
        applyDraftChange { $0["video_clean_voice"] = on ? .bool(true) : nil }
    }

    // MARK: Voice-overs, music and CutSell sounds (draft `audio_clips`)

    var audioClips: [AudioClipItem] {
        displayDraft?["audio_clips"]?.arrayValue?.compactMap(\.objectValue).compactMap(AudioClipItem.init) ?? []
    }

    var voiceClips: [AudioClipItem] { audioClips.filter { $0.kind == .voice } }
    var soundClips: [AudioClipItem] { audioClips.filter { $0.kind != .voice } }

    /// "Voiceover 1", "Tap", "Music 1"…
    func title(of clip: AudioClipItem) -> String {
        switch clip.kind {
        case .voice:
            let index = (voiceClips.sorted { $0.start < $1.start }.firstIndex { $0.id == clip.id } ?? 0) + 1
            return "Voiceover \(index)"
        case .music:
            let index = (audioClips.filter { $0.kind == .music }.firstIndex { $0.id == clip.id } ?? 0) + 1
            return "Music \(index)"
        case .sfx:
            return SfxLibrary.name(of: clip.sfxID ?? "")
        }
    }

    static func setAudioClips(_ object: inout [String: JSONValue], _ items: [AudioClipItem]) {
        object["audio_clips"] = .array(items.sorted { $0.start < $1.start }.map { .object($0.json) })
    }

    private static func mapAudioClips(_ object: inout [String: JSONValue], _ transform: ([AudioClipItem]) -> [AudioClipItem]) {
        let items = object["audio_clips"]?.arrayValue?.compactMap(\.objectValue).compactMap(AudioClipItem.init) ?? []
        setAudioClips(&object, transform(items))
    }

    @discardableResult
    func addAudioClip(_ item: AudioClipItem) -> Task<Void, Never> {
        applyDraftChange { object in Self.mapAudioClips(&object) { $0 + [item] } }
    }

    @discardableResult
    func updateAudioClip(id: String, _ edit: @escaping (inout AudioClipItem) -> Void) -> Task<Void, Never> {
        applyDraftChange { object in
            Self.mapAudioClips(&object) { items in items.map { item in
                guard item.id == id else { return item }
                var changed = item
                edit(&changed)
                return changed
            } }
        }
    }

    @discardableResult
    func removeAudioClip(id: String) -> Task<Void, Never> {
        applyDraftChange { object in Self.mapAudioClips(&object) { $0.filter { $0.id != id } } }
    }

    /// Cuts the clip at timeline `time` (both pieces keep the same recording, nothing is lost).
    @discardableResult
    func splitAudioClip(id: String, at time: Double) -> Bool {
        guard let item = audioClips.first(where: { $0.id == id }),
              time > item.start + 0.05, time < item.end - 0.05 else { return false }
        let cut = item.sourceIn + (time - item.start)
        var left = item, right = item
        left.sourceOut = cut
        right = AudioClipItem(id: "ac_\(UUID().uuidString.prefix(8).lowercased())", kind: item.kind, assetID: item.assetID,
                              sfxID: item.sfxID, start: time, sourceIn: cut, sourceOut: item.sourceOut,
                              volume: item.volume, duckVideo: item.duckVideo, clean: item.clean,
                              enhancedAtRecord: item.enhancedAtRecord)
        applyDraftChange { object in
            Self.mapAudioClips(&object) { items in items.flatMap { $0.id == id ? [left, right] : [$0] } }
        }
        return true
    }

    /// Moves a clip so it starts at `start` (voices go to the first free place: they never overlap).
    @discardableResult
    func moveAudioClip(id: String, to start: Double) -> Task<Void, Never>? {
        guard let item = audioClips.first(where: { $0.id == id }) else { return nil }
        var begin = max(0, start)
        if item.kind == .voice {
            let others = voiceClips.filter { $0.id != id }.map { (start: $0.start, end: $0.end) }
            begin = PendingVoiceOverStore.freeStart(begin, duration: item.length, taken: others)
        }
        let at = begin
        return updateAudioClip(id: id) { $0.start = at }
    }

    /// How long the clip's recording / sound really is (the trim's outer limit).
    func sourceLength(of clip: AudioClipItem) -> Double {
        switch clip.kind {
        case .sfx:
            if clip.sfxID == "typing" || clip.sfxID == "typing_long" { return SfxLibrary.duration(of: "typing_long") ?? 9 }
            return SfxLibrary.duration(of: clip.sfxID ?? "") ?? clip.sourceOut
        case .voice, .music:
            if let assetID = clip.assetID,
               let known = timelineAssetLibrary?.assets.first(where: { $0.assetID == assetID })?.durationSec, known > 0 {
                return known
            }
            return audioFileDurations[clip.assetID ?? ""] ?? clip.sourceOut
        }
    }

    /// Moves one edge of an audio clip to timeline `time` (never past its recording's ends).
    /// Typing stretched past 3 s switches to the 9 s typing sound.
    @discardableResult
    func trimAudioClip(id: String, edge: TrimEdge, to time: Double) -> Task<Void, Never>? {
        guard let item = audioClips.first(where: { $0.id == id }) else { return nil }
        let total = sourceLength(of: item)
        var changed = item
        switch edge {
        case .start:
            let delta = time - item.start
            let newIn = min(item.sourceOut - 0.1, max(0, item.sourceIn + delta))
            changed.start = max(0, item.start + (newIn - item.sourceIn))
            changed.sourceIn = newIn
        case .end:
            let newOut = max(item.sourceIn + 0.1, min(total, item.sourceOut + (time - item.end)))
            changed.sourceOut = newOut
            if item.kind == .sfx {
                if item.sfxID == "typing", newOut > 3.0 + 0.01 { changed.sfxID = "typing_long" }
                if item.sfxID == "typing_long", newOut <= 3.0 + 0.01, item.sourceIn < 0.01 { changed.sfxID = "typing" }
            }
        }
        let result = changed
        return updateAudioClip(id: id) { $0 = result }
    }

    /// The trim limit of an audio clip's edge on the timeline (start can't go before the recording's
    /// first second; end can't go past its last).
    func audioTrimLimits(_ item: AudioClipItem, edge: TrimEdge) -> ClosedRange<Double> {
        switch edge {
        case .start: return max(0, item.start - item.sourceIn)...(item.end - 0.1)
        case .end:
            let total = item.kind == .sfx && (item.sfxID == "typing" || item.sfxID == "typing_long")
                ? (SfxLibrary.duration(of: "typing_long") ?? 9) : sourceLength(of: item)
            return (item.start + 0.1)...max(item.start + 0.1, item.start + (total - item.sourceIn))
        }
    }

    /// A CutSell sound at the cursor (Audio panel "Add at playhead").
    @discardableResult
    func addSound(sfxID: String, at time: Double) -> String {
        let id = "ac_\(UUID().uuidString.prefix(8).lowercased())"
        let length = SfxLibrary.duration(of: sfxID) ?? (sfxID == "typing" ? 3 : 0.6)
        addAudioClip(AudioClipItem(id: id, kind: .sfx, sfxID: sfxID, start: max(0, min(time, max(0, timelineDuration - 0.1))),
                                   sourceIn: 0, sourceOut: length))
        return id
    }

    // MARK: Old voice-overs (D-282A timeline) move onto the draft once

    /// Voice-overs placed before Editor v5 lived on the separate timeline; they join the draft
    /// once (one save), so they cut, trim, move, Undo and export like every other clip.
    func adoptLegacyVoiceOvers() {
        guard displayDraft?["audio_clips"] == nil,
              let placements = timelineComposition?.voiceOverPlacements, !placements.isEmpty else { return }
        let items = placements.map { placement in
            AudioClipItem(id: "ac_\(placement.placementID)", kind: .voice, assetID: placement.assetID,
                          start: placement.timelineStartSec, sourceIn: placement.sourceInSec, sourceOut: placement.sourceOutSec)
        }
        applyDraftChange { object in Self.setAudioClips(&object, items) }
    }

    // MARK: Overlays: layers and look

    /// The style of an overlay (placed, or still uploading).
    func overlayStyle(_ id: String) -> OverlayStyle {
        if let pending = pendingOverlays.first(where: { $0.id == id }) { return pending.style }
        guard let item = mediaOverlays.first(where: { $0["overlay_id"]?.stringValue == id }) else { return OverlayStyle() }
        return OverlayStyle(item)
    }

    @discardableResult
    func setOverlayStyle(id: String, _ edit: @escaping (inout OverlayStyle) -> Void) -> Task<Void, Never>? {
        if pendingOverlays.contains(where: { $0.id == id }) {
            updatePendingOverlayStyle(id: id, edit)
            return nil
        }
        return changeOverlay(id: id) { item in
            var style = OverlayStyle(item)
            edit(&style)
            style.write(into: &item)
        }
    }

    /// Cuts an overlay at timeline `time`: two pieces on the same row (a video's second piece starts
    /// that much later in its own video).
    @discardableResult
    func splitOverlay(id: String, at time: Double) -> Bool {
        guard let item = mediaOverlays.first(where: { $0["overlay_id"]?.stringValue == id }),
              let start = item["start"]?.doubleValue, let end = item["end"]?.doubleValue,
              time > start + 0.1, time < end - 0.1 else { return false }
        var left = item, right = item
        left["end"] = .number((time * 1000).rounded() / 1000)
        right["start"] = .number((time * 1000).rounded() / 1000)
        right["overlay_id"] = .string("ov_\(UUID().uuidString.prefix(10).lowercased())")
        if item["kind"]?.stringValue == "video" {
            let sourceStart = item["source_start"]?.doubleValue ?? 0
            right["source_start"] = .number(((sourceStart + time - start) * 1000).rounded() / 1000)
        }
        applyDraftChange { object in
            let items = object["media_overlays"]?.arrayValue?.compactMap(\.objectValue) ?? []
            object["media_overlays"] = .array(items.flatMap { entry -> [JSONValue] in
                entry["overlay_id"]?.stringValue == id ? [.object(left), .object(right)] : [.object(entry)]
            })
        }
        return true
    }

    /// Rows of overlays: the highest layer in use (0 = one row).
    var topOverlayLayer: Int { overlaysOnScreen.map(\.style.layer).max() ?? 0 }

    /// "Front": this overlay goes above every other one (a new top row when needed).
    @discardableResult
    func bringOverlayToFront(id: String) -> Task<Void, Never>? {
        let others = overlaysOnScreen.filter { $0.id != id }
        let top = others.map(\.style.layer).max() ?? -1
        let mine = overlayStyle(id).layer
        guard mine <= top || others.contains(where: { $0.style.layer == mine }) else { return nil }
        return setOverlayStyle(id: id) { $0.layer = top + 1 }
    }

    /// "Photo 1", "Video 2" (counted per kind, in the order they were added).
    func overlayTitle(_ id: String) -> String {
        let all = overlaysOnScreen
        guard let item = all.first(where: { $0.id == id }) else { return "Overlay" }
        let sameKind = all.filter { $0.kind == item.kind }
        let index = (sameKind.firstIndex { $0.id == id } ?? 0) + 1
        return "\(item.kind == .video ? "Video" : "Photo") \(index)"
    }

    // MARK: Auto Zoom: what the creator moved or deleted is kept when the cut is made again

    /// Deletes a zoom; an AI one leaves a note so a new cut does not put it back.
    func deleteZoomKeyframe(id: String) async {
        guard let item = zoomKeyframes.first(where: { $0["keyframe_id"]?.stringValue == id }) else { return }
        let origin = item["origin"]?.stringValue ?? "ai"
        let source = item["source_asset_id"] ?? .string("")
        let time = item["source_time"] ?? .number(0)
        await applyDraftChange { object in
            let items = object["zoom_keyframes"]?.arrayValue?.compactMap(\.objectValue) ?? []
            object["zoom_keyframes"] = .array(items.filter { $0["keyframe_id"]?.stringValue != id }.map(JSONValue.object))
            if origin == "ai" {
                var removed = object["zoom_removed"]?.arrayValue ?? []
                removed.append(.object(["source_asset_id": source, "source_time": time]))
                object["zoom_removed"] = .array(removed)
            }
        }.value
    }
}

// MARK: - CutSell sounds (Audio panel): manifest with signed links, files kept on this iPhone

struct SfxManifest: Decodable {
    struct Sound: Decodable, Equatable {
        let id: String
        let name: String
        let file: String
        let duration_sec: Double?
        let url: String?
    }
    let version: String
    let sounds: [Sound]
}

@MainActor
final class SfxLibrary: ObservableObject {
    static let shared = SfxLibrary()

    /// The five buttons of the Audio panel (Figma 1119:1973). "typing_long" is the 9 s typing used
    /// when Typing is stretched past 3 s.
    nonisolated static let buttons: [(id: String, name: String)] = [
        ("tap", "Tap"), ("good", "Good"), ("wrong", "Wrong"), ("glass", "Glass"), ("typing", "Typing"),
    ]
    nonisolated static func name(of id: String) -> String {
        id == "typing_long" ? "Typing" : (buttons.first { $0.id == id }?.name ?? "Sound")
    }

    /// Lengths known before the manifest arrives (the files made from the CC0 packs).
    nonisolated static let builtInDurations: [String: Double] = [
        "tap": 0.612, "good": 0.539, "wrong": 0.139, "glass": 0.692, "typing": 3.0, "typing_long": 9.0,
    ]

    @Published private(set) var ready: Set<String> = []
    @Published private(set) var loading = false
    @Published private(set) var failed = false
    private var durations: [String: Double] = SfxLibrary.builtInDurations

    nonisolated static func duration(of id: String) -> Double? { builtInDurations[id] }

    private static var folder: URL {
        let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell/Sounds/v1", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    nonisolated static func localURL(_ id: String) -> URL? {
        let url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell/Sounds/v1/\(id).m4a")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Downloads every sound the first time the Audio panel opens; later opens use the copies.
    func prepare() async {
        let ids = Self.buttons.map(\.id) + ["typing_long"]
        ready = Set(ids.filter { Self.localURL($0) != nil })
        guard ready.count < ids.count, !loading else { return }
        loading = true
        failed = false
        defer { loading = false }
        do {
            let manifest: SfxManifest = try await APIClient.shared.request("/v1/sfx/manifest")
            for sound in manifest.sounds {
                if let seconds = sound.duration_sec { durations[sound.id] = seconds }
                guard Self.localURL(sound.id) == nil, let raw = sound.url, let url = URL(string: raw) else { continue }
                let (file, response) = try await URLSession.shared.download(from: url)
                guard (response as? HTTPURLResponse)?.statusCode == 200 else { continue }
                let target = Self.folder.appendingPathComponent("\(sound.id).m4a")
                try? FileManager.default.removeItem(at: target)
                try FileManager.default.moveItem(at: file, to: target)
                ready.insert(sound.id)
            }
        } catch {
            failed = true
            CutSellDiagnostics.log("sfx_manifest_failed", ["reason": String(describing: type(of: error))])
        }
    }

    private var player: AVAudioPlayer?

    /// Plays the sound once (the button's preview).
    func play(_ id: String) {
        guard let url = Self.localURL(id) else { return }
        EditorAudio.preparePlayback()
        player = try? AVAudioPlayer(contentsOf: url)
        player?.play()
    }
}
