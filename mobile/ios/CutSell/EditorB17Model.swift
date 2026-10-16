import AVFoundation
import Foundation
import MediaToolbox

// MARK: - Build 17: keyframes, per-clip volume / mute, Clean up voice
//
// Draft fields (the server's names, cutsell_worker/keyframes.py · serde.py):
//   keyframes[]      keyframe_id · target video|overlay · source_asset_id + source_time (video) ·
//                    overlay_id + time (overlay) · scale · x · y · rotation · origin ai|user · kind · group
//   audio_clips[]    muted
//   selected[]       clean_voice
//   draft            master_volume (0–2) · master_muted
//
// Interpolation is the server's: inside one video clip (or one overlay) keyframes are sorted by time,
// the value is held before the first and after the last, and between two keyframes it moves with a
// smoothstep ease. Video pans are limited to what the zoom allows (|x| <= (scale - 1) / 2).

/// A framing: zoom, position (fractions of the frame) and a clockwise turn in degrees.
struct TransformValue: Equatable {
    var scale: Double = 1
    var x: Double = 0
    var y: Double = 0
    var rotation: Double = 0

    static let identity = TransformValue()
    var isIdentity: Bool { abs(scale - 1) < 0.0005 && abs(x) < 0.0005 && abs(y) < 0.0005 && abs(rotation) < 0.01 }

    /// The video can't be zoomed out below 100 % nor panned past its edges at that zoom.
    var clampedForVideo: TransformValue {
        var value = self
        value.scale = min(5, max(1, scale))
        let room = (value.scale - 1) / 2
        value.x = min(room, max(-room, x))
        value.y = min(room, max(-room, y))
        value.rotation = min(720, max(-720, rotation))
        return value
    }

    var clampedForOverlay: TransformValue {
        var value = self
        value.scale = min(5, max(0.2, scale))
        value.x = min(1, max(-1, x))
        value.y = min(1, max(-1, y))
        value.rotation = min(720, max(-720, rotation))
        return value
    }
}

struct TransformKeyframe: Identifiable, Equatable {
    enum Target: String { case video, overlay }

    let id: String
    var target: Target
    var sourceAssetID = ""
    var sourceTime = 0.0
    var overlayID = ""
    var time = 0.0
    var value = TransformValue.identity
    var origin = "user"
    var kind = "key"
    var group = ""

    var isAI: Bool { origin == "ai" }

    init(id: String, target: Target, value: TransformValue) {
        self.id = id
        self.target = target
        self.value = value
    }

    init?(_ object: [String: JSONValue]) {
        guard let id = object["keyframe_id"]?.stringValue,
              let target = Target(rawValue: object["target"]?.stringValue ?? "video") else { return nil }
        self.id = id
        self.target = target
        value = TransformValue(scale: object["scale"]?.doubleValue ?? 1, x: object["x"]?.doubleValue ?? 0,
                               y: object["y"]?.doubleValue ?? 0, rotation: object["rotation"]?.doubleValue ?? 0)
        origin = object["origin"]?.stringValue == "ai" ? "ai" : "user"
        kind = object["kind"]?.stringValue ?? "key"
        group = object["group"]?.stringValue ?? ""
        if target == .video {
            sourceAssetID = object["source_asset_id"]?.stringValue ?? ""
            sourceTime = object["source_time"]?.doubleValue ?? 0
        } else {
            guard let overlay = object["overlay_id"]?.stringValue else { return nil }
            overlayID = overlay
            time = object["time"]?.doubleValue ?? 0
        }
    }

    var json: [String: JSONValue] {
        func r(_ value: Double, _ places: Double) -> JSONValue { .number((value * places).rounded() / places) }
        let v = target == .video ? value.clampedForVideo : value.clampedForOverlay
        var object: [String: JSONValue] = [
            "keyframe_id": .string(id), "target": .string(target.rawValue),
            "scale": r(v.scale, 10_000), "x": r(v.x, 10_000), "y": r(v.y, 10_000), "rotation": r(v.rotation, 100),
            "origin": .string(origin), "kind": .string(kind),
        ]
        if !group.isEmpty { object["group"] = .string(group) }
        if target == .video {
            object["source_asset_id"] = .string(sourceAssetID)
            object["source_time"] = r(sourceTime, 1000)
        } else {
            object["overlay_id"] = .string(overlayID)
            object["time"] = r(max(0, time), 1000)
        }
        return object
    }
}

enum KeyframeMath {
    static func ease(_ u: Double) -> Double {
        let v = max(0, min(1, u))
        return v * v * (3 - 2 * v)
    }

    /// The server's keyframes.value_at, for all four values at once.
    static func value(_ points: [(time: Double, value: TransformValue)], at time: Double) -> TransformValue {
        guard let first = points.first, let last = points.last else { return .identity }
        if time <= first.time { return first.value }
        if time >= last.time { return last.value }
        for (a, b) in zip(points, points.dropFirst()) where a.time <= time && time <= b.time {
            let span = b.time - a.time
            let e = span <= 1e-9 ? 0 : ease((time - a.time) / span)
            return TransformValue(scale: a.value.scale + (b.value.scale - a.value.scale) * e,
                                  x: a.value.x + (b.value.x - a.value.x) * e,
                                  y: a.value.y + (b.value.y - a.value.y) * e,
                                  rotation: a.value.rotation + (b.value.rotation - a.value.rotation) * e)
        }
        return last.value
    }

    /// One Auto Zoom AI push-in (the cut engine's `zoom_keyframes` item) as its four keyframes: the
    /// same curve the export always drew (keyframes.zoom_to_keyframes on the server).
    static func keyframes(fromZoom zoom: [String: JSONValue]) -> [TransformKeyframe] {
        guard let t = zoom["source_time"]?.doubleValue else { return [] }
        let d = zoom["duration"]?.doubleValue ?? 1.5
        let s = zoom["scale"]?.doubleValue ?? 1.08
        let base = zoom["keyframe_id"]?.stringValue ?? "z\(Int(t * 1000))"
        let r = min(0.35, d / 3)
        let steps: [(Double, Double)] = [(t, 1), (t + r, s), (t + d - r, s), (t + d, 1)]
        return steps.enumerated().map { index, step in
            var key = TransformKeyframe(id: "\(base)-\(index + 1)", target: .video, value: TransformValue(scale: step.1))
            key.sourceAssetID = zoom["source_asset_id"]?.stringValue ?? ""
            key.sourceTime = (step.0 * 1000).rounded() / 1000
            key.origin = zoom["origin"]?.stringValue == "user" ? "user" : "ai"
            key.kind = zoom["kind"]?.stringValue ?? "key"
            key.group = base
            return key
        }
    }
}

/// What a keyframe belongs to: a video clip or an overlay.
enum KeyframeTarget: Equatable {
    case video(String)
    case overlay(String)
}

/// A ◇ on the timeline.
struct KeyframeMark: Identifiable, Equatable {
    let id: String
    /// Where it is on the edited timeline.
    let time: Double
    let target: KeyframeTarget
    let isAI: Bool
}

/// What the Volume panel works on (Figma 1119:990: "Selected clip" or "Whole video").
enum VolumeTarget: Equatable {
    case whole
    case video(String)
    case audio(String)
    case overlay(String)
}

extension DraftEditorViewModel {
    // MARK: Keyframes

    /// The draft carries the keyframes list (it then replaces the legacy `zoom_keyframes`).
    var keyframesSet: Bool { displayDraft?["keyframes"]?.arrayValue != nil }

    var transformKeyframes: [TransformKeyframe] {
        displayDraft?["keyframes"]?.arrayValue?.compactMap(\.objectValue).compactMap(TransformKeyframe.init) ?? []
    }

    /// The keyframes that are drawn: the creator's always, the AI's only with Auto Zoom AI on.
    var activeKeyframes: [TransformKeyframe] {
        let auto = autoZoomEnabled
        return transformKeyframes.filter { !$0.isAI || auto }
    }

    /// A cut made by the engine has its Auto Zoom AI push-ins as `zoom_keyframes`: they become
    /// normal keyframes once (one save), so the creator can move, edit or delete them like any other.
    func adoptLegacyZoomKeyframes() {
        guard !keyframesSet else { return }
        let zooms = displayDraft?["zoom_keyframes"]?.arrayValue?.compactMap(\.objectValue) ?? []
        guard !zooms.isEmpty else { return }
        let removed = displayDraft?["zoom_removed"]?.arrayValue?.compactMap(\.objectValue) ?? []
        let live = zooms.filter { zoom in
            let source = zoom["source_asset_id"]?.stringValue ?? ""
            let time = zoom["source_time"]?.doubleValue ?? -99
            return !removed.contains { ($0["source_asset_id"]?.stringValue ?? "") == source
                && abs(($0["source_time"]?.doubleValue ?? -1) - time) <= 0.25 }
        }
        let keys = live.flatMap(KeyframeMath.keyframes(fromZoom:))
        applyDraftChange { object in object["keyframes"] = .array(keys.map { .object($0.json) }) }
    }

    private static func setKeyframes(_ object: inout [String: JSONValue], _ transform: ([TransformKeyframe]) -> [TransformKeyframe]) {
        let items = object["keyframes"]?.arrayValue?.compactMap(\.objectValue).compactMap(TransformKeyframe.init) ?? []
        object["keyframes"] = .array(transform(items).map { .object($0.json) })
    }

    private func clip(_ clipID: String) -> [String: JSONValue]? {
        selectedClips.first { $0["clip_id"]?.stringValue == clipID }
    }

    /// The keyframes of one video clip (its source window), sorted.
    func videoKeyframes(clipID: String, active: Bool = true) -> [TransformKeyframe] {
        guard let clip = clip(clipID) else { return [] }
        let source = clip["source_asset_id"]?.stringValue ?? ""
        let start = clip["start"]?.doubleValue ?? 0
        let end = clip["end"]?.doubleValue ?? start
        return (active ? activeKeyframes : transformKeyframes)
            .filter { $0.target == .video && ($0.sourceAssetID.isEmpty || $0.sourceAssetID == source)
                && $0.sourceTime >= start - 1e-6 && $0.sourceTime <= end + 1e-6 }
            .sorted { $0.sourceTime < $1.sourceTime }
    }

    func overlayKeyframes(overlayID: String) -> [TransformKeyframe] {
        transformKeyframes.filter { $0.target == .overlay && $0.overlayID == overlayID }.sorted { $0.time < $1.time }
    }

    /// The clip under the cursor at `time`.
    func clipWindow(at time: Double) -> EditorTracks.ClipWindow? {
        let windows = EditorTracks.clipWindows(self)
        return windows.first { time >= $0.start && time < $0.end } ?? (time >= (windows.last?.end ?? 0) - 0.01 ? windows.last : nil)
    }

    /// The video's framing at timeline `time` (the export's perspective warp).
    func videoTransform(at time: Double) -> TransformValue {
        guard let window = clipWindow(at: time) else { return .identity }
        let keys = videoKeyframes(clipID: window.clipID)
        guard !keys.isEmpty else { return .identity }
        let source = window.source(at: min(time, window.end))
        return KeyframeMath.value(keys.map { ($0.sourceTime, $0.value) }, at: source).clampedForVideo
    }

    /// An overlay's framing at timeline `time` (on top of its own place, size and turn).
    func overlayTransform(id: String, at time: Double) -> TransformValue {
        guard let overlay = overlaysOnScreen.first(where: { $0.id == id }) else { return .identity }
        let keys = overlayKeyframes(overlayID: id)
        guard !keys.isEmpty else { return .identity }
        return KeyframeMath.value(keys.map { ($0.time, $0.value) }, at: time - overlay.start).clampedForOverlay
    }

    /// Every video ◇ on the edited timeline (AI ones dimmed while Auto Zoom AI is off).
    var videoKeyframeMarks: [KeyframeMark] {
        var marks: [KeyframeMark] = []
        for window in EditorTracks.clipWindows(self) {
            for key in videoKeyframes(clipID: window.clipID, active: false) {
                marks.append(KeyframeMark(id: key.id, time: window.time(ofSource: key.sourceTime),
                                          target: .video(window.clipID), isAI: key.isAI))
            }
        }
        return marks.sorted { $0.time < $1.time }
    }

    func overlayKeyframeMarks(overlayID: String) -> [KeyframeMark] {
        guard let overlay = overlaysOnScreen.first(where: { $0.id == overlayID }) else { return [] }
        return overlayKeyframes(overlayID: overlayID).map {
            KeyframeMark(id: $0.id, time: overlay.start + $0.time, target: .overlay(overlayID), isAI: false)
        }
    }

    /// The ◇ under the cursor for this target (filled button: tapping it deletes it).
    func keyframe(at time: Double, target: KeyframeTarget, tolerance: Double = 0.05) -> KeyframeMark? {
        let marks: [KeyframeMark]
        switch target {
        case .video(let clipID): marks = videoKeyframeMarks.filter { $0.target == .video(clipID) }
        case .overlay(let id): marks = overlayKeyframeMarks(overlayID: id)
        }
        return marks.filter { abs($0.time - time) <= tolerance }.min { abs($0.time - time) < abs($1.time - time) }
    }

    /// The current framing of a target at `time` (where a new keyframe starts from).
    func transform(of target: KeyframeTarget, at time: Double) -> TransformValue {
        switch target {
        case .video: return videoTransform(at: time)
        case .overlay(let id): return overlayTransform(id: id, at: time)
        }
    }

    /// ◇: a keyframe with this framing at `time`. Nil when the cursor is not on the target.
    @discardableResult
    func addKeyframe(target: KeyframeTarget, at time: Double, value: TransformValue) -> String? {
        let id = "kf_" + UUID().uuidString.prefix(8).lowercased()
        var key: TransformKeyframe
        switch target {
        case .video(let clipID):
            guard let window = EditorTracks.clipWindows(self).first(where: { $0.clipID == clipID }),
                  time >= window.start - 0.01, time <= window.end + 0.01 else { return nil }
            key = TransformKeyframe(id: id, target: .video, value: value.clampedForVideo)
            key.sourceAssetID = window.sourceID ?? ""
            let into = min(max(0, time - window.start), max(0, window.end - window.start - 0.01))
            key.sourceTime = (window.source(at: window.start + into) * 1000).rounded() / 1000
        case .overlay(let overlayID):
            guard let overlay = overlaysOnScreen.first(where: { $0.id == overlayID }),
                  time >= overlay.start - 0.01, time <= overlay.end + 0.01 else { return nil }
            key = TransformKeyframe(id: id, target: .overlay, value: value.clampedForOverlay)
            key.overlayID = overlayID
            key.time = min(max(0, time - overlay.start), overlay.end - overlay.start)
        }
        let made = key
        // The first keyframe also keeps Auto Zoom AI's state as it is (AI ones stay as they were).
        applyDraftChange { object in Self.setKeyframes(&object) { $0 + [made] } }
        return id
    }

    func updateKeyframe(id: String, value: TransformValue) {
        applyDraftChange { object in
            Self.setKeyframes(&object) { keys in keys.map { key in
                guard key.id == id else { return key }
                var changed = key
                changed.value = value
                return changed
            } }
        }
    }

    /// Drag a ◇ to another moment (inside its own clip / overlay).
    func moveKeyframe(id: String, to time: Double) {
        guard let key = transformKeyframes.first(where: { $0.id == id }) else { return }
        var moved = key
        switch key.target {
        case .video:
            guard let mark = videoKeyframeMarks.first(where: { $0.id == id }), case .video(let clipID) = mark.target,
                  let window = EditorTracks.clipWindows(self).first(where: { $0.clipID == clipID }) else { return }
            let at = min(max(time, window.start), window.end - 0.01)
            moved.sourceTime = (window.source(at: at) * 1000).rounded() / 1000
        case .overlay:
            guard let overlay = overlaysOnScreen.first(where: { $0.id == key.overlayID }) else { return }
            moved.time = min(max(0, time - overlay.start), overlay.end - overlay.start)
        }
        let result = moved
        applyDraftChange { object in Self.setKeyframes(&object) { $0.map { $0.id == id ? result : $0 } } }
    }

    func deleteKeyframe(id: String) {
        applyDraftChange { object in Self.setKeyframes(&object) { $0.filter { $0.id != id } } }
    }

    // MARK: Volume and Mute per clip, or for the whole video

    var masterVolume: Double { displayDraft?["master_volume"]?.doubleValue ?? 1 }
    var masterMuted: Bool { displayDraft?["master_muted"]?.boolValue ?? false }

    func volumeTarget(for selection: EditorSelection?) -> VolumeTarget {
        switch selection {
        case .video(let id): return .video(id)
        case .voiceOver(let id), .sound(let id): return .audio(id)
        case .overlay(let id, isText: false):
            return overlaysOnScreen.first { $0.id == id }?.kind == .video ? .overlay(id) : .whole
        default: return .whole
        }
    }

    func volume(of target: VolumeTarget) -> Double {
        switch target {
        case .whole: return masterVolume
        case .video(let id): return clip(id)?["audio_volume"]?.doubleValue ?? 1
        case .audio(let id): return audioClips.first { $0.id == id }?.volume ?? 1
        case .overlay(let id): return overlayStyle(id).volume
        }
    }

    func isMuted(_ target: VolumeTarget) -> Bool {
        switch target {
        case .whole: return masterMuted
        case .video(let id):
            guard let clip = clip(id) else { return false }
            return clip["audio_muted"]?.boolValue == true || (clip["audio_volume"]?.doubleValue ?? 1) <= 0.001
        case .audio(let id): return audioClips.first { $0.id == id }.map { $0.muted || $0.volume <= 0.001 } ?? false
        case .overlay(let id): return overlayStyle(id).muted
        }
    }

    /// The level (0...2) for the target; a muted target keeps its level for when it is unmuted.
    func setVolume(_ target: VolumeTarget, _ volume: Double) {
        let value = (min(2, max(0, volume)) * 100).rounded() / 100
        switch target {
        case .whole:
            applyDraftChange { $0["master_volume"] = abs(value - 1) < 0.005 ? nil : .number(value) }
        case .video(let id):
            applyDraftChange { object in
                Self.mapClips(&object) { clips in clips.map { clip in
                    guard clip["clip_id"]?.stringValue == id else { return clip }
                    var clip = clip
                    clip["audio_volume"] = .number(value)
                    if value > 0.001 { clip["audio_muted"] = .bool(false) }
                    return clip
                } }
            }
        case .audio(let id):
            updateAudioClip(id: id) { $0.volume = value; if value > 0.001 { $0.muted = false } }
        case .overlay(let id):
            setOverlayStyle(id: id) { $0.volume = value; if value > 0.001 { $0.muted = false } }
        }
    }

    func setMuted(_ target: VolumeTarget, _ muted: Bool) {
        switch target {
        case .whole:
            applyDraftChange { $0["master_muted"] = muted ? .bool(true) : nil }
        case .video(let id):
            applyDraftChange { object in
                Self.mapClips(&object) { clips in clips.map { clip in
                    guard clip["clip_id"]?.stringValue == id else { return clip }
                    var clip = clip
                    clip["audio_muted"] = .bool(muted)
                    // Unmuting a clip that was silenced at 0 % brings it back at 100 %.
                    if !muted, (clip["audio_volume"]?.doubleValue ?? 1) <= 0.001 { clip["audio_volume"] = .number(1) }
                    return clip
                } }
            }
        case .audio(let id):
            updateAudioClip(id: id) { $0.muted = muted; if !muted, $0.volume <= 0.001 { $0.volume = 1 } }
        case .overlay(let id):
            setOverlayStyle(id: id) { $0.muted = muted; if !muted, $0.volume <= 0.001 { $0.volume = 1 } }
        }
    }

    /// A clip whose sound is silent (shows a mute icon on the timeline).
    func clipIsSilent(_ clipID: String) -> Bool { isMuted(.video(clipID)) }

    // MARK: Clean up voice

    /// Which cleaned copies a target plays, and whether its Clean up is on.
    func cleanKeys(for target: VolumeTarget) -> [CleanAudioKey] {
        switch target {
        case .whole:
            return Array(Set(selectedClips.compactMap { $0["source_asset_id"]?.stringValue })).sorted().map { .source($0) }
        case .video(let id):
            return clip(id)?["source_asset_id"]?.stringValue.map { [.source($0)] } ?? []
        case .audio(let id):
            guard let item = audioClips.first(where: { $0.id == id }), item.kind != .sfx, let asset = item.assetID else { return [] }
            return [.asset(asset)]
        case .overlay: return []
        }
    }

    func cleanUpOn(_ target: VolumeTarget) -> Bool {
        switch target {
        case .whole: return videoCleanVoice
        case .video(let id): return videoCleanVoice || clip(id)?["clean_voice"]?.boolValue == true
        case .audio(let id): return audioClips.first { $0.id == id }?.clean ?? false
        case .overlay: return false
        }
    }

    /// Clean up On / Off. On asks the server for the cleaned copy (the preview plays it once ready).
    func setCleanUp(_ target: VolumeTarget, _ on: Bool) {
        switch target {
        case .whole: setVideoCleanVoice(on)
        case .video(let id):
            applyDraftChange { object in
                Self.mapClips(&object) { clips in clips.map { clip in
                    guard clip["clip_id"]?.stringValue == id else { return clip }
                    var clip = clip
                    clip["clean_voice"] = on ? .bool(true) : nil
                    return clip
                } }
            }
        case .audio(let id): updateAudioClip(id: id) { $0.clean = on }
        case .overlay: return
        }
        if on { for key in cleanKeys(for: target) { CleanAudioCenter.shared.request(key, project: project) } }
    }

    /// The editor opened: fetch (or ask for) every cleaned copy the draft already uses.
    func resumeCleanAudio() {
        var keys: [CleanAudioKey] = []
        if videoCleanVoice { keys += cleanKeys(for: .whole) }
        for clip in selectedClips where clip["clean_voice"]?.boolValue == true {
            if let source = clip["source_asset_id"]?.stringValue { keys.append(.source(source)) }
        }
        for item in audioClips where item.clean && item.kind != .sfx {
            if let asset = item.assetID { keys.append(.asset(asset)) }
        }
        for key in Set(keys) { CleanAudioCenter.shared.request(key, project: project) }
    }

    /// Whether a video clip plays the cleaned sound of its source.
    func clipPlaysCleaned(_ clip: [String: JSONValue]) -> Bool {
        videoCleanVoice || clip["clean_voice"]?.boolValue == true
    }
}

// MARK: - Cleaned copies (made on the server, played here)

enum CleanAudioKey: Hashable {
    case source(String)
    case asset(String)

    var target: String { if case .source = self { return "source" } else { return "asset" } }
    var id: String { switch self { case .source(let id), .asset(let id): return id } }
    var fileName: String {
        let safe = String(id.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) || $0 == "_" || $0 == "-" })
        return "\(target)-\(safe).m4a"
    }
}

/// Asks the server for cleaned copies, follows their progress ("Cleaning… 40%") and keeps the
/// finished files on this iPhone for the preview. The export uses the same stored copy.
@MainActor
final class CleanAudioCenter: ObservableObject {
    static let shared = CleanAudioCenter()

    enum State: Equatable {
        case cleaning(Int)
        case ready(URL)
        case failed
    }

    @Published private(set) var states: [String: State] = [:]
    private var running: Set<String> = []

    private struct Answer: Decodable {
        let status: String
        let progress: Int?
        let url: String?
    }

    private struct Body: Encodable {
        let user_id: String
        let target: String
        let id: String
    }

    private func slot(_ key: CleanAudioKey, _ project: Project) -> String { "\(project.projectID)|\(key.target)|\(key.id)" }

    private static func folder(_ projectID: String) -> URL {
        let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell/Clean/\(projectID)", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    func state(_ key: CleanAudioKey, project: Project) -> State? {
        if let state = states[slot(key, project)] { return state }
        let file = Self.folder(project.projectID).appendingPathComponent(key.fileName)
        return FileManager.default.fileExists(atPath: file.path) ? .ready(file) : nil
    }

    /// The cleaned file on this iPhone, when it is ready.
    func localURL(_ key: CleanAudioKey, project: Project) -> URL? {
        if case .ready(let url) = state(key, project: project) { return url }
        return nil
    }

    func request(_ key: CleanAudioKey, project: Project) {
        let name = slot(key, project)
        if localURL(key, project: project) != nil || running.contains(name) { return }
        running.insert(name)
        states[name] = .cleaning(0)
        Task { [weak self] in
            if let userID = await APIClient.shared.session?.userID {
                await self?.follow(key, project: project, userID: userID)
            } else {
                self?.states[name] = .failed
            }
            self?.running.remove(name)
        }
    }

    private func follow(_ key: CleanAudioKey, project: Project, userID: String) async {
        let name = slot(key, project)
        let path = "/v1/projects/\(project.projectID)/clean-audio"
        do {
            var answer: Answer = try await APIClient.shared.request(
                path, method: "POST", body: Body(user_id: userID, target: key.target, id: key.id))
            var waited = 0.0
            while answer.status == "cleaning" && waited < 900 {
                states[name] = .cleaning(max(1, min(99, answer.progress ?? 0)))
                try await Task.sleep(nanoseconds: 1_500_000_000)
                waited += 1.5
                answer = try await APIClient.shared.request(path, query: [
                    URLQueryItem(name: "user_id", value: userID), URLQueryItem(name: "target", value: key.target),
                    URLQueryItem(name: "id", value: key.id),
                ])
            }
            guard answer.status == "ready", let raw = answer.url, let url = URL(string: raw) else {
                states[name] = .failed
                CutSellDiagnostics.log("clean_audio_failed", ["status": answer.status])
                return
            }
            let (file, response) = try await URLSession.shared.download(from: url)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { states[name] = .failed; return }
            let target = Self.folder(project.projectID).appendingPathComponent(key.fileName)
            try? FileManager.default.removeItem(at: target)
            try FileManager.default.moveItem(at: file, to: target)
            states[name] = .ready(target)
        } catch {
            states[name] = .failed
            CutSellDiagnostics.log("clean_audio_failed", ["reason": String(describing: type(of: error))])
        }
    }

    /// "Cleaning… 40%" for a target (the slowest of its copies), nil when nothing is cleaning.
    func progress(_ keys: [CleanAudioKey], project: Project) -> Int? {
        let values = keys.compactMap { key -> Int? in
            if case .cleaning(let p) = state(key, project: project) { return p }
            return nil
        }
        return values.min()
    }

    func failed(_ keys: [CleanAudioKey], project: Project) -> Bool {
        keys.contains { state($0, project: project) == .failed }
    }

    func retry(_ keys: [CleanAudioKey], project: Project) {
        for key in keys where state(key, project: project) == .failed {
            states[slot(key, project)] = nil
            request(key, project: project)
        }
    }
}

// MARK: - Preview gain above 100 % (AVAudioMix stops at 1.0)

/// Multiplies a preview track's sound by a fixed gain (the export's volume filter can go to 2.0).
enum AudioGainTap {
    final class Box {
        let gain: Float
        var isFloat = true
        init(gain: Float) { self.gain = gain }
    }

    static func make(gain: Double) -> MTAudioProcessingTap? {
        let box = Box(gain: Float(gain))
        var callbacks = MTAudioProcessingTapCallbacks(
            version: kMTAudioProcessingTapCallbacksVersion_0,
            clientInfo: UnsafeMutableRawPointer(Unmanaged.passRetained(box).toOpaque()),
            init: { _, clientInfo, storageOut in storageOut.pointee = clientInfo },
            finalize: { tap in
                Unmanaged<Box>.fromOpaque(MTAudioProcessingTapGetStorage(tap)).release()
            },
            prepare: { tap, _, format in
                let box = Unmanaged<Box>.fromOpaque(MTAudioProcessingTapGetStorage(tap)).takeUnretainedValue()
                box.isFloat = format.pointee.mFormatFlags & kAudioFormatFlagIsFloat != 0
                    && format.pointee.mBitsPerChannel == 32
            },
            unprepare: nil,
            process: { tap, frames, _, buffers, framesOut, flagsOut in
                guard MTAudioProcessingTapGetSourceAudio(tap, frames, buffers, flagsOut, nil, framesOut) == noErr else { return }
                let box = Unmanaged<Box>.fromOpaque(MTAudioProcessingTapGetStorage(tap)).takeUnretainedValue()
                guard box.isFloat else { return }
                for buffer in UnsafeMutableAudioBufferListPointer(buffers) {
                    guard let data = buffer.mData else { continue }
                    let count = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
                    let samples = data.assumingMemoryBound(to: Float.self)
                    for index in 0..<count { samples[index] = max(-1, min(1, samples[index] * box.gain)) }
                }
            }
        )
        var tap: MTAudioProcessingTap?
        let status = MTAudioProcessingTapCreate(kCFAllocatorDefault, &callbacks, kMTAudioProcessingTapCreationFlag_PostEffects, &tap)
        return status == noErr ? tap : nil
    }

    /// Volume `level` (0...2) on a mix input: up to 1 by the mix, the rest by a tap.
    static func apply(_ level: Double, to parameters: AVMutableAudioMixInputParameters) {
        if level > 1.001, let tap = make(gain: level) {
            parameters.audioTapProcessor = tap
            parameters.setVolume(1, at: .zero)
        } else {
            parameters.setVolume(Float(max(0, min(1, level))), at: .zero)
        }
    }
}
