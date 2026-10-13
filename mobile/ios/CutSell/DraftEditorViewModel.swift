import Foundation

/// Carries one result out of a queued edit.
@MainActor
final class ResultBox {
    var value = false
}

@MainActor
final class DraftEditorViewModel: ObservableObject {
    @Published var snapshot: DraftSnapshot?
    @Published var isLoading = false
    /// The one playhead of the editor: the video player's current time (the timeline follows it).
    @Published var playheadTime: Double = 0
    @Published var isSaving = false
    @Published var exportJob: JobStatus?
    @Published private(set) var timelineAssetLibrary: TimelineAssetLibrary?
    @Published var errorMessage: String?

    // D-282A Overlay/Voice-over composition (Main Video stays on the
    // pre-existing `draft`/`draft-edits` system above -- see
    // TimelineComposition.swift's own module docstring).
    @Published private(set) var timelineComposition: TimelineComposition?
    @Published private(set) var isSavingTimeline = false
    private var lastCompositionBeforeMutation: TimelineComposition?
    var canUndoTimelineMutation: Bool { lastCompositionBeforeMutation != nil }

    /// The FIRST real `PRIMARY_SOURCE` asset already registered for this
    /// project, if any -- D-282A's own `base_edit_asset_id` authority.
    /// Honestly `nil` (never invented) when no such asset exists yet, in
    /// which case Overlay/Voice-over placement is correctly disabled --
    /// this gate does not build a new PRIMARY_SOURCE ingestion flow.
    var baseEditAssetID: String? { timelineAssetLibrary?.readyPrimarySources.first?.assetID }

    let project: Project
    private let api = APIClient.shared
    private let session: CutSellSession

    init(project: Project, session: CutSellSession) {
        self.project = project
        self.session = session
        if let saved = UserDefaults.standard.object(forKey: "cutsell.voVolume.\(project.projectID)") as? Double {
            voiceOverVolume = saved
        }
    }

    /// The change the creator just made, shown at once while the server confirms it (Editor v4:
    /// every tap reacts immediately). `nil` when nothing is waiting; on a server error the screen
    /// falls back to what the server really has.
    @Published private(set) var optimisticDraft: JSONValue?
    /// What the editor shows: the predicted draft while an edit is on its way, else the server's.
    var displayDraft: JSONValue? { optimisticDraft ?? snapshot?.draft }

    var selectedClips: [[String: JSONValue]] {
        displayDraft?["selected"]?.arrayValue?.compactMap(\.objectValue) ?? []
    }

    var alternateClips: [[String: JSONValue]] {
        displayDraft?["alternates"]?.arrayValue?.compactMap(\.objectValue) ?? []
    }

    var textOverlays: [[String: JSONValue]] {
        displayDraft?["text_overlays"]?.arrayValue?.compactMap(\.objectValue) ?? []
    }

    var mediaOverlays: [[String: JSONValue]] {
        displayDraft?["media_overlays"]?.arrayValue?.compactMap(\.objectValue) ?? []
    }

    /// Mirror (flip left-right) and rotation (0, 90, 180, 270) of the whole video.
    var isMirrored: Bool { displayDraft?["mirror"]?.boolValue ?? false }

    /// Mirror of the clip playing at `time` (its own Mirror, or the older whole-video one).
    func isMirrored(at time: Double) -> Bool {
        if isMirrored { return true }
        let windows = EditorTracks.clipWindows(self)
        guard let window = windows.first(where: { time >= $0.start && time < $0.end }) ?? windows.last,
              let clip = selectedClips.first(where: { $0["clip_id"]?.stringValue == window.clipID }) else { return false }
        return clip["mirror"]?.boolValue ?? false
    }

    func clipIsMirrored(_ clipID: String) -> Bool {
        isMirrored || (selectedClips.first { $0["clip_id"]?.stringValue == clipID }?["mirror"]?.boolValue ?? false)
    }

    /// Editor v4 Mirror: flips the selected clip, or every clip when none is selected; tapping again
    /// takes it off. Saved per clip (`mirror` on the clip); the export flips only those clips' picture.
    @discardableResult
    func toggleMirror(clipID: String?) async -> Bool {
        let clips = selectedClips
        let targets = clipID.map { [$0] } ?? clips.compactMap { $0["clip_id"]?.stringValue }
        let turnOn = !targets.allSatisfy { clipIsMirrored($0) }
        let change: (inout [String: JSONValue]) -> Void = { object in
            let wasWhole = object["mirror"]?.boolValue ?? false
            object["mirror"] = nil   // the whole-video flag becomes per clip
            Self.mapClips(&object) { clips in clips.map { clip in
                var clip = clip
                let id = clip["clip_id"]?.stringValue ?? ""
                if targets.contains(id) {
                    clip["mirror"] = .bool(turnOn)
                } else if wasWhole {
                    clip["mirror"] = .bool(true)
                }
                return clip
            } }
        }
        await perform(predict: change) { [weak self] in
            guard let self, var object = self.snapshot?.draft.objectValue else { return }
            change(&object)
            await self.autosave(.object(object))
        }.value
        return turnOn
    }
    var rotationDegrees: Int { Int(displayDraft?["rotation"]?.doubleValue ?? 0) % 360 }

    /// Volume of the video's own sound (0...2, the server's `audio_volume`), the same on every clip.
    var videoTrackVolume: Double {
        let clips = selectedClips
        guard !clips.isEmpty else { return 1 }
        if clips.allSatisfy({ $0["audio_muted"]?.boolValue == true }) { return 0 }
        return clips.compactMap { $0["audio_volume"]?.doubleValue }.max() ?? 1
    }

    /// Volume of the voice-over track in the preview (kept on this iPhone; 0...1).
    @Published var voiceOverVolume: Double = 1 {
        didSet { UserDefaults.standard.set(voiceOverVolume, forKey: "cutsell.voVolume.\(project.projectID)") }
    }

    // MARK: One queue for every draft edit

    private var actionChain: Task<Void, Never>?
    private var pendingActions = 0
    /// Drafts the server confirmed before each edit (newest last), so Undo can show at once.
    private var confirmedHistory: [JSONValue] = []

    /// Shows `predict`'s result at once, then runs `work` after every earlier edit finished (each
    /// server call builds on the latest revision, so edits never race).
    @discardableResult
    func perform(predict: ((inout [String: JSONValue]) -> Void)?, _ work: @escaping @MainActor () async -> Void) -> Task<Void, Never> {
        if let predict, var object = displayDraft?.objectValue {
            predict(&object)
            optimisticDraft = .object(object)
        }
        pendingActions += 1
        let previous = actionChain
        let task = Task { @MainActor [weak self] in
            _ = await previous?.value
            await work()
            guard let self else { return }
            self.pendingActions -= 1
            if self.pendingActions == 0 { self.optimisticDraft = nil }
        }
        actionChain = task
        return task
    }

    static func mapClips(_ object: inout [String: JSONValue], _ transform: ([[String: JSONValue]]) -> [[String: JSONValue]]) {
        let clips = object["selected"]?.arrayValue?.compactMap(\.objectValue) ?? []
        object["selected"] = .array(transform(clips).map(JSONValue.object))
    }

    /// Sets the video's own volume on every clip (0 mutes) with one save.
    func setVideoTrackVolume(_ volume: Double) async {
        let value = max(0, min(2, volume))
        let change: (inout [String: JSONValue]) -> Void = { object in
            Self.mapClips(&object) { clips in clips.map { clip in
                var clip = clip
                clip["audio_volume"] = .number((value * 1000).rounded() / 1000)
                clip["audio_muted"] = .bool(value <= 0.001)
                return clip
            } }
        }
        await perform(predict: change) { [weak self] in
            guard let self, var object = self.snapshot?.draft.objectValue else { return }
            change(&object)
            await self.autosave(.object(object))
        }.value
    }

    /// Mirror / rotation are whole-video settings saved on the draft (the export reads them).
    func setDraftValue(_ key: String, _ value: JSONValue) async {
        let change: (inout [String: JSONValue]) -> Void = { $0[key] = value }
        await perform(predict: change) { [weak self] in
            guard let self, var object = self.snapshot?.draft.objectValue else { return }
            change(&object)
            await self.autosave(.object(object))
        }.value
    }

    // MARK: Auto Zoom AI keyframes (◇ on the timeline)

    /// Auto Zoom AI on/off for this cut (the export only draws the zooms when it is on).
    var autoZoomEnabled: Bool { displayDraft?["auto_zoom_enabled"]?.boolValue ?? false }

    /// One zoom keyframe placed on the edited timeline. The draft anchors it to the SOURCE time
    /// (so splits and deletes never move it onto other words); `time` is where it falls now.
    struct ZoomMark: Identifiable, Equatable {
        let id: String
        let time: Double
        let duration: Double
        let scale: Double
        let kind: String
    }

    var zoomKeyframes: [[String: JSONValue]] {
        displayDraft?["zoom_keyframes"]?.arrayValue?.compactMap(\.objectValue) ?? []
    }

    /// Keyframes whose source moment is still in the cut, at their place on the edited timeline.
    var zoomMarks: [ZoomMark] {
        let windows = EditorTracks.clipWindows(self)
        return zoomKeyframes.compactMap { item in
            guard let id = item["keyframe_id"]?.stringValue,
                  let sourceTime = item["source_time"]?.doubleValue else { return nil }
            let source = item["source_asset_id"]?.stringValue ?? ""
            guard let window = windows.first(where: { window in
                (source.isEmpty || window.sourceID == source)
                    && sourceTime >= window.sourceStart
                    && sourceTime < window.sourceStart + (window.end - window.start)
            }) else { return nil }
            return ZoomMark(id: id, time: window.start + (sourceTime - window.sourceStart),
                            duration: item["duration"]?.doubleValue ?? 1.2,
                            scale: item["scale"]?.doubleValue ?? 1.08,
                            kind: item["kind"]?.stringValue ?? "key")
        }
        .sorted { $0.time < $1.time }
    }

    /// How much the preview is zoomed at `time` (the same smooth push-in and release as the export).
    func zoomScale(at time: Double) -> Double {
        guard autoZoomEnabled else { return 1 }
        var factor = 1.0
        for mark in zoomMarks where time >= mark.time && time < mark.time + mark.duration {
            let ramp = min(0.35, mark.duration / 3)
            let into = time - mark.time, left = mark.time + mark.duration - time
            func ease(_ x: Double) -> Double { let v = max(0, min(1, x)); return v * v * (3 - 2 * v) }
            factor += (mark.scale - 1) * min(ease(into / ramp), ease(left / ramp))
        }
        return factor
    }

    /// Source position under a point of the edited timeline.
    private func sourcePoint(at time: Double) -> (source: String, seconds: Double)? {
        let windows = EditorTracks.clipWindows(self)
        guard let window = windows.first(where: { time >= $0.start && time < $0.end }) ?? windows.last(where: { time >= $0.end - 0.01 }) else { return nil }
        let into = min(max(0, time - window.start), max(0, window.end - window.start - 0.05))
        return (window.sourceID ?? "", window.sourceStart + into)
    }

    private func saveZoomKeyframes(_ items: [[String: JSONValue]], enable: Bool? = nil) async {
        let change: (inout [String: JSONValue]) -> Void = { object in
            object["zoom_keyframes"] = .array(items.map(JSONValue.object))
            if let enable { object["auto_zoom_enabled"] = .bool(enable) }
        }
        await perform(predict: change) { [weak self] in
            guard let self, var object = self.snapshot?.draft.objectValue else { return }
            change(&object)
            await self.autosave(.object(object))
        }.value
    }

    /// The ◇ control: a zoom keyframe at the cursor (turns Auto Zoom AI on so it shows in the export).
    @discardableResult
    func addZoomKeyframe(at time: Double) async -> String? {
        guard let point = sourcePoint(at: time) else { return nil }
        let id = "zk_" + UUID().uuidString.prefix(8).lowercased()
        var items = zoomKeyframes
        items.append([
            "keyframe_id": .string(id), "source_asset_id": .string(point.source),
            "source_time": .number((point.seconds * 1000).rounded() / 1000),
            "duration": .number(1.2), "scale": .number(1.1),
            "kind": .string("manual"), "origin": .string("user"),
        ])
        await saveZoomKeyframes(items, enable: true)
        return id
    }

    /// Drag a ◇ to another moment of the cut.
    func moveZoomKeyframe(id: String, to time: Double) async {
        guard let point = sourcePoint(at: max(0, time)) else { return }
        let items = zoomKeyframes.map { item -> [String: JSONValue] in
            guard item["keyframe_id"]?.stringValue == id else { return item }
            var moved = item
            moved["source_asset_id"] = .string(point.source)
            moved["source_time"] = .number((point.seconds * 1000).rounded() / 1000)
            return moved
        }
        await saveZoomKeyframes(items)
    }

    func removeZoomKeyframe(id: String) async {
        await saveZoomKeyframes(zoomKeyframes.filter { $0["keyframe_id"]?.stringValue != id })
    }

    #if DEBUG
    /// Editor probe only: shows two sample zoom keyframes on a real cut (the live server keeps
    /// keyframes once the new engine is deployed); nothing is saved.
    func probeShowZoomKeyframes(at times: [Double]) {
        guard var object = displayDraft?.objectValue else { return }
        var items: [JSONValue] = []
        for (index, time) in times.enumerated() {
            guard let point = sourcePoint(at: time) else { continue }
            items.append(.object([
                "keyframe_id": .string("probe-\(index)"), "source_asset_id": .string(point.source),
                "source_time": .number(point.seconds), "duration": .number(1.5), "scale": .number(index == 0 ? 1.08 : 1.12),
                "kind": .string(index == 0 ? "hook" : "price"), "origin": .string("ai"),
            ]))
        }
        object["zoom_keyframes"] = .array(items)
        object["auto_zoom_enabled"] = .bool(true)
        optimisticDraft = .object(object)
    }
    #endif

    /// Editor → Style → Auto Zoom AI. The keyframes stay in the draft either way; off = not drawn.
    func setAutoZoom(_ enabled: Bool) async {
        await setDraftValue("auto_zoom_enabled", .bool(enabled))
    }

    var timelineDuration: Double {
        selectedClips.reduce(0) { total, clip in
            total + max(0, (clip["end"]?.doubleValue ?? 0) - (clip["start"]?.doubleValue ?? 0))
        }
    }

    func alternates(for selectedClip: [String: JSONValue]) -> [[String: JSONValue]] {
        guard let groupID = selectedClip["take_group_id"]?.stringValue, !groupID.isEmpty else { return [] }
        return alternateClips.filter { $0["take_group_id"]?.stringValue == groupID }
    }

    // Caption settings the creator has just chosen, shown at once while the
    // server confirms them. Each one is removed as soon as its save finishes:
    // on success the draft already carries it, on failure the screen falls
    // back to what the server really has (and the error is shown).
    @Published private var captionOverrides: [String: JSONValue] = [:]
    private var captionOverrideTokens: [String: Int] = [:]
    private var captionOverrideCounter = 0

    private func captionValue(_ key: String) -> JSONValue? {
        captionOverrides[key] ?? displayDraft?[key]
    }

    var captionsEnabled: Bool {
        captionValue("captions_enabled")?.boolValue ?? true
    }

    var captionPreset: String {
        captionValue("caption_preset")?.stringValue ?? "classic"
    }

    /// One of `CaptionFontCatalog`'s keys (the server's `caption_font`).
    var captionFont: String {
        captionValue("caption_font")?.stringValue ?? CaptionFontCatalog.defaultKey
    }

    /// Caption centre for the whole video, 0...1 of the frame width / height.
    var captionX: Double {
        captionValue("caption_x")?.doubleValue ?? CaptionLayout.defaultX
    }

    var captionY: Double {
        captionValue("caption_y")?.doubleValue ?? CaptionLayout.defaultY
    }

    /// Caption size for the whole video (1 = the typeface's own size).
    var captionScale: Double {
        captionValue("caption_scale")?.doubleValue ?? CaptionLayout.defaultScale
    }

    /// Which engine produced this draft; short timed captions exist only for "simple".
    var draftEngine: String? {
        displayDraft?["diagnostics"]?["engine"]?.stringValue
    }

    func load() async {
        isLoading = true
        defer { isLoading = false }
        do {
            async let draftRequest: DraftSnapshot = api.request(
                "/v1/projects/\(project.projectID)/draft",
                query: [URLQueryItem(name: "user_id", value: session.userID)]
            )
            async let assetRequest = TimelineAssetRegistryClient.list(
                projectID: project.projectID,
                userID: session.userID,
                api: api
            )
            async let compositionRequest: Void = refreshTimelineComposition()
            let loadedDraft = try await draftRequest
            snapshot = loadedDraft
            let loadedAssets = try await assetRequest
            timelineAssetLibrary = loadedAssets
            await compositionRequest
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// D-282A Stage 6: "refresh the snapshot and library after a confirmed
    /// mutation" -- also called on initial load. A project with no saved
    /// Overlay/Voice-over timeline yet is a legitimate, honest empty state
    /// (`TimelineCompositionError.noTimelineSaved`, from the route's real
    /// 404), never surfaced as an error alert.
    func refreshTimelineComposition() async {
        do {
            timelineComposition = try await TimelineCompositionClient.get(
                projectID: project.projectID, userID: session.userID, api: api
            )
        } catch is TimelineCompositionError {
            timelineComposition = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Same real `TimelineAssetRegistryClient.list` call `saveTimelineComposition`
    /// already runs inline after a confirmed mutation, factored out so a new
    /// asset import (which never itself mutates the timeline composition) can
    /// refresh the library too -- never a second/invented listing authority.
    func refreshTimelineAssetLibrary() async {
        if let loadedAssets = try? await TimelineAssetRegistryClient.list(
            projectID: project.projectID, userID: session.userID, api: api
        ) {
            timelineAssetLibrary = loadedAssets
        }
    }

    /// Imports an already-existing local audio file as a new VOICE_OVER
    /// asset via the real D-282A upload+registration routes
    /// (`VoiceOverImportManager`) -- never a live microphone capture (no
    /// such authority exists). Refreshes the asset library only after the
    /// backend has actually confirmed the new asset; a failed import only
    /// ever sets `errorMessage`, never fakes a new READY asset.
    func importVoiceOverAudio(fileURL: URL) async {
        do {
            _ = try await VoiceOverImportManager.shared.importAudio(
                fileURL: fileURL, projectID: project.projectID, session: session, api: api
            )
            await refreshTimelineAssetLibrary()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    enum RecordedVoiceOverOutcome: Equatable {
        /// Uploaded and placed on the voice-over track at the cursor.
        case placed
        /// Uploaded and saved as a voice-over, but the timeline can't take placements yet
        /// (this project has no registered primary source).
        case savedNotPlaced
        /// Upload or registration failed; `errorMessage` says why.
        case failed
    }

    /// A take recorded with the microphone: uploaded by the SAME path as an imported file
    /// (`/timeline-uploads` + `/timeline-assets`), then placed on the voice-over track starting
    /// at `start` (the timeline cursor) for its full length.
    func addRecordedVoiceOver(fileURL: URL, at start: Double) async -> RecordedVoiceOverOutcome {
        let asset: TimelineMediaAsset
        do {
            asset = try await VoiceOverImportManager.shared.importAudio(
                fileURL: fileURL, projectID: project.projectID, session: session, api: api
            )
        } catch {
            errorMessage = error.localizedDescription
            return .failed
        }
        // Kept on this iPhone so the preview plays it on its own track.
        VoiceOverStore.save(fileURL, assetID: asset.assetID)
        await refreshTimelineAssetLibrary()
        guard asset.isReady, asset.durationSec > 0 else {
            errorMessage = "The recording couldn't be used. Please record again."
            return .failed
        }
        guard baseEditAssetID != nil else { return .savedNotPlaced }
        let begin = max(0, start)
        await addVoiceOverPlacement(
            assetID: asset.assetID, start: begin, end: begin + asset.durationSec,
            sourceIn: 0, sourceOut: asset.durationSec
        )
        return errorMessage == nil ? .placed : .failed
    }

    /// Imports an already-existing local video file as a new
    /// SUPPLEMENTAL_BROLL asset via the real D-282A upload+registration
    /// routes (`BrollImportManager`) -- never a fabricated capture/import
    /// path. Same confirmed-before-refresh discipline as
    /// `importVoiceOverAudio`.
    func importBrollVideo(fileURL: URL) async {
        do {
            _ = try await BrollImportManager.shared.importVideo(
                fileURL: fileURL, projectID: project.projectID, session: session, api: api
            )
            await refreshTimelineAssetLibrary()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Overlay (real, positioned/scaled media overlay -- D-091
    // corrective gate). This is a GENUINELY DIFFERENT feature from B-roll
    // above: `media_overlays` lives on the canonical `draft` dict (the
    // SAME system `selected`/`alternates`/`text_overlays` already use),
    // mutated via `/v1/overlays/add|update|remove`
    // (`cutsell_worker/overlay_edits.py`, confirmed real and ACTIVE:
    // `export_job.py` reads `draft.media_overlays`, downloads each
    // overlay's media, and passes it to `render.render_preview`, which
    // invokes `media_overlay_render.build_final_overlay_command` -- a real
    // ffmpeg `overlay=x=...:y=...` filter with scale (`width`), a timed
    // `enable` window, and `mute_audio`-gated audio mixing). Uses the SAME
    // `edit(path:body:)` + `autosave(_:)` pattern already established for
    // Main Video split/remove/swap -- never the D-282A TimelineComposition
    // PUT-with-revision pattern, since overlays are not part of that
    // system. Undo/Redo therefore reuse the SAME real `/draft/undo` /
    // `/draft/redo` authority as Main Video (`undo()`/`redo()` above),
    // never a second Undo mechanism.

    func addMediaOverlay(
        kind: String, uri: String, start: Double, end: Double,
        x: Double = 0.5, y: Double = 0.5, width: Double = 0.4,
        sourceStart: Double = 0.0, sourceEnd: Double? = nil, muteAudio: Bool = true
    ) async {
        guard let snapshot else { return }
        var body: [String: JSONValue] = [
            "draft": snapshot.draft,
            "project_id": .string(project.projectID),
            "user_id": .string(session.userID),
            "kind": .string(kind),
            "uri": .string(uri),
            "start": .number(start),
            "end": .number(end),
            "x": .number(x),
            "y": .number(y),
            "width": .number(width),
            "source_start": .number(sourceStart),
            "mute_audio": .bool(muteAudio),
        ]
        if let sourceEnd { body["source_end"] = .number(sourceEnd) }
        let edited = await edit(path: "/v1/overlays/add", body: .object(body))
        if let edited { await autosave(edited) }
    }

    func updateMediaOverlay(
        overlayID: String, start: Double? = nil, end: Double? = nil,
        x: Double? = nil, y: Double? = nil, width: Double? = nil, muteAudio: Bool? = nil
    ) async {
        guard let snapshot else { return }
        var body: [String: JSONValue] = [
            "draft": snapshot.draft,
            "overlay_id": .string(overlayID),
        ]
        if let start { body["start"] = .number(start) }
        if let end { body["end"] = .number(end) }
        if let x { body["x"] = .number(x) }
        if let y { body["y"] = .number(y) }
        if let width { body["width"] = .number(width) }
        if let muteAudio { body["mute_audio"] = .bool(muteAudio) }
        let edited = await edit(path: "/v1/overlays/update", body: .object(body))
        if let edited { await autosave(edited) }
    }

    func removeMediaOverlay(overlayID: String) async {
        guard let snapshot else { return }
        let edited = await edit(path: "/v1/overlays/remove", body: .object([
            "draft": snapshot.draft,
            "overlay_id": .string(overlayID),
        ]))
        if let edited { await autosave(edited) }
    }

    /// Imports an already-existing local photo/video file via the real,
    /// ACTIVE `OverlayUploadManager` (`POST /v1/overlays/uploads/presign`
    /// -> real S3 multipart upload), then immediately places it on the
    /// timeline via `addMediaOverlay` -- `/v1/overlays/add` has no
    /// separate "register only" step (unlike D-282A's B-roll/voice-over
    /// two-phase upload+registration), so Import and Add are one real
    /// action here.
    func importOverlayMedia(fileURL: URL) async {
        do {
            let presign = try await OverlayUploadManager.shared.upload(
                fileURL: fileURL, projectID: project.projectID, session: session
            )
            let end = timelineDuration
            await addMediaOverlay(kind: presign.kind, uri: presign.uri, start: max(0, end - 3), end: max(0.5, end))
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Returns whether the real PUT actually succeeded -- callers (e.g.
    /// `undoLastTimelineMutation`) must never infer success merely because
    /// `await` returned; a caught error here means `timelineComposition`
    /// was left exactly as it was and this returns `false`.
    @discardableResult
    private func saveTimelineComposition(
        brollPlacements: [BrollPlacement],
        voiceOverPlacements: [VoiceOverPlacement],
        timelineDurationSec: Double,
        capturePreviousForUndo: Bool
    ) async -> Bool {
        guard let baseEditAssetID else {
            errorMessage = "No primary source asset is registered for this project yet -- overlay and voice-over placement need one first."
            return false
        }
        isSavingTimeline = true
        defer { isSavingTimeline = false }
        let previous = timelineComposition
        do {
            _ = try await TimelineCompositionClient.save(
                projectID: project.projectID,
                userID: session.userID,
                baseEditAssetID: baseEditAssetID,
                timelineDurationSec: max(0.01, timelineDurationSec),
                brollPlacements: brollPlacements,
                voiceOverPlacements: voiceOverPlacements,
                expectedRevisionIdentity: previous?.timelineRevisionIdentity,
                api: api
            )
            if capturePreviousForUndo { lastCompositionBeforeMutation = previous }
            await refreshTimelineComposition()
            let loadedAssets = try? await TimelineAssetRegistryClient.list(
                projectID: project.projectID, userID: session.userID, api: api
            )
            if let loadedAssets { timelineAssetLibrary = loadedAssets }
            return true
        } catch {
            // D-282A: never show a mutation as saved before backend
            // confirmation -- `timelineComposition` is left exactly as it
            // was before this attempt.
            errorMessage = error.localizedDescription
            return false
        }
    }

    private var currentBrollPlacements: [BrollPlacement] { timelineComposition?.brollPlacements ?? [] }
    private var currentVoiceOverPlacements: [VoiceOverPlacement] { timelineComposition?.voiceOverPlacements ?? [] }
    private var currentTimelineDuration: Double {
        max(
            timelineComposition?.timelineDurationSec ?? 0,
            (currentBrollPlacements.map(\.timelineEndSec) + currentVoiceOverPlacements.map(\.timelineEndSec)).max() ?? 1
        )
    }

    func addBrollPlacement(assetID: String, start: Double, end: Double, sourceIn: Double, sourceOut: Double) async {
        var placements = currentBrollPlacements
        placements.append(BrollPlacement(
            placementID: "broll_\(UUID().uuidString.prefix(8))", assetID: assetID,
            timelineStartSec: start, timelineEndSec: end,
            sourceInSec: sourceIn, sourceOutSec: sourceOut, audioMode: .keepPrimaryVoice
        ))
        await saveTimelineComposition(
            brollPlacements: placements, voiceOverPlacements: currentVoiceOverPlacements,
            timelineDurationSec: max(currentTimelineDuration, end), capturePreviousForUndo: false
        )
    }

    func addVoiceOverPlacement(assetID: String, start: Double, end: Double, sourceIn: Double, sourceOut: Double) async {
        var placements = currentVoiceOverPlacements
        placements.append(VoiceOverPlacement(
            placementID: "vo_\(UUID().uuidString.prefix(8))", assetID: assetID,
            timelineStartSec: start, timelineEndSec: end,
            sourceInSec: sourceIn, sourceOutSec: sourceOut, transcriptReference: nil
        ))
        await saveTimelineComposition(
            brollPlacements: currentBrollPlacements, voiceOverPlacements: placements,
            timelineDurationSec: max(currentTimelineDuration, end), capturePreviousForUndo: false
        )
    }

    func removeBrollPlacement(id: String) async {
        let placements = currentBrollPlacements.filter { $0.placementID != id }
        await saveTimelineComposition(
            brollPlacements: placements, voiceOverPlacements: currentVoiceOverPlacements,
            timelineDurationSec: currentTimelineDuration, capturePreviousForUndo: true
        )
    }

    func removeVoiceOverPlacement(id: String) async {
        let placements = currentVoiceOverPlacements.filter { $0.placementID != id }
        await saveTimelineComposition(
            brollPlacements: currentBrollPlacements, voiceOverPlacements: placements,
            timelineDurationSec: currentTimelineDuration, capturePreviousForUndo: true
        )
    }

    /// Split affects ONLY the selected placement, at the playhead time --
    /// client-computed into two placements, saved through the SAME real
    /// PUT operation (never a new backend authority).
    func splitBrollPlacement(id: String, at playheadTime: Double) async {
        var placements = currentBrollPlacements
        guard let index = placements.firstIndex(where: { $0.placementID == id }) else { return }
        let original = placements[index]
        guard playheadTime > original.timelineStartSec + 0.05, playheadTime < original.timelineEndSec - 0.05 else { return }
        let sourceSplit = original.sourceInSec + (playheadTime - original.timelineStartSec)
        let first = BrollPlacement(
            placementID: original.placementID, assetID: original.assetID,
            timelineStartSec: original.timelineStartSec, timelineEndSec: playheadTime,
            sourceInSec: original.sourceInSec, sourceOutSec: sourceSplit, audioMode: original.audioMode
        )
        let second = BrollPlacement(
            placementID: "broll_\(UUID().uuidString.prefix(8))", assetID: original.assetID,
            timelineStartSec: playheadTime, timelineEndSec: original.timelineEndSec,
            sourceInSec: sourceSplit, sourceOutSec: original.sourceOutSec, audioMode: original.audioMode
        )
        placements.replaceSubrange(index...index, with: [first, second])
        await saveTimelineComposition(
            brollPlacements: placements, voiceOverPlacements: currentVoiceOverPlacements,
            timelineDurationSec: currentTimelineDuration, capturePreviousForUndo: true
        )
    }

    /// The ONLY real, backend-accepted editable field on a B-roll
    /// placement besides timing: `BrollPlacementModel.audio_mode`
    /// (`cutsell_app/timeline_routes.py`), one of the three real
    /// `TimelineAudioMode` values. Position, scale, rotation, opacity, and
    /// keyframes have no field anywhere on `BrollPlacement` -- never
    /// exposed as a real mutation.
    func setBrollAudioMode(id: String, mode: TimelineAudioMode) async {
        var placements = currentBrollPlacements
        guard let index = placements.firstIndex(where: { $0.placementID == id }) else { return }
        let original = placements[index]
        placements[index] = BrollPlacement(
            placementID: original.placementID, assetID: original.assetID,
            timelineStartSec: original.timelineStartSec, timelineEndSec: original.timelineEndSec,
            sourceInSec: original.sourceInSec, sourceOutSec: original.sourceOutSec, audioMode: mode
        )
        await saveTimelineComposition(
            brollPlacements: placements, voiceOverPlacements: currentVoiceOverPlacements,
            timelineDurationSec: currentTimelineDuration, capturePreviousForUndo: true
        )
    }

    func splitVoiceOverPlacement(id: String, at playheadTime: Double) async {
        var placements = currentVoiceOverPlacements
        guard let index = placements.firstIndex(where: { $0.placementID == id }) else { return }
        let original = placements[index]
        guard playheadTime > original.timelineStartSec + 0.05, playheadTime < original.timelineEndSec - 0.05 else { return }
        let sourceSplit = original.sourceInSec + (playheadTime - original.timelineStartSec)
        let first = VoiceOverPlacement(
            placementID: original.placementID, assetID: original.assetID,
            timelineStartSec: original.timelineStartSec, timelineEndSec: playheadTime,
            sourceInSec: original.sourceInSec, sourceOutSec: sourceSplit, transcriptReference: original.transcriptReference
        )
        let second = VoiceOverPlacement(
            placementID: "vo_\(UUID().uuidString.prefix(8))", assetID: original.assetID,
            timelineStartSec: playheadTime, timelineEndSec: original.timelineEndSec,
            sourceInSec: sourceSplit, sourceOutSec: original.sourceOutSec, transcriptReference: nil
        )
        placements.replaceSubrange(index...index, with: [first, second])
        await saveTimelineComposition(
            brollPlacements: currentBrollPlacements, voiceOverPlacements: placements,
            timelineDurationSec: currentTimelineDuration, capturePreviousForUndo: true
        )
    }

    /// Reverts to the composition captured just before the last
    /// delete/split -- re-saved through the SAME real PUT operation
    /// (never a fabricated client-only undo). A failed revert surfaces
    /// the real backend error and leaves the captured snapshot intact (so
    /// the user can retry) -- it never pretends to have undone anything,
    /// and never silently discards the one recorded undo opportunity on a
    /// failed attempt.
    func undoLastTimelineMutation() async {
        guard let previous = lastCompositionBeforeMutation else { return }
        let succeeded = await saveTimelineComposition(
            brollPlacements: previous.brollPlacements, voiceOverPlacements: previous.voiceOverPlacements,
            timelineDurationSec: previous.timelineDurationSec, capturePreviousForUndo: false
        )
        if succeeded {
            lastCompositionBeforeMutation = nil
        }
    }

    func swap(selectedClipID: String, replacementClipID: String) async {
        guard let snapshot else { return }
        let edited = await edit(path: "/v1/draft-edits/swap", body: .object([
            "draft": snapshot.draft,
            "selected_clip_id": .string(selectedClipID),
            "replacement_clip_id": .string(replacementClipID)
        ]))
        if let edited { await autosave(edited) }
    }

    func remove(clipID: String) async {
        await perform(predict: { object in
            Self.mapClips(&object) { $0.filter { $0["clip_id"]?.stringValue != clipID } }
        }) { [weak self] in await self?.removeNow(clipID: clipID) }.value
    }

    private func removeNow(clipID: String) async {
        guard let snapshot else { return }
        let edited = await edit(path: "/v1/draft-edits/remove", body: .object([
            "draft": snapshot.draft,
            "clip_id": .string(clipID)
        ]))
        if let edited { await autosave(edited) }
    }

    func reorder(clipIDs: [String]) async {
        guard let snapshot else { return }
        let edited = await edit(path: "/v1/draft-edits/reorder", body: .object([
            "draft": snapshot.draft,
            "ordered_clip_ids": .array(clipIDs.map(JSONValue.string))
        ]))
        if let edited { await autosave(edited) }
    }

    func trim(clipID: String, start: Double, end: Double) async {
        guard let snapshot else { return }
        let edited = await edit(path: "/v1/draft-edits/trim", body: .object([
            "draft": snapshot.draft,
            "clip_id": .string(clipID),
            "start": .number(start),
            "end": .number(end)
        ]))
        if let edited { await autosave(edited) }
    }

    /// The closest place to `sourceTime` (source seconds) where a selected
    /// clip can really be split: the middle of a gap between two spoken
    /// words, leaving a usable piece on both sides. The backend refuses a
    /// split that cuts through a word, so the playhead position is snapped
    /// here first. `nil` when the clip has no safe place to split.
    func safeSplitTime(clipID: String, near sourceTime: Double) -> Double? {
        guard let clip = selectedClips.first(where: { $0["clip_id"]?.stringValue == clipID }) else { return nil }
        let start = clip["start"]?.doubleValue ?? 0
        let end = clip["end"]?.doubleValue ?? start
        let words = clip["words"]?.arrayValue?.compactMap(\.objectValue) ?? []
        if words.isEmpty {
            // No speech in this clip: any point that leaves both sides usable is safe.
            return (sourceTime - start >= 0.15 && end - sourceTime >= 0.15) ? sourceTime : nil
        }
        var gaps: [Double] = []
        for (left, right) in zip(words, words.dropFirst()) {
            guard let leftEnd = left["end"]?.doubleValue,
                  let rightStart = right["start"]?.doubleValue else { continue }
            let candidate = (leftEnd + rightStart) / 2
            if candidate - start >= 0.15, end - candidate >= 0.15 { gaps.append(candidate) }
        }
        return gaps.min(by: { abs($0 - sourceTime) < abs($1 - sourceTime) })
    }

    func split(clipID: String, at sourceTime: Double) async {
        await perform(predict: { object in
            Self.mapClips(&object) { clips in clips.flatMap { clip -> [[String: JSONValue]] in
                guard clip["clip_id"]?.stringValue == clipID else { return [clip] }
                return Self.predictedSplit(clip, at: sourceTime)
            } }
        }) { [weak self] in await self?.splitNow(clipID: clipID, at: sourceTime) }.value
    }

    /// The two halves the server will make (shown until it answers).
    static func predictedSplit(_ clip: [String: JSONValue], at time: Double) -> [[String: JSONValue]] {
        var first = clip, second = clip
        let words = clip["words"]?.arrayValue?.compactMap(\.objectValue) ?? []
        let before = words.filter { ($0["end"]?.doubleValue ?? 0) <= time }
        let after = words.filter { ($0["end"]?.doubleValue ?? 0) > time }
        first["end"] = .number(time)
        second["start"] = .number(time)
        second["clip_id"] = .string((clip["clip_id"]?.stringValue ?? "clip") + "-b")
        first["words"] = .array(before.map(JSONValue.object))
        second["words"] = .array(after.map(JSONValue.object))
        func text(_ items: [[String: JSONValue]]) -> String {
            items.compactMap { $0["word"]?.stringValue ?? $0["text"]?.stringValue }.joined(separator: " ")
        }
        if !words.isEmpty {
            first["caption_text"] = .string(text(before))
            second["caption_text"] = .string(text(after))
        }
        return [first, second]
    }

    private func splitNow(clipID: String, at sourceTime: Double) async {
        guard let snapshot else { return }
        let edited = await edit(path: "/v1/draft-edits/split", body: .object([
            "draft": snapshot.draft,
            "clip_id": .string(clipID),
            "split_time": .number(sourceTime)
        ]))
        if let edited { await autosave(edited) }
    }

    func editCaption(clipID: String, text: String) async {
        // Same queue as every other edit, so a text fix never races a look/typeface save.
        await perform(predict: { object in
            Self.mapClips(&object) { clips in clips.map { clip in
                guard clip["clip_id"]?.stringValue == clipID else { return clip }
                var clip = clip
                clip["caption_text"] = .string(text)
                return clip
            } }
        }) { [weak self] in await self?.sendCaptionText(clipID: clipID, text: text) }.value
    }

    private func sendCaptionText(clipID: String, text: String) async {
        guard let snapshot else { return }
        let edited = await edit(path: "/v1/draft-edits/captions", body: .object([
            "draft": snapshot.draft,
            "edits": .array([.object([
                "clip_id": .string(clipID),
                "text": .string(text)
            ])])
        ]))
        if let edited { await autosave(edited) }
    }

    /// Whole-video caption settings through the real
    /// `/v1/draft-edits/caption-settings` route, then the usual autosave.
    /// Only the values passed are changed.
    func setCaptionSettings(
        enabled: Bool? = nil,
        preset: String? = nil,
        font: String? = nil,
        x: Double? = nil,
        y: Double? = nil,
        scale: Double? = nil
    ) async {
        let task = applyCaptionSettings(enabled: enabled, preset: preset, font: font, x: x, y: y, scale: scale)
        _ = await task?.value
    }

    /// Same as `setCaptionSettings`, but returns right away: the new values
    /// are on screen before this returns (used when fingers lift off the
    /// caption, so it never jumps back), and the save runs in the queue.
    @discardableResult
    func applyCaptionSettings(
        enabled: Bool? = nil,
        preset: String? = nil,
        font: String? = nil,
        x: Double? = nil,
        y: Double? = nil,
        scale: Double? = nil
    ) -> Task<Void, Never>? {
        var changes: [String: JSONValue] = [:]
        if let enabled { changes["captions_enabled"] = .bool(enabled) }
        if let preset { changes["caption_preset"] = .string(preset) }
        if let font { changes["caption_font"] = .string(font) }
        if let x { changes["caption_x"] = .number(x) }
        if let y { changes["caption_y"] = .number(y) }
        if let scale { changes["caption_scale"] = .number(scale) }
        guard !changes.isEmpty else { return nil }

        captionOverrideCounter += 1
        let token = captionOverrideCounter
        for (key, value) in changes {
            captionOverrides[key] = value
            captionOverrideTokens[key] = token
        }

        let changedKeys = Array(changes.keys)
        return perform(predict: nil) { [weak self] in
            guard let self else { return }
            await self.sendCaptionSettings(enabled: enabled, preset: preset, font: font, x: x, y: y, scale: scale)
            for key in changedKeys where self.captionOverrideTokens[key] == token {
                self.captionOverrides[key] = nil
                self.captionOverrideTokens[key] = nil
            }
        }
    }

    private func sendCaptionSettings(
        enabled: Bool?, preset: String?, font: String?, x: Double?, y: Double?, scale: Double?
    ) async {
        guard let snapshot else { return }
        var object: [String: JSONValue] = ["draft": snapshot.draft]
        object["enabled"] = enabled.map(JSONValue.bool) ?? .null
        object["preset"] = preset.map(JSONValue.string) ?? .null
        object["font"] = font.map(JSONValue.string) ?? .null
        object["x"] = x.map(JSONValue.number) ?? .null
        object["y"] = y.map(JSONValue.number) ?? .null
        object["scale"] = scale.map(JSONValue.number) ?? .null
        let edited = await edit(path: "/v1/draft-edits/caption-settings", body: .object(object))
        if let edited { await autosave(edited) }
    }

    func setAudio(clipID: String, muted: Bool? = nil, volume: Double? = nil) async {
        await perform(predict: { object in
            Self.mapClips(&object) { clips in clips.map { clip in
                guard clip["clip_id"]?.stringValue == clipID else { return clip }
                var clip = clip
                if let muted { clip["audio_muted"] = .bool(muted) }
                if let volume { clip["audio_volume"] = .number(volume) }
                return clip
            } }
        }) { [weak self] in await self?.setAudioNow(clipID: clipID, muted: muted, volume: volume) }.value
    }

    private func setAudioNow(clipID: String, muted: Bool?, volume: Double?) async {
        guard let snapshot else { return }
        var object: [String: JSONValue] = ["draft": snapshot.draft, "clip_id": .string(clipID)]
        object["muted"] = muted.map(JSONValue.bool) ?? .null
        object["volume"] = volume.map(JSONValue.number) ?? .null
        let edited = await edit(path: "/v1/draft-edits/audio", body: .object(object))
        if let edited { await autosave(edited) }
    }

    func addTextOverlay(text: String, start: Double = 0, end: Double? = nil) async {
        guard let snapshot else { return }
        let resolvedEnd = max(start + 0.15, min(end ?? max(1, timelineDuration), max(1, timelineDuration)))
        let edited = await edit(path: "/v1/draft-edits/text/add", body: .object([
            "draft": snapshot.draft,
            "text": .string(text),
            "start": .number(start),
            "end": .number(resolvedEnd),
            "x": .number(0.5),
            "y": .number(0.2),
            "scale": .number(1.0)
        ]))
        if let edited { await autosave(edited) }
    }

    func updateTextOverlay(id: String, text: String? = nil, x: Double? = nil, y: Double? = nil, scale: Double? = nil) async {
        guard let snapshot else { return }
        var body: [String: JSONValue] = ["draft": snapshot.draft, "overlay_id": .string(id)]
        body["text"] = text.map(JSONValue.string) ?? .null
        body["x"] = x.map(JSONValue.number) ?? .null
        body["y"] = y.map(JSONValue.number) ?? .null
        body["scale"] = scale.map(JSONValue.number) ?? .null
        let edited = await edit(path: "/v1/draft-edits/text/update", body: .object(body))
        if let edited { await autosave(edited) }
    }

    func removeTextOverlay(id: String) async {
        await perform(predict: { object in
            let items = object["text_overlays"]?.arrayValue?.compactMap(\.objectValue) ?? []
            object["text_overlays"] = .array(items.filter { $0["overlay_id"]?.stringValue != id }.map(JSONValue.object))
        }) { [weak self] in await self?.removeTextOverlayNow(id: id) }.value
    }

    private func removeTextOverlayNow(id: String) async {
        guard let snapshot else { return }
        let edited = await edit(path: "/v1/draft-edits/text/remove", body: .object([
            "draft": snapshot.draft,
            "overlay_id": .string(id)
        ]))
        if let edited { await autosave(edited) }
    }

    func addMediaOverlay(fileURL: URL, start: Double = 0, end: Double? = nil) async {
        guard let snapshot else { return }
        do {
            let uploaded = try await OverlayUploadManager.shared.upload(
                fileURL: fileURL,
                projectID: project.projectID,
                session: session
            )
            let resolvedEnd = max(start + 0.15, min(end ?? max(1, timelineDuration), max(1, timelineDuration)))
            let edited: JSONValue = try await api.request(
                "/v1/overlays/add",
                method: "POST",
                body: JSONValue.object([
                    "project_id": .string(project.projectID),
                    "user_id": .string(session.userID),
                    "draft": snapshot.draft,
                    "kind": .string(uploaded.kind),
                    "uri": .string(uploaded.uri),
                    "start": .number(start),
                    "end": .number(resolvedEnd),
                    "x": .number(0.5),
                    "y": .number(0.5),
                    "width": .number(0.4),
                    "source_start": .number(0),
                    "source_end": .null,
                    "mute_audio": .bool(true)
                ])
            )
            await autosave(edited)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func updateMediaOverlay(id: String, x: Double? = nil, y: Double? = nil, width: Double? = nil, muted: Bool? = nil) async {
        guard let snapshot else { return }
        var body: [String: JSONValue] = ["draft": snapshot.draft, "overlay_id": .string(id)]
        body["x"] = x.map(JSONValue.number) ?? .null
        body["y"] = y.map(JSONValue.number) ?? .null
        body["width"] = width.map(JSONValue.number) ?? .null
        body["mute_audio"] = muted.map(JSONValue.bool) ?? .null
        let edited = await edit(path: "/v1/overlays/update", body: .object(body))
        if let edited { await autosave(edited) }
    }

    func removeMediaOverlay(id: String) async {
        await perform(predict: { object in
            let items = object["media_overlays"]?.arrayValue?.compactMap(\.objectValue) ?? []
            object["media_overlays"] = .array(items.filter { $0["overlay_id"]?.stringValue != id }.map(JSONValue.object))
        }) { [weak self] in await self?.removeMediaOverlayNow(id: id) }.value
    }

    private func removeMediaOverlayNow(id: String) async {
        guard let snapshot else { return }
        let edited = await edit(path: "/v1/overlays/remove", body: .object([
            "draft": snapshot.draft,
            "overlay_id": .string(id)
        ]))
        if let edited { await autosave(edited) }
    }

    /// Returns whether the real `/draft/undo` call actually succeeded --
    /// callers (e.g. TimelineEditorView's Redo gating) must never infer
    /// success merely from `await` returning; a caught error here means
    /// `snapshot` was left exactly as it was and this returns `false`.
    @discardableResult
    func undo() async -> Bool {
        let result = ResultBox()
        var predict: ((inout [String: JSONValue]) -> Void)?
        if let restored = confirmedHistory.last?.objectValue { predict = { $0 = restored } }
        await perform(predict: predict) { [weak self] in result.value = await self?.undoNow() ?? false }.value
        return result.value
    }

    private func undoNow() async -> Bool {
        guard let snapshot else { return false }
        do {
            if !confirmedHistory.isEmpty { confirmedHistory.removeLast() }
            self.snapshot = try await api.request(
                "/v1/projects/\(project.projectID)/draft/undo",
                method: "POST",
                body: JSONValue.object([
                    "user_id": .string(session.userID),
                    "expected_revision": .number(Double(snapshot.revision))
                ])
            )
            return true
        } catch {
            errorMessage = error.localizedDescription
            return false
        }
    }

    /// Same honest success/failure contract as `undo()` -- see above.
    @discardableResult
    func redo() async -> Bool {
        let result = ResultBox()
        await perform(predict: nil) { [weak self] in result.value = await self?.redoNow() ?? false }.value
        return result.value
    }

    private func redoNow() async -> Bool {
        guard let snapshot else { return false }
        do {
            self.snapshot = try await api.request(
                "/v1/projects/\(project.projectID)/draft/redo",
                method: "POST",
                body: JSONValue.object([
                    "user_id": .string(session.userID),
                    "expected_revision": .number(Double(snapshot.revision))
                ])
            )
            return true
        } catch {
            errorMessage = error.localizedDescription
            return false
        }
    }

    func export() async {
        guard let snapshot else { return }
        do {
            let filteredSources = snapshot.sources.compactMap { source -> JSONValue? in
                guard let object = source.objectValue,
                      let sourceID = object["source_asset_id"],
                      let originalName = object["original_name"],
                      let uri = object["uri"] else { return nil }
                return .object(["source_asset_id": sourceID, "original_name": originalName, "uri": uri])
            }
            let response: ExportSubmitResponse = try await api.request(
                "/v1/exports/jobs",
                method: "POST",
                body: JSONValue.object([
                    "project_id": .string(project.projectID),
                    "user_id": .string(session.userID),
                    "draft": snapshot.draft,
                    "sources": .array(filteredSources)
                ])
            )
            await pollExport(jobID: response.jobID)
        } catch { errorMessage = error.localizedDescription }
    }

    private func edit(path: String, body: JSONValue) async -> JSONValue? {
        do {
            let edited: JSONValue = try await api.request(path, method: "POST", body: body)
            return edited
        } catch {
            errorMessage = error.localizedDescription
            return nil
        }
    }

    private func autosave(_ draft: JSONValue) async {
        guard let snapshot else { return }
        isSaving = true
        defer { isSaving = false }
        confirmedHistory.append(snapshot.draft)
        if confirmedHistory.count > 20 { confirmedHistory.removeFirst() }
        do {
            self.snapshot = try await api.request(
                "/v1/projects/\(project.projectID)/draft",
                method: "PUT",
                body: JSONValue.object([
                    "user_id": .string(session.userID),
                    "expected_revision": .number(Double(snapshot.revision)),
                    "draft": draft
                ])
            )
        } catch { errorMessage = error.localizedDescription }
    }

    private func pollExport(jobID: String) async {
        while !Task.isCancelled {
            do {
                let status: JobStatus = try await api.request("/v1/jobs/\(jobID)")
                exportJob = status
                if ["finished", "failed", "canceled"].contains(status.state) { break }
            } catch {
                errorMessage = error.localizedDescription
                break
            }
            try? await Task.sleep(for: .seconds(2))
        }
    }
}

#if DEBUG
extension DraftEditorViewModel {
    /// Design screenshots only: show sample voice-over placements without a server.
    func setPreviewComposition(_ composition: TimelineComposition) {
        timelineComposition = composition
    }
}
#endif
