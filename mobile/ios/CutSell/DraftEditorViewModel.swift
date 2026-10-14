import AVFoundation
import Foundation

/// Carries one result out of a queued edit.
@MainActor
final class ResultBox {
    var value = false
}

@MainActor
final class DraftEditorViewModel: ObservableObject {
    @Published var snapshot: DraftSnapshot? {
        didSet {
            // Kept on the phone so the editor opens at once next time.
            if let snapshot, snapshot.revision != oldValue?.revision || oldValue == nil {
                Task.detached(priority: .utility) { DraftSnapshotCache.save(snapshot) }
            }
        }
    }
    /// Bumped when the server's draft arrives after a cached one was shown (fresh video links).
    @Published private(set) var loadGeneration = 0
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
    /// Saves that met a 409 and were retried quietly (shown by the editor probe).
    @Published private(set) var saveConflicts = 0
    /// Edits that no longer applied to the latest draft (their clip was undone or deleted); skipped.
    @Published private(set) var skippedEdits = 0
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
        await perform(predict: change) { [weak self] in _ = await self?.saveChange(change) }.value
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

    // MARK: Overlays, full-screen B-roll, one undo history

    /// Photos / videos picked for an overlay that are still uploading (shown at once).
    @Published private(set) var pendingOverlays: [PendingOverlay] = []
    private var overlayUploads: Set<String> = []
    /// The overlay just added (the editor selects it).
    @Published var lastAddedOverlayID: String?
    /// A full-screen B-roll video going into the main track (upload %).
    @Published private(set) var pendingBroll: PendingBroll?
    /// What Export is doing before the server takes over ("Saving your changes…").
    @Published var exportPreparing: String?

    /// One Undo / Redo history across the draft (clips, captions, overlays...) and the voice row.
    enum UndoStep { case draft; case voices(TimelineComposition) }
    private var undoSteps: [UndoStep] = []
    private var redoSteps: [UndoStep] = []

    // MARK: Overlay: photo or video from the phone

    /// A photo or video picked in the gallery goes over the video at `start`: on screen at once
    /// (from the copy kept on this iPhone), uploaded in the background with its %, and placed on
    /// the overlay row. A failed upload keeps the copy and tries again.
    func addOverlay(fileURL: URL, at start: Double) {
        let id = "pov_\(UUID().uuidString.prefix(10))"
        guard let name = OverlayMediaStore.keep(fileURL, id: id) else {
            errorMessage = "There isn't enough space on this iPhone for that file."
            return
        }
        let kind = EditorMediaInfo.kind(of: fileURL)
        Task { @MainActor in
            let local = OverlayMediaStore.fileURL(name)
            let length = kind == .video ? await EditorMediaInfo.duration(of: local) : 0
            let total = max(1, timelineDuration)
            let begin = min(max(0, start), max(0, total - 0.5))
            let end = min(total, begin + (kind == .video ? max(0.5, length) : 4))
            var item = PendingOverlay(id: id, fileName: name, kind: kind, start: begin, end: end,
                                      sourceDuration: kind == .video ? length : nil)
            let aspect = await EditorMediaInfo.aspect(of: local, kind: kind)
            // Tall media a little narrower so it never covers the whole frame.
            item.width = aspect < 0.8 ? 0.34 : 0.42
            pendingOverlays.append(item)
            lastAddedOverlayID = id
            uploadPendingOverlay(id: id)
        }
    }

    func retryPendingOverlay(id: String) { uploadPendingOverlay(id: id) }

    func discardPendingOverlay(id: String) {
        pendingOverlays.removeAll { $0.id == id }
    }

    /// Moves / resizes an overlay that is still uploading (kept for when it is placed).
    func updatePendingOverlay(id: String, x: Double? = nil, y: Double? = nil, width: Double? = nil,
                              start: Double? = nil, end: Double? = nil) {
        guard let index = pendingOverlays.firstIndex(where: { $0.id == id }) else { return }
        if let x { pendingOverlays[index].x = x }
        if let y { pendingOverlays[index].y = y }
        if let width { pendingOverlays[index].width = width }
        if let start { pendingOverlays[index].start = start }
        if let end { pendingOverlays[index].end = end }
    }

    private func uploadPendingOverlay(id: String) {
        guard !overlayUploads.contains(id), pendingOverlays.contains(where: { $0.id == id }) else { return }
        overlayUploads.insert(id)
        if let index = pendingOverlays.firstIndex(where: { $0.id == id }) { pendingOverlays[index].failed = false }
        Task { @MainActor [weak self] in
            guard let self else { return }
            defer { self.overlayUploads.remove(id) }
            for attempt in 0..<PendingVoiceOverStore.attemptsPerRound {
                let wait = PendingVoiceOverStore.retryDelay(afterAttempt: attempt)
                if wait > 0 { try? await Task.sleep(nanoseconds: UInt64(wait * 1_000_000_000)) }
                guard let item = self.pendingOverlays.first(where: { $0.id == id }) else { return }
                if await self.finishPendingOverlay(item) { return }
            }
            if let index = self.pendingOverlays.firstIndex(where: { $0.id == id }) { self.pendingOverlays[index].failed = true }
        }
    }

    private func setOverlayProgress(_ id: String, _ value: Double) {
        guard let index = pendingOverlays.firstIndex(where: { $0.id == id }) else { return }
        pendingOverlays[index].progress = max(pendingOverlays[index].progress, min(1, value))
    }

    private func finishPendingOverlay(_ item: PendingOverlay) async -> Bool {
        let uploaded: OverlayPresignResponse
        do {
            uploaded = try await OverlayUploadManager.shared.upload(
                fileURL: item.fileURL, projectID: project.projectID, session: session
            ) { [weak self] value in Task { @MainActor in self?.setOverlayProgress(item.id, value) } }
        } catch {
            CutSellDiagnostics.log("overlay_upload_retry", ["attempt": "\(item.attempts)"])
            return false
        }
        OverlayMediaStore.link(uri: uploaded.uri, fileName: item.fileName)
        // Where it is now (the creator may have moved it while it uploaded).
        let current = pendingOverlays.first { $0.id == item.id } ?? item
        let total = max(1, timelineDuration)
        var fields: [String: JSONValue] = [
            "project_id": .string(project.projectID), "user_id": .string(session.userID),
            "kind": .string(uploaded.kind), "uri": .string(uploaded.uri),
            "start": .number(min(current.start, max(0, total - 0.2))), "end": .number(min(current.end, total)),
            "x": .number(current.x), "y": .number(current.y), "width": .number(max(0.1, min(1, current.width))),
            "source_start": .number(0), "mute_audio": .bool(true),
        ]
        if let length = current.sourceDuration, length > 0 { fields["source_end"] = .number(length) }
        let shownError = errorMessage
        let added = ResultBox()
        let request = fields
        await perform(predict: nil) { [weak self] in
            added.value = await self?.editAndSave("/v1/overlays/add", request) ?? false
        }.value
        errorMessage = shownError
        guard added.value,
              let placed = mediaOverlays.last(where: { $0["uri"]?.stringValue == uploaded.uri })?["overlay_id"]?.stringValue
        else { return false }
        pendingOverlays.removeAll { $0.id == item.id }
        if lastAddedOverlayID == item.id { lastAddedOverlayID = placed }
        return true
    }

    /// Changes one overlay on the draft (shown at once, one save in the edit queue).
    @discardableResult
    private func changeOverlay(id: String, _ edit: @escaping (inout [String: JSONValue]) -> Void) -> Task<Void, Never> {
        let change: (inout [String: JSONValue]) -> Void = { object in
            let items = object["media_overlays"]?.arrayValue?.compactMap(\.objectValue) ?? []
            object["media_overlays"] = .array(items.map { item in
                guard item["overlay_id"]?.stringValue == id else { return .object(item) }
                var changed = item
                edit(&changed)
                return .object(changed)
            })
        }
        return perform(predict: change) { [weak self] in _ = await self?.saveChange(change) }
    }

    /// Where the overlay sits on the video (centre, 0...1) and how wide it is (share of the frame).
    @discardableResult
    func setOverlayFrame(id: String, x: Double, y: Double, width: Double) -> Task<Void, Never> {
        changeOverlay(id: id) { item in
            item["x"] = .number((min(1, max(0, x)) * 10_000).rounded() / 10_000)
            item["y"] = .number((min(1, max(0, y)) * 10_000).rounded() / 10_000)
            item["width"] = .number((min(1, max(0.1, width)) * 10_000).rounded() / 10_000)
        }
    }

    /// When the overlay shows on the timeline (move or trim). A video overlay trimmed at its start
    /// starts that much later in its own video.
    @discardableResult
    func setOverlayTime(id: String, start: Double, end: Double, sourceStart: Double? = nil) -> Task<Void, Never> {
        changeOverlay(id: id) { item in
            item["start"] = .number((max(0, start) * 1000).rounded() / 1000)
            item["end"] = .number((max(start + 0.2, end) * 1000).rounded() / 1000)
            if let sourceStart { item["source_start"] = .number((max(0, sourceStart) * 1000).rounded() / 1000) }
        }
    }

    /// Overlays (photo, video and text) end by the end of the cut; any wholly past it go.
    static func fitOverlaysToCut(_ draft: JSONValue) -> JSONValue {
        guard var object = draft.objectValue else { return draft }
        let clips = object["selected"]?.arrayValue?.compactMap(\.objectValue) ?? []
        let total = clips.reduce(0.0) { $0 + max(0, ($1["end"]?.doubleValue ?? 0) - ($1["start"]?.doubleValue ?? 0)) }
        guard total > 0 else { return draft }
        for key in ["media_overlays", "text_overlays"] {
            guard let items = object[key]?.arrayValue?.compactMap(\.objectValue), !items.isEmpty else { continue }
            var changed = false
            let fitted: [JSONValue] = items.compactMap { item in
                let start = item["start"]?.doubleValue ?? 0
                let end = item["end"]?.doubleValue ?? start
                if start >= total - 0.05 { changed = true; return nil }
                guard end > total else { return .object(item) }
                var copy = item
                copy["end"] = .number((total * 1000).rounded(.down) / 1000)
                changed = true
                return .object(copy)
            }
            if changed { object[key] = .array(fitted) }
        }
        return .object(object)
    }

    // MARK: Full-screen B-roll: a video from the phone as a clip of the main track

    /// The picked video goes into the main track at the cursor (between clips, or splitting the
    /// clip under the cursor), uploaded with its %; it plays at once from this iPhone.
    func addBrollClip(fileURL: URL, at time: Double) {
        let id = "broll_\(UUID().uuidString.prefix(8).lowercased())"
        let folder = LocalCaches.folder("BrollPicks")
        let local = folder.appendingPathComponent("\(id).\(fileURL.pathExtension.isEmpty ? "mp4" : fileURL.pathExtension.lowercased())")
        try? FileManager.default.removeItem(at: local)
        guard (try? FileManager.default.copyItem(at: fileURL, to: local)) != nil else {
            errorMessage = "There isn't enough space on this iPhone for that video."
            return
        }
        pendingBroll = PendingBroll(id: id, progress: 0)
        Task { @MainActor [weak self] in
            guard let self else { return }
            let length = await EditorMediaInfo.duration(of: local)
            guard length > 0.2 else {
                self.pendingBroll = nil
                self.errorMessage = "That video couldn't be read. Please pick another one."
                return
            }
            var uri: String?
            for attempt in 0..<PendingVoiceOverStore.attemptsPerRound where uri == nil {
                let wait = PendingVoiceOverStore.retryDelay(afterAttempt: attempt)
                if wait > 0 { try? await Task.sleep(nanoseconds: UInt64(wait * 1_000_000_000)) }
                let progress: @Sendable (Double) -> Void = { [weak self] value in
                    Task { @MainActor in
                        if let current = self?.pendingBroll, current.id == id {
                            self?.pendingBroll?.progress = max(current.progress, min(1, value))
                        }
                    }
                }
                let result = try? await MultipartUploadManager.shared.upload(
                    fileURL: local, projectID: self.project.projectID, session: self.session,
                    contentType: EditorMediaInfo.contentType(for: local), progress: progress
                )
                if let result { uri = result.sourceURI }
            }
            guard let uri else {
                self.pendingBroll?.failed = true
                self.brollRetry = (fileURL: local, time: time)
                return
            }
            await SourceVideoCache.shared.store(fileURL: local, uri: uri)
            let source: [String: JSONValue] = [
                "source_asset_id": .string("src_\(id)"), "original_name": .string(local.lastPathComponent),
                "uri": .string(uri), "duration_sec": .number((length * 1000).rounded() / 1000),
            ]
            let change: (inout [String: JSONValue]) -> Void = { object in
                Self.insertBroll(&object, id: id, source: source, length: length, at: time)
            }
            self.pendingBroll = nil
            self.brollRetry = nil
            await self.perform(predict: change) { [weak self] in _ = await self?.saveChange(change) }.value
        }
    }

    /// The last B-roll whose upload failed (tap to try again).
    private(set) var brollRetry: (fileURL: URL, time: Double)?

    func retryBroll() {
        guard let retry = brollRetry else { return }
        pendingBroll = nil
        addBrollClip(fileURL: retry.fileURL, at: retry.time)
    }

    func dismissFailedBroll() { pendingBroll = nil; brollRetry = nil }

    /// Puts a B-roll clip into `selected` at timeline `time` (splitting the clip under it).
    static func insertBroll(_ object: inout [String: JSONValue], id: String, source: [String: JSONValue],
                            length: Double, at time: Double) {
        var clips = object["selected"]?.arrayValue?.compactMap(\.objectValue) ?? []
        var sources = object["broll_sources"]?.arrayValue?.compactMap(\.objectValue) ?? []
        sources.append(source)
        object["broll_sources"] = .array(sources.map(JSONValue.object))
        let order = (clips.compactMap { $0["source_order"]?.doubleValue }.max() ?? 0) + 1
        let clip: [String: JSONValue] = [
            "clip_id": .string(id), "source_asset_id": source["source_asset_id"] ?? .string("src_\(id)"),
            "source_order": .number(order), "start": .number(0), "end": .number((length * 1000).rounded() / 1000),
            "text": .string(""), "caption_text": .string(""), "words": .array([]), "selected": .bool(true),
            "audio_muted": .bool(false), "audio_volume": .number(1),
        ]
        var cursor = 0.0
        var insertAt = clips.count
        for (index, item) in clips.enumerated() {
            let start = item["start"]?.doubleValue ?? 0
            let end = item["end"]?.doubleValue ?? start
            let clipLength = max(0, end - start)
            if time <= cursor + 0.05 { insertAt = index; break }
            if time < cursor + clipLength - 0.05 {
                // Inside this clip: split it at the cursor, B-roll in between.
                let at = start + (time - cursor)
                var left = item, right = item
                left["end"] = .number((at * 1000).rounded() / 1000)
                left.removeValue(forKey: "manual_end")
                right["start"] = .number((at * 1000).rounded() / 1000)
                right["clip_id"] = .string((item["clip_id"]?.stringValue ?? "clip") + "-\(id.suffix(4))")
                let words = item["words"]?.arrayValue?.compactMap(\.objectValue) ?? []
                let before = words.filter { ($0["end"]?.doubleValue ?? 0) <= at }
                let after = words.filter { ($0["end"]?.doubleValue ?? 0) > at }
                left["words"] = .array(before.map(JSONValue.object))
                right["words"] = .array(after.map(JSONValue.object))
                if !TrimMath.captionEditedByHand(item) {
                    left["text"] = .string(TrimMath.text(of: before)); left["caption_text"] = left["text"]
                    right["text"] = .string(TrimMath.text(of: after)); right["caption_text"] = right["text"]
                }
                clips.replaceSubrange(index...index, with: [left, clip, right])
                object["selected"] = .array(clips.map(JSONValue.object))
                return
            }
            cursor += clipLength
            insertAt = index + 1
        }
        clips.insert(clip, at: min(insertAt, clips.count))
        object["selected"] = .array(clips.map(JSONValue.object))
    }

    // MARK: Voice row: move a voice in time

    /// Moves a voice-over so it starts at `start` (its length kept); one save. A 409 (the row
    /// changed elsewhere) reads the row again and tries once more. Undo puts it back.
    func moveVoiceOver(id: String, to start: Double) async {
        for attempt in 0..<2 {
            guard let previous = timelineComposition,
                  let index = previous.voiceOverPlacements.firstIndex(where: { $0.placementID == id }) else { return }
            var placements = previous.voiceOverPlacements
            let length = placements[index].timelineEndSec - placements[index].timelineStartSec
            let begin = (max(0, start) * 1000).rounded() / 1000
            placements[index].timelineStartSec = begin
            placements[index].timelineEndSec = begin + length
            let shownError = errorMessage
            let saved = await saveTimelineComposition(
                brollPlacements: previous.brollPlacements, voiceOverPlacements: placements,
                timelineDurationSec: max(previous.timelineDurationSec, begin + length), capturePreviousForUndo: false
            )
            if saved {
                undoSteps.append(.voices(previous))
                redoSteps.removeAll()
                return
            }
            if attempt == 0 { errorMessage = shownError; await refreshTimelineComposition() }
        }
    }

    /// Puts the voice row back to `composition` (Undo / Redo of a voice change).
    private func restoreVoices(_ composition: TimelineComposition) async -> Bool {
        await saveTimelineComposition(
            brollPlacements: composition.brollPlacements, voiceOverPlacements: composition.voiceOverPlacements,
            timelineDurationSec: composition.timelineDurationSec, capturePreviousForUndo: false
        )
    }

    // MARK: Trim by the edges

    /// While a clip's handle is dragged: the original's frame to show on the video, and how much
    /// material the drag adds. `nil` otherwise. (Not part of the preview's rebuild key.)
    @Published var trimScrub: TrimScrub?
    /// Length of each original the phone measured itself (when the server sent none).
    @Published private(set) var knownSourceDurations: [String: Double] = [:]

    /// Moves one edge of a clip to `value` (seconds of its original). Shown at once; ONE save
    /// through the edit queue (quiet reload + retry on 409), so Undo / Redo work as usual.
    @discardableResult
    func trimClip(clipID: String, edge: TrimEdge, to value: Double) -> Task<Void, Never> {
        let change: (inout [String: JSONValue]) -> Void = { object in
            guard let clip = object["selected"]?.arrayValue?.compactMap(\.objectValue)
                    .first(where: { $0["clip_id"]?.stringValue == clipID }) else { return }
            let pool = TrimMath.wordPool(object, sourceID: clip["source_asset_id"]?.stringValue)
            Self.mapClips(&object) { clips in clips.map { item in
                guard item["clip_id"]?.stringValue == clipID else { return item }
                let start = edge == .start ? value : (item["start"]?.doubleValue ?? 0)
                let end = edge == .end ? value : (item["end"]?.doubleValue ?? start)
                guard end - start >= TrimMath.minLength - 1e-6 else { return item }
                return TrimMath.trimmed(item, start: start, end: end, edge: edge, pool: pool)
            } }
        }
        // The new size is shown right here (before this returns); the save runs in the queue.
        return perform(predict: change) { [weak self] in _ = await self?.saveChange(change) }
    }

    /// Measures the originals whose length the server did not send (for the trim's end limit).
    func measureSourceDurations() async {
        for source in snapshot?.sources.compactMap(\.objectValue) ?? [] {
            guard let id = source["source_asset_id"]?.stringValue, sourceDuration(forSource: id) == nil,
                  let url = previewVideoURL(forSource: id),
                  let time = try? await AVURLAsset(url: url).load(.duration) else { continue }
            let seconds = CMTimeGetSeconds(time)
            if seconds.isFinite, seconds > 0 { knownSourceDurations[id] = seconds }
        }
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
        await perform(predict: change) { [weak self] in _ = await self?.saveChange(change) }.value
    }

    /// Mirror / rotation are whole-video settings saved on the draft (the export reads them).
    func setDraftValue(_ key: String, _ value: JSONValue) async {
        let change: (inout [String: JSONValue]) -> Void = { $0[key] = value }
        await perform(predict: change) { [weak self] in _ = await self?.saveChange(change) }.value
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
        await perform(predict: change) { [weak self] in _ = await self?.saveChange(change) }.value
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
    /// Editor probe only: saves the draft on the server WITHOUT taking the answer, so this screen's
    /// revision is one behind -- the same state as a save whose answer was lost (the 409 report).
    func probeMakeLocalRevisionStale() async {
        guard let base = snapshot else { return }
        let _: DraftSnapshot? = try? await api.request(
            "/v1/projects/\(project.projectID)/draft",
            method: "PUT",
            body: JSONValue.object([
                "user_id": .string(session.userID),
                "expected_revision": .number(Double(base.revision)),
                "draft": base.draft
            ])
        )
    }

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
        // Open at once with the draft kept on this phone; the server's copy replaces it right after.
        let showedCached: Bool
        if snapshot == nil, let cached = DraftSnapshotCache.load(projectID: project.projectID) {
            snapshot = cached
            showedCached = true
        } else {
            showedCached = false
            isLoading = snapshot == nil
        }
        defer { isLoading = false }
        async let assetRequest = TimelineAssetRegistryClient.list(
            projectID: project.projectID,
            userID: session.userID,
            api: api
        )
        async let compositionRequest: Void = refreshTimelineComposition()
        do {
            let fresh: DraftSnapshot = try await api.request(
                "/v1/projects/\(project.projectID)/draft",
                query: [URLQueryItem(name: "user_id", value: session.userID)]
            )
            // Never step back behind a save made meanwhile.
            if fresh.revision >= (snapshot?.revision ?? 0) { snapshot = fresh }
            if showedCached { loadGeneration += 1 }
        } catch {
            if !showedCached { errorMessage = error.localizedDescription }
        }
        if let loadedAssets = try? await assetRequest { timelineAssetLibrary = loadedAssets }
        await compositionRequest
        // Voice-overs saved earlier that did not finish uploading: show them and try again.
        resumePendingVoiceOvers()
        await measureSourceDurations()
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

    // MARK: Voice-over saved at once, uploaded in the background

    /// Voice-overs saved on this iPhone that are still uploading (or waiting to retry). They show on
    /// the mic row and play in the preview right away.
    @Published private(set) var pendingVoiceOvers: [PendingVoiceOver] = []
    private var pendingUploads: Set<String> = []

    /// Every voice the preview plays: placed voice-overs whose recording is on this iPhone, then
    /// the ones still uploading (from their own file).
    var previewVoices: [PreviewVoice] {
        var voices: [PreviewVoice] = []
        for placement in timelineComposition?.voiceOverPlacements ?? [] {
            guard let url = VoiceOverStore.url(for: placement.assetID) else { continue }
            voices.append(PreviewVoice(key: placement.placementID, url: url, start: placement.timelineStartSec,
                                       sourceIn: placement.sourceInSec, sourceOut: placement.sourceOutSec))
        }
        for item in pendingVoiceOvers {
            voices.append(PreviewVoice(key: "pending_\(item.id)", url: PendingVoiceOverStore.fileURL(item),
                                       start: item.start, sourceIn: 0, sourceOut: item.duration))
        }
        return voices
    }

    /// A take recorded with the microphone. It goes on the mic row at once (kept on this iPhone),
    /// then uploads by the SAME path as an imported file (`/timeline-uploads` +
    /// `/timeline-assets`) and is placed on the voice-over track at `start` for its full length.
    /// `.failed` only when the recording itself can't be read or kept.
    func addRecordedVoiceOver(fileURL: URL, at start: Double) async -> RecordedVoiceOverOutcome {
        let duration = await PendingVoiceOverStore.duration(of: fileURL)
        guard duration > 0.05 else {
            errorMessage = "The recording couldn't be used. Please record again."
            return .failed
        }
        let id = "pvo_\(UUID().uuidString.prefix(12))"
        guard let fileName = PendingVoiceOverStore.keep(fileURL, id: id) else {
            errorMessage = "There isn't enough space on this iPhone to keep the recording."
            return .failed
        }
        // At the cursor, or right after the voice already there (voices never overlap on the row).
        let at = PendingVoiceOverStore.freeStart(start, duration: duration, taken: takenVoiceRanges(except: nil))
        let item = PendingVoiceOver(id: id, projectID: project.projectID, start: at,
                                    duration: duration, fileName: fileName, createdAt: Date())
        pendingVoiceOvers.append(item)
        PendingVoiceOverStore.save(pendingVoiceOvers, projectID: project.projectID)
        uploadPendingVoiceOver(id: id)
        return .placed
    }

    /// Uploads waiting recordings again (editor opened, or the creator tapped "retry").
    func resumePendingVoiceOvers() {
        if pendingVoiceOvers.isEmpty {
            pendingVoiceOvers = PendingVoiceOverStore.load(projectID: project.projectID)
        }
        for item in pendingVoiceOvers { uploadPendingVoiceOver(id: item.id) }
    }

    func retryPendingVoiceOver(id: String) { uploadPendingVoiceOver(id: id) }

    /// Timeline ranges already used on the mic row (placed voices and other waiting ones).
    private func takenVoiceRanges(except pendingID: String?) -> [(start: Double, end: Double)] {
        let placed = currentVoiceOverPlacements.map { (start: $0.timelineStartSec, end: $0.timelineEndSec) }
        let waiting = pendingVoiceOvers.filter { $0.id != pendingID }.map { (start: $0.start, end: $0.end) }
        return placed + waiting
    }

    func discardPendingVoiceOver(id: String) {
        guard let item = pendingVoiceOvers.first(where: { $0.id == id }) else { return }
        pendingVoiceOvers.removeAll { $0.id == id }
        PendingVoiceOverStore.save(pendingVoiceOvers, projectID: project.projectID)
        PendingVoiceOverStore.discardFile(item)
    }

    private func updatePending(_ id: String, _ change: (inout PendingVoiceOver) -> Void) {
        guard let index = pendingVoiceOvers.firstIndex(where: { $0.id == id }) else { return }
        change(&pendingVoiceOvers[index])
        PendingVoiceOverStore.save(pendingVoiceOvers, projectID: project.projectID)
    }

    /// One background upload per recording; a few tries with a short wait, then "tap to retry".
    private func uploadPendingVoiceOver(id: String) {
        guard !pendingUploads.contains(id), pendingVoiceOvers.contains(where: { $0.id == id }) else { return }
        pendingUploads.insert(id)
        updatePending(id) { $0.failed = false }
        Task { @MainActor [weak self] in
            guard let self else { return }
            defer { self.pendingUploads.remove(id) }
            for attempt in 0..<PendingVoiceOverStore.attemptsPerRound {
                let wait = PendingVoiceOverStore.retryDelay(afterAttempt: attempt)
                if wait > 0 { try? await Task.sleep(nanoseconds: UInt64(wait * 1_000_000_000)) }
                guard let item = self.pendingVoiceOvers.first(where: { $0.id == id }) else { return }
                self.updatePending(id) { $0.attempts += 1 }
                if await self.finishPendingVoiceOver(item) { return }
            }
            self.updatePending(id) { $0.failed = true }
        }
    }

    /// Upload + registration + placement of one waiting recording. `true` when it is done (placed,
    /// or kept in the library when the cut can't take placements yet).
    private func finishPendingVoiceOver(_ item: PendingVoiceOver) async -> Bool {
        let file = PendingVoiceOverStore.fileURL(item)
        let asset: TimelineMediaAsset
        do {
            asset = try await VoiceOverImportManager.shared.importAudio(
                fileURL: file, projectID: project.projectID, session: session, api: api
            )
        } catch {
            CutSellDiagnostics.log("voice_over_upload_retry", ["attempt": "\(item.attempts)"])
            return false
        }
        // Kept on this iPhone so the preview plays it on its own track.
        VoiceOverStore.save(file, assetID: asset.assetID)
        await refreshTimelineAssetLibrary()
        if baseEditAssetID != nil {
            let length = asset.durationSec > 0 ? asset.durationSec : item.duration
            // A background upload never pops an alert: a failure shows on the block (tap to retry).
            let shownError = errorMessage
            var placed = false
            for _ in 0..<2 where !placed {
                // Read the timeline again (another change may have moved it on: 409) and keep
                // the voice clear of the voices already on the row.
                await refreshTimelineComposition()
                let at = PendingVoiceOverStore.freeStart(item.start, duration: length, taken: takenVoiceRanges(except: item.id))
                placed = await addVoiceOverPlacement(
                    assetID: asset.assetID, start: at, end: at + length, sourceIn: 0, sourceOut: length
                )
            }
            errorMessage = shownError
            if !placed { return false }
        }
        pendingVoiceOvers.removeAll { $0.id == item.id }
        PendingVoiceOverStore.save(pendingVoiceOvers, projectID: project.projectID)
        PendingVoiceOverStore.discardFile(item)
        return true
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
        var body: [String: JSONValue] = [
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
        let fields = body
        await perform(predict: nil) { [weak self] in _ = await self?.editAndSave("/v1/overlays/add", fields) }.value
    }

    func updateMediaOverlay(
        overlayID: String, start: Double? = nil, end: Double? = nil,
        x: Double? = nil, y: Double? = nil, width: Double? = nil, muteAudio: Bool? = nil
    ) async {
        var body: [String: JSONValue] = [
            "overlay_id": .string(overlayID),
        ]
        if let start { body["start"] = .number(start) }
        if let end { body["end"] = .number(end) }
        if let x { body["x"] = .number(x) }
        if let y { body["y"] = .number(y) }
        if let width { body["width"] = .number(width) }
        if let muteAudio { body["mute_audio"] = .bool(muteAudio) }
        let fields = body
        await perform(predict: nil) { [weak self] in _ = await self?.editAndSave("/v1/overlays/update", fields) }.value
    }

    func removeMediaOverlay(overlayID: String) async {
        await perform(predict: nil) { [weak self] in _ = await self?.editAndSave("/v1/overlays/remove", [
            "overlay_id": .string(overlayID),
        ]) }.value
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
            errorMessage = "This cut isn't ready for that yet. Please try again in a moment."
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

    @discardableResult
    func addVoiceOverPlacement(assetID: String, start: Double, end: Double, sourceIn: Double, sourceOut: Double) async -> Bool {
        var placements = currentVoiceOverPlacements
        placements.append(VoiceOverPlacement(
            placementID: "vo_\(UUID().uuidString.prefix(8))", assetID: assetID,
            timelineStartSec: start, timelineEndSec: end,
            sourceInSec: sourceIn, sourceOutSec: sourceOut, transcriptReference: nil
        ))
        return await saveTimelineComposition(
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
        let previous = timelineComposition
        let placements = currentVoiceOverPlacements.filter { $0.placementID != id }
        let saved = await saveTimelineComposition(
            brollPlacements: currentBrollPlacements, voiceOverPlacements: placements,
            timelineDurationSec: currentTimelineDuration, capturePreviousForUndo: true
        )
        if saved, let previous { undoSteps.append(.voices(previous)); redoSteps.removeAll() }
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
        let previous = timelineComposition
        let saved = await saveTimelineComposition(
            brollPlacements: currentBrollPlacements, voiceOverPlacements: placements,
            timelineDurationSec: currentTimelineDuration, capturePreviousForUndo: true
        )
        if saved, let previous { undoSteps.append(.voices(previous)); redoSteps.removeAll() }
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
        await perform(predict: nil) { [weak self] in _ = await self?.editAndSave("/v1/draft-edits/swap", [
            "selected_clip_id": .string(selectedClipID),
            "replacement_clip_id": .string(replacementClipID)
        ]) }.value
    }

    func remove(clipID: String) async {
        await perform(predict: { object in
            Self.mapClips(&object) { $0.filter { $0["clip_id"]?.stringValue != clipID } }
        }) { [weak self] in await self?.removeNow(clipID: clipID) }.value
    }

    private func removeNow(clipID: String) async {
        await editAndSave("/v1/draft-edits/remove", [
            "clip_id": .string(clipID)
        ])
    }

    func reorder(clipIDs: [String]) async {
        await reorderNow(clipIDs: clipIDs).value
    }

    /// New clip order, shown at once (the others make room); one save in the edit queue.
    @discardableResult
    func reorderNow(clipIDs: [String]) -> Task<Void, Never> {
        let predict: (inout [String: JSONValue]) -> Void = { object in
            Self.mapClips(&object) { clips in
                clipIDs.compactMap { id in clips.first { $0["clip_id"]?.stringValue == id } }
                    + clips.filter { !clipIDs.contains($0["clip_id"]?.stringValue ?? "") }
            }
        }
        let fields: [String: JSONValue] = ["ordered_clip_ids": .array(clipIDs.map(JSONValue.string))]
        return perform(predict: predict) { [weak self] in _ = await self?.editAndSave("/v1/draft-edits/reorder", fields) }
    }

    func trim(clipID: String, start: Double, end: Double) async {
        await perform(predict: nil) { [weak self] in _ = await self?.editAndSave("/v1/draft-edits/trim", [
            "clip_id": .string(clipID),
            "start": .number(start),
            "end": .number(end)
        ]) }.value
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
        // An end dragged by hand stays with the right piece (the left one now ends at the split).
        first.removeValue(forKey: "manual_end")
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
        await editAndSave("/v1/draft-edits/split", [
            "clip_id": .string(clipID),
            "split_time": .number(sourceTime)
        ])
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
        await editAndSave("/v1/draft-edits/captions", [
            "edits": .array([.object([
                "clip_id": .string(clipID),
                "text": .string(text)
            ])])
        ])
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
        // Tapping what is already chosen (or lifting a finger without moving) saves nothing.
        changes = changes.filter { key, value in
            guard let current = captionValue(key) else { return true }
            if case .number(let a) = current, case .number(let b) = value { return abs(a - b) > 0.0005 }
            return current != value
        }
        guard !changes.isEmpty else { return nil }

        captionOverrideCounter += 1
        let token = captionOverrideCounter
        for (key, value) in changes {
            captionOverrides[key] = value
            captionOverrideTokens[key] = token
        }

        let changedKeys = Array(changes.keys)
        let changed = changes
        return perform(predict: nil) { [weak self] in
            guard let self else { return }
            await self.sendCaptionSettings(
                enabled: changed["captions_enabled"]?.boolValue, preset: changed["caption_preset"]?.stringValue,
                font: changed["caption_font"]?.stringValue, x: changed["caption_x"]?.doubleValue,
                y: changed["caption_y"]?.doubleValue, scale: changed["caption_scale"]?.doubleValue
            )
            for key in changedKeys where self.captionOverrideTokens[key] == token {
                self.captionOverrides[key] = nil
                self.captionOverrideTokens[key] = nil
            }
        }
    }

    private func sendCaptionSettings(
        enabled: Bool?, preset: String?, font: String?, x: Double?, y: Double?, scale: Double?
    ) async {
        var object: [String: JSONValue] = [:]
        object["enabled"] = enabled.map(JSONValue.bool) ?? .null
        object["preset"] = preset.map(JSONValue.string) ?? .null
        object["font"] = font.map(JSONValue.string) ?? .null
        object["x"] = x.map(JSONValue.number) ?? .null
        object["y"] = y.map(JSONValue.number) ?? .null
        object["scale"] = scale.map(JSONValue.number) ?? .null
        await editAndSave("/v1/draft-edits/caption-settings", object)
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
        var object: [String: JSONValue] = ["clip_id": .string(clipID)]
        object["muted"] = muted.map(JSONValue.bool) ?? .null
        object["volume"] = volume.map(JSONValue.number) ?? .null
        await editAndSave("/v1/draft-edits/audio", object)
    }

    func addTextOverlay(text: String, start: Double = 0, end: Double? = nil) async {
        let resolvedEnd = max(start + 0.15, min(end ?? max(1, timelineDuration), max(1, timelineDuration)))
        await perform(predict: nil) { [weak self] in _ = await self?.editAndSave("/v1/draft-edits/text/add", [
            "text": .string(text),
            "start": .number(start),
            "end": .number(resolvedEnd),
            "x": .number(0.5),
            "y": .number(0.2),
            "scale": .number(1.0)
        ]) }.value
    }

    func updateTextOverlay(id: String, text: String? = nil, x: Double? = nil, y: Double? = nil, scale: Double? = nil) async {
        var body: [String: JSONValue] = ["overlay_id": .string(id)]
        body["text"] = text.map(JSONValue.string) ?? .null
        body["x"] = x.map(JSONValue.number) ?? .null
        body["y"] = y.map(JSONValue.number) ?? .null
        body["scale"] = scale.map(JSONValue.number) ?? .null
        let fields = body
        await perform(predict: nil) { [weak self] in _ = await self?.editAndSave("/v1/draft-edits/text/update", fields) }.value
    }

    func removeTextOverlay(id: String) async {
        await perform(predict: { object in
            let items = object["text_overlays"]?.arrayValue?.compactMap(\.objectValue) ?? []
            object["text_overlays"] = .array(items.filter { $0["overlay_id"]?.stringValue != id }.map(JSONValue.object))
        }) { [weak self] in await self?.removeTextOverlayNow(id: id) }.value
    }

    private func removeTextOverlayNow(id: String) async {
        await editAndSave("/v1/draft-edits/text/remove", [
            "overlay_id": .string(id)
        ])
    }

    func addMediaOverlay(fileURL: URL, start: Double = 0, end: Double? = nil) async {
        do {
            let uploaded = try await OverlayUploadManager.shared.upload(
                fileURL: fileURL,
                projectID: project.projectID,
                session: session
            )
            let resolvedEnd = max(start + 0.15, min(end ?? max(1, timelineDuration), max(1, timelineDuration)))
            await addMediaOverlay(kind: uploaded.kind, uri: uploaded.uri, start: start, end: resolvedEnd)
        } catch {
            errorMessage = Self.friendly(error)
        }
    }

    func updateMediaOverlay(id: String, x: Double? = nil, y: Double? = nil, width: Double? = nil, muted: Bool? = nil) async {
        var body: [String: JSONValue] = ["overlay_id": .string(id)]
        body["x"] = x.map(JSONValue.number) ?? .null
        body["y"] = y.map(JSONValue.number) ?? .null
        body["width"] = width.map(JSONValue.number) ?? .null
        body["mute_audio"] = muted.map(JSONValue.bool) ?? .null
        let fields = body
        await perform(predict: nil) { [weak self] in _ = await self?.editAndSave("/v1/overlays/update", fields) }.value
    }

    func removeMediaOverlay(id: String) async {
        await perform(predict: { object in
            let items = object["media_overlays"]?.arrayValue?.compactMap(\.objectValue) ?? []
            object["media_overlays"] = .array(items.filter { $0["overlay_id"]?.stringValue != id }.map(JSONValue.object))
        }) { [weak self] in await self?.removeMediaOverlayNow(id: id) }.value
    }

    private func removeMediaOverlayNow(id: String) async {
        await editAndSave("/v1/overlays/remove", [
            "overlay_id": .string(id)
        ])
    }

    /// Returns whether the real `/draft/undo` call actually succeeded --
    /// callers (e.g. TimelineEditorView's Redo gating) must never infer
    /// success merely from `await` returning; a caught error here means
    /// `snapshot` was left exactly as it was and this returns `false`.
    @discardableResult
    func undo() async -> Bool {
        // The last change was on the voice row: put that row back as it was.
        if case .voices(let previous)? = undoSteps.last {
            undoSteps.removeLast()
            let current = timelineComposition
            if await restoreVoices(previous) {
                if let current { redoSteps.append(.voices(current)) }
                return true
            }
            undoSteps.append(.voices(previous))
            return false
        }
        if case .draft? = undoSteps.last { undoSteps.removeLast() }
        redoSteps.append(.draft)
        let result = ResultBox()
        var predict: ((inout [String: JSONValue]) -> Void)?
        if let restored = confirmedHistory.last?.objectValue { predict = { $0 = restored } }
        await perform(predict: predict) { [weak self] in result.value = await self?.undoNow() ?? false }.value
        return result.value
    }

    private func undoNow() async -> Bool {
        let done = await history("undo")
        if done, !confirmedHistory.isEmpty { confirmedHistory.removeLast() }
        return done
    }

    /// Same honest success/failure contract as `undo()` -- see above.
    @discardableResult
    func redo() async -> Bool {
        if case .voices(let next)? = redoSteps.last {
            redoSteps.removeLast()
            let current = timelineComposition
            if await restoreVoices(next) {
                if let current { undoSteps.append(.voices(current)) }
                return true
            }
            redoSteps.append(.voices(next))
            return false
        }
        if case .draft? = redoSteps.last { redoSteps.removeLast() }
        undoSteps.append(.draft)
        let result = ResultBox()
        await perform(predict: nil) { [weak self] in result.value = await self?.redoNow() ?? false }.value
        return result.value
    }

    private func redoNow() async -> Bool {
        await history("redo")
    }

    /// Every source the cut uses: the uploaded originals, plus full-screen B-roll videos added in
    /// the editor (kept in the draft's `broll_sources`).
    var allSources: [[String: JSONValue]] {
        let base = snapshot?.sources.compactMap(\.objectValue) ?? []
        let known = Set(base.compactMap { $0["source_asset_id"]?.stringValue })
        let added = (displayDraft?["broll_sources"]?.arrayValue?.compactMap(\.objectValue) ?? [])
            .filter { !known.contains($0["source_asset_id"]?.stringValue ?? "") }
        return base + added
    }

    /// Export starts as soon as it is tapped: it first lets the last edits and uploads finish
    /// (so the video has everything on screen), then the server renders it.
    func export() async {
        let started = Date()
        exportJob = nil
        exportPreparing = "Saving your changes…"
        _ = await actionChain?.value
        var waited = 0.0
        while (!pendingVoiceOvers.filter { !$0.failed }.isEmpty || !pendingOverlays.filter { !$0.failed }.isEmpty
               || (pendingBroll.map { !$0.failed } ?? false)), waited < 90 {
            exportPreparing = "Uploading your media…"
            try? await Task.sleep(nanoseconds: 500_000_000)
            waited += 0.5
        }
        _ = await actionChain?.value
        let preparedSeconds = Date().timeIntervalSince(started)
        guard let snapshot else { exportPreparing = nil; return }
        exportPreparing = "Starting the export…"
        do {
            let sources = allSources.compactMap { object -> JSONValue? in
                guard let sourceID = object["source_asset_id"], let uri = object["uri"] else { return nil }
                return .object(["source_asset_id": sourceID,
                                "original_name": object["original_name"] ?? .string("video.mp4"), "uri": uri])
            }
            // Nothing changed since the last export of this cut: that video, at once.
            let key = ExportTracker.contentKey(draft: snapshot.draft, sources: sources)
            if let done = ExportTracker.shared.reusable(projectID: project.projectID, key: key) {
                exportPreparing = nil
                exportJob = done
                NSLog("CUTSELLEXPORT reused %.1f", Date().timeIntervalSince(started))
                return
            }
            let response: ExportSubmitResponse = try await api.request(
                "/v1/exports/jobs",
                method: "POST",
                body: JSONValue.object([
                    "project_id": .string(project.projectID),
                    "user_id": .string(session.userID),
                    "draft": snapshot.draft,
                    "sources": .array(sources)
                ])
            )
            exportPreparing = nil
            ExportTracker.shared.remember(jobID: response.jobID, projectID: project.projectID, key: key)
            ExportTracker.shared.follow(jobID: response.jobID, projectID: project.projectID,
                                        title: project.title, preparedSeconds: preparedSeconds,
                                        submittedSeconds: Date().timeIntervalSince(started))
            await pollExport(jobID: response.jobID)
        } catch {
            exportPreparing = nil
            errorMessage = Self.friendly(error)
        }
    }

    // MARK: One save path for every draft change
    //
    // Every change builds its new draft from the server's latest draft, then saves it with ONE PUT
    // carrying that draft's revision; the answer becomes the local draft (so the next change uses
    // the new revision). Changes run one at a time (`perform`). If the server still says the
    // draft moved on (409: an earlier save whose answer was lost, another device...), the draft is
    // reloaded, the change is made again on top of it and saved once more -- with no alert. Only a
    // second failure in a row is shown, in plain words.

    private enum SaveOutcome { case saved, conflict, failed(Error) }

    static func isConflict(_ error: Error) -> Bool {
        if case APIError.http(409, _) = error { return true }
        return false
    }

    /// Plain words instead of "draft revision conflict ... (409)".
    /// What the creator reads for any message: plain words, never a server body, code or field name.
    static func plainWords(_ message: String?) -> String {
        guard let message, !message.isEmpty else { return "" }
        let technical = ["{", "}", "detail", "TIMELINE", "_id", "(4", "(5", "uri", "draft", "revision",
                         "placement", "asset", "Error Domain", "422", "409", "500", "503"]
        if technical.contains(where: { message.contains($0) }) { return "Something didn't save. Please try again." }
        return message
    }

    static func friendly(_ error: Error) -> String {
        if isConflict(error) {
            return "This cut was changed somewhere else. We reloaded it — please try your change again."
        }
        if let api = error as? APIError, case .http(let code, _) = api, code >= 500 {
            return "The server is busy. Please try again in a moment."
        }
        if (error as NSError).domain == NSURLErrorDomain {
            return "No connection. Check your internet and try again."
        }
        return "Couldn’t save this change. Please try again."
    }

    private func reloadDraft() async -> Bool {
        do {
            snapshot = try await api.request(
                "/v1/projects/\(project.projectID)/draft",
                query: [URLQueryItem(name: "user_id", value: session.userID)]
            )
            return true
        } catch {
            return false
        }
    }

    private func put(_ draft: JSONValue, over base: DraftSnapshot) async -> SaveOutcome {
        do {
            let saved: DraftSnapshot = try await api.request(
                "/v1/projects/\(project.projectID)/draft",
                method: "PUT",
                body: JSONValue.object([
                    "user_id": .string(session.userID),
                    "expected_revision": .number(Double(base.revision)),
                    "draft": draft
                ])
            )
            snapshot = saved
            return .saved
        } catch {
            return Self.isConflict(error) ? .conflict : .failed(error)
        }
    }

    /// `build` makes the edited draft from `base` (a server edit route or a local change);
    /// nil = nothing to save. Returns whether the change is saved.
    @discardableResult
    private func commit(_ build: @escaping (JSONValue) async throws -> JSONValue?) async -> Bool {
        for attempt in 0..<2 {
            guard let base = snapshot else { return false }
            let built: JSONValue?
            do {
                built = try await build(base.draft)
            } catch {
                // The edit route refused this change on the latest draft (409/422: e.g. the clip it
                // was for was just undone or deleted). Nothing is lost that is still in the cut, and
                // it is not a save failure: skip it quietly instead of showing an error.
                if case APIError.http(let code, _) = error, code == 409 || code == 422 {
                    skippedEdits += 1
                    return false
                }
                errorMessage = Self.friendly(error)
                return false
            }
            guard let built else { return false }
            // Overlays never run past the end of the cut (the server refuses such a draft).
            let edited = Self.fitOverlaysToCut(built)
            if edited == base.draft { return true }   // nothing changed: nothing to save
            isSaving = true
            let outcome = await put(edited, over: base)
            isSaving = false
            switch outcome {
            case .saved:
                confirmedHistory.append(base.draft)
                if confirmedHistory.count > 20 { confirmedHistory.removeFirst() }
                undoSteps.append(.draft)
                redoSteps.removeAll()
                return true
            case .conflict:
                saveConflicts += 1
                if attempt == 0, await reloadDraft() { continue }
            case .failed(let error):
                errorMessage = Self.friendly(error)
                return false
            }
        }
        _ = await reloadDraft()
        errorMessage = Self.friendly(APIError.http(409, ""))
        return false
    }

    /// A server edit route (`/v1/draft-edits/...`, `/v1/overlays/...`) applied to the latest draft, then saved.
    @discardableResult
    private func editAndSave(_ path: String, _ fields: [String: JSONValue]) async -> Bool {
        await commit { [api] base in
            var body = fields
            body["draft"] = base
            let edited: JSONValue = try await api.request(path, method: "POST", body: JSONValue.object(body))
            return edited
        }
    }

    /// A local change applied to the latest draft, then saved.
    @discardableResult
    private func saveChange(_ change: @escaping (inout [String: JSONValue]) -> Void) async -> Bool {
        await commit { base in
            guard var object = base.objectValue else { return nil }
            change(&object)
            return .object(object)
        }
    }

    /// Undo / redo on the server, with the same quiet reload-and-retry after a 409.
    private func history(_ action: String) async -> Bool {
        for attempt in 0..<2 {
            guard let base = snapshot else { return false }
            do {
                snapshot = try await api.request(
                    "/v1/projects/\(project.projectID)/draft/\(action)",
                    method: "POST",
                    body: JSONValue.object([
                        "user_id": .string(session.userID),
                        "expected_revision": .number(Double(base.revision))
                    ])
                )
                return true
            } catch {
                if Self.isConflict(error), attempt == 0, await reloadDraft() { saveConflicts += 1; continue }
                if case APIError.http(let code, _) = error, code == 409 || code == 400 || code == 422 {
                    // Nothing (left) to undo / redo on the server: not worth an alert.
                    _ = await reloadDraft()
                    return false
                }
                errorMessage = Self.friendly(error)
                return false
            }
        }
        return false
    }

    /// Shows the export's progress while this editor is open (the app-wide tracker keeps
    /// following it, and tells the creator when it is ready, after the editor closes).
    private func pollExport(jobID: String) async {
        while !Task.isCancelled {
            if let status = ExportTracker.shared.jobs[jobID] {
                exportJob = status
                if ["finished", "failed", "canceled"].contains(status.state) { break }
            }
            try? await Task.sleep(for: .milliseconds(500))
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
