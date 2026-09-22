import Foundation

@MainActor
final class DraftEditorViewModel: ObservableObject {
    @Published var snapshot: DraftSnapshot?
    @Published var isLoading = false
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
    }

    var selectedClips: [[String: JSONValue]] {
        snapshot?.draft["selected"]?.arrayValue?.compactMap(\.objectValue) ?? []
    }

    var alternateClips: [[String: JSONValue]] {
        snapshot?.draft["alternates"]?.arrayValue?.compactMap(\.objectValue) ?? []
    }

    var textOverlays: [[String: JSONValue]] {
        snapshot?.draft["text_overlays"]?.arrayValue?.compactMap(\.objectValue) ?? []
    }

    var mediaOverlays: [[String: JSONValue]] {
        snapshot?.draft["media_overlays"]?.arrayValue?.compactMap(\.objectValue) ?? []
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

    var captionsEnabled: Bool {
        snapshot?.draft["captions_enabled"]?.boolValue ?? true
    }

    var captionPreset: String {
        snapshot?.draft["caption_preset"]?.stringValue ?? "classic"
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
            let loadedDraft = try await draftRequest
            snapshot = loadedDraft
            let loadedAssets = try await assetRequest
            timelineAssetLibrary = loadedAssets
        } catch {
            errorMessage = error.localizedDescription
        }
        await refreshTimelineComposition()
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

    /// Imports an already-existing local video file as a new
    /// SUPPLEMENTAL_BROLL asset via the real D-282A upload+registration
    /// routes (`OverlayImportManager`) -- never a fabricated capture/import
    /// path. Same confirmed-before-refresh discipline as
    /// `importVoiceOverAudio`.
    func importOverlayVideo(fileURL: URL) async {
        do {
            _ = try await OverlayImportManager.shared.importVideo(
                fileURL: fileURL, projectID: project.projectID, session: session, api: api
            )
            await refreshTimelineAssetLibrary()
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

    func split(clipID: String, at sourceTime: Double) async {
        guard let snapshot else { return }
        let edited = await edit(path: "/v1/draft-edits/split", body: .object([
            "draft": snapshot.draft,
            "clip_id": .string(clipID),
            "split_time": .number(sourceTime)
        ]))
        if let edited { await autosave(edited) }
    }

    func editCaption(clipID: String, text: String) async {
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

    func setCaptionSettings(enabled: Bool? = nil, preset: String? = nil) async {
        guard let snapshot else { return }
        var object: [String: JSONValue] = ["draft": snapshot.draft]
        object["enabled"] = enabled.map(JSONValue.bool) ?? .null
        object["preset"] = preset.map(JSONValue.string) ?? .null
        let edited = await edit(path: "/v1/draft-edits/caption-settings", body: .object(object))
        if let edited { await autosave(edited) }
    }

    func setAudio(clipID: String, muted: Bool? = nil, volume: Double? = nil) async {
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
        guard let snapshot else { return false }
        do {
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
