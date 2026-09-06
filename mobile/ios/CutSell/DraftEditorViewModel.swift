import Foundation

@MainActor
final class DraftEditorViewModel: ObservableObject {
    @Published var snapshot: DraftSnapshot?
    @Published var isLoading = false
    @Published var isSaving = false
    @Published var exportJob: JobStatus?
    @Published var errorMessage: String?

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
            snapshot = try await api.request(
                "/v1/projects/\(project.projectID)/draft",
                query: [URLQueryItem(name: "user_id", value: session.userID)]
            )
        } catch {
            errorMessage = error.localizedDescription
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

    func undo() async {
        guard let snapshot else { return }
        do {
            self.snapshot = try await api.request(
                "/v1/projects/\(project.projectID)/draft/undo",
                method: "POST",
                body: JSONValue.object([
                    "user_id": .string(session.userID),
                    "expected_revision": .number(Double(snapshot.revision))
                ])
            )
        } catch { errorMessage = error.localizedDescription }
    }

    func redo() async {
        guard let snapshot else { return }
        do {
            self.snapshot = try await api.request(
                "/v1/projects/\(project.projectID)/draft/redo",
                method: "POST",
                body: JSONValue.object([
                    "user_id": .string(session.userID),
                    "expected_revision": .number(Double(snapshot.revision))
                ])
            )
        } catch { errorMessage = error.localizedDescription }
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
