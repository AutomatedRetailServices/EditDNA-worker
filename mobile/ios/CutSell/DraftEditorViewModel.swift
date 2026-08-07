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

    func remove(clipID: String) async {
        guard let snapshot else { return }
        let edited = await edit(
            path: "/v1/draft-edits/remove",
            body: .object([
                "draft": snapshot.draft,
                "clip_id": .string(clipID)
            ])
        )
        if let edited { await autosave(edited) }
    }

    func reorder(clipIDs: [String]) async {
        guard let snapshot else { return }
        let edited = await edit(
            path: "/v1/draft-edits/reorder",
            body: .object([
                "draft": snapshot.draft,
                "ordered_clip_ids": .array(clipIDs.map(JSONValue.string))
            ])
        )
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
        var object: [String: JSONValue] = [
            "draft": snapshot.draft,
            "clip_id": .string(clipID)
        ]
        object["muted"] = muted.map(JSONValue.bool) ?? .null
        object["volume"] = volume.map(JSONValue.number) ?? .null
        let edited = await edit(path: "/v1/draft-edits/audio", body: .object(object))
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
                return .object([
                    "source_asset_id": sourceID,
                    "original_name": originalName,
                    "uri": uri
                ])
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
