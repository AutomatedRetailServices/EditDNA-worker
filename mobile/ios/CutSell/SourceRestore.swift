import Foundation
import Photos

/// 22 Oct: the server removes a cut's light copy 30 days after its last use (the cut itself stays). Opening such a cut
/// makes the copy again on this iPhone from the original in Photos and uploads it to the same place
/// ("Restoring video…"); then the cut plays and exports as before. Without the original, it says so.
@MainActor
final class SourceRestore: ObservableObject {
    static let shared = SourceRestore()

    /// projectID → 0...1 while restoring.
    @Published private(set) var progress: [String: Double] = [:]
    /// projectID → what went wrong, in plain words.
    @Published private(set) var problem: [String: String] = [:]
    private var running: Set<String> = []

    func restoreIfNeeded(model: DraftEditorViewModel) async {
        let projectID = model.project.projectID
        let missing = (model.snapshot?.sources ?? []).compactMap { $0.objectValue }
            .filter { $0["available"]?.boolValue == false }
            .compactMap { $0["uri"]?.stringValue }
        guard !missing.isEmpty, !running.contains(projectID) else { return }
        running.insert(projectID)
        defer { running.remove(projectID); progress[projectID] = nil }
        progress[projectID] = 0.01
        problem[projectID] = nil
        CutSellDiagnostics.log("source_restore_started", ["sources": String(missing.count)])
        for (index, uri) in missing.enumerated() {
            let base = Double(index) / Double(missing.count)
            let share = 1 / Double(missing.count)
            guard let asset = OriginalLinks.asset(forURI: uri) else {
                problem[projectID] = "This video's original isn't in Photos anymore, so it can't be restored."
                CutSellDiagnostics.log("source_restore_failed", ["why": "no_original"])
                return
            }
            do {
                let original = try await GalleryExport.export(asset, name: nil, progress: { [weak self] value in
                    Task { @MainActor in self?.progress[projectID] = base + share * 0.3 * value }
                })
                let lighter = try await VideoPreparation.shared.prepare(fileURL: original.url, contentType: original.contentType) { [weak self] value in
                    Task { @MainActor in self?.progress[projectID] = base + share * (0.3 + 0.3 * value) }
                }
                await MultipartUploadManager.shared.setRestore(uri: uri, for: lighter.url)
                let result = try await MultipartUploadManager.shared.upload(
                    fileURL: lighter.url, projectID: projectID, session: model.session, contentType: lighter.contentType,
                    foreground: true
                ) { [weak self] value in
                    Task { @MainActor in self?.progress[projectID] = base + share * (0.6 + 0.4 * value) }
                }
                await SourceVideoCache.shared.replace(fileURL: lighter.url, uri: result.sourceURI)
                if lighter.url != original.url { try? FileManager.default.removeItem(at: original.url) }
            } catch {
                problem[projectID] = "The video couldn't be restored. Try again."
                CutSellDiagnostics.log("source_restore_failed", Retry.fields(error, ["why": "upload"]))
                return
            }
        }
        CutSellDiagnostics.log("source_restore_done", ["sources": String(missing.count)])
        await model.load()
    }
}
