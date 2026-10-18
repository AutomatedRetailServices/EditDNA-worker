import Foundation

struct PendingCutVideo: Codable, Hashable {
    let filePath: String
    let name: String
    let duration: Double
    let contentType: String
    /// Build 23 ("iCloud light"): the original is still being downloaded from iCloud to filePath.
    var cloudAssetID: String? = nil
    var expectedBytes: Int64? = nil

    /// Build 22: found again after an app update moves CutSell's folder.
    var fileURL: URL { ContainerPath.resolve(filePath) }
}

struct PendingCutRecord: Codable, Hashable, Identifiable {
    let projectID: String
    let title: String
    let mode: String
    let audioOverlap: Bool
    /// Auto Zoom AI chosen in "Choose your style" (nil on records saved before it existed = on).
    var autoZoom: Bool? = nil
    let videos: [PendingCutVideo]
    /// Build 23 (audio first): the cut already started on the server from the voice; only the full
    /// video is still uploading (in the background). nil = nothing started yet ("Resume upload").
    var jobID: String? = nil
    let createdAt: Date

    var id: String { projectID }
}

actor PendingCutStore {
    static let shared = PendingCutStore()

    private var storeURL: URL {
        let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.appendingPathComponent("pending-cuts.json")
    }

    func list() -> [PendingCutRecord] {
        guard let data = try? Data(contentsOf: storeURL),
              let values = try? JSONDecoder().decode([PendingCutRecord].self, from: data) else {
            return []
        }
        return values.filter { record in
            !record.videos.isEmpty && record.videos.allSatisfy { FileManager.default.fileExists(atPath: $0.fileURL.path) || $0.cloudAssetID != nil }
        }
    }

    func record(projectID: String) -> PendingCutRecord? {
        list().first { $0.projectID == projectID }
    }

    /// Build 23: the cut started from the voice; the record stays until the video is uploaded.
    func markStarted(projectID: String, jobID: String) {
        var values = list()
        guard let index = values.firstIndex(where: { $0.projectID == projectID }) else { return }
        values[index].jobID = jobID
        persist(values)
    }

    func save(_ record: PendingCutRecord) {
        var values = list().filter { $0.projectID != record.projectID }
        values.insert(record, at: 0)
        persist(values)
    }

    func remove(projectID: String, deleteLocalFiles: Bool = false) {
        let current = list()
        if deleteLocalFiles, let record = current.first(where: { $0.projectID == projectID }) {
            for video in record.videos {
                try? FileManager.default.removeItem(at: video.fileURL)
            }
        }
        persist(current.filter { $0.projectID != projectID })
    }

    private func persist(_ values: [PendingCutRecord]) {
        guard let data = try? JSONEncoder().encode(values) else { return }
        try? data.write(to: storeURL, options: .atomic)
    }
}
