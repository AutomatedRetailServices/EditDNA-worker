import Foundation

struct PendingCutVideo: Codable, Hashable {
    let filePath: String
    let name: String
    let duration: Double
    let contentType: String

    var fileURL: URL { URL(fileURLWithPath: filePath) }
}

struct PendingCutRecord: Codable, Hashable, Identifiable {
    let projectID: String
    let title: String
    let mode: String
    let audioOverlap: Bool
    let videos: [PendingCutVideo]
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
            !record.videos.isEmpty && record.videos.allSatisfy { FileManager.default.fileExists(atPath: $0.filePath) }
        }
    }

    func record(projectID: String) -> PendingCutRecord? {
        list().first { $0.projectID == projectID }
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
