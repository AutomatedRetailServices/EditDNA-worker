import Foundation

struct UploadResumeRecord: Codable {
    let identity: String
    let fileURL: String
    let start: MultipartStartResponse
    let updatedAt: Date
}

actor UploadResumeStore {
    static let shared = UploadResumeStore()

    private let defaults = UserDefaults.standard
    private let storageKey = "cutsell.multipart.resume.v1"
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    func record(fileURL: URL, projectID: String, size: Int64) -> UploadResumeRecord? {
        let identity = Self.identity(fileURL: fileURL, projectID: projectID, size: size)
        return loadAll()[identity]
    }

    func record(uploadID: String) -> UploadResumeRecord? {
        loadAll().values.first { $0.start.uploadID == uploadID }
    }

    func save(start: MultipartStartResponse, fileURL: URL, projectID: String, size: Int64) {
        var all = loadAll()
        let identity = Self.identity(fileURL: fileURL, projectID: projectID, size: size)
        all[identity] = UploadResumeRecord(
            identity: identity,
            fileURL: fileURL.path,
            start: start,
            updatedAt: Date()
        )
        persist(all)
    }

    func remove(fileURL: URL, projectID: String, size: Int64) {
        var all = loadAll()
        all.removeValue(forKey: Self.identity(fileURL: fileURL, projectID: projectID, size: size))
        persist(all)
    }

    func remove(uploadID: String) {
        var all = loadAll()
        for key in all.keys where all[key]?.start.uploadID == uploadID {
            all.removeValue(forKey: key)
        }
        persist(all)
    }

    private func loadAll() -> [String: UploadResumeRecord] {
        guard let data = defaults.data(forKey: storageKey),
              let decoded = try? decoder.decode([String: UploadResumeRecord].self, from: data) else {
            return [:]
        }
        return decoded.filter { FileManager.default.fileExists(atPath: $0.value.fileURL) }
    }

    private func persist(_ value: [String: UploadResumeRecord]) {
        if value.isEmpty {
            defaults.removeObject(forKey: storageKey)
        } else if let data = try? encoder.encode(value) {
            defaults.set(data, forKey: storageKey)
        }
    }

    private static func identity(fileURL: URL, projectID: String, size: Int64) -> String {
        "\(projectID)|\(size)|\(fileURL.lastPathComponent)"
    }
}
