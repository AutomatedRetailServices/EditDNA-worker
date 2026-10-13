import AVFoundation
import SwiftUI
import UIKit

/// Thumbnails for the Edits cards and Recent cuts, cached on this iPhone (one small JPEG per cut).
/// A cut made on this iPhone uses the first frame captured when it started; any other ready cut
/// gets a frame from its first kept clip (the draft's signed video link).
@MainActor
final class ThumbnailStore: ObservableObject {
    static let shared = ThumbnailStore()

    @Published private(set) var images: [String: UIImage] = [:]
    /// Read from disk during drawing; kept out of `images` so drawing never publishes changes.
    private var diskCache: [String: UIImage] = [:]
    private var loading: Set<String> = []

    private var folder: URL {
        let root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell/Thumbnails", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func file(_ projectID: String) -> URL {
        folder.appendingPathComponent("\(projectID).jpg")
    }

    func save(_ image: UIImage, for projectID: String) {
        images[projectID] = image
        if let data = image.jpegData(compressionQuality: 0.8) {
            try? data.write(to: file(projectID), options: .atomic)
        }
    }

    /// The cached thumbnail, or nil (call `load` to fetch one for a ready cut).
    func image(for projectID: String) -> UIImage? {
        if let image = images[projectID] ?? diskCache[projectID] { return image }
        if let data = try? Data(contentsOf: file(projectID)), let image = UIImage(data: data) {
            diskCache[projectID] = image
            return image
        }
        return nil
    }

    /// For a ready cut without a thumbnail: one frame from the start of its first kept clip.
    func load(project: Project, session: CutSellSession?) async {
        guard project.state == "draft_ready", image(for: project.projectID) == nil,
              !loading.contains(project.projectID), let session else { return }
        loading.insert(project.projectID)
        defer { loading.remove(project.projectID) }
        guard let snapshot: DraftSnapshot = try? await APIClient.shared.request(
            "/v1/projects/\(project.projectID)/draft",
            query: [URLQueryItem(name: "user_id", value: session.userID)]
        ) else { return }
        let first = snapshot.draft["selected"]?.arrayValue?.first?.objectValue
        let sourceID = first?["source_asset_id"]?.stringValue
        let start = first?["start"]?.doubleValue ?? 0
        let sources = snapshot.sources.compactMap(\.objectValue)
        let source = sources.first { $0["source_asset_id"]?.stringValue == sourceID } ?? sources.first
        guard let raw = source?["playback_url"]?.stringValue, let url = URL(string: raw) else { return }
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 300, height: 540)
        let time = CMTime(seconds: max(0, start + 0.2), preferredTimescale: 600)
        guard let frame = try? await generator.image(at: time).image else { return }
        save(UIImage(cgImage: frame), for: project.projectID)
    }
}
