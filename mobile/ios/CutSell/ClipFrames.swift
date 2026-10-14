import AVFoundation
import SwiftUI
import UIKit

/// Real frames for clip thumbnails in the editor, taken from the cut's own video (the draft's
/// signed playback link) when the server sent no preview frames. Cached in memory.
actor ClipFrameCache {
    static let shared = ClipFrameCache()
    private let cache = NSCache<NSString, UIImage>()
    private var generators: [String: AVAssetImageGenerator] = [:]

    func frame(url: URL, at seconds: Double) async -> UIImage? {
        // The link's signature changes; the file path does not.
        let key = "\(url.path)|\(Int((seconds * 10).rounded()))" as NSString
        if let hit = cache.object(forKey: key) { return hit }
        // One opened video per file for all its frames (each frame used to open the video again).
        let generator: AVAssetImageGenerator
        if let existing = generators[url.path] {
            generator = existing
        } else {
            generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: 180, height: 320)
            generator.requestedTimeToleranceBefore = CMTime(seconds: 0.5, preferredTimescale: 600)
            generator.requestedTimeToleranceAfter = CMTime(seconds: 0.5, preferredTimescale: 600)
            generators[url.path] = generator
        }
        guard let cg = try? await generator.image(at: CMTime(seconds: max(0, seconds), preferredTimescale: 600)).image else {
            // A request cut short (the view went away) is not a broken video: keep the opened one.
            // Otherwise the link may have expired: open it again next time.
            if !Task.isCancelled { generators[url.path] = nil }
            return nil
        }
        let image = UIImage(cgImage: cg)
        cache.setObject(image, forKey: key)
        return image
    }
}

/// One real frame of a clip (falls back to a plain tile while it loads or if there is no link).
struct ClipFrameImage: View {
    let url: URL?
    let seconds: Double
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Rectangle().fill(Color.secondary.opacity(0.15))
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            }
        }
        .clipped()
        .task(id: "\(url?.path ?? "")|\(seconds)") {
            guard let url else { return }
            // Frames must always be there: keep the one shown while a new one loads, and try
            // again a few times if the video could not give it (build 14: selecting a voice-over
            // left the video row without frames).
            for attempt in 0..<4 {
                if Task.isCancelled { return }
                if let frame = await ClipFrameCache.shared.frame(url: url, at: seconds) {
                    image = frame
                    return
                }
                try? await Task.sleep(nanoseconds: UInt64(400_000_000 * (attempt + 1)))
            }
        }
    }
}

extension DraftEditorViewModel {
    /// The signed playback link of one source of this draft.
    func playbackURL(forSource sourceID: String?) -> URL? {
        let sources = allSources
        // Never another video's link for a clip whose own source is not known here.
        let source = sources.first { $0["source_asset_id"]?.stringValue == sourceID }
            ?? (sourceID == nil ? sources.first : nil)
        return source?["playback_url"]?.stringValue.flatMap(URL.init(string:))
    }

    /// The video's own name (instead of "Untitled Cut"), without the extension.
    var videoTitle: String? {
        let names = (snapshot?.sources ?? []).compactMap { $0.objectValue?["original_name"]?.stringValue }
        guard let first = names.first, !first.isEmpty else { return nil }
        let base = (first as NSString).deletingPathExtension
        return names.count > 1 ? "\(base) +\(names.count - 1)" : base
    }
}
