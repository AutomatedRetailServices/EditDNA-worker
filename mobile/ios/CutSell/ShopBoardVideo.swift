@preconcurrency import AVFoundation
import CoreImage
import SwiftUI
import UIKit
import UniformTypeIdentifiers

// Wear it with a VIDEO background (PO 14 Nov 2026, Figma W1 3065:201 · W2 3065:301 · W4 3065:383).
// Her short video (up to 12 s, with its sound) plays behind the products, the title and her brand line, all there from
// the first frame. The board is shared as a 9:16 MP4: the board without its background (products, texts, the shade)
// is drawn once as a transparent picture and laid over every frame of her video.
enum ShopVideoBackground {
    /// PO 14 Nov: "hasta 12 seg". A longer video uses its first 12 s.
    static let maxSeconds: Double = 12

    struct Loaded {
        let url: URL
        let poster: UIImage
        let duration: Double
    }

    static func isVideo(_ provider: NSItemProvider) -> Bool {
        provider.hasItemConformingToTypeIdentifier(UTType.movie.identifier)
    }

    /// Where the picked videos wait until the board is saved (the board's folder keeps its own copy).
    static var workFolder: URL {
        let root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ShopBoardVideos", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    /// The picked video copied here, its first picture and its length (nil when it cannot be read).
    static func load(_ provider: NSItemProvider) async -> Loaded? {
        let copy: URL? = await withCheckedContinuation { continuation in
            provider.loadFileRepresentation(forTypeIdentifier: UTType.movie.identifier) { file, _ in
                guard let file else { continuation.resume(returning: nil); return }
                let ext = file.pathExtension.isEmpty ? "mov" : file.pathExtension
                let destination = workFolder.appendingPathComponent("\(UUID().uuidString).\(ext)")
                do {
                    try FileManager.default.copyItem(at: file, to: destination)       // the picker's file is temporary
                    continuation.resume(returning: destination)
                } catch {
                    continuation.resume(returning: nil)
                }
            }
        }
        guard let copy else {
            CutSellDiagnostics.log("shop_video_failed", ["step": "copy"])
            return nil
        }
        return await describe(copy)
    }

    /// First picture + length of a video already on the phone.
    static func describe(_ url: URL) async -> Loaded? {
        let asset = AVURLAsset(url: url)
        guard let duration = try? await asset.load(.duration).seconds, duration > 0,
              let poster = await poster(of: asset) else {
            CutSellDiagnostics.log("shop_video_failed", ["step": "read"])
            return nil
        }
        CutSellDiagnostics.log("shop_video_loaded", ["seconds": String(format: "%.1f", duration)])
        return Loaded(url: url, poster: poster, duration: duration)
    }

    static func poster(of asset: AVAsset) async -> UIImage? {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 1080, height: 1920)
        guard let image = try? await generator.image(at: CMTime(seconds: 0.1, preferredTimescale: 600)).image else { return nil }
        return UIImage(cgImage: image)
    }

    /// How long the shared video is.
    static func clipLength(_ duration: Double) -> Double { min(max(duration, 0), maxSeconds) }

    /// "0:12" on the board's chip.
    static func clock(_ seconds: Double) -> String {
        let s = Int(clipLength(seconds).rounded())
        return String(format: "%d:%02d", s / 60, s % 60)
    }

    /// Where a video frame of `size` goes to fill a 9:16 board of `board` (centred, cropped): scale and offset.
    static func fill(_ size: CGSize, into board: CGSize) -> (scale: CGFloat, offset: CGPoint) {
        guard size.width > 0, size.height > 0 else { return (1, .zero) }
        let scale = max(board.width / size.width, board.height / size.height)
        return (scale, CGPoint(x: (board.width - size.width * scale) / 2, y: (board.height - size.height * scale) / 2))
    }

    // MARK: Export

    enum ExportError: Error { case noSession, failed }

    /// The board as a 1080 × 1920 MP4: her video (first 12 s, its sound) under `overlay` (the board without background).
    static func export(video url: URL, overlay: UIImage) async throws -> URL {
        let asset = AVURLAsset(url: url)
        let size = ShopLayout.exportSize
        guard let cg = overlay.cgImage else { throw ExportError.failed }
        let top = Box(CIImage(cgImage: cg))
        let composition = try await AVMutableVideoComposition.videoComposition(with: asset) { request in
            let source = request.sourceImage
            let extent = source.extent
            let placed = fill(extent.size, into: size)
            let frame = source
                .transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
                .transformed(by: CGAffineTransform(scaleX: placed.scale, y: placed.scale))
                .transformed(by: CGAffineTransform(translationX: placed.offset.x, y: placed.offset.y))
                .cropped(to: CGRect(origin: .zero, size: size))
            request.finish(with: top.value.composited(over: frame), context: nil)
        }
        composition.renderSize = size
        composition.frameDuration = CMTime(value: 1, timescale: 30)
        guard let session = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetHighestQuality) else {
            throw ExportError.noSession
        }
        let length = clipLength(try await asset.load(.duration).seconds)
        let destination = workFolder.appendingPathComponent("board-\(UUID().uuidString).mp4")
        session.outputURL = destination
        session.outputFileType = .mp4
        session.videoComposition = composition
        session.timeRange = CMTimeRange(start: .zero, duration: CMTime(seconds: length, preferredTimescale: 600))
        let box = Box(session)
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            box.value.exportAsynchronously {
                if box.value.status == .completed { continuation.resume() } else {
                    continuation.resume(throwing: box.value.error ?? ExportError.failed)
                }
            }
        }
        CutSellDiagnostics.log("shop_video_exported", ["seconds": String(format: "%.1f", length)])
        return destination
    }

    final class Box<T>: @unchecked Sendable {
        let value: T
        init(_ value: T) { self.value = value }
    }
}

/// Her video playing behind the board on "Your board" (looped; quiet while she edits: the sound is in the shared video).
struct ShopVideoBackgroundView: UIViewRepresentable {
    let url: URL
    let duration: Double

    func makeUIView(context: Context) -> PlayerView {
        let view = PlayerView()
        view.play(url, duration: duration)
        return view
    }

    func updateUIView(_ view: PlayerView, context: Context) {
        if view.url != url { view.play(url, duration: duration) }
    }

    static func dismantleUIView(_ view: PlayerView, coordinator: ()) { view.stop() }

    final class PlayerView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        private var looper: AVPlayerLooper?
        private let player = AVQueuePlayer()
        private(set) var url: URL?

        func play(_ url: URL, duration: Double) {
            self.url = url
            let layer = self.layer as! AVPlayerLayer
            layer.videoGravity = .resizeAspectFill
            layer.player = player
            player.isMuted = true
            player.removeAllItems()
            let item = AVPlayerItem(url: url)
            looper = AVPlayerLooper(player: player, templateItem: item,
                                    timeRange: CMTimeRange(start: .zero, duration: CMTime(seconds: ShopVideoBackground.clipLength(duration),
                                                                                         preferredTimescale: 600)))
            player.play()
        }

        func stop() {
            player.pause()
            looper = nil
        }
    }
}

/// Figma W2: "▶ 0:12 · 🔊" on the board (black 50 %, radius 10).
struct ShopVideoChip: View {
    let seconds: Double
    var body: some View {
        Text(verbatim: "▶ \(ShopVideoBackground.clock(seconds)) · 🔊")
            .font(CS.inter(10, .bold))
            .foregroundStyle(.white)
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .accessibilityIdentifier("shopVideoChip")
    }
}
