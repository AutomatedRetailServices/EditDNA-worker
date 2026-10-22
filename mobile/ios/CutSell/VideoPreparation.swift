import Foundation
import AVFoundation

struct PreparedVideo: Sendable {
    let url: URL
    let contentType: String
    let wasTranscoded: Bool
}

actor VideoPreparation {
    static let shared = VideoPreparation()

    /// `progress` (0...1) while a 4K (or not H.264/HEVC) video is made smaller (it can take minutes:
    /// build 19 shows it instead of a frozen 0 %).
    func prepare(fileURL: URL, contentType: String,
                 progress: @Sendable @escaping (Double) -> Void = { _ in }) async throws -> PreparedVideo {
        let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        guard size > 0 else { throw VideoPreparationError.emptyFile }

        let asset = AVURLAsset(url: fileURL)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else { throw VideoPreparationError.missingVideoTrack }
        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let displaySize = naturalSize.applying(transform)
        let width = abs(displaySize.width)
        let height = abs(displaySize.height)

        // Build 21: a video that is already fine (H.264 or HEVC, under 4K) is uploaded as it is,
        // never re-encoded. (Build 19 re-encoded a 1180x2556 screen recording and it grew from
        // 72 MB to 103 MB.)
        let codec = try await Self.codec(of: track)
        let under4K = max(width, height) < 3839.5 && min(width, height) < 2159.5
        let fineCodec = codec == kCMVideoCodecType_H264 || codec == kCMVideoCodecType_HEVC
        // 22 Oct (production): a 6:39 1080p iPhone video at ~25 Mb/s weighs 1.2 GB; the server refused it
        // (1 GB limit) and a 5:22 one needed ~80 parts. A heavy video (over ~14 Mb/s) is made lighter too.
        let seconds0 = max(1, CMTimeGetSeconds(try await asset.load(.duration)))
        let heavy = Self.isHeavy(bytes: size, seconds: seconds0)
        if fineCodec && under4K && !heavy {
            CutSellDiagnostics.log("prepare_skipped", ["width": String(Int(width)), "height": String(Int(height)),
                                                       "codec": Self.fourCC(codec), "mb": String(size / 1_048_576)])
            return PreparedVideo(url: fileURL, contentType: contentType, wasTranscoded: false)
        }

        // Otherwise a LIGHTER COPY is made for cutting and preview (the export uses the original in
        // Photos): HEVC at most 1080p, ~10 Mb/s, lower for long videos so the file stays under ~950 MB.
        let seconds = seconds0
        let videoBits = Self.videoBitrate(seconds: seconds)
        let destination = try preparedDestination()
        try? FileManager.default.removeItem(at: destination)
        CutSellDiagnostics.log("prepare_started", ["width": String(Int(width)), "height": String(Int(height)), "heavy": heavy ? "1" : "0",
                                                   "codec": Self.fourCC(codec), "mb": String(size / 1_048_576),
                                                   "kbps": String(Int(videoBits / 1000))])
        let started = Date()
        do {
            try await LightCopyWriter.write(asset: asset, videoTrack: track, to: destination, videoBitsPerSecond: videoBits,
                                            seconds: seconds, progress: progress)
        } catch {
            // A video CutSell can't make lighter is uploaded as it is.
            try? FileManager.default.removeItem(at: destination)
            CutSellDiagnostics.log("prepare_failed_original_kept", ["mb": String(size / 1_048_576),
                                                                     "code": String((error as NSError).code),
                                                                     "error": String(describing: error).prefix(120).description])
            progress(1)
            return PreparedVideo(url: fileURL, contentType: contentType, wasTranscoded: false)
        }

        let preparedAttributes = try? FileManager.default.attributesOfItem(atPath: destination.path)
        let preparedSize = (preparedAttributes?[.size] as? NSNumber)?.int64Value ?? 0
        progress(1)
        // A copy shorter than the original (her last words cut off) is never used.
        let preparedSeconds = CMTimeGetSeconds((try? await AVURLAsset(url: destination).load(.duration)) ?? .zero)
        let complete = preparedSeconds >= seconds - 0.5
        // Never upload a file heavier than the original.
        guard preparedSize > 0, preparedSize < size, complete else {
            try? FileManager.default.removeItem(at: destination)
            CutSellDiagnostics.log("prepare_finished", ["mb_before": String(size / 1_048_576),
                                                        "mb_after": String(preparedSize / 1_048_576), "used": "original",
                                                        "complete": complete ? "1" : "0"])
            return PreparedVideo(url: fileURL, contentType: contentType, wasTranscoded: false)
        }
        CutSellDiagnostics.log("prepare_finished", ["mb_before": String(size / 1_048_576),
                                                    "mb_after": String(preparedSize / 1_048_576), "used": "smaller",
                                                    "seconds": String(format: "%.1f", Date().timeIntervalSince(started))])
        OriginalLinks.carry(from: fileURL, to: destination)
        return PreparedVideo(url: destination, contentType: "video/mp4", wasTranscoded: true)
    }

    /// ~10 Mb/s for 1080p.
    static let targetBitsPerSecond: Double = 10_000_000
    /// The light copy stays under this (the server accepts up to 1 GiB).
    static let maxCopyBytes: Double = 950_000_000
    static let audioBitsPerSecond: Double = 128_000

    /// ~10 Mb/s; for a video longer than ~12.5 min, what fits in 950 MB (20 min: ~6.2 Mb/s; 40 min: ~3 Mb/s),
    /// never under 1.5 Mb/s (enough to cut and preview; the export uses the original).
    static func videoBitrate(seconds: Double) -> Double {
        let fits = maxCopyBytes * 8 / max(1, seconds) - audioBitsPerSecond
        return max(1_500_000, min(targetBitsPerSecond, fits * 0.97))
    }
    /// Over this (video + audio) a video is re-encoded before upload even when it is under 4K.
    static let heavyBitsPerSecond: Double = 14_000_000

    static func isHeavy(bytes: Int64, seconds: Double) -> Bool {
        seconds > 0 && Double(bytes) * 8 / seconds > heavyBitsPerSecond
    }

    static func codec(of track: AVAssetTrack) async throws -> FourCharCode {
        let descriptions = try await track.load(.formatDescriptions)
        return descriptions.first.map { CMFormatDescriptionGetMediaSubType($0) } ?? 0
    }

    static func fourCC(_ code: FourCharCode) -> String {
        let bytes = [24, 16, 8, 0].map { UInt8((code >> $0) & 0xFF) }
        let text = String(bytes: bytes, encoding: .ascii) ?? ""
        return text.trimmingCharacters(in: .whitespaces).isEmpty ? "unknown" : text
    }

    private func preparedDestination() throws -> URL {
        let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell", isDirectory: true)
            .appendingPathComponent("PreparedUploads", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.appendingPathComponent("\(UUID().uuidString).mp4")
    }
}

enum VideoPreparationError: LocalizedError {
    case emptyFile
    case missingVideoTrack
    case exportUnavailable
    case exportFailed
    case cancelled

    var errorDescription: String? {
        switch self {
        case .emptyFile: return "The selected video is empty."
        case .missingVideoTrack: return "CutSell couldn’t read the video track."
        case .exportUnavailable: return "This video can’t be prepared on this device."
        case .exportFailed: return "CutSell couldn’t prepare the video for upload."
        case .cancelled: return "Video preparation was cancelled."
        }
    }
}
