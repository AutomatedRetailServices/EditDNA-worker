import Foundation
import AVFoundation

struct PreparedVideo: Sendable {
    let url: URL
    let contentType: String
    let wasTranscoded: Bool
}

actor VideoPreparation {
    static let shared = VideoPreparation()

    private let maxUploadBytesWithoutPreparation: Int64 = 500 * 1024 * 1024

    func prepare(fileURL: URL, contentType: String) async throws -> PreparedVideo {
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

        let exceeds1080WorkingResolution = min(width, height) > 1080.5 || max(width, height) > 1920.5
        let oversizedFile = size > maxUploadBytesWithoutPreparation
        guard exceeds1080WorkingResolution || oversizedFile else {
            return PreparedVideo(url: fileURL, contentType: contentType, wasTranscoded: false)
        }

        guard let exporter = AVAssetExportSession(asset: asset, presetName: AVAssetExportPreset1920x1080) else {
            throw VideoPreparationError.exportUnavailable
        }
        let destination = try preparedDestination()
        try? FileManager.default.removeItem(at: destination)
        exporter.outputURL = destination
        exporter.outputFileType = .mp4
        exporter.shouldOptimizeForNetworkUse = true

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            exporter.exportAsynchronously {
                switch exporter.status {
                case .completed:
                    continuation.resume(returning: ())
                case .failed:
                    continuation.resume(throwing: exporter.error ?? VideoPreparationError.exportFailed)
                case .cancelled:
                    continuation.resume(throwing: VideoPreparationError.cancelled)
                default:
                    continuation.resume(throwing: exporter.error ?? VideoPreparationError.exportFailed)
                }
            }
        }

        let preparedAttributes = try FileManager.default.attributesOfItem(atPath: destination.path)
        let preparedSize = (preparedAttributes[.size] as? NSNumber)?.int64Value ?? 0
        guard preparedSize > 0 else { throw VideoPreparationError.exportFailed }
        return PreparedVideo(url: destination, contentType: "video/mp4", wasTranscoded: true)
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
