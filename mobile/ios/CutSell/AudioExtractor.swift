import AVFoundation
import Foundation

/// Build 23 ("audio first"): the voice of a video as a small AAC file (mono, 44.1 kHz, ~64 kb/s:
/// ~0.5 MB per minute instead of ~60 MB), made on the iPhone in a few seconds. It is uploaded first
/// so the server can start cutting (Deepgram + Claude only need the voice) while the full video
/// keeps uploading in the background.
enum AudioExtractor {
    enum Failure: LocalizedError {
        case noAudioTrack
        case cannotRead(String)
        case cannotWrite(String)

        var errorDescription: String? {
            switch self {
            case .noAudioTrack: return "The video has no sound track."
            case .cannotRead(let detail): return "CutSell couldn’t read the video’s sound (\(detail))."
            case .cannotWrite(let detail): return "CutSell couldn’t save the sound (\(detail))."
            }
        }
    }

    static let bitRate = 64_000
    static let sampleRate = 44_100.0

    /// Writes `<name>.m4a` next to the other prepared uploads and returns its URL.
    static func extract(from videoURL: URL, progress: (@Sendable (Double) -> Void)? = nil) async throws -> URL {
        let started = Date()
        let asset = AVURLAsset(url: videoURL)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else { throw Failure.noAudioTrack }
        let duration = max(0.1, CMTimeGetSeconds(try await asset.load(.duration)))
        let destination = try folder().appendingPathComponent("\(UUID().uuidString).m4a")
        try? FileManager.default.removeItem(at: destination)

        let reader: AVAssetReader
        do { reader = try AVAssetReader(asset: asset) } catch { throw Failure.cannotRead((error as NSError).localizedDescription) }
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
            AVNumberOfChannelsKey: 1,
            AVSampleRateKey: sampleRate,
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw Failure.cannotRead("no reader output") }
        reader.add(output)

        let writer: AVAssetWriter
        do { writer = try AVAssetWriter(outputURL: destination, fileType: .m4a) } catch {
            throw Failure.cannotWrite((error as NSError).localizedDescription)
        }
        let input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVNumberOfChannelsKey: 1,
            AVSampleRateKey: sampleRate,
            AVEncoderBitRateKey: bitRate,
        ])
        input.expectsMediaDataInRealTime = false
        guard writer.canAdd(input) else { throw Failure.cannotWrite("no writer input") }
        writer.add(input)

        guard reader.startReading() else { throw Failure.cannotRead(reader.error?.localizedDescription ?? "start") }
        guard writer.startWriting() else { throw Failure.cannotWrite(writer.error?.localizedDescription ?? "start") }
        writer.startSession(atSourceTime: .zero)

        let queue = DispatchQueue(label: "ai.cutsell.audio-extract")
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let once = StallWatch.Once(continuation)
            input.requestMediaDataWhenReady(on: queue) {
                while input.isReadyForMoreMediaData {
                    guard let buffer = output.copyNextSampleBuffer() else {
                        input.markAsFinished()
                        writer.finishWriting {
                            if writer.status == .completed {
                                once.finish(.success(()))
                            } else {
                                once.finish(.failure(Failure.cannotWrite(writer.error?.localizedDescription ?? "finish")))
                            }
                        }
                        return
                    }
                    let time = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(buffer))
                    progress?(min(1, max(0, time / duration)))
                    if !input.append(buffer) {
                        reader.cancelReading()
                        once.finish(.failure(Failure.cannotWrite(writer.error?.localizedDescription ?? "append")))
                        return
                    }
                }
            }
        }
        if reader.status == .failed { throw Failure.cannotRead(reader.error?.localizedDescription ?? "read") }
        let bytes = ((try? FileManager.default.attributesOfItem(atPath: destination.path))?[.size] as? NSNumber)?.int64Value ?? 0
        CutSellDiagnostics.log("audio_extracted", ["seconds": String(format: "%.1f", Date().timeIntervalSince(started)),
                                                   "kb": String(bytes / 1024),
                                                   "video_seconds": String(format: "%.0f", duration)])
        progress?(1)
        return destination
    }

    private static func folder() throws -> URL {
        let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell", isDirectory: true)
            .appendingPathComponent("PreparedAudio", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
