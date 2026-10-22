import AVFoundation
import Foundation

/// 22 Oct: the light copy CutSell uploads to cut and preview a video (the export uses the original in
/// Photos). Reader -> writer with an exact bitrate (an export preset can't be given one, and its size
/// limit stops the file early instead). HEVC (H.264 when HEVC can't be written), at most 1920 on the long
/// side, the source's own frame rate and rotation, AAC 128 kb/s.
enum LightCopyWriter {
    enum Failure: Error { case cannotRead, cannotWrite(String), cancelled }

    static func write(asset: AVAsset, videoTrack: AVAssetTrack, to destination: URL, videoBitsPerSecond: Double,
                      seconds: Double, progress: @Sendable @escaping (Double) -> Void) async throws {
        let natural = try await videoTrack.load(.naturalSize)
        let transform = try await videoTrack.load(.preferredTransform)
        let fps = (try? await videoTrack.load(.nominalFrameRate)) ?? 30
        let size = targetSize(natural: natural)
        let audioTrack = try await asset.loadTracks(withMediaType: .audio).first

        let reader = try AVAssetReader(asset: asset)
        let videoOut = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        ])
        videoOut.alwaysCopiesSampleData = false
        guard reader.canAdd(videoOut) else { throw Failure.cannotRead }
        reader.add(videoOut)
        var audioOut: AVAssetReaderTrackOutput?
        if let audioTrack {
            let output = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: [AVFormatIDKey: kAudioFormatLinearPCM])
            output.alwaysCopiesSampleData = false
            if reader.canAdd(output) { reader.add(output); audioOut = output }
        }

        let writer = try AVAssetWriter(outputURL: destination, fileType: .mp4)
        writer.shouldOptimizeForNetworkUse = true
        var compression: [String: Any] = [
            AVVideoAverageBitRateKey: Int(videoBitsPerSecond),
            AVVideoExpectedSourceFrameRateKey: Int(max(1, fps.rounded())),
            AVVideoMaxKeyFrameIntervalDurationKey: 2,
        ]
        var videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: Int(size.width),
            AVVideoHeightKey: Int(size.height),
            AVVideoScalingModeKey: AVVideoScalingModeResizeAspect,
            AVVideoCompressionPropertiesKey: compression,
        ]
        #if targetEnvironment(simulator)
        let preferH264 = true   // the simulator has no video chip: HEVC by software is far too slow for the probes
        #else
        let preferH264 = false
        #endif
        if preferH264 || !writer.canApply(outputSettings: videoSettings, forMediaType: .video) {
            compression[AVVideoProfileLevelKey] = AVVideoProfileLevelH264HighAutoLevel
            videoSettings[AVVideoCodecKey] = AVVideoCodecType.h264
            videoSettings[AVVideoCompressionPropertiesKey] = compression
        }
        let videoIn = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoIn.expectsMediaDataInRealTime = false
        videoIn.transform = transform
        guard writer.canAdd(videoIn) else { throw Failure.cannotWrite("video input") }
        writer.add(videoIn)
        var audioIn: AVAssetWriterInput?
        if audioOut != nil {
            var channels = 1
            if let audioTrack, let description = try? await audioTrack.load(.formatDescriptions).first,
               let basic = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee {
                channels = min(2, max(1, Int(basic.mChannelsPerFrame)))
            }
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC, AVNumberOfChannelsKey: channels, AVSampleRateKey: 44_100,
                AVEncoderBitRateKey: Int(VideoPreparation.audioBitsPerSecond),
            ])
            input.expectsMediaDataInRealTime = false
            if writer.canAdd(input) { writer.add(input); audioIn = input }
        }

        guard reader.startReading() else { throw reader.error ?? Failure.cannotRead }
        guard writer.startWriting() else { throw writer.error ?? Failure.cannotWrite("start") }
        let firstTime = (try? await videoTrack.load(.timeRange).start) ?? .zero
        writer.startSession(atSourceTime: firstTime.isValid ? firstTime : .zero)

        let total = max(0.1, seconds)
        let pumpVideo = Pump(input: videoIn, output: videoOut, label: "cutsell.lightcopy.video") { time in
            progress(min(0.99, CMTimeGetSeconds(time) / total))
        }
        let pumpAudio = audioIn.flatMap { input in audioOut.map { Pump(input: input, output: $0, label: "cutsell.lightcopy.audio") { _ in } } }
        async let videoDone: Void = pumpVideo.run()
        async let audioDone: Void = pumpAudio?.run() ?? ()
        _ = await (videoDone, audioDone)

        if reader.status == .failed {
            writer.cancelWriting()
            throw reader.error ?? Failure.cannotRead
        }
        if Task.isCancelled {
            reader.cancelReading(); writer.cancelWriting()
            throw Failure.cancelled
        }
        await writer.finishWriting()
        if writer.status != .completed { throw writer.error ?? Failure.cannotWrite("finish") }
    }

    /// At most 1920 on the long side and 1080 on the short side, even numbers, never larger than the source.
    static func targetSize(natural: CGSize) -> CGSize {
        let w = abs(natural.width), h = abs(natural.height)
        guard w > 0, h > 0 else { return CGSize(width: 1920, height: 1080) }
        let scale = min(1, 1920 / max(w, h), 1080 / min(w, h))
        func even(_ v: CGFloat) -> CGFloat { max(2, (v * scale / 2).rounded() * 2) }
        return CGSize(width: even(w), height: even(h))
    }

    /// Copies one track's samples into a writer input until the track ends.
    final class Pump: @unchecked Sendable {
        let input: AVAssetWriterInput
        let output: AVAssetReaderOutput
        let queue: DispatchQueue
        let onTime: (CMTime) -> Void

        init(input: AVAssetWriterInput, output: AVAssetReaderOutput, label: String, onTime: @escaping (CMTime) -> Void) {
            self.input = input; self.output = output; self.queue = DispatchQueue(label: label); self.onTime = onTime
        }

        func run() async {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                var finished = false
                input.requestMediaDataWhenReady(on: queue) { [self] in
                    guard !finished else { return }
                    while input.isReadyForMoreMediaData {
                        guard let buffer = output.copyNextSampleBuffer() else {
                            finished = true
                            input.markAsFinished()
                            continuation.resume()
                            return
                        }
                        onTime(CMSampleBufferGetPresentationTimeStamp(buffer))
                        if !input.append(buffer) {
                            finished = true
                            input.markAsFinished()
                            continuation.resume()
                            return
                        }
                    }
                }
            }
        }
    }
}
