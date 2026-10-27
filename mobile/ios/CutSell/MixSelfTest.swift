#if DEBUG
import AVFoundation
import Foundation

/// Debug builds only (`-CutSellMixTest`): the sample editor plays a generated movie (a steady 440 Hz
/// tone, no user media) so a probe can measure what the preview's sound really does per part, e.g.
/// that Mute on one part silences only that part (PO 27 Oct: on the iPhone the voice kept playing).
enum MixSelfTest {
    static var enabled: Bool { ProcessInfo.processInfo.arguments.contains("-CutSellMixTest") }

    /// A small movie with a steady tone: video 160x160 at 10 fps, AAC mono 44.1 kHz.
    static func makeMovie(seconds: Double) async throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("cutsell-mixtest-\(Int(seconds)).mp4")
        if FileManager.default.fileExists(atPath: url.path) { return url }
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 160, AVVideoHeightKey: 160,
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: video, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: 160, kCVPixelBufferHeightKey as String: 160,
        ])
        let audio = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 44_100, AVNumberOfChannelsKey: 1, AVEncoderBitRateKey: 64_000,
        ])
        video.expectsMediaDataInRealTime = false
        audio.expectsMediaDataInRealTime = false
        writer.add(video); writer.add(audio)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        let fps = 10
        for frame in 0..<Int(seconds * Double(fps)) {
            while !video.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(5)) }
            var buffer: CVPixelBuffer?
            CVPixelBufferCreate(nil, 160, 160, kCVPixelFormatType_32BGRA, nil, &buffer)
            guard let buffer else { continue }
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                memset(base, Int32((frame * 7) % 255), CVPixelBufferGetDataSize(buffer))
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: CMTimeScale(fps)))
        }
        video.markAsFinished()

        let rate = 44_100.0
        let chunk = 4_410
        var format = AudioStreamBasicDescription(mSampleRate: rate, mFormatID: kAudioFormatLinearPCM,
                                                 mFormatFlags: kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked,
                                                 mBytesPerPacket: 2, mFramesPerPacket: 1, mBytesPerFrame: 2,
                                                 mChannelsPerFrame: 1, mBitsPerChannel: 16, mReserved: 0)
        var description: CMAudioFormatDescription?
        CMAudioFormatDescriptionCreate(allocator: nil, asbd: &format, layoutSize: 0, layout: nil, magicCookieSize: 0,
                                       magicCookie: nil, extensions: nil, formatDescriptionOut: &description)
        let total = Int(seconds * rate)
        var written = 0
        while written < total, let description {
            while !audio.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(5)) }
            let count = min(chunk, total - written)
            var samples = [Int16](repeating: 0, count: count)
            for i in 0..<count { samples[i] = Int16(12_000 * sin(2 * .pi * 440 * Double(written + i) / rate)) }
            var block: CMBlockBuffer?
            CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: count * 2, blockAllocator: nil,
                                               customBlockSource: nil, offsetToData: 0, dataLength: count * 2, flags: 0,
                                               blockBufferOut: &block)
            guard let block else { break }
            samples.withUnsafeBytes { raw in
                _ = CMBlockBufferReplaceDataBytes(with: raw.baseAddress!, blockBuffer: block, offsetIntoDestination: 0,
                                                  dataLength: count * 2)
            }
            var sample: CMSampleBuffer?
            CMAudioSampleBufferCreateReadyWithPacketDescriptions(
                allocator: nil, dataBuffer: block, formatDescription: description, sampleCount: count,
                presentationTimeStamp: CMTime(value: CMTimeValue(written), timescale: 44_100),
                packetDescriptions: nil, sampleBufferOut: &sample)
            if let sample { audio.append(sample) }
            written += count
        }
        audio.markAsFinished()
        await writer.finishWriting()
        if writer.status != .completed { throw writer.error ?? CocoaError(.fileWriteUnknown) }
        return url
    }

    /// Loudness (dBFS) of each second of `asset` as mixed by `mix` -- what the preview plays.
    static func loudness(of asset: AVAsset, mix: AVAudioMix?) async -> [Int] {
        guard let tracks = try? await asset.loadTracks(withMediaType: .audio), !tracks.isEmpty,
              let reader = try? AVAssetReader(asset: asset) else { return [] }
        let output = AVAssetReaderAudioMixOutput(audioTracks: tracks, audioSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false, AVLinearPCMIsBigEndianKey: false, AVSampleRateKey: 44_100,
            AVNumberOfChannelsKey: 1,
        ])
        output.audioMix = mix
        reader.add(output)
        guard reader.startReading() else { return [] }
        var sums: [Double] = [], counts: [Int] = []
        var index = 0
        while let buffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            var data = [Float](repeating: 0, count: length / 4)
            data.withUnsafeMutableBytes { _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!) }
            for value in data {
                let second = index / 44_100
                while sums.count <= second { sums.append(0); counts.append(0) }
                sums[second] += Double(value * value); counts[second] += 1
                index += 1
            }
        }
        return zip(sums, counts).map { sum, count in
            let rms = sqrt(sum / Double(max(1, count)))
            return rms < 1e-6 ? -120 : Int((20 * log10(rms)).rounded())
        }
    }
}
#endif
