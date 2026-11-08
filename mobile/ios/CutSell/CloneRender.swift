@preconcurrency import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import simd

/// Clon AR: the videos made from the takes, all on the iPhone.
/// - `makeLiveClone`: after each take, its cut-out (transparent HEVC) for the live preview of the next takes.
/// - `render`: the final video. Background = the LAST take (room + you, with its camera movement); on top, the
///   earlier takes projected with the last take's camera frame by frame, soft edges, light/colour matched; the sound
///   of every take mixed at the same loudness. Same size and format as Record (1080 x 1920 portrait .mov).
enum CloneRender {
    enum RenderError: Error { case noTakes, reader, writer }

    static let context = CIContext(options: [.cacheIntermediates: false])

    // MARK: Reading the takes

    final class FrameReader {
        private let reader: AVAssetReader
        private let output: AVAssetReaderTrackOutput
        private(set) var index = -1
        private(set) var image: CIImage?

        init(url: URL) async throws {
            let asset = AVURLAsset(url: url)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw RenderError.reader }
            reader = try AVAssetReader(asset: asset)
            output = AVAssetReaderTrackOutput(track: track, outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
            output.alwaysCopiesSampleData = false
            guard reader.canAdd(output) else { throw RenderError.reader }
            reader.add(output)
            guard reader.startReading() else { throw RenderError.reader }
        }

        /// Frame `target` (frames come in order; a take shorter than the last one keeps its last frame).
        func frame(_ target: Int) -> CIImage? {
            while index < target, let sample = output.copyNextSampleBuffer() {
                index += 1
                if let pixels = CMSampleBufferGetImageBuffer(sample) { image = CIImage(cvPixelBuffer: pixels) }
            }
            return image
        }
    }

    /// The person cut-out of frame `i`: the mask, scaled to the picture and softened (feather ~4 px).
    static func mask(_ take: CloneTake, frame i: Int, feather: Double = 4) -> CIImage? {
        guard let data = take.mask(i), data.count == take.maskWidth * take.maskHeight else { return nil }
        let raw = CIImage(bitmapData: data, bytesPerRow: take.maskWidth,
                          size: CGSize(width: take.maskWidth, height: take.maskHeight), format: .L8, colorSpace: nil)
        let scaled = raw.transformed(by: CGAffineTransform(scaleX: CGFloat(take.width) / CGFloat(take.maskWidth),
                                                           y: CGFloat(take.height) / CGFloat(take.maskHeight)))
        let blur = CIFilter.gaussianBlur()
        blur.inputImage = scaled.clampedToExtent()
        blur.radius = Float(feather)
        return blur.outputImage?.cropped(to: CGRect(x: 0, y: 0, width: CGFloat(take.width), height: CGFloat(take.height)))
    }

    static func cutOut(_ image: CIImage, mask: CIImage) -> CIImage {
        let blend = CIFilter.blendWithMask()
        blend.inputImage = image
        blend.backgroundImage = CIImage.empty()
        blend.maskImage = mask
        return blend.outputImage ?? image
    }

    // MARK: Live clone (after each take)

    /// The take as a transparent video (only the person), half size, for the live preview.
    static func makeLiveClone(_ take: CloneTake) async throws {
        let reader = try await FrameReader(url: take.videoURL)
        let w = Int(take.width / 2), h = Int(take.height / 2)
        try? FileManager.default.removeItem(at: take.alphaURL)
        let writer = try AVAssetWriter(outputURL: take.alphaURL, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevcWithAlpha, AVVideoWidthKey: w, AVVideoHeightKey: h])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h])
        guard writer.canAdd(input) else { throw RenderError.writer }
        writer.add(input)
        guard writer.startWriting() else { throw RenderError.writer }
        writer.startSession(atSourceTime: .zero)
        let half = CGAffineTransform(scaleX: 0.5, y: 0.5)
        for i in take.frames.indices {
            guard let image = reader.frame(i), let mask = CloneRender.mask(take, frame: i) else { continue }
            while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 2_000_000) }
            guard let pool = adaptor.pixelBufferPool else { throw RenderError.writer }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
            guard let buffer else { continue }
            let cut = cutOut(image, mask: mask).transformed(by: half)
            context.render(cut, to: buffer, bounds: CGRect(x: 0, y: 0, width: w, height: h), colorSpace: nil)
            adaptor.append(buffer, withPresentationTime: CMTime(seconds: take.frames[i].t, preferredTimescale: 90_000))
        }
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else { throw writer.error ?? RenderError.writer }
    }

    // MARK: Final video

    /// Output: 1080 x 1920 portrait (the 3:4 sensor picture turned upright and its centre cut to 9:16).
    static let outputSize = CGSize(width: 1080, height: 1920)

    static func portrait(_ landscape: CIImage, width: CGFloat, height: CGFloat) -> CIImage {
        // Sensor picture (landscape) -> upright portrait.
        let upright = landscape.oriented(.right)
        let w = height, h = width                   // portrait size
        let scale = outputSize.height / h
        let scaled = upright.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let cropX = (w * scale - outputSize.width) / 2
        return scaled.transformed(by: CGAffineTransform(translationX: -cropX - scaled.extent.minX, y: -scaled.extent.minY))
            .cropped(to: CGRect(origin: .zero, size: outputSize))
    }

    /// Average colour of a picture (for matching the takes' light and colour).
    static func average(_ image: CIImage) -> SIMD3<Float> {
        let filter = CIFilter.areaAverage()
        filter.inputImage = image
        filter.extent = image.extent
        guard let out = filter.outputImage else { return SIMD3<Float>(0.5, 0.5, 0.5) }
        var pixel = [UInt8](repeating: 0, count: 4)
        context.render(out, toBitmap: &pixel, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
                       format: .RGBA8, colorSpace: nil)
        return SIMD3<Float>(Float(pixel[0]), Float(pixel[1]), Float(pixel[2])) / 255
    }

    static func matched(_ image: CIImage, gain: SIMD3<Float>) -> CIImage {
        let f = CIFilter.colorMatrix()
        f.inputImage = image
        f.rVector = CIVector(x: CGFloat(gain.x), y: 0, z: 0, w: 0)
        f.gVector = CIVector(x: 0, y: CGFloat(gain.y), z: 0, w: 0)
        f.bVector = CIVector(x: 0, y: 0, z: CGFloat(gain.z), w: 0)
        return f.outputImage ?? image
    }

    /// One earlier take, cut out and placed into the last take's camera.
    static func placed(_ clone: CIImage, mask: CIImage, take: CloneTake, frame i: Int,
                       target: CloneMath.Camera, gain: SIMD3<Float>) -> CIImage? {
        guard let corners = CloneMath.cornersInTarget(clone: take.camera(i), anchor: take.anchorPoint, target: target)
        else { return nil }
        let h = CGFloat(target.height)
        func ci(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x, y: h - p.y) }   // Core Image: origin at the bottom
        let warp = CIFilter.perspectiveTransform()
        warp.inputImage = cutOut(matched(clone, gain: gain), mask: mask)
        warp.topLeft = ci(corners[0]); warp.topRight = ci(corners[1])
        warp.bottomRight = ci(corners[2]); warp.bottomLeft = ci(corners[3])
        return warp.outputImage?.cropped(to: CGRect(x: 0, y: 0, width: CGFloat(target.width), height: h))
    }

    /// The final Clon video. `progress` 0...1 on the main queue.
    static func render(_ takes: [CloneTake], progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        guard let last = takes.last else { throw RenderError.noTakes }
        let started = Date()
        let earlier = Array(takes.dropLast())
        let background = try await FrameReader(url: last.videoURL)
        var readers: [FrameReader] = []
        for take in earlier { readers.append(try await FrameReader(url: take.videoURL)) }

        // Light / colour: each earlier take matched to the last one (gain per channel, kept gentle).
        var gains: [SIMD3<Float>] = []
        if let ref = background.frame(0) {
            let target = average(ref)
            for reader in readers {
                let mean = reader.frame(0).map(average) ?? target
                gains.append(simd_clamp(target / simd_max(mean, SIMD3<Float>(repeating: 0.02)),
                                        SIMD3<Float>(repeating: 0.85), SIMD3<Float>(repeating: 1.18)))
            }
        }

        let url = try persistentMediaURL(extension: "mov")
        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let w = Int(outputSize.width), h = Int(outputSize.height)
        let videoIn = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: w, AVVideoHeightKey: h,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 10_000_000]])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: videoIn, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h])
        let audioIn = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 48_000, AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 128_000])
        guard writer.canAdd(videoIn), writer.canAdd(audioIn) else { throw RenderError.writer }
        writer.add(videoIn); writer.add(audioIn)
        guard writer.startWriting() else { throw writer.error ?? RenderError.writer }
        writer.startSession(atSourceTime: .zero)

        // Sound first (small): every take from 0 s, same loudness, as long as the last take.
        let sound = try await mixedSound(takes)

        // Pictures.
        let frameCount = last.frames.count
        for i in 0..<frameCount {
            guard let original = background.frame(i) else { break }
            var picture = original
            let target = last.camera(i)
            let t = last.frames[i].t
            // Farthest clone first; the real you of the last take stays in front of clones behind you.
            let order = CloneMath.drawOrder(anchors: earlier.map(\.anchorPoint), target: target)
            let youDepth = CloneMath.depth(of: last.anchorPoint, from: target)
            var youOnTop = false
            for k in order {
                let take = earlier[k]
                let j = CloneMath.frameIndex(at: t, frameTimes: take.frameTimes)
                if !youOnTop, CloneMath.depth(of: take.anchorPoint, from: target) < youDepth,
                   let mask = CloneRender.mask(last, frame: i) {
                    // From here on clones are in front of you: put you back over the farther ones first.
                    picture = cutOut(original, mask: mask).composited(over: picture)
                    youOnTop = true
                }
                guard let image = readers[k].frame(j), let cut = CloneRender.mask(take, frame: j),
                      let placedClone = placed(image, mask: cut, take: take, frame: j, target: target,
                                               gain: k < gains.count ? gains[k] : SIMD3<Float>(repeating: 1))
                else { continue }
                picture = placedClone.composited(over: picture)
            }
            if !youOnTop, let mask = CloneRender.mask(last, frame: i),
               order.contains(where: { CloneMath.depth(of: earlier[$0].anchorPoint, from: target) > youDepth }) {
                picture = cutOut(original, mask: mask).composited(over: picture)
            }
            let out = portrait(picture, width: CGFloat(last.width), height: CGFloat(last.height))
            while !videoIn.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 2_000_000) }
            guard let pool = adaptor.pixelBufferPool else { throw RenderError.writer }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
            guard let buffer else { continue }
            context.render(out, to: buffer, bounds: CGRect(origin: .zero, size: outputSize), colorSpace: nil)
            adaptor.append(buffer, withPresentationTime: CMTime(seconds: t, preferredTimescale: 90_000))
            // Sound for this stretch of time (interleaved with the pictures).
            try await appendSound(sound, upTo: t + 0.5, input: audioIn)
            if i % 15 == 0 {
                let p = Double(i) / Double(max(1, frameCount))
                DispatchQueue.main.async { progress(p) }
            }
        }
        videoIn.markAsFinished()
        try await appendSound(sound, upTo: .infinity, input: audioIn)
        audioIn.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else { throw writer.error ?? RenderError.writer }
        CutSellDiagnostics.log("clone_render", [
            "takes": "\(takes.count)", "seconds": String(format: "%.1f", last.duration),
            "render_s": String(format: "%.1f", Date().timeIntervalSince(started))])
        return url
    }

    // MARK: Sound

    final class SoundCursor {
        let samples: [Float]
        var written = 0
        init(_ samples: [Float]) { self.samples = samples }
    }

    static let sampleRate = 48_000.0

    static func mixedSound(_ takes: [CloneTake]) async throws -> SoundCursor {
        var all: [[Float]] = []
        for take in takes { all.append(try await readSound(take.videoURL, seconds: takes.last?.duration ?? take.duration)) }
        let gains = all.map { CloneMath.gain(rms: CloneMath.rms($0)) }
        return SoundCursor(CloneMath.mix(takes: all, gains: gains))
    }

    /// A take's sound as mono 48 kHz floats (silence if it has none).
    static func readSound(_ url: URL, seconds: Double) async throws -> [Float] {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            return [Float](repeating: 0, count: Int(seconds * sampleRate))
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: sampleRate, AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true, AVLinearPCMIsNonInterleaved: false,
            AVLinearPCMIsBigEndianKey: false])
        guard reader.canAdd(output) else { throw RenderError.reader }
        reader.add(output)
        guard reader.startReading() else { throw RenderError.reader }
        var out: [Float] = []
        while let sample = output.copyNextSampleBuffer(), let block = CMSampleBufferGetDataBuffer(sample) {
            let length = CMBlockBufferGetDataLength(block)
            var chunk = [Float](repeating: 0, count: length / 4)
            chunk.withUnsafeMutableBytes { raw in
                _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: raw.baseAddress!)
            }
            out.append(contentsOf: chunk)
        }
        return out
    }

    static func appendSound(_ sound: SoundCursor, upTo seconds: Double, input: AVAssetWriterInput) async throws {
        let limit = seconds.isFinite ? min(sound.samples.count, Int(seconds * sampleRate)) : sound.samples.count
        while sound.written < limit {
            while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 2_000_000) }
            let count = min(4_800, limit - sound.written)
            guard let buffer = sampleBuffer(Array(sound.samples[sound.written..<(sound.written + count)]),
                                            at: sound.written) else { return }
            input.append(buffer)
            sound.written += count
        }
    }

    /// Floats -> an audio sample buffer (mono 48 kHz) starting at sample `start`.
    static func sampleBuffer(_ samples: [Float], at start: Int) -> CMSampleBuffer? {
        var asbd = AudioStreamBasicDescription(mSampleRate: sampleRate, mFormatID: kAudioFormatLinearPCM,
                                               mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
                                               mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
                                               mChannelsPerFrame: 1, mBitsPerChannel: 32, mReserved: 0)
        var format: CMAudioFormatDescription?
        guard CMAudioFormatDescriptionCreate(allocator: nil, asbd: &asbd, layoutSize: 0, layout: nil, magicCookieSize: 0,
                                             magicCookie: nil, extensions: nil, formatDescriptionOut: &format) == noErr,
              let format else { return nil }
        let bytes = samples.count * 4
        var block: CMBlockBuffer?
        guard CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: bytes, blockAllocator: nil,
                                                 customBlockSource: nil, offsetToData: 0, dataLength: bytes,
                                                 flags: kCMBlockBufferAssureMemoryNowFlag,
                                                 blockBufferOut: &block) == kCMBlockBufferNoErr, let block else { return nil }
        let copied = samples.withUnsafeBytes { raw in
            CMBlockBufferReplaceDataBytes(with: raw.baseAddress!, blockBuffer: block, offsetIntoDestination: 0, dataLength: bytes)
        }
        guard copied == kCMBlockBufferNoErr else { return nil }
        var sample: CMSampleBuffer?
        guard CMAudioSampleBufferCreateReadyWithPacketDescriptions(
            allocator: nil, dataBuffer: block, formatDescription: format, sampleCount: samples.count,
            presentationTimeStamp: CMTime(value: CMTimeValue(start), timescale: CMTimeScale(sampleRate)),
            packetDescriptions: nil, sampleBufferOut: &sample) == noErr else { return nil }
        return sample
    }
}
