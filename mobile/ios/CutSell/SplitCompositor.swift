@preconcurrency import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import Vision

/// Split clone (PO 27 Oct): joins part 1 (left) and part 2 (right) of one fixed shot into one video,
/// on the iPhone.
/// - Basic: a soft horizontal mask in the middle (~60 px at 1080 wide) — no visible line.
/// - Pro (beta): the person of part 1, cut out frame by frame (Vision person segmentation), over the
///   whole of part 2 — the two "yous" may cross the middle.
/// The small shift measured before part 2 (the phone moved a little) is applied to part 1.
/// Length = part 2's; a shorter part 1 holds its last frame. Both sounds are mixed at an even level.

/// How far part 1 must move to line up with part 2: fractions of the width / height, in CoreImage
/// coordinates (x to the right, y up).
struct SplitOffset: Equatable, Sendable {
    var x: Double = 0
    var y: Double = 0
    static let zero = SplitOffset()
    /// The size of the move, as a fraction of the width (what "your phone moved" compares).
    var magnitude: Double { (x * x + y * y).squareRoot() }
}

enum SplitBlend: String, Sendable { case basic, pro }

enum SplitError: LocalizedError {
    case noVideo, exportFailed(String), timedOut
    var errorDescription: String? {
        switch self {
        case .noVideo: return "A part has no video."
        case .exportFailed(let detail): return "The video could not be joined. \(detail)"
        case .timedOut: return "Joining took too long."
        }
    }
}

// MARK: - Alignment

enum SplitAligner {
    private static let context = CIContext(options: [.useSoftwareRenderer: false])

    /// The move that lines `part1` up with `live` (both full frames, same orientation), or nil when
    /// Vision can't tell. Measured on small copies (270 px wide) — enough for a ~2 % check.
    static func offset(part1: CIImage, live: CIImage) -> SplitOffset? {
        let width: CGFloat = 270
        guard let floating = small(part1, width: width), let reference = small(live, width: width) else { return nil }
        let request = VNTranslationalImageRegistrationRequest(targetedCGImage: floating, options: [:])
        let handler = VNImageRequestHandler(cgImage: reference, options: [:])
        do { try handler.perform([request]) } catch { return nil }
        guard let observation = request.results?.first as? VNImageTranslationAlignmentObservation else { return nil }
        let t = observation.alignmentTransform
        return SplitOffset(x: Double(t.tx) / Double(floating.width), y: Double(t.ty) / Double(floating.height))
    }

    private static func small(_ image: CIImage, width: CGFloat) -> CGImage? {
        let extent = image.extent
        guard extent.width > 1, extent.height > 1 else { return nil }
        let scale = width / extent.width
        let scaled = image.transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        return context.createCGImage(scaled, from: CGRect(x: 0, y: 0, width: width, height: (extent.height * scale).rounded()))
    }

    /// Part 1's first frame, upright (the ghost lines up with it).
    static func firstFrame(of url: URL) async -> CIImage? {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = CMTime(value: 1, timescale: 30)
        guard let image = try? await generator.image(at: .zero).image else { return nil }
        return CIImage(cgImage: image)
    }
}

// MARK: - Joining

enum SplitCompositor {
    /// Joins the two parts into a new .mov (same codec family as part 2). `progress` is 0...1.
    static func join(part1: URL, part2: URL, offset: SplitOffset, blend: SplitBlend,
                     progress: (@Sendable (Double) -> Void)? = nil) async throws -> URL {
        let asset1 = AVURLAsset(url: part1), asset2 = AVURLAsset(url: part2)
        guard let video1 = try await asset1.loadTracks(withMediaType: .video).first,
              let video2 = try await asset2.loadTracks(withMediaType: .video).first else { throw SplitError.noVideo }
        let audio1 = try? await asset1.loadTracks(withMediaType: .audio).first
        let audio2 = try? await asset2.loadTracks(withMediaType: .audio).first
        let range1 = try await video1.load(.timeRange), range2 = try await video2.load(.timeRange)
        let d1 = range1.duration, d2 = range2.duration
        guard d2.seconds > 0.1, d1.seconds > 0.05 else { throw SplitError.noVideo }

        let composition = AVMutableComposition()
        guard let track1 = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid),
              let track2 = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)
        else { throw SplitError.noVideo }
        try track2.insertTimeRange(CMTimeRange(start: range2.start, duration: d2), of: video2, at: .zero)
        if CMTimeCompare(d1, d2) >= 0 {
            try track1.insertTimeRange(CMTimeRange(start: range1.start, duration: d2), of: video1, at: .zero)
        } else {
            // Part 1 is shorter: it holds its last frame until part 2 ends.
            try track1.insertTimeRange(CMTimeRange(start: range1.start, duration: d1), of: video1, at: .zero)
            let fps = Double(try await video1.load(.nominalFrameRate))
            let frame = CMTime(seconds: 1 / max(1, fps > 0 ? fps : 30), preferredTimescale: 600)
            let lastStart = CMTimeSubtract(CMTimeAdd(range1.start, d1), frame)
            try track1.insertTimeRange(CMTimeRange(start: lastStart, duration: frame), of: video1, at: d1)
            // Only part 1's track (the composition's scale would stretch part 2 and the sounds too).
            track1.scaleTimeRange(CMTimeRange(start: d1, duration: frame), toDuration: CMTimeSubtract(d2, d1))
        }

        // Sound: part 2 whole, part 1 while it lasts, at an even level.
        var mixInputs: [AVMutableAudioMixInputParameters] = []
        let (gain1, gain2) = await evenGains(audio1, audio2)
        if let audio2, let a2 = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) {
            let r = try await audio2.load(.timeRange)
            try? a2.insertTimeRange(CMTimeRange(start: r.start, duration: CMTimeMinimum(r.duration, d2)), of: audio2, at: .zero)
            let p = AVMutableAudioMixInputParameters(track: a2)
            p.setVolumeRamp(fromStartVolume: gain2, toEndVolume: gain2, timeRange: CMTimeRange(start: .zero, duration: d2))
            mixInputs.append(p)
        }
        if let audio1, let a1 = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) {
            let r = try await audio1.load(.timeRange)
            let length = CMTimeMinimum(r.duration, CMTimeMinimum(d1, d2))
            try? a1.insertTimeRange(CMTimeRange(start: r.start, duration: length), of: audio1, at: .zero)
            let p = AVMutableAudioMixInputParameters(track: a1)
            p.setVolumeRamp(fromStartVolume: gain1, toEndVolume: gain1, timeRange: CMTimeRange(start: .zero, duration: length))
            mixInputs.append(p)
        }
        let mix = AVMutableAudioMix()
        mix.inputParameters = mixInputs
        // Exactly part 2's length (the one instruction must cover the whole composition).
        if CMTimeCompare(composition.duration, d2) > 0 {
            composition.removeTimeRange(CMTimeRange(start: d2, end: composition.duration))
        }

        // Picture: part 2's upright size.
        let size2 = try await video2.load(.naturalSize), transform2 = try await video2.load(.preferredTransform)
        let transform1 = try await video1.load(.preferredTransform)
        let upright = CGRect(origin: .zero, size: size2).applying(transform2)
        let renderSize = CGSize(width: (abs(upright.width) / 2).rounded() * 2, height: (abs(upright.height) / 2).rounded() * 2)
        let fps2 = Double(try await video2.load(.nominalFrameRate))
        let videoComposition = AVMutableVideoComposition()
        videoComposition.customVideoCompositorClass = SplitVideoCompositor.self
        videoComposition.renderSize = renderSize
        videoComposition.frameDuration = CMTime(value: 1, timescale: CMTimeScale(max(15, min(60, fps2 > 0 ? fps2.rounded() : 30))))
        videoComposition.instructions = [SplitInstruction(
            timeRange: CMTimeRange(start: .zero, duration: d2), part1: track1.trackID, part2: track2.trackID,
            orientation1: Self.orientation(transform1), orientation2: Self.orientation(transform2),
            renderSize: renderSize, offset: offset, blend: blend)]

        let hevc = await isHEVC(video2)
        let preset = hevc ? AVAssetExportPresetHEVCHighestQuality : AVAssetExportPresetHighestQuality
        guard let export = AVAssetExportSession(asset: composition, presetName: preset) else {
            throw SplitError.exportFailed("No exporter.")
        }
        let output = try persistentMediaURL(extension: "mov")
        try? FileManager.default.removeItem(at: output)
        export.outputURL = output
        export.outputFileType = .mov
        export.videoComposition = videoComposition
        export.audioMix = mix
        let watcher = Task.detached {
            while !Task.isCancelled {
                progress?(Double(export.progress))
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
        await withTaskCancellationHandler {
            await export.export()
        } onCancel: {
            export.cancelExport()
        }
        watcher.cancel()
        if Task.isCancelled { try? FileManager.default.removeItem(at: output); throw CancellationError() }
        guard export.status == .completed else {
            throw SplitError.exportFailed(export.error?.localizedDescription ?? "")
        }
        progress?(1)
        return output
    }

    static func orientation(_ t: CGAffineTransform) -> CGImagePropertyOrientation {
        switch (t.a.rounded(), t.b.rounded(), t.c.rounded(), t.d.rounded()) {
        case (0, 1, -1, 0): return .right
        case (0, -1, 1, 0): return .left
        case (-1, 0, 0, -1): return .down
        case (-1, 0, 0, 1): return .upMirrored
        default: return .up
        }
    }

    private static func isHEVC(_ track: AVAssetTrack) async -> Bool {
        guard let formats = try? await track.load(.formatDescriptions) else { return false }
        return formats.contains { CMFormatDescriptionGetMediaSubType($0) == kCMVideoCodecType_HEVC }
    }

    /// Volumes that bring the two voices to the same level (the louder part is turned down; never
    /// below 30 %, so a mostly silent part doesn't mute the other).
    static func evenGains(_ a: AVAssetTrack?, _ b: AVAssetTrack?) async -> (Float, Float) {
        guard let a, let b, let ra = await rms(a), let rb = await rms(b), ra > 1e-4, rb > 1e-4 else { return (1, 1) }
        let ga = Float(max(0.3, min(1, rb / ra))), gb = Float(max(0.3, min(1, ra / rb)))
        return (ga, gb)
    }

    /// The track's loudness (RMS of its samples, mono).
    static func rms(_ track: AVAssetTrack) async -> Double? {
        guard let asset = track.asset, let reader = try? AVAssetReader(asset: asset) else { return nil }
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false, AVLinearPCMIsBigEndianKey: false, AVNumberOfChannelsKey: 1,
            AVSampleRateKey: 16_000,
        ])
        guard reader.canAdd(output) else { return nil }
        reader.add(output)
        guard reader.startReading() else { return nil }
        var sum = 0.0, count = 0
        while let buffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            var data = [Float](repeating: 0, count: length / 4)
            data.withUnsafeMutableBytes { _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!) }
            for v in data { sum += Double(v * v) }
            count += data.count
        }
        return count > 0 ? (sum / Double(count)).squareRoot() : nil
    }
}

// MARK: - The custom compositor

final class SplitInstruction: NSObject, AVVideoCompositionInstructionProtocol {
    let timeRange: CMTimeRange
    let enablePostProcessing = false
    let containsTweening = true
    let requiredSourceTrackIDs: [NSValue]?
    let passthroughTrackID = kCMPersistentTrackID_Invalid
    let part1: CMPersistentTrackID
    let part2: CMPersistentTrackID
    let orientation1: CGImagePropertyOrientation
    let orientation2: CGImagePropertyOrientation
    let renderSize: CGSize
    let offset: SplitOffset
    let blend: SplitBlend

    init(timeRange: CMTimeRange, part1: CMPersistentTrackID, part2: CMPersistentTrackID,
         orientation1: CGImagePropertyOrientation, orientation2: CGImagePropertyOrientation,
         renderSize: CGSize, offset: SplitOffset, blend: SplitBlend) {
        self.timeRange = timeRange
        self.part1 = part1; self.part2 = part2
        self.orientation1 = orientation1; self.orientation2 = orientation2
        self.renderSize = renderSize; self.offset = offset; self.blend = blend
        self.requiredSourceTrackIDs = [NSNumber(value: part1), NSNumber(value: part2)]
    }
}

final class SplitVideoCompositor: NSObject, AVVideoCompositing {
    let sourcePixelBufferAttributes: [String: Any]? = [
        kCVPixelBufferPixelFormatTypeKey as String: [kCVPixelFormatType_32BGRA, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                                     kCVPixelFormatType_420YpCbCr8BiPlanarFullRange],
        kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
    ]
    let requiredPixelBufferAttributesForRenderContext: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
    ]
    private let queue = DispatchQueue(label: "cutsell.split.compositor")
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let segmentation: VNGeneratePersonSegmentationRequest = {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .balanced
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        return request
    }()

    func renderContextChanged(_ newRenderContext: AVVideoCompositionRenderContext) {}

    func startRequest(_ request: AVAsynchronousVideoCompositionRequest) {
        queue.async { [self] in
            guard let instruction = request.videoCompositionInstruction as? SplitInstruction,
                  let output = request.renderContext.newPixelBuffer(),
                  let frame2 = request.sourceFrame(byTrackID: instruction.part2) else {
                request.finish(with: NSError(domain: "CutSellSplit", code: 1))
                return
            }
            let size = instruction.renderSize
            let canvas = CGRect(origin: .zero, size: size)
            let image2 = Self.fill(CIImage(cvPixelBuffer: frame2).oriented(instruction.orientation2), into: size)
            var result = image2
            if let frame1 = request.sourceFrame(byTrackID: instruction.part1) {
                let upright1 = CIImage(cvPixelBuffer: frame1).oriented(instruction.orientation1)
                let move = CGAffineTransform(translationX: CGFloat(instruction.offset.x) * size.width,
                                             y: CGFloat(instruction.offset.y) * size.height)
                let image1 = Self.fill(upright1, into: size).transformed(by: move)
                let mask: CIImage
                if instruction.blend == .pro, let person = personMask(upright1) {
                    mask = Self.fill(person, into: size).transformed(by: move)
                } else {
                    mask = Self.middleMask(size: size)
                }
                result = image1.applyingFilter("CIBlendWithMask", parameters: [
                    kCIInputBackgroundImageKey: image2, kCIInputMaskImageKey: mask,
                ])
            }
            context.render(result.cropped(to: canvas), to: output, bounds: canvas, colorSpace: CGColorSpaceCreateDeviceRGB())
            request.finish(withComposedVideoFrame: output)
        }
    }

    func cancelAllPendingVideoCompositionRequests() {}

    /// White on the left (part 1), black on the right (part 2), with a soft ~60 px (at 1080) edge.
    static func middleMask(size: CGSize) -> CIImage {
        let feather = max(4, 60 * size.width / 1080)
        let gradient = CIFilter.linearGradient()
        gradient.point0 = CGPoint(x: size.width / 2 - feather / 2, y: 0)
        gradient.point1 = CGPoint(x: size.width / 2 + feather / 2, y: 0)
        gradient.color0 = CIColor(red: 1, green: 1, blue: 1)
        gradient.color1 = CIColor(red: 0, green: 0, blue: 0)
        return (gradient.outputImage ?? CIImage(color: .white)).cropped(to: CGRect(origin: .zero, size: size))
    }

    /// `image` scaled to cover `size` (centred), its corner at the origin.
    static func fill(_ image: CIImage, into size: CGSize) -> CIImage {
        let extent = image.extent
        guard extent.width > 0, extent.height > 0 else { return image }
        let scale = max(size.width / extent.width, size.height / extent.height)
        let w = extent.width * scale, h = extent.height * scale
        return image.transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: (size.width - w) / 2, y: (size.height - h) / 2))
    }

    /// Pro: where the person of part 1 is (white), at the frame's size; nil when Vision can't tell.
    private func personMask(_ image: CIImage) -> CIImage? {
        let handler = VNImageRequestHandler(ciImage: image, options: [:])
        do { try handler.perform([segmentation]) } catch { return nil }
        guard let buffer = segmentation.results?.first?.pixelBuffer else { return nil }
        let mask = CIImage(cvPixelBuffer: buffer)
        let sx = image.extent.width / mask.extent.width, sy = image.extent.height / mask.extent.height
        return mask.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            .transformed(by: CGAffineTransform(translationX: image.extent.minX, y: image.extent.minY))
            .clampedToExtent().applyingGaussianBlur(sigma: 2).cropped(to: image.extent)
    }
}
