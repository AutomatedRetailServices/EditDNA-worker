@preconcurrency import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import Vision

/// Clon (beta, PO 29 Oct; was "Split"): joins part 1 (left) and part 2 (right) of one fixed shot into one
/// video, on the iPhone.
/// - Silhouette (default): the WHOLE of part 2 (room + person 2) as the background and, on top, ONLY the person
///   of part 1, cut out frame by frame (Vision person segmentation, .accurate, or .balanced when too slow),
///   with a soft 3–6 px edge, lined up with part 2 (the measured shift) and matched to part 2's colour/brightness.
/// - Halves (fallback): left half part 1 + right half part 2 with a soft ~60 px (at 1080) gradient in the middle,
///   used when the cut-out fails or the phone moved too much.
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

enum SplitBlend: String, Sendable { case silhouette, halves }

/// What the silhouette pass saw, frame by frame (the caller falls back to halves when it went badly).
final class SplitStats: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var frames = 0, failed = 0, empty = 0, balancedFrames = 0
    private(set) var segmentSeconds = 0.0
    func add(failed f: Bool, empty e: Bool, balanced: Bool, seconds: Double) {
        lock.lock(); defer { lock.unlock() }
        frames += 1
        if f { failed += 1 }
        if e { empty += 1 }
        if balanced { balancedFrames += 1 }
        segmentSeconds += seconds
    }
    /// The cut-out worked: Vision answered on most frames and found a person on most of them.
    var usable: Bool {
        lock.lock(); defer { lock.unlock() }
        guard frames > 0 else { return false }
        return Double(failed) / Double(frames) < 0.2 && Double(empty) / Double(frames) < 0.5
    }
    var summary: String {
        lock.lock(); defer { lock.unlock() }
        return "frames \(frames) failed \(failed) empty \(empty) balanced \(balancedFrames) seg \(String(format: "%.1f", segmentSeconds)) s"
    }
}

enum SplitError: LocalizedError {
    case noVideo, exportFailed(String), timedOut, silhouetteFailed(String)
    var errorDescription: String? {
        switch self {
        case .noVideo: return "A part has no video."
        case .exportFailed(let detail): return "The video could not be joined. \(detail)"
        case .timedOut: return "Joining took too long."
        case .silhouetteFailed(let detail): return "The cut-out did not work. \(detail)"
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
        let offset = SplitOffset(x: Double(t.tx) / Double(floating.width), y: Double(t.ty) / Double(floating.height))
        // A quarter of the frame or more is not a nudged phone: Vision found no real match.
        return offset.magnitude < 0.25 ? offset : nil
    }

    #if DEBUG
    /// Probe only: Vision's raw answer for two images at `width` (nil = full size).
    static func raw(part1: CIImage, live: CIImage, width: CGFloat?) -> String {
        let w1 = width ?? part1.extent.width
        guard let floating = small(part1, width: w1), let reference = small(live, width: w1) else { return "no images" }
        let request = VNTranslationalImageRegistrationRequest(targetedCGImage: floating, options: [:])
        let handler = VNImageRequestHandler(cgImage: reference, options: [:])
        do { try handler.perform([request]) } catch { return "error \(error)" }
        guard let o = request.results?.first else { return "no result" }
        let t = o.alignmentTransform
        return "w \(floating.width)x\(floating.height) a \(t.a) b \(t.b) c \(t.c) d \(t.d) tx \(t.tx) ty \(t.ty) conf \(o.confidence)"
    }
    #endif

    private static func small(_ image: CIImage, width: CGFloat) -> CGImage? {
        let extent = image.extent
        guard extent.width > 1, extent.height > 1 else { return nil }
        let scale = width / extent.width
        let scaled = image.transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        return context.createCGImage(scaled, from: CGRect(x: 0, y: 0, width: width, height: (extent.height * scale).rounded()))
    }

    /// The average colour (linear 0...1 RGB) of a frame.
    static func average(_ image: CIImage) -> (Double, Double, Double)? {
        let filter = CIFilter.areaAverage()
        filter.inputImage = image
        filter.extent = image.extent
        guard let out = filter.outputImage else { return nil }
        var px = [Float](repeating: 0, count: 4)
        context.render(out, toBitmap: &px, rowBytes: 16, bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
                       format: .RGBAf, colorSpace: nil)
        return (Double(px[0]), Double(px[1]), Double(px[2]))
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
    /// More than this (a fraction of the width) and the phone moved too much for a clean cut-out.
    static let tooFarForSilhouette = 0.08

    /// The shift between the two RECORDED parts (first frames), or nil when Vision can't tell.
    static func measuredOffset(part1: URL, part2: URL) async -> SplitOffset? {
        guard let a = await SplitAligner.firstFrame(of: part1), let b = await SplitAligner.firstFrame(of: part2) else { return nil }
        return SplitAligner.offset(part1: a, live: b)
    }

    /// Per-channel gains that bring part 1's colour and brightness to part 2's (simple exposure + temperature
    /// match from the average colour of their first frames), kept within ±18 %.
    static func colourGains(part1: URL, part2: URL) async -> (Double, Double, Double) {
        guard let a = await SplitAligner.firstFrame(of: part1), let b = await SplitAligner.firstFrame(of: part2),
              let ma = SplitAligner.average(a), let mb = SplitAligner.average(b) else { return (1, 1, 1) }
        func g(_ x: Double, _ y: Double) -> Double { x > 0.02 ? max(0.82, min(1.18, y / x)) : 1 }
        return (g(ma.0, mb.0), g(ma.1, mb.1), g(ma.2, mb.2))
    }

    static func join(part1: URL, part2: URL, offset: SplitOffset, blend: SplitBlend,
                     gains: (Double, Double, Double) = (1, 1, 1), stats: SplitStats? = nil,
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
        videoComposition.colorPrimaries = AVVideoColorPrimaries_ITU_R_709_2
        videoComposition.colorTransferFunction = AVVideoTransferFunction_ITU_R_709_2
        videoComposition.colorYCbCrMatrix = AVVideoYCbCrMatrix_ITU_R_709_2
        videoComposition.frameDuration = CMTime(value: 1, timescale: CMTimeScale(max(15, min(60, fps2 > 0 ? fps2.rounded() : 30))))
        videoComposition.instructions = [SplitInstruction(
            timeRange: CMTimeRange(start: .zero, duration: d2), part1: track1.trackID, part2: track2.trackID,
            orientation1: Self.orientation(transform1), orientation2: Self.orientation(transform2),
            renderSize: renderSize, offset: offset, blend: blend, gains: gains, stats: stats ?? SplitStats())]

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
        if blend == .silhouette, let stats, !stats.usable {
            try? FileManager.default.removeItem(at: output)
            throw SplitError.silhouetteFailed(stats.summary)
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
    let gains: (Double, Double, Double)
    let stats: SplitStats

    init(timeRange: CMTimeRange, part1: CMPersistentTrackID, part2: CMPersistentTrackID,
         orientation1: CGImagePropertyOrientation, orientation2: CGImagePropertyOrientation,
         renderSize: CGSize, offset: SplitOffset, blend: SplitBlend, gains: (Double, Double, Double), stats: SplitStats) {
        self.timeRange = timeRange
        self.part1 = part1; self.part2 = part2
        self.orientation1 = orientation1; self.orientation2 = orientation2
        self.renderSize = renderSize; self.offset = offset; self.blend = blend
        self.gains = gains; self.stats = stats
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
    /// .accurate first; after a few frames, .balanced if one frame takes longer than ~0.12 s (too slow for a
    /// 30 s video). One request for the whole video (Vision keeps the edges steady from frame to frame).
    private var segmentation: VNGeneratePersonSegmentationRequest = SplitVideoCompositor.request(.accurate)
    private var balanced = false
    private var timed: [Double] = []
    private var lastMask: CIImage?

    private static func request(_ quality: VNGeneratePersonSegmentationRequest.QualityLevel) -> VNGeneratePersonSegmentationRequest {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = quality
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        return request
    }

    #if DEBUG
    /// Self-test only: a stand-in for Vision on synthetic videos (no real person in them).
    nonisolated(unsafe) static var testMask: ((CIImage) -> CIImage?)?
    #endif

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
                let g = instruction.gains
                let image1 = Self.fill(upright1, into: size).transformed(by: move)
                    .applyingFilter("CIColorMatrix", parameters: [
                        "inputRVector": CIVector(x: g.0, y: 0, z: 0, w: 0),
                        "inputGVector": CIVector(x: 0, y: g.1, z: 0, w: 0),
                        "inputBVector": CIVector(x: 0, y: 0, z: g.2, w: 0),
                    ])
                let mask: CIImage
                if instruction.blend == .silhouette {
                    let started = Date()
                    let person = personMask(upright1)
                    let seconds = Date().timeIntervalSince(started)
                    adapt(seconds)
                    let found = person.map(Self.hasPerson) ?? false
                    instruction.stats.add(failed: person == nil, empty: person != nil && !found, balanced: balanced, seconds: seconds)
                    // A frame Vision misses keeps the last good outline (no flicker); none yet = part 2 only.
                    let shape = (found ? person : nil) ?? lastMask
                    if found { lastMask = person }
                    if let shape {
                        // A soft 3–6 px edge so the cut-out never looks cut.
                        let feather = max(1.5, 4 * size.width / 1080)
                        mask = Self.fill(shape, into: size).transformed(by: move)
                            .clampedToExtent().applyingGaussianBlur(sigma: feather).cropped(to: canvas)
                    } else {
                        mask = CIImage(color: .black).cropped(to: canvas)
                    }
                } else {
                    mask = Self.middleMask(size: size)
                }
                result = image1.applyingFilter("CIBlendWithMask", parameters: [
                    kCIInputBackgroundImageKey: image2, kCIInputMaskImageKey: mask,
                ])
            }
            // sRGB (≈ the video's Rec. 709 curve): rendering to device RGB brightened the whole video (120 -> 143).
            context.render(result.cropped(to: canvas), to: output, bounds: canvas, colorSpace: Self.outputSpace)
            request.finish(withComposedVideoFrame: output)
        }
    }

    func cancelAllPendingVideoCompositionRequests() {}

    private static let outputSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

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

    /// .accurate → .balanced once the first frames show it is too slow for the phone.
    private func adapt(_ seconds: Double) {
        guard !balanced else { return }
        timed.append(seconds)
        if timed.count >= 8, timed.reduce(0, +) / Double(timed.count) > 0.12 {
            balanced = true
            segmentation = Self.request(.balanced)
        }
    }

    /// Where the person of part 1 is (white), at the frame's size; nil when Vision can't tell.
    private func personMask(_ image: CIImage) -> CIImage? {
        #if DEBUG
        if let testMask = Self.testMask { return testMask(image) }
        #endif
        let handler = VNImageRequestHandler(ciImage: image, options: [:])
        do { try handler.perform([segmentation]) } catch { return nil }
        guard let buffer = segmentation.results?.first?.pixelBuffer else { return nil }
        let mask = CIImage(cvPixelBuffer: buffer)
        let sx = image.extent.width / mask.extent.width, sy = image.extent.height / mask.extent.height
        return mask.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            .transformed(by: CGAffineTransform(translationX: image.extent.minX, y: image.extent.minY))
            .cropped(to: image.extent)
    }

    /// The mask shows a person (more than ~1 % of the frame).
    private static let probe = CIContext(options: [.useSoftwareRenderer: false])
    static func hasPerson(_ mask: CIImage) -> Bool {
        let filter = CIFilter.areaAverage()
        filter.inputImage = mask
        filter.extent = mask.extent
        guard let out = filter.outputImage else { return false }
        var px = [Float](repeating: 0, count: 4)
        probe.render(out, toBitmap: &px, rowBytes: 16, bounds: CGRect(x: 0, y: 0, width: 1, height: 1), format: .RGBAf, colorSpace: nil)
        return px[0] > 0.01
    }
}
