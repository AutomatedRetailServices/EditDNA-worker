import AVFoundation
import CoreImage
import simd
import XCTest
@testable import CutSell

/// Clon AR + auto-pause + B-roll (PO 8 Nov 2026). Every input is synthetic and made here; nothing is stored in the
/// repo. AR, camera and microphone themselves need a real iPhone.
final class CloneMathTests: XCTestCase {
    private func camera(at position: SIMD3<Float>, yaw: Float = 0) -> CloneMath.Camera {
        var t = simd_float4x4(simd_quatf(angle: yaw, axis: SIMD3<Float>(0, 1, 0)))
        t.columns.3 = SIMD4<Float>(position, 1)
        return CloneMath.Camera(transform: t, fx: 1500, fy: 1500, cx: 960, cy: 720, width: 1920, height: 1440)
    }

    private func close(_ a: CGPoint, _ b: CGPoint, _ tol: CGFloat = 0.5, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.x, b.x, accuracy: tol, file: file, line: line)
        XCTAssertEqual(a.y, b.y, accuracy: tol, file: file, line: line)
    }

    func testProjectAndUnprojectAgree() {
        let cam = camera(at: SIMD3<Float>(0.3, 1.4, 0.2), yaw: 0.4)
        for p in [CGPoint(x: 100, y: 50), CGPoint(x: 960, y: 720), CGPoint(x: 1800, y: 1300)] {
            close(CloneMath.project(CloneMath.unproject(p, depth: 2.3, cam), cam)!, p)
        }
    }

    /// The anchor stays where it is in the room while the camera moves: its pixel follows the camera exactly.
    func testAnchorProjectionWithAMovingCamera() {
        let anchor = SIMD3<Float>(0, 1.4, -2)                  // 2 m in front of the start position
        let start = camera(at: SIMD3<Float>(0, 1.4, 0))
        close(CloneMath.project(anchor, start)!, CGPoint(x: 960, y: 720))
        // 0.5 m to the right: the anchor moves left by fx * 0.5 / 2 = 375 px.
        let right = camera(at: SIMD3<Float>(0.5, 1.4, 0))
        close(CloneMath.project(anchor, right)!, CGPoint(x: 960 - 375, y: 720))
        // 1 m closer: twice as far from the centre for points beside it, the centre itself stays.
        let closer = camera(at: SIMD3<Float>(0, 1.4, -1))
        close(CloneMath.project(anchor, closer)!, CGPoint(x: 960, y: 720))
        XCTAssertEqual(CloneMath.depth(of: anchor, from: closer), 1, accuracy: 1e-4)
        // Turned 90 degrees away: behind the camera, nothing to draw.
        XCTAssertNil(CloneMath.project(anchor, camera(at: SIMD3<Float>(0, 1.4, 0), yaw: .pi)))
    }

    func testCornersInTheSameCameraAreTheImageCorners() {
        let cam = camera(at: SIMD3<Float>(0, 1.4, 0))
        let corners = CloneMath.cornersInTarget(clone: cam, anchor: SIMD3<Float>(0, 1.4, -2), target: cam)!
        close(corners[0], CGPoint(x: 0, y: 0)); close(corners[1], CGPoint(x: 1920, y: 0))
        close(corners[2], CGPoint(x: 1920, y: 1440)); close(corners[3], CGPoint(x: 0, y: 1440))
    }

    /// A clone recorded from one spot, seen from another: its anchor lands where the room says it should.
    func testCloneIsPlacedAtItsAnchorFromTheNewCamera() {
        let anchor = SIMD3<Float>(-0.6, 1.4, -2.5)
        let cloneCam = camera(at: SIMD3<Float>(-0.6, 1.4, 0))   // looked straight at it
        let target = camera(at: SIMD3<Float>(0.4, 1.4, 0.3), yaw: 0.2)
        let corners = CloneMath.cornersInTarget(clone: cloneCam, anchor: anchor, target: target)!
        // The clone image centre (where the anchor was) maps to the anchor's pixel in the target camera:
        // diagonals of the projected quad meet at the projected centre (the plane is flat).
        let expected = CloneMath.project(anchor, target)!
        let centre = Self.intersection(corners[0], corners[2], corners[1], corners[3])
        close(centre, expected, 1.0)
        let plane = CloneMath.plane(clone: cloneCam, anchor: anchor)
        XCTAssertEqual(plane.width, 1920 / 1500 * 2.5, accuracy: 1e-3)
        XCTAssertEqual(plane.height, 1440 / 1500 * 2.5, accuracy: 1e-3)
        let c = plane.transform.columns.3
        XCTAssertLessThan(simd_distance(SIMD3<Float>(c.x, c.y, c.z), anchor), 1e-3)
    }

    func testFartherClonesAreDrawnFirst() {
        let target = camera(at: SIMD3<Float>(0, 1.4, 0))
        let order = CloneMath.drawOrder(anchors: [SIMD3<Float>(0, 1.4, -1), SIMD3<Float>(0.5, 1.4, -3),
                                                  SIMD3<Float>(-0.5, 1.4, -2)], target: target)
        XCTAssertEqual(order, [1, 2, 0])
    }

    static func intersection(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint, _ d: CGPoint) -> CGPoint {
        let d1 = CGPoint(x: b.x - a.x, y: b.y - a.y), d2 = CGPoint(x: d.x - c.x, y: d.y - c.y)
        let den = d1.x * d2.y - d1.y * d2.x
        let t = ((c.x - a.x) * d2.y - (c.y - a.y) * d2.x) / den
        return CGPoint(x: a.x + t * d1.x, y: a.y + t * d1.y)
    }

    // MARK: Mask, duration, sound

    /// The person mask keeps the person's pixels and makes the rest transparent.
    func testMaskCutsThePersonOut() {
        let red = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(to: CGRect(x: 0, y: 0, width: 8, height: 8))
        // Left half white (person), right half black.
        let white = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 8))
        let mask = white.composited(over: CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: 8, height: 8)))
        let out = CloneRender.cutOut(red, mask: mask)
        var px = [UInt8](repeating: 0, count: 8 * 8 * 4)
        CloneRender.context.render(out, toBitmap: &px, rowBytes: 32, bounds: CGRect(x: 0, y: 0, width: 8, height: 8),
                                   format: .RGBA8, colorSpace: nil)
        let row = 4                                      // any row
        XCTAssertEqual(px[(row * 8 + 1) * 4 + 3], 255)   // person: opaque
        XCTAssertGreaterThan(px[(row * 8 + 1) * 4 + 0], 200)
        XCTAssertEqual(px[(row * 8 + 6) * 4 + 3], 0)     // not the person: transparent
    }

    func testDurationIsTheLastTakeAndShortTakesHoldTheirLastFrame() {
        XCTAssertEqual(CloneMath.finalDuration(takeDurations: [12, 30, 21]), 21)
        let short = [0.0, 0.0333, 0.0667, 0.1]
        XCTAssertEqual(CloneMath.frameIndex(at: 0, frameTimes: short), 0)
        XCTAssertEqual(CloneMath.frameIndex(at: 0.05, frameTimes: short), 1)
        XCTAssertEqual(CloneMath.frameIndex(at: 5, frameTimes: short), 3)   // holds the last frame
    }

    func testSoundIsMixedFromZeroAtTheSameLoudnessAndNeverClips() {
        let rate = 48_000
        let quiet = (0..<rate).map { Float(sin(Double($0) * 0.05)) * 0.02 }      // 1 s, quiet
        let loud = (0..<(2 * rate)).map { Float(sin(Double($0) * 0.07)) * 0.6 }  // 2 s, loud (the last take)
        let gains = [quiet, loud].map { CloneMath.gain(rms: CloneMath.rms($0)) }
        XCTAssertEqual(gains[0], 3, accuracy: 1e-4)                       // raised (capped at x3)
        XCTAssertLessThan(gains[1], 1)                                    // lowered
        let mix = CloneMath.mix(takes: [quiet, loud], gains: gains)
        XCTAssertEqual(mix.count, loud.count)                             // as long as the last take
        XCTAssertLessThanOrEqual(mix.map { abs($0) }.max()!, 1)
        // The first take starts at 0 s: sample 10 has both.
        XCTAssertEqual(mix[10], quiet[10] * gains[0] + loud[10] * gains[1], accuracy: 1e-4)
        // After the first take ended only the last one is heard.
        XCTAssertEqual(mix[rate + 10], loud[rate + 10] * gains[1], accuracy: 1e-4)
    }

    /// d) Inside Clon the auto-pause is off: takes are kept whole and in sync.
    func testAutoPauseIsOffInsideClone() {
        XCTAssertFalse(CloneMath.autoPauseInsideClone)
        XCTAssertEqual(CloneMath.maxTakes, 4)
    }
}

/// Ranges equal to the millisecond.
func assertRanges(_ a: [AutoPausePlan.Range], _ b: [AutoPausePlan.Range], file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(a.count, b.count, "\(a) vs \(b)", file: file, line: line)
    for (x, y) in zip(a, b) {
        XCTAssertEqual(x.start, y.start, accuracy: 1e-3, file: file, line: line)
        XCTAssertEqual(x.end, y.end, accuracy: 1e-3, file: file, line: line)
    }
}

final class AutoPauseTests: XCTestCase {
    private typealias R = AutoPausePlan.Range
    private let everyone = [R(start: 0, end: 600)]                     // a person on camera the whole time

    /// b) The 0.3 s before the voice is kept: the first word is never cut.
    func testTheFirstWordIsNeverCut() {
        let keep = AutoPausePlan.keep(voice: [R(start: 2, end: 3)], presence: everyone, duration: 5, brollMode: false)
        XCTAssertEqual(keep.count, 1)
        XCTAssertEqual(keep[0].start, 1.7, accuracy: 1e-9)
        XCTAssertEqual(keep[0].end, 3.3, accuracy: 1e-9)
    }

    func testShortPausesAreKeptWhole() {
        let keep = AutoPausePlan.keep(voice: [R(start: 1, end: 2), R(start: 2.7, end: 4)], presence: everyone,
                                      duration: 6, brollMode: false)
        assertRanges(keep, [R(start: 0.7, end: 4.3)])
    }

    func testLongSilenceIsNotSaved() {
        let keep = AutoPausePlan.keep(voice: [R(start: 1, end: 2), R(start: 10, end: 11)], presence: everyone,
                                      duration: 12, brollMode: false)
        assertRanges(keep, [R(start: 0.7, end: 2.3), R(start: 9.7, end: 11.3)])
        XCTAssertEqual(AutoPausePlan.savedSeconds(keep), 3.2, accuracy: 1e-9)
    }

    func testAutoStopsAfterOneMinuteWithoutVoice() {
        XCTAssertFalse(AutoPausePlan.shouldAutoStop(now: 59, lastVoice: nil, presence: everyone, brollMode: false))
        XCTAssertTrue(AutoPausePlan.shouldAutoStop(now: 60, lastVoice: nil, presence: everyone, brollMode: false))
        XCTAssertFalse(AutoPausePlan.shouldAutoStop(now: 90, lastVoice: 40, presence: everyone, brollMode: false))
        XCTAssertTrue(AutoPausePlan.shouldAutoStop(now: 100.5, lastVoice: 40, presence: everyone, brollMode: false))
    }

    func testNoVoiceAtAllKeepsTheWholeVideo() {
        assertRanges(AutoPausePlan.keep(voice: [], presence: everyone, duration: 20, brollMode: false),
                     [R(start: 0, end: 20)])
    }

    func testTheBadgeSaysRecOrWaiting() {
        XCTAssertTrue(AutoPausePlan.isSavingNow(now: 5, lastVoice: 4.6, personOnCamera: true, brollMode: false))
        XCTAssertFalse(AutoPausePlan.isSavingNow(now: 5, lastVoice: 3, personOnCamera: true, brollMode: false))
        XCTAssertTrue(AutoPausePlan.isSavingNow(now: 5, lastVoice: 3, personOnCamera: false, brollMode: false))
    }

    /// The kept parts are joined into one file without re-encoding (a synthetic 4 s video with sound).
    func testKeptPartsAreJoinedIntoOneVideo() async throws {
        let source = try await SyntheticVideo.make(seconds: 4)
        defer { try? FileManager.default.removeItem(at: source) }
        let out = try await AutoPauseRecorder.join(asset: AVURLAsset(url: source),
                                                   ranges: [R(start: 0, end: 1), R(start: 2.5, end: 3.5)])
        defer { try? FileManager.default.removeItem(at: out) }
        let asset = AVURLAsset(url: out)
        let seconds = try await asset.load(.duration).seconds
        XCTAssertEqual(seconds, 2, accuracy: 0.15)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        XCTAssertFalse(audioTracks.isEmpty)
        let size = try await asset.loadTracks(withMediaType: .video).first!.load(.naturalSize)
        XCTAssertEqual(size, CGSize(width: 192, height: 108))         // same size: not re-encoded
    }
}

final class BrollTests: XCTestCase {
    private typealias R = AutoPausePlan.Range

    /// c) No person and no voice (the product, the room): nothing is cut, and no auto-stop.
    func testNoPersonNoVoiceIsNeverCut() {
        let samples = stride(from: 0.0, to: 120, by: 0.2).map { (time: $0, present: false) }
        let presence = AutoPausePlan.presence(samples: samples, duration: 120)
        assertRanges(AutoPausePlan.keep(voice: [R(start: 50, end: 51)], presence: presence, duration: 120,
                                        brollMode: false), [R(start: 0, end: 120)])
        XCTAssertFalse(AutoPausePlan.shouldAutoStop(now: 120, lastVoice: nil, presence: presence, brollMode: false))
    }

    /// A person on camera in a long silence IS cut; when the person leaves, the shot of the product is kept.
    func testPersonInLongSilenceIsCutButTheProductShotIsKept() {
        // Person 0-10 s, product only 10-20 s, person again 20-30 s; voice at 1-2 and 25-26.
        let samples = stride(from: 0.0, to: 30, by: 0.2).map { (time: $0, present: !($0 >= 10 && $0 < 20)) }
        let presence = AutoPausePlan.presence(samples: samples, duration: 30)
        let keep = AutoPausePlan.keep(voice: [R(start: 1, end: 2), R(start: 25, end: 26)], presence: presence,
                                      duration: 30, brollMode: false)
        XCTAssertEqual(keep.count, 3, "\(keep)")
        XCTAssertEqual(keep[0].start, 0.7, accuracy: 1e-6)
        XCTAssertEqual(keep[0].end, 2.3, accuracy: 1e-6)
        XCTAssertEqual(keep[1].start, 10, accuracy: 0.21)      // the product shot (no person) stays whole
        XCTAssertEqual(keep[1].end, 20, accuracy: 0.21)
        XCTAssertEqual(keep[2].start, 24.7, accuracy: 1e-6)    // the second phrase with its 0.3 s margins
        XCTAssertEqual(keep[2].end, 26.3, accuracy: 1e-6)
        XCTAssertEqual(AutoPausePlan.savedSeconds(keep), 1.6 + 10 + 1.6, accuracy: 0.45)
    }

    /// B-roll stretches are marked with the times of the final, joined video.
    func testBrollStretchesAreMarkedInTheFinalTimeline() {
        let samples = stride(from: 0.0, to: 30, by: 0.2).map { (time: $0, present: !($0 >= 10 && $0 < 20)) }
        let presence = AutoPausePlan.presence(samples: samples, duration: 30)
        let keep = AutoPausePlan.keep(voice: [R(start: 1, end: 2), R(start: 25, end: 26)], presence: presence,
                                      duration: 30, brollMode: false)
        let broll = AutoPausePlan.brollRanges(keep: keep, presence: presence, duration: 30, brollMode: false)
        XCTAssertEqual(broll.count, 1)
        XCTAssertEqual(broll[0].start, 1.6, accuracy: 0.21)       // after the first kept phrase (1.6 s)
        XCTAssertEqual(broll[0].end, 11.6, accuracy: 0.21)
        XCTAssertEqual(AutoPausePlan.metadataText([R(start: 1.6, end: 11.6)]), "1.60-11.60")
        let whole = AutoPausePlan.brollRanges(keep: [R(start: 0, end: 40)], presence: [], duration: 40, brollMode: true)
        assertRanges(whole, [R(start: 0, end: 40)])
    }

    /// The marks are written into the joined video's metadata.
    func testBrollMarksAreWrittenInTheFile() async throws {
        let source = try await SyntheticVideo.make(seconds: 3)
        defer { try? FileManager.default.removeItem(at: source) }
        let out = try await AutoPauseRecorder.join(asset: AVURLAsset(url: source), ranges: [R(start: 0, end: 2)],
                                                   broll: [R(start: 0.5, end: 1.5)])
        defer { try? FileManager.default.removeItem(at: out) }
        let items = try await AVURLAsset(url: out).load(.metadata)
        var found: String?
        for item in items where (item.key as? String) == AutoPauseRecorder.brollMetadataKey {
            found = try await item.load(.stringValue)
        }
        XCTAssertEqual(found, "0.50-1.50")
    }

    func testBrollButtonKeepsEverything() {
        let everyone = [R(start: 0, end: 100)]
        assertRanges(AutoPausePlan.keep(voice: [R(start: 1, end: 2)], presence: everyone, duration: 100,
                                        brollMode: true), [R(start: 0, end: 100)])
        XCTAssertFalse(AutoPausePlan.shouldAutoStop(now: 100, lastVoice: nil, presence: everyone, brollMode: true))
    }

    /// The voice path is the same for the back and the front camera: the microphone's 16-bit and float sound give
    /// the same level.
    func testBackAndFrontCameraSoundIsReadTheSame() throws {
        let samples = (0..<4_800).map { Float(sin(Double($0) * 0.1)) * 0.5 }
        let float = try XCTUnwrap(CloneRender.sampleBuffer(samples, at: 0))
        let pcm = try XCTUnwrap(AutoPauseRecorder.pcmBuffer(from: float))
        XCTAssertEqual(Int(pcm.frameLength), samples.count)
        let int16 = try XCTUnwrap(SyntheticVideo.int16Buffer(samples))
        let pcm16 = try XCTUnwrap(AutoPauseRecorder.pcmBuffer(from: int16))
        XCTAssertEqual(AutoPauseRecorder.rmsDB(pcm), AutoPauseRecorder.rmsDB(pcm16), accuracy: 0.2)
        XCTAssertGreaterThan(AutoPauseRecorder.rmsDB(pcm), -26)            // loud enough to count as voice
    }
}

/// Small synthetic media, made in the test and deleted after it.
enum SyntheticVideo {
    static func make(seconds: Double) async throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("synthetic-\(UUID().uuidString).mov")
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 192, AVVideoHeightKey: 108])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: video, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: 192, kCVPixelBufferHeightKey as String: 108])
        let audio = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 48_000, AVNumberOfChannelsKey: 1])
        writer.add(video); writer.add(audio)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        let frames = Int(seconds * 30)
        let total = Int(seconds * 48_000)
        var written = 0
        // Pictures and sound interleaved (the writer waits for both).
        for i in 0..<frames {
            while !video.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &buffer)
            let shade = CIImage(color: CIColor(red: CGFloat(i % 30) / 30, green: 0.4, blue: 0.6))
                .cropped(to: CGRect(x: 0, y: 0, width: 192, height: 108))
            CloneRender.context.render(shade, to: buffer!)
            adaptor.append(buffer!, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: 30))
            let upTo = min(total, (i + 1) * 1_600)
            while written < upTo {
                while !audio.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
                let n = min(1_600, upTo - written)
                let chunk = (written..<(written + n)).map { Float(sin(Double($0) * 0.05)) * 0.3 }
                audio.append(CloneRender.sampleBuffer(chunk, at: written)!)
                written += n
            }
        }
        video.markAsFinished()
        audio.markAsFinished()
        await writer.finishWriting()
        if writer.status != .completed { throw writer.error ?? CloneRender.RenderError.writer }
        return url
    }

    static func int16Buffer(_ samples: [Float]) -> CMSampleBuffer? {
        var asbd = AudioStreamBasicDescription(mSampleRate: 48_000, mFormatID: kAudioFormatLinearPCM,
                                               mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
                                               mBytesPerPacket: 2, mFramesPerPacket: 1, mBytesPerFrame: 2,
                                               mChannelsPerFrame: 1, mBitsPerChannel: 16, mReserved: 0)
        var format: CMAudioFormatDescription?
        CMAudioFormatDescriptionCreate(allocator: nil, asbd: &asbd, layoutSize: 0, layout: nil, magicCookieSize: 0,
                                       magicCookie: nil, extensions: nil, formatDescriptionOut: &format)
        let ints = samples.map { Int16(max(-1, min(1, $0)) * 32767) }
        let bytes = ints.count * 2
        var block: CMBlockBuffer?
        CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: bytes, blockAllocator: nil,
                                           customBlockSource: nil, offsetToData: 0, dataLength: bytes,
                                           flags: kCMBlockBufferAssureMemoryNowFlag, blockBufferOut: &block)
        guard let block, let format else { return nil }
        _ = ints.withUnsafeBytes { CMBlockBufferReplaceDataBytes(with: $0.baseAddress!, blockBuffer: block,
                                                                 offsetIntoDestination: 0, dataLength: bytes) }
        var sample: CMSampleBuffer?
        CMAudioSampleBufferCreateReadyWithPacketDescriptions(allocator: nil, dataBuffer: block, formatDescription: format,
                                                             sampleCount: ints.count, presentationTimeStamp: .zero,
                                                             packetDescriptions: nil, sampleBufferOut: &sample)
        return sample
    }
}
