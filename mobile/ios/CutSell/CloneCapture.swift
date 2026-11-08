@preconcurrency import ARKit
@preconcurrency import AVFoundation
import Combine
import Foundation
import RealityKit
import simd

/// Clon AR (beta, PO 8 Nov 2026). One iPhone, up to 4 takes, all of you in the same room: ARKit tracks the room,
/// cuts the person out of every frame and remembers where the camera was, so each earlier take stands at its own
/// spot while you record the next one, and the final video puts every take into the last one's camera.
enum CloneSupport {
    /// The Clon control is shown only when the iPhone's AR can cut a person out of the picture.
    static var isSupported: Bool {
        ARWorldTrackingConfiguration.isSupported && ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentation)
    }
    /// LiDAR iPhones (Pro): person depth too, so a farther clone goes behind you.
    static var hasLiDAR: Bool { ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) }

    static func configuration() -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        if hasLiDAR, ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) {
            config.frameSemantics = .personSegmentationWithDepth
        } else {
            config.frameSemantics = .personSegmentation
        }
        config.providesAudioData = true
        // 30 frames a second at the largest 4:3 size: half the work of 60 and the same frame rate as Record.
        if let format = ARWorldTrackingConfiguration.supportedVideoFormats
            .filter({ $0.framesPerSecond == 30 })
            .max(by: { $0.imageResolution.width < $1.imageResolution.width }) {
            config.videoFormat = format
        }
        return config
    }

    /// Exposure and white balance locked from the first take on, so every take has the same light and colour.
    static func lockCamera(_ locked: Bool) {
        guard let device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera else { return }
        do {
            try device.lockForConfiguration()
            if locked {
                if device.isExposureModeSupported(.locked) { device.exposureMode = .locked }
                if device.isWhiteBalanceModeSupported(.locked) { device.whiteBalanceMode = .locked }
            } else {
                if device.isExposureModeSupported(.continuousAutoExposure) { device.exposureMode = .continuousAutoExposure }
                if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                    device.whiteBalanceMode = .continuousAutoWhiteBalance
                }
            }
            device.unlockForConfiguration()
        } catch {
            // Optional: takes still work, only the colour may differ a little.
        }
    }

    /// Clone sound is heard only with headphones (otherwise the microphone would record it again).
    static var headphonesConnected: Bool {
        let wired: Set<AVAudioSession.Port> = [.headphones, .bluetoothA2DP, .bluetoothLE, .bluetoothHFP, .usbAudio]
        return AVAudioSession.sharedInstance().currentRoute.outputs.contains { wired.contains($0.portType) }
    }
}

/// One recorded take: the room video (sensor orientation), the person mask of every frame, where the camera was
/// at every frame, and the spot in the room where the person stood (its anchor).
struct CloneTake: Codable, Equatable {
    struct Frame: Codable, Equatable {
        var t: Double                    // seconds from the take's start
        var m: [Float]                   // camera -> world, 16 numbers (column major)
        var k: [Float]                   // fx, fy, cx, cy
    }

    var folder: URL
    var width: Float                     // captured image size (landscape sensor pixels)
    var height: Float
    var maskWidth: Int
    var maskHeight: Int
    var frames: [Frame]
    var anchor: [Float]                  // x, y, z (world)
    var duration: Double

    var videoURL: URL { folder.appendingPathComponent("video.mov") }
    var masksURL: URL { folder.appendingPathComponent("masks.bin") }
    var alphaURL: URL { folder.appendingPathComponent("clone.mov") }
    var anchorPoint: SIMD3<Float> { SIMD3<Float>(anchor[0], anchor[1], anchor[2]) }
    var frameTimes: [Double] { frames.map(\.t) }

    func camera(_ i: Int) -> CloneMath.Camera {
        let f = frames[max(0, min(frames.count - 1, i))]
        let m = f.m
        let transform = simd_float4x4(columns: (SIMD4<Float>(m[0], m[1], m[2], m[3]), SIMD4<Float>(m[4], m[5], m[6], m[7]),
                                                SIMD4<Float>(m[8], m[9], m[10], m[11]), SIMD4<Float>(m[12], m[13], m[14], m[15])))
        return CloneMath.Camera(transform: transform, fx: f.k[0], fy: f.k[1], cx: f.k[2], cy: f.k[3], width: width, height: height)
    }

    /// The mask of frame `i` (one byte a pixel; 255 = person).
    func mask(_ i: Int) -> Data? {
        guard let handle = try? FileHandle(forReadingFrom: masksURL) else { return nil }
        defer { try? handle.close() }
        let size = maskWidth * maskHeight
        try? handle.seek(toOffset: UInt64(max(0, min(frames.count - 1, i)) * size))
        return try? handle.read(upToCount: size)
    }
}

/// Where the takes of one Clon session live (deleted when the session ends; the final video is kept apart).
enum CloneFolder {
    static func root() throws -> URL {
        let url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell/Clone", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func newTake() throws -> URL {
        let url = try root().appendingPathComponent("take-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func remove(_ take: CloneTake) { try? FileManager.default.removeItem(at: take.folder) }
}

/// Records takes from the AR session (on its own queue) and reports the tracking state.
final class CloneCaptureEngine: NSObject, ARSessionDelegate, @unchecked Sendable {
    enum Tracking: Equatable { case scanning, normal, unstable }

    let queue = DispatchQueue(label: "ai.cutsell.clone.capture")
    /// Main queue: tracking changes, elapsed seconds while recording.
    var onTracking: ((Tracking) -> Void)?
    var onElapsed: ((Double) -> Void)?
    weak var session: ARSession?

    private var writer: AVAssetWriter?
    private var videoIn: AVAssetWriterInput?
    private var audioIn: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var masks: FileHandle?
    private var folder: URL?
    private var startTime: Double?
    private var frames: [CloneTake.Frame] = []
    private var size = CGSize.zero
    private var maskSize = (w: 0, h: 0)
    private var anchorSamples: [SIMD3<Float>] = []
    private var recording = false
    private var lastTracking: Tracking = .scanning
    private var lastReported: Double = -1

    func begin() throws {
        let folder = try CloneFolder.newTake()
        let masks = folder.appendingPathComponent("masks.bin")
        FileManager.default.createFile(atPath: masks.path, contents: nil)
        let handle = try FileHandle(forWritingTo: masks)
        queue.sync {
            self.folder = folder; self.masks = handle
            self.writer = nil; self.videoIn = nil; self.audioIn = nil; self.adaptor = nil
            self.startTime = nil; self.frames = []; self.anchorSamples = []
            self.recording = true
        }
    }

    /// Stops the take and returns it (nil if nothing was recorded).
    func end() async -> CloneTake? {
        let state = queue.sync { () -> (AVAssetWriter?, URL?, [CloneTake.Frame], CGSize, (w: Int, h: Int), [SIMD3<Float>]) in
            self.recording = false
            self.videoIn?.markAsFinished(); self.audioIn?.markAsFinished()
            try? self.masks?.close(); self.masks = nil
            return (self.writer, self.folder, self.frames, self.size, self.maskSize, self.anchorSamples)
        }
        guard let writer = state.0, let folder = state.1, !state.2.isEmpty else { return nil }
        if writer.status == .writing { await writer.finishWriting() }
        guard writer.status == .completed else { return nil }
        let anchor = Self.median(state.5) ?? Self.fallbackAnchor(state.2)
        return CloneTake(folder: folder, width: Float(state.3.width), height: Float(state.3.height),
                         maskWidth: state.4.w, maskHeight: state.4.h, frames: state.2,
                         anchor: [anchor.x, anchor.y, anchor.z], duration: state.2.last?.t ?? 0)
    }

    static func median(_ points: [SIMD3<Float>]) -> SIMD3<Float>? {
        guard !points.isEmpty else { return nil }
        func mid(_ v: [Float]) -> Float { let s = v.sorted(); return s[s.count / 2] }
        return SIMD3<Float>(mid(points.map(\.x)), mid(points.map(\.y)), mid(points.map(\.z)))
    }

    /// No person found: 1.5 m in front of the first frame's camera.
    static func fallbackAnchor(_ frames: [CloneTake.Frame]) -> SIMD3<Float> {
        guard let m = frames.first?.m else { return SIMD3<Float>(0, 0, -1.5) }
        let position = SIMD3<Float>(m[12], m[13], m[14]), forward = -SIMD3<Float>(m[8], m[9], m[10])
        return position + simd_normalize(forward) * 1.5
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        let tracking: Tracking
        switch camera.trackingState {
        case .normal: tracking = .normal
        case .notAvailable: tracking = .scanning
        case .limited(let reason):
            tracking = (reason == .initializing || reason == .relocalizing) ? .scanning : .unstable
        }
        guard tracking != lastTracking else { return }
        lastTracking = tracking
        let report = onTracking
        DispatchQueue.main.async { report?(tracking) }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard recording, let masks else { return }
        let pixels = frame.capturedImage
        if writer == nil { startWriter(for: pixels) }
        guard let writer, let videoIn, let adaptor else { return }
        if startTime == nil {
            guard writer.startWriting() else { return }
            writer.startSession(atSourceTime: Self.time(frame.timestamp))
            startTime = frame.timestamp
        }
        guard let start = startTime, videoIn.isReadyForMoreMediaData,
              adaptor.append(pixels, withPresentationTime: Self.time(frame.timestamp)) else { return }
        // The mask and camera of exactly the frames written (so frame i of the video is frame i here).
        masks.write(Self.maskBytes(frame.segmentationBuffer, size: &maskSize))
        let m = frame.camera.transform, k = frame.camera.intrinsics
        frames.append(CloneTake.Frame(t: frame.timestamp - start,
                                      m: [m.columns.0.x, m.columns.0.y, m.columns.0.z, m.columns.0.w,
                                          m.columns.1.x, m.columns.1.y, m.columns.1.z, m.columns.1.w,
                                          m.columns.2.x, m.columns.2.y, m.columns.2.z, m.columns.2.w,
                                          m.columns.3.x, m.columns.3.y, m.columns.3.z, m.columns.3.w],
                                      k: [k.columns.0.x, k.columns.1.y, k.columns.2.x, k.columns.2.y]))
        if frames.count % 10 == 1, let point = findPerson(frame) { anchorSamples.append(point) }
        let elapsed = frame.timestamp - start
        if elapsed - lastReported >= 0.25 {
            lastReported = elapsed
            let report = onElapsed
            DispatchQueue.main.async { report?(elapsed) }
        }
    }

    func session(_ session: ARSession, didOutputAudioSampleBuffer audioSampleBuffer: CMSampleBuffer) {
        guard recording, let audioIn, let startTime, audioIn.isReadyForMoreMediaData else { return }
        guard CMSampleBufferGetPresentationTimeStamp(audioSampleBuffer).seconds >= startTime else { return }
        audioIn.append(audioSampleBuffer)
    }

    // MARK: Writing

    static func time(_ seconds: TimeInterval) -> CMTime { CMTime(seconds: seconds, preferredTimescale: 90_000) }

    private func startWriter(for pixels: CVPixelBuffer) {
        guard let folder else { return }
        let width = CVPixelBufferGetWidth(pixels), height = CVPixelBufferGetHeight(pixels)
        size = CGSize(width: width, height: height)
        let url = folder.appendingPathComponent("video.mov")
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mov) else { return }
        let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: width, AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 14_000_000]])
        video.expectsMediaDataInRealTime = true
        // Shown upright (portrait) by players; the pixels stay in sensor orientation for the clone maths.
        video.transform = CGAffineTransform(rotationAngle: .pi / 2)
        let audio = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 48_000, AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 128_000])
        audio.expectsMediaDataInRealTime = true
        guard writer.canAdd(video), writer.canAdd(audio) else { return }
        writer.add(video); writer.add(audio)
        self.writer = writer; videoIn = video; audioIn = audio
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: video, sourcePixelBufferAttributes: nil)
    }

    /// The person mask (one byte a pixel), rows packed.
    static func maskBytes(_ buffer: CVPixelBuffer?, size: inout (w: Int, h: Int)) -> Data {
        guard let buffer else { return Data(count: max(1, size.w * size.h)) }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let row = CVPixelBufferGetBytesPerRow(buffer)
        size = (w, h)
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return Data(count: w * h) }
        var out = Data(count: w * h)
        out.withUnsafeMutableBytes { dst in
            guard let d = dst.baseAddress else { return }
            for y in 0..<h { memcpy(d + y * w, base + y * row, w) }
        }
        return out
    }

    /// Where the person stands in the room: the person's depth (LiDAR / depth estimate) at the middle of the
    /// mask, or the floor under the person's feet.
    private func findPerson(_ frame: ARFrame) -> SIMD3<Float>? {
        guard let seg = frame.segmentationBuffer else { return nil }
        CVPixelBufferLockBaseAddress(seg, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(seg, .readOnly) }
        let w = CVPixelBufferGetWidth(seg), h = CVPixelBufferGetHeight(seg), row = CVPixelBufferGetBytesPerRow(seg)
        guard let base = CVPixelBufferGetBaseAddress(seg)?.assumingMemoryBound(to: UInt8.self) else { return nil }
        var sx = 0, sy = 0, count = 0, maxY = 0
        for y in stride(from: 0, to: h, by: 2) {
            for x in stride(from: 0, to: w, by: 2) where base[y * row + x] > 127 {
                sx += x; sy += y; count += 1; maxY = max(maxY, y)
            }
        }
        guard count > (w * h) / 400 else { return nil }      // at least ~1 % of the picture
        let cx = Double(sx) / Double(count), cy = Double(sy) / Double(count)
        let scaleX = Double(CVPixelBufferGetWidth(frame.capturedImage)) / Double(w)
        let scaleY = Double(CVPixelBufferGetHeight(frame.capturedImage)) / Double(h)
        let cam = CloneMath.Camera(transform: frame.camera.transform, fx: frame.camera.intrinsics.columns.0.x,
                                   fy: frame.camera.intrinsics.columns.1.y, cx: frame.camera.intrinsics.columns.2.x,
                                   cy: frame.camera.intrinsics.columns.2.y,
                                   width: Float(CVPixelBufferGetWidth(frame.capturedImage)),
                                   height: Float(CVPixelBufferGetHeight(frame.capturedImage)))
        // 1. The person's own depth (LiDAR or ARKit's estimate).
        if let depth = frame.estimatedDepthData ?? frame.sceneDepth?.depthMap,
           let metres = Self.medianDepth(depth, mask: base, maskWidth: w, maskHeight: h, maskRow: row) {
            return CloneMath.unproject(CGPoint(x: cx * scaleX, y: cy * scaleY), depth: metres, cam)
        }
        // 2. The floor under the feet.
        if let session {
            let feet = CGPoint(x: cx * scaleX, y: Double(maxY) * scaleY)
            let through = CloneMath.unproject(feet, depth: 1, cam)
            let origin = SIMD3<Float>(frame.camera.transform.columns.3.x, frame.camera.transform.columns.3.y,
                                      frame.camera.transform.columns.3.z)
            let query = ARRaycastQuery(origin: origin, direction: simd_normalize(through - origin),
                                       allowing: .estimatedPlane, alignment: .horizontal)
            if let hit = session.raycast(query).first {
                let p = hit.worldTransform.columns.3
                return SIMD3<Float>(p.x, p.y, p.z)
            }
        }
        return nil
    }

    /// Median depth (metres) of the person's pixels.
    static func medianDepth(_ depth: CVPixelBuffer, mask: UnsafePointer<UInt8>, maskWidth: Int, maskHeight: Int,
                            maskRow: Int) -> Float? {
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        guard CVPixelBufferGetPixelFormatType(depth) == kCVPixelFormatType_DepthFloat32,
              let base = CVPixelBufferGetBaseAddress(depth) else { return nil }
        let dw = CVPixelBufferGetWidth(depth), dh = CVPixelBufferGetHeight(depth), drow = CVPixelBufferGetBytesPerRow(depth)
        var values: [Float] = []
        for y in stride(from: 0, to: maskHeight, by: 3) {
            for x in stride(from: 0, to: maskWidth, by: 3) where mask[y * maskRow + x] > 127 {
                let dx = x * dw / maskWidth, dy = y * dh / maskHeight
                let v = (base + dy * drow).assumingMemoryBound(to: Float.self)[dx]
                if v.isFinite, v > 0.2, v < 8 { values.append(v) }
            }
        }
        guard values.count > 20 else { return nil }
        values.sort()
        return values[values.count / 2]
    }
}

/// The earlier takes shown in the room while you record the next one: each is a cut-out video on a plane that
/// follows its recorded camera, frame by frame, at its anchor's depth. Sound only with headphones.
@MainActor
final class CloneLiveClones {
    private struct Live {
        let take: CloneTake
        let entity: ModelEntity
        let video: AVPlayer
        let sound: AVPlayer
    }

    private let anchor = AnchorEntity(world: .zero)
    private var clones: [Live] = []
    private var updates: Cancellable?

    func attach(to view: ARView) {
        if anchor.scene == nil { view.scene.addAnchor(anchor) }
        updates = view.scene.subscribe(to: SceneEvents.Update.self) { [weak self] _ in
            MainActor.assumeIsolated { self?.follow() }
        }
    }

    /// Shows `takes` (the earlier clones), standing still at their first frame until `play()`.
    func show(_ takes: [CloneTake]) {
        clear()
        for take in takes where FileManager.default.fileExists(atPath: take.alphaURL.path) {
            let video = AVPlayer(url: take.alphaURL)
            video.isMuted = true
            let sound = AVPlayer(url: take.videoURL)
            sound.isMuted = !CloneSupport.headphonesConnected
            let entity = ModelEntity(mesh: .generatePlane(width: 1, height: 1), materials: [VideoMaterial(avPlayer: video)])
            anchor.addChild(entity)
            clones.append(Live(take: take, entity: entity, video: video, sound: sound))
        }
        follow()
    }

    func play() {
        let start = CMClockGetTime(CMClockGetHostTimeClock()) + CMTime(seconds: 0.1, preferredTimescale: 600)
        for clone in clones {
            for player in [clone.video, clone.sound] {
                player.automaticallyWaitsToMinimizeStalling = false
                player.seek(to: .zero, toleranceBefore: .zero, toleranceAfter: .zero)
                player.setRate(1, time: .invalid, atHostTime: start)
            }
        }
    }

    func stop() {
        for clone in clones {
            clone.video.pause(); clone.sound.pause()
            clone.video.seek(to: .zero); clone.sound.seek(to: .zero)
        }
    }

    func refreshSound() {
        let on = CloneSupport.headphonesConnected
        for clone in clones { clone.sound.isMuted = !on }
    }

    func clear() {
        for clone in clones { clone.video.pause(); clone.sound.pause(); clone.entity.removeFromParent() }
        clones = []
    }

    private func follow() {
        for clone in clones {
            let t = clone.video.currentTime().seconds
            let i = CloneMath.frameIndex(at: t.isFinite ? t : 0, frameTimes: clone.take.frameTimes)
            let plane = CloneMath.plane(clone: clone.take.camera(i), anchor: clone.take.anchorPoint)
            let scale = simd_float4x4(diagonal: SIMD4<Float>(plane.width, plane.height, 1, 1))
            clone.entity.transform = Transform(matrix: plane.transform * scale)
        }
    }
}
