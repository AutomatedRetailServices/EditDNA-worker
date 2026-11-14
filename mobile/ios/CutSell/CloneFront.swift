@preconcurrency import AVFoundation
import CoreImage
import UIKit
import Vision
import simd

/// Clon with the FRONT camera (PO 14 Nov: "the clone camera only uses the back camera, it must be both").
///
/// ARKit tracks the room only with the back camera, so the front camera works differently: the iPhone stands
/// still (on a stand or propped up) and every take is filmed from the same spot. Vision cuts the person out of
/// every frame. Each take is saved in the same form as an AR take (sensor-landscape video + masks + one camera
/// per frame), with the same camera for every frame and every take, so `CloneRender` puts the earlier takes
/// exactly where they were — the clones stand where you stood.
enum CloneFrontMath {
    /// The one camera of every front take (same for every frame): looking down -Z from the origin.
    static func camera(width: Int, height: Int) -> CloneTake.Frame {
        let f = Float(width)                     // any focal length works: every take uses the same camera
        return CloneTake.Frame(t: 0, m: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                               k: [f, f, Float(width) / 2, Float(height) / 2])
    }

    /// Where a front take's person stands: 1.5 m in front of the camera (all takes the same depth).
    static let anchor: [Float] = [0, 0, -1.5]
}

/// The front camera, the live preview (mirrored like a selfie) with the earlier clones on top, and the takes.
final class CloneFrontCamera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate,
    AVCaptureAudioDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    let queue = DispatchQueue(label: "ai.cutsell.clone.front", qos: .userInitiated)
    /// Main queue: elapsed seconds while recording.
    var onElapsed: ((Double) -> Void)?

    private let videoOut = AVCaptureVideoDataOutput()
    private let audioOut = AVCaptureAudioDataOutput()
    private var device: AVCaptureDevice?
    private let context = CIContext(options: [.cacheIntermediates: false])
    private var pool: CVPixelBufferPool?
    private let segmentation: VNGeneratePersonSegmentationRequest = {
        let r = VNGeneratePersonSegmentationRequest()
        r.qualityLevel = .balanced
        r.outputPixelFormat = kCVPixelFormatType_OneComponent8
        return r
    }()

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
    private var recording = false
    private var lastReported: Double = -1

    static var isAvailable: Bool {
        AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) != nil
    }

    func configure() throws {
        guard device == nil else { return }
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            throw CloneFrontError.noCamera
        }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = session.canSetSessionPreset(.hd1920x1080) ? .hd1920x1080 : .high
        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else { throw CloneFrontError.noCamera }
        session.addInput(input)
        if let mic = AVCaptureDevice.default(for: .audio), let micIn = try? AVCaptureDeviceInput(device: mic),
           session.canAddInput(micIn) {
            session.addInput(micIn)
        }
        videoOut.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOut.alwaysDiscardsLateVideoFrames = true
        videoOut.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(videoOut) else { throw CloneFrontError.noCamera }
        session.addOutput(videoOut)
        if session.canAddOutput(audioOut) {
            audioOut.setSampleBufferDelegate(self, queue: queue)
            session.addOutput(audioOut)
        }
        // Upright portrait, NOT mirrored: the saved takes show the real you (text reads the right way);
        // only the preview on screen is mirrored, like every selfie camera.
        if let connection = videoOut.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }
            if connection.isVideoMirroringSupported {
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = false
            }
        }
        try? camera.lockForConfiguration()
        camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
        camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
        camera.unlockForConfiguration()
        device = camera
    }

    func startRunning() { queue.async { if !self.session.isRunning { self.session.startRunning() } } }
    func stopRunning() { queue.async { if self.session.isRunning { self.session.stopRunning() } } }

    /// Exposure and white balance locked from the first take on (same light in every take).
    func lockCamera(_ locked: Bool) {
        guard let device, (try? device.lockForConfiguration()) != nil else { return }
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
    }

    // MARK: Takes

    func begin() throws {
        let folder = try CloneFolder.newTake()
        let masksURL = folder.appendingPathComponent("masks.bin")
        FileManager.default.createFile(atPath: masksURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: masksURL)
        queue.sync {
            self.folder = folder; self.masks = handle
            self.writer = nil; self.videoIn = nil; self.audioIn = nil; self.adaptor = nil
            self.startTime = nil; self.frames = []; self.lastReported = -1
            self.recording = true
        }
    }

    func end() async -> CloneTake? {
        let state = queue.sync { () -> (AVAssetWriter?, URL?, [CloneTake.Frame], CGSize, (w: Int, h: Int)) in
            self.recording = false
            self.videoIn?.markAsFinished(); self.audioIn?.markAsFinished()
            try? self.masks?.close(); self.masks = nil
            return (self.writer, self.folder, self.frames, self.size, self.maskSize)
        }
        guard let writer = state.0, let folder = state.1, !state.2.isEmpty else { return nil }
        if writer.status == .writing { await writer.finishWriting() }
        guard writer.status == .completed else { return nil }
        return CloneTake(folder: folder, width: Float(state.3.width), height: Float(state.3.height),
                         maskWidth: state.4.w, maskHeight: state.4.h, frames: state.2,
                         anchor: CloneFrontMath.anchor, duration: state.2.last?.t ?? 0)
    }

    // MARK: Capture

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard recording else { return }
        if output === audioOut {
            guard let audioIn, let startTime, audioIn.isReadyForMoreMediaData,
                  CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds >= startTime else { return }
            audioIn.append(sampleBuffer)
            return
        }
        guard let upright = CMSampleBufferGetImageBuffer(sampleBuffer), let masks,
              let landscape = sensorLandscape(upright) else { return }
        let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        if writer == nil { startWriter(width: CVPixelBufferGetWidth(landscape), height: CVPixelBufferGetHeight(landscape)) }
        guard let writer, let videoIn, let adaptor else { return }
        if startTime == nil {
            guard writer.startWriting() else { return }
            writer.startSession(atSourceTime: CloneCaptureEngine.time(time))
            startTime = time
        }
        guard let start = startTime, videoIn.isReadyForMoreMediaData else { return }
        // The person cut-out of this very frame (Vision), stored like ARKit's mask.
        let handler = VNImageRequestHandler(cvPixelBuffer: landscape, options: [:])
        try? handler.perform([segmentation])
        let mask = segmentation.results?.first?.pixelBuffer
        guard adaptor.append(landscape, withPresentationTime: CloneCaptureEngine.time(time)) else { return }
        masks.write(CloneCaptureEngine.maskBytes(mask, size: &maskSize))
        var frame = CloneFrontMath.camera(width: CVPixelBufferGetWidth(landscape), height: CVPixelBufferGetHeight(landscape))
        frame.t = time - start
        frames.append(frame)
        let elapsed = time - start
        if elapsed - lastReported >= 0.25 {
            lastReported = elapsed
            let report = onElapsed
            DispatchQueue.main.async { report?(elapsed) }
        }
    }

    /// The upright portrait frame turned into the same "sensor landscape" orientation as an AR take (so the
    /// existing Clon maths and render work unchanged: `.right` turns it upright again).
    private func sensorLandscape(_ upright: CVPixelBuffer) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(upright), h = CVPixelBufferGetHeight(upright)
        let lw = max(w, h), lh = min(w, h)
        if pool == nil {
            let attrs: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                                        kCVPixelBufferWidthKey as String: lw, kCVPixelBufferHeightKey as String: lh,
                                        kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
            CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &pool)
        }
        guard let pool else { return nil }
        var out: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out)
        guard let out else { return nil }
        let image = CIImage(cvPixelBuffer: upright)
        let landscape = w < h ? image.oriented(.left) : image       // portrait -> landscape (undone by .right)
        let placed = landscape.transformed(by: CGAffineTransform(translationX: -landscape.extent.minX,
                                                                 y: -landscape.extent.minY))
        context.render(placed, to: out, bounds: CGRect(x: 0, y: 0, width: lw, height: lh),
                       colorSpace: CGColorSpaceCreateDeviceRGB())
        return out
    }

    private func startWriter(width: Int, height: Int) {
        guard let folder else { return }
        size = CGSize(width: width, height: height)
        guard let writer = try? AVAssetWriter(outputURL: folder.appendingPathComponent("video.mov"), fileType: .mov)
        else { return }
        let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: width, AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 14_000_000]])
        video.expectsMediaDataInRealTime = true
        video.transform = CGAffineTransform(rotationAngle: .pi / 2)        // shown upright, like an AR take
        let audio = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 48_000, AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 128_000])
        audio.expectsMediaDataInRealTime = true
        guard writer.canAdd(video), writer.canAdd(audio) else { return }
        writer.add(video); writer.add(audio)
        self.writer = writer; videoIn = video; audioIn = audio
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: video, sourcePixelBufferAttributes: nil)
    }
}

enum CloneFrontError: Error { case noCamera }

/// The front preview (mirrored) with the earlier clones playing on top, at the same spots.
final class CloneFrontPreviewView: UIView {
    let previewLayer = AVCaptureVideoPreviewLayer()
    /// Holds one player layer per earlier clone, turned like the preview (upright + mirrored).
    private let clonesLayer = CALayer()
    private var players: [(video: AVPlayer, sound: AVPlayer, layer: AVPlayerLayer)] = []

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .black
        previewLayer.videoGravity = .resizeAspectFill
        layer.addSublayer(previewLayer)
        layer.addSublayer(clonesLayer)
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
        // The 9:16 picture filling the screen, in landscape (as the clone videos are stored), then turned
        // upright and mirrored like the preview.
        let fill = CloneFrontLayout.fill(screen: bounds.size)
        CATransaction.begin(); CATransaction.setDisableActions(true)
        clonesLayer.bounds = CGRect(x: 0, y: 0, width: fill.height, height: fill.width)
        clonesLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        clonesLayer.setAffineTransform(CloneFrontLayout.uprightMirrored)
        for p in players { p.layer.frame = clonesLayer.bounds }
        CATransaction.commit()
    }

    func show(_ takes: [CloneTake]) {
        clear()
        for take in takes where FileManager.default.fileExists(atPath: take.alphaURL.path) {
            let video = AVPlayer(url: take.alphaURL)
            video.isMuted = true
            let sound = AVPlayer(url: take.videoURL)
            sound.isMuted = !CloneSupport.headphonesConnected
            let layer = AVPlayerLayer(player: video)
            layer.videoGravity = .resize
            layer.pixelBufferAttributes = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            layer.frame = clonesLayer.bounds
            clonesLayer.addSublayer(layer)
            players.append((video, sound, layer))
        }
        setNeedsLayout()
    }

    func play() {
        let start = CMClockGetTime(CMClockGetHostTimeClock()) + CMTime(seconds: 0.1, preferredTimescale: 600)
        for p in players {
            for player in [p.video, p.sound] {
                player.automaticallyWaitsToMinimizeStalling = false
                player.seek(to: .zero, toleranceBefore: .zero, toleranceAfter: .zero)
                player.setRate(1, time: .invalid, atHostTime: start)
            }
        }
    }

    func stop() {
        for p in players { p.video.pause(); p.sound.pause(); p.video.seek(to: .zero); p.sound.seek(to: .zero) }
    }

    func refreshSound() { for p in players { p.sound.isMuted = !CloneSupport.headphonesConnected } }

    func clear() {
        for p in players { p.video.pause(); p.sound.pause(); p.layer.removeFromSuperlayer() }
        players = []
    }
}

enum CloneFrontLayout {
    /// The 9:16 portrait picture scaled to fill `screen` (like the preview's aspect fill).
    static func fill(screen: CGSize) -> CGSize {
        let aspect: CGFloat = 9.0 / 16.0
        guard screen.width > 0, screen.height > 0 else { return .zero }
        return screen.width / screen.height > aspect
            ? CGSize(width: screen.width, height: screen.width / aspect)
            : CGSize(width: screen.height * aspect, height: screen.height)
    }

    /// Landscape (sensor) clone video -> upright on screen (quarter turn clockwise), then mirrored like the
    /// selfie preview.
    static let uprightMirrored = CGAffineTransform(rotationAngle: .pi / 2).concatenating(CGAffineTransform(scaleX: -1, y: 1))
}
