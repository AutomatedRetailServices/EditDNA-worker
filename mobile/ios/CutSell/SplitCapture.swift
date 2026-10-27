@preconcurrency import AVFoundation
import CoreImage
import Foundation

/// Split clone (PO 27 Oct): the camera for the two parts. It records with sample-buffer outputs and an
/// AVAssetWriter (not a movie file output) so that:
/// - part 2 can start at an exact host time, the same instant part 1 starts playing as the ghost;
/// - live frames are available for the alignment check (phone moved?) while the preview runs.
/// Exposure, white balance and focus are locked when part 1 starts and stay locked for part 2, so both
/// halves look like one shot.
@MainActor
final class SplitCaptureController: NSObject, ObservableObject {
    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "ai.cutsell.split.session")
    private let bufferQueue = DispatchQueue(label: "ai.cutsell.split.buffers")
    private let videoOutput = AVCaptureVideoDataOutput()
    private let audioOutput = AVCaptureAudioDataOutput()
    private var videoInput: AVCaptureDeviceInput?

    @Published private(set) var isConfigured = false
    @Published private(set) var isRecording = false
    @Published private(set) var cameraPosition: AVCaptureDevice.Position = .front
    @Published private(set) var isLocked = false
    @Published var errorMessage: String?
    /// Same Mirror setting as the Record camera (what is saved is what the preview shows).
    let mirrorOn = UserDefaults.standard.bool(forKey: CameraController.mirrorKey)

    /// Touched only on `bufferQueue`.
    private nonisolated(unsafe) let state = WriterState()

    func prepare() async {
        guard !isConfigured else { startSession(); return }
        configureAudioSession()
        do {
            try await configure(position: cameraPosition)
            isConfigured = true
            startSession()
        } catch {
            errorMessage = "CutSell could not configure the camera."
        }
    }

    /// Plays part 1 to headphones while the microphone records part 2 (no echo: without headphones
    /// the ghost is silent, see `SplitCloneView`). Bluetooth headphones stay in high quality (A2DP);
    /// the iPhone's own microphone records.
    private func configureAudioSession() {
        session.automaticallyConfiguresApplicationAudioSession = false
        let audio = AVAudioSession.sharedInstance()
        try? audio.setCategory(.playAndRecord, mode: .videoRecording, options: [.allowBluetoothA2DP])
        try? audio.setActive(true)
    }

    func startSession() {
        sessionQueue.async { [session] in if !session.isRunning { session.startRunning() } }
    }

    func stopSession() {
        sessionQueue.async { [session] in if session.isRunning { session.stopRunning() } }
    }

    func flipCamera() async {
        guard !isRecording, !isLocked else { return }
        let next: AVCaptureDevice.Position = cameraPosition == .back ? .front : .back
        do {
            try await configure(position: next)
            cameraPosition = next
        } catch {
            errorMessage = "CutSell could not configure the camera."
        }
    }

    /// Exposure, white balance and focus stay exactly as they are now (both parts look the same).
    func lockCamera() {
        guard let device = videoInput?.device else { return }
        sessionQueue.async {
            do {
                try device.lockForConfiguration()
                if device.isExposureModeSupported(.locked) { device.exposureMode = .locked }
                if device.isWhiteBalanceModeSupported(.locked) { device.whiteBalanceMode = .locked }
                if device.isFocusModeSupported(.locked) { device.focusMode = .locked }
                device.unlockForConfiguration()
            } catch {
                // Not fatal: the parts may differ a little in light.
            }
        }
        isLocked = true
    }

    /// Back to automatic (leaving Split, or "Retake part 1").
    func unlockCamera() {
        guard let device = videoInput?.device else { isLocked = false; return }
        sessionQueue.async {
            do {
                try device.lockForConfiguration()
                if device.isExposureModeSupported(.continuousAutoExposure) { device.exposureMode = .continuousAutoExposure }
                if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) { device.whiteBalanceMode = .continuousAutoWhiteBalance }
                if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
                device.unlockForConfiguration()
            } catch {}
        }
        isLocked = false
    }

    /// The newest camera frame (portrait, as saved), for the alignment check.
    func latestFrame() async -> CIImage? {
        await withCheckedContinuation { continuation in
            bufferQueue.async { [state] in
                continuation.resume(returning: state.latest.map { CIImage(cvPixelBuffer: $0) })
            }
        }
    }

    /// A host time a little ahead (time for the player and the writer to get ready).
    nonisolated static func hostTime(after seconds: Double) -> CMTime {
        CMTimeAdd(CMClockGetTime(CMClockGetHostTimeClock()), CMTime(seconds: seconds, preferredTimescale: 1_000_000_000))
    }

    /// Starts writing a new part. With `startHostTime`, the part begins exactly at that host instant
    /// (frames before it are dropped); otherwise at the first frame.
    func startRecording(startHostTime: CMTime? = nil) throws {
        guard isConfigured, !isRecording else { return }
        let url = try persistentMediaURL(extension: "mov")
        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let codec: AVVideoCodecType = videoOutput.availableVideoCodecTypes.contains(.hevc) ? .hevc : .h264
        // The frames' real size (portrait after the connection's rotation): the recommended settings
        // may describe the sensor's landscape size, and a writer scales frames to its own size.
        let size: CGSize = bufferQueue.sync { [state] in
            state.latest.map { CGSize(width: CVPixelBufferGetWidth($0), height: CVPixelBufferGetHeight($0)) }
        } ?? CGSize(width: 1080, height: 1920)
        var videoSettings = videoOutput.recommendedVideoSettings(forVideoCodecType: codec, assetWriterOutputFileType: .mov)
            ?? [AVVideoCodecKey: codec]
        videoSettings[AVVideoWidthKey] = Int(size.width)
        videoSettings[AVVideoHeightKey] = Int(size.height)
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoInput.expectsMediaDataInRealTime = true
        let audioSettings = audioOutput.recommendedAudioSettingsForAssetWriter(writingTo: .mov)
            ?? [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 44_100, AVNumberOfChannelsKey: 1]
        let audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
        audioInput.expectsMediaDataInRealTime = true
        guard writer.canAdd(videoInput), writer.canAdd(audioInput) else { throw CocoaError(.fileWriteUnknown) }
        writer.add(videoInput)
        writer.add(audioInput)
        guard writer.startWriting() else { throw writer.error ?? CocoaError(.fileWriteUnknown) }
        // Buffer times are on the session's clock; the start instant is a host time.
        var start: CMTime? = nil
        if let startHostTime {
            let clock = session.synchronizationClock ?? CMClockGetHostTimeClock()
            start = CMSyncConvertTime(startHostTime, from: CMClockGetHostTimeClock(), to: clock)
        }
        let pending = start
        bufferQueue.sync { [state] in
            state.writer = writer
            state.video = videoInput
            state.audio = audioInput
            state.url = url
            state.requestedStart = pending
            state.started = false
            state.lastVideoTime = .invalid
        }
        isRecording = true
    }

    /// Finishes the part and returns its file.
    func stopRecording() async throws -> URL {
        guard isRecording else { throw CocoaError(.fileWriteUnknown) }
        let (writer, video, audio, url, started, last): (AVAssetWriter?, AVAssetWriterInput?, AVAssetWriterInput?, URL?, Bool, CMTime) =
            await withCheckedContinuation { continuation in
                bufferQueue.async { [state] in
                    let values = (state.writer, state.video, state.audio, state.url, state.started, state.lastVideoTime)
                    state.writer = nil; state.video = nil; state.audio = nil
                    continuation.resume(returning: values)
                }
            }
        isRecording = false
        guard let writer, let url, started else { writer?.cancelWriting(); throw CocoaError(.fileWriteUnknown) }
        video?.markAsFinished()
        audio?.markAsFinished()
        if last.isValid { writer.endSession(atSourceTime: last) }
        await writer.finishWriting()
        guard writer.status == .completed else { throw writer.error ?? CocoaError(.fileWriteUnknown) }
        return url
    }

    private func configure(position: AVCaptureDevice.Position) async throws {
        let mirror = mirrorOn
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            sessionQueue.async { [weak self] in
                guard let self else { continuation.resume(throwing: CocoaError(.featureUnsupported)); return }
                let session = self.session
                session.beginConfiguration()
                defer { session.commitConfiguration() }
                if session.canSetSessionPreset(.hd1920x1080) { session.sessionPreset = .hd1920x1080 } else { session.sessionPreset = .high }
                if let old = self.videoInput { session.removeInput(old) }
                guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
                      let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) else {
                    continuation.resume(throwing: CocoaError(.featureUnsupported)); return
                }
                session.addInput(input)
                self.videoInput = input
                if !session.inputs.contains(where: { ($0 as? AVCaptureDeviceInput)?.device.hasMediaType(.audio) == true }),
                   let mic = AVCaptureDevice.default(for: .audio), let micInput = try? AVCaptureDeviceInput(device: mic),
                   session.canAddInput(micInput) {
                    session.addInput(micInput)
                }
                if !session.outputs.contains(self.videoOutput), session.canAddOutput(self.videoOutput) {
                    self.videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange]
                    self.videoOutput.alwaysDiscardsLateVideoFrames = false
                    self.videoOutput.setSampleBufferDelegate(self, queue: self.bufferQueue)
                    session.addOutput(self.videoOutput)
                }
                if !session.outputs.contains(self.audioOutput), session.canAddOutput(self.audioOutput) {
                    self.audioOutput.setSampleBufferDelegate(self, queue: self.bufferQueue)
                    session.addOutput(self.audioOutput)
                }
                // Portrait frames, flipped only with Mirror on (the same as the preview).
                if let connection = self.videoOutput.connection(with: .video) {
                    if connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }
                    if connection.isVideoMirroringSupported {
                        connection.automaticallyAdjustsVideoMirroring = false
                        connection.isVideoMirrored = mirror
                    }
                    if connection.isVideoStabilizationSupported { connection.preferredVideoStabilizationMode = .off }
                }
                continuation.resume(returning: ())
            }
        }
    }
}

/// What the buffer queue writes into.
private final class WriterState {
    var writer: AVAssetWriter?
    var video: AVAssetWriterInput?
    var audio: AVAssetWriterInput?
    var url: URL?
    var requestedStart: CMTime?
    var started = false
    var lastVideoTime: CMTime = .invalid
    var latest: CVPixelBuffer?
}

extension SplitCaptureController: AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureAudioDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        let state = self.state
        let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let isVideo = output is AVCaptureVideoDataOutput
        if isVideo, let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) { state.latest = pixels }
        guard let writer = state.writer, writer.status == .writing else { return }
        if !state.started {
            // The part starts on a video frame: at the requested instant, or at the first frame.
            guard isVideo else { return }
            if let start = state.requestedStart, CMTimeCompare(time, start) < 0 { return }
            writer.startSession(atSourceTime: state.requestedStart ?? time)
            state.started = true
        }
        if isVideo {
            if let input = state.video, input.isReadyForMoreMediaData, input.append(sampleBuffer) {
                state.lastVideoTime = CMTimeAdd(time, CMSampleBufferGetDuration(sampleBuffer).isValid
                                                ? CMSampleBufferGetDuration(sampleBuffer) : CMTime(value: 1, timescale: 30))
            }
        } else if let input = state.audio, input.isReadyForMoreMediaData {
            input.append(sampleBuffer)
        }
    }
}
