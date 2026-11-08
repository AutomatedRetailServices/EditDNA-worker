@preconcurrency import AVFoundation
import Foundation

@MainActor
final class CameraController: NSObject, ObservableObject {
    enum CameraError: LocalizedError {
        case permissionDenied
        case configurationFailed
        case recordingFailed

        var errorDescription: String? {
            switch self {
            case .permissionDenied: return "Camera and microphone permission are required to record."
            case .configurationFailed: return "CutSell could not configure the camera."
            case .recordingFailed: return "The recording could not be saved."
            }
        }
    }

    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "ai.cutsell.camera.session")
    private let movieOutput = AVCaptureMovieFileOutput()
    private var videoInput: AVCaptureDeviceInput?
    private var completion: ((Result<URL, Error>) -> Void)?

    @Published private(set) var isConfigured = false
    @Published private(set) var isRecording = false
    @Published private(set) var cameraPosition: AVCaptureDevice.Position = .back
    @Published var errorMessage: String?
    /// Auto-pause by voice (PO 8 Nov 2026): on by default; off = Record exactly as before (one movie file).
    @Published var autoPauseOn: Bool = UserDefaults.standard.object(forKey: CameraController.autoPauseKey) as? Bool ?? true {
        didSet { UserDefaults.standard.set(autoPauseOn, forKey: Self.autoPauseKey) }
    }
    static let autoPauseKey = "cutsell.camera.autopause"
    /// B-roll: auto-pause off for this recording, everything is kept.
    @Published var brollOn = false
    /// While recording with auto-pause: "● REC" or "Waiting for voice…", and the seconds kept so far.
    @Published private(set) var live = AutoPauseRecorder.Live()
    /// The last auto-pause recording: what it kept (for the "no voice" / "stopped" notices and diagnostics).
    @Published private(set) var lastOutcome: AutoPauseRecorder.Outcome?
    /// One minute without voice stopped the recording.
    @Published var autoStoppedNotice = false
    private let pauseRecorder = AutoPauseRecorder()
    /// The outputs in the session: the movie file (auto-pause off) or the frame + sound outputs (auto-pause on).
    private var usingPauseOutputs = false
    /// Recording through the auto-pause recorder right now.
    private var pauseRecording = false
    /// Mirror (FICHA_03): off by default so the saved file shows text the right way round; when on,
    /// the preview and the recorded file are flipped the same way. The last choice is remembered.
    @Published var mirrorOn: Bool = UserDefaults.standard.bool(forKey: CameraController.mirrorKey) {
        didSet { UserDefaults.standard.set(mirrorOn, forKey: Self.mirrorKey) }
    }
    static let mirrorKey = "cutsell.camera.mirror"

    func prepare() async {
        guard !isConfigured else {
            startSession()
            return
        }
        let cameraAllowed = await AVCaptureDevice.requestAccess(for: .video)
        let micAllowed = await AVCaptureDevice.requestAccess(for: .audio)
        guard cameraAllowed && micAllowed else {
            errorMessage = CameraError.permissionDenied.localizedDescription
            return
        }
        do {
            try await configure(position: .back)
            usingPauseOutputs = autoPauseOn
            isConfigured = true
            startSession()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func startSession() {
        sessionQueue.async { [session] in
            if !session.isRunning { session.startRunning() }
        }
    }

    func stopSession() {
        sessionQueue.async { [session] in
            if session.isRunning { session.stopRunning() }
        }
    }

    func flipCamera() async {
        guard !isRecording else { return }
        let next: AVCaptureDevice.Position = cameraPosition == .back ? .front : .back
        do {
            try await configure(position: next)
            cameraPosition = next
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Whether the current camera has a light (the front camera usually does not).
    var hasTorch: Bool {
        videoInput?.device.hasTorch == true
    }

    /// Turns the camera light on or off (used as "Flash" while recording video).
    func setTorch(_ on: Bool) {
        guard let device = videoInput?.device, device.hasTorch else { return }
        sessionQueue.async {
            do {
                try device.lockForConfiguration()
                device.torchMode = on ? .on : .off
                device.unlockForConfiguration()
            } catch {
                // The light is optional; recording works without it.
            }
        }
    }

    /// Auto-pause on (and not B-roll) uses the auto-pause recorder; otherwise the movie file, as before.
    var usesAutoPause: Bool { autoPauseOn && !brollOn }

    /// Switches the session's outputs when Auto-pause is turned on / off (never while recording).
    func applyAutoPauseChoice() async {
        guard isConfigured, !isRecording else { return }
        let wanted = autoPauseOn
        guard wanted != usingPauseOutputs else { return }
        let session = self.session, movie = self.movieOutput
        let video = pauseRecorder.videoOutput, audio = pauseRecorder.audioOutput
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            sessionQueue.async {
                session.beginConfiguration()
                Self.installOutputs(session: session, movie: movie, video: video, audio: audio, autoPause: wanted)
                session.commitConfiguration()
                done.resume()
            }
        }
        usingPauseOutputs = wanted
    }

    /// Called on the session queue inside a configuration block.
    nonisolated private static func installOutputs(session: AVCaptureSession, movie: AVCaptureMovieFileOutput,
                                                   video: AVCaptureVideoDataOutput, audio: AVCaptureAudioDataOutput,
                                                   autoPause: Bool) {
        if autoPause {
            if session.outputs.contains(movie) { session.removeOutput(movie) }
            if !session.outputs.contains(video), session.canAddOutput(video) { session.addOutput(video) }
            if !session.outputs.contains(audio), session.canAddOutput(audio) { session.addOutput(audio) }
        } else {
            if session.outputs.contains(video) { session.removeOutput(video) }
            if session.outputs.contains(audio) { session.removeOutput(audio) }
            if !session.outputs.contains(movie), session.canAddOutput(movie) { session.addOutput(movie) }
        }
    }

    func record(completion: @escaping (Result<URL, Error>) -> Void) {
        if usingPauseOutputs {
            recordWithAutoPause(completion: completion)
            return
        }
        guard isConfigured, !movieOutput.isRecording else { return }
        self.completion = completion
        let url: URL
        do {
            url = try persistentMediaURL(extension: "mov")
        } catch {
            self.completion = nil
            errorMessage = error.localizedDescription
            return
        }
        try? FileManager.default.removeItem(at: url)
        if let connection = movieOutput.connection(with: .video) {
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .auto
            }
            if connection.isVideoMirroringSupported {
                // What is saved is what the preview shows: flipped only with Mirror on.
                connection.automaticallyAdjustsVideoMirroring = false
                connection.isVideoMirrored = mirrorOn
            }
        }
        isRecording = true
        movieOutput.startRecording(to: url, recordingDelegate: self)
    }

    func stopRecording() {
        if pauseRecording {
            pauseRecording = false
            Task { await finishAutoPause() }
            return
        }
        guard movieOutput.isRecording else { return }
        movieOutput.stopRecording()
    }

    private func recordWithAutoPause(completion: @escaping (Result<URL, Error>) -> Void) {
        guard isConfigured, !pauseRecording else { return }
        self.completion = completion
        do {
            let url = try persistentMediaURL(extension: "mov")
            try? FileManager.default.removeItem(at: url)
            pauseRecorder.prepareConnections(mirrored: mirrorOn)
            pauseRecorder.brollMode = brollOn
            pauseRecorder.onLive = { [weak self] live in self?.live = live }
            pauseRecorder.onAutoStop = { [weak self] in
                guard let self, self.pauseRecording else { return }
                self.autoStoppedNotice = true
                self.stopRecording()
            }
            try pauseRecorder.start(to: url)
        } catch {
            self.completion = nil
            errorMessage = error.localizedDescription
            return
        }
        live = AutoPauseRecorder.Live()
        lastOutcome = nil
        autoStoppedNotice = false
        pauseRecording = true
        isRecording = true
    }

    private func finishAutoPause() async {
        let callback = completion
        completion = nil
        do {
            let outcome = try await pauseRecorder.stop()
            lastOutcome = outcome
            isRecording = false
            CutSellDiagnostics.log("autopause_recording", [
                "recorded_s": String(format: "%.1f", outcome.recordedSeconds),
                "kept_s": String(format: "%.1f", outcome.keptSeconds),
                "original_mb": String(format: "%.1f", Double(outcome.originalBytes) / 1_048_576),
                "final_mb": String(format: "%.1f", Double(outcome.finalBytes) / 1_048_576),
                "voice": outcome.heardVoice ? "yes" : "no",
                "auto_stopped": outcome.autoStopped ? "yes" : "no",
                "broll": brollOn ? "yes" : "no",
            ])
            callback?(.success(outcome.url))
        } catch {
            isRecording = false
            callback?(.failure(error))
        }
    }

    private func configure(position: AVCaptureDevice.Position) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            sessionQueue.async { [weak self] in
                guard let self else {
                    continuation.resume(throwing: CameraError.configurationFailed)
                    return
                }
                self.session.beginConfiguration()
                self.session.sessionPreset = .high
                defer { self.session.commitConfiguration() }

                if let old = self.videoInput {
                    self.session.removeInput(old)
                }

                guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
                      let input = try? AVCaptureDeviceInput(device: videoDevice),
                      self.session.canAddInput(input) else {
                    continuation.resume(throwing: CameraError.configurationFailed)
                    return
                }
                self.session.addInput(input)
                self.videoInput = input

                if !self.session.inputs.contains(where: { ($0 as? AVCaptureDeviceInput)?.device.hasMediaType(.audio) == true }) {
                    if let audioDevice = AVCaptureDevice.default(for: .audio),
                       let audioInput = try? AVCaptureDeviceInput(device: audioDevice),
                       self.session.canAddInput(audioInput) {
                        self.session.addInput(audioInput)
                    }
                }

                let autoPause = UserDefaults.standard.object(forKey: CameraController.autoPauseKey) as? Bool ?? true
                Self.installOutputs(session: self.session, movie: self.movieOutput, video: self.pauseRecorder.videoOutput,
                                    audio: self.pauseRecorder.audioOutput, autoPause: autoPause)
                let ready = autoPause ? self.session.outputs.contains(self.pauseRecorder.videoOutput)
                                      : self.session.outputs.contains(self.movieOutput)
                guard ready else {
                    continuation.resume(throwing: CameraError.configurationFailed)
                    return
                }
                continuation.resume(returning: ())
            }
        }
    }
}

extension CameraController: AVCaptureFileOutputRecordingDelegate {
    nonisolated func fileOutput(
        _ output: AVCaptureFileOutput,
        didFinishRecordingTo outputFileURL: URL,
        from connections: [AVCaptureConnection],
        error: Error?
    ) {
        Task { @MainActor in
            self.isRecording = false
            let callback = self.completion
            self.completion = nil
            if let error {
                callback?(.failure(error))
            } else if FileManager.default.fileExists(atPath: outputFileURL.path) {
                callback?(.success(outputFileURL))
            } else {
                callback?(.failure(CameraError.recordingFailed))
            }
        }
    }
}
