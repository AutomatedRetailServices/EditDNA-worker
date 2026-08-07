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

    func record(completion: @escaping (Result<URL, Error>) -> Void) {
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
                connection.isVideoMirrored = cameraPosition == .front
            }
        }
        isRecording = true
        movieOutput.startRecording(to: url, recordingDelegate: self)
    }

    func stopRecording() {
        guard movieOutput.isRecording else { return }
        movieOutput.stopRecording()
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

                if !self.session.outputs.contains(self.movieOutput), self.session.canAddOutput(self.movieOutput) {
                    self.session.addOutput(self.movieOutput)
                }

                guard self.session.outputs.contains(self.movieOutput) else {
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
