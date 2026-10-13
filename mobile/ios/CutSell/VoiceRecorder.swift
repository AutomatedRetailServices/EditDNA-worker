import AVFoundation
import Foundation

/// Records one voice-over take with the microphone as an .m4a (AAC) file, the format the
/// server's voice-over upload already accepts (`ALLOWED_VOICE_OVER_AUDIO_EXTENSIONS = {".m4a"}`).
@MainActor
final class VoiceRecorder: ObservableObject {
    enum State: Equatable {
        case idle
        case recording
        case recorded(URL)
        case permissionDenied
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    @Published private(set) var elapsed: TimeInterval = 0
    @Published private(set) var isPaused = false

    private var recorder: AVAudioRecorder?
    private var timer: Timer?

    var recordedURL: URL? {
        if case .recorded(let url) = state { return url }
        return nil
    }

    /// Asks for the microphone the first time, then starts a new take (any previous take is discarded).
    func start() async {
        guard await Self.requestPermission() else {
            state = .permissionDenied
            return
        }
        discardTake()
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .spokenAudio, options: [.defaultToSpeaker, .allowBluetooth])
            try session.setActive(true)
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("voiceover-\(UUID().uuidString).m4a")
            let settings: [String: Any] = [
                AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
                AVSampleRateKey: 44_100,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
            ]
            let recorder = try AVAudioRecorder(url: url, settings: settings)
            guard recorder.record() else {
                state = .failed("The microphone couldn't start. Please try again.")
                return
            }
            self.recorder = recorder
            elapsed = 0
            state = .recording
            timer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    guard let self, let recorder = self.recorder else { return }
                    self.elapsed = recorder.currentTime
                }
            }
        } catch {
            state = .failed("The microphone couldn't start. Please try again.")
        }
    }

    /// Pauses the take (Editor v2 "Tap to pause"); `resume()` continues the same file.
    func pause() {
        guard let recorder, state == .recording, recorder.isRecording else { return }
        recorder.pause()
        isPaused = true
    }

    func resume() {
        guard let recorder, state == .recording, isPaused else { return }
        recorder.record()
        isPaused = false
    }

    /// Stops the take and keeps the file for "Use" or "Record again".
    func stop() {
        guard let recorder, state == .recording else { return }
        isPaused = false
        elapsed = recorder.currentTime
        recorder.stop()
        timer?.invalidate()
        timer = nil
        self.recorder = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        state = elapsed >= 0.3 ? .recorded(recorder.url) : .failed("That take was too short. Please record again.")
    }

    /// Deletes the current take's file and returns to the start.
    func discardTake() {
        if state == .recording { stop() }
        if let url = recordedURL { try? FileManager.default.removeItem(at: url) }
        state = .idle
        elapsed = 0
    }

    /// After "Use" uploaded the take: forget it without touching the upload.
    func finishTake() {
        if let url = recordedURL { try? FileManager.default.removeItem(at: url) }
        state = .idle
        elapsed = 0
    }

    static func requestPermission() async -> Bool {
        switch AVAudioApplication.shared.recordPermission {
        case .granted: return true
        case .denied: return false
        default:
            return await withCheckedContinuation { continuation in
                AVAudioApplication.requestRecordPermission { continuation.resume(returning: $0) }
            }
        }
    }

    static func formatted(_ seconds: TimeInterval) -> String {
        let total = Int(seconds.rounded(.down))
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}
