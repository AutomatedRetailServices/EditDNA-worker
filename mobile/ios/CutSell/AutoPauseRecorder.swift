@preconcurrency import AVFoundation
import Foundation
import SoundAnalysis
import Vision

/// Record with auto-pause (PO 8 Nov 2026). The camera writes one temporary file while it listens (SoundAnalysis
/// speech + the sound level) and looks for a person (Vision, ~5 times a second). When recording stops, only the
/// parts `AutoPausePlan.keep` chooses are joined into the final video WITHOUT re-encoding (same resolution and
/// format as Record), and the temporary file is deleted. Front and back camera work the same.
final class AutoPauseRecorder: NSObject, @unchecked Sendable {
    struct Live: Equatable {
        var saving: Bool = true          // "● REC" (true) or "Waiting for voice…" (false)
        var savedSeconds: Double = 0
        var heardVoice = false
    }

    struct Outcome {
        var url: URL
        var recordedSeconds: Double
        var keptSeconds: Double
        var heardVoice: Bool
        var autoStopped: Bool
        var originalBytes: Int64
        var finalBytes: Int64
    }

    let videoOutput = AVCaptureVideoDataOutput()
    let audioOutput = AVCaptureAudioDataOutput()
    private let queue = DispatchQueue(label: "ai.cutsell.autopause.capture")
    private let visionQueue = DispatchQueue(label: "ai.cutsell.autopause.vision")

    // Writer (touched on `queue` only).
    private var writer: AVAssetWriter?
    private var videoIn: AVAssetWriterInput?
    private var audioIn: AVAssetWriterInput?
    private var startPTS: CMTime?
    private var lastPTS: Double = 0
    private var recording = false
    private var fileURL: URL?

    // Analysis (touched on `queue`, except the Vision callback which hops back to `queue`).
    private var analyzer: SNAudioStreamAnalyzer?
    private var analyzerFormat: AVAudioFormat?
    private var analyzedFrames: AVAudioFramePosition = 0
    private var audioStartSeconds: Double?
    private let speechObserver = SpeechObserver()
    private var loudRanges: [AutoPausePlan.Range] = []
    private var presenceSamples: [(time: Double, present: Bool)] = []
    private var lastVisionTime: Double = -1
    private var visionBusy = false
    private var autoStopped = false

    var brollMode = false
    /// Called on the main queue ~4 times a second while recording.
    var onLive: ((Live) -> Void)?
    /// Called on the main queue when one minute passed without voice (with a person on camera).
    var onAutoStop: (() -> Void)?

    override init() {
        super.init()
        videoOutput.alwaysDiscardsLateVideoFrames = false
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        audioOutput.setSampleBufferDelegate(self, queue: queue)
    }

    /// Orientation / mirroring of what is written, like Record.
    func prepareConnections(mirrored: Bool) {
        if let c = videoOutput.connection(with: .video) {
            if c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
            if c.isVideoStabilizationSupported { c.preferredVideoStabilizationMode = .auto }
            if c.isVideoMirroringSupported {
                c.automaticallyAdjustsVideoMirroring = false
                c.isVideoMirrored = mirrored
            }
        }
    }

    func start(to url: URL) throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let vSettings = videoOutput.recommendedVideoSettingsForAssetWriter(writingTo: .mov)
        let aSettings = audioOutput.recommendedAudioSettingsForAssetWriter(writingTo: .mov) as? [String: Any]
        let v = AVAssetWriterInput(mediaType: .video, outputSettings: vSettings)
        v.expectsMediaDataInRealTime = true
        let a = AVAssetWriterInput(mediaType: .audio, outputSettings: aSettings)
        a.expectsMediaDataInRealTime = true
        guard writer.canAdd(v), writer.canAdd(a) else { throw CameraController.CameraError.recordingFailed }
        writer.add(v); writer.add(a)
        queue.sync {
            self.writer = writer; self.videoIn = v; self.audioIn = a
            self.startPTS = nil; self.lastPTS = 0; self.fileURL = url
            self.analyzer = nil; self.analyzerFormat = nil; self.analyzedFrames = 0; self.audioStartSeconds = nil
            self.speechObserver.reset()
            self.loudRanges = []; self.presenceSamples = []; self.lastVisionTime = -1; self.autoStopped = false
            self.recording = true
        }
    }

    /// Stops, keeps only the chosen parts and returns the final file (or the whole one if nothing is cut).
    func stop() async throws -> Outcome {
        let (writer, url) = queue.sync { () -> (AVAssetWriter?, URL?) in
            self.recording = false
            self.videoIn?.markAsFinished(); self.audioIn?.markAsFinished()
            return (self.writer, self.fileURL)
        }
        guard let writer, let url else { throw CameraController.CameraError.recordingFailed }
        if writer.status == .writing { await writer.finishWriting() }
        guard writer.status == .completed else { throw writer.error ?? CameraController.CameraError.recordingFailed }
        let (voice, presence, duration, stopped, broll) = queue.sync {
            (self.voiceRanges(), AutoPausePlan.presence(samples: self.presenceSamples, duration: self.lastPTS),
             self.lastPTS, self.autoStopped, self.brollMode)
        }
        let asset = AVURLAsset(url: url)
        let total = (try? await asset.load(.duration).seconds) ?? duration
        let keep = AutoPausePlan.keep(voice: voice, presence: presence, duration: total, brollMode: broll)
        let kept = AutoPausePlan.savedSeconds(keep)
        let originalBytes = Self.size(url)
        // Nothing (or almost nothing) to remove: the recording is used as it is.
        if keep.count == 1, keep[0].start < 0.05, keep[0].end > total - 0.05 {
            return Outcome(url: url, recordedSeconds: total, keptSeconds: total, heardVoice: !voice.isEmpty,
                           autoStopped: stopped, originalBytes: originalBytes, finalBytes: originalBytes)
        }
        let out = try await Self.join(asset: asset, ranges: keep)
        try? FileManager.default.removeItem(at: url)
        return Outcome(url: out, recordedSeconds: total, keptSeconds: kept, heardVoice: !voice.isEmpty,
                       autoStopped: stopped, originalBytes: originalBytes, finalBytes: Self.size(out))
    }

    /// Joins the kept parts into one .mov without re-encoding (passthrough).
    static func join(asset: AVURLAsset, ranges: [AutoPausePlan.Range]) async throws -> URL {
        let composition = AVMutableComposition()
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        guard let sourceVideo = videoTracks.first,
              let video = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)
        else { throw CameraController.CameraError.recordingFailed }
        video.preferredTransform = try await sourceVideo.load(.preferredTransform)
        let audio = audioTracks.isEmpty ? nil
            : composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
        var cursor = CMTime.zero
        for r in ranges {
            let range = CMTimeRange(start: CMTime(seconds: r.start, preferredTimescale: 600),
                                    end: CMTime(seconds: r.end, preferredTimescale: 600))
            try video.insertTimeRange(range, of: sourceVideo, at: cursor)
            if let audio, let sourceAudio = audioTracks.first { try audio.insertTimeRange(range, of: sourceAudio, at: cursor) }
            cursor = cursor + range.duration
        }
        let out = try persistentMediaURL(extension: "mov")
        try? FileManager.default.removeItem(at: out)
        guard let export = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetPassthrough) else {
            throw CameraController.CameraError.recordingFailed
        }
        export.outputURL = out
        export.outputFileType = .mov
        await export.export()
        guard export.status == .completed else { throw export.error ?? CameraController.CameraError.recordingFailed }
        return out
    }

    static func size(_ url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber)?.int64Value ?? 0
    }

    // MARK: Voice

    /// Voice = speech from Apple's classifier, or clearly loud sound (when unsure, keep).
    private func voiceRanges() -> [AutoPausePlan.Range] {
        AutoPausePlan.merged(speechObserver.ranges + loudRanges)
    }

    private func analyze(_ sample: CMSampleBuffer, at seconds: Double) {
        guard let buffer = Self.pcmBuffer(from: sample) else { return }
        if analyzer == nil || analyzerFormat != buffer.format {
            let a = SNAudioStreamAnalyzer(format: buffer.format)
            if let request = try? SNClassifySoundRequest(classifierIdentifier: .version1) {
                request.overlapFactor = 0.75            // a result about every 0.25 s
                try? a.add(request, withObserver: speechObserver)
            }
            analyzer = a; analyzerFormat = buffer.format; analyzedFrames = 0
            audioStartSeconds = seconds
            speechObserver.sampleRate = buffer.format.sampleRate
            speechObserver.offset = seconds
        }
        analyzer?.analyze(buffer, atAudioFramePosition: analyzedFrames)
        analyzedFrames += AVAudioFramePosition(buffer.frameLength)
        // Sound level: loud enough to be a voice close to the phone (~ -26 dBFS).
        let level = Self.rmsDB(buffer)
        let length = Double(buffer.frameLength) / buffer.format.sampleRate
        if level > -26 { loudRanges.append(AutoPausePlan.Range(start: seconds, end: seconds + length)) }
    }

    static func rmsDB(_ buffer: AVAudioPCMBuffer) -> Float {
        guard let data = buffer.floatChannelData, buffer.frameLength > 0 else { return -120 }
        let n = Int(buffer.frameLength)
        var sum: Float = 0
        for i in 0..<n { sum += data[0][i] * data[0][i] }
        let rms = (sum / Float(n)).squareRoot()
        return rms > 0 ? 20 * log10(rms) : -120
    }

    /// Microphone sample buffer -> mono Float32 PCM (what SoundAnalysis and the level meter read).
    static func pcmBuffer(from sample: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard let description = CMSampleBufferGetFormatDescription(sample),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee else { return nil }
        let frames = CMSampleBufferGetNumSamples(sample)
        guard frames > 0, let format = AVAudioFormat(standardFormatWithSampleRate: asbd.mSampleRate, channels: 1),
              let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)),
              let block = CMSampleBufferGetDataBuffer(sample) else { return nil }
        var length = 0
        var pointer: UnsafeMutablePointer<Int8>?
        guard CMBlockBufferGetDataPointer(block, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &length,
                                          dataPointerOut: &pointer) == kCMBlockBufferNoErr, let pointer else { return nil }
        let channels = max(1, Int(asbd.mChannelsPerFrame))
        let isFloat = asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0
        let bits = Int(asbd.mBitsPerChannel)
        guard let dst = out.floatChannelData?[0] else { return nil }
        let count = min(frames, length / max(1, channels * bits / 8))
        if isFloat && bits == 32 {
            pointer.withMemoryRebound(to: Float.self, capacity: count * channels) { src in
                for i in 0..<count { dst[i] = src[i * channels] }
            }
        } else if bits == 16 {
            pointer.withMemoryRebound(to: Int16.self, capacity: count * channels) { src in
                for i in 0..<count { dst[i] = Float(src[i * channels]) / 32768 }
            }
        } else {
            return nil
        }
        out.frameLength = AVAudioFrameCount(count)
        return out
    }

    // MARK: Person on camera (B-roll)

    private func lookForPerson(_ pixels: CVPixelBuffer, at seconds: Double) {
        guard seconds - lastVisionTime >= 0.2, !visionBusy else { return }
        lastVisionTime = seconds
        visionBusy = true
        visionQueue.async { [weak self] in
            let faces = VNDetectFaceRectanglesRequest()
            let people = VNDetectHumanRectanglesRequest()
            people.upperBodyOnly = true
            let handler = VNImageRequestHandler(cvPixelBuffer: pixels, orientation: .up)
            try? handler.perform([faces, people])
            let present = !(faces.results ?? []).isEmpty || !(people.results ?? []).isEmpty
            self?.queue.async {
                guard let self else { return }
                self.visionBusy = false
                self.presenceSamples.append((time: seconds, present: present))
            }
        }
    }

    // MARK: Live state

    private var lastLive: Double = -1

    private func publish(now: Double) {
        guard now - lastLive >= 0.25 else { return }
        lastLive = now
        let voice = voiceRanges()
        let lastVoice = voice.last?.end
        let presence = AutoPausePlan.presence(samples: presenceSamples, duration: now)
        let person = presenceSamples.last?.present ?? true
        let saving = AutoPausePlan.isSavingNow(now: now, lastVoice: lastVoice, personOnCamera: person, brollMode: brollMode)
        let keep = AutoPausePlan.keep(voice: voice, presence: presence, duration: now, brollMode: brollMode)
        let live = Live(saving: saving, savedSeconds: AutoPausePlan.savedSeconds(keep), heardVoice: !voice.isEmpty)
        let stop = !autoStopped && AutoPausePlan.shouldAutoStop(now: now, lastVoice: lastVoice, presence: presence,
                                                                brollMode: brollMode)
        if stop { autoStopped = true }
        let onLive = self.onLive, onAutoStop = self.onAutoStop
        DispatchQueue.main.async {
            onLive?(live)
            if stop { onAutoStop?() }
        }
    }
}

extension AutoPauseRecorder: AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureAudioDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard recording, let writer else { return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        if output === videoOutput {
            if startPTS == nil {
                guard writer.startWriting() else { return }
                writer.startSession(atSourceTime: pts)
                startPTS = pts
            }
            guard let start = startPTS else { return }
            let seconds = (pts - start).seconds
            lastPTS = max(lastPTS, seconds)
            if let videoIn, videoIn.isReadyForMoreMediaData { videoIn.append(sampleBuffer) }
            if let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) { lookForPerson(pixels, at: seconds) }
            publish(now: seconds)
        } else if output === audioOutput {
            guard let start = startPTS, pts >= start else { return }
            if let audioIn, audioIn.isReadyForMoreMediaData { audioIn.append(sampleBuffer) }
            analyze(sampleBuffer, at: (pts - start).seconds)
        }
    }
}

/// Collects the windows Apple's sound classifier calls "speech".
final class SpeechObserver: NSObject, SNResultsObserving, @unchecked Sendable {
    private let lock = NSLock()
    private var found: [AutoPausePlan.Range] = []
    var sampleRate: Double = 48_000
    var offset: Double = 0

    var ranges: [AutoPausePlan.Range] { lock.lock(); defer { lock.unlock() }; return found }

    func reset() { lock.lock(); found = []; lock.unlock() }

    func request(_ request: SNRequest, didProduce result: SNResult) {
        guard let result = result as? SNClassificationResult,
              let speech = result.classification(forIdentifier: "speech"), speech.confidence >= 0.5 else { return }
        let start = result.timeRange.start.seconds, end = (result.timeRange.start + result.timeRange.duration).seconds
        guard start.isFinite, end.isFinite else { return }
        lock.lock()
        found.append(AutoPausePlan.Range(start: offset + start, end: offset + end))
        lock.unlock()
    }
}
