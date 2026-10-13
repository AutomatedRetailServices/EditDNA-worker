import AVFoundation
import Foundation

/// Sound for the editor preview. The default iOS audio session ("solo ambient") is silenced by the
/// ring/silent switch, so with the switch on silent the video played with no sound until a voice-over
/// recording changed the session (TestFlight build 10). The editor now asks for normal playback sound.
enum EditorAudio {
    static func preparePlayback() {
        let session = AVAudioSession.sharedInstance()
        do {
            if session.category != .playback {
                try session.setCategory(.playback, mode: .moviePlayback, options: [])
            }
            try session.setActive(true)
        } catch {
            CutSellDiagnostics.log("audio_session_failed", ["reason": String(describing: type(of: error))])
        }
    }
}

/// Voice-over takes recorded on this iPhone, kept by asset id so the preview can play them on their
/// own track (the server keeps the uploaded copy; this is only for playback here).
enum VoiceOverStore {
    private static var folder: URL {
        let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell/VoiceOvers", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    static func url(for assetID: String) -> URL? {
        let url = folder.appendingPathComponent("\(safe(assetID)).m4a")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    static func save(_ file: URL, assetID: String) {
        let destination = folder.appendingPathComponent("\(safe(assetID)).m4a")
        try? FileManager.default.removeItem(at: destination)
        try? FileManager.default.copyItem(at: file, to: destination)
    }

    private static func safe(_ id: String) -> String {
        String(id.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) || $0 == "_" || $0 == "-" })
    }
}

/// Real waveform of an audio file: `count` loudness values between 0 and 1.
enum Waveform {
    private static let cache = NSCache<NSString, NSArray>()

    static func samples(url: URL, count: Int) async -> [Float] {
        let key = "\(url.path)|\(count)" as NSString
        if let hit = cache.object(forKey: key) as? [Float] { return hit }
        let result = await Task.detached(priority: .userInitiated) { () -> [Float] in
            guard let file = try? AVAudioFile(forReading: url) else { return [] }
            let format = file.processingFormat
            let frames = AVAudioFrameCount(file.length)
            guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { return [] }
            do { try file.read(into: buffer) } catch { return [] }
            guard let data = buffer.floatChannelData?[0] else { return [] }
            let total = Int(buffer.frameLength)
            let bucket = max(1, total / max(1, count))
            var values: [Float] = []
            values.reserveCapacity(count)
            var index = 0
            while index < total && values.count < count {
                let end = min(total, index + bucket)
                var sum: Float = 0
                for i in index..<end { sum += data[i] * data[i] }
                values.append(sqrt(sum / Float(max(1, end - index))))
                index = end
            }
            let peak = max(0.0001, values.max() ?? 1)
            return values.map { min(1, max(0.06, $0 / peak)) }
        }.value
        cache.setObject(result as NSArray, forKey: key)
        return result
    }
}

/// "Enhance voice" (Figma Editor v4/4): done on this iPhone before upload. Cuts rumble below 90 Hz,
/// evens out loud and quiet words (compressor) and brings the take to a clear, even level.
enum VoiceEnhancer {
    static func enhance(_ input: URL) async -> URL? {
        await Task.detached(priority: .userInitiated) { () -> URL? in
            do {
                let source = try AVAudioFile(forReading: input)
                let format = source.processingFormat
                let engine = AVAudioEngine()
                let player = AVAudioPlayerNode()
                let eq = AVAudioUnitEQ(numberOfBands: 2)
                eq.bands[0].filterType = .highPass
                eq.bands[0].frequency = 90
                eq.bands[0].bypass = false
                eq.bands[1].filterType = .parametric
                eq.bands[1].frequency = 3000
                eq.bands[1].bandwidth = 1.2
                eq.bands[1].gain = 3
                eq.bands[1].bypass = false
                let compressor = AVAudioUnitEffect(audioComponentDescription: AudioComponentDescription(
                    componentType: kAudioUnitType_Effect, componentSubType: kAudioUnitSubType_DynamicsProcessor,
                    componentManufacturer: kAudioUnitManufacturer_Apple, componentFlags: 0, componentFlagsMask: 0))
                engine.attach(player); engine.attach(eq); engine.attach(compressor)
                engine.connect(player, to: eq, format: format)
                engine.connect(eq, to: compressor, format: format)
                engine.connect(compressor, to: engine.mainMixerNode, format: format)
                let au = compressor.audioUnit
                AudioUnitSetParameter(au, kDynamicsProcessorParam_Threshold, kAudioUnitScope_Global, 0, -24, 0)
                AudioUnitSetParameter(au, kDynamicsProcessorParam_HeadRoom, kAudioUnitScope_Global, 0, 6, 0)
                AudioUnitSetParameter(au, kDynamicsProcessorParam_OverallGain, kAudioUnitScope_Global, 0, 8, 0)
                try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
                try engine.start()
                player.scheduleFile(source, at: nil)
                player.play()

                let output = FileManager.default.temporaryDirectory.appendingPathComponent("voiceover-enhanced-\(UUID().uuidString).m4a")
                let settings: [String: Any] = [
                    AVFormatIDKey: Int(kAudioFormatMPEG4AAC), AVSampleRateKey: format.sampleRate,
                    AVNumberOfChannelsKey: Int(format.channelCount), AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
                ]
                let target = try AVAudioFile(forWriting: output, settings: settings,
                                             commonFormat: .pcmFormatFloat32, interleaved: false)
                guard let buffer = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat,
                                                    frameCapacity: engine.manualRenderingMaximumFrameCount) else { return nil }
                while engine.manualRenderingSampleTime < source.length {
                    let remaining = source.length - engine.manualRenderingSampleTime
                    let frames = min(AVAudioFrameCount(remaining), buffer.frameCapacity)
                    let status = try engine.renderOffline(frames, to: buffer)
                    guard status == .success else { break }
                    try target.write(from: buffer)
                }
                player.stop(); engine.stop()
                return output
            } catch {
                return nil
            }
        }.value
    }
}
