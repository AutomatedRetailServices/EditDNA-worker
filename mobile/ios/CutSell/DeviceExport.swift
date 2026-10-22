import AVFoundation
import CoreImage
import Photos
import SwiftUI
import UIKit

// MARK: - Settings (Figma 950:8730 / 950:8790)

/// 22 Oct: exports are made on this iPhone from the ORIGINAL in Photos (4K or 1080p), with the same cuts,
/// captions, zoom, keyframes, sounds and voices as the preview. The server export stays as the fallback.
enum ExportResolution: String, CaseIterable, Identifiable {
    case match, p720, p1080, p1440, k4
    var id: String { rawValue }
    var label: String {
        switch self {
        case .match: return "Match"
        case .p720: return "720p"
        case .p1080: return "1080p"
        case .p1440: return "1440p"
        case .k4: return "4K"
        }
    }
    /// The short side of the 9:16 frame (nil = the source's).
    var shortSide: Int? {
        switch self {
        case .match: return nil
        case .p720: return 720
        case .p1080: return 1080
        case .p1440: return 1440
        case .k4: return 2160
        }
    }
}

enum ExportFrameRate: String, CaseIterable, Identifiable {
    case match, fps24, fps30, fps60
    var id: String { rawValue }
    var label: String {
        switch self {
        case .match: return "Match"
        case .fps24: return "24"
        case .fps30: return "30"
        case .fps60: return "60"
        }
    }
    var value: Int? {
        switch self {
        case .match: return nil
        case .fps24: return 24
        case .fps30: return 30
        case .fps60: return 60
        }
    }
}

enum ExportQuality: String, CaseIterable, Identifiable {
    case auto, high, maximum
    var id: String { rawValue }
    var label: String {
        switch self {
        case .auto: return "Auto"
        case .high: return "High"
        case .maximum: return "Maximum"
        }
    }
    /// Bits per pixel per frame (HEVC).
    var bitsPerPixel: Double {
        switch self {
        case .auto: return 0.12
        case .high: return 0.2
        case .maximum: return 0.32
        }
    }
}

struct ExportSettings: Equatable {
    var resolution: ExportResolution = .p1080
    var frameRate: ExportFrameRate = .match
    var quality: ExportQuality = .auto
}

/// What the export reads: the original's quality (or the copy's, when the original is gone).
struct ExportSourceInfo: Equatable {
    var width = 1080
    var height = 1920
    var fps = 30
    var codec = "HEVC"
    var hdr = false
    /// Every source's original was found in Photos (else the light copy on the server is used).
    var originals = true
    /// Bytes still to download from iCloud before exporting.
    var cloudBytes: Int64 = 0
    var shortSide: Int { min(width, height) }

    static func standardShortSide(_ side: Int) -> Int {
        // 2160 / 1440 / 1080 / 720 or the source's own when smaller.
        for value in [2160, 1440, 1080, 720] where side >= value - 8 { return value }
        return max(2, side / 2 * 2)
    }
}

extension ExportSettings {
    func canvas(for info: ExportSourceInfo) -> CGSize {
        let maxShort = ExportSourceInfo.standardShortSide(info.shortSide)
        let short = min(resolution.shortSide ?? maxShort, maxShort)
        let s = CGFloat(short / 2 * 2)
        return CGSize(width: s, height: (s * 16 / 9 / 2).rounded() * 2)
    }

    func fps(for info: ExportSourceInfo) -> Int {
        min(frameRate.value ?? info.fps, max(24, info.fps), 60)
    }

    func videoBitrate(for info: ExportSourceInfo) -> Double {
        let size = canvas(for: info)
        let bits = Double(size.width * size.height) * Double(fps(for: info)) * quality.bitsPerPixel
        return max(4_000_000, min(120_000_000, bits))
    }

    func estimatedBytes(for info: ExportSourceInfo, seconds: Double) -> Int64 {
        Int64((videoBitrate(for: info) + 192_000) * max(0, seconds) / 8)
    }
}

// MARK: - The export

enum DeviceExportError: LocalizedError {
    case noClips
    case noSource
    case notEnoughSpace(neededBytes: Int64)
    case failed(String)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .noClips: return "This cut has no clips to export."
        case .noSource: return "The video for this cut isn't on this iPhone."
        case .notEnoughSpace(let bytes):
            return "Not enough space on your iPhone. Free up \(String(format: "%.1f", Double(bytes) / 1e9)) GB"
        case .failed(let why): return why
        case .cancelled: return "Export cancelled."
        }
    }
}

@MainActor
final class DeviceExport: ObservableObject {
    enum Phase: Equatable {
        case idle
        case original(Double)          // getting the original from iCloud
        case exporting(Double)
        case saving
        case done(URL)
        case failed(String)
        case noSpace(Int64)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var info = ExportSourceInfo()
    @Published private(set) var usedCopy = false
    private var task: Task<Void, Never>?
    private var reader: AVAssetReader?
    private var writer: AVAssetWriter?

    var running: Bool {
        switch phase {
        case .original, .exporting, .saving: return true
        default: return false
        }
    }

    // MARK: Source detection

    /// The quality of what the export will read (the first clip's source drives the choices).
    func detect(model: DraftEditorViewModel) async {
        var found = ExportSourceInfo()
        var cloud: Int64 = 0
        var allOriginals = true
        var first = true
        for sourceID in Self.neededSources(model) {
            guard let uri = Self.uri(of: sourceID, in: model) else { allOriginals = false; continue }
            if let asset = OriginalLinks.asset(forURI: uri) {
                if first {
                    found.width = asset.pixelWidth
                    found.height = asset.pixelHeight
                    if let resource = PHAssetResource.assetResources(for: asset).first(where: { $0.type == .video }),
                       (resource.originalFilename as NSString).pathExtension.lowercased() == "mp4" { found.codec = "H.264" }
                }
                if await GalleryExport.isOnlyInCloud(asset),
                   let bytes = PHAssetResource.assetResources(for: asset).first(where: { $0.type == .video })
                    .flatMap({ $0.value(forKey: "fileSize") as? NSNumber })?.int64Value {
                    cloud += bytes
                }
            } else {
                allOriginals = false
            }
            if first, let url = Self.localFile(forURI: uri), let track = try? await AVURLAsset(url: url).loadTracks(withMediaType: .video).first {
                if let (size, transform) = try? await track.load(.naturalSize, .preferredTransform), OriginalLinks.asset(forURI: uri) == nil {
                    let shown = size.applying(transform)
                    found.width = Int(abs(shown.width)); found.height = Int(abs(shown.height))
                }
                if let rate = try? await track.load(.nominalFrameRate), rate > 1 { found.fps = Int(rate.rounded()) }
                if let descriptions = try? await track.load(.formatDescriptions), let d = descriptions.first {
                    let code = CMFormatDescriptionGetMediaSubType(d)
                    if OriginalLinks.asset(forURI: uri) == nil { found.codec = code == kCMVideoCodecType_HEVC ? "HEVC" : "H.264" }
                }
                if let hdr = try? await track.load(.mediaCharacteristics) { found.hdr = hdr.contains(.containsHDRVideo) }
            }
            first = false
        }
        found.originals = allOriginals
        found.cloudBytes = cloud
        info = found
    }

    // MARK: Run

    func start(model: DraftEditorViewModel, settings: ExportSettings) {
        guard !running else { return }
        phase = .exporting(0)
        task = Task { [weak self] in
            guard let self else { return }
            do {
                let url = try await self.run(model: model, settings: settings)
                self.phase = .saving
                try await Self.saveToPhotos(url)
                self.phase = .done(url)
                CutSellDiagnostics.log("device_export_done", ["used_copy": self.usedCopy ? "1" : "0"])
            } catch DeviceExportError.notEnoughSpace(let bytes) {
                self.phase = .noSpace(bytes)
            } catch DeviceExportError.cancelled {
                self.phase = .idle
            } catch {
                CutSellDiagnostics.log("device_export_failed", ["error": String(describing: error).prefix(160).description])
                self.phase = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() {
        task?.cancel()
        reader?.cancelReading()
        writer?.cancelWriting()
        phase = .idle
    }

    func reset() { if !running { phase = .idle } }

    private func run(model: DraftEditorViewModel, settings: ExportSettings) async throws -> URL {
        let started = Date()
        guard !model.selectedClips.isEmpty else { throw DeviceExportError.noClips }
        await detect(model: model)
        let source = info
        let size = settings.canvas(for: source)
        let fps = settings.fps(for: source)
        let videoBits = settings.videoBitrate(for: source)

        // Space: the file (+20 %), the originals still in iCloud, and a margin.
        let seconds = model.selectedClips.reduce(0.0) { sum, clip in
            let start = clip["start"]?.doubleValue ?? 0, end = clip["end"]?.doubleValue ?? start
            return sum + max(0, end - start) / ClipSpeed.of(clip)
        }
        let needed = Int64(Double(settings.estimatedBytes(for: source, seconds: seconds)) * 1.2) + source.cloudBytes + 200_000_000
        if let free = Self.freeBytes(), free < needed { throw DeviceExportError.notEnoughSpace(neededBytes: needed - free) }

        // 1) The originals (from iCloud when needed), else the light copy.
        var assets: [String: AVAsset] = [:]
        var copy = false
        let sources = Self.neededSources(model)
        for (index, sourceID) in sources.enumerated() {
            try Task.checkCancellation()
            guard let uri = Self.uri(of: sourceID, in: model) else { continue }
            if let original = OriginalLinks.asset(forURI: uri),
               let asset = await Self.originalAsset(original, progress: { [weak self] value in
                   Task { @MainActor in self?.phase = .original((Double(index) + value) / Double(max(1, sources.count))) }
               }) {
                assets[sourceID] = asset
                continue
            }
            copy = true
            if let local = Self.localFile(forURI: uri) {
                assets[sourceID] = AVURLAsset(url: local)
            } else if let remote = Self.playbackURL(of: sourceID, in: model) {
                let (temp, _) = try await URLSession.shared.download(from: remote)
                let kept = FileManager.default.temporaryDirectory.appendingPathComponent("cutsell-src-\(UUID().uuidString).mp4")
                try? FileManager.default.moveItem(at: temp, to: kept)
                assets[sourceID] = AVURLAsset(url: kept)
            }
        }
        usedCopy = copy
        guard !assets.isEmpty else { throw DeviceExportError.noSource }
        phase = .exporting(0)

        // 2) The same timeline, mix and picture as the preview.
        let built = try await ExportTimeline.build(model: model, assets: assets)
        let plan = ExportFramePlan.make(model: model, timeline: built, fps: fps)
        let renderer = ExportFrameRenderer(size: size, timeline: built, plan: plan, model: model)

        let videoComposition = AVMutableVideoComposition()
        videoComposition.customVideoCompositorClass = ExportCompositor.self
        videoComposition.renderSize = size
        videoComposition.frameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))
        let instruction = ExportInstruction(timeRange: CMTimeRange(start: .zero, duration: built.composition.duration),
                                            trackID: built.videoTrackID, renderer: renderer)
        videoComposition.instructions = [instruction]

        let output = FileManager.default.temporaryDirectory.appendingPathComponent("CutSell-\(UUID().uuidString).mp4")
        try await Self.encode(composition: built.composition, videoComposition: videoComposition, audioMix: built.audioMix,
                              to: output, size: size, fps: fps, videoBits: videoBits, seconds: built.duration,
                              owner: self) { [weak self] value in
            Task { @MainActor in if case .exporting = self?.phase { self?.phase = .exporting(value) } }
        }
        CutSellDiagnostics.log("device_export_encoded", [
            "w": String(Int(size.width)), "h": String(Int(size.height)), "fps": String(fps),
            "mbps": String(format: "%.1f", videoBits / 1e6), "seconds": String(format: "%.1f", built.duration),
            "took_s": String(format: "%.1f", Date().timeIntervalSince(started)), "copy": copy ? "1" : "0",
        ])
        return output
    }

    // MARK: Encode

    private static func encode(composition: AVComposition, videoComposition: AVVideoComposition, audioMix: AVAudioMix,
                               to url: URL, size: CGSize, fps: Int, videoBits: Double, seconds: Double, owner: DeviceExport,
                               progress: @escaping @Sendable (Double) -> Void) async throws {
        let reader = try AVAssetReader(asset: composition)
        let videoTracks = composition.tracks(withMediaType: .video)
        let videoOut = AVAssetReaderVideoCompositionOutput(videoTracks: videoTracks, videoSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        videoOut.videoComposition = videoComposition
        videoOut.alwaysCopiesSampleData = false
        guard reader.canAdd(videoOut) else { throw DeviceExportError.failed("Couldn't read the video.") }
        reader.add(videoOut)
        var audioOut: AVAssetReaderAudioMixOutput?
        let audioTracks = composition.tracks(withMediaType: .audio)
        if !audioTracks.isEmpty {
            let out = AVAssetReaderAudioMixOutput(audioTracks: audioTracks, audioSettings: [
                AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 44_100, AVNumberOfChannelsKey: 2,
                AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsNonInterleaved: false,
            ])
            out.audioMix = audioMix
            out.audioTimePitchAlgorithm = .spectral
            if reader.canAdd(out) { reader.add(out); audioOut = out }
        }

        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        writer.shouldOptimizeForNetworkUse = true
        let videoIn = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: Int(size.width),
            AVVideoHeightKey: Int(size.height),
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: Int(videoBits),
                AVVideoExpectedSourceFrameRateKey: fps,
                AVVideoMaxKeyFrameIntervalDurationKey: 2,
            ] as [String: Any],
        ])
        videoIn.expectsMediaDataInRealTime = false
        guard writer.canAdd(videoIn) else { throw DeviceExportError.failed("Couldn't write the video.") }
        writer.add(videoIn)
        var audioIn: AVAssetWriterInput?
        if audioOut != nil {
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC, AVNumberOfChannelsKey: 2, AVSampleRateKey: 44_100, AVEncoderBitRateKey: 192_000,
            ])
            input.expectsMediaDataInRealTime = false
            if writer.canAdd(input) { writer.add(input); audioIn = input }
        }
        await MainActor.run { owner.reader = reader; owner.writer = writer }

        guard reader.startReading() else { throw reader.error ?? DeviceExportError.failed("Couldn't start reading.") }
        guard writer.startWriting() else { throw writer.error ?? DeviceExportError.failed("Couldn't start writing.") }
        writer.startSession(atSourceTime: .zero)
        let total = max(0.1, seconds)
        let videoPump = LightCopyWriter.Pump(input: videoIn, output: videoOut, label: "cutsell.export.video") { time in
            progress(min(0.99, CMTimeGetSeconds(time) / total))
        }
        let audioPump = audioIn.flatMap { input in audioOut.map { LightCopyWriter.Pump(input: input, output: $0, label: "cutsell.export.audio") { _ in } } }
        async let v: Void = videoPump.run()
        async let a: Void = audioPump?.run() ?? ()
        _ = await (v, a)
        if Task.isCancelled || reader.status == .cancelled {
            writer.cancelWriting()
            throw DeviceExportError.cancelled
        }
        if reader.status == .failed {
            writer.cancelWriting()
            throw reader.error ?? DeviceExportError.failed("The export stopped.")
        }
        await writer.finishWriting()
        if writer.status != .completed { throw writer.error ?? DeviceExportError.failed("The export didn't finish.") }
    }

    // MARK: Helpers

    static func neededSources(_ model: DraftEditorViewModel) -> [String] {
        var seen = Set<String>(), out: [String] = []
        for clip in model.selectedClips {
            if let id = clip["source_asset_id"]?.stringValue, seen.insert(id).inserted { out.append(id) }
        }
        return out
    }

    static func uri(of sourceID: String, in model: DraftEditorViewModel) -> String? {
        model.allSources.first { $0["source_asset_id"]?.stringValue == sourceID }?["uri"]?.stringValue
    }

    static func playbackURL(of sourceID: String, in model: DraftEditorViewModel) -> URL? {
        model.allSources.first { $0["source_asset_id"]?.stringValue == sourceID }?["playback_url"]?.stringValue.flatMap(URL.init(string:))
    }

    /// The copy kept on this iPhone (what was uploaded), never the 720p preview proxy.
    static func localFile(forURI uri: String) -> URL? { SourceVideoCache.localURL(forURI: uri) }

    static func freeBytes() -> Int64? {
        let home = URL(fileURLWithPath: NSHomeDirectory())
        let values = try? home.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    /// The original video (downloaded from iCloud when needed). nil when Photos can't give it.
    static func originalAsset(_ asset: PHAsset, progress: @escaping @Sendable (Double) -> Void) async -> AVAsset? {
        await withCheckedContinuation { continuation in
            let options = PHVideoRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .highQualityFormat
            options.version = .original
            options.progressHandler = { value, _, _, _ in progress(value) }
            PHImageManager.default().requestAVAsset(forVideo: asset, options: options) { avAsset, _, _ in
                continuation.resume(returning: avAsset)
            }
        }
    }

    static func saveToPhotos(_ url: URL) async throws {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            throw DeviceExportError.failed("Allow CutSell to add the finished video to Photos.")
        }
        try await PHPhotoLibrary.shared().performChanges {
            PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
        }
    }
}

// MARK: - Timeline (the preview's composition, from the originals)

struct ExportSegment {
    let start: Double
    let duration: Double
    let sourceID: String
    let transform: CGAffineTransform
}

struct ExportTimeline {
    let composition: AVMutableComposition
    let audioMix: AVMutableAudioMix
    let videoTrackID: CMPersistentTrackID
    let segments: [ExportSegment]
    let windows: [CaptionClipWindow]
    let duration: Double
    /// Assets kept alive with their tracks.
    let retained: [AVAsset]

    /// Same as `DraftPlaybackController.rebuild` (cuts, speed, Clean up, volumes, ducking, voices, sounds,
    /// overlay sound), with each source's own orientation.
    @MainActor
    static func build(model: DraftEditorViewModel, assets: [String: AVAsset]) async throws -> ExportTimeline {
        let composition = AVMutableComposition()
        guard let videoTrack = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw DeviceExportError.failed("Couldn't build the video.")
        }
        let audioTrack = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
        var retained: [AVAsset] = Array(assets.values)
        var segments: [ExportSegment] = []
        var windows: [CaptionClipWindow] = []
        var baseLevels: [(start: Double, end: Double, level: Double)] = []
        var mixInputs: [AVMutableAudioMixInputParameters] = []
        let master = model.masterMuted ? 0 : model.masterVolume
        let videoGain = max(0, min(2, model.videoVolume)) * master
        var cursor = CMTime.zero
        var loaded: [String: (video: AVAssetTrack, audio: AVAssetTrack?, transform: CGAffineTransform)] = [:]
        var cleanTracks: [String: AVAssetTrack] = [:]

        for (clipIndex, clip) in model.selectedClips.enumerated() {
            try Task.checkCancellation()
            guard let sourceID = clip["source_asset_id"]?.stringValue, let asset = assets[sourceID] else { continue }
            if loaded[sourceID] == nil {
                guard let video = try await asset.loadTracks(withMediaType: .video).first else { continue }
                let audio = try? await asset.loadTracks(withMediaType: .audio).first
                let transform = (try? await video.load(.preferredTransform)) ?? .identity
                loaded[sourceID] = (video, audio, transform)
            }
            guard let source = loaded[sourceID] else { continue }
            let startSeconds = max(0, clip["start"]?.doubleValue ?? 0)
            let endSeconds = max(startSeconds, clip["end"]?.doubleValue ?? startSeconds)
            let clipDuration = endSeconds - startSeconds
            guard clipDuration >= 0.05 else { continue }
            let range = CMTimeRange(start: CMTime(seconds: startSeconds, preferredTimescale: 600),
                                    duration: CMTime(seconds: clipDuration, preferredTimescale: 600))
            let speed = ClipSpeed.of(clip)
            let played = CMTime(seconds: clipDuration / speed, preferredTimescale: 600)
            try videoTrack.insertTimeRange(range, of: source.video, at: cursor)
            segments.append(ExportSegment(start: CMTimeGetSeconds(cursor), duration: clipDuration / speed, sourceID: sourceID,
                                          transform: source.transform))
            windows.append(CaptionClipWindow(clipIndex: clipIndex, clipID: clip["clip_id"]?.stringValue ?? "",
                                             timelineStart: CMTimeGetSeconds(cursor), duration: clipDuration / speed, speed: speed))
            if let audioTrack {
                var sound = source.audio
                if model.clipPlaysCleaned(clip), let cleanURL = CleanAudioCenter.shared.localURL(.source(sourceID), project: model.project) {
                    if cleanTracks[sourceID] == nil {
                        let clean = AVURLAsset(url: cleanURL)
                        retained.append(clean)
                        cleanTracks[sourceID] = try? await clean.loadTracks(withMediaType: .audio).first
                    }
                    sound = cleanTracks[sourceID] ?? sound
                }
                if let sound { try? audioTrack.insertTimeRange(range, of: sound, at: cursor) }
            }
            if speed != 1 {
                composition.scaleTimeRange(CMTimeRange(start: cursor, duration: range.duration), toDuration: played)
            }
            let muted = clip["audio_muted"]?.boolValue ?? false
            let level = max(0, min(2, clip["audio_volume"]?.doubleValue ?? 1)) * videoGain
            baseLevels.append((CMTimeGetSeconds(cursor), CMTimeGetSeconds(cursor) + clipDuration / speed, muted ? 0 : level))
            cursor = CMTimeAdd(cursor, played)
        }
        let cutSeconds = CMTimeGetSeconds(cursor)
        guard cutSeconds > 0 else { throw DeviceExportError.noSource }

        if let audioTrack {
            let parameters = AVMutableAudioMixInputParameters(track: audioTrack)
            let ducks = model.previewVoices.filter(\.ducksVideo).map { ($0.start, $0.start + ($0.sourceOut - $0.sourceIn)) }
            let peak = baseLevels.map(\.level).max() ?? 1
            let tapGain = max(1, peak)
            if tapGain > 1.001, let tap = AudioGainTap.make(gain: tapGain) { parameters.audioTapProcessor = tap }
            let clips = baseLevels.map { (start: $0.start, end: $0.end, volume: Float($0.level / tapGain)) }
            DraftPlaybackController.applyBaseVolumes(parameters, clips: clips, ducks: ducks)
            mixInputs.append(parameters)
        }

        func addAudio(_ url: URL, from sourceIn: Double, to sourceOut: Double, at start: Double, volume: Double) async {
            let asset = AVURLAsset(url: url)
            retained.append(asset)
            guard let source = try? await asset.loadTracks(withMediaType: .audio).first else { return }
            let range = (try? await source.load(.timeRange)) ?? CMTimeRange(start: .zero, duration: .zero)
            let from = max(sourceIn, max(0, CMTimeGetSeconds(range.start)))
            let to = min(sourceOut, CMTimeGetSeconds(range.end))
            let at = start + (from - sourceIn)
            let length = min(to - from, cutSeconds - at)
            guard length > 0.02,
                  let track = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) else { return }
            do {
                try track.insertTimeRange(CMTimeRange(start: CMTime(seconds: from, preferredTimescale: 44_100),
                                                      duration: CMTime(seconds: length, preferredTimescale: 44_100)),
                                          of: source, at: CMTime(seconds: at, preferredTimescale: 44_100))
            } catch {
                composition.removeTrack(track)
                return
            }
            let parameters = AVMutableAudioMixInputParameters(track: track)
            AudioGainTap.apply(volume, to: parameters)
            mixInputs.append(parameters)
        }
        for voice in model.previewVoices {
            await addAudio(voice.url, from: voice.sourceIn, to: voice.sourceOut, at: voice.start,
                           volume: max(0, min(2, voice.volume)) * master)
        }
        for overlay in model.overlaysOnScreen where overlay.kind == .video && !overlay.style.muted {
            guard let url = overlay.url, url.isFileURL else { continue }
            await addAudio(url, from: overlay.sourceStart, to: overlay.sourceStart + (overlay.end - overlay.start),
                           at: overlay.start, volume: max(0, min(2, overlay.style.volume)) * master)
        }
        if composition.duration > cursor {
            composition.removeTimeRange(CMTimeRange(start: cursor, end: composition.duration))
        }
        let mix = AVMutableAudioMix()
        mix.inputParameters = mixInputs
        return ExportTimeline(composition: composition, audioMix: mix, videoTrackID: videoTrack.trackID, segments: segments,
                              windows: windows, duration: cutSeconds, retained: retained)
    }

    func segment(at time: Double) -> ExportSegment? {
        segments.last { time >= $0.start - 0.0001 } ?? segments.first
    }
}

// MARK: - Per-frame plan (computed once, on the main actor)

struct ExportFrameParams {
    var mirrored = false
    var degrees = 0
    var key = TransformValue.identity
    /// The video's region while a Stacked / Side by side overlay shows (fractions, y down).
    var region: CGRect?
    /// What the layer above the video shows (empty = nothing).
    var layerKey = ""
}

struct ExportFramePlan {
    let fps: Int
    let frames: [ExportFrameParams]
    let cues: [CaptionPreviewCue]

    @MainActor
    static func make(model: DraftEditorViewModel, timeline: ExportTimeline, fps: Int) -> ExportFramePlan {
        let cues = model.captionsEnabled
            ? CaptionPreviewRules.cues(clips: model.selectedClips, windows: timeline.windows, engine: model.draftEngine) : []
        let count = max(1, Int((timeline.duration * Double(fps)).rounded(.up)))
        let overlays = model.overlaysOnScreen.filter { $0.uploading == nil && !$0.failed }
        let texts = model.textOverlays
        var frames: [ExportFrameParams] = []
        frames.reserveCapacity(count)
        for i in 0..<count {
            let t = Double(i) / Double(fps)
            var p = ExportFrameParams()
            p.mirrored = model.isMirrored(at: t)
            p.degrees = model.rotation(at: t)
            p.key = model.videoTransform(at: t)
            var parts: [String] = []
            if let cue = cues.first(where: { t >= $0.start && t < $0.end }) {
                parts.append("c\(cue.id):\(cue.activeWordIndex(at: t) ?? -1)")
            }
            for (index, item) in texts.enumerated() {
                let start = item["start"]?.doubleValue ?? 0, end = item["end"]?.doubleValue ?? start
                if t >= start && t < end { parts.append("t\(index)") }
            }
            var split: OverlayOnScreen?
            for overlay in overlays where t >= overlay.start && t < overlay.end {
                let k = model.overlayTransform(id: overlay.id, at: t)
                let step = overlay.kind == .video ? Int(((overlay.sourceStart + t - overlay.start) * 5).rounded()) : 0
                parts.append(String(format: "o%@:%.3f:%.4f:%.4f:%.1f:%d", overlay.id, k.scale, k.x, k.y, k.rotation, step))
                if overlay.style.layout.baseRegion != nil, split == nil || overlay.style.layer > split!.style.layer { split = overlay }
            }
            p.region = split?.style.layout.baseRegion
            p.layerKey = parts.joined(separator: "|")
            frames.append(p)
        }
        return ExportFramePlan(fps: fps, frames: frames, cues: cues)
    }

    func params(at time: Double) -> ExportFrameParams {
        let i = min(frames.count - 1, max(0, Int((time * Double(fps)).rounded())))
        return frames[i]
    }
}

// MARK: - Drawing one frame

/// Draws the video like the editor's `videoLayer` (fit / fill, mirror, rotation, keyframes) and the
/// layer above it (text, photo / video overlays, captions) drawn by SwiftUI exactly like the preview.
final class ExportFrameRenderer: @unchecked Sendable {
    let size: CGSize
    let timeline: ExportTimeline
    let plan: ExportFramePlan
    private weak var model: DraftEditorViewModel?
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private var layerKey = ""
    private var layerImage: CIImage?
    private var overlayFrames: [String: UIImage] = [:]
    private var generators: [String: AVAssetImageGenerator] = [:]
    private var photos: [String: UIImage] = [:]

    init(size: CGSize, timeline: ExportTimeline, plan: ExportFramePlan, model: DraftEditorViewModel) {
        self.size = size; self.timeline = timeline; self.plan = plan; self.model = model
    }

    func render(time: Double, source: CVPixelBuffer?, into output: CVPixelBuffer) {
        let W = size.width, H = size.height
        let params = plan.params(at: time)
        var image = CIImage(color: CIColor(red: 0, green: 0, blue: 0)).cropped(to: CGRect(x: 0, y: 0, width: W, height: H))
        if let source, let segment = timeline.segment(at: time) {
            image = videoLayer(CIImage(cvPixelBuffer: source), transform: segment.transform, params: params).composited(over: image)
        }
        if !params.layerKey.isEmpty, let layer = layer(for: params, time: time) {
            image = layer.composited(over: image)
        }
        context.render(image, to: output, bounds: CGRect(x: 0, y: 0, width: W, height: H), colorSpace: colorSpace)
    }

    // The video, exactly the editor's modifier chain (SwiftUI is y-down, Core Image y-up).
    private func videoLayer(_ frame: CIImage, transform: CGAffineTransform, params: ExportFrameParams) -> CIImage {
        let W = size.width, H = size.height
        var picture = frame.oriented(Self.orientation(of: transform))
        picture = picture.transformed(by: CGAffineTransform(translationX: -picture.extent.minX, y: -picture.extent.minY))
        let w = picture.extent.width, h = picture.extent.height
        guard w > 0, h > 0 else { return picture }
        // The box the video lives in (whole frame, or its half while Stacked / Side by side shows).
        let region = params.region ?? CGRect(x: 0, y: 0, width: 1, height: 1)
        let box = CGRect(x: region.minX * W, y: (1 - region.maxY) * H, width: region.width * W, height: region.height * H)
        let fill = params.region != nil
        let fit = fill ? max(box.width / w, box.height / h) : min(box.width / w, box.height / h)
        let quarter = params.degrees % 180 != 0
        var t = CGAffineTransform(translationX: -w / 2, y: -h / 2)
        t = t.concatenating(CGAffineTransform(scaleX: fit, y: fit))
        if params.mirrored { t = t.concatenating(CGAffineTransform(scaleX: -1, y: 1)) }
        t = t.concatenating(CGAffineTransform(rotationAngle: -CGFloat(params.degrees) * .pi / 180))
        if quarter { t = t.concatenating(CGAffineTransform(scaleX: 266.0 / 472.0, y: 266.0 / 472.0)) }
        t = t.concatenating(CGAffineTransform(scaleX: params.key.scale, y: params.key.scale))
        t = t.concatenating(CGAffineTransform(rotationAngle: -CGFloat(params.key.rotation) * .pi / 180))
        t = t.concatenating(CGAffineTransform(translationX: box.midX + box.width * params.key.x, y: box.midY - box.height * params.key.y))
        return picture.transformed(by: t).cropped(to: box)
    }

    static func orientation(of t: CGAffineTransform) -> CGImagePropertyOrientation {
        switch (t.a.rounded(), t.b.rounded(), t.c.rounded(), t.d.rounded()) {
        case (0, 1, -1, 0): return .right
        case (0, -1, 1, 0): return .left
        case (-1, 0, 0, -1): return .down
        case (-1, 0, 0, 1): return .upMirrored
        case (1, 0, 0, -1): return .downMirrored
        case (0, 1, 1, 0): return .rightMirrored
        case (0, -1, -1, 0): return .leftMirrored
        default: return .up
        }
    }

    // The layer above the video: drawn by SwiftUI (the preview's own views) when what it shows changes.
    private func layer(for params: ExportFrameParams, time: Double) -> CIImage? {
        if params.layerKey == layerKey { return layerImage }
        let frames = videoOverlayFrames(at: time)
        let photos = photoOverlays()
        let size = self.size
        let image: CGImage? = DispatchQueue.main.sync {
            MainActor.assumeIsolated {
                guard let model else { return nil }
                return ExportLayerSnapshot.render(model: model, time: time, cues: plan.cues, size: size, pictures: frames.merging(photos) { a, _ in a })
            }
        }
        layerKey = params.layerKey
        layerImage = image.map { CIImage(cgImage: $0) }
        return layerImage
    }

    private func photoOverlays() -> [String: UIImage] {
        guard photos.isEmpty else { return photos }
        let overlays: [OverlayOnScreen] = DispatchQueue.main.sync { MainActor.assumeIsolated { model?.overlaysOnScreen ?? [] } }
        for overlay in overlays where overlay.kind == .photo {
            if let url = overlay.url, let image = UIImage(contentsOfFile: url.path) { photos[overlay.id] = image }
        }
        return photos
    }

    /// Video overlays as the preview shows them: a still every 0.2 s of their own time.
    private func videoOverlayFrames(at time: Double) -> [String: UIImage] {
        let overlays: [OverlayOnScreen] = DispatchQueue.main.sync { MainActor.assumeIsolated { model?.overlaysOnScreen ?? [] } }
        var out: [String: UIImage] = [:]
        for overlay in overlays where overlay.kind == .video && time >= overlay.start && time < overlay.end {
            guard let url = overlay.url else { continue }
            let seconds = (overlay.sourceStart + time - overlay.start)
            let step = (seconds * 5).rounded() / 5
            let key = "\(overlay.id)|\(step)"
            if let hit = overlayFrames[key] { out[overlay.id] = hit; continue }
            let generator = generators[overlay.id] ?? {
                let g = AVAssetImageGenerator(asset: AVURLAsset(url: url))
                g.appliesPreferredTrackTransform = true
                g.requestedTimeToleranceBefore = CMTime(value: 1, timescale: 10)
                g.requestedTimeToleranceAfter = CMTime(value: 1, timescale: 10)
                g.maximumSize = CGSize(width: 1080, height: 1920)
                generators[overlay.id] = g
                return g
            }()
            if let cg = try? generator.copyCGImage(at: CMTime(seconds: step, preferredTimescale: 600), actualTime: nil) {
                let image = UIImage(cgImage: cg)
                if overlayFrames.count > 40 { overlayFrames.removeAll() }
                overlayFrames[key] = image
                out[overlay.id] = image
            }
        }
        return out
    }
}

/// The layer above the video, as one transparent picture at the export's size.
@MainActor
enum ExportLayerSnapshot {
    /// The preview card's size (text overlays are sized in its points); scaled up to the export.
    static let card = CGSize(width: 266, height: 266 * 16 / 9)

    static func render(model: DraftEditorViewModel, time: Double, cues: [CaptionPreviewCue], size: CGSize,
                       pictures: [String: UIImage]) -> CGImage? {
        let card = CGSize(width: Self.card.width, height: Self.card.width * size.height / size.width)
        let view = ExportOverlayLayer(model: model, time: time, cues: cues, frame: card, pictures: pictures)
            .frame(width: card.width, height: card.height)
            .environment(\.colorScheme, .dark)
        let renderer = ImageRenderer(content: view)
        renderer.scale = size.width / card.width
        renderer.isOpaque = false
        return renderer.cgImage
    }
}

/// The editor's text overlays, photo / video overlays and captions for one moment, without any controls.
struct ExportOverlayLayer: View {
    @ObservedObject var model: DraftEditorViewModel
    let time: Double
    let cues: [CaptionPreviewCue]
    let frame: CGSize
    let pictures: [String: UIImage]

    var body: some View {
        ZStack {
            ForEach(Array(model.textOverlays.enumerated()), id: \.offset) { _, item in
                let start = item["start"]?.doubleValue ?? 0
                let end = item["end"]?.doubleValue ?? start
                if time >= start && time < end {
                    Text(L10n.tr(item["text"]?.stringValue ?? ""))
                        .font(CS.inter(CGFloat(18 * (item["scale"]?.doubleValue ?? 1)), .bold))
                        .foregroundStyle(.white)
                        .shadow(color: .black.opacity(0.6), radius: 2)
                        .position(x: frame.width * CGFloat(item["x"]?.doubleValue ?? 0.5),
                                  y: frame.height * CGFloat(item["y"]?.doubleValue ?? 0.2))
                }
            }
            ForEach(model.overlaysOnScreen.sorted { $0.style.layer < $1.style.layer }) { overlay in
                if time >= overlay.start && time < overlay.end, let picture = pictures[overlay.id] {
                    ExportMediaOverlay(overlay: overlay, frame: frame, picture: picture,
                                       key: model.overlayTransform(id: overlay.id, at: time))
                }
            }
            if model.captionsEnabled {
                CaptionOverlayView(cue: cues.first { time >= $0.start && time < $0.end }, time: time, preset: model.captionPreset,
                                   fontKey: model.captionFont, x: model.captionX, y: model.captionY, scale: model.captionScale)
            }
        }
        .frame(width: frame.width, height: frame.height)
    }
}

/// `MediaOverlayOnVideo` without its controls, with its picture already loaded.
struct ExportMediaOverlay: View {
    let overlay: OverlayOnScreen
    let frame: CGSize
    let picture: UIImage
    let key: TransformValue

    private var style: OverlayStyle { overlay.style }
    private var aspect: CGFloat { picture.size.height > 0 ? picture.size.width / picture.size.height : 1 }
    private var feather: CGFloat { style.edge.featherOf1080 * frame.width / 1080 }

    var body: some View {
        if let region = style.layout.region {
            let size = CGSize(width: frame.width * region.width, height: frame.height * region.height)
            ZStack {
                if !style.fill && !style.layout.transparentPad { Color.black }
                if style.fill {
                    Image(uiImage: picture).resizable().scaledToFill().frame(width: size.width, height: size.height).clipped()
                } else {
                    Image(uiImage: picture).resizable().aspectRatio(aspect, contentMode: .fit).frame(width: size.width, height: size.height)
                }
            }
            .frame(width: size.width, height: size.height)
            .mask { featherMask(rounded: false) }
            .position(x: frame.width * region.midX, y: frame.height * region.midY)
        } else {
            let w = frame.width * CGFloat(overlay.width) * CGFloat(key.scale)
            let h = w / max(0.1, aspect)
            Image(uiImage: picture).resizable().scaledToFill()
                .frame(width: w, height: h)
                .clipShape(RoundedRectangle(cornerRadius: style.edge == .sharp ? 4 : 0))
                .mask { featherMask(rounded: true) }
                .rotationEffect(.degrees(style.rotation + key.rotation))
                .position(x: frame.width * CGFloat(overlay.x + key.x), y: frame.height * CGFloat(overlay.y + key.y))
        }
    }

    @ViewBuilder
    private func featherMask(rounded: Bool) -> some View {
        if feather > 0.5 {
            if rounded {
                RoundedRectangle(cornerRadius: feather).padding(feather / 2).blur(radius: feather / 2)
            } else {
                Rectangle().padding(feather / 2).blur(radius: feather / 2)
            }
        } else {
            Rectangle()
        }
    }
}

// MARK: - Custom compositor

final class ExportInstruction: NSObject, AVVideoCompositionInstructionProtocol {
    let timeRange: CMTimeRange
    let enablePostProcessing = false
    let containsTweening = true
    let requiredSourceTrackIDs: [NSValue]?
    let passthroughTrackID = kCMPersistentTrackID_Invalid
    let trackID: CMPersistentTrackID
    let renderer: ExportFrameRenderer

    init(timeRange: CMTimeRange, trackID: CMPersistentTrackID, renderer: ExportFrameRenderer) {
        self.timeRange = timeRange
        self.trackID = trackID
        self.renderer = renderer
        self.requiredSourceTrackIDs = [NSNumber(value: trackID)]
    }
}

final class ExportCompositor: NSObject, AVVideoCompositing {
    let sourcePixelBufferAttributes: [String: Any]? = [
        kCVPixelBufferPixelFormatTypeKey as String: [kCVPixelFormatType_32BGRA, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                                     kCVPixelFormatType_420YpCbCr8BiPlanarFullRange],
        kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
    ]
    let requiredPixelBufferAttributesForRenderContext: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
    ]
    private let queue = DispatchQueue(label: "cutsell.export.compositor")

    func renderContextChanged(_ newRenderContext: AVVideoCompositionRenderContext) {}

    func startRequest(_ request: AVAsynchronousVideoCompositionRequest) {
        queue.async {
            guard let instruction = request.videoCompositionInstruction as? ExportInstruction,
                  let output = request.renderContext.newPixelBuffer() else {
                request.finish(with: NSError(domain: "CutSellExport", code: 1))
                return
            }
            let source = request.sourceFrame(byTrackID: instruction.trackID)
            instruction.renderer.render(time: CMTimeGetSeconds(request.compositionTime), source: source, into: output)
            request.finish(withComposedVideoFrame: output)
        }
    }

    func cancelAllPendingVideoCompositionRequests() {}
}
