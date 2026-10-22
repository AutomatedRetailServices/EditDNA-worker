import SwiftUI
import AVKit
import AVFoundation

struct DraftPlaybackView: View {
    @ObservedObject var model: DraftEditorViewModel
    /// While the Captions panel is open the caption on the video can be
    /// tapped (the clip it belongs to is handed back to fix its words),
    /// dragged and pinched (position and size for the whole video).
    var onCaptionTap: ((String) -> Void)? = nil
    @StateObject private var playback = DraftPlaybackController()

    var body: some View {
        VStack(spacing: 10) {
            VideoPlayer(player: playback.player)
                .aspectRatio(9.0 / 16.0, contentMode: .fit)
                .frame(maxHeight: 430)
                .overlay {
                    if model.captionsEnabled {
                        // Captions as the export will burn them in. The export
                        // fits every video into a 9:16 frame and places captions
                        // on that frame, which is exactly this 9:16 box.
                        let cue = playback.captionCue(at: playback.currentTime)
                        CaptionOverlayView(
                            cue: cue,
                            time: playback.currentTime,
                            preset: model.captionPreset,
                            fontKey: model.captionFont,
                            x: model.captionX,
                            y: model.captionY,
                            scale: model.captionScale,
                            editing: editing(for: cue)
                        )
                    }
                }
                .background(.black, in: RoundedRectangle(cornerRadius: 16))
                .clipShape(RoundedRectangle(cornerRadius: 16))

            HStack(spacing: 12) {
                Button {
                    playback.togglePlayback()
                } label: {
                    Image(systemName: playback.isPlaying ? "pause.fill" : "play.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!playback.isReady)

                Slider(
                    value: Binding(
                        get: { playback.currentTime },
                        set: { playback.seek(to: $0) }
                    ),
                    in: 0...max(0.01, playback.duration)
                )
                .disabled(!playback.isReady)

                Text(L10n.tr("\(time(playback.currentTime)) / \(time(playback.duration))"))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 84, alignment: .trailing)
            }

            if playback.isBuilding {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(L10n.tr("Preparing preview…"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else if let message = playback.message {
                Text(L10n.tr(message))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal)
        .onChange(of: playback.currentTime) { _, time in
            if abs(model.playheadTime - time) >= 0.1 || time == 0 { model.playheadTime = time }
        }
        .task(id: model.snapshot?.revision) {
            await playback.rebuild(from: model)
        }
        .onDisappear { playback.pause() }
    }

    @MainActor
    private func editing(for cue: CaptionPreviewCue?) -> CaptionEditing? {
        guard let onCaptionTap, let cue else { return nil }
        let controller = playback
        let draft = model
        // Moving and resizing only exist for the short timed captions; the
        // server draws a whole-clip caption in a fixed place.
        let movable = cue.isTimed
        return CaptionEditing(
            onTap: {
                controller.pause()
                onCaptionTap(cue.clipID)
            },
            onBegin: { controller.pause() },
            onLayoutChange: { x, y, scale in
                guard movable else { return }
                draft.applyCaptionSettings(x: x, y: y, scale: scale)
            }
        )
    }

    private func time(_ seconds: Double) -> String {
        let value = max(0, Int(seconds.rounded(.down)))
        return String(format: "%d:%02d", value / 60, value % 60)
    }
}

@MainActor
final class DraftPlaybackController: ObservableObject {
    @Published private(set) var player = AVPlayer()
    @Published private(set) var isPlaying = false
    @Published private(set) var isBuilding = false
    @Published private(set) var isReady = false
    @Published private(set) var currentTime = 0.0
    @Published private(set) var duration = 0.0
    @Published private(set) var message: String?
    /// Captions of the whole preview, in timeline order (`CaptionPreviewRules`).
    @Published private(set) var captionCues: [CaptionPreviewCue] = []

    private var timeObserver: Any?
    /// Where each clip sits on the preview timeline (same order and same
    /// skipped clips as the composition built below).
    private var clipWindows: [CaptionClipWindow] = []
    private var compositionSignature: String?
    private var compositionBuiltAt: Date?
    /// How long the video links of the current composition can be trusted.
    private var compositionReuseSec = 0.0

    private var endObserver: NSObjectProtocol?
    /// Playing when a rebuild started (a rebuild replaced by a newer one hands this on).
    private var resumeAfterBuild = false

    init() {
        CaptionFontLoader.registerIfNeeded()
        installTimeObserver()
        endObserver = NotificationCenter.default.addObserver(
            forName: AVPlayerItem.didPlayToEndTimeNotification, object: nil, queue: .main
        ) { [weak self] note in
            Task { @MainActor in
                guard let self, (note.object as? AVPlayerItem) === self.player.currentItem else { return }
                self.stopAtEnd()
            }
        }
    }

    func captionCue(at time: Double) -> CaptionPreviewCue? {
        captionCues.first { time >= $0.start && time < $0.end }
    }

    /// Clip under the playhead, if any.
    func clipID(at time: Double) -> String? {
        clipWindows.first { time >= $0.timelineStart && time < $0.timelineStart + $0.duration }?.clipID
    }

    private func refreshCaptionCues(from model: DraftEditorViewModel) {
        captionCues = CaptionPreviewRules.cues(
            clips: model.selectedClips,
            windows: clipWindows,
            engine: model.draftEngine
        )
    }

    /// Everything the picture and sound of the preview depend on. A change
    /// that leaves this untouched (caption style, typeface, text, position)
    /// keeps the player where it is instead of rebuilding it from zero.
    private static func signature(for model: DraftEditorViewModel, sourceURLs: [String: URL]) -> String {
        var parts: [String] = []
        for clip in model.selectedClips {
            let clipID: String = clip["clip_id"]?.stringValue ?? ""
            let sourceID: String = clip["source_asset_id"]?.stringValue ?? ""
            let start: Double = clip["start"]?.doubleValue ?? 0
            let end: Double = clip["end"]?.doubleValue ?? 0
            let muted: Bool = clip["audio_muted"]?.boolValue ?? false
            let volume: Double = clip["audio_volume"]?.doubleValue ?? 1
            let playable: Bool = sourceURLs[sourceID] != nil
            let speed: Double = ClipSpeed.of(clip)
            parts.append("\(clipID)|\(sourceID)|\(start)|\(end)|\(muted)|\(volume)|\(playable)|\(speed)")
        }
        for voice in model.previewVoices {
            parts.append("vo|\(voice.key)|\(voice.start)|\(voice.sourceIn)|\(voice.sourceOut)|\(voice.volume)|\(voice.ducksVideo)|\(voice.url.lastPathComponent)")
        }
        // Build 17: Clean up (which clips play a cleaned copy that is ready) and the whole video's volume.
        for clip in model.selectedClips where model.clipPlaysCleaned(clip) {
            let source = clip["source_asset_id"]?.stringValue ?? ""
            let ready = CleanAudioCenter.shared.localURL(.source(source), project: model.project) != nil
            parts.append("clean|\(clip["clip_id"]?.stringValue ?? "")|\(ready)")
        }
        parts.append("master|\(model.masterVolume)|\(model.masterMuted)")
        for overlay in model.overlaysOnScreen where overlay.kind == .video && !overlay.style.muted {
            parts.append("ova|\(overlay.id)|\(overlay.start)|\(overlay.end)|\(overlay.sourceStart)|\(overlay.style.volume)")
        }
        parts.append("vovol|\(model.voiceOverVolume)|vv|\(model.videoVolume)")
        return parts.joined(separator: ";")
    }

    /// Video links are signed and expire; reuse a composition only for half
    /// of the shortest lifetime the server announced (5 minutes if unknown).
    private static func reuseWindow(for snapshot: DraftSnapshot?) -> Double {
        var shortest: Double?
        for source in snapshot?.sources ?? [] {
            guard let seconds = source["playback_expires_in"]?.doubleValue, seconds > 0 else { continue }
            shortest = min(shortest ?? seconds, seconds)
        }
        return (shortest ?? 600) / 2
    }

    /// One opened video per source, reused by every rebuild while its link is still valid, so an
    /// edit does not reopen every clip over the network (it used to, once per clip, one by one).
    private struct LoadedSource {
        /// Kept alive with its tracks: a track whose asset is gone cannot be inserted (black preview).
        let asset: AVURLAsset
        let video: AVAssetTrack
        let audio: AVAssetTrack?
        let transform: CGAffineTransform
        let openedAt: Date
    }
    private var sourceCache: [String: LoadedSource] = [:]
    /// Cleaned copies used by the current composition (kept alive with it).
    private var retainedCleanAssets: [AVURLAsset] = []

    private func loadSources(_ urls: [String: URL], needed: Set<String>, validFor: Double) async -> [String: LoadedSource] {
        var result: [String: LoadedSource] = [:]
        var missing: [String: URL] = [:]
        for id in needed {
            let wanted = urls[id]
            let reusable: Bool = {
                guard let hit = sourceCache[id] else { return false }
                if hit.asset.url.isFileURL { return hit.asset.url == wanted }        // local copy: never expires
                if wanted?.isFileURL == true { return false }                       // a local copy is ready now
                return Date().timeIntervalSince(hit.openedAt) < validFor            // signed link still valid
            }()
            if reusable, let hit = sourceCache[id] { result[id] = hit }
            else if let url = urls[id] { missing[id] = url }
        }
        await withTaskGroup(of: (String, LoadedSource?).self) { group in
            for (id, url) in missing {
                group.addTask {
                    let asset = AVURLAsset(url: url)
                    guard let video = try? await asset.loadTracks(withMediaType: .video).first else { return (id, nil) }
                    let audio = try? await asset.loadTracks(withMediaType: .audio).first
                    let transform = (try? await video.load(.preferredTransform)) ?? .identity
                    return (id, LoadedSource(asset: asset, video: video, audio: audio, transform: transform, openedAt: Date()))
                }
            }
            for await (id, loaded) in group {
                if let loaded { result[id] = loaded; sourceCache[id] = loaded }
            }
        }
        return result
    }

    /// Each rebuild's number: a newer one makes an older one (still waiting on a file or the
    /// network) stop before it replaces the player -- otherwise the older mix could win and play
    /// without the sound just added, or with the video's sound back on (build 17 on a real iPhone).
    private var buildGeneration = 0
    /// Audio files (voices, sounds, overlay videos) used by the current composition. They must stay
    /// alive: a track whose asset is gone is not inserted (try? hid it), and an optimized build frees
    /// a temporary asset at once -- the sounds were silent on the iPhone while the Debug Simulator
    /// build still played them.
    private var audioAssetCache: [String: AVURLAsset] = [:]

    /// The same opened file for the same file version (a voice recorded again at the same place is
    /// a new version: its modification date changes).
    private func audioAsset(for url: URL, used: inout Set<String>) -> AVURLAsset {
        let modified = (try? FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date)?
            .timeIntervalSince1970 ?? 0
        let key = "\(url.path)|\(modified)"
        #if DEBUG
        // Build 18 probe only: build 17's way (a temporary asset nobody keeps), to show the difference
        // in an optimized build.
        if ProcessInfo.processInfo.environment["CUTSELL_PROBE_TEMP_ASSET"] == "1" { return AVURLAsset(url: url) }
        #endif
        used.insert(key)
        if let hit = audioAssetCache[key] { return hit }
        let asset = AVURLAsset(url: url)
        audioAssetCache[key] = asset
        return asset
    }

    /// One sound / voice / overlay sound put on its own track, where the timeline says, cut to what
    /// its file really holds (an AAC file can be a few ms shorter than the length the draft gives).
    private func insertAudio(_ url: URL, from sourceIn: Double, to sourceOut: Double, at start: Double, cutEnd: Double,
                             composition: AVMutableComposition, entry: inout EditorMixEntry,
                             used: inout Set<String>) async -> AVMutableCompositionTrack? {
        let asset = audioAsset(for: url, used: &used)
        guard let source = try? await asset.loadTracks(withMediaType: .audio).first else {
            entry.problem = "no sound in the file"
            return nil
        }
        let range = (try? await source.load(.timeRange)) ?? CMTimeRange(start: .zero, duration: .zero)
        entry.fileSeconds = CMTimeGetSeconds(range.duration)
        let fileStart = max(0, CMTimeGetSeconds(range.start))
        let fileEnd = CMTimeGetSeconds(range.end)
        let from = max(sourceIn, fileStart)
        let to = min(sourceOut, fileEnd)
        let at = start + (from - sourceIn)
        let length = min(to - from, cutEnd - at)
        guard length > 0.02 else {
            entry.problem = at >= cutEnd ? "after the end of the cut" : "empty range"
            return nil
        }
        guard let track = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) else {
            entry.problem = "no track"
            return nil
        }
        do {
            try track.insertTimeRange(CMTimeRange(start: CMTime(seconds: from, preferredTimescale: 44_100),
                                                  duration: CMTime(seconds: length, preferredTimescale: 44_100)),
                                      of: source, at: CMTime(seconds: at, preferredTimescale: 44_100))
        } catch {
            entry.problem = "insert failed (\((error as NSError).code))"
            composition.removeTrack(track)
            CutSellDiagnostics.log("preview_audio_insert_failed", ["kind": entry.kind.rawValue, "code": String((error as NSError).code)])
            return nil
        }
        // What the composition really holds (not what was asked).
        entry.insertedSeconds = track.segments.filter { !$0.isEmpty }.reduce(0) { $0 + CMTimeGetSeconds($1.timeMapping.target.duration) }
        return track
    }

    func rebuild(from model: DraftEditorViewModel) async {
        EditorDiagnostics.shared.keyframeCount = model.transformKeyframes.count
        EditorDiagnostics.shared.aiKeyframeCount = model.transformKeyframes.filter(\.isAI).count
        let sourceURLs = sourceURLCatalog(sources: model.allSources)
        let signature = Self.signature(for: model, sourceURLs: sourceURLs)
        if isReady, !isBuilding, signature == compositionSignature, let builtAt = compositionBuiltAt,
           Date().timeIntervalSince(builtAt) < compositionReuseSec {
            refreshCaptionCues(from: model)
            return
        }
        buildGeneration += 1
        let generation = buildGeneration
        func superseded() -> Bool { generation != buildGeneration }
        let resumeAt = currentTime
        let wasPlaying = isPlaying || resumeAfterBuild
        resumeAfterBuild = wasPlaying
        compositionSignature = nil
        pause()
        isBuilding = true
        message = nil
        defer { if !superseded() { isBuilding = false; resumeAfterBuild = false } }

        guard !model.selectedClips.isEmpty else {
            player.replaceCurrentItem(with: nil)
            isReady = false
            message = "No selected clips to preview."
            return
        }

        do {
            let composition = AVMutableComposition()
            guard let videoCompositionTrack = composition.addMutableTrack(
                withMediaType: .video,
                preferredTrackID: kCMPersistentTrackID_Invalid
            ) else {
                throw DraftPlaybackError.compositionTrackUnavailable
            }
            // The video's own sound and the voice-over are separate tracks, each with its volume.
            let audioCompositionTrack = composition.addMutableTrack(
                withMediaType: .audio,
                preferredTrackID: kCMPersistentTrackID_Invalid
            )
            let audioParameters = audioCompositionTrack.map { AVMutableAudioMixInputParameters(track: $0) }
            var mixInputs: [AVMutableAudioMixInputParameters] = []

            let needed = Set(model.selectedClips.compactMap { $0["source_asset_id"]?.stringValue })
            let sources = await loadSources(sourceURLs, needed: needed, validFor: Self.reuseWindow(for: model.snapshot))
            if superseded() { return }

            var cursor = CMTime.zero
            var insertedVideo = false
            var setVideoTransform = false
            var windows: [CaptionClipWindow] = []
            /// The video's own sound per clip on the played timeline: (start, end, volume).
            var baseLevels: [(start: Double, end: Double, level: Double)] = []
            let master = model.masterMuted ? 0 : model.masterVolume
            let videoGain = max(0, min(2, model.videoVolume)) * master
            /// Cleaned copies of the sources (Clean up voice), opened once per rebuild.
            var cleanTracks: [String: AVAssetTrack] = [:]
            var cleanAssets: [AVURLAsset] = []

            for (clipIndex, clip) in model.selectedClips.enumerated() {
                guard let sourceID = clip["source_asset_id"]?.stringValue,
                      let source = sources[sourceID] else {
                    continue
                }
                let startSeconds = max(0, clip["start"]?.doubleValue ?? 0)
                let endSeconds = max(startSeconds, clip["end"]?.doubleValue ?? startSeconds)
                let clipDuration = endSeconds - startSeconds
                guard clipDuration >= 0.05 else { continue }

                let start = CMTime(seconds: startSeconds, preferredTimescale: 600)
                let durationTime = CMTime(seconds: clipDuration, preferredTimescale: 600)
                let range = CMTimeRange(start: start, duration: durationTime)
                // Speed: the clip plays (end - start) / speed seconds, like the export.
                let speed = ClipSpeed.of(clip)
                let played = CMTime(seconds: clipDuration / speed, preferredTimescale: 600)
                try videoCompositionTrack.insertTimeRange(range, of: source.video, at: cursor)
                insertedVideo = true

                windows.append(CaptionClipWindow(
                    clipIndex: clipIndex,
                    clipID: clip["clip_id"]?.stringValue ?? "",
                    timelineStart: CMTimeGetSeconds(cursor),
                    duration: clipDuration / speed,
                    speed: speed
                ))

                if !setVideoTransform {
                    videoCompositionTrack.preferredTransform = source.transform
                    setVideoTransform = true
                }

                if let audioCompositionTrack {
                    var sound = source.audio
                    if model.clipPlaysCleaned(clip),
                       let cleanURL = CleanAudioCenter.shared.localURL(.source(sourceID), project: model.project) {
                        if cleanTracks[sourceID] == nil {
                            let asset = AVURLAsset(url: cleanURL)
                            cleanAssets.append(asset)   // a track whose asset is gone can't be inserted
                            cleanTracks[sourceID] = try? await asset.loadTracks(withMediaType: .audio).first
                            if superseded() { return }
                        }
                        sound = cleanTracks[sourceID] ?? sound
                    }
                    if let sound { try audioCompositionTrack.insertTimeRange(range, of: sound, at: cursor) }
                }
                if speed != 1 {
                    composition.scaleTimeRange(CMTimeRange(start: cursor, duration: durationTime), toDuration: played)
                }
                let muted = clip["audio_muted"]?.boolValue ?? false
                let level = max(0, min(2, clip["audio_volume"]?.doubleValue ?? 1)) * videoGain
                baseLevels.append((CMTimeGetSeconds(cursor), CMTimeGetSeconds(cursor) + clipDuration / speed, muted ? 0 : level))
                cursor = CMTimeAdd(cursor, played)
            }

            guard insertedVideo, CMTimeGetSeconds(cursor) > 0 else {
                player.replaceCurrentItem(with: nil)
                isReady = false
                message = "Preview video is temporarily unavailable."
                CutSellDiagnostics.log("playback_failed", ["reason": "no_playable_clips"])
                return
            }
            let cutSeconds = CMTimeGetSeconds(cursor)
            // The video's own sound: each clip's volume, lowered (-12 dB, short ramps) while a
            // voice-over with "Lower video sound" plays -- the same as the export.
            if let audioParameters {
                let ducks = model.previewVoices.filter(\.ducksVideo).map { ($0.start, $0.start + ($0.sourceOut - $0.sourceIn)) }
                // Above 100 % (up to 200 %, like the export) the extra gain comes from a tap on the track.
                let peak = baseLevels.map(\.level).max() ?? 1
                let tapGain = max(1, peak)
                if tapGain > 1.001, let tap = AudioGainTap.make(gain: tapGain) { audioParameters.audioTapProcessor = tap }
                let clips = baseLevels.map { (start: $0.start, end: $0.end, volume: Float($0.level / tapGain)) }
                Self.applyBaseVolumes(audioParameters, clips: clips, ducks: ducks)
                mixInputs.append(audioParameters)
            }

            // Voices, music and CutSell sounds: each on its own track, where the timeline says. A
            // sound's file must be on this iPhone (SfxLibrary); when it arrives the preview rebuilds.
            var usedAudio: Set<String> = []
            var report = EditorMixReport()
            PlaybackMeter.shared.reset()
            var measured: [(index: Int, track: AVMutableCompositionTrack, start: Double, length: Double, volume: Double)] = []
            for placement in model.previewVoices {
                let effective = max(0, min(2, placement.volume)) * master
                var entry = EditorMixEntry(id: placement.key, kind: EditorMixEntry.Kind(rawValue: placement.kind) ?? .voice,
                                           label: placement.label, start: placement.start,
                                           length: max(0, placement.sourceOut - placement.sourceIn),
                                           fileOnPhone: FileManager.default.fileExists(atPath: placement.url.path),
                                           fileSeconds: nil, insertedSeconds: 0, volume: effective)
                let track = await insertAudio(placement.url, from: placement.sourceIn, to: placement.sourceOut, at: placement.start,
                                              cutEnd: cutSeconds, composition: composition, entry: &entry, used: &usedAudio)
                if superseded() { return }
                if let track {
                    let parameters = AVMutableAudioMixInputParameters(track: track)
                    AudioGainTap.apply(effective, to: parameters, meter: placement.key)
                    mixInputs.append(parameters)
                    measured.append((report.entries.count, track, entry.start, entry.insertedSeconds, effective))
                }
                report.entries.append(entry)
            }
            // Sounds on the timeline whose file is not on this iPhone yet (shown in Diagnostics).
            for clip in model.soundClips where clip.sfxID.flatMap({ SfxLibrary.localURL($0) }) == nil {
                report.entries.append(EditorMixEntry(id: clip.id, kind: .sound, label: SfxLibrary.name(of: clip.sfxID ?? ""),
                                                     start: clip.start, length: clip.length, fileOnPhone: false, fileSeconds: nil,
                                                     insertedSeconds: 0, volume: clip.muted ? 0 : clip.volume,
                                                     problem: "waiting for the download"))
            }

            // Video overlays with their sound on.
            for overlay in model.overlaysOnScreen where overlay.kind == .video && !overlay.style.muted {
                guard let url = overlay.url, url.isFileURL else { continue }
                let effective = max(0, min(2, overlay.style.volume)) * master
                var entry = EditorMixEntry(id: overlay.id, kind: .overlay, label: "Video overlay", start: overlay.start,
                                           length: overlay.end - overlay.start, fileOnPhone: true, fileSeconds: nil,
                                           insertedSeconds: 0, volume: effective)
                let track = await insertAudio(url, from: overlay.sourceStart, to: overlay.sourceStart + (overlay.end - overlay.start),
                                              at: overlay.start, cutEnd: cutSeconds, composition: composition, entry: &entry,
                                              used: &usedAudio)
                if superseded() { return }
                if let track {
                    let parameters = AVMutableAudioMixInputParameters(track: track)
                    AudioGainTap.apply(effective, to: parameters)
                    mixInputs.append(parameters)
                }
            }

            // The preview is the CUT: nothing past its last clip (a source track can run longer than
            // the ranges taken from it; the build-11 preview kept playing ~19 s of the original).
            let cutEnd = cursor
            if composition.duration > cutEnd {
                composition.removeTimeRange(CMTimeRange(start: cutEnd, end: composition.duration))
            }
            let item = AVPlayerItem(asset: composition)
            // Sped-up / slowed clips keep the voice's pitch (no chipmunk), like the export's atempo.
            item.audioTimePitchAlgorithm = .spectral
            item.forwardPlaybackEndTime = cutEnd
            let mix = AVMutableAudioMix()
            mix.inputParameters = mixInputs
            item.audioMix = mix
            player.replaceCurrentItem(with: item)
            retainedCleanAssets = cleanAssets
            audioAssetCache = audioAssetCache.filter { usedAudio.contains($0.key) }   // files the new mix no longer uses
            duration = max(0, cutSeconds)
            isReady = true
            clipWindows = windows
            compositionSignature = signature
            compositionBuiltAt = Date()
            compositionReuseSec = sourceURLs.values.allSatisfy(\.isFileURL) ? .greatestFiniteMagnitude : Self.reuseWindow(for: model.snapshot)
            refreshCaptionCues(from: model)
            // Stay where the creator was (an edit no longer jumps back to the start).
            seek(to: min(resumeAt, max(0, duration - 0.05)))
            if wasPlaying { togglePlayback() }

            report.videoLevels = baseLevels
            report.videoVolume = model.videoVolume
            report.duration = duration
            EditorDiagnostics.shared.mix = report
            let inMix = report.entries.filter(\.inMix).count
            CutSellDiagnostics.log("playback_ready", ["duration_s": String(format: "%.2f", duration),
                                                     "sounds": String(report.entries.count), "in_mix": String(inMix),
                                                     "missing": String(report.entries.filter { !$0.fileOnPhone }.count),
                                                     "video_volume": String(format: "%.2f", model.videoVolume)])
            for entry in report.entries where !entry.inMix && entry.volume > 0.001 {
                CutSellDiagnostics.log("preview_sound_not_in_mix", ["kind": entry.kind.rawValue, "problem": entry.problem ?? "unknown"])
            }
            // Measure what each sound's track really carries in its window (Diagnostics, probe).
            let frozen = composition.copy() as? AVComposition ?? composition
            let frozenTracks = measured.map { item in frozen.track(withTrackID: item.track.trackID) }
            let jobs = measured
            Task { [weak self] in
                var levels: [Int: Double] = [:]
                for (job, track) in zip(jobs, frozenTracks) {
                    guard let track else { continue }
                    let window = min(job.length, 2.0)
                    if let db = await EditorDiagnostics.measure(asset: frozen, tracks: [track], start: job.start, end: job.start + window) {
                        levels[job.index] = job.volume > 0.0001 ? db + 20 * log10(job.volume) : -120
                    }
                }
                guard let self, generation == self.buildGeneration else { return }
                var updated = EditorDiagnostics.shared.mix
                for (index, db) in levels where index < updated.entries.count { updated.entries[index].measuredDB = db }
                EditorDiagnostics.shared.mix = updated
                #if DEBUG
                NSLog("CUTSELLB18 %@", updated.summary)
                #endif
            }
        } catch {
            if superseded() { return }
            player.replaceCurrentItem(with: nil)
            isReady = false
            message = "Preview is temporarily unavailable. The draft is still safe."
            CutSellDiagnostics.log("playback_failed", ["reason": String(describing: type(of: error))])
        }
    }

    /// Base sound volume over time: each clip's level, with a -12 dB dip (0.12 s ramps) under every
    /// ducking voice.
    static func applyBaseVolumes(_ parameters: AVMutableAudioMixInputParameters,
                                 clips: [(start: Double, end: Double, volume: Float)],
                                 ducks: [(Double, Double)]) {
        func level(at time: Double) -> Float { clips.first { time >= $0.start && time < $0.end }?.volume ?? clips.last?.volume ?? 1 }
        let dip: Float = 0.25   // -12 dB
        let ramp = 0.12
        func cm(_ seconds: Double) -> CMTime { CMTime(seconds: max(0, seconds), preferredTimescale: 600) }
        func ducked(_ time: Double) -> Bool { ducks.contains { time >= $0.0 && time < $0.1 } }
        for clip in clips {
            parameters.setVolume(clip.volume * (ducked(clip.start) ? dip : 1), at: cm(clip.start))
        }
        for (start, end) in ducks where end - start > 0.05 {
            let v = level(at: start)
            parameters.setVolumeRamp(fromStartVolume: v, toEndVolume: v * dip,
                                     timeRange: CMTimeRange(start: cm(start), duration: cm(min(ramp, (end - start) / 2))))
            let back = max(start + min(ramp, (end - start) / 2), end - ramp)
            let w = level(at: back)
            parameters.setVolumeRamp(fromStartVolume: w * dip, toEndVolume: w,
                                     timeRange: CMTimeRange(start: cm(back), duration: cm(end - back)))
        }
    }

    func togglePlayback() {
        guard isReady else { return }
        if isPlaying {
            pause()
        } else {
            if duration > 0, currentTime >= duration - 0.05 {
                seek(to: 0)
            }
            EditorAudio.preparePlayback()
            player.play()
            isPlaying = true
        }
    }

    func pause() {
        player.pause()
        isPlaying = false
        recordHeardLevels()
    }

    /// Diagnostics: what each sound / voice really sounded like while the preview played.
    private func recordHeardLevels() {
        let heard = PlaybackMeter.shared.heard()
        guard !heard.isEmpty else { return }
        var report = EditorDiagnostics.shared.mix
        var changed = false
        for index in report.entries.indices {
            if let db = heard[report.entries[index].id], report.entries[index].heardDB != db {
                report.entries[index].heardDB = db
                changed = true
            }
        }
        guard changed else { return }
        EditorDiagnostics.shared.mix = report
        let levels = report.entries.compactMap(\.heardDB)
        CutSellDiagnostics.log("sounds_heard", ["played": String(levels.count),
                                                "quietest_db": String(format: "%.1f", levels.min() ?? -120),
                                                "loudest_db": String(format: "%.1f", levels.max() ?? -120)])
    }

    /// End of the cut: stop the player (not just the button) and stay on the last frame.
    private func stopAtEnd() {
        player.pause()
        isPlaying = false
        recordHeardLevels()
        currentTime = duration
    }

    func seek(to seconds: Double) {
        guard isReady else { return }
        let target = max(0, min(duration, seconds))
        player.seek(to: CMTime(seconds: target, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
        currentTime = target
    }

    private func installTimeObserver() {
        timeObserver = player.addPeriodicTimeObserver(
            // Often enough for a three-word caption (and the word being
            // spoken in the Highlight look) to change on time.
            forInterval: CMTime(seconds: 1.0 / 30.0, preferredTimescale: 600),
            queue: .main
        ) { [weak self] time in
            guard let self else { return }
            Task { @MainActor in
                let seconds = max(0, CMTimeGetSeconds(time))
                // Never past the end of the cut (the timeline stops there too).
                self.currentTime = self.duration > 0 ? min(seconds, self.duration) : seconds
                if self.isPlaying, self.duration > 0, seconds >= self.duration - 0.03 {
                    self.stopAtEnd()
                }
            }
        }
    }

    /// The video of each source: the copy on this iPhone when there is one (no download, no expiry),
    /// else the signed link -- and that one is downloaded in the background for next time.
    private func sourceURLCatalog(from snapshot: DraftSnapshot?) -> [String: URL] {
        sourceURLCatalog(sources: snapshot?.sources.compactMap(\.objectValue) ?? [])
    }

    /// `sources`: the cut's originals plus full-screen B-roll videos added in the editor.
    private func sourceURLCatalog(sources: [[String: JSONValue]]) -> [String: URL] {
        var output: [String: URL] = [:]
        for object in sources {
            guard let sourceID = object["source_asset_id"]?.stringValue else { continue }
            let uri = object["uri"]?.stringValue
            // The 720p copy of the whole original first (lighter, and a stretched clip has its
            // hidden material), then the original kept on this iPhone (the copy is made from it).
            if let uri, let proxy = PreviewProxyCache.localURL(forURI: uri) {
                output[sourceID] = proxy
                continue
            }
            if let uri, let local = SourceVideoCache.localURL(forURI: uri) {
                output[sourceID] = local
                Task.detached(priority: .utility) { await PreviewProxyCache.shared.ensure(uri: uri, original: local) }
                continue
            }
            guard let rawURL = object["playback_url"]?.stringValue, let url = URL(string: rawURL) else { continue }
            output[sourceID] = url
            if let uri { Task.detached(priority: .utility) { await SourceVideoCache.shared.prefetch(uri: uri, remote: url) } }
        }
        return output
    }

    /// Where the preview's video comes from (for the probe and diagnostics).
    func sourcesAreLocal(_ snapshot: DraftSnapshot?) -> Bool {
        let urls = sourceURLCatalog(from: snapshot)
        return !urls.isEmpty && urls.values.allSatisfy(\.isFileURL)
    }
}

enum DraftPlaybackError: Error {
    case compositionTrackUnavailable
}
