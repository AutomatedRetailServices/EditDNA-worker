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

                Text("\(time(playback.currentTime)) / \(time(playback.duration))")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 84, alignment: .trailing)
            }

            if playback.isBuilding {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Preparing preview…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else if let message = playback.message {
                Text(message)
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

    init() {
        CaptionFontLoader.registerIfNeeded()
        installTimeObserver()
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
            parts.append("\(clipID)|\(sourceID)|\(start)|\(end)|\(muted)|\(volume)|\(playable)")
        }
        for placement in model.timelineComposition?.voiceOverPlacements ?? [] {
            let local = VoiceOverStore.url(for: placement.assetID) != nil
            parts.append("vo|\(placement.placementID)|\(placement.timelineStartSec)|\(placement.timelineEndSec)|\(placement.sourceInSec)|\(local)")
        }
        parts.append("vovol|\(model.voiceOverVolume)")
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

    private func loadSources(_ urls: [String: URL], needed: Set<String>, validFor: Double) async -> [String: LoadedSource] {
        var result: [String: LoadedSource] = [:]
        var missing: [String: URL] = [:]
        for id in needed {
            if let hit = sourceCache[id], Date().timeIntervalSince(hit.openedAt) < validFor { result[id] = hit }
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

    func rebuild(from model: DraftEditorViewModel) async {
        let sourceURLs = sourceURLCatalog(from: model.snapshot)
        let signature = Self.signature(for: model, sourceURLs: sourceURLs)
        if isReady, signature == compositionSignature, let builtAt = compositionBuiltAt,
           Date().timeIntervalSince(builtAt) < compositionReuseSec {
            refreshCaptionCues(from: model)
            return
        }
        let resumeAt = currentTime
        let wasPlaying = isPlaying
        compositionSignature = nil
        pause()
        isBuilding = true
        message = nil
        defer { isBuilding = false }

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

            var cursor = CMTime.zero
            var insertedVideo = false
            var setVideoTransform = false
            var windows: [CaptionClipWindow] = []

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
                try videoCompositionTrack.insertTimeRange(range, of: source.video, at: cursor)
                insertedVideo = true

                windows.append(CaptionClipWindow(
                    clipIndex: clipIndex,
                    clipID: clip["clip_id"]?.stringValue ?? "",
                    timelineStart: CMTimeGetSeconds(cursor),
                    duration: clipDuration
                ))

                if !setVideoTransform {
                    videoCompositionTrack.preferredTransform = source.transform
                    setVideoTransform = true
                }

                if let audioCompositionTrack, let sourceAudio = source.audio {
                    try audioCompositionTrack.insertTimeRange(range, of: sourceAudio, at: cursor)
                    let muted = clip["audio_muted"]?.boolValue ?? false
                    let volume = Float(max(0, min(1, clip["audio_volume"]?.doubleValue ?? 1)))
                    audioParameters?.setVolume(muted ? 0 : volume, at: cursor)
                }
                cursor = CMTimeAdd(cursor, durationTime)
            }

            guard insertedVideo, CMTimeGetSeconds(cursor) > 0 else {
                player.replaceCurrentItem(with: nil)
                isReady = false
                message = "Preview video is temporarily unavailable."
                CutSellDiagnostics.log("playback_failed", ["reason": "no_playable_clips"])
                return
            }
            if let audioParameters { mixInputs.append(audioParameters) }

            // Voice-over track(s): recordings kept on this iPhone, placed where the timeline says.
            for placement in model.timelineComposition?.voiceOverPlacements ?? [] {
                guard let url = VoiceOverStore.url(for: placement.assetID),
                      let voice = try? await AVURLAsset(url: url).loadTracks(withMediaType: .audio).first,
                      let track = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
                else { continue }
                let length = min(placement.sourceOutSec - placement.sourceInSec,
                                 max(0, CMTimeGetSeconds(cursor) - placement.timelineStartSec))
                guard length > 0.05 else { continue }
                try? track.insertTimeRange(
                    CMTimeRange(start: CMTime(seconds: placement.sourceInSec, preferredTimescale: 600),
                                duration: CMTime(seconds: length, preferredTimescale: 600)),
                    of: voice, at: CMTime(seconds: placement.timelineStartSec, preferredTimescale: 600)
                )
                let parameters = AVMutableAudioMixInputParameters(track: track)
                parameters.setVolume(Float(max(0, min(1, model.voiceOverVolume))), at: .zero)
                mixInputs.append(parameters)
            }

            let item = AVPlayerItem(asset: composition)
            let mix = AVMutableAudioMix()
            mix.inputParameters = mixInputs
            item.audioMix = mix
            player.replaceCurrentItem(with: item)
            duration = max(0, CMTimeGetSeconds(cursor))
            isReady = true
            clipWindows = windows
            compositionSignature = signature
            compositionBuiltAt = Date()
            compositionReuseSec = Self.reuseWindow(for: model.snapshot)
            refreshCaptionCues(from: model)
            // Stay where the creator was (an edit no longer jumps back to the start).
            seek(to: min(resumeAt, max(0, duration - 0.05)))
            if wasPlaying { togglePlayback() }
            CutSellDiagnostics.log("playback_ready", ["duration_s": String(format: "%.2f", duration)])
        } catch {
            player.replaceCurrentItem(with: nil)
            isReady = false
            message = "Preview is temporarily unavailable. The draft is still safe."
            CutSellDiagnostics.log("playback_failed", ["reason": String(describing: type(of: error))])
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
                self.currentTime = max(0, CMTimeGetSeconds(time))
                if self.duration > 0, self.currentTime >= self.duration - 0.05 {
                    self.isPlaying = false
                }
            }
        }
    }

    private func sourceURLCatalog(from snapshot: DraftSnapshot?) -> [String: URL] {
        var output: [String: URL] = [:]
        for source in snapshot?.sources ?? [] {
            guard let object = source.objectValue,
                  let sourceID = object["source_asset_id"]?.stringValue,
                  let rawURL = object["playback_url"]?.stringValue,
                  let url = URL(string: rawURL) else { continue }
            output[sourceID] = url
        }
        return output
    }
}

enum DraftPlaybackError: Error {
    case compositionTrackUnavailable
}
