import SwiftUI
import AVKit
import AVFoundation

struct DraftPlaybackView: View {
    @ObservedObject var model: DraftEditorViewModel
    @StateObject private var playback = DraftPlaybackController()

    var body: some View {
        VStack(spacing: 10) {
            VideoPlayer(player: playback.player)
                .aspectRatio(9.0 / 16.0, contentMode: .fit)
                .frame(maxHeight: 430)
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
        .task(id: model.snapshot?.revision) {
            await playback.rebuild(from: model)
        }
        .onDisappear { playback.pause() }
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

    private var timeObserver: Any?

    init() {
        installTimeObserver()
    }

    func rebuild(from model: DraftEditorViewModel) async {
        pause()
        isBuilding = true
        isReady = false
        message = nil
        currentTime = 0
        duration = 0
        defer { isBuilding = false }

        let sourceURLs = sourceURLCatalog(from: model.snapshot)
        guard !model.selectedClips.isEmpty else {
            player.replaceCurrentItem(with: nil)
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
            let audioCompositionTrack = composition.addMutableTrack(
                withMediaType: .audio,
                preferredTrackID: kCMPersistentTrackID_Invalid
            )
            let audioParameters = audioCompositionTrack.map { AVMutableAudioMixInputParameters(track: $0) }

            var cursor = CMTime.zero
            var insertedVideo = false
            var setVideoTransform = false

            for clip in model.selectedClips {
                guard let sourceID = clip["source_asset_id"]?.stringValue,
                      let sourceURL = sourceURLs[sourceID] else {
                    continue
                }
                let startSeconds = max(0, clip["start"]?.doubleValue ?? 0)
                let endSeconds = max(startSeconds, clip["end"]?.doubleValue ?? startSeconds)
                let clipDuration = endSeconds - startSeconds
                guard clipDuration >= 0.05 else { continue }

                let asset = AVURLAsset(url: sourceURL)
                let videoTracks = try await asset.loadTracks(withMediaType: .video)
                guard let sourceVideo = videoTracks.first else { continue }
                let start = CMTime(seconds: startSeconds, preferredTimescale: 600)
                let durationTime = CMTime(seconds: clipDuration, preferredTimescale: 600)
                let range = CMTimeRange(start: start, duration: durationTime)
                try videoCompositionTrack.insertTimeRange(range, of: sourceVideo, at: cursor)
                insertedVideo = true

                if !setVideoTransform {
                    videoCompositionTrack.preferredTransform = try await sourceVideo.load(.preferredTransform)
                    setVideoTransform = true
                }

                if let audioCompositionTrack {
                    let audioTracks = try await asset.loadTracks(withMediaType: .audio)
                    if let sourceAudio = audioTracks.first {
                        try audioCompositionTrack.insertTimeRange(range, of: sourceAudio, at: cursor)
                        let muted = clip["audio_muted"]?.boolValue ?? false
                        let volume = Float(max(0, min(1, clip["audio_volume"]?.doubleValue ?? 1)))
                        audioParameters?.setVolume(muted ? 0 : volume, at: cursor)
                    }
                }
                cursor = CMTimeAdd(cursor, durationTime)
            }

            guard insertedVideo, CMTimeGetSeconds(cursor) > 0 else {
                player.replaceCurrentItem(with: nil)
                message = "Preview video is temporarily unavailable."
                return
            }

            let item = AVPlayerItem(asset: composition)
            if let audioParameters {
                let mix = AVMutableAudioMix()
                mix.inputParameters = [audioParameters]
                item.audioMix = mix
            }
            player.replaceCurrentItem(with: item)
            duration = max(0, CMTimeGetSeconds(cursor))
            isReady = true
        } catch {
            player.replaceCurrentItem(with: nil)
            message = "Preview is temporarily unavailable. The draft is still safe."
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
            forInterval: CMTime(seconds: 0.1, preferredTimescale: 600),
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
