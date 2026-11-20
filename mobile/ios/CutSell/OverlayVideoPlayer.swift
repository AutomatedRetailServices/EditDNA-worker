import AVFoundation
import SwiftUI
import UIKit

/// PO 20 Nov 2026 ("el error sigue en play… el broll… es lento el video"): a video overlay (B-roll over the voice,
/// or one she added) used to be drawn in the editor as still frames read one by one (at most 5 a second, from the
/// server's full-size copy), so it looked slow and jumpy. Now it is a real muted video that follows the editor's play
/// head; the sound still comes from the editor's own mix.
enum OverlayVideoSync {
    enum Step: Equatable { case none, seek(Double), seekAndPlay(Double), pauseAndSeek(Double), pause }

    /// Playing: start with the play head, and seek only when it drifts more than `playingDrift` (a seek every frame
    /// would stutter). Paused: show the exact frame under the play head.
    static let playingDrift = 0.3
    static let pausedDrift = 0.04

    static func step(playerTime: Double, wanted: Double, editorPlaying: Bool, playerRunning: Bool) -> Step {
        let off = playerTime.isFinite ? abs(playerTime - wanted) : .infinity
        if editorPlaying {
            if !playerRunning { return .seekAndPlay(wanted) }
            return off > playingDrift ? .seek(wanted) : .none
        }
        if playerRunning { return .pauseAndSeek(wanted) }
        return off > pausedDrift ? .seek(wanted) : .none
    }
}

struct OverlaySyncedVideo: UIViewRepresentable {
    let url: URL
    /// Second of the overlay's own video to show now.
    let seconds: Double
    let playing: Bool

    final class LayerView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }

    final class Coordinator {
        let player = AVPlayer()
        private var current: URL?
        private var seeking = false

        init() {
            player.isMuted = true                     // the editor's mix plays the overlay's sound
            player.actionAtItemEnd = .pause
        }

        func sync(url: URL, seconds: Double, playing: Bool) {
            if current != url {
                current = url
                player.replaceCurrentItem(with: AVPlayerItem(url: url))
            }
            guard !seeking else { return }
            let step = OverlayVideoSync.step(playerTime: player.currentTime().seconds, wanted: seconds,
                                             editorPlaying: playing, playerRunning: player.rate != 0)
            switch step {
            case .none: break
            case .pause: player.pause()
            case .seek(let t): seek(t, exact: !playing, then: nil)
            case .seekAndPlay(let t): seek(t, exact: false) { [weak self] in self?.player.play() }
            case .pauseAndSeek(let t): player.pause(); seek(t, exact: true, then: nil)
            }
        }

        private func seek(_ t: Double, exact: Bool, then: (() -> Void)?) {
            seeking = true
            let tolerance = exact ? CMTime.zero : CMTime(seconds: 0.1, preferredTimescale: 600)
            player.seek(to: CMTime(seconds: max(0, t), preferredTimescale: 600),
                        toleranceBefore: tolerance, toleranceAfter: tolerance) { [weak self] _ in
                DispatchQueue.main.async {
                    self?.seeking = false
                    then?()
                }
            }
        }

        func stop() {
            player.pause()
            player.replaceCurrentItem(with: nil)
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> LayerView {
        let view = LayerView()
        view.isUserInteractionEnabled = false
        view.playerLayer.videoGravity = .resizeAspectFill
        view.playerLayer.player = context.coordinator.player
        return view
    }

    func updateUIView(_ view: LayerView, context: Context) {
        context.coordinator.sync(url: url, seconds: seconds, playing: playing)
    }

    static func dismantleUIView(_ view: LayerView, coordinator: Coordinator) {
        coordinator.stop()
    }
}
