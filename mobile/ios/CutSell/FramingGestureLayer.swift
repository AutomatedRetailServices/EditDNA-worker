import SwiftUI
import UIKit

/// Build 18: pinch (zoom), two-finger twist (turn) and drag (move) on the video, with UIKit's own
/// recognizers. Build 17 used SwiftUI gestures stacked on the video card; on Swanny's iPhone they
/// did nothing, while the Simulator's synthesized touches passed. UIKit pinch / rotation / pan
/// recognize together (like Photos and CapCut) and report every step.
///
/// The layer sits over the picture and under overlays and captions, so a touch that starts on an
/// overlay still goes to that overlay. A tap is left to SwiftUI (play / pause).
struct FramingGestureLayer: UIViewRepresentable {
    struct Change {
        /// Pinch since the gesture began (1 = unchanged).
        var scale: CGFloat = 1
        /// Twist since the gesture began, degrees clockwise.
        var rotation: Double = 0
        /// Finger movement since the gesture began, points.
        var translation: CGSize = .zero
        var touches = 0
    }

    let onBegin: () -> Void
    let onChange: (Change) -> Void
    let onEnd: (Change) -> Void
    /// A plain tap (play / pause), in case SwiftUI's own tap does not see touches on this view.
    var onTap: () -> Void = {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .clear
        view.isMultipleTouchEnabled = true
        view.accessibilityIdentifier = "videoFramingLayer"
        let pinch = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.pinched(_:)))
        let rotation = UIRotationGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.rotated(_:)))
        let pan = UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.panned(_:)))
        pan.maximumNumberOfTouches = 2
        for recognizer in [pinch, rotation, pan] as [UIGestureRecognizer] {
            recognizer.delegate = context.coordinator
            recognizer.cancelsTouchesInView = false
            view.addGestureRecognizer(recognizer)
        }
        let tap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.tapped(_:)))
        tap.cancelsTouchesInView = false
        view.addGestureRecognizer(tap)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.parent = self
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var parent: FramingGestureLayer
        private var change = Change()
        private var active: Set<ObjectIdentifier> = []

        init(_ parent: FramingGestureLayer) { self.parent = parent }

        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                               shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
            // Pinch, twist and move together; never together with the timeline's or a sheet's own gestures.
            other.view === gestureRecognizer.view
        }

        private func track(_ recognizer: UIGestureRecognizer, update: () -> Void) {
            let id = ObjectIdentifier(recognizer)
            switch recognizer.state {
            case .began:
                if active.isEmpty {
                    change = Change()
                    parent.onBegin()
                }
                active.insert(id)
                update()
                change.touches = max(change.touches, recognizer.numberOfTouches)
                parent.onChange(change)
            case .changed:
                update()
                change.touches = max(change.touches, recognizer.numberOfTouches)
                parent.onChange(change)
            case .ended, .cancelled, .failed:
                guard active.remove(id) != nil else { return }
                if active.isEmpty { parent.onEnd(change) }
            default:
                break
            }
        }

        @objc func tapped(_ recognizer: UITapGestureRecognizer) {
            if recognizer.state == .ended { parent.onTap() }
        }

        @objc func pinched(_ recognizer: UIPinchGestureRecognizer) {
            track(recognizer) { change.scale = recognizer.scale }
        }

        @objc func rotated(_ recognizer: UIRotationGestureRecognizer) {
            track(recognizer) { change.rotation = Double(recognizer.rotation) * 180 / .pi }
        }

        @objc func panned(_ recognizer: UIPanGestureRecognizer) {
            track(recognizer) {
                let point = recognizer.translation(in: recognizer.view)
                change.translation = CGSize(width: point.x, height: point.y)
            }
        }
    }
}

/// How a twist becomes a turn: a pinch always twists a little, so turning starts past 6°; close to
/// straight (0 / 90 / 180 / 270, within 5°) it snaps.
enum FramingMath {
    static let twistDeadZone = 6.0

    static func turned(base: Double, twist: Double) -> Double {
        guard abs(twist) > twistDeadZone else { return base }
        var turned = base + twist - (twist > 0 ? twistDeadZone : -twistDeadZone)
        for snap in stride(from: -360.0, through: 360.0, by: 90) where abs(turned - snap) < 5 { turned = snap }
        return turned
    }

    /// The framing after a gesture on a video clip: zoom about the centre, turn, then move by a
    /// fraction of the frame (the export's order).
    static func apply(_ change: FramingGestureLayer.Change, to base: TransformValue, frame: CGSize) -> TransformValue {
        var next = base
        next.scale = base.scale * Double(change.scale)
        next.rotation = turned(base: base.rotation, twist: change.rotation)
        next.x = base.x + Double(change.translation.width / max(1, frame.width))
        next.y = base.y + Double(change.translation.height / max(1, frame.height))
        return next
    }
}
