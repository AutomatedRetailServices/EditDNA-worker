import CoreGraphics
import Foundation

/// PO 21 Nov 2026 (Figma 14d `3092:648`): "la línea negra que se mueve en play… debe moverse independiente… y que con el
/// dedo se pueda desplazar". The play line now moves over the tracks (it used to stand still while the tracks slid
/// under it, CapCut style). The tracks show the time from `viewStart` on; they turn the page only when the line reaches
/// the right side while playing. A finger on the line (or the time ruler) drags it; a finger elsewhere slides the tracks
/// without changing the time.
enum TimelineViewport {
    /// The line turns the page when it passes this share of the visible tracks ...
    static let pageAt = 0.85
    /// ... and lands here after the page turn.
    static let landAt = 0.15
    /// How close (points) a finger must start to the line to grab it.
    static let grabDistance: CGFloat = 24

    /// Seconds of the cut the tracks show at once.
    static func visibleSeconds(tracksWidth: CGFloat, pixelsPerSecond: CGFloat) -> Double {
        Double(max(40, tracksWidth) / max(0.001, pixelsPerSecond))
    }

    /// Latest `viewStart` that still makes sense: the end of the cut may reach the middle of the tracks.
    static func maxStart(duration: Double, visible: Double) -> Double { max(0, duration - visible / 2) }

    static func clamp(_ start: Double, duration: Double, visible: Double) -> Double {
        min(max(0, start), maxStart(duration: duration, visible: visible))
    }

    /// Keeps the line on screen: when it goes past the right side (playing, or a jump from elsewhere) or before the
    /// left side, the tracks move so it stands near the left again.
    static func follow(viewStart: Double, now: Double, visible: Double, duration: Double) -> Double {
        if now > viewStart + visible * pageAt || now < viewStart {
            return clamp(now - visible * landAt, duration: duration, visible: visible)
        }
        return viewStart
    }

    /// Time under a finger at `x` (points from the left of the tracks).
    static func time(atX x: CGFloat, viewStart: Double, pixelsPerSecond: CGFloat, duration: Double) -> Double {
        min(duration, max(0, viewStart + Double(x / max(0.001, pixelsPerSecond))))
    }

    /// A drag that starts this close to the line (or on the ruler) moves the line.
    static func grabsLine(startX: CGFloat, lineX: CGFloat, onRuler: Bool) -> Bool {
        onRuler || abs(startX - lineX) <= grabDistance
    }

    /// Two-finger zoom: the line keeps its place on screen.
    static func startKeepingLine(now: Double, lineOffset: CGFloat, pixelsPerSecond: CGFloat,
                                 duration: Double, visible: Double) -> Double {
        clamp(now - Double(lineOffset / max(0.001, pixelsPerSecond)), duration: duration, visible: visible)
    }
}
