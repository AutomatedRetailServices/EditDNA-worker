import CoreGraphics
import Foundation

/// PO 21 Nov 2026 (Figma 14d `3092:648`): "la línea negra que se mueve en play… debe moverse independiente… y que con el
/// dedo se pueda desplazar". The play line now moves over the tracks (it used to stand still while the tracks slid
/// under it, CapCut style). The tracks show the time from `viewStart` on; they turn the page only when the line reaches
/// the right side while playing. A finger on the line (or the time ruler) drags it; a finger elsewhere slides the tracks
/// without changing the time.
enum TimelineViewport {
    /// PO 21 Nov, build 107 ("el time line no se mueve al compás… cuando termina es como que se mueve de bloque"):
    /// no more page turns. The line walks to this share of the visible tracks; from there the tracks slide smoothly
    /// under it (frame by frame) and the line stays put, until the end of the cut comes into view.
    static let holdAt = 0.5
    /// The end of the cut may come this far across the tracks (then the line walks on to it).
    static let endAt = 0.85
    /// After a jump back (before the left side) the line lands here.
    static let landAt = 0.15
    /// How close (points) a finger must start to the line to grab it.
    static let grabDistance: CGFloat = 24
    /// Moving the line with a finger: this close to a side (points), the tracks slide along with it.
    static let edgeZone: CGFloat = 44
    /// Fastest slide at the very side: share of the visible tracks per second.
    static let edgeSpeed = 1.2

    /// Seconds of the cut the tracks show at once.
    static func visibleSeconds(tracksWidth: CGFloat, pixelsPerSecond: CGFloat) -> Double {
        Double(max(40, tracksWidth) / max(0.001, pixelsPerSecond))
    }

    /// Latest `viewStart`: the end of the cut stands `endAt` across the tracks.
    static func maxStart(duration: Double, visible: Double) -> Double { max(0, duration - visible * endAt) }

    static func clamp(_ start: Double, duration: Double, visible: Double) -> Double {
        min(max(0, start), maxStart(duration: duration, visible: visible))
    }

    /// Keeps the line on screen without jumps: past the middle, the tracks follow it smoothly; before the left
    /// side (a jump back from elsewhere) they come back so it stands near the left.
    static func follow(viewStart: Double, now: Double, visible: Double, duration: Double) -> Double {
        if now < viewStart { return clamp(now - visible * landAt, duration: duration, visible: visible) }
        if now > viewStart + visible * holdAt {
            return clamp(max(viewStart, now - visible * holdAt), duration: duration, visible: visible)
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

    /// Moving the line with a finger near a side: how far (seconds) the tracks slide in `dt` seconds.
    /// Negative = back (left side), 0 = the finger is not near a side.
    static func edgeSlide(fingerX: CGFloat, tracksWidth: CGFloat, visible: Double, dt: Double) -> Double {
        let fromRight = tracksWidth - fingerX
        if fromRight < edgeZone {
            let depth = Double(min(1, max(0, (edgeZone - fromRight) / edgeZone)))
            return visible * edgeSpeed * depth * dt
        }
        if fingerX < edgeZone {
            let depth = Double(min(1, max(0, (edgeZone - fingerX) / edgeZone)))
            return -visible * edgeSpeed * depth * dt
        }
        return 0
    }

    /// Two-finger zoom: the line keeps its place on screen.
    static func startKeepingLine(now: Double, lineOffset: CGFloat, pixelsPerSecond: CGFloat,
                                 duration: Double, visible: Double) -> Double {
        clamp(now - Double(lineOffset / max(0.001, pixelsPerSecond)), duration: duration, visible: visible)
    }
}
