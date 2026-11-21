import XCTest
@testable import CutSell

/// PO 21 Nov 2026 (Figma 14d): the play line moves by itself and can be dragged with a finger.
final class TimelineViewportTests: XCTestCase {
    func testWhilePlayingTheLineWalksToTheMiddleThenTheTracksSlideSmoothly() {
        // 10 s visible, a 60 s cut.
        XCTAssertEqual(TimelineViewport.follow(viewStart: 0, now: 4, visible: 10, duration: 60), 0)   // line walks
        XCTAssertEqual(TimelineViewport.follow(viewStart: 0, now: 5.1, visible: 10, duration: 60), 0.1, accuracy: 0.001)
        // Frame after frame (1/30 s) the tracks move 1/30 s: no jumps ("de bloque").
        let a = TimelineViewport.follow(viewStart: 10, now: 15.0, visible: 10, duration: 60)
        let b = TimelineViewport.follow(viewStart: a, now: 15.0 + 1.0 / 30, visible: 10, duration: 60)
        XCTAssertEqual(b - a, 1.0 / 30, accuracy: 0.0001)
    }

    func testAtTheEndTheLineWalksOnToTheEndOfTheCut() {
        // maxStart = 60 - 8.5 = 51.5: the tracks stop, the line goes on to 60.
        XCTAssertEqual(TimelineViewport.follow(viewStart: 51.5, now: 59, visible: 10, duration: 60), 51.5, accuracy: 0.001)
    }

    func testAJumpBackBringsTheLineIntoView() {
        XCTAssertEqual(TimelineViewport.follow(viewStart: 30, now: 2, visible: 10, duration: 60), 0.5, accuracy: 0.001)
        XCTAssertEqual(TimelineViewport.follow(viewStart: 30, now: 0, visible: 10, duration: 60), 0)
    }

    func testTheTracksNeverSlidePastTheEnd() {
        XCTAssertEqual(TimelineViewport.clamp(80, duration: 60, visible: 10), 51.5, accuracy: 0.001)
        XCTAssertEqual(TimelineViewport.clamp(-3, duration: 60, visible: 10), 0)
    }

    func testDraggingTheLineToASideSlidesTheTracksAlong() {
        // 300 pt of tracks, 10 s visible, one frame (1/30 s).
        XCTAssertEqual(TimelineViewport.edgeSlide(fingerX: 150, tracksWidth: 300, visible: 10, dt: 1.0 / 30), 0)
        XCTAssertGreaterThan(TimelineViewport.edgeSlide(fingerX: 290, tracksWidth: 300, visible: 10, dt: 1.0 / 30), 0)
        XCTAssertLessThan(TimelineViewport.edgeSlide(fingerX: 10, tracksWidth: 300, visible: 10, dt: 1.0 / 30), 0)
        // Deeper into the side = faster.
        XCTAssertGreaterThan(TimelineViewport.edgeSlide(fingerX: 299, tracksWidth: 300, visible: 10, dt: 1.0 / 30),
                             TimelineViewport.edgeSlide(fingerX: 270, tracksWidth: 300, visible: 10, dt: 1.0 / 30))
    }

    func testAFingerOnTheLineOrTheRulerGrabsTheLine() {
        XCTAssertTrue(TimelineViewport.grabsLine(startX: 210, lineX: 200, onRuler: false))
        XCTAssertFalse(TimelineViewport.grabsLine(startX: 260, lineX: 200, onRuler: false))   // slides the tracks
        XCTAssertTrue(TimelineViewport.grabsLine(startX: 360, lineX: 200, onRuler: true))
    }

    func testTheLineGoesToTheTimeUnderTheFinger() {
        XCTAssertEqual(TimelineViewport.time(atX: 60, viewStart: 4, pixelsPerSecond: 30, duration: 60), 6, accuracy: 0.001)
        XCTAssertEqual(TimelineViewport.time(atX: 9000, viewStart: 4, pixelsPerSecond: 30, duration: 60), 60)
        XCTAssertEqual(TimelineViewport.time(atX: -50, viewStart: 0, pixelsPerSecond: 30, duration: 60), 0)
    }

    func testZoomKeepsTheLineInPlace() {
        // Line 90 pt from the left at 30 pt/s (3 s); zoomed to 60 pt/s it stays 90 pt from the left: 1.5 s.
        XCTAssertEqual(TimelineViewport.startKeepingLine(now: 13, lineOffset: 90, pixelsPerSecond: 60,
                                                         duration: 60, visible: 5), 11.5, accuracy: 0.001)
    }
}
