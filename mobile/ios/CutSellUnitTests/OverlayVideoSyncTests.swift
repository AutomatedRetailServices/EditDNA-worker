import XCTest
@testable import CutSell

/// PO 20 Nov 2026: B-roll over the voice plays as real video in the editor, following the play head.
final class OverlayVideoSyncTests: XCTestCase {
    func testPlayStartsTheOverlayAtThePlayHead() {
        XCTAssertEqual(OverlayVideoSync.step(playerTime: 0, wanted: 2.4, editorPlaying: true, playerRunning: false),
                       .seekAndPlay(2.4))
    }

    func testWhilePlayingSmallDriftIsLeftAloneAndBigDriftIsFixed() {
        XCTAssertEqual(OverlayVideoSync.step(playerTime: 2.5, wanted: 2.6, editorPlaying: true, playerRunning: true), .none)
        XCTAssertEqual(OverlayVideoSync.step(playerTime: 1.0, wanted: 2.6, editorPlaying: true, playerRunning: true),
                       .seek(2.6))
    }

    func testPauseStopsItOnTheExactFrame() {
        XCTAssertEqual(OverlayVideoSync.step(playerTime: 3.0, wanted: 3.0, editorPlaying: false, playerRunning: true),
                       .pauseAndSeek(3.0))
        XCTAssertEqual(OverlayVideoSync.step(playerTime: 3.0, wanted: 3.5, editorPlaying: false, playerRunning: false),
                       .seek(3.5))
        XCTAssertEqual(OverlayVideoSync.step(playerTime: 3.5, wanted: 3.51, editorPlaying: false, playerRunning: false), .none)
    }

    func testANotYetLoadedVideoIsMovedToThePlayHead() {
        XCTAssertEqual(OverlayVideoSync.step(playerTime: .nan, wanted: 1, editorPlaying: false, playerRunning: false),
                       .seek(1))
    }
}
