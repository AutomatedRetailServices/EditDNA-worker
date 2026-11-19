import XCTest
import AVFoundation
@testable import CutSell

/// PO 19 Nov 2026: clips from a video filmed the other way showed sideways in the editor preview.
final class PreviewOrientationTests: XCTestCase {
    /// An iPhone portrait video stored landscape: 1920 × 1080 turned 90°.
    private let portraitStored = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 1080, ty: 0)

    func testATurnedSourceFillsTheUprightFrame() {
        let render = CGSize(width: 1080, height: 1920)
        let segment = PreviewOrientation.Segment(start: .zero, transform: portraitStored, naturalSize: CGSize(width: 1920, height: 1080))
        let out = CGRect(origin: .zero, size: segment.naturalSize).applying(PreviewOrientation.fitted(segment, render: render))
        XCTAssertEqual(out.minX, 0, accuracy: 0.5)
        XCTAssertEqual(out.minY, 0, accuracy: 0.5)
        XCTAssertEqual(out.width, 1080, accuracy: 0.5)
        XCTAssertEqual(out.height, 1920, accuracy: 0.5)
    }

    func testAnUprightSourceBesideATurnedOneIsNotTurned() {
        let render = CGSize(width: 1080, height: 1920)
        let upright = PreviewOrientation.Segment(start: .zero, transform: .identity, naturalSize: CGSize(width: 1080, height: 1920))
        let t = PreviewOrientation.fitted(upright, render: render)
        XCTAssertEqual(t, .identity)
    }

    func testOneOrientationNeedsNoVideoComposition() throws {
        let composition = AVMutableComposition()
        let track = try XCTUnwrap(composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid))
        let a = PreviewOrientation.Segment(start: .zero, transform: portraitStored, naturalSize: CGSize(width: 1920, height: 1080))
        let b = PreviewOrientation.Segment(start: CMTime(seconds: 3, preferredTimescale: 600), transform: portraitStored,
                                           naturalSize: CGSize(width: 1920, height: 1080))
        XCTAssertNil(PreviewOrientation.videoComposition(for: [a, b], track: track, duration: CMTime(seconds: 6, preferredTimescale: 600)))
        let c = PreviewOrientation.Segment(start: CMTime(seconds: 6, preferredTimescale: 600), transform: .identity,
                                           naturalSize: CGSize(width: 1080, height: 1920))
        let mixed = try XCTUnwrap(PreviewOrientation.videoComposition(for: [a, b, c], track: track,
                                                                       duration: CMTime(seconds: 9, preferredTimescale: 600)))
        XCTAssertEqual(mixed.renderSize, CGSize(width: 1080, height: 1920))
    }
}
