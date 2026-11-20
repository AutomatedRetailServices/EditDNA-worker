import XCTest
import AVFoundation
@testable import CutSell

/// PO 20 Nov 2026: Mirror is ON by default (the product-reading selfie view); OFF = like the iPhone camera.
@MainActor
final class CameraMirrorTests: XCTestCase {
    override func setUp() { UserDefaults.standard.removeObject(forKey: CameraController.mirrorKey) }
    override func tearDown() { UserDefaults.standard.removeObject(forKey: CameraController.mirrorKey) }

    func testMirrorStartsOnAndThePreviewReadsRight() {
        let camera = CameraController()
        XCTAssertTrue(camera.mirrorOn, "Mirror is lit by default")
        XCTAssertFalse(camera.previewMirrored(position: .front), "on: the selfie preview is the real image (text reads right)")
        camera.mirrorOn = false
        XCTAssertTrue(camera.previewMirrored(position: .front), "off: a mirror, like the iPhone camera")
        XCTAssertFalse(camera.previewMirrored(position: .back), "the back camera is never a mirror")
        XCTAssertFalse(CameraController().mirrorOn, "the last choice is remembered")
    }
}
