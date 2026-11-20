import XCTest
@testable import CutSell

/// PO 20 Nov 2026: a clip recorded with the CutSell camera is saved to Photos and remembered as the cut's original,
/// so a missing light copy can be restored ("Restoring video…") instead of "isn't in Photos anymore".
final class CameraOriginalsTests: XCTestCase {
    func testACameraClipStaysLinkedToItsPhotosCopyThroughTheLightCopyAndTheUpload() {
        let recorded = FileManager.default.temporaryDirectory.appendingPathComponent("cam-\(UUID().uuidString).mov")
        let lighter = FileManager.default.temporaryDirectory.appendingPathComponent("light-\(UUID().uuidString).mp4")
        let uri = "s3://bucket/uploads/\(UUID().uuidString).mp4"
        OriginalLinks.note(file: recorded, photosID: "PHOTOS-ID-1/L0/001")   // what CameraOriginals.keep records
        OriginalLinks.carry(from: recorded, to: lighter)                       // VideoPreparation
        OriginalLinks.link(uri: uri, file: lighter)                            // upload done
        XCTAssertEqual(OriginalLinks.photosID(forURI: uri), "PHOTOS-ID-1/L0/001")
    }

    func testNoPhotosCopyMeansNoLink() {
        let recorded = FileManager.default.temporaryDirectory.appendingPathComponent("cam-\(UUID().uuidString).mov")
        let uri = "s3://bucket/uploads/\(UUID().uuidString).mp4"
        OriginalLinks.note(file: recorded, photosID: nil)                      // permission refused
        OriginalLinks.link(uri: uri, file: recorded)
        XCTAssertNil(OriginalLinks.photosID(forURI: uri))
    }
}
