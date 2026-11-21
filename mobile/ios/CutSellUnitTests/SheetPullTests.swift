import XCTest
@testable import CutSell

/// PO 21 Nov 2026: pulling a sheet down by its top closes it.
final class SheetPullTests: XCTestCase {
    func testAFullPullCloses() { XCTAssertTrue(SheetPull.closes(translation: 130, predicted: 140)) }
    func testAQuickFlickCloses() { XCTAssertTrue(SheetPull.closes(translation: 40, predicted: 300)) }
    func testASmallPullSpringsBack() { XCTAssertFalse(SheetPull.closes(translation: 50, predicted: 90)) }
}
