import XCTest
@testable import CutSell

/// PO 21 Nov 2026 (Figma 3094:357, option A): one list, the last used products first, then A→Z.
final class PromotingOrderTests: XCTestCase {
    private let rows: [(id: String, brand: String, product: String)] = [
        ("p1", "CutSell", "CutSell"), ("p2", "Bloom", "Bloom"), ("p3", "CeraVe", "Moisturizing Cream"),
        ("p4", "CeraVe", "Hydrating Cleanser"),
    ]

    func testTheLastUsedComeFirstInTheOrderTheyWereUsed() {
        let ids = PromotingOrder.sorted(rows, recent: ["p3", "p1"]).map(\.id)
        XCTAssertEqual(ids, ["p3", "p1", "p2", "p4"])
    }

    func testWithoutRecentItIsByBrandThenProduct() {
        let ids = PromotingOrder.sorted(rows, recent: []).map(\.id)
        XCTAssertEqual(ids, ["p2", "p4", "p3", "p1"])
    }

    func testAnOldRecentIdThatIsGoneChangesNothing() {
        let ids = PromotingOrder.sorted(rows, recent: ["gone", "p4"]).map(\.id)
        XCTAssertEqual(ids.first, "p4")
    }
}
