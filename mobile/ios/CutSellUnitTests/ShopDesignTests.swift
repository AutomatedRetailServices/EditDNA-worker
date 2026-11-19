import XCTest
@testable import CutSell

/// Title & design + Background › Cut (PO 18 Nov 2026, Figma T1 – T5).
@MainActor
final class ShopDesignTests: XCTestCase {
    private func pixel(_ image: UIImage, _ x: Int, _ y: Int) -> (r: Int, g: Int, b: Int) {
        let cg = image.cgImage!
        var data = [UInt8](repeating: 0, count: 4)
        let ctx = CGContext(data: &data, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: -x, y: -(cg.height - 1 - y), width: cg.width, height: cg.height))
        return (Int(data[0]), Int(data[1]), Int(data[2]))
    }

    func testEveryLetterIsBundledAndRegistered() {
        ShopFonts.registerIfNeeded()
        XCTAssertEqual(ShopLetter.allCases.count, 19)
        for face in ShopLetter.allCases.map(\.face) + [ShopLetter.dmSerifItalic] {
            XCTAssertNotNil(Bundle.main.url(forResource: face.fileName, withExtension: "ttf"), face.fileName)
            XCTAssertNotNil(UIFont(name: face.postScriptName, size: 20), "\(face.postScriptName) is registered")
            XCTAssertNotEqual(ShopFonts.uiFont(face, size: 20).familyName, UIFont.systemFont(ofSize: 20).familyName, face.postScriptName)
        }
        XCTAssertEqual(ShopFontCredit.all.count, 18)
        // Figma T2: the six new faces + Playfair's italic carry a badge.
        XCTAssertEqual(ShopLetter.allCases.filter { $0.badge != nil }.count, 7)
    }

    func testEveryLayoutDrawsTheTitle() throws {
        let plain = try XCTUnwrap(ShopBoardExport.image(ShopBoardTestData.snapshot()))
        for layout in ShopTitleLayout.allCases {
            var snapshot = ShopBoardTestData.snapshot()
            snapshot.bigTitle = "THE FALL EDIT"
            snapshot.design.layout = layout
            let image = try XCTUnwrap(ShopBoardExport.image(snapshot), layout.rawValue)
            XCTAssertEqual(image.cgImage?.width, 1080, layout.rawValue)
            XCTAssertNotEqual(image.pngData(), plain.pngData(), "\(layout.rawValue) looks different from the first look")
        }
    }

    func testTallSplitsInTwoEvenLines() {
        XCTAssertEqual(ShopTextMeasure.twoLines("DUPES THAT DON'T LOOK").0, "DUPES THAT")
        XCTAssertEqual(ShopTextMeasure.twoLines("DUPES THAT DON'T LOOK").1, "DON'T LOOK")
        XCTAssertEqual(ShopTextMeasure.twoLines("AMAZON").1, "")
    }

    func testDoubleFrameIsDrawnInItsColor() throws {
        var snapshot = ShopBoardTestData.snapshot()
        snapshot.top = ShopRGB(hex: 0x8B7D6E); snapshot.bottom = ShopRGB(hex: 0x8B7D6E)
        snapshot.design.cut = .doubleFrame
        snapshot.design.cutColor = ShopRGB(hex: 0xFFFFFF)
        let image = try XCTUnwrap(ShopBoardExport.image(snapshot))
        // The outer line: 14 design units in (×1080/390 ≈ 39 px), 3.4 thick, on the left side above the products.
        let p = pixel(image, 42, 200)
        XCTAssertGreaterThan(p.r, 230)
        XCTAssertGreaterThan(p.g, 230)
        let gap = pixel(image, 52, 200)                 // between the two lines: the background
        XCTAssertLessThan(gap.r, 160)
    }

    func testEveryCutAndEveryAddedThingRenders() throws {
        for cut in ShopCut.allCases {
            var snapshot = ShopBoardTestData.snapshot()
            snapshot.design.cut = cut
            XCTAssertEqual(try XCTUnwrap(ShopBoardExport.image(snapshot), cut.rawValue).cgImage?.width, 1080)
        }
        var snapshot = ShopBoardTestData.snapshot()
        snapshot.design.decor = ShopDecor.Kind.allCases.map { ShopDecor.make($0, color: ShopRGB(hex: 0x6CC7D1)) }
        snapshot.design.decorAdjusts = snapshot.design.decor.map { _ in ShopAdjust() }
        let image = try XCTUnwrap(ShopBoardExport.image(snapshot))
        XCTAssertEqual(image.cgImage?.height, 1920)
    }

    func testAddDuplicateFrontBackDelete() {
        let model = ShopBoardModel()
        let tag = model.addDecor(.tag)
        XCTAssertEqual(model.selectedDecor, tag)
        XCTAssertTrue(model.selectedDecorHasWords)
        let shape = model.addDecor(.shape)
        XCTAssertFalse(model.decor.last!.front, "a shape starts behind the products")
        let second = model.addDecor(.shape)
        XCTAssertTrue(model.decor.first { $0.id == second }!.circle, "a second shape in a row is a circle")
        model.duplicateDecor(tag)
        XCTAssertEqual(model.decor.count, 4)
        XCTAssertEqual(model.decor.last?.text, "Comment YES")
        model.bringToFront(shape)
        XCTAssertEqual(model.decor.last?.id, shape)
        XCTAssertTrue(model.decor.last!.front)
        model.sendToBack(tag)
        XCTAssertEqual(model.decor.first?.id, tag)
        XCTAssertFalse(model.decor.first!.front)
        model.setText("Reply YES", for: tag)
        XCTAssertEqual(model.decor.first?.text, "Reply YES")
        model.deleteDecor(tag)
        XCTAssertFalse(model.decor.contains { $0.id == tag })
        // Shuffle starts the products over but keeps what she added where she put it.
        model.adjusts[shape] = ShopAdjust(offset: CGSize(width: 30, height: 10))
        model.shuffle()
        XCTAssertEqual(model.adjusts[shape]?.offset.width, 30)
    }

    func testProductsCanBeSentBack() {
        let ids = [UUID(), UUID(), UUID()]
        XCTAssertEqual(ShopDrawOrder.order(itemIDs: ids, raised: [], lowered: [ids[2]]), [2, 0, 1])
        XCTAssertEqual(ShopDrawOrder.order(itemIDs: ids, raised: [ids[0]], lowered: [ids[1]]), [1, 2, 0])
    }

    func testPastTheEdgeAllowsBiggerPieces() {
        let base = CGSize(width: 120, height: 160)
        XCTAssertEqual(ShopAdjust.scaleLimits(base: base).upperBound, 390 * 1.1 / 120, accuracy: 0.001)
        XCTAssertEqual(ShopAdjust.scaleLimits(base: base, pastEdge: true).upperBound, 390 * 2.2 / 120, accuracy: 0.001)
    }

    func testGrabOrderAndFrameIsGrabbedByItsLines() {
        var snapshot = ShopBoardTestData.snapshot()
        let size = CGSize(width: 390, height: 693)
        var frame = ShopDecor.make(.frame, color: ShopRGB(hex: 0xFFFFFF))
        frame.front = true
        var tag = ShopDecor.make(.tag, color: ShopRGB(hex: 0x6CC7D1))
        tag.y = 60                                                   // over the title
        snapshot.design.decor = [frame, tag]
        snapshot.design.decorAdjusts = [ShopAdjust(), ShopAdjust()]
        let titleID = UUID()
        snapshot.itemIDs = snapshot.placements.map { _ in UUID() }
        XCTAssertEqual(snapshot.grab(at: CGPoint(x: 195, y: 60 * 693 / 844), board: size, titleID: titleID), tag.id)
        let edge = snapshot.decorFrame(frame.id, board: size)!.rect
        XCTAssertEqual(snapshot.grab(at: CGPoint(x: edge.minX + 4, y: edge.midY), board: size, titleID: titleID), frame.id)
        XCTAssertNotEqual(snapshot.grab(at: CGPoint(x: edge.midX, y: edge.midY), board: size, titleID: titleID), frame.id,
                          "inside the frame a finger gets the product, not the frame")
    }

    func testSaveAndOpenKeepsTheDesign() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("ShopDesignTest-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ShopBoardStore(root: root)
        let model = ShopBoardModel()
        let r = UIGraphicsImageRenderer(size: CGSize(width: 40, height: 60))
        model.items = [ShopItem(source: .photos, cutout: r.image { c in UIColor.red.setFill(); c.fill(CGRect(x: 0, y: 0, width: 40, height: 60)) }.cgImage!)]
        model.titleLayout = .scriptInside
        model.bigLetter = .instrument
        model.titleSize = 1.2
        model.titleSpacing = 3
        let label = model.addDecor(.label)
        model.adjusts[label] = ShopAdjust(offset: CGSize(width: 10, height: 20), scale: 1.4, rotation: 8)
        model.cut = .doubleFrame
        model.cutColor = ShopRGB(hex: 0xB8955A)
        model.pastEdge = false
        model.sendToBack(model.items[0].id)
        store.save(model, thumbnail: nil, synchronously: true)
        let opened = try XCTUnwrap(ShopBoardStore(root: root).open(model.boardID))
        XCTAssertEqual(opened.titleLayout, .scriptInside)
        XCTAssertEqual(opened.bigLetter, .instrument)
        XCTAssertEqual(opened.titleSize, 1.2)
        XCTAssertEqual(opened.titleSpacing, 3)
        XCTAssertEqual(opened.decor, model.decor)
        XCTAssertEqual(opened.adjusts[label], model.adjusts[label])
        XCTAssertEqual(opened.cut, .doubleFrame)
        XCTAssertEqual(opened.cutColor, ShopRGB(hex: 0xB8955A))
        XCTAssertFalse(opened.pastEdge)
        XCTAssertEqual(opened.lowered, [model.items[0].id])
    }

    func testEveryNewWordHasSpanish() {
        let words = ShopTitleLayout.allCases.map(\.name) + ShopCut.allCases.map(\.name) + ShopDecor.Kind.allCases.map(\.name)
            + ["Title & design", "Words", "Layout", "Letters", "Add", "Cut", "Frame color", "Letter space", "Change letters of",
               "Selected text", "Italic now", "Edit words", "Products can go past the edge", "Big pieces cut by the board, like the pros",
               "Hold anything: Front · Back · Duplicate · Delete", "Add to your board · drag, pinch and turn like a product",
               "Cut the background · boxes inside it", "Title layout · like the boards you love"]
        for w in words { XCTAssertNotNil(L10nES.table[w], "missing Spanish for \(w)") }
    }
}
