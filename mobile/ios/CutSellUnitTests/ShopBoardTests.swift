import CoreImage
import SwiftUI
import UIKit
import Vision
import XCTest
@testable import CutSell

/// Shop board (PO 11 Nov 2026). Every picture here is made in memory by the test; nothing is stored in the repo.

// MARK: - Synthetic pictures

enum ShopSynthetic {
    /// A `size` × `size` picture: `background` everywhere, a `subject` disc in the middle (radius = size × 0.3).
    static func disc(size: Int = 256, subject: UIColor = UIColor(red: 0.85, green: 0.1, blue: 0.15, alpha: 1),
                     background: UIColor = .white) -> CGImage {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        format.opaque = true
        let s = CGFloat(size)
        return UIGraphicsImageRenderer(size: CGSize(width: s, height: s), format: format).image { ctx in
            background.setFill()
            ctx.fill(CGRect(x: 0, y: 0, width: s, height: s))
            subject.setFill()
            ctx.cgContext.fillEllipse(in: CGRect(x: s * 0.2, y: s * 0.2, width: s * 0.6, height: s * 0.6))
        }.cgImage!
    }

    /// A grey mask: white square in the middle half, black elsewhere.
    static func squareMask(size: Int = 256) -> CGImage {
        let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: size,
                            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        ctx.setFillColor(gray: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        ctx.setFillColor(gray: 1, alpha: 1)
        ctx.fill(CGRect(x: size / 4, y: size / 4, width: size / 2, height: size / 2))
        return ctx.makeImage()!
    }

    /// A transparent cut-out of one colour (an item), `w` × `h`.
    static func cutout(_ color: UIColor, w: Int = 120, h: Int = 160) -> CGImage {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        format.opaque = false
        return UIGraphicsImageRenderer(size: CGSize(width: w, height: h), format: format).image { ctx in
            color.setFill()
            ctx.cgContext.fill(CGRect(x: CGFloat(w) * 0.1, y: CGFloat(h) * 0.05, width: CGFloat(w) * 0.8, height: CGFloat(h) * 0.9))
        }.cgImage!
    }

    static func alpha(_ image: CGImage, x: Int, y: Int) -> UInt8 {
        let (bytes, w, _) = ShopCutout.rgba(image)!
        return bytes[(y * w + x) * 4 + 3]
    }
}

// MARK: - a) Background removal

final class ShopBackgroundRemovalTests: XCTestCase {
    /// The mask code alone (always runs): white keeps the pixel, black makes it transparent.
    func testMaskRemovesTheBackground() throws {
        let photo = ShopSynthetic.disc()
        let mask = CIImage(cgImage: ShopSynthetic.squareMask())
        let out = try XCTUnwrap(ShopCutout.apply(mask: mask, to: photo))
        XCTAssertEqual(out.width, 256)
        XCTAssertEqual(ShopSynthetic.alpha(out, x: 2, y: 2), 0, "corner (background) is transparent")
        XCTAssertEqual(ShopSynthetic.alpha(out, x: 250, y: 128), 0, "outside the mask is transparent")
        XCTAssertGreaterThan(ShopSynthetic.alpha(out, x: 128, y: 128), 250, "the subject stays opaque")
        // Cropped to the subject: the kept square (+ a 4 px margin).
        let trimmed = try XCTUnwrap(ShopCutout.trimmed(out))
        XCTAssertEqual(trimmed.width, 128 + 8, accuracy: 2)
        XCTAssertEqual(trimmed.height, 128 + 8, accuracy: 2)
    }

    /// A smaller mask (Vision's masks can come at another size) is stretched to the photo.
    func testMaskOfAnotherSizeIsScaled() throws {
        let photo = ShopSynthetic.disc(size: 256)
        let out = try XCTUnwrap(ShopCutout.apply(mask: CIImage(cgImage: ShopSynthetic.squareMask(size: 64)), to: photo))
        XCTAssertEqual(ShopSynthetic.alpha(out, x: 10, y: 10), 0)
        XCTAssertGreaterThan(ShopSynthetic.alpha(out, x: 128, y: 128), 250)
    }

    /// Vision itself. Its foreground model needs the Neural Engine / a GPU context that the CI simulator often does not
    /// provide: then the request throws and the test is skipped (the mask code is covered above).
    func testVisionForegroundMask() throws {
        let photo = ShopSynthetic.disc(size: 512)
        let cut: CGImage
        do {
            cut = try ShopCutout.removeBackground(photo)
        } catch {
            throw XCTSkip("VNGenerateForegroundInstanceMaskRequest is not available here (\(error)); mask code tested separately.")
        }
        // The disc's bounding box corners are background: transparent after the cut.
        XCTAssertEqual(Int(ShopSynthetic.alpha(cut, x: 1, y: 1)), 0, accuracy: 30)
        XCTAssertGreaterThan(ShopSynthetic.alpha(cut, x: cut.width / 2, y: cut.height / 2), 200)
        XCTAssertLessThan(cut.width, 512, "cropped to the subject")
    }
}

// MARK: - b) Paste a link never downloads anything

private final class NetworkSpy: ShopNetworkLoading {
    var calls = 0
    func data(from url: URL) async throws -> Data { calls += 1; return Data() }
}

/// Counts every request made through the URL loading system while installed.
final class ShopRequestCounter: URLProtocol {
    nonisolated(unsafe) static var count = 0
    override class func canInit(with request: URLRequest) -> Bool { count += 1; return false }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
}

final class ShopPasteLinkTests: XCTestCase {
    func testLinkIsOnlyStoredNeverDownloaded() {
        let spy = NetworkSpy()
        URLProtocol.registerClass(ShopRequestCounter.self)
        defer { URLProtocol.unregisterClass(ShopRequestCounter.self) }
        ShopRequestCounter.count = 0
        let handler = ShopLinkHandler(network: spy)
        let item = handler.item(fromPasted: "  https://www.amazon.com/dp/B0TEST1234?tag=me-20  ")
        XCTAssertEqual(item?.affiliateLink?.absoluteString, "https://www.amazon.com/dp/B0TEST1234?tag=me-20")
        XCTAssertEqual(item?.source, .link)
        XCTAssertNil(item?.cutout, "no picture from the link")
        XCTAssertTrue(item?.needsPhoto == true, "the photo comes from Photos")
        XCTAssertEqual(item?.name, "Amazon")
        XCTAssertEqual(spy.calls, 0, "the handler never loads anything")
        XCTAssertEqual(ShopRequestCounter.count, 0, "no request reached the URL loading system")
    }

    func testOtherStoresAndBareLinks() {
        let handler = ShopLinkHandler()
        XCTAssertEqual(handler.item(fromPasted: "walmart.com/ip/123")?.affiliateLink?.absoluteString, "https://walmart.com/ip/123")
        XCTAssertEqual(handler.item(fromPasted: "https://www.walmart.com/ip/123")?.name, "Walmart")
        XCTAssertEqual(handler.item(fromPasted: "Look at this https://shop.example.co/x")?.affiliateLink?.host, "shop.example.co")
        XCTAssertNil(handler.item(fromPasted: "not a link"))
        XCTAssertNil(handler.item(fromPasted: ""))
    }

    @MainActor
    func testModelKeepsTheLinkAsAffiliateLink() {
        let model = ShopBoardModel()
        let id = model.addLink("https://amzn.to/3abc")
        XCTAssertNotNil(id)
        XCTAssertEqual(model.items.first?.affiliateLink?.absoluteString, "https://amzn.to/3abc")
        XCTAssertEqual(model.affiliateLink, "https://amzn.to/3abc")
        XCTAssertFalse(ShopBoardRules.canChooseStyle(model.items), "it still needs its photo")
    }
}

// MARK: - c) Shape it only for clothing and shoes

final class ShopShapeItTests: XCTestCase {
    private func kind(_ labels: [(String, Float)]) -> ShopItemKind {
        ShopItemClassifier.kind(fromLabels: labels.map { (identifier: $0.0, confidence: $0.1) })
    }

    func testClothingAndShoesCanBeShaped() {
        XCTAssertEqual(kind([("shirt", 0.8), ("clothing", 0.9)]), .top)
        XCTAssertEqual(kind([("jeans", 0.7), ("clothing", 0.8)]), .bottom)
        XCTAssertEqual(kind([("dress", 0.6)]), .clothing)
        XCTAssertEqual(kind([("sneaker", 0.85), ("footwear", 0.8), ("clothing", 0.4)]), .shoes)
        XCTAssertEqual(kind([("Running Shoe", 0.6)]), .shoes)
        for k in [ShopItemKind.top, .bottom, .clothing, .shoes] { XCTAssertTrue(k.canShape, "\(k)") }
        XCTAssertEqual(ShopItemKind.top.shapeKind, "clothing")
        XCTAssertEqual(ShopItemKind.shoes.shapeKind, "shoes")
    }

    func testBagsAccessoriesAndUnsureAreNot() {
        XCTAssertEqual(kind([("handbag", 0.8), ("bag", 0.7)]), .bag)
        XCTAssertEqual(kind([("jewelry", 0.7), ("earring", 0.6)]), .accessory)
        XCTAssertEqual(kind([("sunglasses", 0.9)]), .accessory)
        XCTAssertEqual(kind([("handbag", 0.55), ("clothing", 0.6)]), .unknown, "too close: unsure")
        XCTAssertEqual(kind([("shirt", 0.2)]), .unknown, "too weak")
        XCTAssertEqual(kind([("cat", 0.9), ("animal", 0.9)]), .unknown)
        XCTAssertEqual(kind([]), .unknown)
        for k in [ShopItemKind.bag, .accessory, .unknown] {
            XCTAssertFalse(k.canShape, "\(k)")
            XCTAssertNil(k.shapeKind)
        }
    }

    func testServerFailuresHaveTheirOwnMessage() {
        XCTAssertEqual(ShopShapeFailure.from(status: 429, body: #"{"error": "shape_daily_limit", "limit": 20}"#), .dailyLimit(20))
        XCTAssertEqual(ShopShapeFailure.from(status: 503, body: #"{"error": "shape_disabled"}"#), .disabled)
        XCTAssertEqual(ShopShapeFailure.from(status: 401, body: ""), .signedOut)
        XCTAssertEqual(ShopShapeFailure.from(status: 413, body: ""), .tooLarge)
        XCTAssertEqual(ShopShapeFailure.from(status: 404, body: "Not Found"), .failed, "not deployed yet: Retry")
        let messages = Set([ShopShapeFailure.dailyLimit(20), .disabled, .signedOut, .tooLarge, .failed].map(\.message))
        XCTAssertEqual(messages.count, 5)
        XCTAssertTrue(ShopShapeFailure.dailyLimit(20).message.contains("20"))
    }

    func testShapePNGKeepsAlphaAndLongSide() throws {
        let big = ShopSynthetic.cutout(.blue, w: 2000, h: 1000)
        let png = try XCTUnwrap(ShopCutout.shapePNG(big))
        let image = try XCTUnwrap(UIImage(data: png)?.cgImage)
        XCTAssertEqual(max(image.width, image.height), 1536)
        XCTAssertNotEqual(image.alphaInfo, .none)
        XCTAssertEqual(ShopSynthetic.alpha(image, x: 0, y: 0), 0)
    }
}

// MARK: - d) Match my products / e) picking by hand turns Match off

final class ShopMatchTests: XCTestCase {
    private let navy = ShopRGB(hex: 0x1E2A4A), grey = ShopRGB(hex: 0x9AA0A8)

    func testNavyAndGreyGiveASoftNeutralBackgroundAndATitleFromThePalette() {
        let pixels = [ShopCutout.opaquePixels(ShopSynthetic.cutout(navy.uiColor)),
                      ShopCutout.opaquePixels(ShopSynthetic.cutout(grey.uiColor))]
        XCTAssertFalse(pixels[0].isEmpty)
        let palette = ShopPalette.palette(perItem: pixels)
        XCTAssertTrue(palette.contains { $0.distance2(navy) < 0.002 }, "navy found: \(palette.map(\.hex))")
        XCTAssertTrue(palette.contains { $0.distance2(grey) < 0.002 }, "grey found")

        let bg = ShopPalette.softBackground(for: palette)
        XCTAssertGreaterThan(bg.top.hsl.l, 0.85, "soft = light")
        XCTAssertLessThan(bg.top.hsl.s, 0.2, "neutral = little colour")
        XCTAssertGreaterThan(bg.bottom.hsl.l, 0.8)

        let middle = bg.top.mixed(bg.bottom, 0.5)
        let title = ShopPalette.titleColor(for: palette, on: middle)
        XCTAssertTrue(palette.contains(title), "the title colour comes from the products")
        XCTAssertEqual(title, palette.first { $0.distance2(navy) < 0.002 }, "navy reads best")
        XCTAssertGreaterThanOrEqual(ShopPalette.contrast(title, middle), 4.5)
    }

    func testTransparentPixelsAreIgnored() {
        let cut = ShopSynthetic.cutout(navy.uiColor)          // transparent margins around a navy block
        let pixels = ShopCutout.opaquePixels(cut)
        XCTAssertTrue(pixels.allSatisfy { $0.distance2(navy) < 0.002 }, "no black from the transparent margins")
    }

    func testPaleProductsStillGetAReadableTitle() {
        let palette = [ShopRGB(hex: 0xF4F4F2), ShopRGB(hex: 0xE8E0D0)]
        let bg = ShopPalette.softBackground(for: palette)
        let title = ShopPalette.titleColor(for: palette, on: bg.top)
        XCTAssertGreaterThanOrEqual(ShopPalette.contrast(title, bg.top), 4.5)
    }

    func testKMeansSplitsTwoColours() {
        let pixels = Array(repeating: navy, count: 300) + Array(repeating: grey, count: 100)
        let clusters = ShopPalette.kMeans(pixels, k: 2)
        XCTAssertEqual(clusters.count, 2)
        XCTAssertEqual(clusters[0].color, navy)
        XCTAssertEqual(clusters[0].weight, 0.75, accuracy: 0.001)
    }

    func testMatchIsOnByDefaultAndPickingByHandTurnsItOff() {
        var style = ShopBoardStyle()
        XCTAssertTrue(style.matchBackground, "✨ Match my products: DEFAULT · ON")
        XCTAssertTrue(style.matchTitle)
        style.pick(.color(navy))
        XCTAssertFalse(style.matchBackground)
        XCTAssertEqual(style.backgroundColors(palette: [navy, grey]).top, navy)
        style.turnOnMatch()
        XCTAssertTrue(style.matchBackground)
        style.pick(.classic(.sage))
        XCTAssertFalse(style.matchBackground)
        XCTAssertEqual(style.backgroundColors(palette: []).top, ShopRGB(hex: 0xE6ECE2))
        style.pick(.texture(.linen))
        XCTAssertFalse(style.matchBackground)
        style.pickTitleColor(grey)
        XCTAssertFalse(style.matchTitle)
        XCTAssertEqual(style.resolvedTitleColor(palette: [navy, grey], template: .everydayChic), grey)
    }

    func testColourNamesFromFigma() {
        XCTAssertEqual(ShopPalette.name(of: ShopRGB(hex: 0x1E2A4A)), "Navy")
        XCTAssertEqual(ShopPalette.name(of: ShopRGB(hex: 0x7DA4E6)), "Sky")
        XCTAssertEqual(ShopPalette.name(of: ShopRGB(hex: 0x9AA0A8)), "Gray")
        XCTAssertEqual(ShopPalette.name(of: ShopRGB(hex: 0x6E4E36)), "Brown")
    }
}

// MARK: - f) Letters

@MainActor
final class ShopFontTests: XCTestCase {
    func testAllTwelveFacesAreRegistered() {
        ShopFonts.registerIfNeeded()
        var names = Set<String>()
        for pair in ShopFontPair.allCases {
            for face in [pair.titleFace, pair.scriptFace] {
                XCTAssertNotNil(Bundle.main.url(forResource: face.fileName, withExtension: "ttf"), "\(face.fileName) is in the app")
                XCTAssertNotNil(UIFont(name: face.postScriptName, size: 20), "\(face.postScriptName) is registered")
                names.insert(face.postScriptName)
            }
        }
        XCTAssertEqual(names.count, 12)
        XCTAssertNotNil(UIFont(name: ShopFontPair.footerFace.postScriptName, size: 14))
        // The faces really are ours (not the system fallback).
        for face in ShopFonts.allFaces {
            XCTAssertEqual(ShopFonts.uiFont(face, size: 20).fontName.isEmpty, false)
            XCTAssertNotEqual(ShopFonts.uiFont(face, size: 20).familyName, UIFont.systemFont(ofSize: 20).familyName)
        }
    }

    func testEveryLicenceIsBundled() {
        for credit in ShopFontCredit.all {
            let text = ShopFontCreditsView.licenseText(credit)
            XCTAssertTrue(text.contains("SIL OPEN FONT LICENSE"), credit.family)
        }
    }

    func testBoardRendersWithEveryPair() throws {
        for pair in ShopFontPair.allCases {
            var snapshot = ShopBoardTestData.snapshot()
            snapshot.pair = pair
            let image = try XCTUnwrap(ShopBoardExport.image(snapshot), pair.rawValue)
            XCTAssertEqual(image.cgImage?.width, 1080, pair.rawValue)
        }
    }
}

// MARK: - g) Caption

final class ShopCaptionTests: XCTestCase {
    func testAddsAdOnce() {
        XCTAssertEqual(ShopCaption.compliant("Shop the look", title: "My picks", link: "https://shop.example.com/a"),
                       "Shop the look #ad")
        XCTAssertEqual(ShopCaption.compliant("Shop the look #ad", title: "My picks", link: ""), "Shop the look #ad")
        XCTAssertEqual(ShopCaption.compliant("So #adorable", title: "", link: ""), "So #adorable #ad")
        XCTAssertEqual(ShopCaption.compliant("", title: "", link: ""), "#ad")
        let once = ShopCaption.compliant("Hi", title: "Amazon finds", link: "")
        XCTAssertEqual(ShopCaption.compliant(once, title: "Amazon finds", link: ""), once, "idempotent")
    }

    func testAmazonAddsTheAssociatesLine() {
        let line = "As an Amazon Associate I earn from qualifying purchases."
        for link in ["https://www.amazon.com/dp/B0X", "https://amazon.co.uk/x", "amazon.com.mx/x", "https://amzn.to/3abc",
                     "https://a.co/d/xyz", "https://smile.amazon.de/x"] {
            XCTAssertTrue(ShopCaption.compliant("Look", title: "My picks", link: link).hasSuffix(line), link)
        }
        XCTAssertTrue(ShopCaption.compliant("Look", title: "Amazon finds", link: "").contains(line), "title says Amazon")
        XCTAssertTrue(ShopCaption.compliant("Look", title: "", link: "https://walmart.com/ip/1").contains("#ad"))
        XCTAssertFalse(ShopCaption.compliant("Look", title: "Walmart", link: "https://walmart.com/ip/1").contains(line))
        XCTAssertFalse(ShopCaption.compliant("Look", title: "", link: "https://notamazon.com/x").contains(line))
        XCTAssertFalse(ShopCaption.compliant("Look", title: "", link: "https://amazon.example.com/x").contains(line))
        XCTAssertTrue(ShopCaption.compliant("Look", title: "", link: "https://amzn.to/x").contains("#ad"))
    }

    func testFigmaDefaultCaption() {
        XCTAssertEqual(ShopCaption.defaultCaption(scriptLine: "Everyday chic"),
                       "Everyday chic for fall 🍂 Shop the look — link in bio. #ad #affiliate")
        XCTAssertTrue(ShopCaption.clipboard(caption: "Look", title: "", link: "https://x.co/1").hasSuffix("\n\nhttps://x.co/1"))
    }
}

// MARK: - h) Export 9:16 + layout

enum ShopBoardTestData {
    static func snapshot() -> ShopBoardSnapshot {
        let images = [UIColor.blue, .gray, .lightGray, .darkGray, .brown, .white].map { ShopSynthetic.cutout($0) }
        let kinds: [ShopItemKind] = [.top, .top, .accessory, .bottom, .bag, .shoes]
        return ShopBoardSnapshot(images: images, placements: ShopLayout.placements(kinds: kinds),
                                 top: ShopRGB(hex: 0xF1ECE4), bottom: ShopRGB(hex: 0xE4DDD2), texture: nil, photo: nil,
                                 bigTitle: "CUTSELL", scriptLine: "Everyday chic", pair: .classic, titleColor: ShopRGB(hex: 0x2A2320))
    }
}

@MainActor
final class ShopExportTests: XCTestCase {
    func testBoardExportsAt1080x1920() throws {
        let image = try XCTUnwrap(ShopBoardExport.image(ShopBoardTestData.snapshot()))
        let cg = try XCTUnwrap(image.cgImage)
        XCTAssertEqual(cg.width, 1080)
        XCTAssertEqual(cg.height, 1920)
        XCTAssertEqual(Double(cg.width) / Double(cg.height), 9.0 / 16.0, accuracy: 0.0001)
        XCTAssertNotNil(image.pngData())
    }

    func testEveryTextureRendersAndExports() throws {
        for texture in ShopTexture.allCases {
            let swatch = ShopTextureRenderer.image(texture, size: CGSize(width: 74, height: 74))
            XCTAssertEqual(swatch.cgImage?.width, 74, texture.rawValue)
            var snapshot = ShopBoardTestData.snapshot()
            snapshot.texture = texture
            XCTAssertEqual(ShopBoardExport.image(snapshot)?.cgImage?.height, 1920, texture.rawValue)
        }
    }
}

final class ShopLayoutTests: XCTestCase {
    func testFigmaArrangementPutsEachKindInItsPlace() {
        // Items given out of order: shoes, bag, pants, earrings, sweater, shirt.
        let kinds: [ShopItemKind] = [.shoes, .bag, .bottom, .accessory, .top, .top]
        let placed = ShopLayout.placements(kinds: kinds)
        XCTAssertEqual(placed.count, 6)
        func slot(of item: Int) -> ShopSlot { placed.first { $0.itemIndex == item }!.slot }
        XCTAssertEqual(slot(of: 0), ShopLayout.figmaSlots[5], "shoes at the bottom")
        XCTAssertEqual(slot(of: 1), ShopLayout.figmaSlots[4], "bag on the left")
        XCTAssertEqual(slot(of: 2), ShopLayout.figmaSlots[3], "pants on the long place")
        XCTAssertEqual(slot(of: 3), ShopLayout.figmaSlots[2], "earrings top left")
        XCTAssertEqual(Set(placed.map(\.itemIndex)), Set(0..<6))
    }

    func testShuffleGivesAnotherArrangementWithEveryItem() {
        let kinds: [ShopItemKind] = [.top, .top, .accessory, .bottom, .bag, .shoes]
        let first = ShopLayout.placements(kinds: kinds)
        let second = ShopLayout.placements(kinds: kinds, variant: 1)
        XCTAssertNotEqual(first, second)
        XCTAssertEqual(Set(second.map(\.itemIndex)), Set(0..<6))
        XCTAssertEqual(ShopLayout.placements(kinds: kinds, variant: 2), ShopLayout.placements(kinds: kinds, variant: 2), "repeatable")
    }

    func testThreeToEightItems() {
        XCTAssertEqual(ShopLayout.placements(kinds: Array(repeating: .unknown, count: 3)).count, 3)
        XCTAssertEqual(Set(ShopLayout.placements(kinds: Array(repeating: .unknown, count: 8)).map(\.itemIndex)), Set(0..<8))
        let items = (0..<3).map { _ in ShopItem(source: .photos, cutout: ShopSynthetic.cutout(.red)) }
        XCTAssertTrue(ShopBoardRules.canChooseStyle(items))
        XCTAssertFalse(ShopBoardRules.canChooseStyle(Array(items.prefix(2))))
        XCTAssertEqual(ShopBoardRules.remaining(items), 5)
    }

    func testItemsFitTheirPlaceOnA916Board() {
        let slot = ShopLayout.figmaSlots[3]
        let frame = ShopLayout.frame(for: slot, aspect: 0.5, board: ShopLayout.exportSize)
        let sx = 1080 / 390.0, sy = 1920 / 844.0
        XCTAssertLessThanOrEqual(frame.width, slot.size.width * sx + 0.01)
        XCTAssertLessThanOrEqual(frame.height, slot.size.height * sy + 0.01)
        XCTAssertEqual(frame.width / frame.height, 0.5, accuracy: 0.001)
        XCTAssertEqual(frame.midX, slot.center.x * sx, accuracy: 0.01)
    }
}
