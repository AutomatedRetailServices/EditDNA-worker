import XCTest

/// PO 19 Nov: "no me deja mover seleccionar nada de ese menu". The real Shop board flow (sample products) driven like a
/// finger would: Title & design (Layout, Letters, Add), the added thing moved and held, Background › Cut.
/// Frames + a PASS/FAIL log go to SD_OUT.
final class ShopDesignUITests: XCTestCase {
    private var app: XCUIApplication!
    private lazy var out: URL = {
        let url = URL(fileURLWithPath: ProcessInfo.processInfo.environment["SD_OUT"] ?? NSTemporaryDirectory(), isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()
    private var step = 0

    private func log(_ text: String) {
        let line = "SDUI \(text)\n"
        print(line, terminator: "")
        let file = out.appendingPathComponent("log.txt")
        if let h = try? FileHandle(forWritingTo: file) { h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); try? h.close() }
        else { try? line.data(using: .utf8)!.write(to: file) }
    }
    private func check(_ ok: Bool, _ name: String) { log("\(ok ? "PASS" : "FAIL") \(name)") }
    private func shot(_ name: String) {
        step += 1
        try? XCUIScreen.main.screenshot().pngRepresentation.write(to: out.appendingPathComponent(String(format: "%02d_%@.png", step, name)))
    }
    private func el(_ id: String) -> XCUIElement {
        let byID = app.descendants(matching: .any).matching(identifier: id).firstMatch
        if byID.exists { return byID }
        if let label = fallbackLabels[id] {
            let byLabel = app.descendants(matching: .any).matching(NSPredicate(format: "label == %@ OR label BEGINSWITH %@", label, label + ",")).firstMatch
            if byLabel.exists { return byLabel }
        }
        return byID
    }
    private func pause(_ s: Double) { RunLoop.current.run(until: Date().addingTimeInterval(s)) }
    private func tap(_ id: String) {
        var e = el(id)
        if !e.waitForExistence(timeout: 2), let label = fallbackLabels[id] {
            e = app.descendants(matching: .any).matching(NSPredicate(format: "label == %@ OR label BEGINSWITH %@", label, label + ",")).firstMatch
        }
        let ok = e.waitForExistence(timeout: 5)
        log("tap \(id): exists=\(ok) hittable=\(ok && e.isHittable) frame=\(ok ? "\(e.frame)" : "-")")
        if ok { e.tap() }
        pause(0.8)
    }

    private let fallbackLabels = ["shopTitle": "Title", "shopBackground": "Background", "shopTitleDone": "Done",
                                  "shopBackgroundDone": "Done", "shopDesignTab_2": "Letters", "shopDesignTab_3": "Add",
                                  "shopLetter_instrument": "DUPES THAT DON'T", "shopAdd_tag": "Tag", "shopAdd_sparkle": "Sparkle",
                                  "shopBgTab_Cut": "Cut", "shopCut_doubleFrame": "Double frame", "shopHoldDuplicate": "Duplicate",
                                  "shopLayout_tall": "Tall"]

    func testTitleAndDesignWorksByHand() {
        app = XCUIApplication()
        app.launchArguments = ["-CutSellScreen", "shop_flow"]
        app.launch()
        pause(3)
        shot("board")
        log("buttons: " + app.buttons.allElementsBoundByIndex.prefix(25).map { "\($0.identifier)=\($0.label)" }.joined(separator: " | "))
        tap("shopTitle")
        check(el("shopTitleSheet").waitForExistence(timeout: 4), "Title & design opens")
        shot("sheet")
        tap("shopLayout_tall")
        check(el("shopLayout_tall").isSelected, "Layout › Tall gets picked")
        shot("layout_tall")
        tap("shopDesignTab_2")
        check(el("shopLetter_caveat").waitForExistence(timeout: 3), "Letters tab opens")
        tap("shopLetter_instrument")
        check(el("shopLetter_instrument").isSelected, "Letters › Instrument gets picked")
        shot("letters")
        tap("shopDesignTab_3")
        check(el("shopAdd_tag").waitForExistence(timeout: 3), "Add tab opens")
        tap("shopAdd_tag")
        pause(1)
        check(!el("shopTitleSheet").exists, "Add closes the sheet: the tag waits on the board")
        shot("board_with_tag")
        // The tag starts in the middle of the board (design 195 × 430 of 390 × 844).
        // The board on Your board (iPhone Pro, 402 × 874 points): x 70, y 127, 262 × 466.
        let window = app.windows.firstMatch
        func board(_ x: CGFloat, _ y: CGFloat) -> XCUICoordinate {
            window.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: 70 + 262 * x, dy: 127 + 466 * y))
        }
        let start = board(0.5, 430.0 / 844)
        let end = board(0.7, 0.3)
        start.press(forDuration: 0.1, thenDragTo: end)
        pause(1)
        shot("tag_dragged")
        end.press(forDuration: 1.0)
        pause(1)
        check(el("shopHoldMenu").exists, "hold menu shows")
        shot("hold_menu")
        tap("shopHoldDuplicate")
        shot("duplicated")
        // Instagram-style bin: drag the copy (just below-right of the first) to the bottom of the board.
        let copy = board(0.7 + 16.0 / 390, 0.3 + 16.0 / 693)
        let bin = board(0.5, 1 - 46.0 / 466)
        copy.press(forDuration: 0.15, thenDragTo: bin, withVelocity: .slow, thenHoldForDuration: 0.6)
        pause(1)
        shot("dropped_in_bin")
        tap("shopBackground")
        tap("shopBgTab_Cut")
        check(el("shopCut_doubleFrame").waitForExistence(timeout: 3), "Cut tab opens")
        tap("shopCut_doubleFrame")
        check(el("shopCut_doubleFrame").isSelected, "Cut › Double frame gets picked")
        shot("cut")
        tap("shopBackgroundDone")
        shot("final")
    }
}
