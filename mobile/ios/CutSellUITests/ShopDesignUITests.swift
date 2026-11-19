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
    private func el(_ id: String) -> XCUIElement { app.descendants(matching: .any).matching(identifier: id).firstMatch }
    private func pause(_ s: Double) { RunLoop.current.run(until: Date().addingTimeInterval(s)) }
    private func tap(_ id: String) {
        var e = el(id)
        if !e.waitForExistence(timeout: 2), let label = fallbackLabels[id] {
            e = app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", label)).firstMatch
        }
        let ok = e.waitForExistence(timeout: 5)
        log("tap \(id): exists=\(ok) hittable=\(ok && e.isHittable) frame=\(ok ? "\(e.frame)" : "-")")
        if ok { e.tap() }
        pause(0.8)
    }

    private let fallbackLabels = ["shopTitle": "Title", "shopBackground": "Background", "shopTitleDone": "Done",
                                  "shopBackgroundDone": "Done"]

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
        shot("added_tag")
        tap("shopTitleDone")
        check(!el("shopTitleSheet").exists, "Done closes the sheet")
        shot("board_with_tag")
        // The tag starts in the middle of the board (design 195 × 430 of 390 × 844).
        let layer = el("shopBoardTouchLayer")
        check(layer.waitForExistence(timeout: 3), "board touch layer")
        log("layer frame \(layer.frame) hittable=\(layer.isHittable)")
        let start = layer.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 430.0 / 844))
        let end = layer.coordinate(withNormalizedOffset: CGVector(dx: 0.7, dy: 0.3))
        start.press(forDuration: 0.1, thenDragTo: end)
        pause(1)
        shot("tag_dragged")
        end.press(forDuration: 1.0)
        pause(1)
        check(el("shopHoldMenu").exists, "hold menu shows")
        shot("hold_menu")
        tap("shopHoldDuplicate")
        shot("duplicated")
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
