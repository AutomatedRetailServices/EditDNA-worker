import XCTest

/// Build 21: the upload in REAL conditions. Swanny's iPhone (build 19) uploaded at ~1 MB/s and the
/// app cancelled its own parts after 30 s. Here the Mac's network is limited to ~1 MB/s (dummynet,
/// set by the workflow), the video is over 100 MB, and the creator LEAVES CutSell (Home) in the
/// middle of the upload for 25 s, then comes back. Pass: the upload finishes with no "Upload paused"
/// and no "No data sent" notice, the % never goes back to 0, the cut is processed and opens in the
/// editor. Frames and a log go to B24_OUT; no video ever leaves the test machine.
final class Build24UITests: XCTestCase {
    private var app: XCUIApplication!
    private lazy var out: URL = {
        let path = ProcessInfo.processInfo.environment["B24_OUT"] ?? NSTemporaryDirectory()
        let url = URL(fileURLWithPath: path, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()
    private let started = Date()
    private var step = 0
    private var failures: [String] = []

    override func setUp() { continueAfterFailure = true }

    private func log(_ text: String) {
        let line = "B24UI \(String(format: "%7.1f", Date().timeIntervalSince(started))) \(text)\n"
        print(line, terminator: "")
        let file = out.appendingPathComponent("b24_ui_log.txt")
        if let handle = try? FileHandle(forWritingTo: file) {
            handle.seekToEndOfFile(); handle.write(line.data(using: .utf8)!); try? handle.close()
        } else {
            try? line.data(using: .utf8)!.write(to: file)
        }
    }

    private func check(_ ok: Bool, _ name: String, _ detail: String = "") {
        log("\(ok ? "PASS" : "FAIL") \(name)\(detail.isEmpty ? "" : " · \(detail)")")
        if !ok { failures.append(name) }
    }

    private func shot(_ name: String) {
        step += 1
        try? XCUIScreen.main.screenshot().pngRepresentation
            .write(to: out.appendingPathComponent(String(format: "b24_%02d_%@.png", step, name)))
    }

    private func element(_ id: String) -> XCUIElement { app.descendants(matching: .any).matching(identifier: id).firstMatch }
    private func pause(_ seconds: Double) { RunLoop.current.run(until: Date().addingTimeInterval(seconds)) }

    @discardableResult
    private func wait(_ seconds: Double, until condition: () -> Bool) -> Bool {
        let end = Date().addingTimeInterval(seconds)
        while Date() < end {
            if condition() { return true }
            RunLoop.current.run(until: Date().addingTimeInterval(0.5))
        }
        return condition()
    }

    /// The texts on screen (to see where a flow stopped).
    private func screenTexts() -> String {
        app.staticTexts.allElementsBoundByIndex.prefix(30).map(\.label).filter { !$0.isEmpty }.joined(separator: " | ")
    }

    private func launch() {
        app = XCUIApplication()
        app.launchArguments = ["-CutSellScreen", "create_probe"]
        let env = ProcessInfo.processInfo.environment
        for key in ["CUTSELL_PROBE_TOKEN", "CUTSELL_PROBE_USER", "CUTSELL_PROBE_API"] {
            if let value = env[key] { app.launchEnvironment[key] = value }
        }
        app.launch()
    }

    private func allowPhotos() {
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        for _ in 0..<30 {
            for source in [springboard, app!] {
                let allow = source.buttons.matching(NSPredicate(format: "label CONTAINS[c] %@", "Full Access")).firstMatch
                if allow.exists { allow.tap(); log("Photos access allowed"); return }
            }
            if element("galleryTile").exists { return }
            pause(1)
        }
    }

    private func launchApp(fresh: Bool, extra: [String] = []) {
        app = XCUIApplication()
        // The whole app (four tabs + start-up steps), with the probe's session.
        app.launchArguments = ["-CutSellScreen", "app_probe"] + extra
        let env = ProcessInfo.processInfo.environment
        for key in ["CUTSELL_PROBE_TOKEN", "CUTSELL_PROBE_USER", "CUTSELL_PROBE_API"] {
            if let value = env[key] { app.launchEnvironment[key] = value }
        }
        app.launch()
        _ = element("hubUpload").waitForExistence(timeout: 60)
    }

    private func pickAndCreate(mode: String, count: Int) -> Bool {
        // A mode chip on the Home Hub opens the gallery in that mode.
        let modeChip = element("hubMode_\(mode)")
        if modeChip.waitForExistence(timeout: 10) { modeChip.tap() } else { element("hubUpload").tap() }
        allowPhotos()
        let tiles = app.descendants(matching: .any).matching(identifier: "galleryTile")
        _ = wait(30) { tiles.count >= count }
        guard tiles.count >= count else { check(false, "\(count) videos in the gallery", "\(tiles.count)"); return false }
        if mode != "single" {
            // The chip must open the gallery in its own mode (Single's helper is "One video → one edit.").
            let openedInSingle = app.staticTexts["One video → one edit."].exists
            check(!openedInSingle, "the \(mode) chip opened the gallery in \(mode)")
            if openedInSingle { app.buttons[mode.capitalized].firstMatch.tap(); pause(0.5) }
        }
        for index in 0..<count { tiles.element(boundBy: index).tap(); pause(0.8) }
        let create = element("galleryCreateButton")
        log("gallery CTA: \(create.label)")
        shot("gallery_\(mode)")
        create.tap()
        let style = element("styleCreateButton")
        guard style.waitForExistence(timeout: 10) else { check(false, "style sheet"); return false }
        style.tap()
        if wait(20, until: { self.element("aiConsentAgree").exists || self.app.buttons["Agree"].exists }) {
            (element("aiConsentAgree").exists ? element("aiConsentAgree") : app.buttons["Agree"]).tap()
        }
        return true
    }

    private func openEdits() {
        let edits = element("tab_edits")
        if edits.waitForExistence(timeout: 5) { edits.tap() } else { log("no Edits tab button") }
        pause(1.5)
        // Edits opens on All: show Processing when it is there.
        let processing = element("editsFilterProcessing")
        if processing.waitForExistence(timeout: 5) { processing.tap(); pause(1) } else { log("no Processing chip") }
    }

    /// The status text of each Processing row, top to bottom. Read from one snapshot of the screen, so a
    /// row that moves to Ready while it is being read cannot fail the test ("no matches for index 2").
    private func rowStatuses() -> [String] {
        guard let root = try? app.snapshot() else { return [] }
        var found: [(y: CGFloat, label: String)] = []
        func walk(_ node: XCUIElementSnapshot) {
            if node.identifier == "queueRowStatus", node.elementType == .staticText {
                found.append((node.frame.minY, node.label))
            }
            node.children.forEach(walk)
        }
        walk(root)
        return found.sorted { $0.y < $1.y }.map(\.label)
    }

    /// Waits until Edits · Processing is empty; logs the rows and every "is ready" notice.
    private func waitForProcessingToEmpty(_ label: String, minutes: Double) -> (done: Bool, toasts: Int) {
        var toasts = 0
        var last = ""
        var toastShown = false
        let deadline = Date().addingTimeInterval(minutes * 60)
        while Date() < deadline {
            if element("readyToast").exists {
                if !toastShown { toasts += 1; toastShown = true; log("\(label) notice: \(element("readyToast").label)"); shot("\(label)_toast") }
            } else { toastShown = false }
            let now = rowStatuses().joined(separator: ", ")
            if now != last { last = now; log("\(label) rows: [\(now)]") }
            // The Processing chip disappears when nothing is being made any more.
            if !element("editsFilterProcessing").exists { return (true, toasts) }
            if !element("editsProcessingList").exists, element("editsFilterProcessing").isHittable {
                element("editsFilterProcessing").tap()
            }
            if rowStatuses().contains(where: { $0 == "FAILED" }) { shot("\(label)_failed") }
            pause(2)
        }
        return (false, toasts)
    }

    /// Bulk: the "being edited" sheet → Go to Edits → Processing in order → each cut moves to Ready.
    func testBulkSheetAndProcessingTab() throws {
        launchApp(fresh: true)
        guard pickAndCreate(mode: "bulk", count: 3) else { return }
        let sheet = element("bulkStartedSheet")
        check(sheet.waitForExistence(timeout: 30), "bulk sheet shown")
        shot("bulk_sheet")
        check(!element("processingStage").exists, "Bulk shows no single-video Processing screen (Step x of 8)")
        element("bulkGoToEdits").tap()
        let list = element("editsProcessingList")
        check(list.waitForExistence(timeout: 15), "Edits opens on Processing")
        pause(2)
        shot("processing_tab")
        let first = rowStatuses()
        log("rows at start: \(first)")
        check(first.count == 3, "3 rows in order", first.joined(separator: ", "))
        let result = waitForProcessingToEmpty("bulk", minutes: 5)
        check(result.done, "every cut left Processing")
        check(result.toasts >= 1, "ready notice shown", "\(result.toasts)")
        let ready = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Ready ('")).firstMatch
        log("ready chip: \(ready.exists ? ready.label : "?")")
        shot("ready_tab")
        log("FAILURES: \(failures.isEmpty ? "none" : failures.joined(separator: " | "))")
    }

    /// Single video: Continue in background while it uploads, leave CutSell, close it (terminate)
    /// while it uploads and again while it is being cut; reopen each time. The cut must finish.
    func testLeaveAndCloseAtEachStage() throws {
        launchApp(fresh: true)
        guard pickAndCreate(mode: "single", count: 1) else { return }
        // 1) Continue in background as soon as the Processing screen shows.
        let cont = element("continueInBackground")
        check(cont.waitForExistence(timeout: 30), "Processing screen")
        let stage1 = element("processingStage").exists ? element("processingStage").label : "?"
        check(cont.isHittable, "Continue in background active while: \(stage1)")
        cont.tap()
        pause(1)
        check(!element("continueInBackground").exists, "screen closed, work goes on", stage1)
        openEdits()
        shot("after_continue")
        log("rows after continue: \(rowStatuses())")
        // 2) Leave CutSell (Home) for 40 s.
        XCUIDevice.shared.press(.home)
        pause(40)
        app.activate()
        pause(3)
        log("rows after Home: \(rowStatuses())")
        // 3) Close CutSell while the cut is uploading or being made, then reopen.
        let beforeClose = rowStatuses()
        log("closing CutSell with rows: \(beforeClose)")
        app.terminate()
        pause(5)
        launchApp(fresh: false)
        openEdits()
        let processing = element("editsFilterProcessing")
        if processing.waitForExistence(timeout: 20) { processing.tap() }
        pause(3)
        log("rows after reopen: \(rowStatuses())")
        shot("after_reopen")
        // 4) While it is being cut on the server: close again and reopen.
        if wait(240, until: { self.rowStatuses().contains { $0.hasPrefix("PROCESSING") || $0 == "QUEUED" } }) {
            log("closing CutSell during: \(rowStatuses())")
            app.terminate()
            pause(20)
            launchApp(fresh: false)
            openEdits()
            if element("editsFilterProcessing").waitForExistence(timeout: 10) { element("editsFilterProcessing").tap() }
        }
        let result = waitForProcessingToEmpty("single", minutes: 6)
        check(result.done, "the cut finished after leaving and closing CutSell")
        shot("single_end")
        log("FAILURES: \(failures.isEmpty ? "none" : failures.joined(separator: " | "))")
    }

    /// Build 27: one cut, then Export all the way to the video saved in Photos (the export of 19 Oct failed on the
    /// server: a cut made from the voice file ended 0.06 s past the video's end).
    func testExportSavedToPhotos() throws {
        launchApp(fresh: true)
        guard pickAndCreate(mode: "single", count: 1) else { return }
        let review = element("readyReviewButton")
        check(review.waitForExistence(timeout: 600), "the cut is ready")
        review.tap()
        check(element("timelineV4").waitForExistence(timeout: 120), "editor open")
        pause(2)
        let started = Date()
        let export = element("editorExport")
        check(export.waitForExistence(timeout: 20), "Export button")
        export.tap()
        let done = wait(600) {
            self.app.staticTexts["Your video is ready."].exists || self.app.staticTexts["The export didn't finish. Please try again."].exists
                || self.element("exportError").exists
        }
        let ready = app.staticTexts["Your video is ready."].exists
        check(done && ready, "export finished", String(format: "%.0f s", Date().timeIntervalSince(started)))
        shot("export_result")
        guard ready else { return }
        let save = app.buttons["Save"].firstMatch
        check(save.waitForExistence(timeout: 120), "video downloaded to the phone (Save shown)")
        save.tap()
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        for _ in 0..<15 {
            for label in ["Allow", "Allow Access", "OK", "Allow Full Access"] {
                let button = springboard.buttons[label]
                if button.exists { button.tap(); log("Photos add access: \(label)"); break }
            }
            if app.buttons["Saved"].exists { break }
            pause(1)
        }
        check(app.buttons["Saved"].waitForExistence(timeout: 60), "saved to Photos")
        shot("export_saved")
        log("FAILURES: \(failures.isEmpty ? "none" : failures.joined(separator: " | "))")
    }

    /// Edits · Products (Figma 1135:2686 / 2876 / 3059) with the in-memory catalog (-CutSellCatalogMock): the
    /// Products list, a product, renaming it, assigning the videos that need a product (new brand + product), and
    /// moving 3 videos to another product.
    func testProductsFlow() throws {
        launchApp(fresh: true, extra: ["-CutSellCatalogMock"])
        element("tab_edits").tap()
        pause(2)
        let chip = element("editsFilterProducts")
        check(chip.waitForExistence(timeout: 20), "Products filter in Edits")
        chip.tap()
        check(element("productsList").waitForExistence(timeout: 20), "Products list")
        pause(2)
        shot("products_list")
        let banner = element("productsAssignBanner")
        check(banner.waitForExistence(timeout: 10), "‘need a product’ notice", banner.exists ? banner.label : "")

        // A product: its videos, then Rename.
        let row = element("productRow_Hydrating Cleanser")
        check(row.waitForExistence(timeout: 10), "product row")
        row.tap()
        check(element("productTitle").waitForExistence(timeout: 15), "product view")
        pause(2)
        shot("product_detail")
        let tiles = app.descendants(matching: .any).matching(identifier: "productVideo")
        log("videos in the product: \(tiles.count)")
        element("productRename").tap()
        let field = app.alerts.textFields.firstMatch
        if field.waitForExistence(timeout: 5) {
            field.tap(); field.typeText(" Gel")
            app.alerts.buttons["Save"].firstMatch.tap()
        }
        pause(2)
        let title = element("productTitle")
        check(title.label.contains("Gel"), "renamed", title.label)

        // Select 3 videos and move them to another product.
        let videosBefore = tiles.count
        element("productSelect").tap()
        pause(1)
        for index in 0..<min(3, tiles.count) { tiles.element(boundBy: index).tap(); pause(0.4) }
        shot("product_selected")
        element("productSelect").tap()                 // "Move N"
        let option = element("assignOption_Dior · Miss Dior")
        check(option.waitForExistence(timeout: 10), "assign sheet with her products")
        shot("assign_sheet_move")
        option.tap()
        element("assignMove").tap()
        pause(3)
        let videosAfter = app.descendants(matching: .any).matching(identifier: "productVideo").count
        check(videosAfter == max(0, videosBefore - 3), "3 videos moved out", "\(videosBefore) → \(videosAfter)")
        element("productBack").tap()
        pause(2)

        // The videos that need a product: suggestion + a new brand and product.
        let assign = element("productsAssignBanner")
        if assign.waitForExistence(timeout: 10) {
            assign.tap()
            check(element("assignConfirmSuggestion").waitForExistence(timeout: 10), "suggestion from the video shown")
            shot("assign_sheet_need")
            element("assignNew").tap()
            let brand = app.alerts.textFields.element(boundBy: 0)
            if brand.waitForExistence(timeout: 5) {
                brand.tap(); brand.typeText("Glow Recipe")
                let product = app.alerts.textFields.element(boundBy: 1)
                product.tap(); product.typeText("Watermelon Toner")
                app.alerts.buttons["Move here"].firstMatch.tap()
            }
            pause(3)
            check(element("productRow_Watermelon Toner").waitForExistence(timeout: 10), "new brand and product created")
            check(!element("productsAssignBanner").exists, "no video left without a product")
        }
        shot("products_after")
        log("FAILURES: \(failures.isEmpty ? "none" : failures.joined(separator: " | "))")
    }
    /// Mix (Figma 1136:2724 / 2804 / 2916) with in-memory catalog + Mix (-CutSellCatalogMock -CutSellMixMock): the
    /// product, clips from Photos in 4 sections (3 hooks, 2 bodies, 2 benefits, 2 CTAs), rename / add / move / delete a
    /// section, "Pick how many" → 12, Processing shows ONE group, Results (MIX tiles with codes) and Export all.
    func testMixFlow() throws {
        launchApp(fresh: true, extra: ["-CutSellCatalogMock", "-CutSellMixMock"])
        let chip = element("hubMode_mix")
        check(chip.waitForExistence(timeout: 20), "Mix on Create")
        chip.tap()
        check(element("mixProduct").waitForExistence(timeout: 15), "Mix setup")
        pause(2)
        let product = element("mixProductName")
        check(product.exists && product.label.contains("·"), "a product chosen", product.label)
        shot("mix_setup_empty")
        for (section, count) in [(0, 3), (1, 2), (2, 2), (3, 2)] {
            element("mixAddClip_\(section)").tap()
            let photos = app.buttons["Choose from Photos"].firstMatch
            guard photos.waitForExistence(timeout: 10) else { check(false, "add-clip menu \(section)"); continue }
            photos.tap()
            allowPhotos()
            let tiles = app.descendants(matching: .any).matching(identifier: "galleryTile")
            _ = wait(30) { tiles.count >= count }
            for index in 0..<min(count, tiles.count) { tiles.element(boundBy: index).tap(); pause(0.5) }
            let add = element("galleryCreateButton")
            check(add.label.contains("Add \(count) clip"), "gallery says Add \(count) clips", add.label)
            add.tap()
            _ = wait(30) { self.element("mixProduct").exists && !self.element("galleryCreateButton").exists }
            pause(2)
        }
        let clips = app.descendants(matching: .any).matching(identifier: "mixClip")
        check(wait(20) { clips.count == 9 }, "9 clips in 4 sections", "\(clips.count)")
        let summary = element("mixSummary")
        check(summary.label.hasPrefix("24 videos"), "24 videos · about N s each", summary.label)
        shot("mix_setup_filled")

        // Rename Benefit → Results, add a section and delete it, move CTA up and back.
        element("mixSectionMore_2").tap()
        if app.buttons["Rename"].firstMatch.waitForExistence(timeout: 5) {
            app.buttons["Rename"].firstMatch.tap()
            let field = app.alerts.textFields.firstMatch
            if field.waitForExistence(timeout: 5) {
                field.tap(); field.clearAndType("Results")
                app.alerts.buttons["Save"].firstMatch.tap()
            }
        }
        pause(1)
        check(element("mixSectionName_2").label == "Results", "section renamed", element("mixSectionName_2").label)
        element("mixAddSection").tap(); pause(1)
        check(element("mixSection_4").exists, "section added")
        element("mixSectionMore_4").tap()
        app.buttons["Delete section"].firstMatch.tap(); pause(1)
        check(!element("mixSection_4").exists, "section deleted")
        element("mixSectionMove_3").tap()
        app.buttons["Move up"].firstMatch.tap(); pause(1)
        check(element("mixSectionName_2").label == "CTA", "section moved up", element("mixSectionName_2").label)
        element("mixSectionMove_2").tap()
        app.buttons["Move down"].firstMatch.tap(); pause(1)
        check(element("mixSectionName_3").label == "CTA", "section moved back", element("mixSectionName_3").label)

        // How many videos?
        element("mixPickHowMany").tap()
        check(element("mixHowMany").waitForExistence(timeout: 10), "How many videos? sheet")
        pause(2)
        let rows = app.descendants(matching: .any).matching(identifier: "mixPreviewRow")
        check(wait(10) { rows.count == 4 }, "preview of the first 4 videos", "\(rows.count)")
        check(element("mixHowManyMake").label.contains("12"), "12 chosen", element("mixHowManyMake").label)
        shot("mix_how_many")
        element("mixHowManyMake").tap()

        // Edits · Processing: one group for the Mix, then its videos.
        let group = element("mixGroup")
        check(group.waitForExistence(timeout: 30), "Mix group in Processing")
        check(element("mixGroupTitle").label == "Mix · 12 videos", "Mix · 12 videos", element("mixGroupTitle").label)
        shot("mix_processing")
        group.tap()
        check(element("mixBanner").waitForExistence(timeout: 15), "Mix results")
        _ = wait(30) { self.element("mixBanner").label.contains("Mix ready") }
        check(element("mixBanner").label.contains("Mix ready: 12 videos from 9 clips"), "Mix ready banner", element("mixBanner").label)
        let badges = app.descendants(matching: .any).matching(identifier: "mixBadge")
        check(badges.count >= 6, "MIX tiles with their code", "\(badges.count)")
        check(element("productSubtitle").label.contains("Mix · 12 videos"), "Brand · Mix · 12 videos", element("productSubtitle").label)
        shot("mix_results")
        element("mixExportAll").tap()
        check(element("mixExportMessage").waitForExistence(timeout: 30), "Export all", element("mixExportMessage").label)
        shot("mix_exported")
        log("FAILURES: \(failures.isEmpty ? "none" : failures.joined(separator: " | "))")
    }

}

extension XCUIElement {
    /// Replaces the text in a field.
    func clearAndType(_ text: String) {
        if let current = value as? String, !current.isEmpty {
            typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: current.count))
        }
        typeText(text)
    }
}
