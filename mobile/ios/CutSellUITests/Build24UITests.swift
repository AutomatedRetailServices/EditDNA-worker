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
        for key in ["CUTSELL_PROBE_TOKEN", "CUTSELL_PROBE_USER", "CUTSELL_PROBE_API", "CUTSELL_PLAN_FIXTURE"] {
            if let value = env[key] { app.launchEnvironment[key] = value }
        }
        app.launch()
    }

    private func allowPhotos() {
        // The Photos prompt can come up late (a busy simulator): wait up to 60 s for it or for the gallery, in
        // SpringBoard, in the app and in the Photos picker service; an interruption monitor catches it too.
        let monitor = addUIInterruptionMonitor(withDescription: "Photos access") { alert in
            for label in ["Allow Full Access", "Allow Access to All Photos", "Allow"] where alert.buttons[label].exists {
                alert.buttons[label].tap(); return true
            }
            return false
        }
        defer { removeUIInterruptionMonitor(monitor) }
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let photosService = XCUIApplication(bundleIdentifier: "com.apple.mobileslideshow")
        for _ in 0..<60 {
            for source in [springboard, app!, photosService] {
                let allow = source.buttons.matching(NSPredicate(format: "label CONTAINS[c] %@", "Full Access")).firstMatch
                if allow.exists { allow.tap(); log("Photos access allowed"); return }
            }
            if element("galleryTile").exists { return }
            pause(1)
        }
    }

    private func launchApp(fresh: Bool, extra: [String] = [], env extraEnv: [String: String] = [:]) {
        app = XCUIApplication()
        // The whole app (four tabs + start-up steps), with the probe's session.
        app.launchArguments = ["-CutSellScreen", "app_probe"] + extra
        let env = ProcessInfo.processInfo.environment
        for key in ["CUTSELL_PROBE_TOKEN", "CUTSELL_PROBE_USER", "CUTSELL_PROBE_API", "CUTSELL_PLAN_FIXTURE"] {
            if let value = env[key] { app.launchEnvironment[key] = value }
        }
        for (key, value) in extraEnv { app.launchEnvironment[key] = value }
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
        check(!element("hubMode_bulk").exists && !element("hubMode_mix").exists, "Create is for one video (no Bulk, no Mix)")
        element("tab_plan").tap()
        let newMix = element("planNewMix")
        check(newMix.waitForExistence(timeout: 20), "Plan · New Mix")
        check(element("planBulk").exists && element("planMyWeek").exists, "Plan · Bulk and This week")
        shot("plan_home")
        // Bulk opens the Bulk gallery (the same pipeline as before); closed again here.
        element("planBulk").tap()
        allowPhotos()
        let bulkCTA = element("galleryCreateButton")
        check(bulkCTA.waitForExistence(timeout: 20), "Plan · Bulk opens the gallery")
        if app.buttons["Close"].firstMatch.exists { app.buttons["Close"].firstMatch.tap() }
        pause(2)
        newMix.tap()
        check(element("mixProduct").waitForExistence(timeout: 15), "Mix setup")
        pause(2)
        let product = element("mixProductName")
        check(product.exists && product.label.contains("·"), "a product chosen", product.label)
        shot("mix_setup_empty")
        let clips = app.descendants(matching: .any).matching(identifier: "mixClip")
        var expected = 0
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
            expected += count
            // the clips are copied from Photos first ("Getting your clips…"), then shown in their section
            let label = element("mixSection_\(section)").staticTexts["\(count) clips"]
            check(wait(90) { label.exists }, "section \(section) has its \(count) clips")
            pause(1)
        }
        check(expected == 9 && element("mixSummary").label.hasPrefix("24 videos"), "9 clips in 4 sections")
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
        check(wait(10) { rows.count >= 4 } && app.staticTexts["Video 4"].exists && !app.staticTexts["Video 5"].exists,
              "preview of the first 4 videos")
        check(element("mixHowManyMake").label.contains("12"), "12 chosen", element("mixHowManyMake").label)
        shot("mix_how_many")
        element("mixHowManyMake").tap()

        // Edits · Processing showing this Mix (as from Plan · In progress), one group for it.
        let group = element("mixGroup")
        check(group.waitForExistence(timeout: 30), "Mix group in Processing")
        check(element("processingFocus").exists, "Processing shows this Mix only")
        check(element("mixGroupTitle").label == "Mix · 12 videos", "Mix · 12 videos", element("mixGroupTitle").label)
        shot("mix_processing")
        element("tab_plan").tap()
        let card = element("planProgressMix")
        check(card.waitForExistence(timeout: 10), "Plan · In progress shows the Mix", card.exists ? card.label : "")
        shot("plan_in_progress")
        card.tap()
        check(group.waitForExistence(timeout: 15), "In progress → Edits · Processing (this Mix)")
        group.tap()
        check(element("mixBanner").waitForExistence(timeout: 15), "Mix results")
        _ = wait(60) { self.element("mixBanner").label.contains("Mix ready") }
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

    /// Plan my week (Figma 1138:2749 / 2798 / 2872) with an in-memory plan (-CutSellPlanMock): Goal → Build my week →
    /// Week (videos of the 4 origins) → move a video → Day: caption edit / copy, Share to TikTok (share sheet with the
    /// video), Save, Mark as posted → rebuild keeps it posted → a "Time to post" reminder opens its day.
    func testCreateHub() throws {
        launchApp(fresh: true, extra: ["-CutSellProjectsMock"])
        pause(2)
        shot("create_hub")
        check(element("planMeter").exists, "meter X / N cuts · Plan", element("planMeter").label)
        for mode in ["single", "bulk", "combine"] { check(element("hubMode_\(mode)").exists, "Upload videos · \(mode) chip") }
        check(element("hubMix").exists, "Mix row")
        check(element("hubRecord").exists, "Record with CutSell")
        check(!element("hubVoiceOver").exists && !element("hubFaceless").exists, "Faceless video hidden until it is built")
        check(app.buttons["See all in Edits"].exists, "Recent cuts · See all in Edits")
        // Mix row → Mix · Setup (the same screen as New Mix in Plan)
        element("hubMix").tap()
        check(element("mixProduct").waitForExistence(timeout: 15), "Mix row opens Mix · Setup")
        shot("create_mix_setup")
        element("mixClose").tap()
        pause(1)
        // Bulk → the gallery, with Single · Bulk · Combine at the bottom (Figma 484:214)
        element("hubMode_bulk").tap()
        allowPhotos()
        let picker = element("createModePicker")
        if !picker.waitForExistence(timeout: 20) {
            app.swipeUp(velocity: .slow)                 // nudges the interruption monitor if the prompt is still up
            allowPhotos()
        }
        check(picker.waitForExistence(timeout: 30), "gallery mode picker")
        for title in ["Single", "Bulk", "Combine"] {
            check(picker.buttons[title].exists || app.buttons[title].exists, "gallery shows \(title)")
        }
        check(app.staticTexts["Each video becomes a separate edit."].exists, "Bulk chip opened the gallery in Bulk")
        shot("create_gallery_bulk")
        done()
    }

    // MARK: Language: You → Language (Automatic · English · Español); the main screens in Spanish

    func testLanguage() throws {
        launchApp(fresh: true, extra: ["-CutSellProjectsMock"])
        pause(2)
        element("tab_you").tap(); pause(1.5)
        check(element("youOverview").waitForExistence(timeout: 10), "You · Overview v2")
        youTour("en")
        // Preferences → App language → Español: the app re-draws in Spanish at once and comes back on Language
        element("youRow_preferences").tap(); pause(1)
        element("prefLanguage").tap(); pause(1)
        check(element("language_automatic").exists && element("language_en").exists && element("language_es").exists,
              "System Default · English · Español")
        element("language_es").tap(); pause(3)
        check(app.staticTexts["Idioma de la app"].waitForExistence(timeout: 10), "Español: re-drawn in Spanish, back on App Language")
        check(element("language_es").isSelected, "Español is chosen")
        shot("es_you_language")
        element("youBack").tap(); pause(1.5)
        youTour("es")
        for (tab, word, name) in [("create", "Subir videos", "crear"), ("edits", "Mis videos", "misvideos"), ("plan", "Plan", "plan")] {
            element("tab_\(tab)").tap(); pause(2)
            check(app.staticTexts[word].exists || app.buttons[word].exists, "\(tab) in Spanish", word)
            shot("es_\(name)")
        }
        done()
    }




    /// Intro (Figma 1145:2773-2858) on the first visit, then Perfil → Help → How to use CutSell (1145:2898 / 2958).
    func testIntroGuide() throws {
        for lang in ["en", "es"] {
            launchApp(fresh: true, extra: ["-CutSellProjectsMock"], env: ["CUTSELL_INTRO": "1", "CUTSELL_LANGUAGE": lang])
            check(element("introView").waitForExistence(timeout: 20), "\(lang): intro shows on the first visit")
            for page in 1...4 {
                pause(1.2)
                shot("\(lang)_intro_\(page)")
                check(element("intro_skip").exists, "\(lang): Skip on screen \(page)")
                element("intro_next").tap()
            }
            pause(1.5)
            check(!element("introView").exists && element("hubUpload").exists, "\(lang): Get started ends on Create")
            element("tab_you").tap(); pause(1.2)
            let help = element("youRow_help")
            for _ in 0..<3 where !help.isHittable { app.swipeUp(velocity: .slow); pause(0.4) }
            help.tap(); pause(1.2)
            element("help_guide").tap(); pause(1.2)
            check(element("guidePage").exists, "\(lang): How to use CutSell opens the list")
            shot("\(lang)_guide_list")
            element("guide_mix").tap(); pause(1)
            check(element("guideDetail_mix").exists && element("guideStep_5").exists, "\(lang): Mix guide, 5 steps")
            shot("\(lang)_guide_mix")
            for id in ["upload", "editor", "keyframes", "voiceover", "teleprompter", "products", "planweek", "tiktok"] {
                element("guideDetailBack").tap(); pause(0.8)
                let row = element("guide_\(id)")
                for _ in 0..<3 where !row.isHittable { app.swipeUp(velocity: .slow); pause(0.3) }
                row.tap(); pause(0.9)
                shot("\(lang)_guide_\(id)")
            }
            element("guideDetailBack").tap(); pause(0.8)
            app.swipeDown(velocity: .fast); pause(0.4)
            let search = element("guideSearch")
            search.tap(); search.typeText(lang == "es" ? "voz" : "voice"); pause(0.8)
            check(element("guide_voiceover").exists && !element("guide_mix").exists, "\(lang): search finds Voice over")
            shot("\(lang)_guide_search")
            // Try Products now → My videos › Products
            search.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: 6)); pause(0.5)
            element("guide_products").tap(); pause(0.8)
            element("guideTry").tap(); pause(2.5)
            check(element("editsFilterProducts").exists && element("editsFilterProducts").isSelected,
                  "\(lang): Try Products now opens My videos › Products")
            element("tab_you").tap(); pause(1.2)
            for _ in 0..<3 where !help.isHittable { app.swipeUp(velocity: .slow); pause(0.4) }
            help.tap(); pause(1.2)
            element("help_intro").tap(); pause(1.5)
            check(element("introView").waitForExistence(timeout: 5), "\(lang): Replay intro shows the 4 screens again")
            element("intro_skip").tap(); pause(1.5)
            check(!element("introView").exists && element("hubUpload").exists, "\(lang): Skip ends on Create")
            app.terminate()
        }
        done()
    }

    /// Teleprompter (Figma 1145:2988 sheet, 1151:2824 over the camera), EN and ES.
    func testTeleprompter() throws {
        for lang in ["en", "es"] {
            app = XCUIApplication()
            app.launchArguments = ["-CutSellScreen", "camera", "-cutsell.prompter.scripts", "", "-cutsell.prompter.on", "NO",
                                   "-cutsell.prompter.speed2", "1.0", "-cutsell.prompter.size2", "28", "-cutsell.prompter.frame", "",
                                   "-cutsell.prompter.color", "white"]
            app.launchEnvironment["CUTSELL_LANGUAGE"] = lang
            app.launch()
            check(element("cameraScript").waitForExistence(timeout: 20), "\(lang): Script on the camera rail")
            shot("\(lang)_prompter_rail")
            element("cameraScript").tap(); pause(1.2)
            let editor = element("prompterEditorText")
            check(editor.waitForExistence(timeout: 5), "\(lang): no script yet → the script sheet opens")
            editor.tap()
            editor.typeText(lang == "es"
                ? "Deja de perder tiempo con correas viejas. Estas correas Rhino se traban de un jalón. Llevé mi kayak sin que se moviera nada. Vienen en paquete de 4 y hoy están en oferta. Toca el carrito naranja antes de que se acaben."
                : "Stop wasting time with old straps. These Rhino retractable straps lock in one pull. I hauled my whole kayak with zero slipping. They come in a 4-pack, and right now they're on sale. Tap the orange cart before they're gone.")
            app.swipeDown(velocity: .slow)   // keyboard away (interactive)
            pause(0.8)
            element("prompterSpeedPlus").tap(); pause(0.3)
            check(element("prompterSpeedValue").label == "1.1×", "\(lang): speed + → 1.1×", element("prompterSpeedValue").label)
            element("prompterSpeedMinus").tap(); pause(0.3)
            shot("\(lang)_prompter_sheet")
            element("prompterRecord").tap(); pause(1.5)
            check(element("prompterOverlay").waitForExistence(timeout: 5), "\(lang): the script shows over the camera")
            check(["On", "Activado", "Sí"].contains(element("cameraScript").value as? String ?? ""), "\(lang): Script is on",
                  element("cameraScript").value as? String ?? "")
            pause(3.5)                                     // 3·2·1 (no camera in the simulator: nothing records)
            shot("\(lang)_prompter_on")
            element("prompterText").tap(); pause(0.6)
            if !element("prompterAutoScroll").exists { element("prompterText").tap(); pause(0.6) }
            check(element("prompterAutoScroll").exists && element("prompterEdit").exists && element("prompterSettings").exists,
                  "\(lang): tap shows ✎ · Auto-scroll · ⚙ and the frame")
            shot("\(lang)_prompter_controls")
            pause(2.6)
            check(!element("prompterAutoScroll").exists, "\(lang): controls hide after 2 s")
            element("cameraScriptSpeed").tap(); pause(0.4)
            check(element("cameraScriptSpeed").label.contains("1.2"), "\(lang): Speed on the rail → 1.25×",
                  element("cameraScriptSpeed").label)
            element("prompterText").tap(); pause(0.5)
            element("prompterSettings").tap(); pause(1)
            shot("\(lang)_prompter_settings")
            element("prompterColor_yellow").tap(); pause(0.4)
            app.swipeDown(velocity: .fast); pause(1)
            element("cameraScript").tap(); pause(0.8)
            check(!element("prompterOverlay").exists, "\(lang): Script off hides it")
            element("cameraScript").tap(); pause(0.8)
            check(element("prompterOverlay").exists, "\(lang): Script on shows it again (script kept)")
            app.terminate()
        }
        done()
    }

    /// Day mode (Figma 1146:2 Create, 1146:91 My videos › Products, 1146:178 Plan) and every Perfil page.
    func testLightMode() throws {
        launchApp(fresh: true, extra: ["-CutSellProjectsMock", "-CutSellCatalogMock", "-cutsell.pref.appearance", "light"])
        pause(2)
        shot("light_create")
        element("tab_edits").tap(); pause(2); shot("light_edits")
        element("editsFilterProducts").tap(); pause(2); shot("light_products")
        element("tab_plan").tap(); pause(2); shot("light_plan")
        element("tab_you").tap(); pause(1.5)
        check(element("youOverview").waitForExistence(timeout: 10), "light: Perfil opens")
        youTour("light")
        let help = element("youRow_help")
        for _ in 0..<3 where !help.isHittable { app.swipeUp(velocity: .slow); pause(0.4) }
        help.tap(); pause(1.2)
        element("help_guide").tap(); pause(1.2); shot("light_guide_list")
        element("guide_mix").tap(); pause(1); shot("light_guide_mix")
        element("guideDetailBack").tap(); pause(0.6)
        element("guideBack").tap(); pause(0.8)
        element("youBack").tap(); pause(1)
        // Back to Dark from Appearance: the app changes at once.
        element("youRow_preferences").tap(); pause(1)
        element("prefAppearance").tap(); pause(1)
        element("appearance_dark").tap(); pause(1.5)
        check(element("appearance_dark").isSelected, "light → Dark chosen")
        shot("light_to_dark")
        done()
    }

    /// Overview and every You page, one screenshot each (lang = "en" / "es").
    private func youTour(_ lang: String) {
        element("tab_you").tap(); pause(1)
        shot("\(lang)_you_overview")
        for page in ["account", "plan", "privacy", "preferences", "connections", "help"] {
            let row = element("youRow_\(page)")
            for _ in 0..<3 where !row.isHittable { app.swipeUp(velocity: .slow); pause(0.4) }
            row.tap(); pause(1.2)
            shot("\(lang)_you_\(page)")
            if page == "preferences" {
                element("prefAppearance").tap(); pause(1); shot("\(lang)_you_appearance"); element("youBack").tap(); pause(0.8)
                if lang == "en" { element("prefLanguage").tap(); pause(1); shot("en_you_language"); element("youBack").tap(); pause(0.8) }
            }
            element("youBack").tap(); pause(1)
        }
        app.swipeDown(velocity: .fast); pause(0.5)
    }

    /// The editor in Spanish (a real cut; Automatic follows CUTSELL_LANGUAGE=es on the probe iPhone).
    func testSpanishEditor() throws {
        app = XCUIApplication()
        app.launchArguments = ["-CutSellScreen", "editor_probe"]
        let env = ProcessInfo.processInfo.environment
        for key in ["CUTSELL_PROBE_TOKEN", "CUTSELL_PROBE_USER", "CUTSELL_PROBE_PROJECT", "CUTSELL_PROBE_API"] {
            if let value = env[key] { app.launchEnvironment[key] = value }
        }
        app.launchEnvironment["CUTSELL_PROBE_STATE"] = "b18_es"
        app.launchEnvironment["CUTSELL_LANGUAGE"] = "es"
        app.launch()
        check(element("timelineV4").waitForExistence(timeout: 120), "editor open on a real cut")
        _ = wait(90) { self.element("editorPlay").exists && self.element("editorPlay").isEnabled }
        pause(3)
        check(app.staticTexts["Dividir"].exists && app.staticTexts["Exportar"].exists || app.buttons["Exportar"].exists, "editor tools in Spanish")
        shot("es_editor")
        editorTool("voiceOver")
        if element("voiceRecordSheet").waitForExistence(timeout: 10) {
            check(app.staticTexts["Añade tu voz a este video"].exists, "voice-over sheet in Spanish")
            shot("es_voiceover")
            element("voiceRecordSheet").swipeDown(velocity: .fast); pause(1)
        }
        editorTool("captions"); pause(1.5); shot("es_captions")
        done()
    }

    // MARK: Voice-over on a real cut (editor probe): record in 2 steps, timeline, panel, export

    private func editorState() -> String { let e = element("b17State"); return e.exists ? e.label : "no-state" }
    private func voices() -> [String] {
        let st = editorState()
        guard let range = st.range(of: "voices=") else { return [] }
        let rest = String(st[range.upperBound...].prefix { $0 != " " })      // "voices= muted…" = none
        return rest.isEmpty ? [] : rest.split(separator: ",").map(String.init)
    }
    private func editorTool(_ name: String) {
        let button = element("tool_\(name)")
        guard button.waitForExistence(timeout: 5) else { log("missing tool_\(name)"); return }
        let screen = app.windows.firstMatch.frame
        for _ in 0..<4 {
            let f = button.frame
            if f.minX >= 0 && f.maxX <= screen.maxX - 2 && button.isHittable { break }
            // swipe the tool row itself (at the button's height) toward the button
            let y = f.midY / max(1, screen.height)
            let from = app.coordinate(withNormalizedOffset: CGVector(dx: f.minX < 0 ? 0.2 : 0.8, dy: y))
            let to = app.coordinate(withNormalizedOffset: CGVector(dx: f.minX < 0 ? 0.8 : 0.2, dy: y))
            from.press(forDuration: 0.05, thenDragTo: to)
            pause(0.8)
        }
        button.tap()
        pause(1.0)
    }
    private func waitExport(_ label: String) -> Bool {
        element("editorExport").tap()
        let ready = app.staticTexts["Your video is ready."]
        let ok = ready.waitForExistence(timeout: 420)
        check(ok, "export \(label) finished")
        _ = element("postToTikTok").waitForExistence(timeout: 60)
        pause(3)
        shot("export_\(label)")
        app.swipeDown(velocity: .fast)              // close the Export sheet
        pause(1.5)
        return ok
    }

    func testVoiceOverFlow() throws {
        app = XCUIApplication()
        app.launchArguments = ["-CutSellScreen", "editor_probe"]
        let env = ProcessInfo.processInfo.environment
        for key in ["CUTSELL_PROBE_TOKEN", "CUTSELL_PROBE_USER", "CUTSELL_PROBE_PROJECT", "CUTSELL_PROBE_API"] {
            if let value = env[key] { app.launchEnvironment[key] = value }
        }
        app.launchEnvironment["CUTSELL_PROBE_STATE"] = "b18_vo"
        app.launchEnvironment["CUTSELL_PROBE_FAKE_MIC"] = "1"
        app.launch()
        check(element("timelineV4").waitForExistence(timeout: 120), "editor open on a real cut")
        _ = wait(90) { self.element("editorPlay").exists && self.element("editorPlay").isEnabled }
        pause(4)
        shot("vo_editor_open")
        log("start \(editorState())")

        // (d) baseline export (no voice): the workflow compares the video's sound with the one after
        _ = waitExport("before")

        // Closing the sheet (iPhone report, build 31): X, swipe down, tap outside; countdown cancel; discard asks first
        element("b17Step").tap(); pause(0.6)                       // cursor at 1 s
        let sheetOpen = { self.element("voiceRecordSheet").exists || self.element("voiceReviewSheet").exists }
        editorTool("voiceOver")
        check(element("voiceRecordSheet").waitForExistence(timeout: 10), "sheet opens")
        check(!element("voiceClose").exists, "no X on the sheet (Figma)")
        element("voiceRecordSheet").swipeDown(velocity: .fast)
        check(wait(3) { !sheetOpen() } && voices().isEmpty, "close: slide down from the handle before recording → editor, no change")
        editorTool("voiceOver")
        _ = element("voiceRecordSheet").waitForExistence(timeout: 5)
        element("voiceRecordSheet").swipeDown(velocity: .fast)
        check(wait(3) { !sheetOpen() } && voices().isEmpty, "close: swipe down before recording")
        editorTool("voiceOver")
        _ = element("voiceRecordSheet").waitForExistence(timeout: 5)
        app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.25)).tap()
        check(wait(3) { !sheetOpen() } && voices().isEmpty, "close: tap outside before recording")
        editorTool("voiceOver")
        _ = element("voiceRecordSheet").waitForExistence(timeout: 5)
        element("voiceRecord").tap()
        // a quick drag from the handle (top of the sheet) while 3·2·1 runs
        let sheetTop = element("voiceRecordSheet").frame.minY
        let from = app.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: 195, dy: sheetTop + 14))
        from.press(forDuration: 0.02, thenDragTo: from.withOffset(CGVector(dx: 0, dy: 260)), withVelocity: .fast, thenHoldForDuration: 0)
        pause(4)
        let late = app.alerts["Discard recording?"]
        if late.exists { log("the drag landed after 3·2·1 had ended (recording had started): Discard asked"); late.buttons["Discard"].tap(); pause(1) }
        check(!sheetOpen() && !element("liveWave").exists && voices().isEmpty, "sliding down cancels the 3·2·1 countdown (nothing recorded)")
        editorTool("voiceOver")
        _ = element("voiceRecordSheet").waitForExistence(timeout: 5)
        element("voiceRecord").tap()
        _ = element("liveWave").waitForExistence(timeout: 6)
        pause(1.5)
        element("voiceRecordSheet").swipeDown(velocity: .fast)
        let discard = app.alerts["Discard recording?"]
        check(discard.waitForExistence(timeout: 3), "closing mid-recording asks Discard recording?")
        shot("vo_discard_alert")
        discard.buttons["Keep recording"].tap(); pause(1)
        check(element("liveWave").exists, "Keep recording: still recording")
        element("voiceFinish").tap()
        _ = element("voiceReviewSheet").waitForExistence(timeout: 5)
        element("voiceReviewSheet").swipeDown(velocity: .fast)
        check(discard.waitForExistence(timeout: 3), "closing with an unsaved take asks Discard recording?")
        discard.buttons["Discard"].tap()
        check(wait(3) { !sheetOpen() } && voices().isEmpty, "Discard: back to the editor, nothing saved")

        // (a) record in 2 steps: countdown 3·2·1, Enhance voice, Stop → Play / Retake / Save
        editorTool("voiceOver")
        check(element("voiceRecordSheet").waitForExistence(timeout: 10), "a) step 1: Recording voice-over sheet")
        check(element("enhanceVoice").exists, "a) Enhance voice switch")
        shot("vo_step1")
        element("voiceRecord").tap()
        check(element("voiceCountdown").waitForExistence(timeout: 3), "a) countdown 3·2·1 over the video")
        pause(0.4)
        shot("vo_countdown")
        check(element("liveWave").waitForExistence(timeout: 6), "a) recording (live wave)")
        pause(2)
        element("voicePause").tap(); pause(1); element("voicePause").tap()      // pause / resume
        pause(2)
        shot("vo_recording")
        element("voiceFinish").tap()
        check(element("voiceReviewSheet").waitForExistence(timeout: 5), "a) step 2: Play / Retake / Save")
        check(element("voicePlay").exists && element("voiceRetake").exists && element("voiceSave").exists, "a) Play · Retake · Save")
        element("voicePlay").tap(); pause(1.5)
        shot("vo_step2")
        element("voiceRetake").tap()
        check(element("voiceRecordSheet").waitForExistence(timeout: 5), "a) Retake back to step 1")
        element("voiceRecord").tap()
        _ = element("liveWave").waitForExistence(timeout: 6)
        pause(5)
        element("voiceFinish").tap()
        _ = element("voiceSave").waitForExistence(timeout: 5)
        element("voiceSave").tap()
        check(wait(30) { self.voices().count == 1 }, "a) Save puts the voice on the timeline", editorState())
        _ = wait(60) { self.element("voiceBlock").exists }
        shot("vo_saved")
        log("saved \(voices())")

        // (c) panel of the selected voice-over (Figma 1119:1634)
        element("voiceBlock").tap(); pause(1)
        check(element("voiceClipPanel").waitForExistence(timeout: 5), "c) voice-over panel")
        for id in ["lowerVideoSound", "cleanUpToggle", "recordAgain", "panelClose", "voiceDelete"] {
            check(element(id).exists, "c) \(id)")
        }
        element("lowerVideoSound").tap(); pause(1)
        check(voices().first?.contains("d") == true, "c) Lower video sound on", voices().joined(separator: ","))
        shot("vo_panel")

        // (b) timeline: move, trim, split, delete, undo
        let before = voices().first ?? ""
        let block = element("voiceBlock")
        let start = block.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        start.press(forDuration: 0.6, thenDragTo: start.withOffset(CGVector(dx: 60, dy: 0)))
        pause(1.5)
        let moved = voices().first ?? ""
        check(moved != before && moved.split(separator: "+").first != before.split(separator: "+").first, "b) moved by dragging", "\(before) → \(moved)")
        // the voice now runs past the right side of the screen: its START edge is the one in view
        if !element("audioTrimStart").exists { element("voiceBlock").tap(); pause(0.8) }
        let handle = element("audioTrimStart")
        if handle.exists {
            let h = handle.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
            h.press(forDuration: 0.1, thenDragTo: h.withOffset(CGVector(dx: 30, dy: 0)))
            pause(1.5)
        }
        let trimmed = voices().first ?? ""
        check(handle.exists && trimmed != moved, "b) edge trimmed (start edge dragged in)", "\(moved) → \(trimmed)")
        shot("vo_moved_trimmed")
        // cursor into the middle of the voice, then Split
        let parts = trimmed.split(separator: "+")
        let vStart = Double(parts.first ?? "0") ?? 0
        let vLen = Double(parts.count > 1 ? String(parts[1].prefix { "0123456789.".contains($0) }) : "0") ?? 0
        element("b18Start").tap(); pause(0.5)
        let target = vStart + vLen / 2
        for _ in 0..<Int(target) { element("b17Step").tap(); pause(0.25) }
        if target - Double(Int(target)) >= 0.5 { element("b18Half").tap(); pause(0.3) }
        if !(editorState().contains("sel=voice")) { element("voiceBlock").tap(); pause(0.8) }
        editorTool("split")
        check(wait(10) { self.voices().count == 2 }, "b) Split cuts the voice in two", voices().joined(separator: ","))
        shot("vo_split")
        let pieces = app.descendants(matching: .any).matching(identifier: "voiceBlock")
        if pieces.count > 1 { pieces.element(boundBy: 1).tap(); pause(0.8) }
        if element("voiceDelete").exists { element("voiceDelete").tap() } else { editorTool("delete") }
        check(wait(10) { self.voices().count == 1 }, "b) a piece deleted", voices().joined(separator: ","))
        element("editorUndo").tap()
        check(wait(10) { self.voices().count == 2 }, "b) Undo brings it back", voices().joined(separator: ","))
        shot("vo_undo")

        // (c) Record again on the first piece
        app.descendants(matching: .any).matching(identifier: "voiceBlock").element(boundBy: 0).tap(); pause(1)
        if element("recordAgain").waitForExistence(timeout: 5) {
            element("recordAgain").tap()
            if element("voiceRecord").waitForExistence(timeout: 5) {
                element("voiceRecord").tap()
                _ = element("liveWave").waitForExistence(timeout: 6)
                pause(3)
                element("voiceFinish").tap()
                _ = element("voiceSave").waitForExistence(timeout: 5)
                element("voiceSave").tap()
            }
        }
        check(wait(30) { self.voices().count == 2 }, "c) Record again replaces that voice (still 2)", voices().joined(separator: ","))
        log("before export \(voices())")
        // make sure every piece lowers the video sound for the export check
        for i in 0..<app.descendants(matching: .any).matching(identifier: "voiceBlock").count {
            app.descendants(matching: .any).matching(identifier: "voiceBlock").element(boundBy: i).tap(); pause(0.8)
            if let v = voices().sorted().dropFirst(i).first, !v.contains("d"), element("lowerVideoSound").exists {
                element("lowerVideoSound").tap(); pause(0.8)
            }
        }
        log("export with \(voices())")
        shot("vo_before_export")

        // (d) export with the voice
        _ = waitExport("after")
        log("end \(editorState())")
        done()
    }

    /// Plan · Day → Post to TikTok: video to Photos, caption copied with the notice, TikTok (here its App Store page,
    /// the simulator has no TikTok) → back in CutSell "Did you post it?" → Yes, posted.
    // MARK: Keyframes like CapCut on a real cut: ◇, framing anywhere makes a ◇, ‹ ›, curve, smooth, centred line

    private func stateField(_ name: String) -> String? {
        // whole field names only ("zoom=" must not read "auto_zoom=", "t=" not "user_rot=")
        for part in editorState().split(separator: " ") where part.hasPrefix(name + "=") {
            return String(part.dropFirst(name.count + 1))
        }
        return nil
    }
    private func stateNumber(_ name: String) -> Double? { stateField(name).flatMap(Double.init) }
    private func seekTo(_ seconds: Double) {
        element("b18Start").tap(); pause(0.4)
        for _ in 0..<Int(seconds) { element("b17Step").tap(); pause(0.2) }
        if seconds - Double(Int(seconds)) >= 0.5 { element("b18Half").tap(); pause(0.3) }
        pause(0.4)
    }

    func testKeyframesCapCut() throws {
        app = XCUIApplication()
        app.launchArguments = ["-CutSellScreen", "editor_probe"]
        let env = ProcessInfo.processInfo.environment
        for key in ["CUTSELL_PROBE_TOKEN", "CUTSELL_PROBE_USER", "CUTSELL_PROBE_PROJECT", "CUTSELL_PROBE_API"] {
            if let value = env[key] { app.launchEnvironment[key] = value }
        }
        app.launchEnvironment["CUTSELL_PROBE_STATE"] = "b18_kf"
        app.launch()
        check(element("timelineV4").waitForExistence(timeout: 120), "editor open on a real cut")
        _ = wait(90) { self.element("editorPlay").exists && self.element("editorPlay").isEnabled }
        pause(4)
        log("start \(editorState())")
        // The cut's own Auto Zoom AI ◇ (the engine marks key moments): how many, before anything else.
        log("auto zoom ◇ from the engine: ai=\(stateField("ai") ?? "?") auto_zoom=\(stateField("auto_zoom") ?? "?")")
        if stateField("auto_zoom") == "true" { element("b18AutoZoom").tap() }   // AI zooms off: only her ◇ move the picture
        pause(1)
        shot("kf_open")

        // 1-2. clip under the line, ◇ keeps the current framing (100 %)
        seekTo(0.5)
        element("editorKeyframe").tap()
        check(wait(8) { self.stateField("user") == "1" }, "2) ◇ adds a keyframe at the line (100 %)", editorState())
        check(stateField("key_btn") == "delete", "5) on a ◇ the button removes it")
        // 3. move the line, pinch: the keyframe is made there by itself (no ◇ tap)
        seekTo(2.0)
        check(stateField("key_btn") == "add", "off a ◇ the button adds")
        element("editorVideo").pinch(withScale: 1.32, velocity: 1.0)
        check(wait(8) { self.stateField("user") == "2" }, "3) pinch away from a ◇ makes a keyframe there", editorState())
        let peak = stateNumber("zoom") ?? 1
        check(peak > 1.15, "second keyframe zoomed in", String(format: "zoom %.3f", peak))
        seekTo(3.5)
        element("editorVideo").pinch(withScale: 1 / 1.6, velocity: -1.0)
        check(wait(8) { self.stateField("user") == "3" }, "third keyframe back out by pinching", editorState())
        log("keyframes \(stateField("user_scales") ?? "?")")
        shot("kf_three")

        // 4 + 8. the framing between and outside the ◇ (what the preview and the export show)
        seekTo(0)
        var samples: [(Double, Double)] = []
        for i in 0..<9 {
            if i > 0 { element("b18Half").tap(); pause(0.35) }
            samples.append((stateNumber("t") ?? -1, stateNumber("zoom") ?? -1))
        }
        let text = samples.map { String(format: "%.1f:%.3f", $0.0, $0.1) }.joined(separator: " ")
        let zooms = samples.map(\.1)
        let top = zooms.firstIndex(of: zooms.max() ?? 0) ?? 0
        let steps = zip(zooms, zooms.dropFirst()).map { $1 - $0 }
        check(abs(zooms[0] - zooms[1]) < 0.005, "8) held before the first ◇", text)
        check(steps.prefix(top).allSatisfy { $0 >= -0.003 } && steps.dropFirst(top).allSatisfy { $0 <= 0.003 }
              && (steps.map(abs).max() ?? 1) < 0.2 && zooms.max()! > 1.15,
              "4) smooth in then out (ease in-out), never a jump", text)

        // 6. ‹ › jump between the ◇
        seekTo(3.0)
        element("editorKeyPrev").tap(); pause(0.6)
        let prevT = stateNumber("t") ?? -1
        element("editorKeyPrev").tap(); pause(0.6)
        let prev2 = stateNumber("t") ?? -1
        element("editorKeyNext").tap(); pause(0.6)
        let nextT = stateNumber("t") ?? -1
        check(abs(prevT - 2.0) < 0.15 && abs(prev2 - 0.5) < 0.15 && abs(nextT - 2.0) < 0.15, "6) ‹ › go to the previous / next ◇",
              String(format: "%.2f %.2f %.2f", prevT, prev2, nextT))

        // 7. tap a ◇ on the timeline: curve
        let diamonds = app.descendants(matching: .any).matching(identifier: "userKeyframe")
        if diamonds.count > 1 { diamonds.element(boundBy: 1).tap(); pause(0.8) }
        check(element("keyCurveBar").waitForExistence(timeout: 4), "7) tapping a ◇ shows Linear / Ease in / Ease out / Ease in-out")
        shot("kf_curve")
        element("keyCurve_linear").tap(); pause(0.8)
        check((stateField("curves") ?? "").contains("linear"), "7) curve saved on that ◇", stateField("curves") ?? "")
        element("keyCurve_ease_in_out").tap(); pause(0.8)

        // 10. playback: the line stays in the middle, the timeline slides under it, the zoom moves
        seekTo(0)
        log("PLAY_START epoch=\(Date().timeIntervalSince1970)")
        let before = stateNumber("t") ?? -1
        element("editorPlay").tap()
        pause(2)
        let during = stateNumber("t") ?? -1
        if stateField("playing") == "true" { element("editorPlay").tap() }
        log("PLAY_END epoch=\(Date().timeIntervalSince1970)")
        // (XCUITest waits for the app to idle while a video plays, so the screen recording shows the motion;
        // here: the line moved on from where it started)
        check(during > before + 0.2, "10) the line advances while playing", String(format: "%.2f -> %.2f", before, during))
        shot("kf_after_play")

        // Export (measured in the workflow: the square's size frame by frame)
        _ = waitExport("keyframes")
        // 9. Auto Zoom AI back on: its ◇ (in, hold, out) next to hers
        element("b18AutoZoom").tap(); pause(1.5)
        log("auto zoom on: ai=\(stateField("ai") ?? "?") auto_zoom=\(stateField("auto_zoom") ?? "?")")
        log("end \(editorState())")
        done()
    }

    func testPostToTikTok() throws {
        launchApp(fresh: true, extra: ["-CutSellPlanMock"])
        element("tab_plan").tap()
        let card = element("planMyWeek")
        check(card.waitForExistence(timeout: 20), "Plan card")
        card.tap()
        check(element("goalBuild").waitForExistence(timeout: 10), "Goal")
        element("goalBuild").tap()
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        if springboard.buttons["Allow"].waitForExistence(timeout: 5) { springboard.buttons["Allow"].tap() }
        let firstDay = app.descendants(matching: .any).matching(NSPredicate(format: "identifier BEGINSWITH 'weekDay_'")).firstMatch
        check(firstDay.waitForExistence(timeout: 15), "week built")
        firstDay.tap()
        let post = element("postToTikTok")
        check(post.waitForExistence(timeout: 10), "Day: big Post to TikTok button")
        check(element("dayShare").exists && element("dayShare").label.contains("Share"), "Share below it")
        shot("day_post_to_tiktok")
        post.tap()
        allowPhotos()
        let copied = app.staticTexts["Caption copied, paste it in TikTok"]
        check(copied.waitForExistence(timeout: 60), "Caption copied, paste it in TikTok")
        shot("day_caption_copied")
        // TikTok (or its App Store page) takes over; then she comes back to CutSell.
        _ = wait(20) { self.app.state != .runningForeground }
        log("left CutSell for TikTok / App Store: \(app.state != .runningForeground)")
        pause(2)
        app.activate()
        let ask = app.alerts["Did you post it?"]
        check(ask.waitForExistence(timeout: 15), "back from TikTok: Did you post it?")
        shot("did_you_post_it")
        check(ask.buttons["Yes, posted"].exists && ask.buttons["Not yet"].exists, "Yes, posted / Not yet")
        ask.buttons["Yes, posted"].tap()
        check(wait(10) { self.element("dayPosted").label.contains("Posted") }, "Yes, posted marks it posted", element("dayPosted").label)
        shot("day_posted")
        done()
    }

    /// Her real library (plan computed read-only on the server with the new rules), shown by the app.
    func testPlanRealData() throws {
        launchApp(fresh: true, extra: ["-CutSellPlanMock"])
        check(ProcessInfo.processInfo.environment["CUTSELL_PLAN_FIXTURE"] != nil, "her plan data given to the simulator")
        element("tab_plan").tap()
        let card = element("planMyWeek")
        check(card.waitForExistence(timeout: 20), "Plan card")
        card.tap()
        check(element("goalBuild").waitForExistence(timeout: 10), "Goal v2")
        pause(1)
        log("goal: count \(element("goalCount").label) · \(element("goalSummary").label) · \(element("goalTimesHelp").label)")
        shot("real_goal")
        element("goalBuild").tap()
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        if springboard.buttons["Allow"].waitForExistence(timeout: 5) { springboard.buttons["Allow"].tap() }
        let subtitle = element("weekSubtitle")
        check(subtitle.waitForExistence(timeout: 15), "her plan built", subtitle.label)
        pause(1)
        shot("real_week")
        app.swipeUp(); pause(1)
        shot("real_week_more")
        app.swipeDown(); pause(1)
        let first = app.descendants(matching: .any).matching(NSPredicate(format: "identifier BEGINSWITH 'weekDay_'")).firstMatch
        if first.exists { first.tap(); pause(2); shot("real_day") }
        done()
    }

    func testPersonalMenu() throws {
        launchApp(fresh: true, extra: ["-CutSellPlanMock", "-CutSellProjectsMock"])
        element("tab_edits").tap()
        let options = app.buttons.matching(identifier: "editTileMenuReady")
        check(options.firstMatch.waitForExistence(timeout: 30), "a Ready video's ••• in Edits",
              "\(options.count) · all tiles \(app.descendants(matching: .any).matching(identifier: "editTileMenu").count)")
        guard options.count > 0 else { return done() }
        // the first Ready video: its menu has "Personal · don't plan"
        var index = 0
        var found = false
        while index < min(options.count, 12) {
            options.element(boundBy: index).tap()
            if app.buttons["Personal · don't plan"].waitForExistence(timeout: 3) { found = true; break }
            app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.04)).tap(); pause(0.5)
            index += 1
        }
        if !found { shot("personal_menu_missing"); log("on screen: \(screenTexts()) · buttons: \(app.buttons.allElementsBoundByIndex.prefix(25).map(\.label))") }
        check(found, "••• shows Personal · don't plan")
        guard found else { return done() }
        shot("personal_menu")
        app.buttons["Personal · don't plan"].tap()
        pause(1.5)
        options.element(boundBy: index).tap()
        check(app.buttons["Use in Plan"].waitForExistence(timeout: 5), "after marking it: Use in Plan")
        shot("personal_menu_marked")
        app.buttons["Use in Plan"].tap()
        pause(1.5)
        options.element(boundBy: index).tap()
        check(app.buttons["Personal · don't plan"].waitForExistence(timeout: 5), "back in Plan: Personal · don't plan")
        app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.04)).tap()
        // the video is still in Edits after being marked personal
        check(options.count > index, "personal videos stay in Edits")
        done()
    }

    private func done() {
        log("FAILURES: \(failures.isEmpty ? "none" : failures.joined(separator: " | "))")
        XCTAssertTrue(failures.isEmpty, failures.joined(separator: " | "))
    }

    func testPlanWeekFlow() throws {
        launchApp(fresh: true, extra: ["-CutSellPlanMock", "-CutSellPlanTestReminder"])
        element("tab_plan").tap()
        let card = element("planMyWeek")
        check(card.waitForExistence(timeout: 20), "Plan · This week card")
        check(!element("planSoon").exists, "no SOON any more")
        card.tap()
        check(element("goalBuild").waitForExistence(timeout: 10), "Plan my week · Goal")
        // Goal v2 (Figma 1140:2773): how many videos, free times, days, start, remind
        let count = element("goalCount")
        check(count.label == "9", "HOW MANY VIDEOS starts at all usable", count.label)
        element("goalMinus").tap(); pause(0.3)
        check(count.label == "8", "− takes one away", count.label)
        element("goalPlus").tap(); pause(0.3)
        check(count.label == "9", "+ adds one (never above the usable videos)", count.label)
        element("goalPlus").tap(); pause(0.3)
        check(count.label == "9", "+ stops at all usable", count.label)
        check(!element("goalPerDay_2").exists && !element("goalReminder_2").exists, "no 1/2/3/4+ and no Morning/Evening/Night")
        element("goalAddTime").tap()
        if element("timeSave").waitForExistence(timeout: 5) { element("timeSave").tap() }
        pause(1)
        check(element("goalTimesHelp").label.hasPrefix("3 times = 3 videos per day"), "+ Time adds a time", element("goalTimesHelp").label)
        shot("plan_goal_three_times")
        element("goalTime_0").tap()
        check(element("timeDelete").waitForExistence(timeout: 5), "a time can be changed or deleted")
        element("timeDelete").tap()
        pause(1)
        check(element("goalTimesHelp").label.hasPrefix("2 times = 2 videos per day"), "time deleted", element("goalTimesHelp").label)
        element("goalDay_6").tap()                            // Sunday on
        check(element("goalStart").label.contains("Tomorrow"), "START: Tomorrow", element("goalStart").label)
        shot("plan_goal")
        check(element("goalSummary").label.contains("9 videos · 2 per day"), "live summary N videos · X per day · Y days",
              element("goalSummary").label)
        element("goalBuild").tap()
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let allow = springboard.buttons["Allow"]
        if allow.waitForExistence(timeout: 5) { allow.tap() }
        let subtitle = element("weekSubtitle")
        check(subtitle.waitForExistence(timeout: 15), "Week built")
        check(subtitle.label.hasPrefix("9 videos"), "9 videos in the week", subtitle.label)
        check(!element("weekEndsEarly").exists, "all 9 fit (no ends-early note)")
        shot("plan_week")

        // Move one video (long press → Move).
        let slots = app.descendants(matching: .any).matching(identifier: "weekSlot")
        if slots.count > 2 {
            slots.element(boundBy: 2).press(forDuration: 1.2)
            let moveItem = app.buttons["Move to another day or time"].firstMatch
            if moveItem.waitForExistence(timeout: 5) {
                moveItem.tap()
                check(element("moveSave").waitForExistence(timeout: 5), "move sheet")
                element("moveSave").tap()
                pause(1.5)
            }
        }
        check(element("weekSubtitle").label.hasPrefix("9 videos"), "still 9 videos after the move")

        // The first day.
        let firstDay = app.descendants(matching: .any).matching(NSPredicate(format: "identifier BEGINSWITH 'weekDay_'")).firstMatch
        firstDay.tap()
        check(element("dayShare").waitForExistence(timeout: 10), "Day")
        check(element("dayCaption").label.contains("#tiktokshop"), "suggested caption", element("dayCaption").label)
        shot("plan_day")
        element("captionEdit").tap()
        let field = app.alerts.textFields.firstMatch
        if field.waitForExistence(timeout: 5) {
            field.tap(); field.clearAndType("My own caption #mine")
            app.alerts.buttons["Save"].firstMatch.tap()
        }
        check(wait(5) { self.element("dayCaption").label.hasPrefix("My own caption #mine") }, "her caption saved", element("dayCaption").label)
        element("captionCopy").tap()
        check(element("dayNotice").label == "Caption copied", "Copy")
        element("dayShare").tap()
        let sheet = app.otherElements["ActivityListView"]
        check(wait(30) { sheet.exists || self.app.buttons["Save Video"].exists }, "Share opens the share sheet with the video")
        shot("plan_share")
        // Close the share sheet (its × or a swipe down) until the Day is usable again.
        for _ in 0..<4 where !element("dayPosted").isHittable {
            let close = app.descendants(matching: .button).matching(NSPredicate(format: "label == 'Close' OR identifier == 'Close'")).firstMatch
            if close.exists { close.tap() } else {
                app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.35))
                    .press(forDuration: 0.1, thenDragTo: app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.98)))
            }
            pause(1.5)
        }
        element("dayPosted").tap()
        check(wait(5) { self.element("dayPosted").label.contains("Posted") }, "Mark as posted", element("dayPosted").label)

        // The reminder (6 s after the day opened): Home, then tap it.
        XCUIDevice.shared.press(.home)
        let banner = springboard.descendants(matching: .any).matching(NSPredicate(format: "label CONTAINS 'Time to post'")).firstMatch
        let shown = banner.waitForExistence(timeout: 45)
        check(shown, "\"Time to post\" reminder")
        if shown { banner.tap() } else { app.activate() }
        check(element("dayTitle").waitForExistence(timeout: 15), "the reminder opens its day")
        shot("plan_reminder_opened")

        // Build again: the posted video stays posted.
        element("planBack").tap()
        if element("weekSettings").waitForExistence(timeout: 10) { element("weekSettings").tap() }
        if element("goalBuild").waitForExistence(timeout: 10) { element("goalBuild").tap() }
        check(element("weekSubtitle").waitForExistence(timeout: 15), "week built again")
        firstDay.tap()
        check(wait(10) { self.element("dayPosted").label.contains("Posted") || self.app.staticTexts["Posted"].exists },
              "posted stays posted after rebuilding")
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
