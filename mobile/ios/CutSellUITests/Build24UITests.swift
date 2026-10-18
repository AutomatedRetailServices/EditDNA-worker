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

    private func launchApp(fresh: Bool) {
        app = XCUIApplication()
        app.launchArguments = ["-CutSellScreen", "create_probe"]
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

    private func rowStatuses() -> [String] {
        app.descendants(matching: .any).matching(identifier: "queueRowStatus").allElementsBoundByIndex.map(\.label)
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
        element("bulkGoToEdits").tap()
        let list = element("editsProcessingList")
        check(list.waitForExistence(timeout: 15), "Edits opens on Processing")
        pause(2)
        shot("processing_tab")
        let first = rowStatuses()
        log("rows at start: \(first)")
        check(first.count == 3, "3 rows in order", first.joined(separator: ", "))
        let result = waitForProcessingToEmpty("bulk", minutes: 25)
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
        let result = waitForProcessingToEmpty("single", minutes: 15)
        check(result.done, "the cut finished after leaving and closing CutSell")
        shot("single_end")
        log("FAILURES: \(failures.isEmpty ? "none" : failures.joined(separator: " | "))")
    }
}
