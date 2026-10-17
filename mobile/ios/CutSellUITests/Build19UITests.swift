import XCTest

/// Build 19: a NEW cut from the gallery, the way the creator makes one, on the live server (R2).
/// Builds 16-18 stayed at "Uploading 0 %" forever on a real iPhone: the AI notice was asked under
/// the upload screen and never appeared. Every probe before this one made its cut through the API,
/// so none of them went through that path.
///
/// For each video in the Simulator's library (a short 1080p one and a 4K one over 200 MB, longer
/// than a minute): Upload → pick it → Create → style → (AI notice → Agree, the first time) →
/// Processing, logging every step and % the screen shows → "Review & edit" → the editor opens.
/// Frames and a log go to B19_OUT; no video ever leaves the test machine.
final class Build19UITests: XCTestCase {
    private var app: XCUIApplication!
    private lazy var out: URL = {
        let path = ProcessInfo.processInfo.environment["B19_OUT"] ?? NSTemporaryDirectory()
        let url = URL(fileURLWithPath: path, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()
    private let started = Date()
    private var step = 0
    private var failures: [String] = []

    override func setUp() { continueAfterFailure = true }

    private func log(_ text: String) {
        let line = "B19UI \(String(format: "%7.1f", Date().timeIntervalSince(started))) \(text)\n"
        print(line, terminator: "")
        let file = out.appendingPathComponent("b19_ui_log.txt")
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
            .write(to: out.appendingPathComponent(String(format: "b19_%02d_%@.png", step, name)))
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

    /// One cut from the `index`-th video of the gallery (newest first).
    private func makeCut(index: Int, name: String) {
        launch()
        let upload = element("hubUpload")
        guard upload.waitForExistence(timeout: 60) else { check(false, "\(name): Create screen"); return }
        upload.tap()
        // The system's Photos permission prompt, the first time (answered like the creator would).
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        for _ in 0..<10 {
            let allow = springboard.buttons["Allow Full Access"]
            if allow.exists { allow.tap(); log("\(name): Photos access allowed"); break }
            let inApp = app.buttons["Allow Full Access"]
            if inApp.exists { inApp.tap(); log("\(name): Photos access allowed"); break }
            if element("galleryTile").exists { break }
            pause(1)
        }
        let tiles = app.descendants(matching: .any).matching(identifier: "galleryTile")
        _ = wait(30) { tiles.count > index }
        log("\(name): gallery shows \(tiles.count) videos")
        guard tiles.count > index else { check(false, "\(name): video in the gallery"); shot("\(name)_gallery"); return }
        tiles.element(boundBy: index).tap()
        pause(0.8)
        shot("\(name)_picked")
        element("galleryCreateButton").tap()
        let style = element("styleCreateButton")
        guard style.waitForExistence(timeout: 10) else { check(false, "\(name): style sheet"); return }
        style.tap()
        let tappedAt = Date()
        pause(3)
        shot("\(name)_after_create")
        log("\(name): on screen 3 s after Create: \(screenTexts())")

        // The AI notice must APPEAR (builds 16-18: it was hidden and the cut waited forever).
        let agree = element("aiConsentAgree")
        if agree.waitForExistence(timeout: 15) {
            shot("\(name)_ai_notice")
            log("\(name): AI notice shown after \(String(format: "%.1f", Date().timeIntervalSince(tappedAt))) s")
            agree.tap()
        } else {
            log("\(name): no AI notice (already agreed?) on screen: \(screenTexts())")
            shot("\(name)_no_notice")
        }

        // Follow the Processing screen: every step and % it shows, until the cut is ready.
        var seen: [String] = []
        var last = ""
        var firstUploadPercent: (Double, String)?
        let deadline = Date().addingTimeInterval(30 * 60)
        var ready = false
        var problem = ""
        var lastChange = Date()
        while Date() < deadline {
            // Nothing new on the screen for 4 minutes: record what is there and stop.
            if Date().timeIntervalSince(lastChange) > 240 {
                problem = "no change for 4 min: \(screenTexts())"
                shot("\(name)_stuck")
                break
            }
            if element("readyReviewButton").exists { ready = true; break }
            if app.staticTexts["Upload paused"].exists || app.staticTexts["Couldn’t start the cut"].exists {
                problem = app.staticTexts.allElementsBoundByIndex.map(\.label).joined(separator: " | ")
                break
            }
            let stage = element("processingStage")
            let percent = element("processingPercent")
            if stage.exists {
                let now = "\(stage.label) \(percent.exists ? percent.label : "")"
                if now != last {
                    last = now
                    lastChange = Date()
                    let at = Date().timeIntervalSince(tappedAt)
                    seen.append(String(format: "%.0fs %@", at, now))
                    if firstUploadPercent == nil, stage.label.contains("Uploading"), percent.label != "0%" {
                        firstUploadPercent = (at, now)
                    }
                    if seen.count % 12 == 1 { shot("\(name)_processing") }
                }
            }
            pause(1.5)
        }
        let total = Date().timeIntervalSince(tappedAt)
        log("\(name): screen steps: \(seen.joined(separator: " → "))")
        check(firstUploadPercent != nil, "\(name): upload % moves past 0",
              firstUploadPercent.map { String(format: "first at %.0f s: %@", $0.0, $0.1) } ?? "never")
        check(problem.isEmpty, "\(name): no error screen", problem)
        check(ready, "\(name): uploaded and processed", String(format: "%.0f s from Create", total))
        shot("\(name)_ready")
        guard ready else { return }
        element("readyReviewButton").tap()
        let editor = element("timelineV4").waitForExistence(timeout: 120)
        pause(4)
        shot("\(name)_editor")
        check(editor, "\(name): opens in the editor")
        app.terminate()
    }

    func test1ShortVideo() throws {
        makeCut(index: 0, name: "short")
        log("FAILURES (short): \(failures.isEmpty ? "none" : failures.joined(separator: " | "))")
    }

    func test2Big4KVideo() throws {
        makeCut(index: 1, name: "big4k")
        log("FAILURES (big 4K): \(failures.isEmpty ? "none" : failures.joined(separator: " | "))")
    }
}
