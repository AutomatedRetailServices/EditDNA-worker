import XCTest

/// Build 21: the upload in REAL conditions. Swanny's iPhone (build 19) uploaded at ~1 MB/s and the
/// app cancelled its own parts after 30 s. Here the Mac's network is limited to ~1 MB/s (dummynet,
/// set by the workflow), the video is over 100 MB, and the creator LEAVES CutSell (Home) in the
/// middle of the upload for 25 s, then comes back. Pass: the upload finishes with no "Upload paused"
/// and no "No data sent" notice, the % never goes back to 0, the cut is processed and opens in the
/// editor. Frames and a log go to B21_OUT; no video ever leaves the test machine.
final class Build21UITests: XCTestCase {
    private var app: XCUIApplication!
    private lazy var out: URL = {
        let path = ProcessInfo.processInfo.environment["B21_OUT"] ?? NSTemporaryDirectory()
        let url = URL(fileURLWithPath: path, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()
    private let started = Date()
    private var step = 0
    private var failures: [String] = []

    override func setUp() { continueAfterFailure = true }

    private func log(_ text: String) {
        let line = "B21UI \(String(format: "%7.1f", Date().timeIntervalSince(started))) \(text)\n"
        print(line, terminator: "")
        let file = out.appendingPathComponent("b21_ui_log.txt")
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
            .write(to: out.appendingPathComponent(String(format: "b21_%02d_%@.png", step, name)))
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

    func testThrottledUploadLeavingTheApp() throws {
        launch()
        let upload = element("hubUpload")
        guard upload.waitForExistence(timeout: 60) else { check(false, "Create screen"); return }
        upload.tap()
        allowPhotos()
        let tiles = app.descendants(matching: .any).matching(identifier: "galleryTile")
        _ = wait(30) { tiles.count > 0 }
        log("gallery shows \(tiles.count) videos")
        guard tiles.count > 0 else { check(false, "video in the gallery"); shot("gallery"); return }
        tiles.element(boundBy: 0).tap()
        pause(0.8)
        element("galleryCreateButton").tap()
        let style = element("styleCreateButton")
        guard style.waitForExistence(timeout: 10) else { check(false, "style sheet"); return }
        style.tap()
        let tappedAt = Date()
        if wait(20, until: { self.element("aiConsentAgree").exists || self.app.buttons["Agree"].exists }) {
            (element("aiConsentAgree").exists ? element("aiConsentAgree") : app.buttons["Agree"]).tap()
            log("AI notice agreed")
        }

        var seen: [String] = []
        var last = ""
        var left = false
        var uploadStarted: Date?
        var uploadEnded: Date?
        var lastUploadPercent = -1
        var wentBack = false
        var stalledSeen = false
        var problem = ""
        var ready = false
        var lastChange = Date()
        let deadline = Date().addingTimeInterval(40 * 60)
        while Date() < deadline {
            if Date().timeIntervalSince(lastChange) > 300 { problem = "no change for 5 min: \(screenTexts())"; shot("stuck"); break }
            if element("readyReviewButton").exists { ready = true; break }
            if element("uploadStalled").exists && !stalledSeen {
                stalledSeen = true
                log("STALL NOTICE: \(element("uploadStalled").label)")
                shot("stalled")
            }
            if app.staticTexts["Upload paused"].exists || app.staticTexts["Couldn’t start the cut"].exists {
                problem = screenTexts(); shot("error"); break
            }
            let stage = element("processingStage")
            let percentElement = element("processingPercent")
            if stage.exists {
                let percentText = percentElement.exists ? percentElement.label : ""
                let now = "\(stage.label) \(percentText)"
                if now != last {
                    last = now
                    lastChange = Date()
                    seen.append(String(format: "%.0fs %@", Date().timeIntervalSince(tappedAt), now))
                }
                if stage.label.contains("Uploading") {
                    if uploadStarted == nil { uploadStarted = Date(); shot("uploading") }
                    let value = Int(percentText.replacingOccurrences(of: "%", with: "")) ?? -1
                    if value >= 0 {
                        if value < lastUploadPercent { wentBack = true; log("% went back: \(lastUploadPercent) → \(value)") }
                        lastUploadPercent = max(lastUploadPercent, value)
                    }
                    // In the middle of the upload the creator leaves CutSell for 25 s.
                    if !left && value >= 15 {
                        left = true
                        log("leaving CutSell at \(value)%")
                        XCUIDevice.shared.press(.home)
                        pause(25)
                        app.activate()
                        pause(3)
                        let back = element("processingPercent").exists ? element("processingPercent").label : "?"
                        log("back in CutSell: \(element("processingStage").exists ? element("processingStage").label : "?") \(back)")
                        shot("back_in_app")
                        lastChange = Date()
                    }
                } else if uploadStarted != nil && uploadEnded == nil {
                    uploadEnded = Date()
                }
            }
            pause(1.5)
        }
        log("screen steps: \(seen.joined(separator: " → "))")
        let uploadSeconds = uploadStarted.map { (uploadEnded ?? Date()).timeIntervalSince($0) } ?? -1
        check(left, "left CutSell mid-upload and came back")
        check(uploadEnded != nil, "upload finished", String(format: "%.0f s uploading (limited to ~1 MB/s)", uploadSeconds))
        check(!stalledSeen, "no 'No data sent' notice")
        check(!wentBack, "upload % never went back")
        check(problem.isEmpty, "no error screen", problem)
        check(ready, "processed", String(format: "%.0f s from Create", Date().timeIntervalSince(tappedAt)))
        shot("end")
        if ready {
            element("readyReviewButton").tap()
            let editor = element("timelineV4").waitForExistence(timeout: 120)
            pause(3)
            shot("editor")
            check(editor, "opens in the editor")
        }
        log("FAILURES: \(failures.isEmpty ? "none" : failures.joined(separator: " | "))")
    }
}
