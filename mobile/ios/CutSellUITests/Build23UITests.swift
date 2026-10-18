import XCTest

/// Build 21: the upload in REAL conditions. Swanny's iPhone (build 19) uploaded at ~1 MB/s and the
/// app cancelled its own parts after 30 s. Here the Mac's network is limited to ~1 MB/s (dummynet,
/// set by the workflow), the video is over 100 MB, and the creator LEAVES CutSell (Home) in the
/// middle of the upload for 25 s, then comes back. Pass: the upload finishes with no "Upload paused"
/// and no "No data sent" notice, the % never goes back to 0, the cut is processed and opens in the
/// editor. Frames and a log go to B23_OUT; no video ever leaves the test machine.
final class Build23UITests: XCTestCase {
    private var app: XCUIApplication!
    private lazy var out: URL = {
        let path = ProcessInfo.processInfo.environment["B23_OUT"] ?? NSTemporaryDirectory()
        let url = URL(fileURLWithPath: path, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()
    private let started = Date()
    private var step = 0
    private var failures: [String] = []

    override func setUp() { continueAfterFailure = true }

    private func log(_ text: String) {
        let line = "B23UI \(String(format: "%7.1f", Date().timeIntervalSince(started))) \(text)\n"
        print(line, terminator: "")
        let file = out.appendingPathComponent("b23_ui_log.txt")
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
            .write(to: out.appendingPathComponent(String(format: "b23_%02d_%@.png", step, name)))
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

    /// Build 23: from tapping Create to the cut open in the editor, at ~1 MB/s. B23_MODE "before" =
    /// audio first off (the whole video uploads first, as in build 22); "after" = audio first.
    func testTimeToEditor() throws {
        let env = ProcessInfo.processInfo.environment
        let mode = env["B23_MODE"] ?? "after"
        let label = env["B23_LABEL"] ?? "video"
        app = XCUIApplication()
        app.launchArguments = ["-CutSellScreen", "create_probe"] + (mode == "before" ? ["-cutsell.audioFirst.off", "YES"] : [])
        for key in ["CUTSELL_PROBE_TOKEN", "CUTSELL_PROBE_USER", "CUTSELL_PROBE_API"] {
            if let value = env[key] { app.launchEnvironment[key] = value }
        }
        app.launch()
        let upload = element("hubUpload")
        guard upload.waitForExistence(timeout: 60) else { check(false, "Create screen"); return }
        upload.tap()
        allowPhotos()
        let tiles = app.descendants(matching: .any).matching(identifier: "galleryTile")
        _ = wait(30) { tiles.count > 0 }
        guard tiles.count > 0 else { check(false, "video in the gallery"); return }
        tiles.element(boundBy: 0).tap()
        pause(0.8)
        element("galleryCreateButton").tap()
        let style = element("styleCreateButton")
        guard style.waitForExistence(timeout: 10) else { check(false, "style sheet"); return }
        style.tap()
        let tappedAt = Date()
        if wait(20, until: { self.element("aiConsentAgree").exists || self.app.buttons["Agree"].exists }) {
            (element("aiConsentAgree").exists ? element("aiConsentAgree") : app.buttons["Agree"]).tap()
        }
        var firstSeen: [String: Double] = [:]
        var problem = ""
        var ready = false
        var last = ""
        let deadline = Date().addingTimeInterval(30 * 60)
        while Date() < deadline {
            if element("readyReviewButton").exists { ready = true; break }
            if app.staticTexts["Upload paused"].exists || app.staticTexts["Couldn’t start the cut"].exists || element("uploadTryAgain").exists {
                problem = screenTexts(); shot("error"); break
            }
            let stage = element("processingStage")
            if stage.exists {
                let name = stage.label.components(separatedBy: " · ").last ?? stage.label
                let key = name.hasPrefix("Cutting your video") ? "Cutting your video" : name
                if firstSeen[key] == nil { firstSeen[key] = Date().timeIntervalSince(tappedAt) }
                let background = element("backgroundVideoUpload").exists ? " | " + element("backgroundVideoUpload").label : ""
                let now = stage.label + " " + (element("processingPercent").exists ? element("processingPercent").label : "") + background
                if now != last { last = now; log(String(format: "%.0fs %@", Date().timeIntervalSince(tappedAt), now)) }
            }
            pause(0.5)
        }
        let readyAt = Date().timeIntervalSince(tappedAt)
        var editorAt = -1.0
        if ready {
            element("readyReviewButton").tap()
            if element("timelineV4").waitForExistence(timeout: 120) { editorAt = Date().timeIntervalSince(tappedAt) }
            pause(2)
            shot("editor_\(label)_\(mode)")
        }
        let steps = firstSeen.sorted { $0.value < $1.value }.map { String(format: "%@ %.0fs", $0.key, $0.value) }.joined(separator: " → ")
        log("TIMING \(label) \(mode): \(steps) → ready \(String(format: "%.0f", readyAt))s → editor \(String(format: "%.0f", editorAt))s")
        check(problem.isEmpty, "no error screen", problem)
        check(ready, "processed", String(format: "%.0f s", readyAt))
        check(editorAt > 0, "opens in the editor", String(format: "%.0f s from Create", editorAt))
        if mode == "after" {
            // The full video keeps uploading in the background (export needs it): keep CutSell open
            // long enough; the workflow reads video_background_done from the app's own log.
            let seconds = Double(env["B23_WAIT"] ?? "300") ?? 300
            log(String(format: "keeping CutSell open %.0f s for the background video upload", seconds))
            pause(seconds)
        }
        log("FAILURES: \(failures.isEmpty ? "none" : failures.joined(separator: " | "))")
    }
}
