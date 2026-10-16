import XCTest

/// Build 17 on a real cut (live server), with real touches in the Simulator:
/// 1. Typing: add it, cut it (Split), delete a piece, select / hold-and-move / trim the other.
/// 2. Volume per clip: mute one video clip ("Selected clip"), the rest keeps playing ("Whole video").
/// 3. Clean up voice On -> "Cleaning… %" -> clean copy playing -> Off.
/// 4. Auto Zoom AI ◇ on the clips, Style toggle off / on.
/// 5. Two manual keyframes 100 % -> 130 %, then the second deleted with the filled ◇.
/// 6. Export.
/// Every step writes a screenshot (frames only) and a line to b17_ui_log.txt in B17_OUT.
final class Build17UITests: XCTestCase {
    private var app: XCUIApplication!
    private lazy var out: URL = {
        let path = ProcessInfo.processInfo.environment["B17_OUT"] ?? NSTemporaryDirectory()
        let url = URL(fileURLWithPath: path, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()
    private var step = 0

    override func setUp() {
        continueAfterFailure = true
        app = XCUIApplication()
        app.launchArguments = ["-CutSellScreen", "editor_probe"]
        let env = ProcessInfo.processInfo.environment
        for key in ["CUTSELL_PROBE_TOKEN", "CUTSELL_PROBE_USER", "CUTSELL_PROBE_PROJECT", "CUTSELL_PROBE_API"] {
            if let value = env[key] { app.launchEnvironment[key] = value }
        }
        app.launchEnvironment["CUTSELL_PROBE_STATE"] = "b17_ui"
        app.launch()
    }

    // MARK: Helpers

    private func log(_ text: String) {
        let line = "B17UI \(String(format: "%7.1f", Date().timeIntervalSince(started))) \(text)\n"
        print(line, terminator: "")
        let file = out.appendingPathComponent("b17_ui_log.txt")
        if let handle = try? FileHandle(forWritingTo: file) {
            handle.seekToEndOfFile()
            handle.write(line.data(using: .utf8)!)
            try? handle.close()
        } else {
            try? line.data(using: .utf8)!.write(to: file)
        }
    }

    private let started = Date()

    private func shot(_ name: String) {
        step += 1
        let png = XCUIScreen.main.screenshot().pngRepresentation
        try? png.write(to: out.appendingPathComponent(String(format: "b17_%02d_%@.png", step, name)))
    }

    private func element(_ id: String) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: id).firstMatch
    }

    private func count(_ id: String) -> Int {
        app.descendants(matching: .any).matching(identifier: id).count
    }

    @discardableResult
    private func wait(_ seconds: Double, until condition: () -> Bool) -> Bool {
        let end = Date().addingTimeInterval(seconds)
        while Date() < end {
            if condition() { return true }
            RunLoop.current.run(until: Date().addingTimeInterval(0.4))
        }
        return condition()
    }

    private func pause(_ seconds: Double) { RunLoop.current.run(until: Date().addingTimeInterval(seconds)) }

    private var timeline: XCUIElement { element("timelineV4") }

    /// Slide the tracks under the cursor: `seconds` forward (at the default 30 points a second).
    private func scrub(seconds: Double) {
        let from = timeline.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: 220, dy: 28))
        let to = from.withOffset(CGVector(dx: -30 * seconds, dy: 0))
        from.press(forDuration: 0.05, thenDragTo: to)
        pause(0.8)
    }

    /// The video clip under the cursor (tap toggles its selection).
    private func tapClipAtCursor() {
        timeline.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: 62, dy: 62)).tap()
        pause(0.8)
    }

    private func tool(_ name: String) {
        element("tool_\(name)").tap()
        pause(0.8)
    }

    private func soundValues() -> [String] {
        app.descendants(matching: .any).matching(identifier: "soundBlock").allElementsBoundByIndex
            .compactMap { $0.value as? String }
    }

    // MARK: The test

    func testBuild17OnARealCut() throws {
        XCTAssertTrue(timeline.waitForExistence(timeout: 90), "editor did not open")
        XCTAssertTrue(wait(90) { self.element("editorPlay").isEnabled }, "preview not ready")
        pause(6)
        shot("editor_open")
        log("opened zoom_keyframes=\(count("zoomKeyframe")) user_keyframes=\(count("userKeyframe"))")

        // 1. Typing: add, Split, delete a piece, hold-and-move, trim.
        tool("audio")
        XCTAssertTrue(element("audioPanel").waitForExistence(timeout: 10))
        let downloaded = wait(90) { !self.element("soundDownloading").exists }
        log("sounds_downloaded=\(downloaded)")
        shot("audio_panel")
        element("sound_typing").tap()
        pause(1)
        XCTAssertTrue(wait(30) { self.element("addSoundAtPlayhead").isEnabled }, "Add stays disabled")
        element("addSoundAtPlayhead").tap()
        let added = wait(10) { self.count("soundBlock") == 1 }
        log("typing_added=\(added) panel=\(element("soundClipPanel").exists) values=\(soundValues())")
        shot("typing_added_selected")
        scrub(seconds: 1)
        tool("split")
        let split = wait(10) { self.count("soundBlock") == 2 }
        log("typing_split=\(split) values=\(soundValues())")
        shot("typing_split")
        tool("delete")
        let deleted = wait(10) { self.count("soundBlock") == 1 }
        log("typing_piece_deleted=\(deleted) values=\(soundValues())")
        let block = element("soundBlock")
        block.tap()
        pause(1)
        let selected = element("soundClipPanel").waitForExistence(timeout: 5)
        let before = block.value as? String ?? ""
        let start = block.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        start.press(forDuration: 0.6, thenDragTo: start.withOffset(CGVector(dx: 75, dy: 0)))
        pause(1.5)
        let moved = element("soundBlock").value as? String ?? ""
        log("typing_selected=\(selected) move \(before) -> \(moved)")
        shot("typing_moved")
        if !element("audioTrimEnd").exists { element("soundBlock").tap(); pause(0.8) }
        if !element("audioTrimEnd").exists { element("soundBlock").tap(); pause(0.8) }
        let handle = element("audioTrimEnd")
        if handle.exists {
            let h = handle.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
            h.press(forDuration: 0.1, thenDragTo: h.withOffset(CGVector(dx: -24, dy: 0)))
            pause(1.5)
        }
        log("typing_trim handle=\(handle.exists) \(moved) -> \(element("soundBlock").value as? String ?? "")")
        shot("typing_trimmed")
        if element("panelClose").exists { element("panelClose").tap(); pause(0.6) }

        // 2. Mute one video clip; the others keep their sound.
        tapClipAtCursor()
        tool("volume")
        let selectedClip = app.staticTexts["Selected clip"].waitForExistence(timeout: 5)
        element("volumeMute").tap()
        pause(1.5)
        log("clip_selected_panel=\(selectedClip) mute_badges=\(count("muteBadge"))")
        shot("volume_selected_clip_muted")

        // 3. Clean up voice on that clip: Cleaning… % -> playing the clean voice -> Off.
        let toggle = app.switches["cleanUpToggle"]
        if toggle.waitForExistence(timeout: 5) {
            toggle.tap()
            let progressShown = wait(20) { self.element("cleanUpProgress").exists || self.element("cleanUpReady").exists }
            shot("clean_up_cleaning")
            let ready = wait(240) { self.element("cleanUpReady").exists }
            log("clean_up progress_shown=\(progressShown) ready=\(ready) failed=\(element("cleanUpRetry").exists)")
            shot("clean_up_ready")
            toggle.tap()
            pause(1)
            log("clean_up_off ready_text=\(element("cleanUpReady").exists)")
        } else {
            log("clean_up toggle missing")
        }
        element("volumeMute").tap()   // unmute again
        pause(1)
        tapClipAtCursor()             // nothing selected: Whole video
        let whole = app.staticTexts["Whole video"].waitForExistence(timeout: 5)
        log("whole_video_panel=\(whole) mute_badges=\(count("muteBadge"))")
        shot("volume_whole_video")
        if element("panelClose").exists { element("panelClose").tap(); pause(0.6) }

        // 4. Auto Zoom AI ◇ and the Style toggle.
        let aiMarks = count("zoomKeyframe")
        tool("style")
        let autoZoom = app.switches["editorAutoZoomToggle"]
        if autoZoom.waitForExistence(timeout: 5) {
            let wasOn = (autoZoom.value as? String) == "1"
            autoZoom.tap(); pause(1.5)
            shot("style_auto_zoom_toggled")
            let after = (autoZoom.value as? String) == "1"
            autoZoom.tap(); pause(1.5)
            log("auto_zoom ai_marks=\(aiMarks) on_before=\(wasOn) after_toggle=\(after) back=\((autoZoom.value as? String) == "1")")
        }
        tool("style")
        shot("auto_zoom_marks")

        // 5. Two manual keyframes, 100 % then 130 %, then delete the second with the filled ◇.
        tapClipAtCursor()
        let key = element("editorKeyframe")
        key.tap()
        pause(1.2)
        let first = count("userKeyframe")
        scrub(seconds: 1)
        element("editorVideo").pinch(withScale: 1.3, velocity: 1.5)
        pause(1)
        shot("keyframe_pinched_130")
        key.tap()
        pause(1.2)
        let second = count("userKeyframe")
        let filled = key.value as? String ?? ""
        shot("keyframes_two")
        key.tap()
        pause(1.2)
        let afterDelete = count("userKeyframe")
        log("keyframes first=\(first) second=\(second) button_on_second=\(filled) after_delete=\(afterDelete)")
        shot("keyframe_deleted")

        // 6. Export.
        let exportStarted = Date()
        element("editorExport").tap()
        let finished = wait(420) {
            self.app.staticTexts["Your video is ready."].exists || self.app.staticTexts["The export didn't finish. Please try again."].exists
        }
        let ok = app.staticTexts["Your video is ready."].exists
        log(String(format: "export finished=%@ ok=%@ seconds=%.1f", finished ? "yes" : "no", ok ? "yes" : "no",
                   Date().timeIntervalSince(exportStarted)))
        shot("export_result")
    }
}
