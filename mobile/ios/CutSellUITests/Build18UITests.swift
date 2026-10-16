import XCTest

/// Build 18 on a real cut (live server), real touches in the Simulator, the app built OPTIMIZED like
/// TestFlight (the probe passes -O): build 17 passed the Simulator in Debug while the sounds stayed
/// silent on the iPhone, so every audio check here reads what the preview's mix really carries.
///
/// test1: sounds (old way vs new way, measured), sound edits, per-clip mute + reload, the speaker
///        six times, keyframes with pinch / twist / drag (measured on screen), Auto Zoom AI on
///        screen, photo and video overlays on the timeline (select, move, trim both edges, Split,
///        Delete) and with two fingers on the video, the system photo picker, Export.
/// test2: Settings → five taps on the version → Diagnostics → Send report.
/// Every step writes a frame (PNG) and a line to b18_ui_log.txt in B18_OUT; nothing else leaves.
final class Build18UITests: XCTestCase {
    private var app: XCUIApplication!
    private lazy var out: URL = {
        let path = ProcessInfo.processInfo.environment["B18_OUT"] ?? NSTemporaryDirectory()
        let url = URL(fileURLWithPath: path, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()
    private var step = 0
    private let started = Date()
    private var failures: [String] = []

    override func setUp() {
        continueAfterFailure = true
    }

    private func launchEditor(tempAsset: Bool = false) {
        app = XCUIApplication()
        app.launchArguments = ["-CutSellScreen", "editor_probe"]
        let env = ProcessInfo.processInfo.environment
        for key in ["CUTSELL_PROBE_TOKEN", "CUTSELL_PROBE_USER", "CUTSELL_PROBE_PROJECT", "CUTSELL_PROBE_API"] {
            if let value = env[key] { app.launchEnvironment[key] = value }
        }
        app.launchEnvironment["CUTSELL_PROBE_STATE"] = "b18_ui"
        if tempAsset { app.launchEnvironment["CUTSELL_PROBE_TEMP_ASSET"] = "1" }
        app.launch()
    }

    // MARK: Helpers

    private func log(_ text: String) {
        let line = "B18UI \(String(format: "%7.1f", Date().timeIntervalSince(started))) \(text)\n"
        print(line, terminator: "")
        let file = out.appendingPathComponent("b18_ui_log.txt")
        if let handle = try? FileHandle(forWritingTo: file) {
            handle.seekToEndOfFile()
            handle.write(line.data(using: .utf8)!)
            try? handle.close()
        } else {
            try? line.data(using: .utf8)!.write(to: file)
        }
    }

    /// A check: logged PASS / FAIL (the test goes on; failures are listed at the end).
    private func check(_ ok: Bool, _ name: String, _ detail: String = "") {
        log("\(ok ? "PASS" : "FAIL") \(name)\(detail.isEmpty ? "" : " · \(detail)")")
        if !ok { failures.append(name) }
    }

    private func shot(_ name: String) {
        step += 1
        let png = XCUIScreen.main.screenshot().pngRepresentation
        try? png.write(to: out.appendingPathComponent(String(format: "b18_%02d_%@.png", step, name)))
    }

    private func element(_ id: String) -> XCUIElement {
        app.descendants(matching: .any).matching(identifier: id).firstMatch
    }

    private func state() -> String { let e = element("b17State"); return e.exists ? e.label : "no-state" }
    private func mix() -> String { let e = element("b18Mix"); return e.exists ? e.label : "no-mix" }

    private func value(_ id: String) -> String { let e = element(id); return e.exists ? (e.value as? String ?? "") : "missing" }
    private func count(_ id: String) -> Int { app.descendants(matching: .any).matching(identifier: id).count }
    private func values(_ id: String) -> [String] {
        app.descendants(matching: .any).matching(identifier: id).allElementsBoundByIndex.compactMap { $0.exists ? $0.value as? String : nil }
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

    private func tap(_ id: String) {
        let target = element(id)
        if target.waitForExistence(timeout: 4) { target.tap() } else { log("missing \(id)") }
        pause(0.7)
    }

    private func tool(_ name: String) {
        let button = element("tool_\(name)")
        guard button.waitForExistence(timeout: 4) else { log("missing tool_\(name)"); return }
        let screen = app.windows.firstMatch.frame
        if !screen.isEmpty, button.frame.maxX > screen.maxX - 4 { element("tool_overlay").swipeLeft(); pause(0.8) }
        if !screen.isEmpty, button.frame.minX < screen.minX + 4 { element("tool_overlay").swipeRight(); pause(0.8) }
        button.tap()
        pause(1.0)
    }

    private func seek(_ seconds: Int) {
        tap("b18Start")
        for _ in 0..<seconds { tap("b17Step") }
    }

    private var timeline: XCUIElement { element("timelineV4") }

    private func tapClipAtCursor() {
        timeline.coordinate(withNormalizedOffset: .zero).withOffset(CGVector(dx: 62, dy: 86)).tap()
        pause(0.8)
    }

    private func closePanel() { if element("panelClose").exists { tap("panelClose") } }

    /// "field=value" from a text ("vv=1.00 t=2.00 …").
    private func field(_ name: String, in text: String) -> String? {
        // The whole key only ("zoom=" must not match inside "auto_zoom=").
        var search = text.startIndex..<text.endIndex
        while let range = text.range(of: "\(name)=", range: search) {
            let before = range.lowerBound == text.startIndex ? " " : text[text.index(before: range.lowerBound)]
            if before == " " || before == "[" || before == ";" {
                return String(text[range.upperBound...].prefix { $0 != " " && $0 != ";" && $0 != "]" })
            }
            search = range.upperBound..<text.endIndex
        }
        return nil
    }

    /// Pause the preview if it plays (a measurement needs a still frame).
    private func ensurePaused() {
        if field("playing", in: state()) == "true" { tap("editorPlay") }
    }

    private func number(_ name: String, in text: String) -> Double? { field(name, in: text).flatMap(Double.init) }

    /// "photo:1.00-4.00:w0.40:r0" items of one kind on the video (a trailing "u": still uploading).
    private func overlayItems(_ kind: String) -> [String] {
        (field("overlays", in: state()) ?? "").split(separator: ",").map(String.init).filter { $0.hasPrefix("\(kind):") }
    }

    /// The sound entries of the mix summary: label@start+len file= dur= in= vol= db=.
    private func soundEntries(_ text: String = "") -> [String] {
        let source = text.isEmpty ? mix() : text
        guard let a = source.range(of: "sounds=["), let b = source.range(of: "] video=") else { return [] }
        let inner = source[a.upperBound..<b.lowerBound]
        return inner.isEmpty ? [] : inner.split(separator: ";").map(String.init)
    }

    private func videoLevels() -> [Double] {
        let text = mix()
        guard let a = text.range(of: "video=["), let b = text[a.upperBound...].range(of: "]") else { return [] }
        return text[a.upperBound..<b.lowerBound].split(separator: ",").compactMap { $0.split(separator: ":").last.flatMap { Double($0) } }
    }

    /// Fraction of the video card's middle row that is the test video's white square (bigger = zoomed).
    private func whiteWidth() -> Double {
        let video = element("editorVideo")
        guard video.exists, let cg = video.screenshot().image.cgImage else { return -1 }
        let width = cg.width, height = cg.height
        guard let data = cg.dataProvider?.data, let bytes = CFDataGetBytePtr(data) else { return -1 }
        let perRow = cg.bytesPerRow, perPixel = cg.bitsPerPixel / 8
        var best = 0
        for rowFraction in [0.46, 0.5, 0.54] {
            let y = Int(Double(height) * rowFraction)
            var run = 0, longest = 0
            for x in 0..<width {
                let o = y * perRow + x * perPixel
                let white = bytes[o] > 190 && bytes[o + 1] > 190 && bytes[o + 2] > 190
                run = white ? run + 1 : 0
                longest = max(longest, run)
            }
            best = max(best, longest)
        }
        return Double(best) / Double(max(1, width))
    }

    // MARK: test1 — the editor

    func test1EditorOnARealCut() throws {
        // A. The OLD way (build 17: a temporary asset nobody keeps), optimized build: add Typing at 0:02.
        launchEditor(tempAsset: true)
        XCTAssertTrue(timeline.waitForExistence(timeout: 120), "editor did not open")
        XCTAssertTrue(wait(120) { self.element("editorPlay").isEnabled }, "preview not ready")
        pause(5)
        log("opened (old way) \(state())")
        seek(2)
        tool("audio")
        _ = element("audioPanel").waitForExistence(timeout: 10)
        let downloaded = wait(90) { !self.element("soundDownloading").exists }
        tap("sound_typing")
        _ = wait(30) { self.element("addSoundAtPlayhead").exists && self.element("addSoundAtPlayhead").isEnabled }
        tap("addSoundAtPlayhead")
        let added = wait(15) { self.count("soundBlock") == 1 }
        check(added, "typing added at 0:02", "downloaded=\(downloaded) \(state())")
        _ = wait(25) { self.soundEntries().first?.contains("db=?") == false }
        let oldWay = soundEntries().first ?? "none"
        log("OLD WAY (temporary asset) mix: \(oldWay)")
        shot("old_way_typing_added")
        closePanel()
        pause(3)   // the draft is saved
        app.terminate()

        // B. The NEW way (this build): the same draft, Typing at 0:02 must be in the mix with sound.
        launchEditor()
        XCTAssertTrue(timeline.waitForExistence(timeout: 120))
        XCTAssertTrue(wait(120) { self.element("editorPlay").isEnabled })
        _ = wait(40) { (self.soundEntries().first.map { !$0.contains("db=?") }) ?? false }
        let newWay = soundEntries().first ?? "none"
        log("NEW WAY mix: \(newWay)")
        let inSeconds = number("in", in: newWay) ?? 0
        let db = number("db", in: newWay) ?? -120
        check(newWay.hasPrefix("Typing@2.00") && inSeconds > 2.5 && db > -45, "typing in the mix at 2 s with sound",
              String(format: "in=%.2f s level=%.1f dB", inSeconds, db))
        // Play across 0:02 (the session must be Playback / MoviePlayback: sound with the silent switch on).
        seek(1)
        tap("editorPlay")
        pause(2.5)
        tap("editorPlay")
        let session = field("session", in: state()) ?? "?"
        check(session.hasPrefix("Playback/MoviePlayback"), "audio session playback", session)
        // What really went through the player while it played (its tap), not only what the mix holds.
        _ = wait(10) { (self.soundEntries().first.flatMap { self.number("heard", in: $0) }) != nil }
        let heard = soundEntries().first.flatMap { number("heard", in: $0) } ?? -120
        check(heard > -45, "typing heard while the preview played", String(format: "%.1f dB", heard))
        shot("new_way_played")

        // C. Sound edits rebuild the mix: select, plain drag, Split, trim, Delete.
        let block = element("soundBlock")
        if block.waitForExistence(timeout: 5) {
            block.tap(); pause(1)
            let before = soundEntries().first ?? ""
            let c = block.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
            c.press(forDuration: 0.05, thenDragTo: c.withOffset(CGVector(dx: 45, dy: 0)))
            let moved = wait(15) { (self.soundEntries().first ?? "") != before && !(self.soundEntries().first ?? "").contains("db=?") }
            check(moved, "selected sound moves with a plain drag (no hold)", "\(before) -> \(soundEntries().first ?? "")")
            seek(4)
            tool("split")
            let split = wait(15) { self.soundEntries().count == 2 && self.soundEntries().allSatisfy { !$0.contains("in=0.00") } }
            check(split, "split sound: two pieces, both in the mix", mix())
            shot("sound_split")
            if !element("audioTrimEnd").exists, element("soundBlock").exists { element("soundBlock").tap(); pause(0.8) }
            let handle = element("audioTrimEnd")
            if handle.exists {
                let h = handle.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
                h.press(forDuration: 0.1, thenDragTo: h.withOffset(CGVector(dx: -15, dy: 0)))
                pause(2)
            }
            log("after trim: \(soundEntries())")
            tool("delete")
            let deleted = wait(15) { self.soundEntries().count == 1 }
            check(deleted, "deleted one piece: one sound left in the mix", mix())
        } else {
            check(false, "sound block on the timeline")
        }

        // D. Per-clip mute: only that clip goes silent; survives a reload; the speaker always works.
        seek(0)
        tapClipAtCursor()
        tool("volume")
        let selectedClip = app.staticTexts["Selected clip"].waitForExistence(timeout: 5)
        tap("volumeMute")
        let clipMuted = wait(15) { let l = self.videoLevels(); return l.count > 1 && l[0] == 0 && l.dropFirst().contains { $0 > 0 } }
        check(selectedClip && clipMuted, "Selected clip mute: that clip 0, the others play", "levels=\(videoLevels())")
        check(soundEntries().contains { (number("vol", in: $0) ?? 0) > 0.5 }, "the sound keeps its volume (no video mute inherited)")
        shot("clip_muted")
        closePanel()
        tap("b18Reload")
        pause(6)
        let kept = wait(30) { (self.field("muted_clips", in: self.state()) ?? "0") == "1" }
        check(kept, "clip mute kept after reloading the draft", state())
        for round in 1...6 {
            tap("rowVideoSound")
            let expectOff = round % 2 == 1
            let ok = wait(15) {
                let vv = self.number("video_volume", in: self.mix()) ?? -1
                return expectOff ? vv == 0 : vv > 0.5
            }
            let icon = value("rowVideoSound")
            let levels = videoLevels()
            let mixOK = expectOff ? levels.allSatisfy { $0 == 0 } : levels.allSatisfy { $0 > 0 }
            check(ok && mixOK && icon == (expectOff ? "off" : "on"), "speaker tap \(round) -> \(expectOff ? "off" : "on")",
                  "icon=\(icon) levels=\(levels)")
        }
        shot("speaker_on_again")

        // E. Keyframes with the fingers on the video (UIKit pinch / twist / drag), measured on screen.
        ensurePaused()
        seek(0)
        let flatWidth = whiteWidth()
        tapClipAtCursor()
        seek(1)
        element("editorVideo").pinch(withScale: 1.6, velocity: 1.2)
        pause(1)
        let pinched = state()
        check((number("zoom", in: pinched) ?? 1) > 1.2, "pinch zooms the video", "pending=\(field("pending", in: pinched) ?? "?")")
        tap("editorKeyframe")
        let firstKey = wait(10) { (self.field("user", in: self.state()) ?? "0") == "1" }
        check(firstKey, "◇ keeps the pinch as a keyframe", state())
        let zoomedWidth = whiteWidth()
        check(zoomedWidth > flatWidth * 1.15, "zoom visible on screen", String(format: "white %.3f -> %.3f", flatWidth, zoomedWidth))
        seek(3)
        element("editorVideo").rotate(CGFloat.pi / 5, withVelocity: 1.5)
        pause(1)
        let turned = number("rot", in: state()) ?? 0
        check(abs(turned) > 15, "two-finger twist turns the video", String(format: "rot=%.1f", turned))
        let v = element("editorVideo").coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        v.press(forDuration: 0.05, thenDragTo: v.withOffset(CGVector(dx: 30, dy: 0)))
        pause(1)
        check(field("playing", in: state()) == "false", "a drag on the video never starts playback", state())
        tap("editorKeyframe")
        let secondKey = wait(10) { (self.field("user", in: self.state()) ?? "0") == "2" }
        check(secondKey, "second ◇ with the turn", "user_rot=\(field("user_rot", in: state()) ?? "?")")
        shot("keyframes_two")
        // Play from 1 s to 3 s: the framing changes smoothly between the two ◇.
        seek(1)
        var samples: [String] = []
        tap("editorPlay")
        for _ in 0..<8 {
            pause(0.3)
            samples.append(String(format: "%@/%@", field("zoom", in: state()) ?? "?", field("rot", in: state()) ?? "?"))
        }
        tap("editorPlay")
        let distinct = Set(samples).count
        check(distinct >= 4, "smooth change while playing", samples.joined(separator: " "))
        // Auto Zoom AI ◇ on screen.
        ensurePaused()
        tap("b18AIPeak")
        pause(1.5)
        log("at the AI zoom: \(state())")
        let aiZoom = number("zoom", in: state()) ?? 1
        let aiWidth = whiteWidth()
        check(aiZoom > 1.01 && aiWidth > flatWidth * 1.01, "Auto Zoom AI zooms the preview",
              String(format: "zoom=%.3f white %.3f -> %.3f", aiZoom, flatWidth, aiWidth))
        shot("auto_zoom_on_screen")

        // F. Overlays on the timeline: photo and video, select / move / trim / Split / Delete; two fingers.
        for kind in ["photo", "video"] {
            seek(1)
            tap(kind == "photo" ? "b18AddPhoto" : "b18AddVideo")
            let placed = wait(120) {
                let items = self.overlayItems(kind)
                return !items.isEmpty && items.allSatisfy { !$0.hasSuffix("u") }
            }
            check(placed, "\(kind) overlay added and uploaded", field("overlays", in: state()) ?? "")
            let blocks = app.descendants(matching: .any).matching(identifier: "overlayBlock").allElementsBoundByIndex
                .filter { ($0.value as? String ?? "").hasPrefix(kind) }
            guard let blockElement = blocks.first else { check(false, "\(kind) overlay block"); continue }
            log("\(kind) block hittable=\(blockElement.isHittable) frame=\(blockElement.frame)")
            if !(field("sel", in: state()) == "overlay") { blockElement.tap(); pause(1) }
            let start0 = blockElement.value as? String ?? ""
            let b = blockElement.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
            b.press(forDuration: 0.05, thenDragTo: b.withOffset(CGVector(dx: 30, dy: 0)))
            let movedOK = wait(15) { (blocks.first?.value as? String ?? "") != start0 }
            check(movedOK, "\(kind) overlay moves with a plain drag when selected", "\(start0) -> \(blocks.first?.value as? String ?? "")")
            for (edge, dx) in [("overlayTrimEnd", 40.0), ("overlayTrimStart", -20.0)] {
                let before = blocks.first?.value as? String ?? ""
                let handle = element(edge)
                if handle.exists {
                    let h = handle.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
                    h.press(forDuration: 0.1, thenDragTo: h.withOffset(CGVector(dx: dx, dy: 0)))
                }
                let changed = wait(15) { (blocks.first?.value as? String ?? "") != before }
                check(changed, "\(kind) overlay \(edge == "overlayTrimEnd" ? "end" : "start") trimmed (\(dx > 0 ? "longer" : "longer to the left"))",
                      "\(before) -> \(blocks.first?.value as? String ?? "")")
            }
            // Two fingers on the video move / size / turn the selected overlay.
            let overlaysBefore = field("overlays", in: state()) ?? ""
            element("editorVideo").pinch(withScale: 0.7, velocity: -1)
            pause(1)
            element("editorVideo").rotate(CGFloat.pi / 6, withVelocity: 1.5)
            pause(1.5)
            let overlaysAfter = field("overlays", in: state()) ?? ""
            check(overlaysAfter != overlaysBefore, "\(kind) overlay sized / turned with two fingers on the video",
                  "\(overlaysBefore) -> \(overlaysAfter)")
            // Split at the cursor inside it, then Delete one piece.
            let window = (blocks.first?.value as? String ?? "").split(separator: " ").last?.split(separator: "-").compactMap { Double($0) } ?? []
            if window.count == 2 { seek(Int(((window[0] + window[1]) / 2).rounded(.down))) }
            if field("sel", in: state()) != "overlay" { blockElement.tap(); pause(0.8) }
            let before = overlayItems(kind).count
            tool("split")
            let split = wait(15) { self.overlayItems(kind).count == before + 1 }
            check(split, "\(kind) overlay split in two", field("overlays", in: state()) ?? "")
            shot("\(kind)_overlay_split")
            let pieces = app.descendants(matching: .any).matching(identifier: "overlayBlock").allElementsBoundByIndex
                .filter { ($0.value as? String ?? "").hasPrefix(kind) }
            if let piece = pieces.last {
                if field("sel", in: state()) != "overlay" { piece.tap(); pause(0.8) }
                tool("delete")
                let deleted = wait(15) { self.overlayItems(kind).count == before }
                check(deleted, "\(kind) overlay piece deleted", field("overlays", in: state()) ?? "")
            }
        }

        // G. The system photo picker (whole library, no full access needed).
        closePanel()
        tool("overlay")
        tap("overlayAddPhoto")
        let picker = wait(15) { self.element("overlayGallery").exists }
        pause(3)
        shot("system_photo_picker")
        // A photo of the picker: an image that can be touched (the editor behind the sheet can't).
        var image = element("overlayGallery").descendants(matching: .image).firstMatch
        if !image.exists {
            image = app.images.allElementsBoundByIndex.first { $0.isHittable && $0.frame.minY > 160 && $0.frame.width > 60 } ?? image
        }
        log("picker images=\(app.images.count) cells=\(app.cells.count) chosen=\(image.exists ? "\(image.frame)" : "none")")
        var picked = false
        if picker, image.exists {
            let overlaysBefore = overlayItems("photo").count
            image.tap()
            picked = wait(60) { self.overlayItems("photo").count > overlaysBefore }
        }
        check(picker, "system photo picker opens", "photo picked=\(picked)")
        if !picked, element("overlayGallery").exists { app.swipeDown(); pause(1) }

        // H. Export.
        let exportStarted = Date()
        tap("editorExport")
        let finished = wait(480) {
            self.app.staticTexts["Your video is ready."].exists || self.app.staticTexts["The export didn't finish. Please try again."].exists
        }
        check(finished && app.staticTexts["Your video is ready."].exists, "export",
              String(format: "%.1f s", Date().timeIntervalSince(exportStarted)))
        shot("export_result")
        log("FAILURES: \(failures.isEmpty ? "none" : failures.joined(separator: " | "))")
    }

    // MARK: test2 — Diagnostics

    func test2DiagnosticsFromSettings() throws {
        app = XCUIApplication()
        app.launchArguments = ["-CutSellScreen", "you"]
        app.launch()
        let version = element("appVersion")
        XCTAssertTrue(version.waitForExistence(timeout: 20))
        if !version.isHittable { app.swipeUp(); pause(1) }
        for _ in 0..<5 { version.tap() }
        let shown = element("diagnosticsScreen").waitForExistence(timeout: 10)
        pause(1.5)
        shot("diagnostics")
        let build = element("diagBuild").exists ? element("diagBuild").label : "missing"
        let photos = element("diagPhotos").exists ? element("diagPhotos").label : "missing"
        var events: [String] = []
        for _ in 0..<8 {
            events = app.staticTexts.allElementsBoundByIndex.map(\.label).filter { $0.contains("sound_") || $0.contains("keyframe_") || $0.contains("playback_ready") }
            if !events.isEmpty { break }
            app.swipeUp(); pause(0.8)
        }
        shot("diagnostics_events")
        check(shown, "Diagnostics opens with five taps on the version", "build=\(build) photos=\(photos)")
        check(!events.isEmpty, "Diagnostics lists the editor's events", events.prefix(2).joined(separator: " | "))
        let send = element("diagSendReport")
        if send.waitForExistence(timeout: 5) {
            send.tap()
            let sheet = wait(10) { self.app.otherElements["ActivityListView"].exists || self.app.buttons["Copy"].exists
                || self.app.cells.count > 0 && self.app.navigationBars.count == 0 }
            pause(1)
            shot("send_report_sheet")
            check(sheet, "Send report opens the share sheet")
        } else {
            check(false, "Send report button")
        }
        log("FAILURES (diagnostics): \(failures.isEmpty ? "none" : failures.joined(separator: " | "))")
    }
}
