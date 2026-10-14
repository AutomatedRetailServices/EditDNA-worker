import AVFoundation
import AVKit
import SwiftUI

/// Editor v4 (Figma 04_Editor "Editor v4" 1-11, approved by Swanny on 13 Oct 2026).
/// Header without logo (back · cut name, tap to rename · Export) · bigger video · Full screen / Play /
/// Undo / Redo · timeline with the cursor fixed in the middle that slides by itself while playing and
/// zooms with two fingers · video track with each clip's caption inside it · voice-over track ·
/// overlay track · toolbar v4 (Split · Delete · Mirror · Rotate | Overlay · Voice Over · Captions · Style).
struct EditorV4View: View {
    @ObservedObject var model: DraftEditorViewModel
    let project: Project
    /// Design screenshots only (Debug): open in a given state.
    var previewState: EditorPreviewState? = nil

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var appState: AppState
    @StateObject private var playback = DraftPlaybackController()
    @StateObject private var recorder = VoiceRecorder()
    @State private var selection: EditorSelection?
    @State private var toast: EditorToast?
    @State private var sheet: EditorSheet?
    @State private var panel: EditorPanel = .none
    @State private var fullScreen = false
    @State private var renaming = false
    @State private var newTitle = ""
    @State private var displayTitle: String?
    @State private var choosingOverlay = false
    @State private var addingText = false
    @State private var overlayText = ""
    @State private var pixelsPerSecond: CGFloat = 30
    @State private var captionText = ""
    @FocusState private var captionFocused: Bool
    @State private var videoVolume: Double = 1
    /// "Video sound" panel under the timeline: opened only by the speaker or by tapping a voice clip.
    @State private var soundPanel: SoundPanelTarget?

    private var duration: Double { max(0.1, playback.duration > 0 ? playback.duration : model.timelineDuration) }

    /// Rebuild the preview when what it plays changes (the predicted draft included, so it follows at once).
    private var previewKey: String {
        let placements = model.previewVoices.map { "\($0.key)@\($0.start)" }.joined(separator: ",")
        return "\(model.displayDraft?.hashValue ?? 0)|\(placements)|\(model.voiceOverVolume)|\(model.loadGeneration)"
    }

    var body: some View {
        presentations(lifecycle(content))
    }

    private var content: some View {
        ZStack {
            Color(hex: 0x050814).ignoresSafeArea()
            GeometryReader { geo in
                // The video keeps its size whatever is open below it (Figma: 266 x 472 on a 390 x 844
                // iPhone); the timeline, a menu or the voice-over panel use the space under the controls.
                let previewHeight = max(200, min(472, geo.size.height - 48 - 8 - reservedBelowVideo))
                VStack(spacing: 0) {
                    header
                    preview
                        .frame(height: previewHeight)
                        .padding(.top, 8)
                    if showsControls {
                        controlsRow
                            .padding(.top, 9)
                            .padding(.bottom, 9)
                    }
                    bottomArea
                        .frame(maxHeight: .infinity, alignment: .top)
                }
                .animation(.easeOut(duration: 0.22), value: panel)
            }
            .ignoresSafeArea(.container, edges: .bottom)
            if isVoicePanel {
                // Figma 1084:2562: the voice-over sheet comes up over the video.
                VStack(spacing: 0) {
                    Spacer(minLength: 0)
                    VoiceOverPanel(model: model, recorder: recorder, start: playback.currentTime, mode: $panel) { outcome in
                        switch outcome {
                        case .placed: show(.message("Voice-over added"))
                        case .savedNotPlaced: show(.message("Voice-over saved. It will be placed when this cut's main video is registered."))
                        case .failed: break
                        }
                    }
                    .padding(.horizontal, 1)
                }
                .ignoresSafeArea(.container, edges: .bottom)
                .transition(.move(edge: .bottom))
            }
            if panel == .micPermission {
                micPermissionCard
            }
        }
    }

    /// Height used under the video: controls, then the timeline, a menu (captions 2b, style) or the
    /// voice-over panel, then the toolbar (Figma: the video is 472 high with the timeline and while
    /// writing a caption, 408 with a menu open).
    private var reservedBelowVideo: CGFloat {
        let toolbar = ToolbarV4.height
        let controls: CGFloat = 9 + 24 + 9
        switch panel {
        // Captions menu 2b (Figma 1083:5746, 1049:2492...): no controls; the menu takes their place.
        case .captions: return controls + 1 + TimelineV4.height + toolbar
        // Style (Figma 1054:2497): no controls; the panel sits where the controls and timeline were.
        case .style: return controls + 1 + TimelineV4.height + toolbar
        case .none, .micPermission, .voiceRecord, .voiceReview:
            return controls + 1 + TimelineV4.height + toolbar + (soundPanel == nil ? 0 : SoundPanelV4.height)
        }
    }

    private var showsControls: Bool {
        switch panel {
        case .style, .captions: return false
        default: return true
        }
    }

    private var isVoicePanel: Bool { panel == .voiceRecord || panel == .voiceReview }

    private func lifecycle<V: View>(_ view: V) -> some View {
        view
            .toolbar(.hidden, for: .navigationBar)
            .preferredColorScheme(.dark)
            .task(id: previewKey) { await playback.rebuild(from: model) }
            .onChange(of: playback.currentTime) { _, time in
                if abs(model.playheadTime - time) >= 0.1 || time == 0 { model.playheadTime = time }
            }
            .onChange(of: selection) { _, newValue in selectionChanged(newValue) }
            .onAppear { appeared() }
            .onDisappear {
                playback.pause()
                if recorder.state == .recording { recorder.stop() }
            }
    }

    private func appeared() {
        EditorAudio.preparePlayback()
        videoVolume = model.videoTrackVolume
        guard let previewState else { return }
        selection = previewState.selection
        toast = previewState.toast
        #if DEBUG
        if previewState.panel == .voiceRecord {
            recorder.previewRecording(levels: Self.sampleLevels(), elapsed: 7)
        } else if previewState.panel == .voiceReview, let url = Self.sampleTone() {
            recorder.previewRecorded(url: url, elapsed: 8)
        }
        #endif
        var opening = previewState.panel
        if case .captions(.some("first")) = opening, let first = model.selectedClips.first?["clip_id"]?.stringValue {
            opening = .captions(first)
        }
        if case .overlay("first_media", _) = previewState.selection {
            selection = model.mediaOverlays.first?["overlay_id"]?.stringValue.map { .overlay($0, isText: false) }
        }
        if let opened = opening { panel = opened }
        if case .captions(let clipID?) = opening {
            captionText = captionTextFor(clipID)
        }
        #if DEBUG
        if previewState.probe == "stress" {
            Task { await runStressProbe() }
            return
        }
        if previewState.probe == "flow" {
            Task { await runFlowProbe() }
            return
        }
        if previewState.probe == "trim" {
            Task { await runTrimProbe() }
            return
        }
        if previewState.probe == "voiceover" {
            Task { await runVoiceOverProbe() }
            return
        }
        if let probe = previewState.probe {
            switch probe {
            case "cap_highlight": model.applyCaptionSettings(enabled: true, preset: "highlight_green", font: "montserrat")
            case "cap_box": model.applyCaptionSettings(enabled: true, preset: "box", font: "montserrat")
            case "cap_fonts": model.applyCaptionSettings(enabled: true, preset: "classic", font: "bebas_neue")
            case "caption_edit": model.applyCaptionSettings(enabled: true, preset: "highlight_green")
            default: break
            }
            // Move the cursor into the cut, like the Figma screens.
            DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                let at = min(3, playback.duration / 3)
                if probe == "base" || probe == "style" {
                    model.probeShowZoomKeyframes(at: [max(0, at - 0.6), at + 4])
                }
                playback.seek(to: probe == "base" ? 0 : at)
            }
        }
        #endif
        if let opened = previewState.sheet {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { sheet = opened }
        }
    }

    private func presentations<V: View>(_ view: V) -> some View {
        view
            .sheet(item: $sheet) { which in sheetView(which) }
            .fullScreenCover(isPresented: $fullScreen) { fullScreenPlayer }
            .alert("Rename cut", isPresented: $renaming) {
                TextField("Name", text: $newTitle)
                Button("Save") { Task { await rename() } }
                Button("Cancel", role: .cancel) {}
            }
            .confirmationDialog("Add to the video", isPresented: $choosingOverlay, titleVisibility: .visible) {
                Button("Text") { addingText = true }
                Button("Photo or video") { sheet = .overlay }
                Button("B-roll (full screen)") { sheet = .broll }
                Button("Cancel", role: .cancel) {}
            }
            .alert("Add text", isPresented: $addingText) {
                TextField("Text", text: $overlayText)
                Button("Add") { addTextOverlay() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("It appears on the video at the cursor for 3 seconds.")
            }
            .alert("CutSell", isPresented: errorShown) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(model.errorMessage ?? "")
            }
    }

    private var errorShown: Binding<Bool> {
        Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })
    }

    private var fullScreenPlayer: some View {
        ZStack(alignment: .topLeading) {
            Color.black.ignoresSafeArea()
            VideoPlayer(player: playback.player).ignoresSafeArea()
                .scaleEffect(x: model.isMirrored(at: playback.currentTime) ? -1 : 1, y: 1)
            Button { fullScreen = false } label: {
                Image(systemName: "xmark").font(.system(size: 17, weight: .bold)).foregroundStyle(.white)
                    .frame(width: 44, height: 44).background(Color.black.opacity(0.4), in: Circle())
            }
            .padding(16)
            .accessibilityLabel("Close full screen")
        }
    }

    private func addTextOverlay() {
        let text = overlayText.trimmingCharacters(in: .whitespacesAndNewlines)
        overlayText = ""
        guard !text.isEmpty else { return }
        let at = playback.currentTime
        Task { await model.addTextOverlay(text: text, start: at, end: at + 3) }
    }

    #if DEBUG
    /// Editor probe "stress": 10 quick edits in a row (split, delete, caption, mirror, undo/redo...),
    /// started without waiting for each other, right after the local draft was made one revision
    /// stale on purpose. Expected: no alert, every edit saved, the stale revision healed quietly.
    private func runStressProbe() async {
        try? await Task.sleep(nanoseconds: 3_000_000_000)
        await model.probeMakeLocalRevisionStale()
        let started = Date()
        var tasks: [Task<Void, Never>] = []
        func go(_ work: @escaping @MainActor () async -> Void) { tasks.append(Task { @MainActor in await work() }) }
        let windows = EditorTracks.clipWindows(model)
        if let first = windows.first {
            let middle = first.sourceStart + (first.end - first.start) / 2
            if let at = model.safeSplitTime(clipID: first.clipID, near: middle) { go { await model.split(clipID: first.clipID, at: at) } }
        }
        go { await model.editCaption(clipID: windows.first?.clipID ?? "", text: "Stress test caption") }
        go { _ = await model.toggleMirror(clipID: nil) }
        model.applyCaptionSettings(preset: "box")
        model.applyCaptionSettings(preset: "box")          // same choice again: must save nothing
        go { _ = await model.undo() }
        go { _ = await model.redo() }
        if windows.count > 1, let last = windows.last { go { await model.remove(clipID: last.clipID) } }
        go { _ = await model.toggleMirror(clipID: nil) }
        go { _ = await model.undo() }
        model.applyCaptionSettings(preset: "highlight_green")
        for task in tasks { await task.value }
        try? await Task.sleep(nanoseconds: 1_500_000_000)
        let seconds = Date().timeIntervalSince(started)
        let error = model.errorMessage
        let summary = String(format: "10 edits in %.1f s · %d quiet retries · %d skipped · %@", seconds, model.saveConflicts,
                             model.skippedEdits, error == nil ? "no errors" : "ERROR: \(error!)")
        toast = .message(summary)
        NSLog("CUTSELLSTRESS %@", summary)
    }

    /// Editor probe "flow" (screen recording): play the whole cut (it must stop at its end and the
    /// timeline with it), then split, caption, mirror and undo, each shown at once.
    private func runFlowProbe() async {
        func wait(_ seconds: Double) async { try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000)) }
        await wait(4)
        playback.seek(to: 0)
        playback.togglePlayback()
        await wait(max(2, playback.duration) + 2.5)
        show(.message(playback.isPlaying ? "Still playing" : "Stopped at the end of the cut (\(EditorTime.short(playback.currentTime)))"))
        await wait(2.5)
        playback.seek(to: max(0.5, playback.duration / 3))
        await wait(1)
        await split()
        await wait(2)
        if let id = EditorTracks.clipWindows(model).first?.clipID {
            Task { await model.editCaption(clipID: id, text: "Edited on the phone") }
            show(.message("Caption changed"))
        }
        await wait(2)
        toolTapped(.mirror)
        await wait(2.5)
        Task { _ = await model.undo() }
        show(.message("Undo"))
        await wait(2.5)
        show(.message(model.errorMessage == nil ? "Done · no errors" : "Error: \(model.errorMessage!)"))
    }

    /// Editor probe "trim" (real cut): select the first clip (white handles), show the drag frame
    /// with "+0,4 s", stretch its end outward 0.4 s over hidden material (one save), shorten its
    /// start 0.3 s, then Undo and Redo. Logs CUTSELLTRIM lines with the windows and revisions.
    private func runTrimProbe() async {
        func wait(_ seconds: Double) async { try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000)) }
        func describe(_ label: String) {
            let clips = model.selectedClips
            let first = clips.first
            let line = String(format: "%@ rev=%d clips=%d first=%.3f-%.3f manual_end=%@ caption=%@ total=%.2f",
                              label, model.snapshot?.revision ?? -1, clips.count,
                              first?["start"]?.doubleValue ?? -1, first?["end"]?.doubleValue ?? -1,
                              first?["manual_end"]?.boolValue == true ? "yes" : "no",
                              "\"\(first?["caption_text"]?.stringValue ?? "")\"", model.timelineDuration)
            NSLog("CUTSELLTRIM %@", line)
        }
        await wait(4)
        let clips = model.selectedClips
        guard let first = clips.first, let id = first["clip_id"]?.stringValue,
              let start = first["start"]?.doubleValue, let end = first["end"]?.doubleValue else {
            NSLog("CUTSELLTRIM no clip"); return
        }
        describe("before")
        selection = .video(id)
        await wait(2.5)
        let sourceID = first["source_asset_id"]?.stringValue
        let limits = TrimMath.limits(clips: clips, index: 0, edge: .end, sourceDuration: model.sourceDuration(forSource: sourceID))
        let target = TrimMath.resolve(proposed: end + 0.4 + (limits.neighbor.map { $0 < end + 0.4 ? TrimMath.passThrough : 0 } ?? 0),
                                      edge: .end, limits: limits)
        NSLog("CUTSELLTRIM limits lower=%.3f upper=%.3f neighbor=%.3f target=%.3f",
              limits.lower, limits.upper == .greatestFiniteMagnitude ? -1 : limits.upper, limits.neighbor ?? -1, target.value)
        // What the video shows while the handle is dragged.
        model.trimScrub = TrimScrub(sourceID: sourceID, sourceTime: target.value - 1.0 / 30,
                                    delta: target.value - end, mirrored: first["mirror"]?.boolValue == true)
        await wait(4)
        model.trimScrub = nil
        let saveStart = model.snapshot?.revision ?? 0
        await model.trimClip(clipID: id, edge: .end, to: target.value).value
        describe("stretched_end")
        NSLog("CUTSELLTRIM saves_for_one_drag=%d", (model.snapshot?.revision ?? 0) - saveStart)
        show(.message("End stretched \(TrimMath.deltaLabel(target.value - end))"))
        await wait(3)
        await model.trimClip(clipID: id, edge: .start, to: min(start + 0.3, target.value - 0.2)).value
        describe("shortened_start")
        show(.message("Start shortened"))
        await wait(3)
        _ = await model.undo()
        describe("undo")
        show(.message("Undo"))
        await wait(3)
        _ = await model.redo()
        describe("redo")
        await wait(1)
        selection = .video(id)
        NSLog("CUTSELLTRIM done error=%@", model.errorMessage ?? "none")
        show(.message(model.errorMessage == nil ? "Trim done · no errors" : "Error: \(model.errorMessage!)"))
    }

    /// Editor probe "voiceover": Save a recording (it must be on the mic row at once while it
    /// uploads), then the speaker's "Video sound" panel and a voice clip's panel. Logs CUTSELLVO.
    private func runVoiceOverProbe() async {
        func wait(_ seconds: Double) async { try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000)) }
        let opened = Date()
        await wait(4)
        guard let take = Self.sampleVoiceM4A(seconds: 6) else { NSLog("CUTSELLVO no sample"); return }
        let started = Date()
        let before = model.pendingVoiceOvers.count + (model.timelineComposition?.voiceOverPlacements.count ?? 0)
        let outcome = await model.addRecordedVoiceOver(fileURL: take, at: 1)
        let shownAfter = Date().timeIntervalSince(started)
        let after = model.pendingVoiceOvers.count + (model.timelineComposition?.voiceOverPlacements.count ?? 0)
        show(.message(String(format: "Voice-over on the mic row after %.2f s", shownAfter)))
        NSLog("CUTSELLVO shown=%.3f outcome=%@ blocks=%d->%d", shownAfter, "\(outcome)", before, after)
        // Wait for the background upload (or its retries) to finish.
        var waited = 0.0
        while !model.pendingVoiceOvers.isEmpty, waited < 18 { await wait(1); waited += 1 }
        let uploaded = model.pendingVoiceOvers.isEmpty
        NSLog("CUTSELLVO uploaded=%@ after=%.0fs placements=%d", uploaded ? "yes" : "no", waited,
              model.timelineComposition?.voiceOverPlacements.count ?? 0)
        show(.message(uploaded ? "Uploaded in the background" : "Still uploading (kept on the phone)"))
        // Fixed moments for the screenshots: Video sound panel at ~24 s, a voice clip's panel at ~30 s.
        await wait(max(1, 24 - Date().timeIntervalSince(opened)))
        toggleVideoSound()
        await wait(max(1, 30 - Date().timeIntervalSince(opened)))
        if let voice = model.timelineComposition?.voiceOverPlacements.last {
            selection = .voiceOver(voice.placementID)
        }
    }

    /// A short spoken-like tone saved as .m4a (the same format the recorder makes).
    static func sampleVoiceM4A(seconds: Double) -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("probe-voice-\(UUID().uuidString.prefix(6)).m4a")
        let rate = 44_100.0
        let settings: [String: Any] = [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: rate, AVNumberOfChannelsKey: 1]
        guard let format = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(rate * seconds)),
              let file = try? AVAudioFile(forWriting: url, settings: settings) else { return nil }
        buffer.frameLength = buffer.frameCapacity
        let data = buffer.floatChannelData![0]
        for i in 0..<Int(buffer.frameLength) {
            let t = Double(i) / rate
            let envelope = abs(sin(t * 2.3)) * (0.4 + 0.6 * abs(sin(t * 7.1)))
            data[i] = Float(envelope * sin(2 * .pi * 180 * t) * 0.6)
        }
        do { try file.write(from: buffer) } catch { return nil }
        return url
    }

    static func sampleLevels() -> [Float] {
        var levels: [Float] = []
        for i in 0..<80 {
            let a: Double = abs(sin(Double(i) * 0.7))
            let b: Double = 0.5 + 0.5 * abs(cos(Double(i) * 0.23))
            levels.append(Float(0.25 + 0.6 * a * b))
        }
        return levels
    }
    #endif

    // MARK: Header (no logo)

    private var header: some View {
        HStack(spacing: 0) {
            Button { dismiss() } label: {
                Image("ControlBack").resizable().frame(width: 44, height: 44).frame(width: 46, height: 47)
            }
            .accessibilityLabel("Back")
            Button {
                newTitle = shownTitle
                renaming = true
            } label: {
                Text("\(shownTitle) ✎")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(CS.textPrimaryF)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .frame(maxWidth: 124, alignment: .leading)
            }
            .buttonStyle(.plain)
            .padding(.leading, 4)
            .accessibilityIdentifier("editorTitle")
            Spacer(minLength: 6)
            ExportQualityMenu()
            Button { sheet = .export } label: {
                Text("Export")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 74, height: 32)
                    .background(Color(hex: 0x1769FF), in: RoundedRectangle(cornerRadius: 10))
            }
            .buttonStyle(.plain)
            .padding(.leading, 8)
            .accessibilityIdentifier("editorExport")
        }
        .padding(.leading, 8)
        .padding(.trailing, 12)
        .frame(height: 48)
    }

    private var shownTitle: String { displayTitle ?? CutNaming.readable(project: project) }

    // MARK: Video

    private var editingCaptionClip: String? {
        if case .captions(let clipID) = panel { return clipID }
        return nil
    }

    private var preview: some View {
        GeometryReader { proxy in
            videoCard(size: Self.fit(CGSize(width: 266, height: 472), in: proxy.size))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity)
        .animation(.easeOut(duration: 0.2), value: toast)
    }

    private func videoCard(size: CGSize) -> some View {
        videoStack(size: size)
            .frame(width: size.width, height: size.height)
            .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(Self.borderGradient, lineWidth: 2))
            .contentShape(Rectangle())
            .onTapGesture { videoTapped() }
            .overlay(alignment: .topLeading) { zoomBadge }
            .overlay(alignment: .bottom) { videoBottomPill }
    }

    private static let borderGradient = LinearGradient(
        colors: [CS.cyan, Color(hex: 0x7C3AED), Color(hex: 0xFF21D7)], startPoint: .bottomLeading, endPoint: .topTrailing
    )

    private func videoStack(size: CGSize) -> some View {
        ZStack {
            Color(hex: 0x070B16)
            // The cut's first frame (kept on this phone) at once, until the player is ready.
            if !playback.isReady, let still = ThumbnailStore.shared.image(for: project.projectID) {
                Image(uiImage: still).resizable().scaledToFill()
                    .frame(width: size.width, height: size.height).clipped()
                    .accessibilityIdentifier("editorFirstFrame")
            }
            videoLayer
            overlaysOnVideo(frame: size)
            if model.captionsEnabled {
                captionLayer(frame: size)
            }
            // Dragging a clip's edge: the original's exact frame at the handle, and "+0,4 s".
            if let scrub = model.trimScrub {
                TrimFrameOverlay(scrub: scrub, url: model.previewVideoURL(forSource: scrub.sourceID))
                    .frame(width: size.width, height: size.height)
            }
            if playback.isBuilding && !playback.isReady {
                ProgressView().tint(CS.cyan)
            } else if let message = playback.message, !playback.isReady {
                Text(message).font(CS.inter(12)).foregroundStyle(CS.textSecondaryF)
                    .multilineTextAlignment(.center).padding(20)
            }
        }
    }

    @ViewBuilder
    private var videoBottomPill: some View {
        if let toast {
            ToastPillV4(toast: toast) { Task { await undoFromToast() } }
                .padding(.bottom, 12)
                .transition(.opacity)
        } else if case .overlay(_, isText: false) = selection {
            hintPill("Drag to move · Pinch to resize").padding(.bottom, 14)
        }
    }

    private func videoTapped() {
        if captionFocused { finishCaptionEditing(); return }
        playback.togglePlayback()
    }

    static func fit(_ target: CGSize, in box: CGSize) -> CGSize {
        let scale = min(1, box.width / target.width, box.height / target.height)
        return CGSize(width: target.width * scale, height: target.height * scale)
    }

    /// The video, mirrored and rotated as the export will make it.
    private var videoLayer: some View {
        let quarter = model.rotationDegrees % 180 != 0
        return PlayerLayerView(player: playback.player)
            .scaleEffect(zoomNow)
            .scaleEffect(x: model.isMirrored(at: playback.currentTime) ? -1 : 1, y: 1)
            .rotationEffect(.degrees(Double(model.rotationDegrees)))
            .scaleEffect(quarter ? 266.0 / 472.0 : 1)
            .animation(.easeOut(duration: 0.2), value: model.isMirrored(at: playback.currentTime))
            .animation(.easeOut(duration: 0.25), value: model.rotationDegrees)
    }

    /// Auto Zoom AI in the preview: the same push-in as the export at this moment.
    private var zoomNow: CGFloat { CGFloat(model.zoomScale(at: playback.currentTime)) }

    /// "◇ Zoom 4%" while a zoom is on screen (Figma 1054:2497).
    @ViewBuilder
    private var zoomBadge: some View {
        let percent = Int(((zoomNow - 1) * 100).rounded())
        if percent >= 1 {
            Text("◇ Zoom \(percent)%")
                .font(.system(size: 10, weight: .semibold)).foregroundStyle(.white)
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(Color(hex: 0x040812, opacity: 0.85), in: RoundedRectangle(cornerRadius: 10))
                .padding(.leading, 12).padding(.top, 12)
                .accessibilityIdentifier("zoomBadge")
        }
    }

    @ViewBuilder
    private func captionLayer(frame: CGSize) -> some View {
        if let clipID = editingCaptionClip, captionFocusedOrEditing {
            captionEditor(clipID: clipID, frame: frame)
        } else {
            let cue = playback.captionCue(at: playback.currentTime)
            let menuOpen: Bool = { if case .captions = panel { return true } else { return false } }()
            // With the captions menu open the caption can be dragged and pinched (dashed box);
            // otherwise it looks like the export, and tapping it starts writing on it.
            CaptionOverlayView(
                cue: cue, time: playback.currentTime, preset: model.captionPreset, fontKey: model.captionFont,
                x: model.captionX, y: model.captionY, scale: model.captionScale,
                editing: menuOpen ? captionEditing(for: cue) : nil
            )
            if !menuOpen, let cue {
                Color.clear
                    .contentShape(Rectangle())
                    .frame(width: frame.width * 0.8, height: 56)
                    .position(x: frame.width * CGFloat(model.captionX), y: frame.height * CGFloat(model.captionY))
                    .onTapGesture {
                        playback.pause()
                        startCaptionEditing(cue.clipID)
                    }
                    .accessibilityIdentifier("captionTapZone")
            }
        }
    }

    private var captionFocusedOrEditing: Bool { editingCaptionClip != nil }

    /// Tap the caption on the video: write right there; the style menu (2b) opens below.
    private func captionEditing(for cue: CaptionPreviewCue?) -> CaptionEditing? {
        guard let cue else { return nil }
        let movable = cue.isTimed
        return CaptionEditing(
            onTap: {
                playback.pause()
                startCaptionEditing(cue.clipID)
            },
            onBegin: { playback.pause() },
            onLayoutChange: { x, y, scale in
                guard movable else { return }
                model.applyCaptionSettings(x: x, y: y, scale: scale)
            }
        )
    }

    private func captionEditor(clipID: String, frame: CGSize) -> some View {
        let spec = CaptionFontCatalog.spec(for: model.captionFont)
        let highlight = Self.highlightColor(model.captionPreset)
        return VStack(spacing: 6) {
            hintPill("Type here · drag · pinch")
            ZStack {
                if !captionFocused {
                    Self.highlightedText(captionText, color: highlight)
                        .font(.custom(spec.postScriptName, size: 21))
                        .multilineTextAlignment(.center)
                        .lineLimit(3)
                }
                TextField("", text: $captionText, axis: .vertical)
                    .lineLimit(1...3)
                    .font(.custom(spec.postScriptName, size: 21))
                    .multilineTextAlignment(.center)
                    .foregroundStyle(captionFocused ? .white : .clear)
                    .tint(CS.cyan)
                    .focused($captionFocused)
                    .submitLabel(.done)
                    .onSubmit { finishCaptionEditing() }
                    .accessibilityIdentifier("captionInlineField")
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .frame(width: frame.width * 0.88)
            .background(Color.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(style: StrokeStyle(lineWidth: 1.5, dash: [5, 4])).foregroundStyle(.white))
            .overlay { cornerHandles }
            .onTapGesture { captionFocused = true }
        }
        .position(x: frame.width * CGFloat(model.captionX), y: frame.height * CGFloat(min(0.8, max(0.2, model.captionY))))
    }

    private var cornerHandles: some View {
        GeometryReader { proxy in
            ForEach(0..<4, id: \.self) { index in
                Circle().fill(.white).frame(width: 10, height: 10)
                    .position(x: index % 2 == 0 ? 0 : proxy.size.width, y: index < 2 ? 0 : proxy.size.height)
            }
        }
        .allowsHitTesting(false)
    }

    /// Text overlays and photo/video overlays drawn where the export puts them.
    private func overlaysOnVideo(frame: CGSize) -> some View {
        ZStack {
            ForEach(Array(model.textOverlays.enumerated()), id: \.offset) { _, item in
                let start = item["start"]?.doubleValue ?? 0
                let end = item["end"]?.doubleValue ?? start
                if playback.currentTime >= start && playback.currentTime < end {
                    Text(item["text"]?.stringValue ?? "")
                        .font(CS.inter(CGFloat(18 * (item["scale"]?.doubleValue ?? 1)), .bold))
                        .foregroundStyle(.white)
                        .shadow(color: .black.opacity(0.6), radius: 2)
                        .position(x: frame.width * CGFloat(item["x"]?.doubleValue ?? 0.5),
                                  y: frame.height * CGFloat(item["y"]?.doubleValue ?? 0.2))
                }
            }
            ForEach(Array(model.mediaOverlays.enumerated()), id: \.offset) { _, item in
                MediaOverlayOnVideo(
                    item: item, frame: frame, time: playback.currentTime,
                    isSelected: selection == .overlay(item["overlay_id"]?.stringValue ?? "", isText: false),
                    onSelect: { selection = .overlay(item["overlay_id"]?.stringValue ?? "", isText: false) },
                    onChange: { x, y, width in
                        Task { await model.updateMediaOverlay(id: item["overlay_id"]?.stringValue ?? "", x: x, y: y, width: width) }
                    }
                )
            }
        }
        .frame(width: frame.width, height: frame.height)
    }

    // MARK: Controls (Figma: blue line icons, no boxes) — full screen · play · ◇ undo redo

    private var controlsRow: some View {
        ZStack {
            HStack(spacing: 0) {
                control("Ed4CtlExpand", "Full screen") { fullScreen = true }
                Spacer()
            }
            .padding(.leading, 12)
            Button { playback.togglePlayback() } label: {
                Group {
                    if playback.isPlaying {
                        HStack(spacing: 5) {
                            RoundedRectangle(cornerRadius: 1).stroke(Color(hex: 0x1769FF), lineWidth: 2).frame(width: 5, height: 16)
                            RoundedRectangle(cornerRadius: 1).stroke(Color(hex: 0x1769FF), lineWidth: 2).frame(width: 5, height: 16)
                        }
                    } else {
                        Image("Ed4CtlPlay").resizable().frame(width: 24, height: 24)
                    }
                }
                .frame(width: 36, height: 36)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(playback.isPlaying ? "Pause" : "Play")
            .accessibilityIdentifier("editorPlay")
            HStack(spacing: 0) {
                Spacer()
                control("Ed4CtlKeyframe", "Add zoom keyframe") { addKeyframe() }
                    .accessibilityIdentifier("editorKeyframe")
                control("Ed4CtlUndo", "Undo") { Task { await model.undo() } }
                    .accessibilityIdentifier("editorUndo")
                    .padding(.leading, 0)
                control("Ed4CtlRedo", "Redo") { Task { await model.redo() } }
                    .padding(.leading, -4)
            }
            .padding(.trailing, 10)
        }
        .frame(height: 24)
    }

    private func control(_ image: String, _ label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(image).resizable().frame(width: 24, height: 24)
                .frame(width: 36, height: 36)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .frame(width: 36, height: 24)
        .accessibilityLabel(label)
    }

    private func addKeyframe() {
        playback.pause()
        let at = playback.currentTime
        Task {
            if let id = await model.addZoomKeyframe(at: at) {
                selection = .zoom(id)
                show(.message("Zoom added at \(EditorTime.short(at))", undo: true))
            } else {
                show(.message("Move the cursor onto a clip to add a zoom."))
            }
        }
    }

    // MARK: Bottom: timeline, menus, toolbar

    @ViewBuilder
    private var bottomArea: some View {
        switch panel {
        default:
            VStack(spacing: 0) {
                switch panel {
                case .captions:
                    CaptionStylePanel(model: model) { finishCaptionEditing(); panel = .none }
                        .padding(.top, 8)
                case .style:
                    Spacer(minLength: 0)
                    StylePanel(model: model) { panel = .none }
                        .padding(.bottom, 12)
                default:
                    TimelineV4(
                        model: model, playback: playback, selection: $selection,
                        pixelsPerSecond: $pixelsPerSecond,
                        onAddVideo: { sheet = .broll },
                        onAddVoice: { openVoiceOver() },
                        onAddOverlay: { choosingOverlay = true },
                        onToggleVideoSound: { toggleVideoSound() }
                    )
                    .padding(.top, 1)
                    // Under the timeline (never over its rows).
                    if let soundPanel {
                        soundPanelView(soundPanel)
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                }
                if case .style = panel {} else { Spacer(minLength: 0) }
                ToolbarV4(active: activeTool) { tool in toolTapped(tool) }
            }
        }
    }

    /// The "Video sound" panel: the video's own sound (0–200%, 100% in the middle), or the
    /// voice-over's level (0–100%, 100% at the end). The dot sits exactly at the % shown.
    @ViewBuilder
    private func soundPanelView(_ target: SoundPanelTarget) -> some View {
        switch target {
        case .video:
            SoundPanelV4(title: "Video sound", icon: "speaker.wave.2.fill", value: $videoVolume, maximum: 2,
                         onClose: { closeSoundPanel() }) {
                Task { await model.setVideoTrackVolume(videoVolume) }
            }
        case .voice:
            SoundPanelV4(title: "Voice-over", icon: "mic.fill", value: $model.voiceOverVolume, maximum: 1,
                         onClose: { closeSoundPanel() }) {}
        }
    }

    private func closeSoundPanel() {
        withAnimation(.easeOut(duration: 0.2)) { soundPanel = nil }
        if case .voiceOver = selection { selection = nil }
    }

    /// Speaker on the video row: opens (or closes) the "Video sound" panel.
    private func toggleVideoSound() {
        videoVolume = model.videoTrackVolume
        withAnimation(.easeOut(duration: 0.2)) { soundPanel = soundPanel == .video ? nil : .video }
    }

    /// Tapping a voice clip opens its sound panel; selecting anything else closes a voice panel.
    private func selectionChanged(_ newValue: EditorSelection?) {
        withAnimation(.easeOut(duration: 0.2)) {
            if case .voiceOver = newValue {
                soundPanel = .voice
            } else if soundPanel == .voice {
                soundPanel = nil
            }
        }
    }

    private var activeTool: EditorTool? {
        switch panel {
        case .captions: return .captions
        case .style: return .style
        case .voiceRecord, .voiceReview, .micPermission: return .voiceOver
        case .none:
            if case .overlay = selection { return .overlay }
            return nil
        }
    }

    // MARK: Mic permission (Editor v4/10)

    private var micPermissionCard: some View {
        ZStack {
            Color.black.opacity(0.55).ignoresSafeArea()
                .onTapGesture { panel = .none }
            VStack(spacing: 0) {
                Image(systemName: "mic")
                    .font(.system(size: 40, weight: .regular))
                    .foregroundStyle(CS.magenta)
                    .shadow(color: CS.magenta.opacity(0.8), radius: 10)
                    .padding(.top, 30)
                Text("Allow microphone access")
                    .font(CS.inter(21, .bold)).foregroundStyle(.white)
                    .padding(.top, 18)
                Text("CutSell.ai needs microphone access to record your voiceover.")
                    .font(CS.inter(14)).foregroundStyle(CS.textSecondaryF)
                    .multilineTextAlignment(.center)
                    .padding(.top, 10).padding(.horizontal, 22)
                PrimaryButton(title: "Allow microphone", height: 52, cornerRadius: 14, fontSize: 16) {
                    Task {
                        let granted = await VoiceRecorder.requestPermission()
                        panel = granted ? .voiceRecord : .none
                        if !granted { show(.message("Allow the microphone in iOS Settings to record.")) }
                    }
                }
                .padding(.top, 24).padding(.horizontal, 20)
                .accessibilityIdentifier("allowMicrophone")
                Text("You can change this later in iOS Settings.")
                    .font(CS.inter(12)).foregroundStyle(CS.textMutedF)
                    .padding(.top, 12).padding(.bottom, 22)
            }
            .frame(maxWidth: .infinity)
            .background(Color(hex: 0x0A1326), in: RoundedRectangle(cornerRadius: 24, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 24, style: .continuous).stroke(Color(hex: 0x2A4A80), lineWidth: 1))
            .padding(.horizontal, 24)
        }
    }

    // MARK: Sheets

    @ViewBuilder
    private func sheetView(_ which: EditorSheet) -> some View {
        switch which {
        case .overlay:
            OverlayGallerySheet(onPick: { url in
                let at = playback.currentTime
                Task {
                    await model.addMediaOverlay(fileURL: url, start: at, end: at + 4)
                    if let added = model.mediaOverlays.last?["overlay_id"]?.stringValue {
                        selection = .overlay(added, isText: false)
                    }
                }
            }, sampleTiles: previewState != nil)
        case .broll:
            BrollView(model: model, initialPlacementID: nil)
        case .export:
            ExportSheet(model: model)
        case .captions:
            AddCaptionsSheet(model: model, playhead: playback.currentTime)
        case .voiceOver:
            EmptyView()
        }
    }

    // MARK: Actions

    private func toolTapped(_ tool: EditorTool) {
        switch tool {
        case .split: Task { await split() }
        case .delete: Task { await delete() }
        case .mirror:
            var clipID: String? = nil
            if case .video(let id) = selection { clipID = id }
            Task {
                let on = await model.toggleMirror(clipID: clipID)
                show(.message(clipID == nil ? (on ? "Video mirrored" : "Mirror off") : (on ? "Clip mirrored" : "Clip mirror off")))
            }
        case .rotate:
            let next = (model.rotationDegrees + 90) % 360
            Task { await model.setDraftValue("rotation", .number(Double(next))) }
            show(.message("Rotated \(next)°"))
        case .overlay: choosingOverlay = true
        case .voiceOver: openVoiceOver()
        case .captions:
            playback.pause()
            panel = panel == .captions(nil) ? .none : .captions(nil)
        case .style:
            panel = panel == .style ? .none : .style
        }
    }

    private func openVoiceOver() {
        playback.pause()
        switch AVAudioApplication.shared.recordPermission {
        case .granted: panel = .voiceRecord
        case .denied: show(.message("Allow the microphone in iOS Settings to record a voice-over."))
        default: panel = .micPermission
        }
    }

    private func startCaptionEditing(_ clipID: String) {
        captionText = captionTextFor(clipID)
        panel = .captions(clipID)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { captionFocused = true }
    }

    private func captionTextFor(_ clipID: String) -> String {
        let clip = model.selectedClips.first { $0["clip_id"]?.stringValue == clipID }
        return clip?["caption_text"]?.stringValue ?? clip?["text"]?.stringValue ?? ""
    }

    private func finishCaptionEditing() {
        captionFocused = false
        guard let clipID = editingCaptionClip else { return }
        let text = captionText.trimmingCharacters(in: .whitespacesAndNewlines)
        panel = .captions(nil)
        guard text != captionTextFor(clipID).trimmingCharacters(in: .whitespacesAndNewlines) else { return }
        Task { await model.editCaption(clipID: clipID, text: text) }
    }

    private func split() async {
        let time = playback.currentTime
        switch selection {
        case .voiceOver(let id):
            await model.splitVoiceOverPlacement(id: id, at: time)
            show(.message("Voice-over split at \(EditorTime.short(time))"))
        case .broll(let id):
            await model.splitBrollPlacement(id: id, at: time)
            show(.message("B-roll split at \(EditorTime.short(time))"))
        default:
            guard let window = EditorTracks.clipWindows(model).first(where: { time > $0.start + 0.05 && time < $0.end - 0.05 }) else {
                show(.message("Move the cursor inside a clip to split it."))
                return
            }
            let wanted = window.sourceStart + (time - window.start)
            guard let at = model.safeSplitTime(clipID: window.clipID, near: wanted) else {
                show(.message("This clip can’t be split here. Move the cursor between two words."))
                return
            }
            show(.split(EditorTime.tenths(time)))
            await model.split(clipID: window.clipID, at: at)
        }
    }

    private func delete() async {
        let current = selection
        selection = nil
        switch current {
        case .video(let id):
            show(.message("Clip deleted", undo: true))
            await model.remove(clipID: id)
        case .voiceOver(let id):
            show(.message("Voice-over deleted"))
            await model.removeVoiceOverPlacement(id: id)
        case .broll(let id):
            show(.message("B-roll deleted"))
            await model.removeBrollPlacement(id: id)
        case .overlay(let id, let isText):
            show(.message("Overlay deleted", undo: true))
            if isText { await model.removeTextOverlay(id: id) } else { await model.removeMediaOverlay(id: id) }
        case .caption(let clipID):
            show(.captionDeleted)
            await model.editCaption(clipID: clipID, text: "")
        case .zoom(let id):
            show(.message("Zoom deleted", undo: true))
            await model.removeZoomKeyframe(id: id)
        case nil:
            if let clipID = editingCaptionClip {
                show(.captionDeleted)
                captionText = ""
                panel = .captions(nil)
                await model.editCaption(clipID: clipID, text: "")
            } else {
                show(.message("Select a clip, voice-over, overlay or caption first."))
            }
        }
    }

    private func undoFromToast() async {
        toast = nil
        await model.undo()
    }

    private func show(_ value: EditorToast) {
        toast = value
        let shown = value
        Task {
            try? await Task.sleep(nanoseconds: 2_600_000_000)
            if toast == shown { toast = nil }
        }
    }

    private func hintPill(_ text: String) -> some View {
        Text(text)
            .font(CS.inter(11, .semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 14).frame(height: 26)
            .background(Color(hex: 0x050814, opacity: 0.92), in: Capsule())
            .overlay(Capsule().stroke(Color(hex: 0x2A4A80), lineWidth: 1))
    }

    private func rename() async {
        let title = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty, let session = appState.session else { return }
        struct Body: Encodable { let user_id: String; let title: String }
        displayTitle = title
        do {
            let updated: Project = try await APIClient.shared.request(
                "/v1/projects/\(project.projectID)", method: "PATCH",
                body: Body(user_id: session.userID, title: String(title.prefix(120)))
            )
            displayTitle = updated.title
            try? await appState.refreshProjects()
        } catch {
            displayTitle = nil
            model.errorMessage = "Couldn’t rename the cut. Please try again."
        }
    }

    #if DEBUG
    /// Design screenshots only: an 8 s spoken-like tone so the review shows a real waveform.
    static func sampleTone() -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sample-voice.caf")
        let rate = 44_100.0
        guard let format = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(rate * 8)),
              let file = try? AVAudioFile(forWriting: url, settings: format.settings) else { return nil }
        buffer.frameLength = buffer.frameCapacity
        let data = buffer.floatChannelData![0]
        for i in 0..<Int(buffer.frameLength) {
            let t = Double(i) / rate
            let envelope = abs(sin(t * 2.3)) * (0.4 + 0.6 * abs(sin(t * 7.1)))
            data[i] = Float(envelope * sin(2 * .pi * 180 * t) * 0.6)
        }
        try? file.write(from: buffer)
        return url
    }
    #endif

    // MARK: Caption highlight helpers

    static func highlightColor(_ preset: String) -> Color? {
        switch preset {
        case "highlight", "highlight_green": return Color(hex: 0x39FF78)
        case "highlight_red": return Color(hex: 0xFF453A)
        case "highlight_blue": return Color(hex: 0x1AA6FF)
        default: return nil
        }
    }

    /// Caption text with one word lit in the highlight color (the longest word, like the Figma sample).
    static func highlightedText(_ text: String, color: Color?) -> Text {
        let words = text.split(separator: " ").map(String.init)
        guard let color, let lit = words.max(by: { $0.count < $1.count }) else {
            return Text(text).foregroundColor(.white)
        }
        var result = Text("")
        var done = false
        for (index, word) in words.enumerated() {
            let piece = Text(word + (index < words.count - 1 ? " " : ""))
            if !done && word == lit { result = result + piece.foregroundColor(color); done = true }
            else { result = result + piece.foregroundColor(.white) }
        }
        return result
    }
}

// MARK: - Panels and tools

enum EditorPanel: Equatable {
    case none
    /// Captions style menu (2b); with a clip id, that caption is being written on the video.
    case captions(String?)
    case voiceRecord
    case voiceReview
    case micPermission
    case style
}

enum EditorTool: String, CaseIterable {
    case split, delete, mirror, rotate, overlay, voiceOver, captions, style

    var title: String {
        switch self {
        case .split: return "Split"
        case .delete: return "Delete"
        case .mirror: return "Mirror"
        case .rotate: return "Rotate"
        case .overlay: return "Overlay"
        case .voiceOver: return "Voice Over"
        case .captions: return "Captions"
        case .style: return "Style"
        }
    }

    var icon: String {
        switch self {
        case .split: return "Ed4Split"
        case .delete: return "Ed4Delete"
        case .mirror: return "Ed4Mirror"
        case .rotate: return "Ed4Rotate"
        case .overlay: return "Ed4Overlay"
        case .voiceOver: return "Ed4Voice"
        case .captions: return "Ed4Captions"
        case .style: return "Ed4Style"
        }
    }
}

/// Editor/Toolbar/v4 (Figma 1087:2573): outline gradient icons without boxes, 53 pt per tool,
/// slides sideways (Style is last), active tool in cyan.
struct ToolbarV4: View {
    /// The home indicator area under the toolbar (the editor draws to the bottom edge).
    static var bottomInset: CGFloat {
        (UIApplication.shared.connectedScenes.first as? UIWindowScene)?.windows.first?.safeAreaInsets.bottom ?? 0
    }

    /// 6 + 51 (icon 34 · 4 · label) + bottom padding (at least 10, or what the home indicator needs).
    static var height: CGFloat { 6 + 51 + max(10, bottomInset - 14) }

    let active: EditorTool?
    let onTap: (EditorTool) -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(alignment: .top, spacing: 0) {
                ForEach(EditorTool.allCases, id: \.self) { tool in
                    Button { onTap(tool) } label: {
                        VStack(spacing: 4) {
                            if active == tool {
                                Image(tool.icon).renderingMode(.template).resizable().foregroundStyle(CS.cyan)
                                    .frame(width: 34, height: 34)
                            } else {
                                Image(tool.icon).resizable().frame(width: 34, height: 34)
                            }
                            Text(tool.title)
                                .font(CS.inter(10.5, .medium))
                                .foregroundStyle(active == tool ? CS.cyan : CS.textSecondaryF)
                                .lineLimit(1)
                                .fixedSize()
                        }
                        .frame(width: 53, height: 51)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("tool_\(tool.rawValue)")
                    if tool == .rotate {
                        Rectangle().fill(Color(hex: 0x14336B)).frame(width: 1, height: 40)
                    }
                }
            }
            .padding(.leading, 4)
            .padding(.trailing, 6)
        }
        .padding(.top, 6)
        .frame(height: Self.height, alignment: .top)
        .background(Color(hex: 0x02040A))
        .overlay(alignment: .top) { Rectangle().fill(Color(hex: 0x14336B, opacity: 0.7)).frame(height: 1) }
    }
}

/// Export quality (Figma: "4K Ultra HD ▾" in the logo gradient). Honest: the export makes 1080p
/// today, so 2K and 4K are listed as coming soon instead of pretending.
struct ExportQualityMenu: View {
    private static let gradient = LinearGradient(
        colors: [CS.cyan, Color(hex: 0x1769FF), Color(hex: 0x7C3AED), Color(hex: 0xFF21D7)],
        startPoint: .leading, endPoint: .trailing
    )

    var body: some View {
        Menu {
            Button {} label: { Label("1080p Full HD", systemImage: "checkmark") }
            Button("2K · coming soon") {}.disabled(true)
            Button("4K Ultra HD · coming soon") {}.disabled(true)
        } label: {
            HStack(spacing: 6) {
                Text("1080p HD")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(Self.gradient)
                    .lineLimit(1)
                Image("Ed4Chevron").resizable().frame(width: 10, height: 10)
            }
            .padding(.leading, 12).padding(.trailing, 10)
            .frame(width: 108, height: 32)
            .background(CS.surfaceCard, in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(CS.borderDefault, lineWidth: 1))
        }
        .accessibilityLabel("Export quality: 1080p")
        .accessibilityIdentifier("exportQuality")
    }
}

enum SoundPanelTarget: Equatable { case video, voice }

/// "Video sound" panel (under the timeline): mute button, title, a slider whose dot is exactly at
/// the % shown, and the %. `maximum` 2 = 0–200% with a mark at 100%; 1 = 0–100%.
struct SoundPanelV4: View {
    static let height: CGFloat = 46

    let title: String
    let icon: String
    @Binding var value: Double
    let maximum: Double
    let onClose: () -> Void
    let onCommit: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Button {
                value = value > 0.001 ? 0 : 1
                onCommit()
            } label: {
                Image(systemName: value > 0.001 ? icon : "speaker.slash.fill")
                    .font(.system(size: 13, weight: .semibold)).foregroundStyle(CS.cyan).frame(width: 26, height: 30)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(value > 0.001 ? "Mute" : "Sound on")
            Text(title).font(CS.inter(11, .semibold)).foregroundStyle(.white).fixedSize()
            SoundSlider(value: $value, maximum: maximum, onCommit: onCommit)
                .frame(height: 30)
            Text(SoundSlider.percentText(value))
                .font(CS.sf(11, .semibold)).monospacedDigit().foregroundStyle(CS.textSecondaryF)
                .frame(width: 40, alignment: .trailing)
                .accessibilityIdentifier("soundPercent")
            Button(action: onClose) {
                Image(systemName: "xmark").font(.system(size: 11, weight: .bold)).foregroundStyle(CS.textSecondaryF)
                    .frame(width: 24, height: 30)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Close")
        }
        .padding(.horizontal, 12)
        .frame(height: 36)
        .background(Color(hex: 0x0A1326, opacity: 0.97), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color(hex: 0x14336B), lineWidth: 1))
        .padding(.horizontal, 16)
        .padding(.vertical, 5)
        .frame(height: Self.height)
        .accessibilityIdentifier("volumeStrip")
    }
}

/// Slider drawn by hand so the dot's centre is exactly at `value / maximum` of the track
/// (the system slider insets its ends, so the dot and the % disagreed). Snaps to 100%.
struct SoundSlider: View {
    @Binding var value: Double
    let maximum: Double
    let onCommit: () -> Void
    static let knob: CGFloat = 18

    static func percentText(_ value: Double) -> String { "\(Int((value * 100).rounded()))%" }

    /// Value for a finger at `x` on a track `width` wide (dot centre = finger); snaps within 3% of 100%.
    static func value(atX x: CGFloat, width: CGFloat, maximum: Double) -> Double {
        let usable = max(1, width - knob)
        let fraction = Double(min(max(0, (x - knob / 2) / usable), 1))
        var result = (fraction * maximum * 100).rounded() / 100
        if abs(result - 1) < 0.03 { result = 1 }
        return result
    }

    /// Dot centre for `value` on a track `width` wide.
    static func knobCenter(value: Double, width: CGFloat, maximum: Double) -> CGFloat {
        let usable = max(1, width - knob)
        return knob / 2 + usable * CGFloat(min(max(0, value / maximum), 1))
    }

    var body: some View {
        GeometryReader { proxy in
            let width = proxy.size.width
            let center = Self.knobCenter(value: value, width: width, maximum: maximum)
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.18)).frame(height: 4)
                    .padding(.horizontal, Self.knob / 2)
                Capsule().fill(CS.cyan).frame(width: max(0, center - Self.knob / 2), height: 4)
                    .offset(x: Self.knob / 2)
                if maximum > 1 {
                    // 100% mark.
                    Rectangle().fill(Color.white.opacity(0.55)).frame(width: 2, height: 10)
                        .offset(x: Self.knobCenter(value: 1, width: width, maximum: maximum) - 1)
                }
                Circle().fill(.white).frame(width: Self.knob, height: Self.knob)
                    .shadow(color: .black.opacity(0.35), radius: 2, y: 1)
                    .offset(x: center - Self.knob / 2)
            }
            .frame(height: proxy.size.height)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { drag in value = Self.value(atX: drag.location.x, width: width, maximum: maximum) }
                    .onEnded { _ in onCommit() }
            )
        }
        .accessibilityElement()
        .accessibilityLabel("Volume")
        .accessibilityValue(Self.percentText(value))
        .accessibilityAdjustableAction { direction in
            let step = 0.05
            switch direction {
            case .increment: value = min(maximum, value + step)
            case .decrement: value = max(0, value - step)
            @unknown default: break
            }
            onCommit()
        }
        .accessibilityIdentifier("soundSlider")
    }
}

private struct ToastPillV4: View {
    let toast: EditorToast
    let onUndo: () -> Void

    var body: some View {
        HStack {
            Text(toast.text)
                .font(CS.inter(13, .medium))
                .foregroundStyle(.white)
                .lineLimit(2)
            if toast.hasUndo {
                Spacer()
                Button("Undo", action: onUndo)
                    .font(CS.inter(13, .semibold))
                    .foregroundStyle(CS.cyan)
            }
        }
        .padding(.horizontal, 16)
        .frame(minHeight: 40)
        .background(Color(hex: 0x040812, opacity: 0.97), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(hex: 0x7C3AED, opacity: 0.8), lineWidth: 1))
        .padding(.horizontal, 10)
        .accessibilityIdentifier("editorToast")
    }
}

/// A photo/video overlay on the video. Selected: white box with corner handles; drag to move,
/// pinch to resize (saved when the fingers lift).
private struct MediaOverlayOnVideo: View {
    let item: [String: JSONValue]
    let frame: CGSize
    let time: Double
    let isSelected: Bool
    let onSelect: () -> Void
    let onChange: (Double, Double, Double) -> Void

    @State private var drag: CGSize = .zero
    @State private var pinch: CGFloat = 1

    private var start: Double { item["start"]?.doubleValue ?? 0 }
    private var end: Double { item["end"]?.doubleValue ?? start }
    private var x: Double { item["x"]?.doubleValue ?? 0.5 }
    private var y: Double { item["y"]?.doubleValue ?? 0.5 }
    private var width: Double { item["width"]?.doubleValue ?? 0.4 }
    private var visible: Bool { isSelected || (time >= start && time < end) }

    var body: some View {
        if visible {
            thumb
                .position(x: frame.width * CGFloat(x) + drag.width, y: frame.height * CGFloat(y) + drag.height)
                .onTapGesture { onSelect() }
                .gesture(moveGesture.simultaneously(with: resizeGesture))
                .accessibilityIdentifier("mediaOverlayOnVideo")
        }
    }

    private var thumb: some View {
        let w = frame.width * CGFloat(width) * pinch
        return OverlayMediaThumb(item: item)
            .frame(width: w, height: w * 0.42)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay { if isSelected { SelectionHandles() } }
    }

    private var moveGesture: some Gesture {
        DragGesture()
            .onChanged { value in if isSelected { drag = value.translation } }
            .onEnded { value in
                guard isSelected else { return }
                let nx = max(0, min(1, x + Double(value.translation.width / frame.width)))
                let ny = max(0, min(1, y + Double(value.translation.height / frame.height)))
                drag = .zero
                onChange(nx, ny, width)
            }
    }

    private var resizeGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in if isSelected { pinch = value } }
            .onEnded { value in
                guard isSelected else { return }
                let nw = max(0.1, min(1, width * Double(value)))
                pinch = 1
                onChange(x, y, nw)
            }
    }
}

/// White box with the four corner handles around a selected item.
private struct SelectionHandles: View {
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 2).stroke(.white, lineWidth: 1.5).padding(-4)
            GeometryReader { proxy in
                ForEach(0..<4, id: \.self) { index in
                    Circle().fill(.white).frame(width: 10, height: 10)
                        .position(x: index % 2 == 0 ? -4 : proxy.size.width + 4, y: index < 2 ? -4 : proxy.size.height + 4)
                }
            }
        }
    }
}

/// The overlay's picture: its preview link if the server sent one, else a labelled tile.
private struct OverlayMediaThumb: View {
    let item: [String: JSONValue]

    var body: some View {
        let link = item["preview_url"]?.stringValue ?? item["playback_url"]?.stringValue
        ZStack {
            LinearGradient(colors: [CS.cyan, Color(hex: 0x7C3AED)], startPoint: .leading, endPoint: .trailing)
            if let link, let url = URL(string: link) {
                AsyncImage(url: url) { image in image.resizable().scaledToFill() } placeholder: { Color.clear }
            } else {
                Text((item["kind"]?.stringValue ?? "photo").capitalized)
                    .font(CS.inter(14, .bold)).foregroundStyle(.white)
            }
        }
    }
}
