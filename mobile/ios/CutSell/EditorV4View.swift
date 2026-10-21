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
    @State private var pixelsPerSecond: CGFloat = 30
    @State private var captionText = ""
    @FocusState private var captionFocused: Bool
    /// Editor v5 panel above the toolbar (overlay, volume, speed, voice-over, sounds…).
    @State private var card: EditorPanelKind?
    /// 3 · 2 · 1 over the video before a voice-over records.
    @State private var countdown: Int?
    /// The voice-over sheet follows a finger dragging it down; past 80 pt it asks to close.
    @State private var voiceSheetDrag: CGFloat = 0
    /// "Discard recording?" (closing with a take in progress or not saved).
    @State private var confirmDiscardVoice = false
    /// "Record again" on a voice-over: the take replaces that one.
    @State private var recordAgainID: String?
    @State private var musicPicker = false
    @State private var musicNotice = false
    @AppStorage("cutsell.musicRightsNoticeShown") private var musicNoticeShown = false
    /// Build 17: the preview rebuilds when a sound finishes downloading or a clean copy is ready.
    @ObservedObject private var sounds = SfxLibrary.shared
    @ObservedObject private var cleaning = CleanAudioCenter.shared
    /// Build 18: what the preview mixes (probe label; Diagnostics reads the same).
    @ObservedObject private var diagnostics = EditorDiagnostics.shared
    /// A framing made with the fingers at the cursor, not kept yet (◇ keeps it as a keyframe).
    @State private var keyPending: TransformValue?
    @State private var keyPendingTarget: KeyframeTarget?
    @State private var keyPendingTime: Double = 0
    @State private var keyGestureBase: TransformValue?
    @State private var lastVideoTap = Date.distantPast
    /// The last moment the fingers framed the video (a tap right after is part of that gesture).
    @State private var lastFraming = Date.distantPast

    private var duration: Double { max(0.1, playback.duration > 0 ? playback.duration : model.timelineDuration) }

    /// Rebuild the preview when what it plays changes (the predicted draft included, so it follows at once).
    private var previewKey: String {
        let placements = model.previewVoices.map { "\($0.key)@\($0.start)" }.joined(separator: ",")
        let cleanReady = cleaning.states.values.filter { if case .ready = $0 { return true } else { return false } }.count
        return "\(model.displayDraft?.hashValue ?? 0)|\(placements)|\(model.voiceOverVolume)|\(model.loadGeneration)|\(sounds.ready.count)|\(cleanReady)"
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
                // Figma 1119:2316 / 1120:4: the voice-over sheet in two steps, over the timeline. It closes like the
                // other panels: drag it down, tap outside it, or its X (a take in progress asks first).
                VStack(spacing: 0) {
                    Color.black.opacity(0.001)
                        .contentShape(Rectangle())
                        .onTapGesture { requestCloseVoiceSheet() }
                        .accessibilityIdentifier("voiceSheetOutside")
                    VoiceOverSheetV5(model: model, recorder: recorder, start: playback.currentTime, replacing: recordAgainID,
                                     countdown: $countdown) { outcome in
                        panel = .none
                        let replaced = recordAgainID != nil
                        recordAgainID = nil
                        if outcome != .failed {
                            let number = model.voiceClips.count + model.pendingVoiceOvers.filter { !$0.isMusic }.count
                            show(.message(replaced ? "Voiceover recorded again · tap ↶ to undo"
                                                   : "Voiceover \(max(1, number)) added · tap ↶ to undo", undo: true))
                        }
                    } onCancel: {
                        requestCloseVoiceSheet()
                    }
                    .padding(.horizontal, 1)
                    .offset(y: max(0, voiceSheetDrag))
                    .gesture(DragGesture(minimumDistance: 12)
                        .onChanged { voiceSheetDrag = $0.translation.height }
                        .onEnded { value in
                            let close = value.translation.height > 80 || value.predictedEndTranslation.height > 200
                            withAnimation(.easeOut(duration: 0.2)) { voiceSheetDrag = 0 }
                            if close { requestCloseVoiceSheet() }
                        })
                }
                .ignoresSafeArea(.container, edges: .bottom)
                .transition(.move(edge: .bottom))
                .alert("Discard recording?", isPresented: $confirmDiscardVoice) {
                    Button("Discard", role: .destructive) { cancelVoiceSheet() }
                    Button("Keep recording", role: .cancel) {}
                } message: {
                    Text("Your voice-over hasn’t been saved.")
                }
            }
            if panel == .micPermission {
                micPermissionCard
            }
            #if DEBUG
            if let probe = previewState?.probe, probe == "b17_ui" || probe.hasPrefix("b18") {
                // Build 17 / 18 probes only: the editor's state as text, read by the UI test.
                Text(b17State).font(.system(size: 1)).opacity(0.02).frame(width: 2, height: 2)
                    .allowsHitTesting(false)
                    .accessibilityLabel(b17State)
                    .accessibilityIdentifier("b17State")
                Text(diagnostics.mix.summary).font(.system(size: 1)).opacity(0.02).frame(width: 2, height: 2)
                    .allowsHitTesting(false)
                    .accessibilityLabel(diagnostics.mix.summary)
                    .accessibilityIdentifier("b18Mix")
                // Probe buttons on the left margin, beside the video (nothing else lives there).
                HStack {
                    VStack(spacing: 4) {
                        Spacer()
                        // Moves the cursor exactly 1 s on (a play / pause tap overshoots: the UI test
                        // waits for the app to go idle while the video plays).
                        probeButton("b17Step") { playback.seek(to: playback.currentTime + 1) }
                        probeButton("b18Back") { playback.seek(to: max(0, playback.currentTime - 1)) }
                        probeButton("b18Half") { playback.seek(to: playback.currentTime + 0.5) }
                        probeButton("b18Start") { playback.seek(to: 0) }
                        probeButton("b18AIPeak") {
                            if let peak = ProbeMedia.firstAIPlateau(model.videoKeyframeMarks) { playback.seek(to: peak) }
                        }
                        probeButton("b18Reload") { Task { await model.load() } }
                        probeButton("b18AddPhoto") {
                            if let url = ProbeMedia.makePhoto() { model.addOverlay(fileURL: url, at: playback.currentTime) }
                        }
                        probeButton("b18AddVideo") {
                            Task { if let url = await ProbeMedia.makeVideo(seconds: 3) { model.addOverlay(fileURL: url, at: playback.currentTime) } }
                        }
                        Spacer()
                    }
                    Spacer()
                }
            }
            #endif
        }
    }

    #if DEBUG
    private var b17State: String {
        let marks = model.videoKeyframeMarks
        let sounds = model.soundClips.map { String(format: "%@@%.2f+%.2f%@", $0.sfxID ?? $0.kind.rawValue, $0.start, $0.length, $0.muted ? "m" : "") }
        let keys = model.transformKeyframes.filter { !$0.isAI }.map { String(format: "%.2f", $0.value.scale) }
        let rotations = model.transformKeyframes.filter { !$0.isAI }.map { String(format: "%.0f", $0.value.rotation) }
        let mutedClips = model.selectedClips.filter { $0["audio_muted"]?.boolValue == true }.count
        let shown = shownVideoTransform
        let overlays = model.overlaysOnScreen.map { String(format: "%@:%.2f-%.2f:w%.2f:r%.0f%@", $0.kind == .video ? "video" : "photo",
                                                          $0.start, $0.end, $0.width, $0.style.rotation, $0.uploading != nil ? "u" : "") }
        let pending = keyPending.map { String(format: "%.3f/%.1f", $0.scale, $0.rotation) } ?? "none"
        let voices = model.audioClips.filter { $0.kind == .voice }.sorted { $0.start < $1.start }
            .map { String(format: "%.2f+%.2f%@%@%@v%.2f", $0.start, $0.length, $0.duckVideo ? "d" : "", $0.clean ? "c" : "", $0.muted ? "m" : "", $0.volume) }
        return "ai=\(marks.filter(\.isAI).count) user=\(marks.filter { !$0.isAI }.count) sounds=\(sounds.joined(separator: ",")) "
            + "voices=\(voices.joined(separator: ",")) "
            + "muted_clips=\(mutedClips) user_scales=\(keys.joined(separator: ",")) user_rot=\(rotations.joined(separator: ",")) "
            + "auto_zoom=\(model.autoZoomEnabled) overlays=\(overlays.joined(separator: ",")) sel=\(selectionName) pending=\(pending) "
            + String(format: "vv=%.2f t=%.2f zoom=%.3f rot=%.1f ", model.videoVolume, playback.currentTime, shown.scale, shown.rotation)
            + "playing=\(playback.isPlaying) "
            + "session=\(AVAudioSession.sharedInstance().category.rawValue.replacingOccurrences(of: "AVAudioSessionCategory", with: ""))/"
            + AVAudioSession.sharedInstance().mode.rawValue.replacingOccurrences(of: "AVAudioSessionMode", with: "")
    }

    private var selectionName: String {
        switch selection {
        case .video?: return "video"
        case .sound?: return "sound"
        case .voiceOver?: return "voice"
        case .overlay(let id, _)?:
            return model.overlaysOnScreen.contains { $0.id == id } ? "overlay" : "overlay_gone"
        case .zoom?: return "keyframe"
        case nil: return "none"
        default: return "other"
        }
    }

    private func probeButton(_ id: String, action: @escaping () -> Void) -> some View {
        Button(action: action) { Color.white.opacity(0.05).frame(width: 28, height: 28) }
            .accessibilityIdentifier(id)
    }
    #endif

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
            // A tall panel makes the video smaller, so the timeline stays in view (Editor v5).
            return controls + 1 + TimelineV4.height(for: model, compact: card != nil) + toolbar + cardHeight
        }
    }

    private var cardHeight: CGFloat {
        guard let card else { return 0 }
        if case .overlay(let id) = card {
            return 8 + card.height(isVideoOverlay: model.overlaysOnScreen.first { $0.id == id }?.kind == .video) + 8
        }
        return 8 + card.height() + 8
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
            .onChange(of: selection) { _, newValue in
                selectionChanged(newValue)
                clearKeyPending()
            }
            .onChange(of: playback.currentTime) { _, time in
                // A framing not kept with ◇ belongs to its moment: moving the cursor drops it.
                if keyPending != nil, keyGestureBase == nil, abs(time - keyPendingTime) > 0.03 { clearKeyPending() }
            }
            // An overlay that finished uploading stays selected under its new id.
            .onChange(of: model.placedOverlayRename) { _, rename in
                guard let rename else { return }
                if selection == .overlay(rename.from, isText: false) { selection = .overlay(rename.to, isText: false) }
                if keyPendingTarget == .overlay(rename.from) { clearKeyPending() }
            }
            // A photo / video just picked for an overlay comes selected (move it, pinch it).
            .onChange(of: model.lastAddedOverlayID) { _, id in
                guard let id else { return }
                selection = .overlay(id, isText: false)
                if model.pendingOverlays.contains(where: { $0.id == id }) {
                    let name = model.overlayTitle(id)
                    show(.message(model.lastAddedOverlayNewLayer ? "\(name) added on a new layer · tap ↶ to undo"
                                                                  : "\(name) added", undo: true))
                }
            }
            .fileImporter(isPresented: $musicPicker, allowedContentTypes: MusicImport.types) { result in
                guard case .success(let url) = result else { return }
                let at = playback.currentTime
                Task {
                    guard let song = await MusicImport.m4a(from: url) else {
                        model.errorMessage = "That song couldn't be read. Please pick another file."
                        return
                    }
                    if await model.addRecordedVoiceOver(fileURL: song, at: at, music: true) != .failed {
                        show(.message("Music added · tap ↶ to undo", undo: true))
                    }
                }
            }
            .alert("Music from Files", isPresented: $musicNotice) {
                Button("OK") { musicNoticeShown = true; musicPicker = true }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Use only music you own or have rights to.")
            }
            .onAppear {
                ExportTracker.shared.visibleProjectID = project.projectID
                appeared()
            }
            .onDisappear {
                if ExportTracker.shared.visibleProjectID == project.projectID { ExportTracker.shared.visibleProjectID = nil }
                playback.pause()
                if recorder.state == .recording { recorder.stop() }
            }
    }

    private func appeared() {
        EditorAudio.preparePlayback()
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
        if previewState.probe == "media" {
            Task { await EditorMediaProbe.run(model: model) { selection = $0 } }
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
        if let probe = previewState.probe, probe.hasPrefix("v5_") {
            // Editor v5 panels on the real cut (design screenshots).
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                switch probe {
                case "v5_overlay_add": openCard(.overlayAdd)
                case "v5_volume": openCard(.volume)
                case "v5_speed":
                    if let first = model.selectedClips.first?["clip_id"]?.stringValue { selection = .video(first) }
                    openCard(.speed)
                case "v5_audio": openCard(.audio)
                case "v5_countdown":
                    panel = .voiceRecord
                    countdown = 3
                default: break
                }
            }
            return
        }
        if previewState.probe == "v5" {
            Task { await EditorV5Probe.run(model: model, playback: playback, select: { selection = $0 },
                                           open: { openCard($0) }, toast: { show(.message($0)) },
                                           countdown: { countdown = $0 }) }
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
            .alert("CutSell", isPresented: errorShown) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(DraftEditorViewModel.plainWords(model.errorMessage))
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
            let middle = first.source(at: (first.start + first.end) / 2)
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
        openCard(.volume)
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
            // Export starts the moment it is tapped; the sheet shows its real progress.
            Button {
                playback.pause()
                sheet = .export
                if !model.isExporting { Task { await model.export() } }
            } label: {
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
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier("editorVideo")
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
            splitAwareVideo(size: size)
            // Build 18: pinch / twist / drag with UIKit's recognizers. Under the overlays while a
            // video clip is framed (an overlay can still be touched); over them while a Float overlay
            // is framed (the fingers anywhere on the video move, size and turn it).
            if framingEnabled, !framingOverlay { framingLayer(size: size) }
            overlaysOnVideo(frame: size)
            if framingEnabled, framingOverlay { framingLayer(size: size) }
            if model.captionsEnabled {
                captionLayer(frame: size)
            }
            if let broll = model.pendingBroll {
                brollPill(broll).frame(maxHeight: .infinity, alignment: .bottom).padding(.bottom, 14)
            }
            // Dragging a clip's edge: the original's exact frame at the handle, and "+0,4 s".
            if let scrub = model.trimScrub {
                TrimFrameOverlay(scrub: scrub, url: model.previewVideoURL(forSource: scrub.sourceID))
                    .frame(width: size.width, height: size.height)
            }
            if let countdown {
                CountdownOverlayV5(value: countdown).frame(width: size.width, height: size.height)
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
        } else if soundsMissing {
            soundDownloadPill.padding(.bottom, 12)
        } else if case .overlay(let id, isText: false) = selection, model.overlayStyle(id).layout == .float {
            hintPill("Drag · pinch · turn with two fingers").padding(.bottom, 14)
        }
    }

    /// A sound on the timeline whose file is not on this iPhone yet.
    private var soundsMissing: Bool {
        model.soundClips.contains { $0.kind == .sfx && $0.sfxID.flatMap { SfxLibrary.localURL($0) } == nil }
    }

    /// "Getting sounds…" while they download; "Sound not downloaded · Try again" when it failed.
    private var soundDownloadPill: some View {
        HStack(spacing: 10) {
            if sounds.failed && !sounds.loading {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(Color(hex: 0xFFD60A))
                Text("A sound didn't download").font(CS.inter(13, .medium)).foregroundStyle(.white)
                Button("Try again") { sounds.retry() }
                    .font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
                    .accessibilityIdentifier("soundRetry")
            } else {
                ProgressView().controlSize(.small).tint(.white)
                Text("Getting sounds…").font(CS.inter(13, .medium)).foregroundStyle(.white)
            }
        }
        .padding(.horizontal, 14).frame(height: 36)
        .background(Color(hex: 0x040812, opacity: 0.9), in: Capsule())
        .accessibilityIdentifier("soundDownloadPill")
    }

    /// "Adding video… 45%" while a full-screen B-roll uploads; "Try again" if it failed.
    private func brollPill(_ broll: PendingBroll) -> some View {
        HStack(spacing: 10) {
            if broll.failed {
                Text("The video wasn't added").font(CS.inter(13, .medium)).foregroundStyle(.white)
                Button("Try again") { model.retryBroll() }.font(CS.inter(13, .semibold)).foregroundStyle(CS.cyan)
                Button { model.dismissFailedBroll() } label: { Image(systemName: "xmark").foregroundStyle(CS.textMutedF) }
                    .accessibilityLabel("Dismiss")
            } else {
                ProgressView().controlSize(.small).tint(.white)
                Text("Adding video… \(Int((broll.progress * 100).rounded()))%")
                    .font(CS.inter(13, .medium)).monospacedDigit().foregroundStyle(.white)
            }
        }
        .padding(.horizontal, 14).frame(height: 36)
        .background(Color(hex: 0x040812, opacity: 0.9), in: Capsule())
        .accessibilityIdentifier("brollUpload")
    }

    private func videoTapped() {
        // SwiftUI's tap and the framing layer's tap can both see one touch: one toggle only. A touch
        // that pinched, twisted or dragged the video is never a tap.
        guard Date().timeIntervalSince(lastVideoTap) > 0.35, keyGestureBase == nil,
              Date().timeIntervalSince(lastFraming) > 0.6 else { return }
        lastVideoTap = Date()
        if captionFocused { finishCaptionEditing(); return }
        playback.togglePlayback()
    }

    static func fit(_ target: CGSize, in box: CGSize) -> CGSize {
        let scale = min(1, box.width / target.width, box.height / target.height)
        return CGSize(width: target.width * scale, height: target.height * scale)
    }

    /// The overlay on screen now whose layout also moves the video (Stacked / Side by side).
    private var splitOverlay: OverlayOnScreen? {
        let now = playback.currentTime
        return model.overlaysOnScreen
            .filter { now >= $0.start && now < $0.end && $0.style.layout.baseRegion != nil }
            .max { $0.style.layer < $1.style.layer }
    }

    /// The video in the whole frame, or in its half while a Stacked / Side by side overlay shows
    /// (filling that half, centred, like the export).
    @ViewBuilder
    private func splitAwareVideo(size: CGSize) -> some View {
        if let region = splitOverlay?.style.layout.baseRegion {
            videoLayer(fill: true)
                .frame(width: size.width * region.width, height: size.height * region.height)
                .clipped()
                .frame(width: size.width, height: size.height, alignment: region.minX == 0 && region.width < 1 ? .leading : .top)
        } else {
            videoLayer(fill: false)
        }
    }

    /// The video, mirrored and rotated as the export will make it (whole video + this clip).
    private func videoLayer(fill: Bool) -> some View {
        let degrees = model.rotation(at: playback.currentTime)
        let quarter = degrees % 180 != 0
        let key = shownVideoTransform
        return GeometryReader { box in
            PlayerLayerView(player: playback.player, fill: fill)
                .scaleEffect(x: model.isMirrored(at: playback.currentTime) ? -1 : 1, y: 1)
                .rotationEffect(.degrees(Double(degrees)))
                .scaleEffect(quarter ? 266.0 / 472.0 : 1)
                // Keyframes / Auto Zoom AI, as the export draws them: zoom about the centre, turn
                // clockwise, then move by a fraction of the frame.
                .scaleEffect(key.scale)
                .rotationEffect(.degrees(key.rotation))
                .offset(x: box.size.width * key.x, y: box.size.height * key.y)
                .frame(width: box.size.width, height: box.size.height)
                // Zoomed past the card, the picture must not take the taps meant for ◇, Play, Undo.
                .allowsHitTesting(false)
        }
            .animation(.easeOut(duration: 0.2), value: model.isMirrored(at: playback.currentTime))
            .animation(.easeOut(duration: 0.25), value: degrees)
    }

    /// Auto Zoom AI in the preview: the same push-in as the export at this moment.
    private var zoomNow: CGFloat { CGFloat(shownVideoTransform.scale) }

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
            // Higher layer on top (the export draws them in the same order).
            ForEach(model.overlaysOnScreen.sorted { $0.style.layer < $1.style.layer }) { overlay in
                MediaOverlayOnVideo(
                    overlay: overlay, frame: frame, time: playback.currentTime,
                    isSelected: selection == .overlay(overlay.id, isText: false),
                    keyTransform: shownOverlayTransform(overlay.id),
                    keyMode: !model.overlayKeyframes(overlayID: overlay.id).isEmpty,
                    onKeyChange: { value in keyFramingEnded(.overlay(overlay.id), value.clampedForOverlay) },
                    onRotate: { degrees in model.setOverlayStyle(id: overlay.id) { $0.rotation = degrees } },
                    onSelect: { selection = .overlay(overlay.id, isText: false) },
                    onChange: { x, y, width in
                        if overlay.uploading != nil || overlay.failed {
                            model.updatePendingOverlay(id: overlay.id, x: x, y: y, width: width)
                        } else {
                            model.setOverlayFrame(id: overlay.id, x: x, y: y, width: width)
                        }
                    },
                    onRetry: { model.retryPendingOverlay(id: overlay.id) }
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
                keyframeControl
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

    /// ◇: outline = add a keyframe here; filled = the cursor is on one (tap deletes it).
    private var keyframeControl: some View {
        let onKey = keyframeUnderCursor != nil
        return Button { addKeyframe() } label: {
            Group {
                if onKey {
                    ZoomDiamond(isSelected: true).scaleEffect(1.5)
                } else {
                    Image("Ed4CtlKeyframe").resizable().frame(width: 24, height: 24)
                }
            }
            .frame(width: 36, height: 36)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .frame(width: 36, height: 24)
        .accessibilityLabel(onKey ? "Delete keyframe" : "Add keyframe")
        .accessibilityValue(onKey ? "filled" : "empty")
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

    /// The video row's speaker: silences all of the video's own sound, and brings all of it back
    /// (the same setting as Volume → Whole video; sounds and voices keep playing).
    private func toggleVideoSound() {
        let muted = model.toggleVideoSound()
        show(.message(muted ? "Video sound off · tap ↶ to undo" : "Video sound on", undo: muted))
    }

    // MARK: Keyframes (◇): framing with the fingers, kept at the cursor

    /// What ◇ works on: the selected overlay (Float) or video clip, the clip of a selected ◇, or
    /// the clip under the cursor.
    private var keyTarget: KeyframeTarget? {
        switch selection {
        case .overlay(let id, isText: false):
            return model.overlayStyle(id).layout == .float ? .overlay(id) : nil
        case .video(let id):
            // The cursor left the selected clip: ◇ and the fingers work on the clip under the cursor.
            if let under = model.clipWindow(at: playback.currentTime)?.clipID, under != id { return .video(under) }
            return .video(id)
        case .zoom(let id):
            if let mark = model.videoKeyframeMarks.first(where: { $0.id == id }) { return mark.target }
            for overlay in model.overlaysOnScreen where model.overlayKeyframeMarks(overlayID: overlay.id).contains(where: { $0.id == id }) {
                return .overlay(overlay.id)
            }
            return nil
        default:
            return model.clipWindow(at: playback.currentTime).map { .video($0.clipID) }
        }
    }

    private func clearKeyPending() {
        keyPending = nil
        keyPendingTarget = nil
        keyGestureBase = nil
    }

    /// The video's framing shown now (a framing being made with the fingers, else the keyframes').
    private var shownVideoTransform: TransformValue {
        if let keyPending, case .video = keyPendingTarget { return keyPending.clampedForVideo }
        return model.videoTransform(at: playback.currentTime)
    }

    private func shownOverlayTransform(_ id: String) -> TransformValue {
        if let keyPending, keyPendingTarget == .overlay(id) { return keyPending.clampedForOverlay }
        return model.overlayTransform(id: id, at: playback.currentTime)
    }

    /// The cursor is on a ◇ of the target (the button is filled; tapping it deletes that ◇).
    private var keyframeUnderCursor: KeyframeMark? {
        guard let target = keyTarget else { return nil }
        return model.keyframe(at: playback.currentTime, target: target)
    }

    /// A framing finished with the fingers: on a ◇ it changes that keyframe at once; elsewhere it
    /// waits for ◇ (moving the cursor drops it).
    private func keyFramingEnded(_ target: KeyframeTarget, _ value: TransformValue) {
        let at = playback.currentTime
        #if DEBUG
        NSLog("CUTSELLB17 framing_end target=%@ at=%.3f scale=%.3f rot=%.1f marks=%@ windows=%@", "\(target)", at, value.scale,
              value.rotation, model.videoKeyframeMarks.map { "\($0.target):\(String(format: "%.3f", $0.time))" }.joined(separator: ","),
              EditorTracks.clipWindows(model).map { "\($0.clipID):\(String(format: "%.2f-%.2f", $0.start, $0.end))" }.joined(separator: ","))
        #endif
        if let mark = model.keyframe(at: at, target: target) {
            model.updateKeyframe(id: mark.id, value: value)
            clearKeyPending()
            return
        }
        keyPending = value
        keyPendingTarget = target
        keyPendingTime = at
        show(.message("Tap ◇ to keep this framing"))
    }

    private func addKeyframe() {
        playback.pause()
        let at = playback.currentTime
        guard let target = keyTarget else {
            show(.message("Select a clip or an overlay, then tap ◇."))
            return
        }
        #if DEBUG
        NSLog("CUTSELLB17 diamond_tap target=%@ at=%.3f selection=%@ pending=%@", "\(target)", at, "\(String(describing: selection))",
              keyPending.map { String(format: "%.3f", $0.scale) } ?? "none")
        #endif
        if let mark = model.keyframe(at: at, target: target) {
            model.deleteKeyframe(id: mark.id)
            if selection == .zoom(mark.id) { selection = nil }
            clearKeyPending()
            show(.message("Keyframe deleted · tap ↶ to undo", undo: true))
            return
        }
        let value = (keyPendingTarget == target ? keyPending : nil) ?? model.transform(of: target, at: at)
        if model.addKeyframe(target: target, at: at, value: value) != nil {
            clearKeyPending()
            show(.message("Keyframe added at \(EditorTime.short(at)) · tap ↶ to undo", undo: true))
        } else {
            show(.message("Move the cursor onto the clip to add a keyframe."))
        }
    }

    /// The UIKit pinch / twist / drag layer on the video.
    private func framingLayer(size: CGSize) -> some View {
        FramingGestureLayer(
            onBegin: { framingBegan() },
            onChange: { framingChanged($0, size: size) },
            onEnd: { framingEnded($0) },
            onTap: { videoTapped() }
        )
        .frame(width: size.width, height: size.height)
        .accessibilityElement()
        .accessibilityIdentifier("videoFramingLayer")
        .accessibilityValue(keyPending.map { String(format: "scale=%.3f rot=%.1f x=%.3f y=%.3f", $0.scale, $0.rotation, $0.x, $0.y) } ?? "none")
    }

    private func framingBegan() {
        guard let target = keyTarget else { return }
        playback.pause()
        let shown: TransformValue
        switch target {
        case .video: shown = shownVideoTransform
        case .overlay(let id): shown = shownOverlayTransform(id)
        }
        lastFraming = Date()
        keyGestureBase = keyPendingTarget == target ? (keyPending ?? shown) : shown
        keyPendingTarget = target
        keyPendingTime = playback.currentTime
        CutSellDiagnostics.log("framing_began", ["target": target.logName, "at": String(format: "%.2f", playback.currentTime)])
    }

    private func framingChanged(_ change: FramingGestureLayer.Change, size: CGSize) {
        guard let base = keyGestureBase, let target = keyPendingTarget else { return }
        lastFraming = Date()
        let next = FramingMath.apply(change, to: base, frame: size)
        if case .video = target { keyPending = next.clampedForVideo } else { keyPending = next.clampedForOverlay }
    }

    private func framingEnded(_ change: FramingGestureLayer.Change) {
        lastFraming = Date()
        guard keyGestureBase != nil, let target = keyPendingTarget, let value = keyPending else {
            keyGestureBase = nil
            return
        }
        keyGestureBase = nil
        CutSellDiagnostics.log("framing_ended", ["target": target.logName, "touches": String(change.touches),
                                                 "scale": String(format: "%.3f", value.scale),
                                                 "rotation": String(format: "%.1f", value.rotation)])
        if case .overlay(let id) = target, model.overlayKeyframes(overlayID: id).isEmpty,
           let overlay = model.overlaysOnScreen.first(where: { $0.id == id }) {
            // An overlay without keyframes: the fingers change its own place, size and turn.
            let current = model.overlayTransform(id: id, at: playback.currentTime)
            let x = max(0, min(1, overlay.x + (value.x - current.x)))
            let y = max(0, min(1, overlay.y + (value.y - current.y)))
            let width = max(0.1, min(1, overlay.width * value.scale / max(0.01, current.scale)))
            let turn = value.rotation - current.rotation
            if overlay.uploading != nil || overlay.failed {
                model.updatePendingOverlay(id: id, x: x, y: y, width: width)
            } else {
                model.setOverlayFrame(id: id, x: x, y: y, width: width)
            }
            if abs(turn) > 0.01 {
                let rotation = (overlay.style.rotation + turn).truncatingRemainder(dividingBy: 360)
                model.setOverlayStyle(id: id) { $0.rotation = rotation }
            }
            clearKeyPending()
            return
        }
        keyFramingEnded(target, value)
    }

    /// The fingers frame the target at the cursor: a video clip (selected, under the cursor, or
    /// the clip of a selected ◇) or a selected Float overlay.
    private var framingEnabled: Bool {
        guard keyTarget != nil, panel == .none, editingCaptionClip == nil, countdown == nil else { return false }
        switch selection {
        case nil, .video, .zoom, .overlay(_, isText: false): return true
        default: return false
        }
    }

    private var framingOverlay: Bool {
        if case .overlay = keyTarget { return true }
        return false
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
                        onAddVideo: { playback.pause(); sheet = .brollGallery },
                        onAddVoice: { openVoiceOver() },
                        onAddOverlay: { openOverlayPanel() },
                        onToggleVideoSound: { toggleVideoSound() },
                        compact: card != nil
                    )
                    .padding(.top, 1)
                    // Under the timeline, above the toolbar (never over the rows).
                    if let card {
                        cardView(card)
                            .padding(.vertical, 8)
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                }
                if case .style = panel {} else { Spacer(minLength: 0) }
                ToolbarV4(active: activeTool) { tool in toolTapped(tool) }
            }
        }
    }

    // MARK: Editor v5 panels

    @ViewBuilder
    private func cardView(_ kind: EditorPanelKind) -> some View {
        switch kind {
        case .overlayAdd:
            OverlayAddPanelV5(onPhoto: { openGallery(photos: true) }, onVideo: { openGallery(photos: false) },
                              onClose: { closeCard() })
        case .overlay(let id):
            OverlayPanelV5(model: model, overlayID: id, onClose: { closeCard() })
        case .volume:
            VolumePanelV5(model: model, target: model.volumeTarget(for: selection), onClose: { closeCard() })
        case .speed:
            SpeedPanelV5(model: model, clipID: selectedVideoClip, onClose: { closeCard() })
        case .voice(let id):
            VoiceClipPanelV5(model: model, clipID: id, onRecordAgain: {
                recordAgainID = id
                closeCard(keepSelection: true)
                openVoiceOver()
            }, onDelete: {
                closeCard()
                model.removeAudioClip(id: id)
                show(.message("Voice-over deleted", undo: true))
            }, onClose: { closeCard() })
        case .sound(let id):
            SoundClipPanelV5(model: model, clipID: id, onDelete: {
                closeCard()
                model.removeAudioClip(id: id)
                show(.message("Sound deleted", undo: true))
            }, onClose: { closeCard() })
        case .audio:
            AudioPanelV5(model: model, playhead: { playback.currentTime }, onAdded: { name, clipID in
                // The new sound comes selected (handles to trim, hold to move, Split, Delete).
                selection = .sound(clipID)
                show(.message("\(name) added · tap ↶ to undo", undo: true))
            }, onMusic: {
                playback.pause()
                if musicNoticeShown { musicPicker = true } else { musicNotice = true }
            }, onClose: { closeCard() })
        }
    }

    private var selectedVideoClip: String? {
        if case .video(let id) = selection { return id }
        return nil
    }

    private func openCard(_ kind: EditorPanelKind) {
        playback.pause()
        if panel == .captions(nil) || panel == .style { panel = .none }
        withAnimation(.easeOut(duration: 0.2)) { card = card == kind ? nil : kind }
    }

    private func closeCard(keepSelection: Bool = false) {
        let closing = card
        withAnimation(.easeOut(duration: 0.2)) { card = nil }
        guard !keepSelection else { return }
        switch closing {
        case .overlay, .voice, .sound: selection = nil
        default: break
        }
    }

    /// Selecting an overlay, a voice-over or a sound opens its panel; selecting something else closes it.
    /// With Volume open, the panel stays and follows the selection ("Selected clip" / "Whole video").
    private func selectionChanged(_ newValue: EditorSelection?) {
        if card == .volume { return }
        withAnimation(.easeOut(duration: 0.2)) {
            switch newValue {
            case .overlay(let id, isText: false): card = .overlay(id)
            case .voiceOver(let id): card = .voice(id)
            case .sound(let id): card = .sound(id)
            default:
                switch card {
                case .overlay, .voice, .sound: card = nil
                default: break
                }
            }
        }
    }

    private var activeTool: EditorTool? {
        switch panel {
        case .captions: return .captions
        case .style: return .style
        case .voiceRecord, .voiceReview, .micPermission: return .voiceOver
        case .none:
            switch card {
            case .overlayAdd, .overlay: return .overlay
            case .voice: return .voiceOver
            case .audio, .sound: return .audio
            case .volume: return .volume
            case .speed: return .speed
            case nil: return nil
            }
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

    /// Design screenshots show sample tiles; the build 18 probe opens the real system picker.
    private var usesSampleGallery: Bool { previewState != nil && previewState?.probe?.hasPrefix("b18") != true }

    @ViewBuilder
    private func sheetView(_ which: EditorSheet) -> some View {
        switch which {
        case .overlay, .overlayPhoto:
            let photos = which == .overlayPhoto
            OverlayGallerySheet(onPick: { url in
                closeCard()
                model.addOverlay(fileURL: url, at: playback.currentTime)
            }, sampleTiles: usesSampleGallery, videosOnly: !photos, photosOnly: photos,
               title: photos ? "Photos" : "Videos")
        case .broll, .brollGallery:
            // Full-screen B-roll: a video from the phone, as a clip of the main track at the cursor.
            OverlayGallerySheet(onPick: { url in
                model.addBrollClip(fileURL: url, at: playback.currentTime)
            }, sampleTiles: usesSampleGallery, videosOnly: true, title: "Add a video")
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
            // The selected clip turns on its own; with nothing selected, the whole video.
            if let clipID = selectedVideoClip {
                let next = (model.clipRotation(clipID) + 90) % 360
                model.rotateClip(clipID: clipID)
                show(.message("Clip rotated \(next)°", undo: true))
            } else {
                let next = (model.rotationDegrees + 90) % 360
                Task { await model.setDraftValue("rotation", .number(Double(next))) }
                show(.message("Rotated \(next)°", undo: true))
            }
        case .overlay: openOverlayPanel()
        case .voiceOver: openVoiceOver()
        case .audio: openCard(.audio)
        case .volume: openCard(.volume)
        case .speed:
            // The selected clip, or the one under the cursor (selected so its handles show).
            if selectedVideoClip == nil, let clip = playback.clipID(at: playback.currentTime) ?? EditorTracks.clipWindows(model).first(where: { playback.currentTime >= $0.start && playback.currentTime < $0.end })?.clipID {
                selection = .video(clip)
            }
            openCard(.speed)
        case .captions:
            playback.pause()
            card = nil
            panel = panel == .captions(nil) ? .none : .captions(nil)
        case .style:
            card = nil
            panel = panel == .style ? .none : .style
        }
    }

    /// Overlay: the "Add overlay · From your Photos" panel (Figma 1119:670) with Photo and Video.
    private func openOverlayPanel() {
        playback.pause()
        openCard(.overlayAdd)
    }

    private func openGallery(photos: Bool) {
        sheet = photos ? .overlayPhoto : .overlay
    }

    /// Closing the voice-over sheet: before recording it just closes (nothing changes); during the countdown it is
    /// cancelled; with a take recording or not saved yet it asks "Discard recording?" -- it never saves by itself.
    private func requestCloseVoiceSheet() {
        if countdown != nil {
            cancelVoiceSheet()
        } else if recorder.state == .recording || recorder.recordedURL != nil {
            confirmDiscardVoice = true
        } else {
            cancelVoiceSheet()
        }
    }

    private func cancelVoiceSheet() {
        countdown = nil
        if recorder.state == .recording { recorder.stop() }
        recorder.discardTake()
        recordAgainID = nil
        panel = .none
    }

    private func openVoiceOver() {
        playback.pause()
        withAnimation(.easeOut(duration: 0.2)) { card = nil }
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
        case .voiceOver(let id), .sound(let id):
            if model.splitAudioClip(id: id, at: time) {
                show(.message("Split at \(EditorTime.short(time)) · tap ↶ to undo", undo: true))
            } else {
                show(.message("Move the cursor inside the clip to split it."))
            }
        case .overlay(let id, isText: false):
            if model.pendingOverlays.contains(where: { $0.id == id }) {
                show(.message("Wait until it finishes uploading, then split it."))
            } else if model.splitOverlay(id: id, at: time) {
                show(.message("Overlay split at \(EditorTime.short(time)) · tap ↶ to undo", undo: true))
            } else {
                show(.message("Move the cursor inside the overlay to split it."))
            }
        case .broll(let id):
            await model.splitBrollPlacement(id: id, at: time)
            show(.message("B-roll split at \(EditorTime.short(time))"))
        default:
            guard let window = EditorTracks.clipWindows(model).first(where: { time > $0.start + 0.05 && time < $0.end - 0.05 }) else {
                show(.message("Move the cursor inside a clip to split it."))
                return
            }
            let wanted = window.source(at: time)
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
            show(.message("Voice-over deleted", undo: true))
            await model.removeAudioClip(id: id).value
        case .sound(let id):
            show(.message("Sound deleted", undo: true))
            await model.removeAudioClip(id: id).value
        case .broll(let id):
            show(.message("B-roll deleted"))
            await model.removeBrollPlacement(id: id)
        case .overlay(let id, let isText):
            show(.message("Overlay deleted", undo: true))
            if isText {
                await model.removeTextOverlay(id: id)
            } else if model.pendingOverlays.contains(where: { $0.id == id }) {
                model.discardPendingOverlay(id: id)
            } else {
                await model.removeMediaOverlay(id: id)
            }
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
    case split, delete, mirror, rotate, overlay, voiceOver, audio, volume, speed, captions, style

    var title: String {
        switch self {
        case .split: return "Split"
        case .delete: return "Delete"
        case .mirror: return "Mirror"
        case .rotate: return "Rotate"
        case .overlay: return "Overlay"
        case .voiceOver: return "Voice Over"
        case .audio: return "Audio"
        case .volume: return "Volume"
        case .speed: return "Speed"
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
        case .audio: return "Ed5Audio"
        case .volume: return "Ed5Volume"
        case .speed: return "Ed5Speed"
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
/// One photo / video overlay on the video (Figma 1026:3400): its real picture at its real
/// shape, moved with one finger and resized with two; white frame with corner dots when selected.
/// While it uploads it shows its %; a failed upload shows "Tap to retry".
struct OverlayOnScreen: Identifiable, Equatable {
    let id: String
    let kind: EditorMediaKind
    let url: URL?
    var x: Double
    var y: Double
    var width: Double
    let start: Double
    let end: Double
    let sourceStart: Double
    var style = OverlayStyle()
    var uploading: Double?
    var failed = false
}

private struct MediaOverlayOnVideo: View {
    let overlay: OverlayOnScreen
    let frame: CGSize
    let time: Double
    let isSelected: Bool
    /// Build 17 keyframes: the overlay's framing now (on top of its own place, size and turn).
    let keyTransform: TransformValue
    /// The overlay has keyframes: the fingers change the framing at the cursor, not its own place.
    let keyMode: Bool
    let onKeyChange: (TransformValue) -> Void
    let onRotate: (Double) -> Void
    let onSelect: () -> Void
    let onChange: (Double, Double, Double) -> Void
    let onRetry: () -> Void

    @State private var drag: CGSize = .zero
    @State private var pinch: CGFloat = 1
    @State private var turning: Double?
    @State private var aspect: Double?

    private var visible: Bool { isSelected || overlay.uploading != nil || (time >= overlay.start && time < overlay.end) }
    private var style: OverlayStyle { overlay.style }
    /// Feather in points of this preview (the export's 12 px / 28 px of a 1080-wide frame).
    private var feather: CGFloat { style.edge.featherOf1080 * frame.width / 1080 }

    var body: some View {
        if visible {
            if let region = style.layout.region {
                regionBody(region)
            } else {
                floatBody
            }
        }
    }

    private var picture: some View {
        OverlayMediaPicture(url: overlay.url, kind: overlay.kind, seconds: overlay.sourceStart + max(0, time - overlay.start))
    }

    /// Soft edges: the picture fades out over `feather` points at its sides.
    private func feathered<V: View>(_ view: V, size: CGSize) -> some View {
        view.mask {
            if feather > 0.5 {
                RoundedRectangle(cornerRadius: feather)
                    .padding(feather / 2)
                    .blur(radius: feather / 2)
            } else {
                Rectangle()
            }
        }
    }

    // Float: its own place and size; one finger moves, two fingers resize, the ↻ button turns it.
    private var floatBody: some View {
        let key = keyTransform
        let w = frame.width * CGFloat(overlay.width) * pinch * CGFloat(key.scale)
        let h = w / CGFloat(max(0.1, aspect ?? 1))
        let angle = (turning ?? style.rotation) + key.rotation
        return feathered(picture.frame(width: w, height: h).clipShape(RoundedRectangle(cornerRadius: style.edge == .sharp ? 4 : 0)),
                         size: CGSize(width: w, height: h))
            .overlay { if isSelected { TornasolSelection(onTurn: turnGesture(center: CGSize(width: w, height: h))) } }
            .overlay { uploadBadge }
            .rotationEffect(.degrees(angle))
            .position(x: frame.width * CGFloat(overlay.x + key.x) + drag.width,
                      y: frame.height * CGFloat(overlay.y + key.y) + drag.height)
            .onTapGesture { overlay.failed ? onRetry() : onSelect() }
            .gesture(moveGesture.simultaneously(with: resizeGesture))
            .task(id: overlay.url?.path ?? overlay.id) {
                if let url = overlay.url { aspect = await EditorMediaInfo.aspect(of: url, kind: overlay.kind) }
            }
            .accessibilityIdentifier("mediaOverlayOnVideo")
    }

    // Region layouts (Full, Top, Bottom, Left, Right, Stacked, Side by side): fill or fit their region.
    private func regionBody(_ region: CGRect) -> some View {
        let size = CGSize(width: frame.width * region.width, height: frame.height * region.height)
        return ZStack {
            if !style.fill && !style.layout.transparentPad { Color.black }
            if style.fill {
                picture.frame(width: size.width, height: size.height).clipped()
            } else {
                picture.aspectRatio(CGFloat(aspect ?? 1), contentMode: .fit).frame(width: size.width, height: size.height)
            }
        }
        .frame(width: size.width, height: size.height)
        .modifier(FeatherModifier(feather: feather))
        .overlay { if isSelected { Rectangle().strokeBorder(Tornasol.angular, lineWidth: 2) } }
        .overlay { uploadBadge }
        .position(x: frame.width * (region.midX), y: frame.height * (region.midY))
        .onTapGesture { overlay.failed ? onRetry() : onSelect() }
        .task(id: overlay.url?.path ?? overlay.id) {
            if let url = overlay.url { aspect = await EditorMediaInfo.aspect(of: url, kind: overlay.kind) }
        }
        .accessibilityIdentifier("mediaOverlayOnVideo")
    }

    @ViewBuilder
    private var uploadBadge: some View {
        if overlay.failed {
            Label("Tap to retry", systemImage: "arrow.clockwise")
                .font(CS.inter(11, .semibold)).foregroundStyle(.white)
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(CS.error.opacity(0.85), in: Capsule())
        } else if let progress = overlay.uploading {
            Text("\(Int((progress * 100).rounded()))%")
                .font(CS.inter(12, .bold)).monospacedDigit().foregroundStyle(.white)
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(Color.black.opacity(0.6), in: Capsule())
                .accessibilityIdentifier("overlayUploadPercent")
        }
    }

    private var moveGesture: some Gesture {
        DragGesture(minimumDistance: 2)
            .onChanged { value in
                guard turning == nil else { return }
                if !isSelected { onSelect() }
                drag = value.translation
            }
            .onEnded { value in
                guard turning == nil else { return }
                if keyMode {
                    var next = keyTransform
                    next.x += Double(value.translation.width / frame.width)
                    next.y += Double(value.translation.height / frame.height)
                    drag = .zero
                    onKeyChange(next)
                    return
                }
                let nx = max(0, min(1, overlay.x + Double(value.translation.width / frame.width)))
                let ny = max(0, min(1, overlay.y + Double(value.translation.height / frame.height)))
                drag = .zero
                onChange(nx, ny, overlay.width)
            }
    }

    private var resizeGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                if !isSelected { onSelect() }
                pinch = value
            }
            .onEnded { value in
                if keyMode {
                    var next = keyTransform
                    next.scale *= Double(value)
                    pinch = 1
                    onKeyChange(next)
                    return
                }
                let nw = max(0.1, min(1, overlay.width * Double(value)))
                pinch = 1
                onChange(overlay.x, overlay.y, nw)
            }
    }

    /// Dragging the ↻ button turns the overlay around its centre (snaps to 0 / 90 / 180 / 270).
    private func turnGesture(center: CGSize) -> some Gesture {
        DragGesture(minimumDistance: 0, coordinateSpace: .named("overlayBox"))
            .onChanged { value in
                let dx = value.location.x - center.width / 2
                let dy = value.location.y - center.height / 2
                // The button sits at the top-right corner: that direction is "no turn".
                let turned = atan2(Double(dy), Double(dx)) - atan2(Double(-center.height / 2 - 18), Double(center.width / 2 + 6))
                var degrees = turned * 180 / Double.pi + style.rotation
                for snap in [0.0, 90, 180, 270, 360, -90, -180, -270] where abs(degrees - snap) < 4 { degrees = snap }
                turning = degrees.truncatingRemainder(dividingBy: 360)
            }
            .onEnded { _ in
                if let turning {
                    if keyMode {
                        var next = keyTransform
                        next.rotation = keyTransform.rotation + (turning - style.rotation)
                        onKeyChange(next)
                    } else {
                        onRotate(turning)
                    }
                }
                turning = nil
            }
    }
}

/// Soft edges for a region layout.
private struct FeatherModifier: ViewModifier {
    let feather: CGFloat
    func body(content: Content) -> some View {
        if feather > 0.5 {
            content.mask { Rectangle().padding(feather / 2).blur(radius: feather / 2) }
        } else {
            content
        }
    }
}

/// A selected overlay on the video (Figma 1124:1191): tornasol border, white corner squares and the
/// ↻ button at the top-right corner.
private struct TornasolSelection<Turn: Gesture>: View {
    let onTurn: Turn

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: 3).strokeBorder(Tornasol.angular, lineWidth: 2)
                ForEach(0..<4, id: \.self) { index in
                    RoundedRectangle(cornerRadius: 2).fill(.white).frame(width: 10, height: 10)
                        .position(x: index % 2 == 0 ? 0 : proxy.size.width, y: index < 2 ? 0 : proxy.size.height)
                }
                Image(systemName: "arrow.clockwise").font(.system(size: 11, weight: .bold)).foregroundStyle(.white)
                    .frame(width: 24, height: 24)
                    .background(Tornasol.angular, in: Circle())
                    .contentShape(Circle().inset(by: -10))
                    .position(x: proxy.size.width - 2, y: -18)
                    .gesture(onTurn)
                    .accessibilityLabel("Turn")
                    .accessibilityIdentifier("overlayRotate")
            }
            .coordinateSpace(name: "overlayBox")
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
