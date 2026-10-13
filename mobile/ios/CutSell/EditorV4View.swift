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

    private var duration: Double { max(0.1, playback.duration > 0 ? playback.duration : model.timelineDuration) }

    /// Rebuild the preview when what it plays changes (the predicted draft included, so it follows at once).
    private var previewKey: String {
        let placements = (model.timelineComposition?.voiceOverPlacements ?? [])
            .map { "\($0.placementID)@\($0.timelineStartSec)" }.joined(separator: ",")
        return "\(model.displayDraft?.hashValue ?? 0)|\(placements)|\(model.voiceOverVolume)"
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
        case .captions(let clipID): return (clipID == nil ? controls : 0) + 236 + 4 + toolbar
        // Style (Figma 1054:2497): no controls; the panel sits where the controls and timeline were.
        case .style: return controls + 1 + TimelineV4.height + toolbar
        case .none, .micPermission, .voiceRecord, .voiceReview: return controls + 1 + TimelineV4.height + toolbar
        }
    }

    private var showsControls: Bool {
        switch panel {
        case .style: return false
        case .captions(let clipID): return clipID == nil
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
        if let probe = previewState.probe {
            switch probe {
            case "cap_highlight": model.applyCaptionSettings(enabled: true, preset: "highlight_green", font: "montserrat")
            case "cap_box": model.applyCaptionSettings(enabled: true, preset: "box", font: "montserrat")
            case "cap_fonts": model.applyCaptionSettings(enabled: true, preset: "classic", font: "bebas_neue")
            case "caption_edit": model.applyCaptionSettings(enabled: true, preset: "highlight_green")
            default: break
            }
            // Move the cursor into the cut, like the Figma screens.
            DispatchQueue.main.asyncAfter(deadline: .now() + 4) { playback.seek(to: min(3, playback.duration / 3)) }
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
                .scaleEffect(x: model.isMirrored ? -1 : 1, y: 1)
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
            videoLayer
            overlaysOnVideo(frame: size)
            if model.captionsEnabled {
                captionLayer(frame: size)
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
            .scaleEffect(x: model.isMirrored ? -1 : 1, y: 1)
            .rotationEffect(.degrees(Double(model.rotationDegrees)))
            .scaleEffect(quarter ? 266.0 / 472.0 : 1)
            .animation(.easeOut(duration: 0.25), value: model.isMirrored)
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
                    .overlay(alignment: .bottom) { volumeStrip }
                }
                if case .style = panel {} else { Spacer(minLength: 0) }
                ToolbarV4(active: activeTool) { tool in toolTapped(tool) }
            }
        }
    }

    /// Volume of the selected track: the video's own sound, or the voice-over.
    @ViewBuilder
    private var volumeStrip: some View {
        switch selection {
        case .video:
            VolumeStrip(title: "Video sound", icon: "speaker.wave.2.fill", value: $videoVolume, range: 0...2) {
                Task { await model.setVideoTrackVolume(videoVolume) }
            }
        case .voiceOver:
            VolumeStrip(title: "Voice-over", icon: "mic.fill", value: $model.voiceOverVolume, range: 0...1) {}
        default:
            EmptyView()
        }
    }

    /// Speaker on the video row: mute the video's own sound, or bring it back.
    private func toggleVideoSound() {
        let muted = model.videoTrackVolume <= 0.001
        videoVolume = muted ? 1 : 0
        Task { await model.setVideoTrackVolume(videoVolume) }
        show(.message(muted ? "Video sound on" : "Video sound muted", undo: !muted))
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
            let mirrored = !model.isMirrored
            Task { await model.setDraftValue("mirror", .bool(mirrored)) }
            show(.message(mirrored ? "Video mirrored" : "Mirror off"))
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

private struct VolumeStrip: View {
    let title: String
    let icon: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let onCommit: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Button {
                value = value > 0.001 ? 0 : 1
                onCommit()
            } label: {
                Image(systemName: value > 0.001 ? icon : "speaker.slash.fill")
                    .font(.system(size: 13, weight: .semibold)).foregroundStyle(CS.cyan).frame(width: 26)
            }
            .buttonStyle(.plain)
            Text(title).font(CS.inter(11, .semibold)).foregroundStyle(.white).fixedSize()
            Slider(value: $value, in: range) { editing in if !editing { onCommit() } }
                .tint(CS.cyan)
            Text("\(Int((value * 100).rounded()))%")
                .font(CS.sf(11, .semibold)).monospacedDigit().foregroundStyle(CS.textSecondaryF)
                .frame(width: 40, alignment: .trailing)
        }
        .padding(.horizontal, 12)
        .frame(height: 34)
        .background(Color(hex: 0x0A1326, opacity: 0.97), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color(hex: 0x14336B), lineWidth: 1))
        .padding(.horizontal, 16)
        .accessibilityIdentifier("volumeStrip")
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
