import AVFoundation
import AVKit
import SwiftUI

/// Editor v2 (Figma 04_Editor: "Editor/BaseCanon" 242:1532 and "Prototype/Canonical/Editor" 950:1165).
/// Top bar (back · logo · Export) · video with the brand border · Expand / Play / Keyframe / Undo / Redo ·
/// time + ruler · tracks Video (Mute clip) / Voiceover / Overlay (+ Captions when on), each with "+" ·
/// toolbar Split · Delete · Overlay · Voice Over · Captions.
/// Every action uses the existing editor model (the same server routes as before).
struct EditorV2View: View {
    @ObservedObject var model: DraftEditorViewModel
    let project: Project

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var appState: AppState
    @StateObject private var playback = DraftPlaybackController()
    @State private var selection: EditorSelection?
    @State private var toast: EditorToast?
    @State private var sheet: EditorSheet?
    @State private var fullScreen = false
    @State private var renaming = false
    @State private var newTitle = ""
    @State private var displayTitle: String?
    /// Design screenshots only (Debug): open in a given state.
    var previewState: EditorPreviewState? = nil

    private var duration: Double { max(0.1, playback.duration > 0 ? playback.duration : model.timelineDuration) }

    var body: some View {
        VStack(spacing: 0) {
            topBar
            titleRow
            preview
                .padding(.top, 6)
            controlsRow
                .padding(.top, 8)
            Rectangle()
                .fill(LinearGradient(colors: [Color(hex: 0x1769FF, opacity: 0.28), Color(hex: 0x7C3AED, opacity: 0.34), Color(hex: 0x19E6FF, opacity: 0.28)],
                                     startPoint: .leading, endPoint: .trailing))
                .frame(height: 1)
                .padding(.top, 8)
            EditorTracks(
                model: model, playback: playback, duration: duration, selection: $selection,
                onAdd: { track in addTapped(track) },
                onToggleMute: toggleMute
            )
            .padding(.top, 4)
            Spacer(minLength: 6)
            toolbar
        }
        .padding(.horizontal, 8)
        .background(Color(hex: 0x02040A).ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
        .preferredColorScheme(.dark)
        .task(id: model.snapshot?.revision) { await playback.rebuild(from: model) }
        .onChange(of: playback.currentTime) { _, time in
            if abs(model.playheadTime - time) >= 0.1 || time == 0 { model.playheadTime = time }
        }
        .onDisappear { playback.pause() }
        .onAppear {
            guard let previewState else { return }
            selection = previewState.selection
            toast = previewState.toast
            if let opened = previewState.sheet {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { sheet = opened }
            }
        }
        .sheet(item: $sheet) { which in sheetView(which) }
        .fullScreenCover(isPresented: $fullScreen) {
            ZStack(alignment: .topLeading) {
                Color.black.ignoresSafeArea()
                VideoPlayer(player: playback.player).ignoresSafeArea()
                Button { fullScreen = false } label: {
                    Image(systemName: "xmark").font(.system(size: 17, weight: .bold)).foregroundStyle(.white)
                        .frame(width: 44, height: 44).background(Color.black.opacity(0.4), in: Circle())
                }
                .padding(16)
                .accessibilityLabel("Close full screen")
            }
        }
        .alert("Rename cut", isPresented: $renaming) {
            TextField("Name", text: $newTitle)
            Button("Save") { Task { await rename() } }
            Button("Cancel", role: .cancel) {}
        }
        .alert("CutSell", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) { Button("OK", role: .cancel) {} } message: { Text(model.errorMessage ?? "") }
    }

    // MARK: Top

    private var topBar: some View {
        ZStack {
            BrandLogo()
            HStack {
                Button { dismiss() } label: {
                    Image("ControlBack").resizable().frame(width: 36, height: 36).frame(width: 44, height: 44)
                }
                .accessibilityLabel("Back")
                Spacer()
                Button { sheet = .export } label: {
                    Text("Export")
                        .font(CS.sf(15, .semibold))
                        .foregroundStyle(Color(hex: 0xFAFCFF))
                        .frame(width: 91, height: 38)
                        .background(Color(hex: 0x040813, opacity: 0.95), in: Capsule())
                        .overlay(Capsule().stroke(LinearGradient(colors: [CS.cyan, Color(hex: 0xFF21D7)], startPoint: .leading, endPoint: .trailing), lineWidth: 1.5))
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("editorExport")
            }
        }
        .padding(.top, 2)
    }

    /// The cut's name (tap to rename): "Cut 13 Oct" instead of an internal file name.
    private var titleRow: some View {
        Button {
            newTitle = shownTitle
            renaming = true
        } label: {
            HStack(spacing: 4) {
                Text(shownTitle).lineLimit(1)
                Image(systemName: "pencil").font(.system(size: 10, weight: .semibold))
            }
            .font(CS.inter(12, .medium))
            .foregroundStyle(CS.textSecondaryF)
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("editorTitle")
    }

    private var shownTitle: String { displayTitle ?? CutNaming.readable(project: project) }

    private var preview: some View {
        ZStack(alignment: .bottom) {
            ZStack {
                Color(hex: 0x070B16)
                PlayerLayerView(player: playback.player)
                if model.captionsEnabled {
                    let cue = playback.captionCue(at: playback.currentTime)
                    CaptionOverlayView(
                        cue: cue, time: playback.currentTime, preset: model.captionPreset, fontKey: model.captionFont,
                        x: model.captionX, y: model.captionY, scale: model.captionScale, editing: nil
                    )
                }
                if !playback.isPlaying && !playback.isBuilding {
                    Image("EdPlayIcon").resizable().frame(width: 44, height: 44)
                        .background(Color(hex: 0x091226), in: RoundedRectangle(cornerRadius: 12))
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(hex: 0x1AA6FF, opacity: 0.9), lineWidth: 1))
                        .opacity(playback.currentTime < 0.05 ? 1 : 0)
                }
                if playback.isBuilding {
                    ProgressView().tint(CS.cyan)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 17, style: .continuous))
            .padding(3.5)
            .background(Color(hex: 0x050711), in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .stroke(LinearGradient(colors: [CS.cyan, Color(hex: 0x7C3AED), Color(hex: 0xFF21D7)], startPoint: .bottomLeading, endPoint: .topTrailing), lineWidth: 1.5)
            )
            .contentShape(Rectangle())
            .onTapGesture { playback.togglePlayback() }

            if let toast {
                ToastPill(toast: toast) {
                    Task { await undoFromToast() }
                }
                .padding(.bottom, 12)
                .transition(.opacity)
            } else if let selection {
                Text(selection.pillTitle)
                    .font(CS.inter(12, .medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 22)
                    .frame(height: 30)
                    .background(Color(hex: 0x040812, opacity: 0.95), in: RoundedRectangle(cornerRadius: 10))
                    .overlay(RoundedRectangle(cornerRadius: 10).stroke(CS.cyan.opacity(0.8), lineWidth: 1))
                    .padding(.bottom, 12)
            }
        }
        .aspectRatio(244.0 / 412.0, contentMode: .fit)
        .frame(maxWidth: .infinity, maxHeight: 412)
        .layoutPriority(-1)
        .animation(.easeOut(duration: 0.2), value: toast)
    }

    private var controlsRow: some View {
        HStack(spacing: 12) {
            control("EdExpand", "Full screen") { fullScreen = true }
            Spacer()
            control(playback.isPlaying ? "EdPause" : "EdPlay", playback.isPlaying ? "Pause" : "Play") { playback.togglePlayback() }
            control("EdKeyframe", "Keyframe") { show(.message("Keyframes are coming soon.")) }
            control("EdUndo", "Undo") { Task { await model.undo() } }
            control("EdRedo", "Redo") { Task { await model.redo() } }
        }
        .padding(.horizontal, 6)
    }

    @ViewBuilder
    private func control(_ image: String, _ label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            if image == "EdPause" {
                Image(systemName: "pause.fill")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(Color(hex: 0x1769FF))
                    .frame(width: 32, height: 32)
                    .background(Color(hex: 0x0A1326), in: RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(hex: 0x1769FF), lineWidth: 1))
                    .frame(width: 44, height: 44)
            } else {
                Image(image).resizable().frame(width: 44, height: 44)
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }

    // MARK: Toolbar (Editor/Toolbar/Current 428:36)

    private var toolbar: some View {
        HStack(spacing: 4) {
            tool("Split", id: "toolSplit", icon: AnyView(SplitIcon())) { Task { await split() } }
            tool("Delete", id: "toolDelete", icon: AnyView(Image("EdToolDelete").resizable().frame(width: 52, height: 52))) { Task { await delete() } }
            tool("Overlay", id: "toolOverlay", icon: AnyView(Image("EdToolOverlay").resizable().frame(width: 52, height: 52))) { sheet = .overlay }
            tool("Voice\nOver", id: "toolVoiceOver", icon: AnyView(Image("EdToolVoice").resizable().frame(width: 52, height: 52))) { sheet = .voiceOver }
            tool("Captions", id: "toolCaptions", icon: AnyView(CCIcon())) { sheet = .captions }
        }
        .padding(4)
        .frame(height: 80)
        .background(Color(hex: 0x1B0916, opacity: 0.98), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(Color(hex: 0x741A52, opacity: 0.95), lineWidth: 1))
    }

    private func tool(_ title: String, id: String, icon: AnyView, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 2) {
                icon.frame(width: 42, height: 42)
                Text(title)
                    .font(CS.inter(10, .medium))
                    .foregroundStyle(Color(hex: 0xEDF5FF))
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(height: 24, alignment: .top)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 72)
            .background(Color(hex: 0x060E1C), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(id)
    }

    // MARK: Sheets

    @ViewBuilder
    private func sheetView(_ which: EditorSheet) -> some View {
        switch which {
        case .captions:
            AddCaptionsSheet(model: model, playhead: playback.currentTime)
        case .voiceOver:
            VoiceOverRecordSheet(model: model, start: playback.currentTime) { outcome in
                switch outcome {
                case .placed: show(.message("Voice-over added"))
                case .savedNotPlaced: show(.message("Voice-over saved. It will be placed when this cut's main video is registered."))
                case .failed: break
                }
            }
        case .overlay:
            OverlayView(model: model)
        case .broll:
            BrollView(model: model, initialPlacementID: nil)
        case .export:
            ExportSheet(model: model)
        }
    }

    // MARK: Actions

    private func addTapped(_ track: EditorTrack) {
        switch track {
        case .video: sheet = .broll
        case .voiceOver: sheet = .voiceOver
        case .overlay: sheet = .overlay
        case .captions: sheet = .captions
        }
    }

    private func toggleMute() {
        let clips = model.selectedClips
        let allMuted = !clips.isEmpty && clips.allSatisfy { $0["audio_muted"]?.boolValue == true }
        Task {
            for clip in clips {
                guard let id = clip["clip_id"]?.stringValue else { continue }
                await model.setAudio(clipID: id, muted: !allMuted)
            }
        }
    }

    private func split() async {
        let time = playback.currentTime
        switch selection {
        case .voiceOver(let id):
            await model.splitVoiceOverPlacement(id: id, at: time)
            show(.message(String(format: "Voice-over split at %@", EditorTime.short(time))))
        case .broll(let id):
            await model.splitBrollPlacement(id: id, at: time)
            show(.message(String(format: "B-roll split at %@", EditorTime.short(time))))
        default:
            guard let window = EditorTracks.clipWindows(model).first(where: { time > $0.start + 0.05 && time < $0.end - 0.05 }) else {
                show(.message("Move the playhead inside a clip to split it."))
                return
            }
            let wanted = window.sourceStart + (time - window.start)
            guard let at = model.safeSplitTime(clipID: window.clipID, near: wanted) else {
                show(.message("This clip can’t be split here. Move the playhead between two words."))
                return
            }
            await model.split(clipID: window.clipID, at: at)
            selection = .video(window.clipID)
            show(.split(EditorTime.tenths(time)))
        }
    }

    private func delete() async {
        switch selection {
        case .video(let id):
            await model.remove(clipID: id)
            show(.message("Clip deleted", undo: true))
        case .voiceOver(let id):
            await model.removeVoiceOverPlacement(id: id)
            show(.message("Voice-over deleted"))
        case .broll(let id):
            await model.removeBrollPlacement(id: id)
            show(.message("B-roll deleted"))
        case .overlay(let id, let isText):
            if isText { await model.removeTextOverlay(id: id) } else { await model.removeMediaOverlay(id: id) }
            show(.message("Overlay deleted", undo: true))
        case .caption(let clipID):
            await model.editCaption(clipID: clipID, text: "")
            show(.captionDeleted)
        case nil:
            show(.message("Select a clip, voice-over, overlay or caption first."))
        }
        selection = nil
    }

    private func undoFromToast() async {
        await model.undo()
        toast = nil
    }

    private func show(_ value: EditorToast) {
        toast = value
        let shown = value
        Task {
            try? await Task.sleep(nanoseconds: 2_600_000_000)
            if toast == shown { toast = nil }
        }
    }

    private func rename() async {
        let title = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty, let session = appState.session else { return }
        struct Body: Encodable { let user_id: String; let title: String }
        do {
            let updated: Project = try await APIClient.shared.request(
                "/v1/projects/\(project.projectID)", method: "PATCH",
                body: Body(user_id: session.userID, title: String(title.prefix(120)))
            )
            displayTitle = updated.title
            try? await appState.refreshProjects()
        } catch {
            model.errorMessage = "Couldn’t rename the cut. Please try again."
        }
    }
}

// MARK: - Tracks

enum EditorTrack { case video, voiceOver, overlay, captions }

enum EditorSelection: Equatable {
    case video(String)
    case broll(String)
    case voiceOver(String)
    case overlay(String, isText: Bool)
    case caption(String)

    var pillTitle: String {
        switch self {
        case .video: return "Video clip selected"
        case .broll: return "B-roll selected"
        case .voiceOver: return "Voice-over selected"
        case .overlay: return "Overlay selected"
        case .caption: return "Caption selected"
        }
    }
}

struct EditorTracks: View {
    @ObservedObject var model: DraftEditorViewModel
    @ObservedObject var playback: DraftPlaybackController
    let duration: Double
    @Binding var selection: EditorSelection?
    let onAdd: (EditorTrack) -> Void
    let onToggleMute: () -> Void

    struct ClipWindow { let clipID: String; let start: Double; let end: Double; let sourceStart: Double; let sourceID: String? }

    /// Clips back to back on the edited timeline.
    static func clipWindows(_ model: DraftEditorViewModel) -> [ClipWindow] {
        var cursor = 0.0
        return model.selectedClips.compactMap { clip in
            let start = clip["start"]?.doubleValue ?? 0
            let end = clip["end"]?.doubleValue ?? start
            let length = max(0, end - start)
            defer { cursor += length }
            guard let id = clip["clip_id"]?.stringValue, length > 0 else { return nil }
            return ClipWindow(clipID: id, start: cursor, end: cursor + length, sourceStart: start,
                              sourceID: clip["source_asset_id"]?.stringValue)
        }
    }

    private var allMuted: Bool {
        let clips = model.selectedClips
        return !clips.isEmpty && clips.allSatisfy { $0["audio_muted"]?.boolValue == true }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Text("\(EditorTime.clock(playback.currentTime)) / \(EditorTime.clock(duration))")
                    .font(CS.sf(13, .semibold))
                    .foregroundStyle(Color(hex: 0xF2F7FF))
                    .monospacedDigit()
                    .frame(width: 92, alignment: .leading)
                Ruler(duration: duration)
                    .frame(height: 24)
                Color.clear.frame(width: 28, height: 24)
            }
            .padding(.leading, 3)

            GeometryReader { proxy in
                let laneWidth = max(10, proxy.size.width - 96 - 34)
                ZStack(alignment: .topLeading) {
                    VStack(spacing: 10) {
                        row(label: muteLabel, height: 42, lane: { videoLane(width: laneWidth) }, add: .video)
                        row(label: trackLabel(icon: "EdMic", title: "Voiceover", iconSize: 18), height: 32, lane: { voiceLane(width: laneWidth) }, add: .voiceOver)
                        row(label: trackLabel(icon: "EdLayers", title: "Overlay", iconSize: 28), height: 32, lane: { overlayLane(width: laneWidth) }, add: .overlay)
                        if model.captionsEnabled {
                            row(label: captionsLabel, height: 26, lane: { captionLane(width: laneWidth) }, add: .captions)
                        }
                    }
                    playhead(height: model.captionsEnabled ? 154 : 126)
                        .offset(x: 96 + laneWidth * CGFloat(min(1, max(0, playback.currentTime / duration))) - 5)
                }
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 6)
                        .onChanged { value in
                            let x = value.location.x - 96
                            guard x >= 0, x <= laneWidth else { return }
                            playback.seek(to: Double(x / laneWidth) * duration)
                        }
                )
            }
            .frame(height: model.captionsEnabled ? 168 : 132)
        }
    }

    private func row<Lane: View>(label: some View, height: CGFloat, @ViewBuilder lane: () -> Lane, add: EditorTrack) -> some View {
        HStack(spacing: 6) {
            label.frame(width: 90, height: height)
            lane().frame(height: height)
            Button { onAdd(add) } label: { PlusIcon() }
                .buttonStyle(.plain)
                .accessibilityLabel("Add to \(String(describing: add))")
        }
    }

    // Labels

    private var muteLabel: some View {
        Button(action: onToggleMute) {
            HStack(spacing: 4) {
                Image("EdVolume").resizable().frame(width: 24, height: 24)
                    .overlay { if allMuted { Rectangle().fill(CS.error).frame(width: 2, height: 26).rotationEffect(.degrees(45)) } }
                VStack(alignment: .leading, spacing: 1) {
                    Text("Mute clip").font(CS.sf(10, .semibold)).foregroundStyle(Color(hex: 0xEBF2FF))
                        .lineLimit(1).fixedSize()
                    Text(allMuted ? "muted" : "audio").font(CS.sf(9)).foregroundStyle(Color(hex: 0x8CA8D6))
                        .lineLimit(1).fixedSize()
                }
                Spacer(minLength: 0)
            }
            .padding(.leading, 6)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(labelBackground)
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("muteClip")
    }

    private func trackLabel(icon: String, title: String, iconSize: CGFloat) -> some View {
        HStack(spacing: 6) {
            Image(icon).resizable().frame(width: iconSize, height: iconSize)
                .frame(width: 24)
            Text(title).font(CS.sf(10, .semibold)).foregroundStyle(Color(hex: 0xEBF2FF))
                .lineLimit(1).fixedSize()
            Spacer(minLength: 0)
        }
        .padding(.leading, 6)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(labelBackground)
    }

    private var captionsLabel: some View {
        HStack(spacing: 6) {
            Image("EdCC").resizable().frame(width: 18, height: 14).frame(width: 24)
            Text("Captions").font(CS.sf(10, .semibold)).foregroundStyle(Color(hex: 0xEBF2FF))
            Spacer(minLength: 0)
        }
        .padding(.leading, 6)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(labelBackground)
    }

    private var labelBackground: some View {
        RoundedRectangle(cornerRadius: 10).fill(Color(hex: 0x040812))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color(hex: 0x14336B), lineWidth: 1))
    }

    // Lanes

    private func x(_ seconds: Double, _ width: CGFloat) -> CGFloat { width * CGFloat(max(0, min(1, seconds / duration))) }

    private func videoLane(width: CGFloat) -> some View {
        let windows = Self.clipWindows(model)
        return ZStack(alignment: .leading) {
            RoundedRectangle(cornerRadius: 8).fill(Color(hex: 0x050A14))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(hex: 0x1459A6), lineWidth: 1))
            ForEach(windows, id: \.clipID) { window in
                let isSelected = selection == .video(window.clipID)
                ClipFrameImage(url: model.playbackURL(forSource: window.sourceID), seconds: window.sourceStart + 0.2)
                    .frame(width: max(4, x(window.end - window.start, width) - 2), height: 34)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay {
                        if isSelected { SelectedClipFrame() }
                    }
                    .offset(x: x(window.start, width) + 1)
                    .onTapGesture {
                        selection = isSelected ? nil : .video(window.clipID)
                    }
            }
            // B-roll placed over the main video: a labelled block on top of the clips.
            ForEach(model.timelineComposition?.brollPlacements ?? []) { placement in
                let isSelected = selection == .broll(placement.placementID)
                Text("B-roll")
                    .font(CS.sf(9, .bold))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                    .padding(.horizontal, 4)
                    .frame(width: max(8, x(placement.timelineEndSec - placement.timelineStartSec, width) - 2), height: 34, alignment: .topLeading)
                    .padding(.top, 2)
                    .background(
                        LinearGradient(colors: [Color(hex: 0xD61FFF, opacity: 0.55), Color(hex: 0x1A6BFF, opacity: 0.45)],
                                       startPoint: .leading, endPoint: .trailing),
                        in: RoundedRectangle(cornerRadius: 6)
                    )
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(isSelected ? CS.cyan : Color(hex: 0xD61FFF), lineWidth: isSelected ? 2 : 1))
                    .offset(x: x(placement.timelineStartSec, width) + 1)
                    .onTapGesture { selection = isSelected ? nil : .broll(placement.placementID) }
                    .accessibilityIdentifier("brollBlock")
            }
        }
        .frame(width: width)
        .accessibilityIdentifier("videoTrack")
    }

    private func voiceLane(width: CGFloat) -> some View {
        ZStack(alignment: .leading) {
            RoundedRectangle(cornerRadius: 7).fill(Color(hex: 0x03060E))
            ForEach(model.timelineComposition?.voiceOverPlacements ?? []) { placement in
                let isSelected = selection == .voiceOver(placement.placementID)
                Image("EdWave").resizable()
                    .frame(width: max(6, x(placement.timelineEndSec - placement.timelineStartSec, width)), height: 30)
                    .clipShape(RoundedRectangle(cornerRadius: 7))
                    .overlay(RoundedRectangle(cornerRadius: 7).stroke(isSelected ? CS.cyan : .clear, lineWidth: 2))
                    .offset(x: x(placement.timelineStartSec, width))
                    .onTapGesture { selection = isSelected ? nil : .voiceOver(placement.placementID) }
            }
        }
        .frame(width: width)
    }

    private func overlayLane(width: CGFloat) -> some View {
        let texts = model.textOverlays.map { (item: $0, isText: true) }
        let media = model.mediaOverlays.map { (item: $0, isText: false) }
        let items = texts + media
        return ZStack(alignment: .leading) {
            RoundedRectangle(cornerRadius: 8).fill(Color(hex: 0x03060E))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(hex: 0x0F4D9E), lineWidth: 1))
            ForEach(Array(items.enumerated()), id: \.offset) { _, entry in
                let id = entry.item["overlay_id"]?.stringValue ?? ""
                let start = entry.item["start"]?.doubleValue ?? 0
                let end = entry.item["end"]?.doubleValue ?? start + 1
                let isSelected = selection == .overlay(id, isText: entry.isText)
                Text(entry.isText ? (entry.item["text"]?.stringValue ?? "Text") : (entry.item["kind"]?.stringValue ?? "Media").capitalized)
                    .font(CS.sf(11, .semibold))
                    .foregroundStyle(entry.isText ? .white : Color(hex: 0xD10F1A))
                    .lineLimit(1)
                    .padding(.horizontal, 6)
                    .frame(width: max(8, x(end - start, width) - 2), height: 30, alignment: .leading)
                    .background(
                        entry.isText
                            ? AnyShapeStyle(LinearGradient(colors: [Color(hex: 0x0A5CDB), Color(hex: 0x1A1F5C)], startPoint: .leading, endPoint: .trailing))
                            : AnyShapeStyle(Color(hex: 0xF0F2F7)),
                        in: RoundedRectangle(cornerRadius: 7)
                    )
                    .overlay(RoundedRectangle(cornerRadius: 7).stroke(isSelected ? CS.cyan : .clear, lineWidth: 2))
                    .offset(x: x(start, width) + 1)
                    .onTapGesture { selection = isSelected ? nil : .overlay(id, isText: entry.isText) }
            }
        }
        .frame(width: width)
    }

    private func captionLane(width: CGFloat) -> some View {
        ZStack(alignment: .leading) {
            RoundedRectangle(cornerRadius: 6).fill(Color(hex: 0x03060E))
            ForEach(captionSegments, id: \.id) { cue in
                let isSelected = selection == .caption(cue.clipID)
                Text(cue.text)
                    .font(CS.sf(9, .medium))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                    .padding(.horizontal, 4)
                    .frame(width: max(6, x(cue.end - cue.start, width) - 2), height: 24)
                    .background(Color(hex: 0x0A3B7A), in: RoundedRectangle(cornerRadius: 5))
                    .overlay(RoundedRectangle(cornerRadius: 5).stroke(isSelected ? CS.cyan : Color(hex: 0x1A6BFF, opacity: 0.6), lineWidth: isSelected ? 2 : 1))
                    .offset(x: x(cue.start, width) + 1)
                    .onTapGesture { selection = isSelected ? nil : .caption(cue.clipID) }
            }
        }
        .frame(width: width)
    }

    private struct CaptionSegment { let id: String; let clipID: String; let start: Double; let end: Double; let text: String }

    /// The preview's caption cues, or (before the preview is built) one caption per clip.
    private var captionSegments: [CaptionSegment] {
        if !playback.captionCues.isEmpty {
            return playback.captionCues.map { CaptionSegment(id: $0.id, clipID: $0.clipID, start: $0.start, end: $0.end, text: $0.text) }
        }
        let clips = model.selectedClips
        return Self.clipWindows(model).compactMap { window in
            let clip = clips.first { $0["clip_id"]?.stringValue == window.clipID }
            guard let text = clip?["caption_text"]?.stringValue ?? clip?["text"]?.stringValue, !text.isEmpty else { return nil }
            return CaptionSegment(id: window.clipID, clipID: window.clipID, start: window.start, end: window.end, text: text)
        }
    }

    private func playhead(height: CGFloat) -> some View {
        VStack(spacing: 0) {
            Circle().fill(.white).frame(width: 10, height: 10)
            Rectangle().fill(.white).frame(width: 2, height: height)
        }
        .padding(.top, 4)
        .allowsHitTesting(false)
    }
}

/// Selected clip: cyan outline with the grey trim handles (Figma Editor/TrimHandle).
private struct SelectedClipFrame: View {
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 6).stroke(CS.cyan, lineWidth: 2)
            HStack {
                handle
                Spacer()
                handle
            }
        }
    }

    private var handle: some View {
        RoundedRectangle(cornerRadius: 5).fill(Color(hex: 0xC7CFDB))
            .frame(width: 12)
            .overlay(Image("EdTrimGrip").resizable().frame(width: 6, height: 18))
    }
}

/// Time ruler (Editor/Ruler).
struct Ruler: View {
    let duration: Double

    var body: some View {
        GeometryReader { proxy in
            let width = proxy.size.width
            let step = Self.labelStep(duration)
            ZStack(alignment: .topLeading) {
                Color(hex: 0x03060E)
                ForEach(0...Int(duration / (step / 2)), id: \.self) { index in
                    let seconds = Double(index) * step / 2
                    let major = index % 2 == 0
                    let x = width * CGFloat(seconds / max(duration, 0.1))
                    Rectangle().fill(Color(hex: 0x5C7AAD))
                        .frame(width: 1, height: major ? 9 : 5)
                        .offset(x: x, y: major ? 9 : 12)
                    if major {
                        Text(EditorTime.clock(seconds))
                            .font(CS.sf(8, .semibold))
                            .foregroundStyle(Color(hex: 0x758FB8))
                            .offset(x: x - (index == 0 ? 0 : 10))
                    }
                }
            }
            .clipped()
        }
    }

    static func labelStep(_ duration: Double) -> Double {
        for step in [2.0, 4, 8, 15, 30, 60, 120] where duration / step <= 4.5 { return step }
        return 300
    }
}

private struct PlusIcon: View {
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 6).fill(Color(hex: 0xE6E9EF))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.18), lineWidth: 1))
                .frame(width: 24.5, height: 24.5)
            Image(systemName: "plus").font(.system(size: 13, weight: .heavy)).foregroundStyle(Color(hex: 0x0B1020))
        }
        .frame(width: 28, height: 28)
        .contentShape(Rectangle())
    }
}

private struct SplitIcon: View {
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12).fill(Color(hex: 0x0B0613))
            RoundedRectangle(cornerRadius: 12).stroke(CS.cyan, lineWidth: 2.5)
            HStack(spacing: 6) {
                RoundedRectangle(cornerRadius: 2.5).stroke(CS.cyan, lineWidth: 2.5).frame(width: 10, height: 26)
                RoundedRectangle(cornerRadius: 2.5).stroke(CS.cyan, lineWidth: 2.5).frame(width: 10, height: 26)
            }
        }
        .frame(width: 42, height: 42)
        .shadow(color: Color(hex: 0x1A94FF, opacity: 0.2), radius: 3)
    }
}

private struct CCIcon: View {
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12).fill(Color(hex: 0x0B0613))
            RoundedRectangle(cornerRadius: 12).stroke(CS.cyan, lineWidth: 2.5)
            Image("EdCC").resizable().frame(width: 30, height: 24)
        }
        .frame(width: 42, height: 42)
    }
}

// MARK: - Toasts

enum EditorToast: Equatable {
    case split(String)
    case captionDeleted
    case message(String, undo: Bool = false)

    var text: String {
        switch self {
        case .split(let at): return "Video split at \(at)"
        case .captionDeleted: return "Caption deleted"
        case .message(let text, _): return text
        }
    }

    var hasUndo: Bool {
        switch self {
        case .captionDeleted: return true
        case .message(_, let undo): return undo
        case .split: return false
        }
    }
}

private struct ToastPill: View {
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
        .frame(minHeight: 42)
        .frame(maxWidth: .infinity)
        .background(Color(hex: 0x040812, opacity: 0.97), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(hex: 0x7C3AED, opacity: 0.8), lineWidth: 1))
        .padding(.horizontal, 10)
        .accessibilityIdentifier("editorToast")
    }
}

struct EditorPreviewState {
    var selection: EditorSelection?
    var toast: EditorToast?
    var sheet: EditorSheet?
    var panel: EditorPanel?
    /// Set when a real cut is opened by the editor probe (Debug only).
    var probe: String?
}

enum EditorSheet: String, Identifiable {
    case captions, voiceOver, overlay, broll, export
    var id: String { rawValue }
}

enum EditorTime {
    /// "00:06"
    static func clock(_ seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded(.down)))
        return String(format: "%02d:%02d", total / 60, total % 60)
    }

    /// "00:06.0"
    static func tenths(_ seconds: Double) -> String {
        let clamped = max(0, seconds)
        let whole = Int(clamped)
        let tenth = Int(((clamped - Double(whole)) * 10).rounded(.down))
        return String(format: "%02d:%02d.%d", whole / 60, whole % 60, tenth)
    }

    static func short(_ seconds: Double) -> String { tenths(seconds) }
}

/// Plays the editor preview without the system controls (Figma: clean video with brand border).
struct PlayerLayerView: UIViewRepresentable {
    let player: AVPlayer

    func makeUIView(context: Context) -> PlayerUIView {
        let view = PlayerUIView()
        view.playerLayer.player = player
        view.playerLayer.videoGravity = .resizeAspect
        return view
    }

    func updateUIView(_ view: PlayerUIView, context: Context) {
        if view.playerLayer.player !== player { view.playerLayer.player = player }
    }

    final class PlayerUIView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }
}

/// Readable names for cuts: the project's own title, or "Cut 13 Oct" (date it was made) instead of
/// "Untitled Cut" or an internal file name like "media-B3E69460-…".
enum CutNaming {
    static func isInternal(_ name: String) -> Bool {
        let lower = name.lowercased()
        return name.isEmpty || lower == "untitled cut" || lower.hasPrefix("media-") || lower == "video 1"
            || lower == "recorded clip" || lower.hasPrefix("prepared-")
    }

    static func dated(_ date: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.setLocalizedDateFormatFromTemplate("d MMM")
        return "Cut \(formatter.string(from: date))"
    }

    static func readable(project: Project) -> String {
        if !isInternal(project.title) { return project.title }
        let raw = project.createdAt ?? project.updatedAt ?? ""
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.dateFormat = "yyyy-MM-dd"
        if let date = formatter.date(from: String(raw.prefix(10))) { return dated(date) }
        return dated()
    }
}
