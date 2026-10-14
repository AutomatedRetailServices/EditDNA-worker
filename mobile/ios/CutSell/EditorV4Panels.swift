import AVFoundation
import Photos
import SwiftUI

// MARK: - Captions menu 2b (Figma 1083:5746 / 1049:2492 / 1049:2638 / 1049:2785)

/// Captions On/Off and Done, the four looks (Classic · Highlight · Box · Yellow), the typefaces that
/// slide sideways, and the look's own option: the color of the spoken word (Highlight) or which box
/// (Box). Every choice uses the server's real caption settings (the export draws the same thing).
struct CaptionStylePanel: View {
    @ObservedObject var model: DraftEditorViewModel
    let onDone: () -> Void

    private enum Look: String, CaseIterable { case classic, highlight, box, yellow }

    private var look: Look {
        let preset = model.captionPreset
        if preset.hasPrefix("highlight") { return .highlight }
        if preset.hasPrefix("box") { return .box }
        if preset == "yellow" { return .yellow }
        return .classic
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text("Captions").font(CS.inter(15, .semibold)).foregroundStyle(.white)
                Spacer()
                Text("On").font(CS.inter(12)).foregroundStyle(CS.textSecondaryF)
                Toggle("", isOn: Binding(
                    get: { model.captionsEnabled },
                    set: { model.applyCaptionSettings(enabled: $0) }
                ))
                .labelsHidden()
                .tint(Color(hex: 0x1677FF))
                .accessibilityIdentifier("captionsOnToggle")
                Button(action: onDone) {
                    Text("Done")
                        .font(CS.inter(14, .semibold)).foregroundStyle(.white)
                        .frame(width: 62, height: 28)
                        .background(Color(hex: 0x040813), in: Capsule())
                        .overlay(Capsule().stroke(LinearGradient(colors: [CS.cyan, Color(hex: 0xFF21D7)], startPoint: .leading, endPoint: .trailing), lineWidth: 1.5))
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("captionsDone")
            }
            .padding(.horizontal, 14)

            HStack(spacing: 6) {
                ForEach(Look.allCases, id: \.self) { item in
                    lookCard(item)
                }
            }
            .padding(.horizontal, 10)

            ScrollViewReader { reader in
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(CaptionFontCatalog.all, id: \.key) { spec in
                        let selected = model.captionFont == spec.key
                        Button { model.applyCaptionSettings(font: spec.key) } label: {
                            Text(spec.displayName)
                                .font(.custom(spec.postScriptName, size: 14))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 12)
                                .frame(height: 34)
                                .background(Color(hex: 0x060B18), in: RoundedRectangle(cornerRadius: 10))
                                .overlay(RoundedRectangle(cornerRadius: 10).stroke(selected ? Color(hex: 0x1AA6FF) : Color(hex: 0x1C2E52), lineWidth: selected ? 2 : 1))
                        }
                        .buttonStyle(.plain)
                        .id(spec.key)
                        .accessibilityIdentifier("captionFont_\(spec.key)")
                    }
                }
                .padding(.horizontal, 10)
            }
            .onAppear { reader.scrollTo(model.captionFont, anchor: .center) }
            .onChange(of: model.captionFont) { _, key in withAnimation { reader.scrollTo(key, anchor: .center) } }
            }

            switch look {
            case .highlight: wordColors
            case .box: boxChoices
            default: EmptyView()
            }

            Text("Drag to move, pinch to resize, tap to fix words.")
                .font(CS.inter(10)).foregroundStyle(CS.textMutedF)
                .frame(maxWidth: .infinity)
        }
        .padding(.top, 6)
        .overlay(alignment: .top) { Rectangle().fill(Color(hex: 0x14336B, opacity: 0.7)).frame(height: 1).offset(y: -4) }
        .accessibilityIdentifier("captionMenu2b")
    }

    private func lookCard(_ item: Look) -> some View {
        let selected = look == item
        return Button {
            switch item {
            case .classic: model.applyCaptionSettings(preset: "classic")
            case .highlight: model.applyCaptionSettings(preset: "highlight_green")
            case .box: model.applyCaptionSettings(preset: "box")
            case .yellow: model.applyCaptionSettings(preset: "yellow")
            }
        } label: {
            VStack(spacing: 4) {
                Group {
                    switch item {
                    case .classic: Text("Aa").foregroundStyle(.white)
                    case .highlight: Text("Aa").foregroundStyle(Color(hex: 0x39FF78))
                    case .box:
                        Text("Aa").foregroundStyle(.black)
                            .padding(.horizontal, 6).background(.white, in: RoundedRectangle(cornerRadius: 3))
                    case .yellow: Text("Aa").foregroundStyle(Color(hex: 0xFFD60A))
                    }
                }
                .font(CS.inter(16, .bold))
                Text(item.rawValue.capitalized).font(CS.inter(10)).foregroundStyle(CS.textSecondaryF)
            }
            .frame(maxWidth: .infinity).frame(height: 54)
            .background(Color(hex: 0x060B18), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(selected ? Color(hex: 0x1AA6FF) : Color(hex: 0x1C2E52), lineWidth: selected ? 2 : 1))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("captionLook_\(item.rawValue)")
    }

    private var wordColors: some View {
        HStack(spacing: 16) {
            Text("Word color").font(CS.inter(11)).foregroundStyle(CS.textSecondaryF)
            ForEach([("highlight_green", 0x39FF78), ("highlight_red", 0xFF453A), ("highlight_blue", 0x1AA6FF)], id: \.0) { preset, color in
                let selected = model.captionPreset == preset || (preset == "highlight_green" && model.captionPreset == "highlight")
                Button { model.applyCaptionSettings(preset: preset) } label: {
                    Circle().fill(Color(hex: UInt32(color))).frame(width: 22, height: 22)
                        .padding(3)
                        .overlay(Circle().stroke(selected ? Color.white : .clear, lineWidth: 2))
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("wordColor_\(preset)")
            }
            Spacer()
        }
        .padding(.horizontal, 14)
    }

    private var boxChoices: some View {
        HStack(spacing: 8) {
            boxChoice("Black box, white words", preset: "box", dark: true)
            boxChoice("White box, black words", preset: "box_light", dark: false)
        }
        .padding(.horizontal, 10)
    }

    private func boxChoice(_ title: String, preset: String, dark: Bool) -> some View {
        let selected = model.captionPreset == preset
        return Button { model.applyCaptionSettings(preset: preset) } label: {
            HStack(spacing: 8) {
                Text("Aa").font(CS.inter(10, .bold))
                    .foregroundStyle(dark ? .white : .black)
                    .frame(width: 22, height: 16)
                    .background(dark ? Color.black : Color.white.opacity(0.85), in: RoundedRectangle(cornerRadius: 2))
                    .overlay(RoundedRectangle(cornerRadius: 2).stroke(Color.white.opacity(0.6), lineWidth: 0.5))
                Text(title).font(CS.inter(10)).foregroundStyle(selected ? .white : CS.textMutedF).lineLimit(1)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity).frame(height: 30)
            .background(Color(hex: 0x060B18), in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(selected ? Color(hex: 0x1AA6FF) : Color(hex: 0x1C2E52), lineWidth: selected ? 2 : 1))
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Style (Figma 1054:2497)

/// Editor → Style.
struct StylePanel: View {
    @ObservedObject var model: DraftEditorViewModel
    let onDone: () -> Void

    /// Editor v4/11 (Figma 1054:2497): only the Auto Zoom AI switch. Overlap audio is chosen
    /// before the cut and is not changed here.
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Style").font(.system(size: 15, weight: .semibold)).foregroundStyle(Color(hex: 0xF2F7FF))
                Spacer()
                Button(action: onDone) {
                    Text("Done").font(.system(size: 13, weight: .semibold)).foregroundStyle(Color(hex: 0xFAFCFF))
                        .frame(width: 62, height: 28)
                        .background(Color(hex: 0x040813, opacity: 0.95), in: Capsule())
                        .overlay(Capsule().stroke(Color(hex: 0x1AA6FF), lineWidth: 1))
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("styleDone")
            }
            .padding(.leading, 4)
            StyleToggleRow(
                title: "Auto Zoom AI", subtitle: "Zooms in on key moments automatically.",
                isOn: Binding(get: { model.autoZoomEnabled }, set: { value in Task { await model.setAutoZoom(value) } }),
                identifier: "editorAutoZoomToggle"
            )
            .frame(height: 66)
            .background(Color(hex: 0x040812), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color(hex: 0x14336B), lineWidth: 1))
        }
        .padding(.horizontal, 10)
    }
}

// MARK: - Voice-over: one sheet to record, listen, retake and save (1084:2562)

struct VoiceOverPanel: View {
    @ObservedObject var model: DraftEditorViewModel
    @ObservedObject var recorder: VoiceRecorder
    let start: Double
    @Binding var mode: EditorPanel
    let onDone: (DraftEditorViewModel.RecordedVoiceOverOutcome) -> Void

    @StateObject private var listener = TakePlayer()
    @State private var samples: [Float] = []
    @State private var enhance = true
    @State private var saving = false
    @State private var holding = false
    @State private var pressID: UUID?

    /// Figma 1084:2562: one sheet over the video — record (pause · mic · finish) on top, and the
    /// take's preview (waveform · Play · Retake · Save) below.
    static let height: CGFloat = 491

    private var takeURL: URL? { recorder.recordedURL }
    private var isRecording: Bool { recorder.state == .recording }

    var body: some View {
        VStack(spacing: 0) {
            Capsule().fill(Color(hex: 0x455987)).frame(width: 38, height: 5).padding(.top, 13)
            recordSection
            previewSection
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity)
        .frame(height: Self.height, alignment: .top)
        .background(Color(hex: 0x060D1F, opacity: 0.98), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 26, style: .continuous).stroke(Color(hex: 0x294D85, opacity: 0.9), lineWidth: 1.5))
        .onChange(of: recorder.recordedURL) { _, url in
            if url != nil { mode = .voiceReview }
        }
        .task(id: takeURL) {
            if let url = takeURL { samples = await Waveform.samples(url: url, count: 60) } else { samples = [] }
        }
        .onDisappear { listener.stop() }
        .accessibilityIdentifier("voiceOverPanel")
    }

    // MARK: Record: pause · mic (tap or hold) · finish

    private var recordSection: some View {
        VStack(spacing: 0) {
            ZStack {
                Text("Recording voice-over").font(CS.inter(18, .bold)).foregroundStyle(Color(hex: 0xF7FAFF))
                Image("Ed4OfficialMic").resizable().frame(width: 28, height: 28)
                    .offset(x: -(195 - 58 - 14))
            }
            .frame(height: 28)
            .padding(.top, 15)
            Text(statusLine)
                .font(CS.inter(13, .bold)).foregroundStyle(Color(hex: 0x9EB5DB))
                .monospacedDigit()
                .padding(.top, 4)
            HStack(spacing: 6) {
                Text("Enhance voice").font(CS.inter(12, .medium)).foregroundStyle(Color(hex: 0xF7FAFF))
                Button { enhance.toggle() } label: {
                    ZStack(alignment: enhance ? .trailing : .leading) {
                        Capsule().fill(enhance ? Color(hex: 0x08CCFA) : Color(hex: 0x2A3A5C))
                            .overlay(Capsule().stroke(Color(hex: 0x59739E, opacity: 0.8), lineWidth: 1))
                        Circle().fill(.white).frame(width: 16, height: 16).padding(3)
                    }
                    .frame(width: 38, height: 22)
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("enhanceVoice")
                Spacer()
            }
            .padding(.leading, 30)
            .padding(.top, 6)
            HStack(alignment: .top, spacing: 0) {
                control(label: recorder.isPaused ? "Tap to resume" : "Tap to pause") {
                    Button {
                        guard isRecording else { return }
                        if recorder.isPaused { recorder.resume() } else { recorder.pause() }
                    } label: {
                        ZStack {
                            Circle().fill(Color(hex: 0x1A0F2E, opacity: 0.98))
                            Circle().stroke(.white, lineWidth: 1.5)
                            if recorder.isPaused {
                                Image(systemName: "play.fill").font(.system(size: 18, weight: .bold)).foregroundStyle(.white)
                            } else {
                                HStack(spacing: 6) {
                                    RoundedRectangle(cornerRadius: 3.5).fill(.white).frame(width: 7, height: 22)
                                    RoundedRectangle(cornerRadius: 3.5).fill(.white).frame(width: 7, height: 22)
                                }
                            }
                        }
                        .frame(width: 48, height: 48)
                        .opacity(isRecording ? 1 : 0.45)
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("voicePause")
                }
                control(label: "Tap / hold to record") { micButton }
                control(label: "Tap to finish") {
                    Button {
                        if isRecording { recorder.stop() }
                    } label: {
                        ZStack {
                            Circle().fill(Color(hex: 0x1A0F2E, opacity: 0.98))
                            Circle().stroke(Color(hex: 0xFF3861, opacity: 0.95), lineWidth: 1.5)
                            RoundedRectangle(cornerRadius: 5).fill(Color(hex: 0xFF3861)).frame(width: 20, height: 20)
                        }
                        .frame(width: 48, height: 48)
                        .opacity(isRecording ? 1 : 0.45)
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("voiceFinish")
                }
            }
            .padding(.top, 2)
            switch recorder.state {
            case .permissionDenied:
                Text("Allow the microphone in Settings to record.").font(CS.inter(11)).foregroundStyle(CS.error)
            case .failed(let message):
                Text(message).font(CS.inter(11)).foregroundStyle(CS.error).lineLimit(1)
            default:
                EmptyView()
            }
        }
    }

    private var statusLine: String {
        if isRecording { return (recorder.isPaused ? "Paused · " : "● REC · ") + VoiceRecorder.formatted(recorder.elapsed) }
        return "Add your voice to this video"
    }

    private func control<Content: View>(label: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(spacing: 8) {
            ZStack { content() }.frame(height: 86)
            Text(label).font(CS.inter(10, .medium)).foregroundStyle(Color(hex: 0xA3B2D6))
                .lineLimit(1).minimumScaleFactor(0.85)
        }
        .frame(maxWidth: .infinity)
    }

    /// The big mic: tap starts (or resumes) recording; holding it records only while held.
    private var micButton: some View {
        ZStack {
            Circle()
                .strokeBorder(AngularGradient(colors: [Color(hex: 0x18E6FF), Color(hex: 0x1669FF), Color(hex: 0x7B39EC), Color(hex: 0xF119D8), Color(hex: 0x18E6FF)], center: .center), lineWidth: 3)
                .frame(width: 84, height: 85)
                .shadow(color: Color(hex: 0x1669FF, opacity: 0.6), radius: isRecording && !recorder.isPaused ? 12 : 6)
            Image("Ed4VoiceMicBig").resizable().frame(width: 84, height: 84)
                .scaleEffect(holding ? 0.94 : 1)
            if isRecording && !recorder.isPaused {
                Circle().stroke(Color(hex: 0xFF3861), lineWidth: 2).frame(width: 92, height: 92)
            }
        }
        .contentShape(Circle())
        .gesture(DragGesture(minimumDistance: 0)
            .onChanged { _ in
                guard !holding, pressID == nil else { return }
                let id = UUID()
                pressID = id
                // Still down after 0.35 s: record while the finger stays on the button.
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                    guard pressID == id, !isRecording else { return }
                    holding = true
                    Task { await beginRecording() }
                }
            }
            .onEnded { _ in
                pressID = nil
                if holding {
                    holding = false
                    if isRecording { recorder.stop() }
                } else {
                    tapMic()
                }
            })
        .accessibilityElement()
        .accessibilityLabel(isRecording ? "Recording" : "Record")
        .accessibilityAddTraits(.isButton)
        .accessibilityIdentifier("voiceRecord")
    }

    private func tapMic() {
        if isRecording {
            if recorder.isPaused { recorder.resume() } else { recorder.pause() }
        } else {
            Task { await beginRecording() }
        }
    }

    private func beginRecording() async {
        guard !isRecording else { return }
        listener.stop()
        if takeURL != nil { recorder.discardTake() }
        mode = .voiceRecord
        await recorder.start()
    }

    // MARK: Preview: waveform · Play · Retake · Save

    private var previewSection: some View {
        let ready = takeURL != nil && !isRecording
        return VStack(spacing: 0) {
            ZStack {
                Text("Voice-over").font(CS.inter(18, .bold)).foregroundStyle(Color(hex: 0xF7FAFF))
                Image("Ed4OfficialMic").resizable().frame(width: 28, height: 28).offset(x: -77)
            }
            .frame(height: 33)
            .padding(.top, 14)
            Text("Preview your recording before saving.")
                .font(CS.inter(13)).foregroundStyle(Color(hex: 0x8FA8D1))
                .padding(.top, 2)
            waveform
                .frame(height: 34)
                .padding(.horizontal, 40)
                .padding(.top, 22)
            HStack {
                Text(VoiceRecorder.formatted(ready ? listener.time : 0))
                Spacer()
                Text(VoiceRecorder.formatted(ready ? (listener.duration > 0 ? listener.duration : recorder.elapsed) : recorder.elapsed))
            }
            .font(CS.inter(12, .medium)).monospacedDigit().foregroundStyle(Color(hex: 0x8FA8D1))
            .padding(.horizontal, 30)
            .padding(.top, 10)
            HStack(spacing: 0) {
                Button {
                    if let url = takeURL { listener.toggle(url: url) }
                } label: {
                    HStack(spacing: 10) {
                        Image(systemName: listener.isPlaying ? "pause.fill" : "play.fill").font(.system(size: 13, weight: .bold)).foregroundStyle(CS.cyan)
                        Text(listener.isPlaying ? "Pause" : "Play").font(CS.inter(14, .semibold)).foregroundStyle(Color(hex: 0xF7FAFF))
                    }
                    .frame(width: 100, height: 46)
                    .background(Color(hex: 0x0F081A), in: RoundedRectangle(cornerRadius: 14))
                    .overlay(RoundedRectangle(cornerRadius: 14).stroke(CS.magenta, lineWidth: 1))
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("voicePlay")
                Spacer(minLength: 0)
                Button {
                    listener.stop()
                    recorder.discardTake()
                    mode = .voiceRecord
                    Task { await recorder.start() }
                } label: {
                    Text("↻  Retake").font(CS.inter(14, .semibold)).foregroundStyle(Color(hex: 0xF7FAFF))
                        .frame(width: 104, height: 46)
                        .background(Color(hex: 0x0F081A), in: RoundedRectangle(cornerRadius: 14))
                        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color(hex: 0x731F8C), lineWidth: 1))
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("voiceRetake")
                Spacer(minLength: 0)
                Button { if let url = takeURL { Task { await save(url) } } } label: {
                    Text(saving ? "Saving…" : "✓  Save").font(CS.inter(14, .semibold)).foregroundStyle(Color(hex: 0xF7FAFF))
                        .frame(width: 112, height: 46)
                        .background(LinearGradient(colors: [Color(hex: 0x0AD6FF), Color(hex: 0x3366FF), Color(hex: 0xF01FD6)], startPoint: .leading, endPoint: .trailing), in: RoundedRectangle(cornerRadius: 14))
                }
                .buttonStyle(.plain)
                .disabled(saving)
                .accessibilityIdentifier("voiceSave")
            }
            .padding(.horizontal, 24)
            .padding(.top, 22)
            .disabled(!ready)
            .opacity(ready ? 1 : 0.45)
            Button("Cancel") {
                listener.stop()
                if isRecording { recorder.stop() }
                recorder.discardTake()
                mode = .none
            }
            .font(CS.inter(12, .semibold)).foregroundStyle(CS.textMutedF)
            .padding(.top, 8)
            .accessibilityIdentifier("voiceCancel")
        }
    }

    /// While recording: the live bars. With a take: its real waveform, the heard part in blue.
    @ViewBuilder
    private var waveform: some View {
        if isRecording {
            LiveWave(levels: recorder.levels)
                .accessibilityIdentifier("liveWave")
        } else {
            GeometryReader { proxy in
                let progress = listener.duration > 0 ? listener.time / listener.duration : 0
                ZStack(alignment: .leading) {
                    PlayedWave(samples: samples, progress: progress)
                    Rectangle().fill(.white).frame(width: 2, height: 34)
                        .offset(x: proxy.size.width * CGFloat(progress))
                }
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onEnded { value in
                    listener.seek(fraction: Double(value.location.x / max(1, proxy.size.width)))
                })
            }
            .accessibilityIdentifier("reviewWave")
        }
    }

    private func save(_ url: URL) async {
        listener.stop()
        saving = true
        var upload = url
        if enhance, let better = await VoiceEnhancer.enhance(url) { upload = better }
        let outcome = await model.addRecordedVoiceOver(fileURL: upload, at: start)
        saving = false
        if upload != url { try? FileManager.default.removeItem(at: upload) }
        if outcome != .failed {
            recorder.finishTake()
            mode = .none
            onDone(outcome)
        }
    }
}

private struct TrailingIconLabel: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 8) { configuration.title; configuration.icon }
    }
}

/// Live bars from the microphone, newest at the right; the newest ones in red.
struct LiveWave: View {
    let levels: [Float]

    var body: some View {
        GeometryReader { proxy in
            let count = max(10, Int(proxy.size.width / 5))
            let shown = Array(levels.suffix(count))
            let padded = Array(repeating: Float(0.06), count: max(0, count - shown.count)) + shown
            HStack(alignment: .center, spacing: 2) {
                ForEach(Array(padded.enumerated()), id: \.offset) { index, level in
                    RoundedRectangle(cornerRadius: 1.5)
                        .fill(index >= count - 6 && level > 0.07 ? Color(hex: 0xFF4D6D) : CS.cyan)
                        .frame(width: 3, height: max(3, proxy.size.height * 0.9 * CGFloat(level)))
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
            .animation(.linear(duration: 0.06), value: levels.count)
        }
    }
}

/// The recording's real waveform; bars already heard in cyan, the rest grey.
struct PlayedWave: View {
    let samples: [Float]
    let progress: Double

    var body: some View {
        GeometryReader { proxy in
            let values = samples.isEmpty ? Array(repeating: Float(0.15), count: 64) : samples
            HStack(alignment: .center, spacing: 2) {
                ForEach(Array(values.enumerated()), id: \.offset) { index, level in
                    RoundedRectangle(cornerRadius: 1.5)
                        .fill(Double(index) / Double(values.count) <= progress ? CS.cyan : Color(hex: 0x7F8DB0))
                        .frame(maxWidth: .infinity)
                        .frame(height: max(3, proxy.size.height * 0.9 * CGFloat(level)))
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
        }
    }
}

/// Plays one take for review and reports where it is.
@MainActor
final class TakePlayer: ObservableObject {
    @Published private(set) var time: Double = 0
    @Published private(set) var duration: Double = 0
    @Published private(set) var isPlaying = false
    private var player: AVAudioPlayer?
    private var timer: Timer?

    func toggle(url: URL) {
        if isPlaying { player?.pause(); isPlaying = false; timer?.invalidate(); return }
        if player?.url != url {
            EditorAudio.preparePlayback()
            player = try? AVAudioPlayer(contentsOf: url)
            duration = player?.duration ?? 0
        }
        player?.play()
        isPlaying = true
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.03, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let player = self.player else { return }
                self.time = player.currentTime
                if !player.isPlaying { self.isPlaying = false; self.time = self.duration; self.timer?.invalidate() }
            }
        }
    }

    func seek(fraction: Double) {
        guard let player else { return }
        player.currentTime = max(0, min(1, fraction)) * player.duration
        time = player.currentTime
    }

    func stop() {
        player?.stop()
        player = nil
        isPlaying = false
        timer?.invalidate()
        time = 0
    }
}

// MARK: - Overlay gallery (Figma 1026:7971)

/// "Photos & Videos": pick one photo or video to place over the video.
struct OverlayGallerySheet: View {
    let onPick: (URL) -> Void
    /// Design screenshots only: show sample tiles instead of the photo library.
    var sampleTiles = false
    /// Full-screen B-roll: only videos.
    var videosOnly = false
    var title = "Photos & Videos"
    @Environment(\.dismiss) private var dismiss
    @State private var assets: [PHAsset] = []
    @State private var status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
    @State private var picking: String?

    private let columns = Array(repeating: GridItem(.flexible(), spacing: 6), count: 3)

    var body: some View {
        VStack(spacing: 0) {
            Capsule().fill(Color(hex: 0x3A5280)).frame(width: 40, height: 4).padding(.top, 8)
            ZStack {
                Text(title).font(CS.inter(17, .bold)).foregroundStyle(.white)
                HStack {
                    Button("Cancel") { dismiss() }.font(CS.inter(15)).foregroundStyle(CS.cyan)
                    Spacer()
                }
            }
            .padding(.horizontal, 20).padding(.vertical, 14)
            if sampleTiles {
                LazyVGrid(columns: columns, spacing: 6) {
                    ForEach(0..<9, id: \.self) { index in
                        RoundedRectangle(cornerRadius: 10)
                            .fill(LinearGradient(colors: [Color(hex: 0x1C3566), Color(hex: 0x2A4F8F)], startPoint: .topLeading, endPoint: .bottomTrailing))
                            .aspectRatio(1, contentMode: .fit)
                            .overlay(RoundedRectangle(cornerRadius: 10).stroke(index == 1 ? CS.cyan : .clear, lineWidth: 2))
                            .overlay(alignment: .bottomTrailing) {
                                if index == 1 || index == 5 {
                                    Text(index == 1 ? "0:08" : "0:15").font(CS.inter(10, .semibold)).foregroundStyle(.white).padding(5)
                                }
                            }
                    }
                }
                .padding(.horizontal, 20)
                Spacer()
            } else             if status == .authorized || status == .limited {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 6) {
                        ForEach(assets, id: \.localIdentifier) { asset in
                            Button { Task { await pick(asset) } } label: {
                                OverlayAssetTile(asset: asset, busy: picking == asset.localIdentifier)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 20)
                }
            } else {
                VStack(spacing: 12) {
                    Text("Allow access to your photos to add one over the video.")
                        .font(CS.inter(14)).foregroundStyle(CS.textSecondaryF).multilineTextAlignment(.center)
                    PrimaryButton(title: "Allow photos", height: 48, cornerRadius: 14, fontSize: 15) {
                        Task {
                            status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
                            load()
                        }
                    }
                }
                .padding(24)
                Spacer()
            }
        }
        .background(Color(hex: 0x0A1326).ignoresSafeArea())
        .presentationDetents([.fraction(0.55), .large])
        .presentationBackground(Color(hex: 0x0A1326))
        .preferredColorScheme(.dark)
        .onAppear { load() }
    }

    private func load() {
        guard status == .authorized || status == .limited else { return }
        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        options.fetchLimit = 300
        let result = PHAsset.fetchAssets(with: options)
        var list: [PHAsset] = []
        result.enumerateObjects { asset, _, _ in
            if asset.mediaType == .video || (!videosOnly && asset.mediaType == .image) { list.append(asset) }
        }
        assets = list
    }

    private func pick(_ asset: PHAsset) async {
        picking = asset.localIdentifier
        defer { picking = nil }
        if asset.mediaType == .video {
            if let video = try? await GalleryExport.export(asset, name: nil) {
                onPick(video.url); dismiss()
            }
            return
        }
        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = true
        options.deliveryMode = .highQualityFormat
        let data: Data? = await withCheckedContinuation { continuation in
            PHImageManager.default().requestImageDataAndOrientation(for: asset, options: options) { data, _, _, _ in
                continuation.resume(returning: data)
            }
        }
        guard let data, let image = UIImage(data: data), let jpeg = image.jpegData(compressionQuality: 0.9) else { return }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("overlay-\(UUID().uuidString).jpg")
        try? jpeg.write(to: url)
        onPick(url)
        dismiss()
    }
}

private struct OverlayAssetTile: View {
    let asset: PHAsset
    let busy: Bool
    @State private var image: UIImage?

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            Rectangle().fill(Color(hex: 0x1C2E52))
            if let image { Image(uiImage: image).resizable().scaledToFill() }
            if asset.mediaType == .video {
                Text(VoiceRecorder.formatted(asset.duration))
                    .font(CS.inter(10, .semibold)).foregroundStyle(.white).padding(5)
            }
            if busy { ProgressView().tint(.white).frame(maxWidth: .infinity, maxHeight: .infinity) }
        }
        .aspectRatio(1, contentMode: .fill)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .task {
            let options = PHImageRequestOptions()
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .opportunistic
            PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: 240, height: 240),
                                                  contentMode: .aspectFill, options: options) { result, _ in
                if let result { image = result }
            }
        }
    }
}
