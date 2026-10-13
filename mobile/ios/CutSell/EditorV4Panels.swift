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

/// Pace and zoom. Honest: the server does not apply pace or zoom to a finished cut yet, so the panel
/// says so instead of pretending; the choice is remembered for when it does.
struct StylePanel: View {
    let onDone: () -> Void
    @AppStorage("cutsell.style.pace") private var pace = "normal"
    @AppStorage("cutsell.style.zoom") private var zoom = "subtle"

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Style").font(CS.inter(15, .semibold)).foregroundStyle(.white)
                Spacer()
                Button(action: onDone) {
                    Text("Done").font(CS.inter(14, .semibold)).foregroundStyle(.white)
                        .frame(width: 62, height: 28)
                        .background(Color(hex: 0x040813), in: Capsule())
                        .overlay(Capsule().stroke(LinearGradient(colors: [CS.cyan, Color(hex: 0xFF21D7)], startPoint: .leading, endPoint: .trailing), lineWidth: 1.5))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 14)
            Text("Pace").font(CS.inter(11)).foregroundStyle(CS.textSecondaryF).padding(.horizontal, 14)
            HStack(spacing: 8) {
                card("Normal", "Natural pauses", selected: pace == "normal") { pace = "normal" }
                card("Fast", "Cuts the pauses", selected: pace == "fast") { pace = "fast" }
            }
            .padding(.horizontal, 10)
            Text("Zoom").font(CS.inter(11)).foregroundStyle(CS.textSecondaryF).padding(.horizontal, 14)
            HStack(spacing: 8) {
                card("Off", "No zoom", selected: zoom == "off") { zoom = "off" }
                card("Subtle", "2–5%", selected: zoom == "subtle") { zoom = "subtle" }
                card("Dynamic", "2–12%", selected: zoom == "dynamic") { zoom = "dynamic" }
            }
            .padding(.horizontal, 10)
            Text("Coming soon: pace and zoom are not applied to this cut yet.")
                .font(CS.inter(10)).foregroundStyle(CS.textMutedF)
                .frame(maxWidth: .infinity)
        }
        .padding(.top, 6)
        .accessibilityIdentifier("stylePanel")
    }

    private func card(_ title: String, _ subtitle: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Text(title).font(CS.inter(14, .semibold)).foregroundStyle(.white)
                Text(subtitle).font(CS.inter(10)).foregroundStyle(CS.textSecondaryF)
            }
            .frame(maxWidth: .infinity).frame(height: 50)
            .background(Color(hex: 0x060B18), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(selected ? Color(hex: 0x1AA6FF) : Color(hex: 0x1C2E52), lineWidth: selected ? 2 : 1))
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Voice-over: recording (1084:2562) and review (1084:2778)

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

    var body: some View {
        VStack(spacing: 0) {
            Capsule().fill(Color(hex: 0x3A5280)).frame(width: 40, height: 4).padding(.top, 8)
            if mode == .voiceReview, let url = recorder.recordedURL {
                review(url)
            } else {
                recording
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(hex: 0x0A1326), in: UnevenRoundedRectangle(topLeadingRadius: 24, topTrailingRadius: 24))
        .overlay(UnevenRoundedRectangle(topLeadingRadius: 24, topTrailingRadius: 24).stroke(Color(hex: 0x1F3A6B), lineWidth: 1))
        .ignoresSafeArea(edges: .bottom)
        .task {
            if recorder.state != .recording && recorder.recordedURL == nil && mode == .voiceRecord {
                await recorder.start()
            }
        }
        .onChange(of: recorder.recordedURL) { _, url in
            if url != nil { mode = .voiceReview }
        }
        .onDisappear { listener.stop() }
        .accessibilityIdentifier("voiceOverPanel")
    }

    // Recording: the wave follows the voice live (newest bars in red).

    private var recording: some View {
        VStack(spacing: 14) {
            HStack(spacing: 8) {
                Circle().fill(Color(hex: 0xFF4D6D)).frame(width: 8, height: 8)
                Text("Recording voice-over").font(CS.inter(16, .bold)).foregroundStyle(.white)
                Spacer()
                Text(VoiceRecorder.formatted(recorder.elapsed))
                    .font(CS.sf(16, .bold)).monospacedDigit().foregroundStyle(Color(hex: 0xFF4D6D))
            }
            .padding(.horizontal, 20).padding(.top, 14)
            LiveWave(levels: recorder.levels)
                .frame(height: 62)
                .padding(.horizontal, 10)
                .background(Color(hex: 0x060B18), in: RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal, 20)
                .accessibilityIdentifier("liveWave")
            Text("The wave follows your voice live")
                .font(CS.inter(11)).foregroundStyle(CS.textMutedF)
            HStack(spacing: 34) {
                VStack(spacing: 6) {
                    Button {
                        if recorder.isPaused { recorder.resume() } else { recorder.pause() }
                    } label: {
                        Image(systemName: recorder.isPaused ? "play.fill" : "pause.fill")
                            .font(.system(size: 17, weight: .bold)).foregroundStyle(.white)
                            .frame(width: 52, height: 52)
                            .background(Color(hex: 0x111B30), in: Circle())
                    }
                    .buttonStyle(.plain)
                    Text(recorder.isPaused ? "Resume" : "Pause").font(CS.inter(11)).foregroundStyle(CS.textSecondaryF)
                }
                VStack(spacing: 6) {
                    Button {
                        if recorder.state == .recording { recorder.stop() } else { Task { await recorder.start() } }
                    } label: {
                        ZStack {
                            Circle().stroke(LinearGradient(colors: [CS.cyan, Color(hex: 0x7C3AED), CS.magenta], startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 4)
                                .frame(width: 72, height: 72)
                            if recorder.state == .recording {
                                RoundedRectangle(cornerRadius: 6).fill(Color(hex: 0xFF4D6D)).frame(width: 26, height: 26)
                            } else {
                                Circle().fill(Color(hex: 0xFF4D6D)).frame(width: 30, height: 30)
                            }
                        }
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("voiceFinish")
                    Text(recorder.state == .recording ? "Finish" : "Record").font(CS.inter(11)).foregroundStyle(CS.textSecondaryF)
                }
            }
            switch recorder.state {
            case .permissionDenied:
                Text("Allow the microphone in Settings to record.").font(CS.inter(12)).foregroundStyle(CS.error)
            case .failed(let message):
                Text(message).font(CS.inter(12)).foregroundStyle(CS.error)
            default:
                EmptyView()
            }
            Button("Cancel") {
                recorder.discardTake()
                mode = .none
            }
            .font(CS.inter(13, .semibold)).foregroundStyle(CS.textSecondaryF)
        }
    }

    // Review: the real waveform; the part already heard lights up and the line moves.

    private func review(_ url: URL) -> some View {
        VStack(spacing: 12) {
            HStack {
                Text("Voice-over \((model.timelineComposition?.voiceOverPlacements.count ?? 0) + 1)")
                    .font(CS.inter(16, .bold)).foregroundStyle(.white)
                Spacer()
                Text("Enhance voice").font(CS.inter(12)).foregroundStyle(CS.textSecondaryF)
                Toggle("", isOn: $enhance).labelsHidden().tint(CS.cyan)
                    .accessibilityIdentifier("enhanceVoice")
            }
            .padding(.horizontal, 20).padding(.top, 12)
            GeometryReader { proxy in
                let progress = listener.duration > 0 ? listener.time / listener.duration : 0
                ZStack(alignment: .leading) {
                    PlayedWave(samples: samples, progress: progress)
                    Rectangle().fill(.white).frame(width: 2, height: 52)
                        .offset(x: proxy.size.width * CGFloat(progress))
                }
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onEnded { value in
                    listener.seek(fraction: Double(value.location.x / max(1, proxy.size.width)))
                })
            }
            .frame(height: 58)
            .padding(.horizontal, 10)
            .background(Color(hex: 0x060B18), in: RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal, 20)
            .accessibilityIdentifier("reviewWave")
            HStack {
                Text(VoiceRecorder.formatted(listener.time))
                Spacer()
                Text(VoiceRecorder.formatted(listener.duration > 0 ? listener.duration : recorder.elapsed))
            }
            .font(CS.sf(11, .semibold)).foregroundStyle(CS.textSecondaryF)
            .padding(.horizontal, 22)
            Text("The part you already heard lights up and the line moves while you listen")
                .font(CS.inter(10)).foregroundStyle(CS.textMutedF)
                .multilineTextAlignment(.center).padding(.horizontal, 20)
            HStack(spacing: 10) {
                outlined(listener.isPlaying ? "Pause" : "Play", icon: listener.isPlaying ? "pause.fill" : "play.fill") {
                    listener.toggle(url: url)
                }
                outlined("Retake", icon: "arrow.counterclockwise") {
                    listener.stop()
                    recorder.discardTake()
                    mode = .voiceRecord
                    Task { await recorder.start() }
                }
                Button { Task { await save(url) } } label: {
                    Label(saving ? "Saving…" : "Save", systemImage: "checkmark")
                        .labelStyle(TrailingIconLabel())
                        .font(CS.inter(15, .semibold)).foregroundStyle(.white)
                        .frame(maxWidth: .infinity).frame(height: 48)
                        .background(LinearGradient(colors: [CS.cyan, Color(hex: 0x7C3AED), CS.magenta], startPoint: .leading, endPoint: .trailing), in: RoundedRectangle(cornerRadius: 14))
                }
                .buttonStyle(.plain)
                .disabled(saving)
                .accessibilityIdentifier("voiceSave")
            }
            .padding(.horizontal, 20)
        }
        .task(id: url) { samples = await Waveform.samples(url: url, count: 64) }
    }

    private func outlined(_ title: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: icon)
                .font(CS.inter(15, .semibold)).foregroundStyle(.white)
                .frame(maxWidth: .infinity).frame(height: 48)
                .background(Color(hex: 0x060B18), in: RoundedRectangle(cornerRadius: 14))
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color(hex: 0x2A3A60), lineWidth: 1.5))
        }
        .buttonStyle(.plain)
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
    @Environment(\.dismiss) private var dismiss
    @State private var assets: [PHAsset] = []
    @State private var status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
    @State private var picking: String?

    private let columns = Array(repeating: GridItem(.flexible(), spacing: 6), count: 3)

    var body: some View {
        VStack(spacing: 0) {
            Capsule().fill(Color(hex: 0x3A5280)).frame(width: 40, height: 4).padding(.top, 8)
            ZStack {
                Text("Photos & Videos").font(CS.inter(17, .bold)).foregroundStyle(.white)
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
            if asset.mediaType == .image || asset.mediaType == .video { list.append(asset) }
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
