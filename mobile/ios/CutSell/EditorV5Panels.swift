import AVFoundation
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Editor v5 panels (Figma 04_Editor · "Editor v5 · Paneles")
//
// Every panel has the same shape: a card above the toolbar and under the timeline, with a title,
// a grey subtitle and ✓ (or X) to close. When a panel is tall the video gets smaller so the
// timeline stays in view; a panel never covers the rows.

/// The tornasol gradient of the brand (#19E6FF → #1769FF → #7C3AED → #FF21D7).
enum Tornasol {
    static let colors = [Color(hex: 0x19E6FF), Color(hex: 0x1769FF), Color(hex: 0x7C3AED), Color(hex: 0xFF21D7)]
    static let horizontal = LinearGradient(colors: colors, startPoint: .leading, endPoint: .trailing)
    static func horizontal(opacity: Double) -> LinearGradient {
        LinearGradient(colors: colors.map { $0.opacity(opacity) }, startPoint: .leading, endPoint: .trailing)
    }
    static let angular = AngularGradient(colors: colors + [colors[0]], center: .center)
}

enum EditorPanelKind: Equatable {
    case overlayAdd
    case overlay(String)
    case volume
    case speed
    case voice(String)
    case sound(String)
    case audio

    /// Card height (Figma); the video gets smaller by this much.
    func height(isVideoOverlay: Bool = true) -> CGFloat {
        switch self {
        case .overlayAdd: return 112
        case .overlay: return isVideoOverlay ? 226 : 214
        case .volume: return 176
        case .speed: return 112
        case .voice: return 226
        case .sound: return 124
        case .audio: return 230
        }
    }
}

/// The card every panel uses.
struct PanelCardV5<Accessory: View, Content: View>: View {
    let title: String
    var subtitle: String = ""
    var closeIsCheck = true
    let onClose: () -> Void
    @ViewBuilder var accessory: () -> Accessory
    @ViewBuilder var content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .center, spacing: 8) {
                Text(title).font(.system(size: 17, weight: .semibold)).foregroundStyle(CS.textPrimaryF).lineLimit(1)
                if !subtitle.isEmpty {
                    Text(subtitle).font(.system(size: 14)).foregroundStyle(CS.textSecondaryF).lineLimit(1)
                }
                Spacer(minLength: 4)
                accessory()
                Button(action: onClose) {
                    Image(closeIsCheck ? "Ed5CloseCheck" : "Ed5CloseX").resizable().frame(width: 22, height: 22)
                        .frame(width: 34, height: 34).contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(closeIsCheck ? "Done" : "Close")
                .accessibilityIdentifier("panelClose")
            }
            .padding(.leading, 15).padding(.trailing, 8).padding(.top, 8)
            content()
                .padding(.horizontal, 15)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(Color(hex: 0x0C1730), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(Color(hex: 0x19345F), lineWidth: 1))
        .padding(.horizontal, 16)
    }
}

extension PanelCardV5 where Accessory == EmptyView {
    init(title: String, subtitle: String = "", closeIsCheck: Bool = true, onClose: @escaping () -> Void,
         @ViewBuilder content: @escaping () -> Content) {
        self.title = title; self.subtitle = subtitle; self.closeIsCheck = closeIsCheck; self.onClose = onClose
        self.accessory = { EmptyView() }; self.content = content
    }
}

/// Dark rounded pill button (Mute, Clean up voice, Record again…); blue when on.
struct PillButtonV5: View {
    let title: String
    let systemImage: String
    var isOn = false
    var destructive = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: systemImage).font(.system(size: 14, weight: .semibold))
                Text(title).font(.system(size: 15, weight: .medium)).lineLimit(1).minimumScaleFactor(0.85)
            }
            .foregroundStyle(destructive ? Color(hex: 0xFF4D6D) : .white)
            .padding(.horizontal, 14)
            .frame(height: 36)
            .background(isOn ? Color(hex: 0x1769FF) : Color(hex: 0x0A1326), in: RoundedRectangle(cornerRadius: 18))
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isOn ? .isSelected : [])
    }
}

/// Slider whose dot sits exactly at the % shown (0...maximum); cyan → magenta fill, white dot.
struct GradientSliderV5: View {
    @Binding var value: Double
    let maximum: Double
    let onCommit: () -> Void
    static let knob: CGFloat = 24

    var body: some View {
        GeometryReader { proxy in
            let width = proxy.size.width
            let usable = max(1, width - Self.knob)
            let center = Self.knob / 2 + usable * CGFloat(min(max(0, value / maximum), 1))
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.18)).frame(height: 4).padding(.horizontal, Self.knob / 2)
                Capsule().fill(LinearGradient(colors: [CS.cyan, Color(hex: 0x7C3AED), Color(hex: 0xFF21D7)],
                                              startPoint: .leading, endPoint: .trailing))
                    .frame(width: max(0, center - Self.knob / 2), height: 4)
                    .offset(x: Self.knob / 2)
                Circle().fill(.white).frame(width: Self.knob, height: Self.knob)
                    .shadow(color: .black.opacity(0.35), radius: 2, y: 1)
                    .offset(x: center - Self.knob / 2)
            }
            .frame(height: proxy.size.height)
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { drag in
                    let fraction = Double(min(max(0, (drag.location.x - Self.knob / 2) / usable), 1))
                    var result = (fraction * maximum * 100).rounded() / 100
                    if abs(result - 1) < 0.03 { result = 1 }
                    value = result
                }
                .onEnded { _ in onCommit() })
        }
        .frame(height: 30)
        .accessibilityElement()
        .accessibilityLabel("Volume")
        .accessibilityValue("\(Int((value * 100).rounded()))%")
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: value = min(maximum, value + 0.05)
            case .decrement: value = max(0, value - 0.05)
            @unknown default: break
            }
            onCommit()
        }
        .accessibilityIdentifier("volumeSlider")
    }
}

// MARK: Overlay · add (Figma 1119:670)

struct OverlayAddPanelV5: View {
    let onPhoto: () -> Void
    let onVideo: () -> Void
    let onClose: () -> Void

    var body: some View {
        PanelCardV5(title: "Add overlay", subtitle: "From your Photos", closeIsCheck: false, onClose: onClose) {
            HStack(spacing: 16) {
                choice("Photo", icon: "Ed5Photo", action: onPhoto).accessibilityIdentifier("overlayAddPhoto")
                choice("Video", icon: "Ed5Video", action: onVideo).accessibilityIdentifier("overlayAddVideo")
            }
            .padding(.top, 6)
        }
        .frame(height: EditorPanelKind.overlayAdd.height())
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("overlayAddPanel")
    }

    private func choice(_ title: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(icon).resizable().frame(width: 26, height: 26)
                Text(title).font(.system(size: 16, weight: .semibold)).foregroundStyle(CS.textPrimaryF)
            }
            .frame(maxWidth: .infinity).frame(height: 52)
            .background(Color(hex: 0x0A1326), in: RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }
}

// MARK: Overlay · selected (Figma 1124:1191 · 1124:1590)

struct OverlayPanelV5: View {
    @ObservedObject var model: DraftEditorViewModel
    let overlayID: String
    let onClose: () -> Void

    private var item: OverlayOnScreen? { model.overlaysOnScreen.first { $0.id == overlayID } }
    private var style: OverlayStyle { model.overlayStyle(overlayID) }
    private var isVideo: Bool { item?.kind == .video }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Rectangle().fill(Color(hex: 0x19345F)).frame(height: 1).padding(.horizontal, 15).padding(.top, 4)
            Text("Layout").font(.system(size: 13, weight: .medium)).foregroundStyle(CS.textSecondaryF)
                .padding(.leading, 15).padding(.top, 9)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(OverlayLayout.allCases) { layout in layoutCard(layout) }
                }
                .padding(.horizontal, 15)
            }
            .frame(height: 74)
            .padding(.top, 2)
            HStack {
                Text("Edge").font(.system(size: 13, weight: .medium)).foregroundStyle(CS.textSecondaryF)
                Spacer()
                edgeSegments
            }
            .padding(.horizontal, 15).padding(.top, 10)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .frame(height: EditorPanelKind.overlay(overlayID).height(isVideoOverlay: isVideo))
        .background(Color(hex: 0x0C1730), in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(Color(hex: 0x19345F), lineWidth: 1))
        .padding(.horizontal, 16)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("overlayPanel")
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 0) {
            Image("Ed5Film").resizable().frame(width: 18, height: 18)
                .frame(width: 30, height: 30)
                .background(Tornasol.horizontal(opacity: 0.5), in: RoundedRectangle(cornerRadius: 8))
                .padding(.top, 4)
            VStack(alignment: .leading, spacing: 2) {
                Text(model.overlayTitle(overlayID)).font(.system(size: 16, weight: .semibold)).foregroundStyle(CS.textPrimaryF)
                Text("Overlay · \(EditorTime.minutes(item?.start ?? 0))–\(EditorTime.minutes(item?.end ?? 0))")
                    .font(.system(size: 11.5)).foregroundStyle(CS.textSecondaryF).monospacedDigit()
            }
            .padding(.leading, 8).padding(.top, 5)
            Spacer(minLength: 4)
            if isVideo {
                roundButton(style.muted ? "Muted" : "Sound", image: "Ed5Muted",
                            systemImage: style.muted ? nil : "speaker.wave.2.fill", isOn: !style.muted) {
                    model.setOverlayStyle(id: overlayID) { $0.muted.toggle() }
                }
                .accessibilityIdentifier("overlaySound")
            }
            roundButton("Fill", image: "Ed5Fill", isOn: style.fill) {
                model.setOverlayStyle(id: overlayID) { $0.fill.toggle() }
            }
            .accessibilityIdentifier("overlayFill")
            roundButton("Front", image: "Ed5Front", isOn: false) {
                model.bringOverlayToFront(id: overlayID)
            }
            .accessibilityIdentifier("overlayFront")
            Button(action: onClose) {
                Image("Ed5CloseCheck").resizable().frame(width: 22, height: 22).frame(width: 34, height: 34)
            }
            .buttonStyle(.plain)
            .padding(.top, 4)
            .accessibilityLabel("Done")
            .accessibilityIdentifier("panelClose")
        }
        .padding(.leading, 15).padding(.trailing, 8).padding(.top, 9)
    }

    private func roundButton(_ title: String, image: String, systemImage: String? = nil, isOn: Bool,
                             action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                ZStack {
                    Circle().fill(Color(hex: 0x0A1326))
                    if isOn { Circle().strokeBorder(Tornasol.angular, lineWidth: 2) }
                    if let systemImage {
                        Image(systemName: systemImage).font(.system(size: 15, weight: .semibold)).foregroundStyle(CS.cyan)
                    } else {
                        Image(image).resizable().frame(width: 20, height: 20)
                    }
                }
                .frame(width: 36, height: 36)
                Text(title).font(.system(size: 10.5, weight: .medium))
                    .foregroundStyle(isOn ? CS.cyan : CS.textSecondaryF)
            }
            .frame(width: 52)
        }
        .buttonStyle(.plain)
    }

    private func layoutCard(_ layout: OverlayLayout) -> some View {
        let chosen = style.layout == layout
        return Button {
            model.setOverlayStyle(id: overlayID) { $0.layout = layout }
        } label: {
            VStack(spacing: 5) {
                LayoutMiniPhone(layout: layout)
                    .frame(width: 33.6, height: 53.76)
                    .overlay(RoundedRectangle(cornerRadius: 6.7).stroke(chosen ? Color.white : .clear, lineWidth: 1.7))
                Text(layout.title).font(.system(size: 8.8, weight: .medium))
                    .foregroundStyle(chosen ? CS.cyan : CS.textSecondaryF).lineLimit(1).fixedSize()
            }
            .frame(width: 52)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Layout \(layout.title)")
        .accessibilityAddTraits(chosen ? .isSelected : [])
        .accessibilityIdentifier("layout_\(layout.rawValue)")
    }

    private var edgeSegments: some View {
        HStack(spacing: 0) {
            ForEach(OverlayEdge.allCases) { edge in
                let chosen = style.edge == edge
                Button { model.setOverlayStyle(id: overlayID) { $0.edge = edge } } label: {
                    Text(edge.title).font(.system(size: 13, weight: .medium))
                        .foregroundStyle(chosen ? .white : CS.textSecondaryF)
                        .frame(maxWidth: .infinity).frame(height: 28)
                        .background { if chosen { Capsule().fill(Tornasol.horizontal) } }
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("edge_\(edge.rawValue)")
            }
        }
        .padding(3)
        .frame(width: 232, height: 34)
        .background(Color(hex: 0x0A1326), in: Capsule())
    }
}

/// The small phone of a layout card: grey = the video, tornasol = the overlay.
struct LayoutMiniPhone: View {
    let layout: OverlayLayout

    var body: some View {
        GeometryReader { proxy in
            let w = proxy.size.width, h = proxy.size.height
            ZStack(alignment: .topLeading) {
                Color(hex: 0x6F86AE)
                if let base = layout.baseRegion {
                    Color(hex: 0x6F86AE).frame(width: w * base.width, height: h * base.height)
                }
                if let region = layout.region {
                    Tornasol.horizontal(opacity: 0.55)
                        .frame(width: w * region.width, height: h * region.height)
                        .offset(x: w * region.minX, y: h * region.minY)
                } else {
                    RoundedRectangle(cornerRadius: 2.5).fill(Tornasol.horizontal)
                        .frame(width: 16, height: 20.4)
                        .offset(x: 14.1, y: 6.45)
                }
            }
        }
        .background(Color(hex: 0x0B1730))
        .clipShape(RoundedRectangle(cornerRadius: 6.7))
    }
}

// MARK: Volume (Figma 1119:990): the whole video, or the selected clip

struct VolumePanelV5: View {
    @ObservedObject var model: DraftEditorViewModel
    /// The selected clip (video, voice, sound or video overlay), or the whole video.
    let target: VolumeTarget
    let onClose: () -> Void
    @State private var value: Double = 1

    var body: some View {
        let muted = model.isMuted(target)
        PanelCardV5(title: "Volume", subtitle: target == .whole ? "Whole video" : "Selected clip", onClose: onClose, accessory: {
            PillButtonV5(title: muted ? "Unmute" : "Mute", systemImage: muted ? "speaker.slash.fill" : "speaker.wave.2",
                         isOn: muted) {
                model.setMuted(target, !muted)
            }
            .accessibilityIdentifier("volumeMute")
        }) {
            HStack(spacing: 12) {
                GradientSliderV5(value: $value, maximum: 2) { model.setVolume(target, value) }
                    .opacity(muted ? 0.45 : 1)
                Text(muted ? "Muted" : "\(Int((value * 100).rounded()))%")
                    .font(.system(size: 15, weight: .semibold)).monospacedDigit().foregroundStyle(.white)
                    .frame(width: 58, alignment: .trailing)
                    .accessibilityIdentifier("volumePercent")
            }
            .padding(.top, 8)
            if !model.cleanKeys(for: target).isEmpty {
                CleanUpRowV5(model: model, target: target).padding(.top, 8)
            }
        }
        .frame(height: EditorPanelKind.volume.height())
        .onAppear { value = model.volume(of: target) }
        .onChange(of: target) { _, new in value = model.volume(of: new) }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("volumePanel")
    }
}

/// Clean up voice: a clear On / Off with what it does, and "Cleaning… 40%" while the server makes
/// the clean copy (the preview plays it as soon as it is ready; Off plays the original).
struct CleanUpRowV5: View {
    @ObservedObject var model: DraftEditorViewModel
    let target: VolumeTarget
    @ObservedObject private var center = CleanAudioCenter.shared

    var body: some View {
        let on = model.cleanUpOn(target)
        let keys = model.cleanKeys(for: target)
        let progress = on ? center.progress(keys, project: model.project) : nil
        let failed = on && center.failed(keys, project: model.project)
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 10) {
                Toggle(isOn: Binding(get: { on }, set: { model.setCleanUp(target, $0) })) {
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Clean up voice").font(.system(size: 15, weight: .medium)).foregroundStyle(.white)
                        Text("Removes background noise from the voice")
                            .font(.system(size: 12)).foregroundStyle(CS.textSecondaryF).lineLimit(1).minimumScaleFactor(0.8)
                    }
                }
                .toggleStyle(SwitchToggleStyle(tint: Color(hex: 0x1769FF)))
                .accessibilityIdentifier("cleanUpToggle")
            }
            if let progress {
                HStack(spacing: 8) {
                    ProgressView(value: Double(progress), total: 100).tint(CS.cyan).frame(maxWidth: 140)
                    Text("Cleaning… \(progress)%").font(.system(size: 12, weight: .medium)).monospacedDigit()
                        .foregroundStyle(CS.textSecondaryF)
                        .accessibilityIdentifier("cleanUpProgress")
                }
            } else if failed {
                Button("Couldn't clean the voice. Tap to try again.") { center.retry(keys, project: model.project) }
                    .font(.system(size: 12, weight: .medium)).foregroundStyle(CS.cyan)
                    .accessibilityIdentifier("cleanUpRetry")
            } else if on {
                Text("Playing the clean voice").font(.system(size: 12)).foregroundStyle(CS.textMutedF)
                    .accessibilityIdentifier("cleanUpReady")
            }
        }
    }
}

// MARK: Speed (Figma 1119:1315)

struct SpeedPanelV5: View {
    @ObservedObject var model: DraftEditorViewModel
    let clipID: String?
    let onClose: () -> Void

    var body: some View {
        PanelCardV5(title: "Speed", subtitle: clipID == nil ? "Select a clip" : "Selected clip", onClose: onClose) {
            HStack(spacing: 8) {
                ForEach(ClipSpeed.choices, id: \.self) { speed in
                    let chosen = clipID.map { model.clipSpeed($0) == speed } ?? (speed == 1)
                    Button {
                        if let clipID { model.setClipSpeed(clipID: clipID, speed: speed) }
                    } label: {
                        Text(ClipSpeed.label(speed)).font(.system(size: 15, weight: .medium)).foregroundStyle(.white)
                            .frame(maxWidth: .infinity).frame(height: 42)
                            .background(chosen ? Color(hex: 0x1769FF) : Color(hex: 0x0A1326), in: RoundedRectangle(cornerRadius: 12))
                    }
                    .buttonStyle(.plain)
                    .disabled(clipID == nil)
                    .accessibilityAddTraits(chosen ? .isSelected : [])
                    .accessibilityIdentifier("speed_\(ClipSpeed.label(speed))")
                }
            }
            .padding(.top, 10)
        }
        .frame(height: EditorPanelKind.speed.height())
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("speedPanel")
    }
}

// MARK: Voiceover selected (Figma 1119:1634)

struct VoiceClipPanelV5: View {
    @ObservedObject var model: DraftEditorViewModel
    let clipID: String
    let onRecordAgain: () -> Void
    let onDelete: () -> Void
    let onClose: () -> Void
    @State private var value: Double = 1

    private var clip: AudioClipItem? { model.audioClips.first { $0.id == clipID } }

    var body: some View {
        PanelCardV5(title: clip.map { model.title(of: $0) } ?? "Voiceover", onClose: onClose) {
            HStack(spacing: 12) {
                GradientSliderV5(value: $value, maximum: 2) { setVolume(value) }
                Text("\(Int((value * 100).rounded()))%")
                    .font(.system(size: 15, weight: .semibold)).monospacedDigit().foregroundStyle(.white)
                    .frame(width: 52, alignment: .trailing)
            }
            .padding(.top, 4)
            HStack(spacing: 8) {
                PillButtonV5(title: "Lower video sound", systemImage: "speaker.wave.1", isOn: clip?.duckVideo ?? false) {
                    model.updateAudioClip(id: clipID) { $0.duckVideo.toggle() }
                }
                .accessibilityIdentifier("lowerVideoSound")
                PillButtonV5(title: "Mute", systemImage: (clip?.muted ?? false) ? "speaker.slash.fill" : "speaker.wave.2",
                             isOn: clip?.muted ?? false) {
                    model.setMuted(.audio(clipID), !(clip?.muted ?? false))
                }
                .accessibilityIdentifier("voiceMute")
            }
            .padding(.top, 8)
            CleanUpRowV5(model: model, target: .audio(clipID)).padding(.top, 6)
            HStack(spacing: 8) {
                PillButtonV5(title: "Record again", systemImage: "arrow.counterclockwise", action: onRecordAgain)
                    .accessibilityIdentifier("recordAgain")
                PillButtonV5(title: "Delete", systemImage: "trash", destructive: true, action: onDelete)
                    .accessibilityIdentifier("voiceDelete")
            }
            .padding(.top, 8)
        }
        .frame(height: EditorPanelKind.voice(clipID).height())
        .onAppear { value = clip?.volume ?? 1 }
        .onChange(of: clipID) { _, _ in value = clip?.volume ?? 1 }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("voiceClipPanel")
    }

    private func setVolume(_ volume: Double) {
        model.updateAudioClip(id: clipID) { $0.volume = volume }
    }
}

// MARK: Sound selected (a CutSell sound or music clip)

struct SoundClipPanelV5: View {
    @ObservedObject var model: DraftEditorViewModel
    let clipID: String
    let onDelete: () -> Void
    let onClose: () -> Void
    @State private var value: Double = 1

    private var clip: AudioClipItem? { model.audioClips.first { $0.id == clipID } }

    var body: some View {
        PanelCardV5(title: clip.map { model.title(of: $0) } ?? "Sound",
                    subtitle: clip?.kind == .music ? "Music" : "CutSell sound", onClose: onClose) {
            HStack(spacing: 12) {
                GradientSliderV5(value: $value, maximum: 2) { setVolume(value) }
                Text("\(Int((value * 100).rounded()))%")
                    .font(.system(size: 15, weight: .semibold)).monospacedDigit().foregroundStyle(.white)
                    .frame(width: 52, alignment: .trailing)
            }
            .padding(.top, 4)
            HStack(spacing: 8) {
                if clip?.kind == .music {
                    PillButtonV5(title: "Lower video sound", systemImage: "speaker.wave.1", isOn: clip?.duckVideo ?? false) {
                        model.updateAudioClip(id: clipID) { $0.duckVideo.toggle() }
                    }
                }
                PillButtonV5(title: (clip?.muted ?? false) ? "Unmute" : "Mute",
                             systemImage: (clip?.muted ?? false) ? "speaker.slash.fill" : "speaker.wave.2",
                             isOn: clip?.muted ?? false) {
                    model.setMuted(.audio(clipID), !(clip?.muted ?? false))
                }
                .accessibilityIdentifier("soundMute")
                PillButtonV5(title: "Delete", systemImage: "trash", destructive: true, action: onDelete)
                    .accessibilityIdentifier("soundDelete")
            }
            .padding(.top, 8)
        }
        .frame(height: EditorPanelKind.sound(clipID).height())
        .onAppear { value = clip?.volume ?? 1 }
        .onChange(of: clipID) { _, _ in value = clip?.volume ?? 1 }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("soundClipPanel")
    }

    private func setVolume(_ volume: Double) {
        model.updateAudioClip(id: clipID) { $0.volume = volume }
    }
}

// MARK: Audio · CutSell sounds (Figma 1119:1973)

struct AudioPanelV5: View {
    @ObservedObject var model: DraftEditorViewModel
    @ObservedObject var library = SfxLibrary.shared
    let playhead: () -> Double
    /// (name, clip id) of the sound just added (the editor selects it).
    let onAdded: (String, String) -> Void
    let onMusic: () -> Void
    let onClose: () -> Void
    @State private var chosen: String?

    private let columns = Array(repeating: GridItem(.flexible(), spacing: 10), count: 3)

    var body: some View {
        PanelCardV5(title: "CutSell sounds", subtitle: "Tap one to hear it", onClose: onClose) {
            LazyVGrid(columns: columns, spacing: 10) {
                ForEach(SfxLibrary.buttons, id: \.id) { sound in
                    Button {
                        chosen = sound.id
                        library.play(sound.id)
                    } label: {
                        Text(sound.name).font(.system(size: 16, weight: .medium)).foregroundStyle(.white)
                            .frame(maxWidth: .infinity).frame(height: 54)
                            .background(Color(hex: 0x0A1326), in: RoundedRectangle(cornerRadius: 14))
                            .overlay(RoundedRectangle(cornerRadius: 14).stroke(chosen == sound.id ? CS.cyan : .clear, lineWidth: 1.5))
                            .overlay(alignment: .topTrailing) {
                                // Still downloading: a spinner, and the sound can't be added yet.
                                if !library.ready.contains(sound.id) {
                                    ProgressView().controlSize(.mini).tint(.white).padding(6)
                                        .accessibilityIdentifier("soundDownloading")
                                }
                            }
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("sound_\(sound.id)")
                }
            }
            .padding(.top, 6)
            HStack(spacing: 10) {
                let wanted = chosen ?? "tap"
                let ready = library.ready.contains(wanted)
                Button {
                    // Never a silent clip: added only once its file is on this iPhone.
                    if let clipID = model.addSound(sfxID: wanted, at: playhead()) {
                        onAdded(SfxLibrary.name(of: wanted), clipID)
                    }
                } label: {
                    HStack(spacing: 8) {
                        if ready {
                            Image(systemName: "plus").font(.system(size: 15, weight: .bold))
                        } else {
                            ProgressView().controlSize(.small).tint(.white)
                        }
                        Text(ready ? "Add at playhead" : "Getting sound…").font(.system(size: 15, weight: .medium))
                    }
                    .foregroundStyle(.white).padding(.horizontal, 14).frame(height: 36)
                    .background(Color(hex: 0x1769FF).opacity(ready ? 1 : 0.55), in: RoundedRectangle(cornerRadius: 12))
                }
                .buttonStyle(.plain)
                .disabled(!ready)
                .accessibilityIdentifier("addSoundAtPlayhead")
                Button(action: onMusic) {
                    HStack(spacing: 8) {
                        Image(systemName: "music.note").font(.system(size: 14, weight: .semibold))
                        Text("Music from Files").font(.system(size: 15, weight: .medium))
                    }
                    .foregroundStyle(.white).padding(.horizontal, 12).frame(height: 36)
                    .background(Color(hex: 0x0A1326), in: RoundedRectangle(cornerRadius: 12))
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("musicFromFiles")
            }
            .padding(.top, 12)
            if library.failed && !library.loading {
                Button("Sounds couldn't download. Tap to try again.") { Task { await library.ensure() } }
                    .font(.system(size: 11, weight: .medium)).foregroundStyle(CS.cyan)
                    .padding(.top, 4)
                    .accessibilityIdentifier("soundsRetry")
            }
        }
        .frame(height: EditorPanelKind.audio.height())
        .task { await library.ensure() }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("audioPanel")
    }
}

// MARK: Voice-over in two steps (Figma 1119:2316 record · 1120:4 preview)

enum VoiceSheetStep: Equatable { case record, review }

struct VoiceOverSheetV5: View {
    @ObservedObject var model: DraftEditorViewModel
    @ObservedObject var recorder: VoiceRecorder
    let start: Double
    /// "Record again": the voice-over this take replaces.
    let replacing: String?
    @Binding var countdown: Int?
    let onDone: (DraftEditorViewModel.RecordedVoiceOverOutcome) -> Void
    let onCancel: () -> Void

    @StateObject private var listener = TakePlayer()
    @State private var step: VoiceSheetStep = .record
    @State private var samples: [Float] = []
    @State private var enhance = true
    @State private var saving = false

    static func height(_ step: VoiceSheetStep) -> CGFloat { step == .record ? 318 : 272 }

    private var isRecording: Bool { recorder.state == .recording }
    private var takeURL: URL? { recorder.recordedURL }

    var body: some View {
        VStack(spacing: 0) {
            Capsule().fill(Color(hex: 0x455987)).frame(width: 38, height: 5).padding(.top, 12)
            if step == .record { recordStep } else { reviewStep }
            Spacer(minLength: 0)
        }
        .overlay(alignment: .topTrailing) {
            // Close (like the other panels); also cancels the 3 · 2 · 1 countdown.
            Button(action: onCancel) {
                Image(systemName: "xmark").font(.system(size: 13, weight: .bold)).foregroundStyle(Color(hex: 0xA3B2D6))
                    .frame(width: 30, height: 30).background(Circle().fill(Color(hex: 0x14203A)))
            }
            .buttonStyle(.plain)
            .padding(.top, 12).padding(.trailing, 14)
            .disabled(saving)
            .accessibilityLabel("Close")
            .accessibilityIdentifier("voiceClose")
        }
        .frame(maxWidth: .infinity)
        .frame(height: Self.height(step), alignment: .top)
        .background(Color(hex: 0x060D1F, opacity: 0.98), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 26, style: .continuous).stroke(Color(hex: 0x294D85, opacity: 0.9), lineWidth: 1.5))
        .onChange(of: recorder.recordedURL) { _, url in if url != nil { step = .review } }
        .task(id: takeURL) {
            if let url = takeURL { samples = await Waveform.samples(url: url, count: 60) } else { samples = [] }
        }
        .onAppear { if recorder.recordedURL != nil { step = .review } }
        .onDisappear { listener.stop() }
        .accessibilityIdentifier(step == .record ? "voiceRecordSheet" : "voiceReviewSheet")
    }

    // Step 1: record
    private var recordStep: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Image("Ed4OfficialMic").resizable().frame(width: 26, height: 26)
                Text("Recording voice-over").font(.system(size: 18, weight: .semibold)).foregroundStyle(CS.textPrimaryF)
            }
            .padding(.top, 14)
            Text(isRecording ? (recorder.isPaused ? "Paused · " : "● REC · ") + VoiceRecorder.formatted(recorder.elapsed)
                             : (countdown != nil ? "Get ready…" : "Add your voice to this video"))
                .font(.system(size: 13, weight: .semibold)).foregroundStyle(Color(hex: 0x9EB5DB)).monospacedDigit()
                .padding(.top, 4)
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Enhance voice").font(.system(size: 15, weight: .medium)).foregroundStyle(CS.textPrimaryF)
                    Text("Removes background noise and sounds, so only your voice is heard.")
                        .font(.system(size: 12)).foregroundStyle(CS.textSecondaryF).fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 4)
                Toggle("", isOn: $enhance).labelsHidden().tint(CS.cyan)
                    .disabled(isRecording || countdown != nil)
                    .accessibilityIdentifier("enhanceVoice")
            }
            .padding(.horizontal, 16).padding(.vertical, 10)
            .background(Color(hex: 0x0C1730), in: RoundedRectangle(cornerRadius: 14))
            .padding(.horizontal, 22).padding(.top, 12)
            if isRecording {
                LiveWave(levels: recorder.levels).frame(height: 22).padding(.horizontal, 40).padding(.top, 8)
                    .accessibilityIdentifier("liveWave")
            }
            HStack(alignment: .top, spacing: 0) {
                control("Tap to pause") {
                    Button {
                        guard isRecording else { return }
                        if recorder.isPaused { recorder.resume() } else { recorder.pause() }
                    } label: {
                        ZStack {
                            Circle().fill(Color(hex: 0x1A0F2E))
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
                        .frame(width: 48, height: 48).opacity(isRecording ? 1 : 0.45)
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("voicePause")
                }
                control("Tap / hold to record") {
                    Button { tapMic() } label: {
                        ZStack {
                            Circle().strokeBorder(Tornasol.angular, lineWidth: 3).frame(width: 84, height: 84)
                                .shadow(color: Color(hex: 0x1769FF, opacity: 0.6), radius: isRecording && !recorder.isPaused ? 12 : 6)
                            Image("Ed4VoiceMicBig").resizable().frame(width: 84, height: 84)
                            if isRecording && !recorder.isPaused {
                                Circle().stroke(Color(hex: 0xFF3861), lineWidth: 2).frame(width: 92, height: 92)
                            }
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(countdown != nil)
                    .accessibilityLabel(isRecording ? "Recording" : "Record")
                    .accessibilityIdentifier("voiceRecord")
                }
                control("Tap to finish") {
                    Button { if isRecording { recorder.stop() } } label: {
                        ZStack {
                            Circle().fill(Color(hex: 0x1A0F2E))
                            Circle().stroke(Color(hex: 0xFF3861), lineWidth: 1.5)
                            RoundedRectangle(cornerRadius: 5).fill(Color(hex: 0xFF3861)).frame(width: 20, height: 20)
                        }
                        .frame(width: 48, height: 48).opacity(isRecording ? 1 : 0.45)
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("voiceFinish")
                }
            }
            .padding(.top, 10)
            if case .permissionDenied = recorder.state {
                Text("Allow the microphone in Settings to record.").font(.system(size: 11)).foregroundStyle(CS.error)
            }
        }
    }

    private func control<Content: View>(_ label: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(spacing: 8) {
            ZStack { content() }.frame(height: 88)
            Text(label).font(.system(size: 10, weight: .medium)).foregroundStyle(Color(hex: 0xA3B2D6)).lineLimit(1)
        }
        .frame(maxWidth: .infinity)
    }

    /// Mic: a 3 · 2 · 1 countdown over the video, then it records (tap again pauses / resumes).
    private func tapMic() {
        if isRecording {
            if recorder.isPaused { recorder.resume() } else { recorder.pause() }
            return
        }
        guard countdown == nil else { return }
        listener.stop()
        if takeURL != nil { recorder.discardTake() }
        Task { @MainActor in
            for value in [3, 2, 1] {
                countdown = value
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard countdown != nil else { return }   // cancelled
            }
            countdown = nil
            await recorder.start(enhance: enhance)
        }
    }

    // Step 2: preview
    private var reviewStep: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Image("Ed4OfficialMic").resizable().frame(width: 26, height: 26)
                Text("Voice-over").font(.system(size: 18, weight: .semibold)).foregroundStyle(CS.textPrimaryF)
            }
            .padding(.top, 14)
            Text("Preview your recording before saving.")
                .font(.system(size: 13)).foregroundStyle(Color(hex: 0x8FA8D1)).padding(.top, 4)
            GeometryReader { proxy in
                let progress = listener.duration > 0 ? listener.time / listener.duration : 0
                ZStack(alignment: .leading) {
                    PlayedWave(samples: samples, progress: progress)
                    Rectangle().fill(.white).frame(width: 2, height: 30).offset(x: proxy.size.width * CGFloat(progress))
                }
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onEnded { value in
                    listener.seek(fraction: Double(value.location.x / max(1, proxy.size.width)))
                })
            }
            .frame(height: 30)
            .padding(.horizontal, 12).padding(.vertical, 2)
            .background(Color(hex: 0x0C1730), in: RoundedRectangle(cornerRadius: 8))
            .padding(.horizontal, 40).padding(.top, 18)
            .accessibilityIdentifier("reviewWave")
            HStack {
                Text(VoiceRecorder.formatted(listener.time))
                Spacer()
                Text(VoiceRecorder.formatted(listener.duration > 0 ? listener.duration : recorder.elapsed))
            }
            .font(.system(size: 12, weight: .medium)).monospacedDigit().foregroundStyle(Color(hex: 0x8FA8D1))
            .padding(.horizontal, 30).padding(.top, 8)
            HStack(spacing: 14) {
                Button { if let url = takeURL { listener.toggle(url: url) } } label: {
                    Label(listener.isPlaying ? "Pause" : "Play", systemImage: listener.isPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 15, weight: .semibold)).foregroundStyle(.white)
                        .frame(maxWidth: .infinity).frame(height: 46)
                        .background(Color(hex: 0x0F081A), in: RoundedRectangle(cornerRadius: 14))
                        .overlay(RoundedRectangle(cornerRadius: 14).stroke(CS.magenta, lineWidth: 1))
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("voicePlay")
                Button {
                    listener.stop()
                    recorder.discardTake()
                    step = .record
                } label: {
                    Label("Retake", systemImage: "arrow.counterclockwise")
                        .font(.system(size: 15, weight: .semibold)).foregroundStyle(.white)
                        .frame(maxWidth: .infinity).frame(height: 46)
                        .background(Color(hex: 0x0F081A), in: RoundedRectangle(cornerRadius: 14))
                        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color(hex: 0x731F8C), lineWidth: 1))
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("voiceRetake")
                Button { if let url = takeURL { Task { await save(url) } } } label: {
                    Label(saving ? "Saving…" : "Save", systemImage: "checkmark")
                        .font(.system(size: 15, weight: .semibold)).foregroundStyle(.white)
                        .frame(maxWidth: .infinity).frame(height: 46)
                        .background(LinearGradient(colors: [Color(hex: 0x0AD6FF), Color(hex: 0x3366FF), Color(hex: 0xF01FD6)],
                                                   startPoint: .leading, endPoint: .trailing), in: RoundedRectangle(cornerRadius: 14))
                }
                .buttonStyle(.plain)
                .disabled(saving)
                .accessibilityIdentifier("voiceSave")
            }
            .padding(.horizontal, 24).padding(.top, 20)
        }
    }

    private func save(_ url: URL) async {
        listener.stop()
        saving = true
        var upload = url
        if enhance, let better = await VoiceEnhancer.enhance(url) { upload = better }
        let outcome = await model.addRecordedVoiceOver(fileURL: upload, at: start, enhanced: enhance, replacing: replacing)
        saving = false
        if upload != url { try? FileManager.default.removeItem(at: upload) }
        if outcome != .failed {
            recorder.finishTake()
            onDone(outcome)
        }
    }
}

/// 3 · 2 · 1 over the video before the voice-over records (tornasol ring, Figma 1119:2316).
struct CountdownOverlayV5: View {
    let value: Int
    @State private var progress: CGFloat = 0

    var body: some View {
        ZStack {
            Color.black.opacity(0.35)
            VStack(spacing: 14) {
                ZStack {
                    Circle().stroke(Color.white.opacity(0.15), lineWidth: 6)
                    Circle().trim(from: 0, to: progress)
                        .stroke(Tornasol.angular, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                    Text("\(value)").font(.system(size: 64, weight: .bold)).foregroundStyle(.white)
                }
                .frame(width: 128, height: 128)
                Text("Get ready…").font(.system(size: 17, weight: .semibold)).foregroundStyle(.white)
                Text("Recording starts in \(value) second\(value == 1 ? "" : "s")")
                    .font(.system(size: 13)).foregroundStyle(CS.textSecondaryF)
            }
        }
        .onAppear { restart() }
        .onChange(of: value) { _, _ in restart() }
        .accessibilityIdentifier("voiceCountdown")
    }

    private func restart() {
        progress = 0
        withAnimation(.linear(duration: 1)) { progress = 1 }
    }
}

// MARK: - Music from Files: kept as .m4a (the voice-over upload takes AAC .m4a)

enum MusicImport {
    /// Copies the picked song into an .m4a the upload accepts (nil if it can't be read).
    static func m4a(from url: URL) async -> URL? {
        let access = url.startAccessingSecurityScopedResource()
        defer { if access { url.stopAccessingSecurityScopedResource() } }
        let local = FileManager.default.temporaryDirectory.appendingPathComponent("music-src-\(UUID().uuidString).\(url.pathExtension.isEmpty ? "m4a" : url.pathExtension)")
        guard (try? FileManager.default.copyItem(at: url, to: local)) != nil else { return nil }
        if local.pathExtension.lowercased() == "m4a" { return local }
        let asset = AVURLAsset(url: local)
        guard let export = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetAppleM4A) else { return nil }
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("music-\(UUID().uuidString).m4a")
        export.outputURL = out
        export.outputFileType = .m4a
        await export.export()
        try? FileManager.default.removeItem(at: local)
        return export.status == .completed ? out : nil
    }

    static let types: [UTType] = [.audio, .mp3, .mpeg4Audio, .wav, .aiff]
}
