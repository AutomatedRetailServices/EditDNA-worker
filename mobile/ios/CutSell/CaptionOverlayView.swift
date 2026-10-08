import SwiftUI
import CoreText

/// Loads the nine caption typefaces that ship inside the app (the server's
/// own files, see `project.yml`). Registering them here, instead of listing
/// them in Info.plist, keeps one single list of faces: `CaptionFontCatalog`.
@MainActor
enum CaptionFontLoader {
    private static var didRegister = false

    static func registerIfNeeded() {
        guard !didRegister else { return }
        didRegister = true
        for spec in CaptionFontCatalog.all {
            guard let url = Bundle.main.url(forResource: spec.fileName, withExtension: spec.fileExtension) else {
                CutSellDiagnostics.log("caption_font_missing", ["font": spec.key])
                continue
            }
            if !CTFontManagerRegisterFontsForURL(url as CFURL, .process, nil) {
                CutSellDiagnostics.log("caption_font_not_registered", ["font": spec.key])
            }
        }
    }
}

extension Color {
    /// 0xRRGGBB.
    init(captionHex hex: UInt32) {
        self.init(
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255
        )
    }
}

/// Draws the caption of the current moment on top of the video preview,
/// following the same rules the server uses for the exported video
/// (`CaptionPreviewRules`). It fills the video frame it is laid over.
struct CaptionOverlayView: View {
    let cue: CaptionPreviewCue?
    let time: Double
    let preset: String
    let fontKey: String
    let x: Double
    let y: Double
    let scale: Double
    /// Set while the Captions panel is open: the caption can be tapped (to
    /// fix its words), dragged and pinched (position and size for the whole video).
    var editing: CaptionEditing? = nil

    /// Live finger movement, applied on top of the saved position until the
    /// fingers lift; then the result is saved and these go back to zero.
    @State private var dragTranslation: CGSize = .zero
    @State private var pinchScale: Double = 1

    private var onTap: (() -> Void)? { editing?.onTap }

    var body: some View {
        GeometryReader { proxy in
            if let cue {
                if cue.isTimed {
                    timedCaption(cue, frame: proxy.size)
                } else {
                    wholeClipCaption(cue, frame: proxy.size)
                }
            }
        }
        .accessibilityHidden(cue == nil)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(cue?.text ?? "")
        .accessibilityAddTraits(onTap == nil ? [] : .isButton)
        .accessibilityAction { onTap?() }
        .accessibilityIdentifier("preview.captionOverlay")
    }

    // MARK: Timed phrase (Editor v2 look)

    private func timedCaption(_ cue: CaptionPreviewCue, frame: CGSize) -> some View {
        let spec = CaptionFontCatalog.spec(for: fontKey)
        let look = CaptionLook.resolve(preset: preset)
        // Start from where the caption really is (the saved point, kept
        // inside the frame), then follow the fingers. `placement` keeps the
        // result inside the frame at every moment, exactly as the export does.
        let saved = CaptionLayout.placement(x: x, y: y, scale: CaptionLayout.clampedScale(scale), fontKey: fontKey)
        let liveX = saved.centerX / CaptionLayout.frameWidth + Double(dragTranslation.width) / max(1, Double(frame.width))
        let liveY = saved.centerY / CaptionLayout.frameHeight + Double(dragTranslation.height) / max(1, Double(frame.height))
        let safeScale = CaptionLayout.clampedScale(CaptionLayout.clampedScale(scale) * pinchScale)
        let placement = CaptionLayout.placement(x: liveX, y: liveY, scale: safeScale, fontKey: fontKey)
        let renderSize = CaptionLayout.fittedSize(
            text: cue.text, size: placement.fontSize, column: placement.column, fontKey: fontKey
        )
        // Server units (1080x1920) -> points of this preview.
        let unitX = Double(frame.width) / CaptionLayout.frameWidth
        let unitY = Double(frame.height) / CaptionLayout.frameHeight

        let em = spec.emSize(forRenderSize: renderSize) * unitY
        let lineHeight = renderSize * unitY
        let naturalHeight = spec.naturalLineHeight(em: em)
        let baselineShift = spec.baselineShift(renderSize: renderSize) * unitY
        let isBox = look.kind == .box
        let boxPadding = isBox ? CaptionLook.boxPadding * safeScale * unitY : 0
        let verticalPadding = boxPadding + max(0, (lineHeight - naturalHeight) / 2)
        let font = Font.custom(spec.postScriptName, fixedSize: CGFloat(max(1, em)))
        let active = look.kind == .highlight ? cue.activeWordIndex(at: time) : nil

        return CaptionGlyphs(
            words: colouredText(cue, look: look, activeIndex: active),
            plain: Text(cue.text),
            font: font,
            outline: CGFloat(isBox ? 0 : CaptionLook.outline * safeScale * unitY),
            shadow: CGFloat(isBox ? 0 : CaptionLook.shadow * safeScale * unitY),
            // The server's renderer spaces wrapped lines by the render size.
            lineSpacing: CGFloat(max(0, lineHeight - naturalHeight))
        )
        .offset(y: CGFloat(baselineShift))
        .padding(.horizontal, CGFloat(boxPadding))
        .padding(.vertical, CGFloat(verticalPadding))
        .background(look.boxHex.map { Color(captionHex: $0) } ?? Color.clear)
        .overlay {
            if editing != nil {
                // Shows the caption can be moved and resized.
                Rectangle()
                    .stroke(Color.white.opacity(0.8), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    .padding(-4)
            }
        }
        .frame(width: CGFloat(placement.column * unitX + 2 * boxPadding))
        .modifier(CaptionTapModifier(onTap: onTap))
        .modifier(CaptionMoveModifier(
            isActive: editing != nil,
            onChange: { translation, magnification in
                if dragTranslation == .zero && pinchScale == 1 { editing?.onBegin() }
                dragTranslation = translation
                pinchScale = magnification
            },
            onEnd: {
                // Read the latest finger movement (not the values of the
                // last drawing) so the saved spot is exactly where it was left.
                let endX = saved.centerX / CaptionLayout.frameWidth + Double(dragTranslation.width) / max(1, Double(frame.width))
                let endY = saved.centerY / CaptionLayout.frameHeight + Double(dragTranslation.height) / max(1, Double(frame.height))
                let endScale = CaptionLayout.clampedScale(CaptionLayout.clampedScale(scale) * pinchScale)
                let ended = CaptionLayout.placement(x: endX, y: endY, scale: endScale, fontKey: fontKey)
                editing?.onLayoutChange(
                    ended.centerX / CaptionLayout.frameWidth,
                    ended.centerY / CaptionLayout.frameHeight,
                    endScale
                )
                dragTranslation = .zero
                pinchScale = 1
            }
        ))
        .position(x: CGFloat(placement.centerX * unitX), y: CGFloat(placement.centerY * unitY))
    }

    private func colouredText(_ cue: CaptionPreviewCue, look: CaptionLook, activeIndex: Int?) -> Text {
        let base = Color(captionHex: look.textHex)
        guard let activeIndex, let highlightHex = look.highlightHex, !cue.words.isEmpty else {
            return Text(cue.text).foregroundColor(base)
        }
        let highlight = Color(captionHex: highlightHex)
        var output = Text("")
        for (index, word) in cue.words.enumerated() {
            if index > 0 { output = output + Text(" ").foregroundColor(base) }
            output = output + Text(word).foregroundColor(index == activeIndex ? highlight : base)
        }
        return output
    }

    // MARK: One caption for the whole clip (pre-v2 look)

    /// Clips without word timings, or text edited by hand: the server burns
    /// one small caption for the whole clip, lower centre, and ignores the
    /// position/size settings. Drawn the same way here.
    private func wholeClipCaption(_ cue: CaptionPreviewCue, frame: CGSize) -> some View {
        let boxed = ["clean", "box", "box_light"].contains(preset)
        // render.py force_style: Fontsize=24, MarginV=120 on the renderer's
        // default 384x288 canvas.
        let fontSize = max(1, frame.height * 24.0 / 288.0 * 0.895)
        let bottom = frame.height * 120.0 / 288.0
        let column = frame.width * 364.0 / 384.0
        return CaptionGlyphs(
            words: Text(cue.text).foregroundColor(.white),
            plain: Text(cue.text),
            font: Font.custom("ArialMT", fixedSize: fontSize),
            outline: boxed ? 0 : frame.height * 2.0 / 288.0,
            shadow: 0,
            lineSpacing: 0
        )
        .padding(boxed ? frame.height * 1.0 / 288.0 : 0)
        .background(boxed ? Color.black.opacity(0.6) : Color.clear)
        .frame(width: column)
        .modifier(CaptionTapModifier(onTap: onTap))
        .frame(width: frame.width, height: max(0, frame.height - bottom), alignment: .bottom)
    }
}

/// What the caption on the video can do while the Captions panel is open.
struct CaptionEditing {
    /// Tap: fix the words of this caption.
    let onTap: () -> Void
    /// Fingers start moving the caption.
    let onBegin: () -> Void
    /// Fingers lifted: new centre (0...1 of the frame) and size, for the whole video.
    let onLayoutChange: (Double, Double, Double) -> Void
}

/// Drag to move, pinch to resize. The touch area is a little taller than
/// the words so two fingers fit on a small caption.
private struct CaptionMoveModifier: ViewModifier {
    let isActive: Bool
    let onChange: (CGSize, Double) -> Void
    let onEnd: () -> Void

    @State private var latestTranslation: CGSize = .zero
    @State private var latestMagnification: Double = 1

    @ViewBuilder
    func body(content: Content) -> some View {
        if isActive {
            content
                .padding(.vertical, 24)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 6)
                        .simultaneously(with: MagnifyGesture())
                        .onChanged { value in
                            // When one finger lifts, its part of the gesture
                            // stops reporting: keep its last value, never reset.
                            if let drag = value.first { latestTranslation = drag.translation }
                            if let pinch = value.second { latestMagnification = Double(pinch.magnification) }
                            onChange(latestTranslation, latestMagnification)
                        }
                        .onEnded { _ in
                            onEnd()
                            latestTranslation = .zero
                            latestMagnification = 1
                        }
                )
                .padding(.vertical, -24)
        } else {
            content
        }
    }
}

/// Makes the caption itself (only the caption, not the whole video) tappable.
private struct CaptionTapModifier: ViewModifier {
    let onTap: (() -> Void)?

    @ViewBuilder
    func body(content: Content) -> some View {
        if let onTap {
            content
                .contentShape(Rectangle())
                .onTapGesture(perform: onTap)
                .accessibilityAddTraits(.isButton)
                .accessibilityHint("Fix the words of this caption")
        } else {
            content.allowsHitTesting(false)
        }
    }
}

/// The words, with the soft dark edge and shadow the server draws around
/// plain (non-box) captions so they stay readable over any video.
private struct CaptionGlyphs: View {
    let words: Text
    let plain: Text
    let font: Font
    let outline: CGFloat
    let shadow: CGFloat
    let lineSpacing: CGFloat

    private static let directions: [CGSize] = [
        CGSize(width: 1, height: 0), CGSize(width: -1, height: 0),
        CGSize(width: 0, height: 1), CGSize(width: 0, height: -1),
        CGSize(width: 0.7, height: 0.7), CGSize(width: -0.7, height: 0.7),
        CGSize(width: 0.7, height: -0.7), CGSize(width: -0.7, height: -0.7),
    ]

    var body: some View {
        ZStack {
            if shadow > 0 {
                styled(plain.foregroundColor(.black))
                    .offset(x: shadow, y: shadow)
                    .opacity(0.85)
            }
            if outline > 0 {
                ZStack {
                    ForEach(0..<CaptionGlyphs.directions.count, id: \.self) { index in
                        styled(plain.foregroundColor(.black))
                            .offset(
                                x: CaptionGlyphs.directions[index].width * outline,
                                y: CaptionGlyphs.directions[index].height * outline
                            )
                    }
                }
                .compositingGroup()
                .opacity(0.75)
            }
            styled(words)
        }
    }

    private func styled(_ text: Text) -> some View {
        text
            .font(font)
            .multilineTextAlignment(.center)
            .lineSpacing(lineSpacing)
            .lineLimit(nil)
            .fixedSize(horizontal: false, vertical: true)
    }
}
