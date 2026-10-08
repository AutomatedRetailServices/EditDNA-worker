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
    /// When set, the caption can be tapped (to fix its words).
    var onTap: (() -> Void)? = nil

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
        let safeScale = CaptionLayout.clampedScale(scale)
        let placement = CaptionLayout.placement(x: x, y: y, scale: safeScale, fontKey: fontKey)
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
        .frame(width: CGFloat(placement.column * unitX + 2 * boxPadding))
        .modifier(CaptionTapModifier(onTap: onTap))
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
