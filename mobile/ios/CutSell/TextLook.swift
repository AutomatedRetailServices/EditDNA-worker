import SwiftUI
import UIKit

/// 30 Oct (PO): the look of a caption or a text, chosen in the Fonts / Style sheet. Stored as a small JSON
/// object (`draft.caption_style`, `text_overlays[i].style`) that the server checks
/// (cutsell_worker/text_style.py). Every field is optional: what is not set keeps the look's default, so a
/// draft without a style draws exactly as before.
struct TextLook: Equatable {
    var preset: String? = nil
    var color: UInt32 = 0xFFFFFF
    var opacity: Double = 1
    var strokeColor: UInt32? = nil
    var strokeWidth: Double = 0          // 1080x1920 units at scale 1
    var bgColor: UInt32? = nil
    var bgOpacity: Double = 0
    var bgRadius: Double = 10
    var shadowColor: UInt32? = nil
    var shadowOpacity: Double = 0
    var shadowBlur: Double = 0
    var shadowDistance: Double = 0
    var letterSpacing: Double = 0        // points at scale 1 on a 1080-wide frame
    var lineSpacing: Double = 0
    var align: String = "center"
    var uppercase = false
    /// Captions only: the colour of the word being spoken (nil = no highlight).
    var highlightColor: UInt32? = nil

    // MARK: JSON

    init() {}

    init(json: JSONValue?) {
        guard let object = json?.objectValue else { return }
        preset = object["preset"]?.stringValue
        if let value = Self.hex(object["color"]) { color = value }
        opacity = object["opacity"]?.doubleValue ?? 1
        strokeColor = Self.hex(object["stroke_color"])
        strokeWidth = object["stroke_width"]?.doubleValue ?? 0
        bgColor = Self.hex(object["bg_color"])
        bgOpacity = object["bg_opacity"]?.doubleValue ?? (bgColor == nil ? 0 : 1)
        bgRadius = object["bg_radius"]?.doubleValue ?? 10
        shadowColor = Self.hex(object["shadow_color"])
        shadowOpacity = object["shadow_opacity"]?.doubleValue ?? (shadowColor == nil ? 0 : 0.6)
        shadowBlur = object["shadow_blur"]?.doubleValue ?? 0
        shadowDistance = object["shadow_distance"]?.doubleValue ?? 0
        letterSpacing = object["letter_spacing"]?.doubleValue ?? 0
        lineSpacing = object["line_spacing"]?.doubleValue ?? 0
        align = object["align"]?.stringValue ?? "center"
        uppercase = object["uppercase"]?.boolValue ?? false
        highlightColor = Self.hex(object["highlight_color"])
    }

    var json: JSONValue {
        var out: [String: JSONValue] = [
            "color": .string(Self.hexString(color)), "opacity": .number(Self.round(opacity, 1, 0, 1)),
            "stroke_width": .number(Self.round(strokeWidth, 12, 0, 12)),
            "bg_opacity": .number(Self.round(bgOpacity, 1, 0, 1)), "bg_radius": .number(Self.round(bgRadius, 40, 0, 40)),
            "shadow_opacity": .number(Self.round(shadowOpacity, 1, 0, 1)),
            "shadow_blur": .number(Self.round(shadowBlur, 20, 0, 20)),
            "shadow_distance": .number(Self.round(shadowDistance, 20, 0, 20)),
            "letter_spacing": .number(Self.round(letterSpacing, 30, -5, 30)),
            "line_spacing": .number(Self.round(lineSpacing, 60, -20, 60)),
            "align": .string(["left", "center", "right"].contains(align) ? align : "center"),
            "uppercase": .bool(uppercase),
        ]
        if let preset, Self.presets.contains(where: { $0.key == preset }) { out["preset"] = .string(preset) }
        if let strokeColor { out["stroke_color"] = .string(Self.hexString(strokeColor)) }
        if let bgColor { out["bg_color"] = .string(Self.hexString(bgColor)) }
        if let shadowColor { out["shadow_color"] = .string(Self.hexString(shadowColor)) }
        if let highlightColor { out["highlight_color"] = .string(Self.hexString(highlightColor)) }
        return .object(out)
    }

    private static func round(_ value: Double, _ _scale: Double, _ low: Double, _ high: Double) -> Double {
        (min(high, max(low, value)) * 100).rounded() / 100
    }

    static func hex(_ value: JSONValue?) -> UInt32? {
        guard let text = value?.stringValue, text.count == 7, text.hasPrefix("#") else { return nil }
        return UInt32(text.dropFirst(), radix: 16)
    }

    static func hexString(_ value: UInt32) -> String { String(format: "#%06X", value & 0xFFFFFF) }

    // MARK: The looks offered as "Aa" chips

    struct Preset: Identifiable {
        let key: String
        let look: TextLook
        var id: String { key }
    }

    static let presets: [Preset] = [
        Preset(key: "plain", look: TextLook()),
        Preset(key: "outline", look: { var l = TextLook(); l.strokeColor = 0x000000; l.strokeWidth = 4; return l }()),
        Preset(key: "thick", look: { var l = TextLook(); l.strokeColor = 0x000000; l.strokeWidth = 9; return l }()),
        Preset(key: "shadow", look: { var l = TextLook(); l.shadowColor = 0x000000; l.shadowOpacity = 0.8
            l.shadowBlur = 3; l.shadowDistance = 4; return l }()),
        Preset(key: "dark_box", look: { var l = TextLook(); l.bgColor = 0x000000; l.bgOpacity = 1; return l }()),
        Preset(key: "light_box", look: { var l = TextLook(); l.color = 0x0B1020; l.bgColor = 0xFFFFFF; l.bgOpacity = 1; return l }()),
        Preset(key: "soft_box", look: { var l = TextLook(); l.bgColor = 0x000000; l.bgOpacity = 0.55; return l }()),
        Preset(key: "pop", look: { var l = TextLook(); l.color = 0xFFD60A; l.strokeColor = 0x000000; l.strokeWidth = 6; return l }()),
        Preset(key: "neon", look: { var l = TextLook(); l.color = 0xFFFFFF; l.shadowColor = 0xFF2D95; l.shadowOpacity = 1
            l.shadowBlur = 10; l.shadowDistance = 0; return l }()),
        Preset(key: "glow", look: { var l = TextLook(); l.color = 0x39FF78; l.shadowColor = 0x39FF78; l.shadowOpacity = 0.9
            l.shadowBlur = 8; return l }()),
    ]

    /// A preset applied on top of the colours she already picked? No: a preset is a fresh start (like the
    /// reference app); highlight and alignment stay.
    func applying(preset key: String) -> TextLook {
        guard var next = Self.presets.first(where: { $0.key == key })?.look else { return self }
        next.preset = key
        next.highlightColor = highlightColor
        next.align = align
        next.uppercase = uppercase
        next.letterSpacing = letterSpacing
        next.lineSpacing = lineSpacing
        return next
    }

    /// The captions' look before this sheet existed (caption_preset), so nothing changes until she styles.
    static func legacyCaption(preset: String) -> TextLook {
        let old = CaptionLook.resolve(preset: preset)
        var look = TextLook()
        look.color = old.textHex
        switch old.kind {
        case .box:
            look.bgColor = old.boxHex; look.bgOpacity = 1; look.bgRadius = 0
        case .plain, .highlight:
            look.strokeColor = 0x000000; look.strokeWidth = CaptionLook.outline
            look.shadowColor = 0x000000; look.shadowOpacity = 0.85; look.shadowDistance = CaptionLook.shadow
        }
        look.highlightColor = old.highlightHex
        return look
    }

    // MARK: Colours offered (plus the colour wheel for any other)

    static let palette: [UInt32] = [
        0xFFFFFF, 0xD9D9D9, 0x9E9E9E, 0x5C5C5C, 0x2B2B2B, 0x000000,
        0xFF3B30, 0xFF2D55, 0xFF6B81, 0xFF9500, 0xFFB340, 0xFFD60A,
        0xFFF176, 0xC6FF00, 0x39FF78, 0x34C759, 0x00C7BE, 0x30D5C8,
        0x5AC8FA, 0x1AA6FF, 0x1769FF, 0x0A3D91, 0x5856D6, 0x7C3AED,
        0xAF52DE, 0xFF21D7, 0xF5A3C7, 0xA2845E, 0x6D4C41, 0xF5E6C8,
    ]
}

extension Color {
    init(lookHex hex: UInt32, opacity: Double = 1) {
        self.init(.sRGB, red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255, opacity: opacity)
    }

    /// The colour as 0xRRGGBB (what the colour wheel picked).
    var lookHex: UInt32 {
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        UIColor(self).getRed(&r, green: &g, blue: &b, alpha: &a)
        let c = { (v: CGFloat) in UInt32(max(0, min(255, (v * 255).rounded()))) }
        return (c(r) << 16) | (c(g) << 8) | c(b)
    }
}

/// Words drawn with a TextLook: colour, outline, shadow / glow, background box, spacing.
/// One view for the preview and the on-phone export (they draw the same thing).
struct LookedText: View {
    let text: String
    let look: TextLook
    let font: Font
    /// 1080x1920 units -> points (the frame's height / 1920), times the text's own scale.
    let unit: CGFloat
    var words: [String] = []
    var activeWord: Int? = nil

    private static let directions: [CGSize] = [
        CGSize(width: 1, height: 0), CGSize(width: -1, height: 0), CGSize(width: 0, height: 1), CGSize(width: 0, height: -1),
        CGSize(width: 0.7, height: 0.7), CGSize(width: -0.7, height: 0.7), CGSize(width: 0.7, height: -0.7),
        CGSize(width: -0.7, height: -0.7),
    ]

    private var shown: String { look.uppercase ? text.uppercased() : text }
    private var alignment: TextAlignment { look.align == "left" ? .leading : look.align == "right" ? .trailing : .center }

    var body: some View {
        let stroke = CGFloat(look.strokeWidth) * unit
        ZStack {
            if let shadow = look.shadowColor, look.shadowOpacity > 0 {
                styled(Text(shown).foregroundColor(Color(lookHex: shadow)))
                    .blur(radius: CGFloat(look.shadowBlur) * unit)
                    .offset(x: CGFloat(look.shadowDistance) * unit, y: CGFloat(look.shadowDistance) * unit)
                    .opacity(look.shadowOpacity)
            }
            if let strokeColor = look.strokeColor, stroke > 0 {
                ZStack {
                    ForEach(0..<Self.directions.count, id: \.self) { index in
                        styled(Text(shown).foregroundColor(Color(lookHex: strokeColor)))
                            .offset(x: Self.directions[index].width * stroke, y: Self.directions[index].height * stroke)
                    }
                    if stroke > 3 {   // a thick outline needs the half-way ring too (no gaps in the corners)
                        ForEach(0..<Self.directions.count, id: \.self) { index in
                            styled(Text(shown).foregroundColor(Color(lookHex: strokeColor)))
                                .offset(x: Self.directions[index].width * stroke * 0.5, y: Self.directions[index].height * stroke * 0.5)
                        }
                    }
                }
                .compositingGroup()
            }
            styled(coloured)
        }
        .opacity(look.opacity)
        .padding(.horizontal, look.bgColor != nil && look.bgOpacity > 0 ? 16 * unit : 0)
        .padding(.vertical, look.bgColor != nil && look.bgOpacity > 0 ? 8 * unit : 0)
        .background {
            if let bg = look.bgColor, look.bgOpacity > 0 {
                RoundedRectangle(cornerRadius: CGFloat(look.bgRadius) * unit, style: .continuous)
                    .fill(Color(lookHex: bg, opacity: look.bgOpacity))
            }
        }
    }

    private var coloured: Text {
        let base = Color(lookHex: look.color)
        guard let activeWord, let highlight = look.highlightColor, !words.isEmpty else {
            return Text(shown).foregroundColor(base)
        }
        var out = Text("")
        for (index, word) in words.enumerated() {
            let w = look.uppercase ? word.uppercased() : word
            if index > 0 { out = out + Text(verbatim: " ").foregroundColor(base) }
            out = out + Text(verbatim: w).foregroundColor(index == activeWord ? Color(lookHex: highlight) : base)
        }
        return out
    }

    private func styled(_ text: Text) -> some View {
        text
            .font(font)
            .tracking(CGFloat(look.letterSpacing) * unit)
            .multilineTextAlignment(alignment)
            .lineSpacing(max(0, CGFloat(look.lineSpacing) * unit))
            .lineLimit(nil)
            .fixedSize(horizontal: false, vertical: true)
    }
}
