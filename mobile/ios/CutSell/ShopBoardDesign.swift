import SwiftUI
import UIKit

// Shop board · "Title & design" and Background › Cut (PO 18 Nov 2026, from her 20 example boards: "mira las letras,
// cómo expanden las imágenes, cómo se colocan una encima de otra, cómo se corta backgrounds, recuadros dentro del
// background"). Everything is free and on the iPhone (no AI). Figma page "💡 10 · Shop board": T1 Layout 3082:242,
// T2 Letters 3083:245, T3 Add 3083:385, T4 Background › Cut 3084:253, T5 example 3084:376.
// The board drawing (ShopBoardCanvas) uses these views, so the screen and the 1080 × 1920 export are the same.

// MARK: - Letters (T2)

/// Every face she can pick for a line or a text (Fonts/ShopBoard, SIL Open Font License). Figma T2 order first.
enum ShopLetter: String, CaseIterable, Codable, Identifiable {
    case bodoni, instrument, dmSerif, playfairItalic, cinzel, monsieur, mrsSaint, greatVibes, caveat, archivo, bebas
    case playfair, cormorant, italiana, parisienne, pinyon, allura, sacramento, dancing
    var id: String { rawValue }

    var face: ShopFontFace {
        switch self {
        case .bodoni: ShopFontFace(postScriptName: "BodoniModa-Regular", fileName: "ShopBoard-BodoniModa-Variable", weight: 400, variable: true)
        case .instrument: ShopFontFace(postScriptName: "InstrumentSerif-Regular", fileName: "ShopBoard-InstrumentSerif-Regular")
        case .dmSerif: ShopFontFace(postScriptName: "DMSerifDisplay-Regular", fileName: "ShopBoard-DMSerifDisplay-Regular")
        case .playfairItalic: ShopFontFace(postScriptName: "PlayfairDisplay-Italic", fileName: "ShopBoard-PlayfairDisplay-Italic-Variable",
                                           weight: 400, variable: true)
        case .cinzel: ShopFontFace(postScriptName: "Cinzel-Regular", fileName: "ShopBoard-Cinzel-Variable", weight: 400, variable: true)
        case .monsieur: ShopFontFace(postScriptName: "MonsieurLaDoulaise-Regular", fileName: "ShopBoard-MonsieurLaDoulaise-Regular")
        case .mrsSaint: ShopFontFace(postScriptName: "MrsSaintDelafield-Regular", fileName: "ShopBoard-MrsSaintDelafield-Regular")
        case .greatVibes: ShopFontFace(postScriptName: "GreatVibes-Regular", fileName: "ShopBoard-GreatVibes-Regular")
        case .caveat: ShopFontFace(postScriptName: "Caveat-Regular", fileName: "ShopBoard-Caveat-Variable", weight: 400, variable: true)
        case .archivo: ShopFontFace(postScriptName: "ArchivoBlack-Regular", fileName: "ShopBoard-ArchivoBlack-Regular")
        case .bebas: ShopFontFace(postScriptName: "BebasNeue-Regular", fileName: "ShopBoard-BebasNeue-Regular")
        case .playfair: ShopFontFace(postScriptName: "PlayfairDisplay-Regular", fileName: "ShopBoard-PlayfairDisplay-Variable", weight: 400, variable: true)
        case .cormorant: ShopFontFace(postScriptName: "CormorantGaramond-Light", fileName: "ShopBoard-CormorantGaramond-Variable", weight: 400, variable: true)
        case .italiana: ShopFontFace(postScriptName: "Italiana-Regular", fileName: "ShopBoard-Italiana-Regular")
        case .parisienne: ShopFontFace(postScriptName: "Parisienne-Regular", fileName: "ShopBoard-Parisienne-Regular")
        case .pinyon: ShopFontFace(postScriptName: "PinyonScript-Regular", fileName: "ShopBoard-PinyonScript-Regular")
        case .allura: ShopFontFace(postScriptName: "Allura-Regular", fileName: "ShopBoard-Allura-Regular")
        case .sacramento: ShopFontFace(postScriptName: "Sacramento-Regular", fileName: "ShopBoard-Sacramento-Regular")
        case .dancing: ShopFontFace(postScriptName: "DancingScript-Regular", fileName: "ShopBoard-DancingScript-Variable", weight: 400, variable: true)
        }
    }

    /// DM Serif Display's italic (T1 "Italic mix": "Mom *Uniform*").
    static let dmSerifItalic = ShopFontFace(postScriptName: "DMSerifDisplay-Italic", fileName: "ShopBoard-DMSerifDisplay-Italic")

    var name: String {
        switch self {
        case .bodoni: "Bodoni Moda"
        case .instrument: "Instrument Serif"
        case .dmSerif: "DM Serif Display"
        case .playfairItalic: "Playfair Display"
        case .cinzel: "Cinzel"
        case .monsieur: "Monsieur La Doulaise"
        case .mrsSaint: "Mrs Saint Delafield"
        case .greatVibes: "Great Vibes"
        case .caveat: "Caveat"
        case .archivo: "Archivo Black"
        case .bebas: "Bebas Neue"
        case .playfair: "Playfair Display"
        case .cormorant: "Cormorant Garamond"
        case .italiana: "Italiana"
        case .parisienne: "Parisienne"
        case .pinyon: "Pinyon Script"
        case .allura: "Allura"
        case .sacramento: "Sacramento"
        case .dancing: "Dancing Script"
        }
    }

    /// The sample on its row (Figma T2: words from her example boards).
    var sample: String {
        switch self {
        case .bodoni: "AMAZON FASHION"
        case .instrument: "DUPES THAT DON'T"
        case .dmSerif: "Mom"
        case .playfairItalic: "Elegant Autumn"
        case .cinzel: "NEW ARRIVALS"
        case .monsieur: "fall girls night"
        case .mrsSaint: "Nothing Here"
        case .greatVibes: "Christmas Chic"
        case .caveat: "Flattering Fit ↙"
        case .archivo: "AMAZON"
        case .bebas: "ELEVATE ANY OUTFIT"
        case .playfair: "Fashion Finds"
        case .cormorant: "Cozy at home"
        case .italiana: "THE EDIT"
        case .parisienne: "cozy + chic"
        case .pinyon: "Holiday Edit"
        case .allura: "fall favorites"
        case .sacramento: "girls night"
        case .dancing: "my picks"
        }
    }

    /// Green badge on the row (T2): the faces added on 18 Nov, and Playfair's new italic.
    var badge: String? {
        switch self {
        case .instrument, .dmSerif, .monsieur, .mrsSaint, .caveat, .archivo: "NEW"
        case .playfairItalic: "Italic now"
        default: nil
        }
    }
}

// MARK: - Title layouts (T1)

/// How the title lines sit, copied from the example boards. nil on the model = the board's first look (the letter
/// style pair, "CUTSELL" over "Everyday chic").
enum ShopTitleLayout: String, CaseIterable, Codable, Identifiable {
    case thinSerif, scriptOver, scriptInside, tall, italicMix, boldScript
    var id: String { rawValue }

    var name: String {
        switch self {
        case .thinSerif: "Thin serif"
        case .scriptOver: "Script over"
        case .scriptInside: "Script inside"
        case .tall: "Tall"
        case .italicMix: "Italic mix"
        case .boldScript: "Bold + script"
        }
    }

    /// The big title's own face.
    var bigLetter: ShopLetter {
        switch self {
        case .thinSerif, .scriptInside: .bodoni
        case .scriptOver, .italicMix: .dmSerif
        case .tall: .instrument
        case .boldScript: .archivo
        }
    }

    /// The second line's own face; nil = small spaced capitals in the app's sans (Tall, Italic mix).
    var scriptLetter: ShopLetter? {
        switch self {
        case .thinSerif: .bodoni
        case .scriptOver: .monsieur
        case .scriptInside: .mrsSaint
        case .boldScript: .greatVibes
        case .tall, .italicMix: nil
        }
    }
}

/// The title as one block (330 × 124 design units at the top of the board), for any layout. Same frame as the first
/// look, so moving, resizing and turning it by hand work the same way.
struct ShopLayoutTitle: View {
    let layout: ShopTitleLayout
    let big: String
    let script: String
    var bigLetter: ShopLetter?
    var scriptLetter: ShopLetter?
    let ink: Color
    /// "Size" (0.6 … 1.5) and "Letter space" (−2 … 8 design units) from the Layout tab.
    var scale: CGFloat = 1
    var spacing: CGFloat = 0
    let width: CGFloat

    private var sx: CGFloat { width / ShopLayout.designSize.width }
    private var sy: CGFloat { width * 16 / 9 / ShopLayout.designSize.height }
    private var bigFace: ShopFontFace { (bigLetter ?? layout.bigLetter).face }

    private func line(_ text: String, face: ShopFontFace, size: CGFloat, tracking: CGFloat = 0, top: CGFloat) -> some View {
        Text(verbatim: text)
            .font(ShopFonts.font(face, size: size * sx * scale, opticalSize: size * scale))
            .tracking((tracking + spacing) * sx)
            .foregroundStyle(ink)
            .lineLimit(1).minimumScaleFactor(0.35)
            .frame(width: width - 24 * sx)
            .padding(.top, top * sy)
    }

    /// Small spaced capitals (T1 "Tall" and "Italic mix" second lines; "Thin serif" in Bodoni).
    private func small(_ text: String, top: CGFloat, tracking: CGFloat) -> some View {
        Group {
            if let letter = scriptLetter ?? layout.scriptLetter {
                line(text.uppercased(), face: letter.face, size: 15, tracking: tracking, top: top)
            } else {
                Text(verbatim: text.uppercased())
                    .font(CS.inter(12 * sx * scale, .bold))
                    .tracking((tracking + spacing) * sx)
                    .foregroundStyle(ink)
                    .lineLimit(1).minimumScaleFactor(0.35)
                    .frame(width: width - 24 * sx)
                    .padding(.top, top * sy)
            }
        }
    }

    var body: some View {
        ZStack(alignment: .top) {
            switch layout {
            case .thinSerif:
                line(big.uppercased(), face: bigFace, size: 52, tracking: -1, top: 20)
                small(script, top: 88, tracking: 4.5)
            case .scriptOver:
                line(big, face: bigFace, size: 52, top: 14)
                line(script, face: (scriptLetter ?? .monsieur).face, size: 50, top: 60)
            case .scriptInside:
                scriptInside
            case .tall:
                tall
            case .italicMix:
                small(script, top: 16, tracking: 7)
                italicMix.padding(.top, 34 * sy)
            case .boldScript:
                line(big.uppercased(), face: bigFace, size: 50, top: 14)
                line(script, face: (scriptLetter ?? .greatVibes).face, size: 44, top: 64)
            }
        }
        .frame(width: width, height: 140 * sy, alignment: .top)
    }

    /// "THE *Fall* EDIT": the big title's words split around the script word, the script drawn over the gap.
    private var scriptInside: some View {
        let words = big.uppercased().split(separator: " ").map(String.init)
        let left = words.prefix(max(1, words.count / 2)).joined(separator: " ")
        let right = words.dropFirst(max(1, words.count / 2)).joined(separator: " ")
        let scriptFace = (scriptLetter ?? .mrsSaint).face
        let gap = ShopTextMeasure.width(script, face: scriptFace, size: 70 * scale) * 0.62 * sx
        return ZStack(alignment: .top) {
            HStack(spacing: gap) {
                Text(verbatim: left)
                Text(verbatim: right)
            }
            .font(ShopFonts.font(bigFace, size: 40 * sx * scale, opticalSize: 40 * scale))
            .tracking((1.5 + spacing) * sx)
            .foregroundStyle(ink)
            .lineLimit(1).minimumScaleFactor(0.35)
            .frame(width: width - 24 * sx)
            .padding(.top, 52 * sy)
            line(script, face: scriptFace, size: 70, top: 14)
        }
    }

    /// "DUPES THAT / DON'T LOOK" in two tight lines, a short rule, then small spaced capitals.
    private var tall: some View {
        let lines = ShopTextMeasure.twoLines(big.uppercased())
        return ZStack(alignment: .top) {
            line(lines.0, face: bigFace, size: 50, top: 6)
            if !lines.1.isEmpty { line(lines.1, face: bigFace, size: 50, top: 50) }
            Rectangle().fill(ink).frame(width: 60 * sx, height: 1.5 * sx).padding(.top, (lines.1.isEmpty ? 64 : 104) * sy)
            small(script, top: lines.1.isEmpty ? 72 : 112, tracking: 5)
        }
    }

    /// "Mom *Uniform*": the first word upright, the rest in italic.
    private var italicMix: some View {
        let words = big.split(separator: " ").map(String.init)
        let first = words.first ?? ""
        let rest = words.dropFirst().joined(separator: " ")
        let italic = (bigLetter == nil || bigLetter == .dmSerif) ? ShopLetter.dmSerifItalic : bigFace
        return HStack(spacing: 6 * sx) {
            Text(verbatim: first).font(ShopFonts.font(bigFace, size: 54 * sx * scale))
            if !rest.isEmpty { Text(verbatim: rest).font(ShopFonts.font(italic, size: 54 * sx * scale)) }
        }
        .tracking(spacing * sx)
        .foregroundStyle(ink)
        .lineLimit(1).minimumScaleFactor(0.35)
        .frame(width: width - 24 * sx)
    }
}

enum ShopTextMeasure {
    /// Width of `text` in `face` at `size` (points).
    static func width(_ text: String, face: ShopFontFace, size: CGFloat) -> CGFloat {
        let font = ShopFonts.uiFont(face, size: max(1, size))
        return ceil((text as NSString).size(withAttributes: [.font: font]).width)
    }

    /// The words in two lines of about the same length ("DUPES THAT" / "DON'T LOOK"); one word stays one line.
    static func twoLines(_ text: String) -> (String, String) {
        let words = text.split(separator: " ").map(String.init)
        guard words.count > 1 else { return (text, "") }
        var best = 1, bestDiff = Int.max
        for cut in 1..<words.count {
            let a = words[..<cut].joined(separator: " ").count, b = words[cut...].joined(separator: " ").count
            if abs(a - b) < bestDiff { bestDiff = abs(a - b); best = cut }
        }
        return (words[..<best].joined(separator: " "), words[best...].joined(separator: " "))
    }
}

// MARK: - Things to add (T3)

/// One thing she added to her board (T3): moves, resizes and turns like a product; its hand edit lives in the
/// model's `adjusts` under its id. `front` = drawn over the products (Hold › Front) or behind them (Hold › Back).
struct ShopDecor: Codable, Identifiable, Equatable {
    enum Kind: String, Codable, CaseIterable {
        case text, label, tag, button, frame, line, sparkle, ribbon, shape
        var name: String {
            switch self {
            case .text: "Text"
            case .label: "Label + arrow"
            case .tag: "Tag"
            case .button: "Button"
            case .frame: "Frame"
            case .line: "Line"
            case .sparkle: "Sparkle"
            case .ribbon: "Ribbon"
            case .shape: "Shape"
            }
        }
        /// Kinds with words she can change (Hold › Edit words).
        var hasWords: Bool { [.text, .label, .tag, .button, .ribbon].contains(self) }
        /// Words when it is added (from the example boards).
        var startText: String {
            switch self {
            case .text: "Your text"
            case .label: "Flattering fit"
            case .tag: "Comment YES"
            case .button: "SHOP NOW"
            case .ribbon: "Early Black Friday Sale"
            default: ""
            }
        }
    }

    var id = UUID()
    var kind: Kind
    var text: String
    var color: ShopRGB
    var letter: ShopLetter?
    var front: Bool
    /// Shape: a circle instead of a rounded square.
    var circle = false
    /// Centre in board design units (390 × 844) and its own tilt before any hand edit.
    var x: Double = 195
    var y: Double = 430
    var rotation: Double = 0

    static func make(_ kind: Kind, color: ShopRGB, circle: Bool = false) -> ShopDecor {
        var d = ShopDecor(kind: kind, text: kind.startText, color: color, letter: nil,
                          front: kind != .frame && kind != .shape, circle: circle)
        switch kind {
        case .ribbon: d.y = 200; d.rotation = 10
        case .label: d.rotation = -6
        case .frame: d.y = 422
        default: break
        }
        return d
    }

    /// The face of its words.
    var face: ShopFontFace {
        if let letter { return letter.face }
        switch kind {
        case .label: return ShopLetter.caveat.face
        case .tag, .ribbon: return ShopLetter.playfair.face
        default: return ShopLetter.dmSerif.face
        }
    }

    /// Words on a coloured box (tag, button, ribbon) are dark or white, whichever reads.
    var ink: ShopRGB { color.luminance > 0.45 ? ShopRGB(hex: 0x0B1020) : ShopRGB(hex: 0xFFFFFF) }

    /// Its size before any hand edit, in points per design unit of width (multiply by board width / 390).
    var baseSize: CGSize {
        switch kind {
        case .text: return CGSize(width: ShopTextMeasure.width(text, face: face, size: 30) + 16, height: 46)
        case .label: return CGSize(width: max(70, ShopTextMeasure.width(text, face: face, size: 24) + 12), height: 76)
        case .tag: return CGSize(width: ShopTextMeasure.width(text, face: face, size: 22) + 30, height: 42)
        case .button: return CGSize(width: ShopTextMeasure.width(text.uppercased(), face: ShopDecor.buttonFace, size: 13) + 13 * 0.2 * CGFloat(text.count) + 40, height: 40)
        case .frame: return CGSize(width: 350, height: 650)
        case .line: return CGSize(width: 130, height: 2)
        case .sparkle: return CGSize(width: 64, height: 64)
        case .ribbon: return CGSize(width: 560, height: 36)
        case .shape: return CGSize(width: 150, height: 150)
        }
    }

    static let buttonFace = ShopFontFace(postScriptName: "PlusJakartaSans-Bold", fileName: "PlusJakartaSans-Bold")

    /// Where it is drawn on a board `size` big (after the hand edit `adjust`), and its tilt.
    func frame(board size: CGSize, adjust: ShopAdjust) -> (rect: CGRect, rotation: Double) {
        let sx = size.width / ShopLayout.designSize.width, sy = size.height / ShopLayout.designSize.height
        let s = baseSize
        let base = CGRect(x: x * sx - s.width * sx / 2, y: y * sy - s.height * sx / 2, width: s.width * sx, height: s.height * sx)
        return (adjust.frame(base: base, board: size), rotation + adjust.rotation)
    }

    /// Is `point` on it? A frame is grabbed only by its lines (the products inside stay easy to move).
    func contains(_ point: CGPoint, board size: CGSize, adjust: ShopAdjust) -> Bool {
        let placed = frame(board: size, adjust: adjust)
        guard ShopBoardSnapshot.contains(placed.rect, rotation: placed.rotation, point, minimum: 44) else { return false }
        guard kind == .frame else { return true }
        let band: CGFloat = 22
        let inner = placed.rect.insetBy(dx: band, dy: band)
        return !ShopBoardSnapshot.contains(inner, rotation: placed.rotation, point, minimum: 0)
    }
}

/// One added thing, drawn at its base size × `scale` (sx = board width / 390).
struct ShopDecorView: View {
    let decor: ShopDecor
    let sx: CGFloat

    var body: some View {
        let s = decor.baseSize
        let w = s.width * sx, h = s.height * sx
        Group {
            switch decor.kind {
            case .text:
                Text(verbatim: decor.text).font(ShopFonts.font(decor.face, size: 30 * sx)).foregroundStyle(decor.color.color)
                    .lineLimit(1).fixedSize()
            case .label:
                VStack(spacing: 2 * sx) {
                    Text(verbatim: decor.text).font(ShopFonts.font(decor.face, size: 24 * sx)).foregroundStyle(decor.color.color)
                        .lineLimit(1).fixedSize()
                    ShopArrow().stroke(decor.color.color, style: StrokeStyle(lineWidth: 2 * sx, lineCap: .round, lineJoin: .round))
                        .frame(width: 26 * sx, height: 40 * sx)
                }
            case .tag:
                Text(verbatim: decor.text).font(ShopFonts.font(decor.face, size: 22 * sx)).foregroundStyle(decor.ink.color)
                    .lineLimit(1).fixedSize()
                    .frame(width: w, height: h).background(decor.color.color)
            case .button:
                Text(verbatim: decor.text.uppercased()).font(ShopFonts.font(ShopDecor.buttonFace, size: 13 * sx))
                    .tracking(13 * 0.2 * sx).foregroundStyle(decor.ink.color).lineLimit(1).fixedSize()
                    .frame(width: w, height: h).background(decor.color.color)
            case .frame:
                ZStack {
                    Rectangle().strokeBorder(decor.color.color, lineWidth: 3.4 * sx)
                    Rectangle().strokeBorder(decor.color.color, lineWidth: 1.4 * sx).padding(8 * sx)
                }
            case .line:
                Rectangle().fill(decor.color.color)
            case .sparkle:
                ZStack(alignment: .topLeading) {
                    ShopSparkle().fill(decor.color.color).frame(width: 40 * sx, height: 40 * sx)
                    ShopSparkle().fill(decor.color.color).frame(width: 18 * sx, height: 18 * sx).offset(x: 40 * sx, y: 40 * sx)
                }
            case .ribbon:
                let piece = decor.text + "  /  "
                Text(verbatim: String(repeating: piece, count: 6)).font(ShopFonts.font(decor.face, size: 20 * sx))
                    .foregroundStyle(decor.ink.color).lineLimit(1).fixedSize()
                    .frame(width: w, height: h, alignment: .leading).clipped()
                    .background(decor.color.color)
            case .shape:
                if decor.circle { Circle().fill(decor.color.color) }
                else { RoundedRectangle(cornerRadius: 28 * sx, style: .continuous).fill(decor.color.color) }
            }
        }
        .frame(width: w, height: h)
    }
}

/// The curved arrow under a label (T3 "Label + arrow").
struct ShopArrow: Shape {
    func path(in r: CGRect) -> Path {
        var p = Path()
        let start = CGPoint(x: r.maxX * 0.85, y: r.minY)
        let end = CGPoint(x: r.minX + r.width * 0.25, y: r.maxY - r.height * 0.04)
        p.move(to: start)
        p.addCurve(to: end, control1: CGPoint(x: r.maxX * 1.05, y: r.height * 0.45), control2: CGPoint(x: r.minX + r.width * 0.2, y: r.height * 0.55))
        p.move(to: end)
        p.addLine(to: CGPoint(x: end.x - r.width * 0.28, y: end.y - r.height * 0.2))
        p.move(to: end)
        p.addLine(to: CGPoint(x: end.x + r.width * 0.32, y: end.y - r.height * 0.14))
        return p
    }
}

/// A four-point star ✦ (T3 "Sparkle").
struct ShopSparkle: Shape {
    func path(in r: CGRect) -> Path {
        var p = Path()
        let c = CGPoint(x: r.midX, y: r.midY), a = r.width / 2, b = r.height / 2, k: CGFloat = 0.16
        p.move(to: CGPoint(x: c.x, y: c.y - b))
        p.addQuadCurve(to: CGPoint(x: c.x + a, y: c.y), control: CGPoint(x: c.x + a * k, y: c.y - b * k))
        p.addQuadCurve(to: CGPoint(x: c.x, y: c.y + b), control: CGPoint(x: c.x + a * k, y: c.y + b * k))
        p.addQuadCurve(to: CGPoint(x: c.x - a, y: c.y), control: CGPoint(x: c.x - a * k, y: c.y + b * k))
        p.addQuadCurve(to: CGPoint(x: c.x, y: c.y - b), control: CGPoint(x: c.x - a * k, y: c.y - b * k))
        p.closeSubpath()
        return p
    }
}

// MARK: - Background › Cut (T4)

enum ShopCut: String, CaseIterable, Codable, Identifiable {
    case full, photoColor, band, innerCard, doubleFrame, arch
    var id: String { rawValue }
    var name: String {
        switch self {
        case .full: "Full"
        case .photoColor: "Photo + color"
        case .band: "Band"
        case .innerCard: "Inner card"
        case .doubleFrame: "Double frame"
        case .arch: "Arch"
        }
    }
}

/// Drawn over the background, under everything else. `photo` = her photo for "Photo + color" (else the frame colour).
struct ShopCutView: View {
    let cut: ShopCut
    let color: ShopRGB
    var photo: UIImage?
    let size: CGSize

    var body: some View {
        let w = size.width, h = size.height, sx = w / ShopLayout.designSize.width
        ZStack(alignment: .topLeading) {
            switch cut {
            case .full:
                Color.clear
            case .photoColor:
                Group {
                    if let photo { Image(uiImage: photo).resizable().scaledToFill() } else { color.color }
                }
                .frame(width: w * 0.38, height: h).clipped()
            case .band:
                color.color.frame(width: w, height: h * 0.2).offset(y: h * 0.74)
            case .innerCard:
                ShopInnerCardShape(inset: w * 0.06, radius: w * 0.03)
                    .fill(color.color, style: FillStyle(eoFill: true))
            case .doubleFrame:
                Rectangle().strokeBorder(color.color, lineWidth: 3.4 * sx)
                    .frame(width: w - 28 * sx, height: h - 28 * sx).offset(x: 14 * sx, y: 14 * sx)
                Rectangle().strokeBorder(color.color, lineWidth: 1.4 * sx)
                    .frame(width: w - 44 * sx, height: h - 44 * sx).offset(x: 22 * sx, y: 22 * sx)
            case .arch:
                UnevenRoundedRectangle(topLeadingRadius: w * 0.3, topTrailingRadius: w * 0.3, style: .continuous)
                    .fill(color.color)
                    .frame(width: w * 0.6, height: h * 0.87).offset(x: w * 0.2, y: h * 0.13)
            }
        }
        .frame(width: w, height: h, alignment: .topLeading)
        .allowsHitTesting(false)
    }
}

/// The whole board minus a card inset on every side (even-odd fill: the colour around the card).
struct ShopInnerCardShape: Shape {
    let inset: CGFloat
    let radius: CGFloat
    func path(in r: CGRect) -> Path {
        var p = Path(r)
        p.addRoundedRect(in: r.insetBy(dx: inset, dy: inset), cornerSize: CGSize(width: radius, height: radius), style: .continuous)
        return p
    }
}

// MARK: - The sheets (T1 · T2 · T3 · T4)

/// The grey segmented tabs of T1 – T4: 342 × 36, radius 12, the picked one on a white pill.
struct ShopSegmentTabs: View {
    let titles: [String]
    @Binding var selected: Int
    var idPrefix = "shopDesignTab"

    var body: some View {
        HStack(spacing: 0) {
            ForEach(titles.indices, id: \.self) { i in
                let on = i == selected
                Button { withAnimation(.easeOut(duration: 0.15)) { selected = i } } label: {
                    Text(L10n.tr(titles[i]))
                        .font(CS.inter(13, on ? .heavy : .semibold))
                        .foregroundStyle(Color(fixed: on ? 0x1769FF : 0x6B7385))
                        .lineLimit(1).minimumScaleFactor(0.8)
                        .frame(maxWidth: .infinity).frame(height: 30)
                        .background {
                            if on {
                                RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color.white)
                                    .shadow(color: Color.black.opacity(0.08), radius: 2, x: 0, y: 1)
                            }
                        }
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(on ? .isSelected : [])
                .accessibilityIdentifier("\(idPrefix)_\(i)")
            }
        }
        .padding(3)
        .frame(height: 36)
        .background(Color(fixed: 0xF3F5F9), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

/// The grey label over each part (Bold 12, #6B7385).
struct ShopSheetLabel: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(L10n.tr(text)).font(CS.inter(12, .bold)).foregroundStyle(Color(fixed: 0x6B7385))
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A row of swatches on the #F3F5F9 bar (T1 Title color, T3 Color, T4 Frame color).
struct ShopSwatchRow<Leading: View>: View {
    let colors: [ShopRGB]
    let isChosen: (Int) -> Bool
    let pick: (Int) -> Void
    var spacing: CGFloat = 8
    var idPrefix = "shopSwatch"
    @ViewBuilder let leading: () -> Leading

    var body: some View {
        HStack(spacing: 0) {
            leading()
            Spacer(minLength: 8)
            HStack(spacing: spacing) {
                ForEach(colors.indices, id: \.self) { i in
                    let chosen = isChosen(i)
                    Button { pick(i) } label: {
                        Circle().fill(colors[i].color).frame(width: 24, height: 24)
                            .overlay(Circle().strokeBorder(chosen ? Color(fixed: 0x1769FF) : Color(fixed: 0xE6E9EF), lineWidth: chosen ? 3 : 1))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(L10n.tr(ShopPalette.name(of: colors[i])))
                    .accessibilityAddTraits(chosen ? .isSelected : [])
                    .accessibilityIdentifier("\(idPrefix)_\(i)")
                }
            }
        }
        .padding(.horizontal, 14)
        .frame(height: 46)
        .background(Color(fixed: 0xF3F5F9), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

/// "Title & design" (T1 – T3): Words · Layout · Letters · Add.
struct ShopTitleDesignSheet: View {
    @ObservedObject var model: ShopBoardModel
    let onDone: () -> Void
    @State private var tab = 1
    /// Letters › "Change letters of": 0 big title, 1 script line, 2 the picked text on the board.
    @State private var target = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(L10n.tr("Title & design")).font(CS.inter(22, .heavy)).foregroundStyle(Color(fixed: 0x0B1020))
            ShopSegmentTabs(titles: ["Words", "Layout", "Letters", "Add"], selected: $tab).padding(.top, 10)
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    switch tab {
                    case 0: words
                    case 1: layoutTab
                    case 2: letters
                    default: addTab
                    }
                }
                .padding(.top, 12)
                .padding(.bottom, 8)
            }
            ShopBigButton(title: "Done", id: "shopTitleDone", action: onDone)
                .padding(.horizontal, -4)
                .padding(.top, 12)
                .padding(.bottom, 16)
        }
        .padding(.horizontal, 24)
        .frame(maxHeight: 680)
        .onAppear { if model.selectedDecorHasWords { target = 2 } }
        .accessibilityIdentifier("shopTitleSheet")
    }

    // MARK: Words (the three lines of the first Title sheet)

    private var words: some View {
        VStack(alignment: .leading, spacing: 0) {
            ShopSheetLabel("Big title  (e.g. Amazon finds, Walmart, your shop)")
            field($model.bigTitle, id: "shopBigTitle").padding(.top, 6)
            ShopSheetLabel("Script line").padding(.top, 10)
            field($model.scriptLine, id: "shopScriptLine").padding(.top, 6)
            ShopSheetLabel("Bottom line  (your brand or shop)").padding(.top, 10)
            field($model.bottomLine, id: "shopBottomLine").padding(.top, 6)
            if let id = model.selectedDecor, let decor = model.decor.first(where: { $0.id == id }), decor.kind.hasWords {
                ShopSheetLabel("Picked text on your board").padding(.top, 10)
                field(Binding(get: { decor.text }, set: { model.setText($0, for: id) }), id: "shopDecorText").padding(.top, 6)
            }
        }
    }

    private func field(_ text: Binding<String>, id: String) -> some View {
        TextField("", text: text)
            .font(CS.inter(15, .semibold))
            .foregroundStyle(Color(fixed: 0x0B1020))
            .padding(.horizontal, 13)
            .frame(height: 44)
            .background(Color(fixed: 0xFFFFFF), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color(fixed: 0xE6E9EF), lineWidth: 1))
            .accessibilityIdentifier(id)
    }

    // MARK: Layout (T1)

    private var layoutTab: some View {
        VStack(alignment: .leading, spacing: 0) {
            ShopSheetLabel("Title layout · like the boards you love")
            LazyVGrid(columns: Array(repeating: GridItem(.fixed(106), spacing: 12), count: 3), spacing: 12) {
                ForEach(ShopTitleLayout.allCases) { layout in layoutTile(layout) }
            }
            .padding(.top, 8)
            slider("Size", value: $model.titleSize, range: 0.6...1.5, id: "shopTitleSize").padding(.top, 18)
            slider("Letter space", value: $model.titleSpacing, range: -2...8, id: "shopTitleSpacing").padding(.top, 14)
            titleColorRow.padding(.top, 16)
            Text(L10n.tr("Tap any line on your board to move it, turn it or change its letters"))
                .font(CS.inter(12, .medium)).foregroundStyle(Color(fixed: 0x1769FF))
                .multilineTextAlignment(.center).frame(maxWidth: .infinity).padding(.top, 14)
        }
    }

    private func layoutTile(_ layout: ShopTitleLayout) -> some View {
        let on = model.titleLayout == layout
        return Button { model.titleLayout = layout; model.bigLetter = nil; model.scriptLetter = nil } label: {
            ZStack(alignment: .top) {
                ShopLayoutSample(layout: layout).frame(width: 106, height: 76)
                VStack {
                    Spacer()
                    Text(L10n.tr(layout.name)).font(CS.inter(11, .bold))
                        .foregroundStyle(Color(fixed: on ? 0x1769FF : 0x6B7385)).padding(.bottom, 6)
                }
            }
            .frame(width: 106, height: 100)
            .background(Color(fixed: on ? 0xEEF3FF : 0xF7F4EF), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color(fixed: on ? 0x1769FF : 0xE6E9EF), lineWidth: on ? 2 : 1))
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
        .buttonStyle(BubblePress())
        .accessibilityAddTraits(on ? .isSelected : [])
        .accessibilityIdentifier("shopLayout_\(layout.rawValue)")
    }

    /// Bold 13 label at the left, the track from x 126 (240 long), a 20 pt knob with a blue ring.
    private func slider(_ title: String, value: Binding<Double>, range: ClosedRange<Double>, id: String) -> some View {
        HStack(spacing: 0) {
            Text(L10n.tr(title)).font(CS.inter(13, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                .frame(width: 102, alignment: .leading)
            ShopSlider(value: value, range: range).frame(height: 24)
        }
        .accessibilityIdentifier(id)
    }

    private var titleColorRow: some View {
        let matched = ShopBoardStyle(matchBackground: model.style.matchBackground, background: model.style.background,
                                     matchTitle: true).resolvedTitleColor(palette: model.palette, template: model.template)
        let white = ShopRGB(hex: 0xFFFFFF)
        let others = ([white] + model.palette).filter { $0.distance2(matched) > 0.01 }.prefix(4)
        let swatches = [matched] + others
        return ShopSwatchRow(colors: Array(swatches), isChosen: { i in
            i == 0 ? model.style.matchTitle : (!model.style.matchTitle && model.style.titleColor == swatches[i])
        }, pick: { i in
            if i == 0 { model.style.matchTitle = true } else { model.style.pickTitleColor(swatches[i]) }
        }, idPrefix: "shopTitleColor") {
            VStack(alignment: .leading, spacing: 2) {
                Text(L10n.tr("Title color")).font(CS.inter(14, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                Text(L10n.tr(model.style.matchTitle ? "✨ Matches your products" : "Your pick"))
                    .font(CS.inter(11, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
            }
        }
    }

    // MARK: Letters (T2)

    private var letters: some View {
        VStack(alignment: .leading, spacing: 0) {
            ShopSheetLabel("Change letters of")
            HStack(spacing: 8) {
                chip("Big title", index: 0, width: 92)
                chip("Script line", index: 1, width: 92)
                chip("Selected text", index: 2, width: 110, enabled: model.selectedDecorHasWords)
            }
            .padding(.top, 8)
            VStack(spacing: 4) {
                ForEach(ShopLetter.allCases) { letter in letterRow(letter) }
            }
            .padding(.top, 12)
        }
    }

    private func chip(_ title: String, index: Int, width: CGFloat, enabled: Bool = true) -> some View {
        let on = target == index
        return Button { target = index } label: {
            Text(L10n.tr(title)).font(CS.inter(12, .bold))
                .foregroundStyle(on ? Color.white : Color(fixed: enabled ? 0x0B1020 : 0xB4BAC6))
                .lineLimit(1).minimumScaleFactor(0.8)
                .frame(width: width, height: 30)
                .background(Color(fixed: on ? 0x1769FF : 0xF3F5F9), in: Capsule())
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .accessibilityAddTraits(on ? .isSelected : [])
        .accessibilityIdentifier("shopLetterTarget_\(index)")
    }

    private var currentLetter: ShopLetter? {
        switch target {
        case 0: return model.bigLetter ?? model.titleLayout?.bigLetter
        case 1: return model.scriptLetter ?? model.titleLayout?.scriptLetter
        default: return model.selectedDecor.flatMap { id in model.decor.first { $0.id == id }?.letter }
        }
    }

    private func letterRow(_ letter: ShopLetter) -> some View {
        let on = currentLetter == letter
        return Button {
            switch target {
            case 0: model.bigLetter = letter
            case 1: model.scriptLetter = letter
            default: if let id = model.selectedDecor { model.setLetter(letter, for: id) }
            }
        } label: {
            HStack(spacing: 5) {
                Text(verbatim: letter.sample).font(ShopFonts.font(letter.face, size: 22, opticalSize: 22))
                    .foregroundStyle(Color(fixed: 0x2A2320)).lineLimit(1)
                if letter == .dmSerif {
                    Text(verbatim: "Uniform").font(ShopFonts.font(ShopLetter.dmSerifItalic, size: 22)).foregroundStyle(Color(fixed: 0x2A2320))
                }
                Spacer(minLength: 6)
                VStack(alignment: .trailing, spacing: 3) {
                    Text(verbatim: letter.name).font(CS.inter(10.5, .semibold)).foregroundStyle(Color(fixed: 0x6B7385)).lineLimit(1)
                    if let badge = letter.badge {
                        Text(L10n.tr(badge)).font(CS.inter(9, .heavy)).foregroundStyle(Color(fixed: 0x1FA971))
                            .padding(.horizontal, 8).frame(height: 16)
                            .background(Color(fixed: 0xE8F7EF), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                    }
                }
            }
            .padding(.horizontal, 14)
            .frame(height: 48)
            .background(Color(fixed: on ? 0xEEF3FF : 0xFFFFFF), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color(fixed: on ? 0x1769FF : 0xE6E9EF), lineWidth: on ? 1.5 : 1))
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(on ? .isSelected : [])
        .accessibilityIdentifier("shopLetter_\(letter.rawValue)")
    }

    // MARK: Add (T3)

    static let addColors: [ShopRGB] = ([0xFFFFFF, 0xEDE6DA, 0xD9C7B0, 0xB8955A, 0x6CC7D1, 0x8A4B32, 0x3F4A2A, 0x0B1020] as [UInt32]).map { ShopRGB(hex: $0) }

    private var addTab: some View {
        VStack(alignment: .leading, spacing: 0) {
            ShopSheetLabel("Add to your board · drag, pinch and turn like a product")
            LazyVGrid(columns: Array(repeating: GridItem(.fixed(106), spacing: 12), count: 3), spacing: 10) {
                ForEach(ShopDecor.Kind.allCases, id: \.self) { kind in addTile(kind) }
            }
            .padding(.top, 8)
            ShopSwatchRow(colors: Self.addColors, isChosen: { model.addColor == Self.addColors[$0] }, pick: { i in
                model.addColor = Self.addColors[i]
                if let id = model.selectedDecor { model.setColor(Self.addColors[i], for: id) }
            }, spacing: 6, idPrefix: "shopAddColor") {
                Text(L10n.tr("Color")).font(CS.inter(14, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
            }
            .padding(.top, 14)
            Text(L10n.tr("Hold anything: Front · Back · Duplicate · Delete"))
                .font(CS.inter(12, .medium)).foregroundStyle(Color(fixed: 0x1769FF))
                .multilineTextAlignment(.center).frame(maxWidth: .infinity).padding(.top, 12)
        }
    }

    private func addTile(_ kind: ShopDecor.Kind) -> some View {
        Button { model.addDecor(kind) } label: {
            ZStack(alignment: .bottom) {
                ShopAddSample(kind: kind).frame(width: 106, height: 72)
                    .frame(maxHeight: .infinity, alignment: .top)
                Text(L10n.tr(kind.name)).font(CS.inter(11, .bold)).foregroundStyle(Color(fixed: 0x6B7385))
                    .frame(width: 106, height: 20)
                    .background(Color.white.opacity(0.85))
            }
            .frame(width: 106, height: 92)
            .background(Color(fixed: kind == .frame ? 0x8B7D6E : 0xF7F4EF))
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(Color(fixed: 0xE6E9EF), lineWidth: 1))
        }
        .buttonStyle(BubblePress())
        .accessibilityIdentifier("shopAdd_\(kind.rawValue)")
    }
}

/// The 240-long track (#E6E9EF, blue part #1769FF) with the white knob of T1.
struct ShopSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let t = CGFloat((value - range.lowerBound) / (range.upperBound - range.lowerBound))
            ZStack(alignment: .leading) {
                Capsule().fill(Color(fixed: 0xE6E9EF)).frame(height: 4)
                Capsule().fill(Color(fixed: 0x1769FF)).frame(width: max(0, w * t), height: 4)
                Circle().fill(Color.white).frame(width: 20, height: 20)
                    .overlay(Circle().strokeBorder(Color(fixed: 0x1769FF), lineWidth: 2))
                    .offset(x: w * t - 10)
            }
            .frame(height: geo.size.height)
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { g in
                let p = min(max(0, g.location.x / max(1, w)), 1)
                value = range.lowerBound + Double(p) * (range.upperBound - range.lowerBound)
            })
        }
        .accessibilityElement()
        .accessibilityValue(String(format: "%.1f", value))
        .accessibilityAdjustableAction { direction in
            let step = (range.upperBound - range.lowerBound) / 10
            value = min(max(range.lowerBound, value + (direction == .increment ? step : -step)), range.upperBound)
        }
    }
}

/// The little pictures on the Layout tiles (T1): words from the example boards.
struct ShopLayoutSample: View {
    let layout: ShopTitleLayout
    private let ink = Color(fixed: 0x2A2320)

    private func t(_ s: String, _ face: ShopFontFace, _ size: CGFloat, _ color: Color? = nil, tracking: CGFloat = 0) -> some View {
        Text(verbatim: s).font(ShopFonts.font(face, size: size, opticalSize: size)).tracking(tracking)
            .foregroundStyle(color ?? ink).lineLimit(1).fixedSize()
    }

    var body: some View {
        ZStack {
            switch layout {
            case .thinSerif:
                VStack(spacing: 2) {
                    t("AMAZON", ShopLetter.bodoni.face, 23, tracking: -0.9)
                    t("FAVORITE FINDS", ShopLetter.bodoni.face, 8, tracking: 1)
                }
            case .scriptOver:
                VStack(spacing: -6) {
                    t("AMAZON", ShopLetter.dmSerif.face, 22, Color(fixed: 0x6B4A33))
                    t("fall girls night", ShopLetter.monsieur.face, 17, Color(fixed: 0x0B1020))
                }
            case .scriptInside:
                ZStack {
                    t("THE      EDIT", ShopLetter.bodoni.face, 15).offset(y: 8)
                    t("Fall", ShopLetter.mrsSaint.face, 26).offset(y: -6)
                }
            case .tall:
                VStack(spacing: 3) {
                    t("DUPES THAT", ShopLetter.instrument.face, 19)
                    t("DON'T LOOK", ShopLetter.instrument.face, 19).padding(.top, -6)
                    Rectangle().fill(Color(fixed: 0xB8955A)).frame(width: 30, height: 1.5)
                    Text(verbatim: "ALL ON AMAZON").font(CS.inter(7, .bold)).tracking(1.5).foregroundStyle(Color(fixed: 0xB8955A))
                }
            case .italicMix:
                VStack(spacing: 2) {
                    Text(verbatim: "AMAZON EDIT").font(CS.inter(7, .medium)).tracking(2.1).foregroundStyle(ink)
                    HStack(spacing: 4) {
                        t("Mom", ShopLetter.dmSerif.face, 19)
                        t("Uniform", ShopLetter.dmSerifItalic, 19)
                    }
                }
            case .boldScript:
                VStack(spacing: -2) {
                    t("AMAZON", ShopLetter.archivo.face, 18, Color(fixed: 0x3F4A2A))
                    t("Early Fall Outfit", ShopLetter.greatVibes.face, 15, Color(fixed: 0x8A4B32))
                }
            }
        }
        .frame(width: 106, height: 76)
    }
}

/// The little pictures on the Add tiles (T3).
struct ShopAddSample: View {
    let kind: ShopDecor.Kind
    private let ink = Color(fixed: 0x2A2320)

    var body: some View {
        ZStack {
            switch kind {
            case .text:
                Text(verbatim: "Aa").font(ShopFonts.font(ShopLetter.dmSerif.face, size: 30)).foregroundStyle(ink)
            case .label:
                HStack(alignment: .top, spacing: 0) {
                    Text(verbatim: "Flattering\nFit").font(ShopFonts.font(ShopLetter.caveat.face, size: 15)).foregroundStyle(ink)
                        .rotationEffect(.degrees(-6))
                    ShopArrow().stroke(ink, style: StrokeStyle(lineWidth: 1.6, lineCap: .round, lineJoin: .round))
                        .frame(width: 18, height: 38).padding(.top, 14)
                }
            case .tag:
                Text(verbatim: "Comment YES").font(ShopFonts.font(ShopLetter.playfair.face, size: 11.5)).foregroundStyle(Color(fixed: 0x0B1020))
                    .frame(width: 82, height: 26).background(Color(fixed: 0x6CC7D1))
            case .button:
                Text(verbatim: "SHOP THE SALE").font(CS.inter(8.5, .bold)).tracking(0.7).foregroundStyle(ink)
                    .frame(width: 82, height: 24).background(Color(fixed: 0xEDE6DA))
            case .frame:
                ZStack {
                    Rectangle().strokeBorder(Color.white, lineWidth: 2.5).frame(width: 70, height: 58)
                    Rectangle().strokeBorder(Color.white, lineWidth: 1.2).frame(width: 60, height: 48)
                }
            case .line:
                VStack(spacing: 8) {
                    Rectangle().fill(Color(fixed: 0xB8955A)).frame(width: 56, height: 2)
                    Text(verbatim: "NOT TO MISS").font(CS.inter(8, .medium)).tracking(2.4).foregroundStyle(ink)
                }
            case .sparkle:
                ZStack(alignment: .topLeading) {
                    ShopSparkle().fill(Color(fixed: 0xC9A24E)).frame(width: 30, height: 30)
                    ShopSparkle().fill(Color(fixed: 0xC9A24E)).frame(width: 13, height: 13).offset(x: 34, y: 28)
                }
                .frame(width: 50, height: 44)
            case .ribbon:
                Text(verbatim: "Early Black Friday Sale /").font(ShopFonts.font(ShopLetter.playfair.face, size: 12)).foregroundStyle(ink)
                    .lineLimit(1).fixedSize()
                    .frame(width: 140, height: 20).background(Color(fixed: 0xEDE6DA))
                    .rotationEffect(.degrees(10))
                    .offset(y: 8)
            case .shape:
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color(fixed: 0xD9C7B0)).frame(width: 40, height: 40).offset(x: -17, y: -2)
                    Circle().fill(Color(fixed: 0x8A9A6A)).frame(width: 36, height: 36).offset(x: 19, y: 8)
                }
            }
        }
        .frame(width: 106, height: 72)
        .clipped()
    }
}

/// Background › Cut (T4).
struct ShopCutTab: View {
    @ObservedObject var model: ShopBoardModel
    static let frameColors: [ShopRGB] = ([0xFFFFFF, 0xF1ECE4, 0xB8955A, 0x8B7D6E, 0x3F4A2A, 0x0B1020] as [UInt32]).map { ShopRGB(hex: $0) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ShopSheetLabel("Cut the background · boxes inside it")
            LazyVGrid(columns: Array(repeating: GridItem(.fixed(106), spacing: 12), count: 3), spacing: 12) {
                ForEach(ShopCut.allCases) { cut in tile(cut) }
            }
            .padding(.top, 8)
            ShopSwatchRow(colors: Self.frameColors, isChosen: { model.cutColor == Self.frameColors[$0] },
                          pick: { model.cutColor = Self.frameColors[$0] }, idPrefix: "shopCutColor") {
                Text(L10n.tr("Frame color")).font(CS.inter(14, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
            }
            .padding(.top, 14)
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(L10n.tr("Products can go past the edge")).font(CS.inter(14, .bold)).foregroundStyle(Color(fixed: 0x0B1020))
                    Text(L10n.tr("Big pieces cut by the board, like the pros")).font(CS.inter(11, .medium)).foregroundStyle(Color(fixed: 0x6B7385))
                }
                Spacer(minLength: 4)
                Toggle("", isOn: $model.pastEdge).labelsHidden().tint(Color(fixed: 0x1FA971))
                    .accessibilityIdentifier("shopPastEdge")
            }
            .padding(.horizontal, 14)
            .frame(height: 50)
            .background(Color(fixed: 0xF3F5F9), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .padding(.top, 10)
        }
    }

    private func tile(_ cut: ShopCut) -> some View {
        let on = model.cut == cut
        return Button { model.cut = cut } label: {
            VStack(spacing: 0) {
                ShopCutSample(cut: cut).frame(width: 106, height: 76)
                Text(L10n.tr(cut.name)).font(CS.inter(11, .bold)).foregroundStyle(Color(fixed: on ? 0x1769FF : 0x6B7385))
                    .frame(width: 106, height: 24)
                    .background(Color(fixed: on ? 0xEEF3FF : 0xFFFFFF))
            }
            .frame(width: 106, height: 100)
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color(fixed: on ? 0x1769FF : 0xE6E9EF), lineWidth: on ? 2 : 1))
        }
        .buttonStyle(BubblePress())
        .accessibilityAddTraits(on ? .isSelected : [])
        .accessibilityIdentifier("shopCut_\(cut.rawValue)")
    }
}

/// The little pictures on the Cut tiles (T4).
struct ShopCutSample: View {
    let cut: ShopCut
    private let cream = Color(fixed: 0xF1ECE4), taupe = Color(fixed: 0x8B7D6E)

    var body: some View {
        ZStack(alignment: .topLeading) {
            switch cut {
            case .full:
                cream
            case .photoColor:
                cream
                LinearGradient(colors: [Color(fixed: 0x9C8C7A), Color(fixed: 0x6E6053)], startPoint: .leading, endPoint: .trailing)
                    .frame(width: 40, height: 76)
                    .overlay(Text(verbatim: L10n.tr("photo")).font(CS.inter(9, .bold)).foregroundStyle(.white))
            case .band:
                cream
                Color(fixed: 0x6E5A48).frame(width: 106, height: 18).offset(y: 54)
            case .innerCard:
                taupe
                RoundedRectangle(cornerRadius: 6, style: .continuous).fill(cream).frame(width: 80, height: 56).offset(x: 13, y: 8)
            case .doubleFrame:
                taupe
                Rectangle().strokeBorder(Color.white, lineWidth: 2.5).frame(width: 86, height: 60).offset(x: 10, y: 6)
                Rectangle().strokeBorder(Color.white, lineWidth: 1).frame(width: 76, height: 50).offset(x: 15, y: 11)
            case .arch:
                Color(fixed: 0xE9E2D7)
                UnevenRoundedRectangle(topLeadingRadius: 32, topTrailingRadius: 32).fill(Color(fixed: 0xCDBFAE))
                    .frame(width: 64, height: 66).offset(x: 21, y: 10)
            }
        }
        .frame(width: 106, height: 76, alignment: .topLeading)
        .clipped()
    }
}

/// Hold a product or a thing you added (T3 "Hold anything: Front · Back · Duplicate · Delete").
struct ShopHoldMenu: View {
    let isDecor: Bool
    let hasWords: Bool
    let onFront: () -> Void
    let onBack: () -> Void
    let onDuplicate: () -> Void
    let onDelete: () -> Void
    let onEditWords: () -> Void

    var body: some View {
        HStack(spacing: 4) {
            button("Front", "square.3.layers.3d.top.filled", id: "shopHoldFront", action: onFront)
            button("Back", "square.3.layers.3d.bottom.filled", id: "shopHoldBack", action: onBack)
            if isDecor {
                if hasWords { button("Edit words", "character.cursor.ibeam", id: "shopHoldWords", action: onEditWords) }
                button("Duplicate", "plus.square.on.square", id: "shopHoldDuplicate", action: onDuplicate)
                button("Delete", "trash", id: "shopHoldDelete", destructive: true, action: onDelete)
            }
        }
        .padding(4)
        .background(Capsule().fill(Color.white).shadow(color: Color(fixed: 0x0D1433, opacity: 0.18), radius: 10, x: 0, y: 4))
        .accessibilityIdentifier("shopHoldMenu")
    }

    private func button(_ title: String, _ icon: String, id: String, destructive: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 1) {
                Image(systemName: icon).font(.system(size: 13, weight: .semibold))
                Text(L10n.tr(title)).font(CS.inter(9.5, .bold)).lineLimit(1)
            }
            .foregroundStyle(Color(fixed: destructive ? 0xE5484D : 0x0B1020))
            .frame(minWidth: 52, minHeight: 40)
            .padding(.horizontal, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(id)
    }
}

// MARK: - In the drawing

/// Title & design + Background › Cut as plain values (part of ShopBoardSnapshot; empty for the style previews).
struct ShopDesignSnapshot {
    var layout: ShopTitleLayout?
    var bigLetter: ShopLetter?
    var scriptLetter: ShopLetter?
    var titleSize: CGFloat = 1
    var titleSpacing: CGFloat = 0
    var decor: [ShopDecor] = []
    /// Same order as `decor`.
    var decorAdjusts: [ShopAdjust] = []
    var cut: ShopCut = .full
    var cutColor = ShopRGB(hex: 0xFFFFFF)
    var cutPhoto: UIImage?

    func adjust(_ index: Int) -> ShopAdjust { decorAdjusts.indices.contains(index) ? decorAdjusts[index] : ShopAdjust() }
}

extension ShopBoardSnapshot {
    /// The added thing under `point` among those drawn over (`front`) or behind the products, the top one first.
    func decor(at point: CGPoint, board size: CGSize, front: Bool) -> UUID? {
        for index in design.decor.indices.reversed() where design.decor[index].front == front {
            if design.decor[index].contains(point, board: size, adjust: design.adjust(index)) { return design.decor[index].id }
        }
        return nil
    }

    func decorFrame(_ id: UUID, board size: CGSize) -> (rect: CGRect, rotation: Double)? {
        guard let index = design.decor.firstIndex(where: { $0.id == id }) else { return nil }
        return design.decor[index].frame(board: size, adjust: design.adjust(index))
    }

    /// What a finger grabs at `point`, top first: things over the products, the title, products, things behind.
    func grab(at point: CGPoint, board size: CGSize, titleID: UUID) -> UUID? {
        if let id = decor(at: point, board: size, front: true) { return id }
        if isOnTitle(point, board: size) { return titleID }
        if let index = item(at: point, board: size) { return itemIDs[index] }
        return decor(at: point, board: size, front: false)
    }
}

/// The added things of one layer, drawn on a board `size` big.
struct ShopDecorLayer: View {
    let design: ShopDesignSnapshot
    let front: Bool
    let size: CGSize

    var body: some View {
        let sx = size.width / ShopLayout.designSize.width
        ZStack(alignment: .topLeading) {
            ForEach(Array(design.decor.enumerated()), id: \.element.id) { index, decor in
                if decor.front == front {
                    let adjust = design.adjust(index)
                    let placed = decor.frame(board: size, adjust: adjust)
                    ShopDecorView(decor: decor, sx: sx)
                        .scaleEffect(adjust.scale)
                        .rotationEffect(.degrees(placed.rotation))
                        .position(x: placed.rect.midX, y: placed.rect.midY)
                }
            }
        }
        .frame(width: size.width, height: size.height, alignment: .topLeading)
        .allowsHitTesting(false)
    }
}
