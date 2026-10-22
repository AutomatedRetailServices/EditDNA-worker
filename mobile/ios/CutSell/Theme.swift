import CoreText
import SwiftUI

/// CutSell design system, taken from the Figma file (03 Create / 05 Edits).
/// Every screen uses these colors, fonts and components instead of its own copies.
enum CS {
    // MARK: Colors (Figma hex values)

    static let background = Color(hex: 0x070910)
    static let backgroundTop = Color(hex: 0x04060E)
    static let backgroundMid = Color(hex: 0x090B1A)
    static let backgroundBottom = Color(hex: 0x020308)
    static let panel = Color(hex: 0x03050B, opacity: 0.98)
    static let sheet = Color(hex: 0x060B18)
    static let card = Color(hex: 0x040812)
    static let cardBorder = Color(hex: 0x14336B)
    static let fieldBackground = Color(hex: 0x0A1326)
    static let fieldBorder = Color(hex: 0x1A335E, opacity: 0.75)
    static let divider = Color(hex: 0x1A386B, opacity: 0.65)
    static let tileBorder = Color(hex: 0x14478C, opacity: 0.9)
    static let chipInactive = Color(hex: 0x03060F)
    static let chipBorder = Color(hex: 0x1F4794)
    static let circleButtonBorder = Color(hex: 0x144080)

    static let textPrimary = Color.ink
    static let textSecondary = Color(hex: 0xA3B8E0)
    static let textMuted = Color(hex: 0x8FA9D6)
    static let textCaption = Color(hex: 0x9EB2E0)
    static let textTab = Color(hex: 0xB0C4E8)

    // Foundations (Figma 01_Foundations 50:2 / FICHA_01 §1)
    static let base = Color(hex: 0x050814)
    static let surfaceCard = Color(hex: 0x0C1730)
    static let surfaceControl = Color(hex: 0x0A1326)
    static let borderDefault = Color(hex: 0x19345F)
    static let textPrimaryF = Color(hex: 0xF7FAFF)
    static let textSecondaryF = Color(hex: 0xAFC4E8)
    static let textMutedF = Color(hex: 0x6F86AE)
    static let magenta = Color(hex: 0xFF21D7)
    static let success = Color(hex: 0x25E6A2)
    static let error = Color(hex: 0xFF4D67)

    /// Cyan text, links and highlights: #1769FF in day mode (never cyan on white).
    static let cyan = Color(dark: 0x19E6FF, light: 0x1769FF)
    static let selection = Color(dark: 0x0DD1FF, light: 0x1769FF)
    static let link = cyan
    static let selectedCardBorder = Color(hex: 0x1AA6FF)
    static let checkbox = Color(hex: 0x1677FF)
    static let ready = Color(hex: 0x0DFF9E)
    static let draft = Color(hex: 0x5C7AA8)
    static let alert = Color(hex: 0xFF2E5E)

    // MARK: Gradients

    /// Main call-to-action ("Create 1 cut", "Allow camera"...).
    static let ctaGradient = LinearGradient(
        colors: [Color(hex: 0x0DC7FF), Color(hex: 0x1459FF), Color(hex: 0xBF1AFF)],
        startPoint: .leading, endPoint: .trailing
    )
    /// Smaller primary buttons on alert screens (cyan → pink).
    static let alertButtonGradient = LinearGradient(
        colors: [Color(hex: 0x19E6FF), Color(hex: 0x1769FF), Color(hex: 0xF01ADB)],
        startPoint: .leading, endPoint: .trailing
    )
    /// Selected segment / filter chip.
    static let selectedGradient = LinearGradient(
        colors: [Color(hex: 0x0DCCFF), Color(hex: 0x1459FF)],
        startPoint: .leading, endPoint: .trailing
    )
    /// Selected recording length chip and record ring.
    static let brandGradient = LinearGradient(
        colors: [Color(hex: 0x7C3AED), Color(hex: 0x1769FF), Color(hex: 0x19E6FF)],
        startPoint: .leading, endPoint: .trailing
    )
    /// Brand gradient: Cyan → ElectricBlue → Violet → Magenta.
    static let brandFull = LinearGradient(
        stops: [
            .init(color: Color(hex: 0x19E6FF), location: 0),
            .init(color: Color(hex: 0x1769FF), location: 0.38),
            .init(color: Color(hex: 0x7C3AED), location: 0.72),
            .init(color: Color(hex: 0xFF21D7), location: 1),
        ],
        startPoint: .leading, endPoint: .trailing
    )
    /// The intro / guide gradient (Figma: #19E5FF → #1769FF 35 % → #7D3BED 70 % → #FF21D6), same in both looks.
    static let brandStroke = LinearGradient(
        stops: [
            .init(color: Color(fixed: 0x19E5FF), location: 0),
            .init(color: Color(fixed: 0x1769FF), location: 0.35),
            .init(color: Color(fixed: 0x7D3BED), location: 0.70),
            .init(color: Color(fixed: 0xFF21D6), location: 1),
        ],
        startPoint: .leading, endPoint: .trailing
    )
    static let screenBackground = LinearGradient(
        stops: [
            .init(color: backgroundTop, location: 0),
            .init(color: backgroundMid, location: 0.58),
            .init(color: backgroundBottom, location: 1),
        ],
        startPoint: .top, endPoint: .bottom
    )

    // MARK: Fonts

    /// Inter, the typeface used across the Figma screens (falls back to the system font).
    static func inter(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        AppFontLoader.registerIfNeeded()
        let name: String
        switch weight {
        case .bold, .heavy, .black: name = "Inter-Bold"
        case .semibold: name = "Inter-SemiBold"
        case .medium: name = "Inter-Medium"
        default: name = "Inter-Regular"
        }
        return .custom(name, fixedSize: size)
    }

    /// SF Pro, used by the camera screen in Figma.
    static func sf(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight)
    }
}

extension Color {
    /// 0xRRGGBB, as written in Figma (the dark CutSell look). In Light appearance the same color is drawn from
    /// `LightPalette` (surfaces turn light, text turns dark, brand colors stay); screens forced dark (editor,
    /// camera, processing) always get the Figma value.
    init(hex: UInt32, opacity: Double = 1) {
        self.init(uiColor: LightPalette.dynamic(hex, opacity))
    }
}

/// Day mode (Perfil → Preferences → Appearance → Light). There is no light Figma: every dark Figma color has a
/// light counterpart by role, decided from its lightness and saturation, with hand-set values for the main
/// tokens.
enum LightPalette {
    nonisolated(unsafe) private static var cache: [UInt64: UIColor] = [:]
    private static let lock = NSLock()

    /// Day-mode rules (Figma 1146:2 / 1146:91 / 1146:178): page #F5F7FB, white cards, text #0B1020, secondary
    /// #5B6275, borders #E2E6EF. Brand colors and the gradient stay as they are; cyan used as text or a link is
    /// `CS.cyan` / `CS.link`, which turn #1769FF in day mode (never cyan on white).
    static let page: UInt32 = 0xF5F7FB, card: UInt32 = 0xFFFFFF, ink: UInt32 = 0x0B1020,
               secondary: UInt32 = 0x5B6275, muted: UInt32 = 0x8A91A3, border: UInt32 = 0xE2E6EF, accent: UInt32 = 0x1769FF

    /// Hand-set light values for the foundation tokens (dark hex → light hex).
    private static let tokens: [UInt32: UInt32] = [
        // backgrounds
        0x070910: page, 0x04060E: page, 0x090B1A: page, 0x020308: page, 0x050814: page, 0x03050D: page, 0x050B1C: page,
        0x030409: page,
        // sheets, cards and controls
        0x03050B: card, 0x060B18: card, 0x03060F: card, 0x060E1D: card, 0x040812: card, 0x0C1730: card, 0x0F1726: card,
        0x0D121F: card, 0x050A17: card,
        0x0A1326: card, 0x10182A: 0xF0F3F8, 0x1F2638: 0xEEF1F7, 0x1A2236: 0xEEF1F7, 0x121A2B: 0xF0F3F8,
        // borders
        0x14336B: border, 0x1A335E: border, 0x1A386B: border, 0x14478C: border, 0x1F4794: border, 0x144080: border,
        0x19345F: border, 0x143866: border, 0x26334D: border, 0x33405C: border,
        // text
        0xF7FAFF: ink, 0xF0F7FF: ink, 0xFFFFFF: ink,
        0xA3B8E0: secondary, 0xAFC4E8: secondary, 0x99A3BD: secondary, 0x9EB2E0: secondary, 0xB0C4E8: secondary,
        0x8FA9D6: muted, 0x6F86AE: muted, 0x5C7AA8: muted,
    ]

    static func dynamic(_ hex: UInt32, _ opacity: Double) -> UIColor {
        let key = UInt64(hex) << 16 | UInt64(min(max(opacity, 0), 1) * 1000)
        lock.lock(); defer { lock.unlock() }
        if let hit = cache[key] { return hit }
        let dark = uiColor(hex, opacity)
        let light = uiColor(tokens[hex] ?? mapped(hex), opacity)
        let color = UIColor { $0.userInterfaceStyle == .light ? light : dark }
        cache[key] = color
        return color
    }

    private static func uiColor(_ hex: UInt32, _ opacity: Double) -> UIColor {
        UIColor(red: CGFloat((hex >> 16) & 0xFF) / 255, green: CGFloat((hex >> 8) & 0xFF) / 255,
                blue: CGFloat(hex & 0xFF) / 255, alpha: CGFloat(opacity))
    }

    /// By role: very dark → page/card, dark-saturated (borders, tints) → light border, light/greyish (text) →
    /// dark ink, vivid colors (brand, states) → kept.
    static func mapped(_ hex: UInt32) -> UInt32 {
        let r = Double((hex >> 16) & 0xFF) / 255, g = Double((hex >> 8) & 0xFF) / 255, b = Double(hex & 0xFF) / 255
        let maxC = max(r, g, b), minC = min(r, g, b)
        let l = (maxC + minC) / 2
        let d = maxC - minC
        let s = d == 0 ? 0 : d / (1 - abs(2 * l - 1))
        var h = 0.0
        if d > 0 {
            if maxC == r { h = ((g - b) / d).truncatingRemainder(dividingBy: 6) }
            else if maxC == g { h = (b - r) / d + 2 } else { h = (r - g) / d + 4 }
            h /= 6; if h < 0 { h += 1 }
        }
        var nl = l, ns = s
        if l < 0.13 {                       // page, cards, panels
            nl = 0.955 + l * 0.35; ns = min(s, 0.45)
        } else if l < 0.42 && s > 0.15 {    // borders, dark tints
            nl = 0.86 - (l - 0.13) * 0.2; ns = min(s, 0.5)
        } else if l < 0.42 {                // dark greys
            nl = 1.0 - l * 0.6
        } else if s < 0.55 || l > 0.8 {     // text and light greys → ink
            nl = max(0.07, 0.07 + (1 - l) * 1.3); ns = min(s, 0.45)
        }                                   // vivid brand / state colors: unchanged (the gradient stays as it is)
        return hslToHex(h, ns, min(max(nl, 0), 1))
    }

    private static func hslToHex(_ h: Double, _ s: Double, _ l: Double) -> UInt32 {
        let c = (1 - abs(2 * l - 1)) * s
        let x = c * (1 - abs((h * 6).truncatingRemainder(dividingBy: 2) - 1))
        let m = l - c / 2
        let (r1, g1, b1): (Double, Double, Double)
        switch Int(h * 6) {
        case 0: (r1, g1, b1) = (c, x, 0)
        case 1: (r1, g1, b1) = (x, c, 0)
        case 2: (r1, g1, b1) = (0, c, x)
        case 3: (r1, g1, b1) = (0, x, c)
        case 4: (r1, g1, b1) = (x, 0, c)
        default: (r1, g1, b1) = (c, 0, x)
        }
        func byte(_ v: Double) -> UInt32 { UInt32(max(0, min(255, ((v + m) * 255).rounded()))) }
        return byte(r1) << 16 | byte(g1) << 8 | byte(b1)
    }
}

extension Color {
    /// White in the dark look, dark ink (#0B1020) in day mode: text and icons drawn on the page or a card.
    /// (White on a gradient, photo or video stays `.white`.)
    static let ink = Color(hex: 0xFFFFFF)

    /// One color in the dark look, another in day mode.
    init(dark: UInt32, light: UInt32, opacity: Double = 1) {
        let d = UIColor(red: CGFloat((dark >> 16) & 0xFF) / 255, green: CGFloat((dark >> 8) & 0xFF) / 255,
                        blue: CGFloat(dark & 0xFF) / 255, alpha: CGFloat(opacity))
        let l = UIColor(red: CGFloat((light >> 16) & 0xFF) / 255, green: CGFloat((light >> 8) & 0xFF) / 255,
                        blue: CGFloat(light & 0xFF) / 255, alpha: CGFloat(opacity))
        self.init(uiColor: UIColor { $0.userInterfaceStyle == .light ? l : d })
    }

    /// Exactly this color in both looks (brand gradients, video tiles).
    init(fixed hex: UInt32, opacity: Double = 1) {
        self.init(.sRGB, red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255, opacity: opacity)
    }
}

extension View {
    /// Day mode: white cards get the soft Figma shadow (0 2 10, #0D1433 at 8 %). Nothing in the dark look.
    func dayCardShadow() -> some View { modifier(DayCardShadow()) }
}

private struct DayCardShadow: ViewModifier {
    @Environment(\.colorScheme) private var scheme
    func body(content: Content) -> some View {
        content.shadow(color: scheme == .light ? Color(fixed: 0x0D1433, opacity: 0.08) : .clear, radius: 5, x: 0, y: 2)
    }
}

/// Registers the app's own Inter weights (the bold weight already ships with the caption fonts).
enum AppFontLoader {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var didRegister = false

    static func registerIfNeeded() {
        lock.lock()
        defer { lock.unlock() }
        guard !didRegister else { return }
        didRegister = true
        for name in ["Inter-Regular", "Inter-Medium", "Inter-SemiBold", "Inter-Bold"] {
            guard let url = Bundle.main.url(forResource: name, withExtension: "otf") else { continue }
            CTFontManagerRegisterFontsForURL(url as CFURL, .process, nil)
        }
    }
}

// MARK: - Components

/// The official CutSell.ai logo (Figma "Brand/Logo/Official", 132 × 44).
struct BrandLogo: View {
    var width: CGFloat = 132
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        Group {
            if scheme == .light {
                // Figma 1147:4422-4424: the bag icon + "CutSell" dark + ".ai" with the gradient.
                HStack(spacing: 2 * width / 132) {
                    Image("BrandLogo").resizable().interpolation(.high)
                        .frame(width: width, height: width / 3)
                        .frame(width: 34 * width / 132, height: width / 3, alignment: .leading)
                        .clipped()
                    Text(verbatim: "CutSell")
                        .font(CS.inter(21 * width / 132, .bold))
                        .tracking(-0.63 * width / 132)
                        .foregroundStyle(Color(fixed: 0x0B1020))
                    + Text(verbatim: ".ai").font(CS.inter(21 * width / 132, .bold))
                        .foregroundStyle(CS.brandStroke)
                }
                .frame(width: width, height: width / 3, alignment: .leading)
            } else {
                Image("BrandLogo")
                    .resizable()
                    .interpolation(.high)
                    .scaledToFit()
                    .frame(width: width, height: width / 3)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(L10n.tr("CutSell.ai"))
    }
}

/// Full-width gradient button (Figma "Create 1 cut", 56 high, radius 20).
struct PrimaryButton: View {
    let title: String
    var height: CGFloat = 56
    var cornerRadius: CGFloat = 20
    var fontSize: CGFloat = 18
    var gradient: LinearGradient = CS.ctaGradient
    var textColor: Color = .white
    var isEnabled: Bool = true
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(L10n.tr(title))
                .font(CS.inter(fontSize, .semibold))
                .foregroundStyle(textColor)
                .frame(maxWidth: .infinity)
                .frame(height: height)
                .background(gradient, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
                .shadow(color: Color(hex: 0x1459FF, opacity: 0.35), radius: 12)
                .opacity(isEnabled ? 1 : 0.45)
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
    }
}

/// Dark outlined secondary button (Figma "Keep Current Selection", "Remove a clip").
struct SecondaryButton: View {
    let title: String
    var height: CGFloat = 46
    var cornerRadius: CGFloat = 16
    var fontSize: CGFloat = 14
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(L10n.tr(title))
                .font(CS.inter(fontSize, .semibold))
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity)
                .frame(height: height)
                .background(Color(hex: 0x050A17), in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .stroke(Color(hex: 0x24529E), lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
    }
}

/// Pill segmented control (Figma gallery "Single · Bulk · Combine").
struct SegmentedPill<Option: Hashable>: View {
    let options: [Option]
    @Binding var selection: Option
    let title: (Option) -> String

    var body: some View {
        HStack(spacing: 0) {
            ForEach(options, id: \.self) { option in
                let isSelected = option == selection
                Button {
                    withAnimation(.easeOut(duration: 0.15)) { selection = option }
                } label: {
                    Text(L10n.tr(title(option)))
                        .font(CS.inter(15, isSelected ? .semibold : .medium))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 36)
                        .background {
                            if isSelected {
                                RoundedRectangle(cornerRadius: 10, style: .continuous).fill(CS.selectedGradient)
                            }
                        }
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(isSelected ? .isSelected : [])
            }
        }
        .padding(2)
        .frame(height: 40)
        .background(CS.fieldBackground, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(CS.fieldBorder, lineWidth: 1))
    }
}

/// Choice card with a title and a small subtitle (Figma style sheet "Normal / Natural pauses").
struct OptionCard: View {
    let title: String
    let subtitle: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Text(L10n.tr(title))
                    .font(CS.inter(15, .semibold))
                    .foregroundStyle(.white)
                Text(L10n.tr(subtitle))
                    .font(CS.inter(10))
                    .foregroundStyle(CS.textMuted)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 56)
            .background(CS.card, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(isSelected ? CS.selectedCardBorder : CS.cardBorder, lineWidth: isSelected ? 2 : 1)
            )
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

/// Filter chip (Figma Edits "All (50) · Ready (28) · Drafts (12)").
struct FilterChip: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(L10n.tr(title))
                .font(CS.inter(13, .semibold))
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .frame(maxWidth: .infinity)
                .frame(height: 40)
                .background {
                    if isSelected {
                        RoundedRectangle(cornerRadius: 19, style: .continuous).fill(CS.selectedGradient)
                            .shadow(color: Color(hex: 0x0DCCFF, opacity: 0.35), radius: 8)
                    } else {
                        RoundedRectangle(cornerRadius: 19, style: .continuous).fill(CS.chipInactive)
                    }
                }
                .overlay {
                    if !isSelected {
                        RoundedRectangle(cornerRadius: 19, style: .continuous).stroke(CS.chipBorder, lineWidth: 1)
                    }
                }
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

/// Round outlined header button (Figma Edits search / more).
struct CircleIconButton<Icon: View>: View {
    var size: CGFloat = 34
    let action: () -> Void
    @ViewBuilder let icon: () -> Icon

    var body: some View {
        Button(action: action) {
            icon()
                .frame(width: size, height: size)
                // Day mode (Figma 1146:91): dark round buttons with white icons.
                .background(Color(dark: 0x050A17, light: 0x0B1020), in: Circle())
                .overlay(Circle().stroke(Color(dark: 0x144080, light: 0x0B1020), lineWidth: 1))
        }
        .buttonStyle(.plain)
    }
}

/// Checkbox row (Figma "Use this style for my next videos").
struct CheckboxRow: View {
    let title: String
    @Binding var isOn: Bool

    var body: some View {
        Button { isOn.toggle() } label: {
            HStack(spacing: 10) {
                ZStack {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(isOn ? CS.checkbox : Color.clear)
                        .overlay(
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .stroke(isOn ? CS.checkbox : CS.cardBorder, lineWidth: 1.5)
                        )
                        .frame(width: 20, height: 20)
                    if isOn {
                        Image("IconCheck").resizable().frame(width: 14, height: 14)
                    }
                }
                Text(L10n.tr(title))
                    .font(CS.inter(13))
                    .foregroundStyle(.white)
                Spacer(minLength: 0)
            }
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isOn ? .isSelected : [])
    }
}

/// Edge-to-edge bottom sheet drawn by the app (Figma "Choose your style"), so it looks the
/// same on every iOS version (the system sheet floats with side margins on newer iOS).
struct BottomSheetOverlay<Content: View>: View {
    @Binding var isPresented: Bool
    @ViewBuilder let content: () -> Content
    @State private var drag: CGFloat = 0

    var body: some View {
        ZStack(alignment: .bottom) {
            if isPresented {
                Color.black.opacity(0.55)
                    .ignoresSafeArea()
                    .onTapGesture { isPresented = false }
                    .transition(.opacity)
                VStack(spacing: 0) {
                    Capsule()
                        .fill(Color(hex: 0x3A5280))
                        .frame(width: 40, height: 4)
                        .frame(maxWidth: .infinity)
                        .padding(.top, 10)
                        .padding(.bottom, 14)
                        .contentShape(Rectangle())
                        .gesture(
                            DragGesture()
                                .onChanged { drag = max(0, $0.translation.height) }
                                .onEnded { value in
                                    if value.translation.height > 80 { isPresented = false }
                                    drag = 0
                                }
                        )
                    content()
                }
                .background(alignment: .top) {
                    UnevenRoundedRectangle(topLeadingRadius: 24, topTrailingRadius: 24, style: .continuous)
                        .fill(CS.sheet)
                        .overlay(
                            UnevenRoundedRectangle(topLeadingRadius: 24, topTrailingRadius: 24, style: .continuous)
                                .stroke(CS.cardBorder, lineWidth: 1)
                        )
                        .ignoresSafeArea(edges: .bottom)
                }
                .offset(y: drag)
                .transition(.move(edge: .bottom))
            }
        }
        .animation(.easeOut(duration: 0.25), value: isPresented)
    }
}

/// Dark full-screen background used by every redesigned screen.
struct ScreenBackground: View {
    var body: some View {
        CS.screenBackground.ignoresSafeArea()
    }
}

/// Follows Perfil → Preferences → Appearance (Automatic / Dark / Light).
struct AppColorSchemeModifier: ViewModifier {
    @ObservedObject private var prefs = AppPrefs.shared
    func body(content: Content) -> some View { content.preferredColorScheme(prefs.colorScheme) }
}

extension View {
    func appColorScheme() -> some View { modifier(AppColorSchemeModifier()) }
}
