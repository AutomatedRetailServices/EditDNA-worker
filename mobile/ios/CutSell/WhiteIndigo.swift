import SwiftUI

// White Indigo (Figma PZnh8DuFcnhMUqPScHHL5q · 12_Prototype): the shared pieces of the main screens.

extension View {
    /// Figma card: #F8F9FB, 1 pt #E6E9EF border, radius 20, soft shadow (0 2 10, #0D1433 at 8 %).
    func wiCard(radius: CGFloat = 20, fill: Color = CS.wiCard, border: Color = CS.wiBorder, shadow: Bool = true) -> some View {
        self
            .background(fill, in: RoundedRectangle(cornerRadius: radius, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: radius, style: .continuous).stroke(border, lineWidth: 1))
            .shadow(color: shadow ? Color(fixed: 0x0D1433, opacity: 0.08) : .clear, radius: 5, x: 0, y: 2)
    }
}

extension View {
    /// The strip behind the clock and battery stays white: content that scrolls up passes UNDER it, never over the
    /// status bar (PO 24 Oct: the Edits logo and title rode over the time and battery).
    func wiStatusBarCover() -> some View {
        overlay(alignment: .top) {
            CS.wiPage.frame(height: 0).ignoresSafeArea(edges: .top).allowsHitTesting(false)
        }
    }
}

/// Header of the main screens: the horizontal logo (172 pt with its glow) left, round buttons right.
struct WIHeader<Trailing: View>: View {
    @ViewBuilder var trailing: () -> Trailing

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            BrandLogo(width: 172).padding(.leading, -11.5)
            Spacer()
            trailing()
        }
        .frame(height: 67)
        .padding(.top, 2)
    }
}

/// 40 pt round header button (#F3F5F9, 1 pt border) with a black 20 pt line icon.
struct WIHeaderIcon: View {
    let image: String
    var body: some View {
        Image(image).renderingMode(.template).resizable().frame(width: 20, height: 20)
            .foregroundStyle(CS.wiInk)
            .frame(width: 40, height: 40)
            .background(CS.wiSurface2, in: Circle())
            .overlay(Circle().stroke(CS.wiBorder, lineWidth: 1))
            .contentShape(Circle())
    }
}

/// Screen title (ExtraBold 32, -2 %) and its grey line (Medium 15).
struct WIScreenTitle: View {
    let title: String
    var subtitle: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(L10n.tr(title))
                .font(CS.inter(32, .heavy))
                .tracking(CS.screenTitleTracking)
                .foregroundStyle(CS.wiInk)
                .fixedSize(horizontal: false, vertical: true)
            if let subtitle {
                Text(L10n.tr(subtitle))
                    .font(CS.inter(15, .medium))
                    .foregroundStyle(CS.wiSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Full-width segmented bar (Figma 1173:2 "Tabs/Segmented"): #F3F5F9, radius 20, 3 pt inset; the active segment is
/// #1769FF with white text; others ink text with a blue count. Every segment fits (the text shrinks before it is cut).
struct WISegmented<ID: Hashable>: View {
    struct Item: Identifiable {
        let id: ID
        let title: String
        var count: Int?
        var accessibilityID: String = ""
    }

    let items: [Item]
    @Binding var selection: ID

    var body: some View {
        HStack(spacing: 2) {
            ForEach(items) { item in
                let active = item.id == selection
                Button { selection = item.id } label: {
                    HStack(spacing: 4) {
                        Text(L10n.tr(item.title)).font(CS.inter(13, .semibold))
                            .foregroundStyle(active ? Color(fixed: 0xFFFFFF) : CS.wiInk)
                        if let count = item.count {
                            Text(verbatim: "\(count)").font(CS.inter(11, .bold))
                                .foregroundStyle(active ? Color(fixed: 0xFFFFFF) : CS.accent)
                        }
                    }
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .padding(.horizontal, 4)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(active ? CS.accent : .clear, in: RoundedRectangle(cornerRadius: 17, style: .continuous))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(active ? .isSelected : [])
                .accessibilityIdentifier(item.accessibilityID)
            }
        }
        .padding(3)
        .frame(height: 40)
        .frame(maxWidth: .infinity)
        .background(CS.wiSurface2, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
    }
}

/// Section title row: Bold 17 left, small grey note right.
struct WISectionTitle: View {
    let title: String
    var note: String?

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(L10n.tr(title)).font(CS.sectionTitle).foregroundStyle(CS.wiInk)
            Spacer(minLength: 8)
            if let note {
                Text(L10n.tr(note)).font(CS.inter(12, .medium)).foregroundStyle(CS.wiSecondary)
                    .lineLimit(1).minimumScaleFactor(0.8)
            }
        }
    }
}

/// Figma 1177:157 "Note": white 36 pt icon tile with the bell, Bold 14 title, grey 12 text.
struct WINote: View {
    let title: String?
    let text: String
    var icon = "WINoteIcon"

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(icon).renderingMode(.template).resizable().frame(width: 20, height: 20)
                .foregroundStyle(CS.wiInk)
                .frame(width: 36, height: 36)
                .background(CS.wiPage, in: RoundedRectangle(cornerRadius: 11, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 11, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
            VStack(alignment: .leading, spacing: 3) {
                if let title {
                    Text(L10n.tr(title)).font(CS.inter(14, .bold)).foregroundStyle(CS.wiInk)
                    Text(L10n.tr(text)).font(CS.inter(12, .medium)).foregroundStyle(CS.wiSecondary)
                        .lineSpacing(2)
                } else {
                    Text(L10n.tr(text)).font(CS.inter(13, .semibold)).foregroundStyle(CS.wiInk)
                }
            }
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .frame(minHeight: 36)
        }
        .padding(.horizontal, 13)
        .padding(.vertical, 13)
        .wiCard(shadow: false)
    }
}

/// Top bar of a pushed screen (Figma 1179:3): 40 pt back button, Bold 17 title centred, optional blue text right.
struct WIBackBar: View {
    var title: String? = nil
    let onBack: () -> Void
    var trailingTitle: String? = nil
    var onTrailing: () -> Void = {}

    var body: some View {
        ZStack {
            if let title {
                Text(L10n.tr(title)).font(CS.inter(17, .bold)).foregroundStyle(CS.wiInk).lineLimit(1)
                    .padding(.horizontal, 60)
            }
            HStack {
                Button(action: onBack) { WIHeaderIcon(image: "WIBack") }
                    .buttonStyle(.plain)
                    .accessibilityLabel(L10n.tr("Back"))
                Spacer()
                if let trailingTitle {
                    Button(L10n.tr(trailingTitle), action: onTrailing)
                        .font(CS.inter(15, .semibold))
                        .foregroundStyle(CS.accent)
                }
            }
        }
        .frame(height: 56)
        .padding(.top, 2)
    }
}

/// Error / warning mark in a coloured circle (Figma 1176:2867 ErrorBadge, 1176:2993 WarnIcon).
struct WIStatusBadge: View {
    enum Kind { case error, warning, info }
    var kind: Kind = .error
    var size: CGFloat = 64

    var body: some View {
        Group {
            switch kind {
            case .error:
                Image("WIErrorMark").renderingMode(.template).resizable().frame(width: size * 34 / 64, height: size * 34 / 64)
                    .foregroundStyle(Color(fixed: 0xFFFFFF))
                    .frame(width: size, height: size)
                    .background(CS.wiError, in: Circle())
            case .warning:
                Image("WIWarn").renderingMode(.template).resizable().frame(width: size / 2, height: size / 2)
                    .foregroundStyle(CS.wiWarning)
                    .frame(width: size, height: size)
                    .background(CS.wiWarningBg, in: Circle())
            case .info:
                Image("WIInfo").renderingMode(.template).resizable().frame(width: size / 2, height: size / 2)
                    .foregroundStyle(CS.accent)
                    .frame(width: size, height: size)
                    .background(CS.accentSoft, in: Circle())
            }
        }
        .accessibilityHidden(true)
    }
}

/// Figma 1176:2874 "Safe": green card "Your original video is safe · Nothing was lost."
struct WIOriginalSafeCard: View {
    var body: some View {
        HStack(spacing: 12) {
            Image("WIShield").renderingMode(.template).resizable().frame(width: 20, height: 20)
                .foregroundStyle(CS.wiSuccess)
                .frame(width: 36, height: 36)
                .background(CS.wiPage, in: RoundedRectangle(cornerRadius: 11, style: .continuous))
            VStack(alignment: .leading, spacing: 2) {
                Text(L10n.tr("Your original video is safe")).font(CS.inter(14, .bold)).foregroundStyle(CS.wiInk)
                Text(L10n.tr("Nothing was lost.")).font(CS.inter(12, .medium)).foregroundStyle(CS.wiSecondary)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 13)
        .frame(height: 72)
        .wiCard(fill: CS.wiSuccessBg, border: Color(fixed: 0xBFEBD5), shadow: false)
        .accessibilityIdentifier("originalSafe")
    }
}

/// 52 pt pill button: blue for the main action, black for neutral ones, outlined for the second choice.
struct WIPillButton: View {
    enum Style { case primary, neutral, outline }
    let title: String
    var style: Style = .primary
    var icon: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if let icon { Image(icon).renderingMode(.template).resizable().frame(width: 20, height: 20) }
                Text(L10n.tr(title)).font(CS.inter(16, .bold)).lineLimit(1).minimumScaleFactor(0.8)
            }
            .foregroundStyle(style == .outline ? CS.wiInk : Color(fixed: 0xFFFFFF))
            .frame(maxWidth: .infinity)
            .frame(height: 52)
            .background(style == .primary ? CS.accent : style == .neutral ? CS.neutralButton : CS.wiPage, in: Capsule())
            .overlay(Capsule().stroke(style == .outline ? CS.wiInk : .clear, lineWidth: 1.5))
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

/// Bottom sheet of the White Indigo look (Figma 1176:2991): white, radius 28 on top, grey handle, 45 % dim behind.
struct WISheetCard<Content: View>: View {
    @ViewBuilder var content: () -> Content

    var body: some View {
        VStack(spacing: 0) {
            Capsule().fill(Color(fixed: 0xD5D9E2)).frame(width: 40, height: 5).padding(.top, 10)
            content()
        }
        .frame(maxWidth: .infinity)
        .background {
            // Reaches the bottom edge of the screen (under the home indicator).
            UnevenRoundedRectangle(topLeadingRadius: 28, topTrailingRadius: 28, style: .continuous)
                .fill(CS.wiPage)
                .ignoresSafeArea(edges: .bottom)
        }
    }
}
