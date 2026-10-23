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
                .layoutPriority(item.count == nil ? 0 : 1)
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
