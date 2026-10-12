import SwiftUI

/// The four tabs at the bottom (Figma): Create · Edits · Plan · You. The app opens on Create.
enum MainTab: String, CaseIterable, Hashable {
    case create, edits, plan, you

    var title: String {
        switch self {
        case .create: return "Create"
        case .edits: return "Edits"
        case .plan: return "Plan"
        case .you: return "You"
        }
    }

    var image: String {
        switch self {
        case .create: return "TabCreate"
        case .edits: return "TabEdits"
        case .plan: return "TabPlan"
        case .you: return "TabYou"
        }
    }
}

struct MainTabView: View {
    @State private var tab: MainTab
    /// The tab to return to from the camera's back arrow.
    @State private var lastNonCreateTab: MainTab = .edits

    init(initialTab: MainTab = .create) {
        _tab = State(initialValue: initialTab)
    }

    var body: some View {
        Group {
            switch tab {
            case .create:
                // Full-screen camera, as in Figma (no tab bar on the camera); the back arrow
                // returns to the last tab.
                CreateFlowView(
                    onBack: { select(lastNonCreateTab) },
                    onStarted: { select(.edits) }
                )
            case .edits:
                ProjectsView(selectTab: select)
            case .plan:
                PlanView(selectTab: select)
            case .you:
                YouView(selectTab: select)
            }
        }
        .preferredColorScheme(.dark)
    }

    private func select(_ next: MainTab) {
        if next != .create { lastNonCreateTab = next }
        tab = next
    }
}

/// Bottom tab bar (Figma "Tab Bar": 358 × 88, radius 26).
struct CSTabBar: View {
    let current: MainTab
    let select: (MainTab) -> Void

    var body: some View {
        HStack(spacing: 0) {
            ForEach(MainTab.allCases, id: \.self) { item in
                let isActive = item == current
                Button {
                    select(item)
                } label: {
                    VStack(spacing: 4) {
                        Image(item.image)
                            .resizable()
                            .interpolation(.high)
                            .scaledToFit()
                            .frame(width: 44, height: 44)
                            .shadow(color: isActive ? CS.cyan.opacity(0.75) : .clear, radius: 8)
                        Text(item.title)
                            .font(CS.inter(13, isActive ? .semibold : .regular))
                            .foregroundStyle(isActive ? CS.cyan : CS.textTab)
                    }
                    .opacity(isActive ? 1 : 0.68)
                    .frame(maxWidth: .infinity)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(isActive ? .isSelected : [])
                .accessibilityIdentifier("tab_\(item.rawValue)")
            }
        }
        .padding(.horizontal, 4)
        .frame(height: 88)
        .background(Color(hex: 0x060E1D, opacity: 0.96), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 26, style: .continuous).stroke(Color(hex: 0x143866, opacity: 0.9), lineWidth: 1))
        .padding(.horizontal, 16)
        .padding(.bottom, 2)
    }
}

/// Plan tab: not designed for this phase yet. No plans or prices are shown.
struct PlanView: View {
    let selectTab: (MainTab) -> Void

    var body: some View {
        ZStack {
            ScreenBackground()
            VStack(alignment: .leading, spacing: 0) {
                BrandLogo().padding(.top, 8)
                Text("Plan")
                    .font(CS.inter(34, .semibold))
                    .foregroundStyle(.white)
                    .padding(.top, 14)
                Spacer()
                VStack(spacing: 10) {
                    Image("TabPlan").resizable().scaledToFit().frame(width: 64, height: 64)
                    Text("Coming soon")
                        .font(CS.inter(19, .semibold))
                        .foregroundStyle(.white)
                }
                .frame(maxWidth: .infinity)
                Spacer()
                Spacer()
            }
            .padding(.horizontal, 20)
        }
        .safeAreaInset(edge: .bottom) { CSTabBar(current: .plan, select: selectTab) }
    }
}
