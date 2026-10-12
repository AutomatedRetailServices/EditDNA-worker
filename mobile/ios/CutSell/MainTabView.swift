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
    @EnvironmentObject private var appState: AppState
    @ObservedObject private var tracker = CutTracker.shared
    /// A ready cut opened from the "Your cut is ready" banner.
    @State private var openReady: ReadyRoute?

    struct ReadyRoute: Identifiable { let id: String }

    init(initialTab: MainTab = .create) {
        _tab = State(initialValue: initialTab)
    }

    var body: some View {
        Group {
            switch tab {
            case .create:
                // Home Hub (Figma 1068:22); camera and gallery open full screen from it.
                CreateFlowView(selectTab: select)
            case .edits:
                ProjectsView(selectTab: select)
            case .plan:
                PlanView(selectTab: select)
            case .you:
                YouView(selectTab: select)
            }
        }
        .overlay(alignment: .top) {
            if let projectID = tracker.readyBanner {
                ReadyBanner {
                    tracker.readyBanner = nil
                    openReady = ReadyRoute(id: projectID)
                } onDismiss: {
                    tracker.readyBanner = nil
                }
                .padding(.horizontal, 16)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.easeOut(duration: 0.25), value: tracker.readyBanner)
        .fullScreenCover(item: $openReady) { route in
            NavigationStack {
                CutProgressView(
                    projectID: route.id, fileName: "", thumbnail: nil,
                    onClose: { openReady = nil },
                    onOpenEdits: { openReady = nil; select(.edits) }
                )
            }
            .environmentObject(appState)
        }
        .task { AppStateBridge.refreshProjects = { try await appState.refreshProjects() } }
        .preferredColorScheme(.dark)
    }

    private func select(_ next: MainTab) {
        tab = next
    }
}

/// "Your cut is ready" banner shown in any tab when a cut finishes in the background.
struct ReadyBanner: View {
    let onOpen: () -> Void
    let onDismiss: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image("ProcessingLogo").resizable().frame(width: 34, height: 34).blendMode(.screen)
            VStack(alignment: .leading, spacing: 2) {
                Text("Your cut is ready").font(CS.inter(15, .semibold)).foregroundStyle(.white)
                Text("Tap to review it").font(CS.inter(12)).foregroundStyle(CS.textSecondaryF)
            }
            Spacer()
            Button(action: onDismiss) {
                Image(systemName: "xmark").font(.system(size: 13, weight: .bold)).foregroundStyle(CS.textMutedF)
                    .frame(width: 32, height: 32)
            }
            .accessibilityLabel("Dismiss")
        }
        .padding(12)
        .background(CS.surfaceCard, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous).stroke(CS.cyan.opacity(0.5), lineWidth: 1))
        .shadow(color: CS.cyan.opacity(0.25), radius: 12)
        .contentShape(Rectangle())
        .onTapGesture(perform: onOpen)
        .accessibilityAddTraits(.isButton)
        .accessibilityIdentifier("readyBanner")
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
