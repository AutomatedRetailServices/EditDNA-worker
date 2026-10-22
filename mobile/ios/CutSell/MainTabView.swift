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
    @ObservedObject private var exports = ExportTracker.shared
    @ObservedObject private var queue = ProcessingQueue.shared
    @ObservedObject private var planWeek = PlanWeekStore.shared
    @ObservedObject private var intro = IntroState.shared
    @ObservedObject private var router = AppRouter.shared
    /// A ready cut opened from the "Your cut is ready" banner.
    @State private var openReady: ReadyRoute?

    struct ReadyRoute: Identifiable { let id: String }

    init(initialTab: MainTab = .create) {
        _tab = State(initialValue: initialTab)
    }

    var body: some View {
        ZStack {
            // Build 24: Create stays alive in every tab, so an upload left with "Continue in background"
            // (or "Go to Edits") keeps going while the creator is elsewhere in CutSell.
            CreateFlowView(selectTab: select)
                .opacity(tab == .create ? 1 : 0)
                .allowsHitTesting(tab == .create)
                .accessibilityHidden(tab != .create)
            switch tab {
            case .create:
                EmptyView()
            case .edits:
                ProjectsView(selectTab: select)
            case .plan:
                PlanView(selectTab: select)
            case .you:
                YouView(selectTab: select)
            }
        }
        .overlay(alignment: .bottom) {
            // "<name> is ready · View" (Figma 1134:3027), above the tab bar, in every tab.
            if let toast = queue.readyToast {
                ReadyToast(name: toast.name) {
                    queue.readyToast = nil
                    openReady = ReadyRoute(id: toast.projectID)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 110)
                .transition(.move(edge: .bottom).combined(with: .opacity))
                .task(id: toast.projectID) {
                    try? await Task.sleep(nanoseconds: 6_000_000_000)
                    if queue.readyToast?.projectID == toast.projectID { queue.readyToast = nil }
                }
            }
        }
        .animation(.easeOut(duration: 0.25), value: queue.readyToast?.projectID)
        .tiktokReturnPrompt()                      // "Did you post it?" after Post to TikTok
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
        .overlay(alignment: .top) {
            // An export finished while the creator was somewhere else in the app.
            if tracker.readyBanner == nil, let text = exports.readyBanner {
                ReadyBanner(title: text, subtitle: "Open the cut in Edits to save or share it") {
                    exports.readyBanner = nil
                    select(.edits)
                } onDismiss: {
                    exports.readyBanner = nil
                }
                .padding(.horizontal, 16)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.easeOut(duration: 0.25), value: tracker.readyBanner)
        .overlay {
            // The 4 welcome screens: first visit, or Perfil → Help & Support → Replay intro. They end on Create.
            if intro.showing {
                IntroView { intro.finish(); select(.create) }
                    .transition(.opacity)
                    .zIndex(10)
            }
        }
        .animation(.easeOut(duration: 0.25), value: intro.showing)
        .onAppear { intro.showIfFirstTime() }
        // "Try … now" in a guide: go to the tab; the screen there opens the feature.
        .onChange(of: router.pending) { _, destination in
            guard let destination else { return }
            tab = router.tab
            if destination == .edits || destination == .plan { router.pending = nil }
        }
        .animation(.easeOut(duration: 0.25), value: exports.readyBanner)
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
        .task {
            AppStateBridge.refreshProjects = { try await appState.refreshProjects() }
            AppStateBridge.markReady = { id in
                if let index = appState.projects.firstIndex(where: { $0.projectID == id }) {
                    appState.projects[index].state = "draft_ready"
                }
            }
            // Build 24: cuts still being made on the server (e.g. CutSell was closed meanwhile) are
            // followed from launch, so they move to Ready with the notice as soon as they finish.
            for project in appState.projects where Project.processingStates.contains(project.state) {
                CutTracker.shared.follow(project: project)
            }
        }
        // A "Time to post" reminder was tapped: Plan opens that day.
        .onChange(of: planWeek.requestedDay) { _, day in if day != nil { tab = .plan } }
        .appColorScheme()
    }

    private func select(_ next: MainTab) {
        tab = next
    }
}

/// "Your cut is ready" banner shown in any tab when a cut finishes in the background.
struct ReadyBanner: View {
    var title = "Your cut is ready"
    var subtitle = "Tap to review it"
    let onOpen: () -> Void
    let onDismiss: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image("ProcessingLogo").resizable().frame(width: 34, height: 34).blendMode(.screen)
            VStack(alignment: .leading, spacing: 2) {
                Text(L10n.tr(title)).font(CS.inter(15, .semibold)).foregroundStyle(Color.ink)
                Text(L10n.tr(subtitle)).font(CS.inter(12)).foregroundStyle(CS.textSecondaryF)
            }
            Spacer()
            Button(action: onDismiss) {
                Image(systemName: "xmark").font(.system(size: 13, weight: .bold)).foregroundStyle(CS.textMutedF)
                    .frame(width: 32, height: 32)
            }
            .accessibilityLabel(L10n.tr("Dismiss"))
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
                        Text(L10n.tr(item.title))
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
