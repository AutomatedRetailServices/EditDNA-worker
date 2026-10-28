import SwiftUI

/// Create tab start screen. 28 Oct (PO): the same 7 actions, restyled "more app / bubbly" in White Indigo:
/// a fan of her latest cuts, a centred two-tone question, Record + Upload pills, round bubbles, her numbers, Recent.
struct HomeHubView: View {
    let onUpload: (CreateMode) -> Void
    var onMix: () -> Void = {}
    let onRecord: () -> Void
    let onOpenProject: (Project) -> Void
    let selectTab: (MainTab) -> Void
    /// 23 Oct (PO): "What are you promoting?" first; the action runs after Continue / Detect for me.
    var askPromoting: (PromotingPurpose, @escaping () -> Void) -> Void = { _, then in then() }

    @EnvironmentObject private var appState: AppState
    @ObservedObject private var tracker = CutTracker.shared
    @ObservedObject private var thumbnails = ThumbnailStore.shared
    @State private var voiceOverSoon = false

    private var recent: [Project] { Array(appState.projects.prefix(3)) }
    /// Turns on the "Faceless video" row once that flow is built.
    static let facelessVideoReady = false

    @ObservedObject private var account = AccountProfile.shared

    private var firstName: String {
        account.name.split(separator: " ").first.map(String.init) ?? ""
    }

    var body: some View {
        ZStack {
            HomeBackdrop()
            ScrollView {
                VStack(spacing: 0) {
                    // 1) Header: only the logo, on the left (PO: no avatar on Create).
                    HStack {
                        BrandLogo(width: 150).padding(.leading, -10)
                        Spacer()
                    }
                    .frame(height: 58)
                    .padding(.top, 2)

                    // 2) Hero (28 Oct, PO: "más app, bubbly"): her latest cuts as a fan of tilted cards.
                    HomeFan(projects: recent, onOpen: onOpenProject)
                        .padding(.top, 10)

                    // 3) Greeting + the two-tone question, centred.
                    Text(firstName.isEmpty ? L10n.tr("Hi there 👋") : L10n.tr("Hi \(firstName) 👋"))
                        .font(CS.inter(15, .medium))
                        .foregroundStyle(CS.wiSecondary)
                        .padding(.top, 18)
                    (Text(L10n.tr("What are we promoting")).foregroundColor(CS.wiInk)
                     + Text(" ") + Text(L10n.tr("today?")).foregroundColor(CS.accent))
                        .font(CS.inter(29, .heavy))
                        .multilineTextAlignment(.center)
                        .lineLimit(2)
                        .minimumScaleFactor(0.8)
                        .padding(.top, 4)
                        .padding(.horizontal, 12)
                    Text(L10n.tr("AI cuts silences, mistakes & repeats."))
                        .font(CS.inter(14, .medium))
                        .foregroundStyle(CS.wiSecondary)
                        .multilineTextAlignment(.center)
                        .padding(.top, 6)

                    // 4) The two big pills: Record (ink) and Upload (blue).
                    HStack(spacing: 10) {
                        bigPill(icon: "WIRecord", title: "Record", fill: AnyShapeStyle(CS.wiInk), id: "hubRecord") {
                            askPromoting(.record, onRecord)
                        }
                        bigPill(icon: "WIUpload", title: "Upload", fill: AnyShapeStyle(CS.accentGradient), id: "hubMode_single") {
                            askPromoting(.upload(.single)) { onUpload(.single) }
                        }
                    }
                    .padding(.top, 22)

                    // 5) More ways, as round bubbles.
                    HStack(alignment: .top, spacing: 0) {
                        bubble(icon: "WIBulk", title: CreateMode.bulk.title, id: "hubMode_bulk") {
                            askPromoting(.upload(.bulk)) { onUpload(.bulk) }
                        }
                        bubble(icon: "WICombine", title: CreateMode.combine.title, id: "hubMode_combine") {
                            askPromoting(.upload(.combine)) { onUpload(.combine) }
                        }
                        bubble(icon: "WISalesMix", title: "Sales Mix", id: "hubMix") { askPromoting(.salesMix, onMix) }
                        bubble(icon: "WIMic", title: "Faceless", id: "hubFaceless") {
                            // The Faceless flow is not built yet: the existing "Coming soon" notice.
                            askPromoting(.faceless) { voiceOverSoon = true }
                        }
                        bubble(icon: "WIPlanCal", title: "Plan my week", id: "hubPlanWeek") {
                            AppRouter.shared.pending = .planWeek
                            selectTab(.plan)
                        }
                    }
                    .padding(.top, 18)
                    .accessibilityElement(children: .contain)
                    .accessibilityIdentifier("hubUpload")

                    // 6) What CutSell already did for her (same numbers as Profile › Your results).
                    if editedCount > 0 { statsCard.padding(.top, 22) }

                    // 7) Recent, in a soft white sheet.
                    recentSheet.padding(.top, 16)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 24)
            }
        }
        .wiStatusBarCover()
        // The account name for "Hi {name} 👋" (people who signed in before it was kept on this iPhone).
        .task { if account.name.isEmpty { await account.refresh(session: appState.session) } }
        .safeAreaInset(edge: .bottom) { CSTabBar(current: .create, select: selectTab) }
        .task(id: recent.filter { Project.processingStates.contains($0.state) }.map(\.id)) {
            for project in recent where Project.processingStates.contains(project.state) {
                CutTracker.shared.follow(project: project)
            }
        }
        .alert(L10n.tr("Faceless video"), isPresented: $voiceOverSoon) {
            Button(L10n.tr("OK"), role: .cancel) {}
        } message: {
            Text(L10n.tr("Coming soon. For now you can record a voice over inside the editor of any cut."))
        }
    }

    private var editedCount: Int { EditStats.edited(appState.projects).count }

    /// Hours saved: 20 minutes of hand editing per video (PO 26 Oct, the Profile number).
    private var savedText: String {
        let minutes = Int(Double(editedCount) * EditStats.minutesSavedPerVideo)
        let h = minutes / 60, m = minutes % 60
        if h == 0 { return "\(m)m" }
        return m == 0 ? "\(h)h" : "\(h)h \(m)m"
    }

    /// Big capsule button, 58 high, white icon + label, soft coloured shadow.
    private func bigPill(icon: String, title: String, fill: AnyShapeStyle, id: String,
                         action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(icon).renderingMode(.template).resizable()
                    .frame(width: 20, height: 20)
                Text(L10n.tr(title))
                    .font(CS.inter(17, .bold))
                    .lineLimit(1).minimumScaleFactor(0.8)
            }
            .foregroundStyle(Color(fixed: 0xFFFFFF))
            .frame(maxWidth: .infinity, minHeight: 58, maxHeight: 58)
            .background(fill, in: Capsule())
            .shadow(color: Color(fixed: 0x1769FF, opacity: 0.22), radius: 12, x: 0, y: 8)
            .contentShape(Capsule())
        }
        .buttonStyle(BubblePress())
        .accessibilityIdentifier(id)
    }

    /// A round white bubble (58 pt) with the blue icon, the name under it.
    private func bubble(icon: String, title: String, id: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 7) {
                Image(icon).renderingMode(.template).resizable()
                    .frame(width: 22, height: 22)
                    .foregroundStyle(CS.accent)
                    .frame(width: 58, height: 58)
                    .background(Color(fixed: 0xFFFFFF), in: Circle())
                    .overlay(Circle().stroke(Color(fixed: 0x1769FF, opacity: 0.10), lineWidth: 1))
                    .shadow(color: Color(fixed: 0x1769FF, opacity: 0.14), radius: 10, x: 0, y: 6)
                Text(L10n.tr(title))
                    .font(CS.inter(11, .semibold))
                    .foregroundStyle(CS.wiInk)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                    .minimumScaleFactor(0.85)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity)
            .contentShape(Rectangle())
        }
        .buttonStyle(BubblePress())
        .accessibilityIdentifier(id)
    }

    private var statsCard: some View {
        HStack(spacing: 0) {
            stat(savedText, L10n.tr("saved editing"))
            Rectangle().fill(CS.wiBorder).frame(width: 1, height: 40)
            stat("\(editedCount)", L10n.tr("videos edited"))
        }
        .padding(.vertical, 16)
        .background(Color(fixed: 0xFFFFFF, opacity: 0.85), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 26, style: .continuous).stroke(Color(fixed: 0xFFFFFF), lineWidth: 1))
        .shadow(color: Color(fixed: 0x0D1433, opacity: 0.06), radius: 14, x: 0, y: 6)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("hubStats")
    }

    private func stat(_ value: String, _ label: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(CS.inter(28, .heavy))
                .foregroundStyle(CS.accent)
                .lineLimit(1).minimumScaleFactor(0.7)
            Text(label)
                .font(CS.inter(12, .medium))
                .foregroundStyle(CS.wiSecondary)
                .lineLimit(1).minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity)
    }

    private var recentSheet: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                Text(L10n.tr("Recent"))
                    .font(CS.inter(19, .heavy))
                    .foregroundStyle(CS.wiInk)
                Spacer()
                Button(L10n.tr("See all")) { selectTab(.edits) }
                    .font(CS.inter(13, .semibold))
                    .foregroundStyle(CS.accent)
                    .accessibilityIdentifier("hubSeeAll")
            }
            if recent.isEmpty {
                Text(L10n.tr("Your cuts will show up here."))
                    .font(CS.inter(13, .medium))
                    .foregroundStyle(CS.wiSecondary)
                    .frame(maxWidth: .infinity, minHeight: 90)
                    .background(CS.wiCard, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .padding(.top, 12)
            } else {
                HStack(spacing: 10) {
                    ForEach(Array(recent.enumerated()), id: \.element.id) { index, project in
                        Button { onOpenProject(project) } label: { recentTile(project, index: index) }
                            .buttonStyle(BubblePress())
                    }
                    ForEach(recent.count..<3, id: \.self) { _ in Color.clear.frame(height: 150) }
                }
                .padding(.top, 12)
            }
        }
        .padding(16)
        .background(Color(fixed: 0xFFFFFF, opacity: 0.85), in: RoundedRectangle(cornerRadius: 28, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 28, style: .continuous).stroke(Color(fixed: 0xFFFFFF), lineWidth: 1))
        .shadow(color: Color(fixed: 0x0D1433, opacity: 0.06), radius: 14, x: 0, y: 6)
    }

    /// "Recent": portrait thumbnails, 150 high, radius 18.
    private func recentTile(_ project: Project, index: Int) -> some View {
        let isProcessing = Project.processingStates.contains(project.state)
        return RoundedRectangle(cornerRadius: 18, style: .continuous)
            .fill(CS.wiSurface2)          // placeholder until the thumbnail loads
            .frame(height: 150)
            .overlay {
                if let image = thumbnails.image(for: project.projectID) {
                    Image(uiImage: image).resizable().scaledToFill()
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .task(id: project.state) { await thumbnails.load(project: project, session: appState.session) }
            .overlay {
                if isProcessing { ProgressView().tint(CS.accent) }
            }
            .accessibilityLabel(project.title)
    }
}

/// Soft White Indigo backdrop: white with a light blue wash at the top and two blurred glows.
struct HomeBackdrop: View {
    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(fixed: 0xE7EEFF), Color(fixed: 0xF5F8FF), Color(fixed: 0xFFFFFF)],
                           startPoint: .top, endPoint: .bottom)
            Circle().fill(Color(fixed: 0x1769FF, opacity: 0.16)).frame(width: 300, height: 300)
                .blur(radius: 80).offset(x: 150, y: -300)
            Circle().fill(Color(fixed: 0x4A5BFF, opacity: 0.12)).frame(width: 260, height: 260)
                .blur(radius: 80).offset(x: -160, y: -120)
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
    }
}

/// The hero: up to three of her cuts as tilted cards, the newest in front. Without cuts, three soft cards
/// with the CutSell icon, so the screen never looks empty.
struct HomeFan: View {
    let projects: [Project]
    let onOpen: (Project) -> Void
    @EnvironmentObject private var appState: AppState
    @ObservedObject private var thumbnails = ThumbnailStore.shared

    private let slots: [(angle: Double, x: CGFloat, y: CGFloat)] = [(-9, -82, 14), (0, 0, 0), (9, 82, 14)]

    var body: some View {
        // Order on screen: 2nd newest left, newest centre (on top), 3rd right.
        let order = [1, 0, 2]
        ZStack {
            ForEach(Array(order.enumerated()), id: \.offset) { position, index in
                let slot = slots[position]
                card(index)
                    .rotationEffect(.degrees(slot.angle))
                    .offset(x: slot.x, y: slot.y)
                    .zIndex(index == 0 ? 2 : 1)
            }
        }
        .frame(height: 196)
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("hubFan")
    }

    @ViewBuilder
    private func card(_ index: Int) -> some View {
        let width: CGFloat = index == 0 ? 116 : 104
        let shape = RoundedRectangle(cornerRadius: 22, style: .continuous)
        if index < projects.count {
            let project = projects[index]
            Button { onOpen(project) } label: {
                shape.fill(CS.wiSurface2)
                    .overlay {
                        if let image = thumbnails.image(for: project.projectID) {
                            Image(uiImage: image).resizable().scaledToFill()
                        } else {
                            BrandIcon(size: 34).opacity(0.5)
                        }
                    }
                    .clipShape(shape)
                    .frame(width: width, height: width * 16 / 10)
                    .overlay(shape.stroke(Color(fixed: 0xFFFFFF), lineWidth: 4))
                    .shadow(color: Color(fixed: 0x1769FF, opacity: 0.20), radius: 14, x: 0, y: 10)
            }
            .buttonStyle(BubblePress())
            .task(id: project.state) { await thumbnails.load(project: project, session: appState.session) }
            .accessibilityLabel(project.title)
        } else {
            shape.fill(LinearGradient(colors: index == 0 ? [Color(fixed: 0x1769FF), Color(fixed: 0x4A5BFF)]
                                                         : [Color(fixed: 0xDCE6FF), Color(fixed: 0xC9D6FF)],
                                      startPoint: .topLeading, endPoint: .bottomTrailing))
                .overlay {
                    if index == 0 {
                        Image(systemName: "sparkles").font(.system(size: 30, weight: .semibold))
                            .foregroundStyle(Color(fixed: 0xFFFFFF))
                    }
                }
                .frame(width: width, height: width * 16 / 10)
                .overlay(shape.stroke(Color(fixed: 0xFFFFFF), lineWidth: 4))
                .shadow(color: Color(fixed: 0x1769FF, opacity: 0.18), radius: 14, x: 0, y: 10)
                .accessibilityHidden(true)
        }
    }
}

/// A gentle "squish" when a bubbly button is pressed.
struct BubblePress: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.95 : 1)
            .animation(.spring(response: 0.25, dampingFraction: 0.6), value: configuration.isPressed)
    }
}

enum PlanMeter {
    /// The monthly number shown in the Figma design (1068:22). Not enforced.
    static let figmaMonthlyCuts = 30

    /// Cuts this account created in the current calendar month (from the server's created_at).
    /// The server writes ISO dates ("2026-10-12T22:28:05.952628+00:00"); the month is the first 7 characters.
    static func cutsThisMonth(_ projects: [Project], now: Date = Date()) -> Int {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.dateFormat = "yyyy-MM"
        let month = formatter.string(from: now)
        return projects.filter { ($0.createdAt ?? $0.updatedAt ?? "").hasPrefix(month) }.count
    }
}
