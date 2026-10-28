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
                    // 1-2) Hero (28 Oct, PO): no logo in the corner; the CutSell logo itself (matte black, puffy)
                    // in the middle, bouncing like jelly, its letters hopping one by one, a few coins.
                    HomeLogoHero()
                        .padding(.top, 6)

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

/// The Create hero (PO 28 Oct): the CutSell logo -- the same bag and name, matte black with a soft puffy volume --
/// bouncing like jelly, the letters of the name hopping one after another, and a few gold coins popping out of
/// the bag. Livelier for the first seconds after opening, then calmer. Reduce Motion: still.
struct HomeLogoHero: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var start = Date()

    /// Each letter of "CutSell.ai" (HomeWord0...9): left edge and width as a share of the name's width.
    static let letters: [(x: CGFloat, w: CGFloat)] = [
        (0.0, 0.1472), (0.1529, 0.123), (0.2834, 0.0923), (0.3765, 0.1298), (0.51, 0.1264),
        (0.6485, 0.0409), (0.7094, 0.0409), (0.7628, 0.0492), (0.8199, 0.1158), (0.9501, 0.0499)]
    static let wordWidth: CGFloat = 200
    static let wordHeight: CGFloat = 200 * 464 / 2643
    /// Coins: where each one flies to (from the top of the bag) and when it starts within its 6 s loop.
    static let coins: [(dx: CGFloat, dy: CGFloat, at: Double)] = [(-96, -46, 0.4), (100, -58, 2.4), (-70, -70, 4.4)]

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60, paused: reduceMotion)) { context in
            let t = reduceMotion ? 0 : context.date.timeIntervalSince(start)
            scene(t: t)
        }
        .frame(height: 270)
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(L10n.tr("CutSell.ai"))
        .accessibilityIdentifier("hubLogoHero")
    }

    /// Seconds into the current bounce: a bounce every 3 s for the first ~10 s, then every 6 s.
    static func bounceTime(_ t: Double) -> Double {
        t < 9 ? t.truncatingRemainder(dividingBy: 3) : (t - 9).truncatingRemainder(dividingBy: 6)
    }

    @ViewBuilder
    private func scene(t: Double) -> some View {
        let local = Self.bounceTime(t)
        let bag = reduceMotion ? Squash.rest : Self.jelly(local, height: 14)
        let float = reduceMotion ? 0 : CGFloat(sin(t * 2 * .pi / 3) * 5)
        ZStack {
            // Soft colour blobs behind (pink, lilac, blue), slowly breathing.
            Circle().fill(Color(fixed: 0xFFB3D9, opacity: 0.55)).frame(width: 170, height: 150)
                .blur(radius: 40).offset(x: -80, y: -10 + CGFloat(sin(t / 1.3) * 6))
            Circle().fill(Color(fixed: 0xB9C8FF, opacity: 0.6)).frame(width: 180, height: 170)
                .blur(radius: 40).offset(x: 90, y: -40 + CGFloat(cos(t / 1.5) * 6))
            Circle().fill(Color(fixed: 0xD9C2FF, opacity: 0.5)).frame(width: 150, height: 140)
                .blur(radius: 40).offset(x: 10, y: 50 + CGFloat(sin(t / 1.7) * 6))

            // The floor shadow shrinks while the bag is up.
            Ellipse().fill(Color(fixed: 0x0D1433, opacity: 0.16)).frame(width: 110, height: 14).blur(radius: 6)
                .scaleEffect(x: 1 - bag.lift * 0.012 + float * 0.01, y: 1)
                .offset(y: 124)

            ForEach(0..<Self.coins.count, id: \.self) { i in
                coin(t: t, Self.coins[i])
            }

            Image("HomeBag").resizable().interpolation(.high).frame(width: 150, height: 154)
                .scaleEffect(x: bag.x, y: bag.y, anchor: .bottom)
                .offset(y: -38 - bag.lift + float)

            // "CutSell.ai": each letter hops a little after the one before it.
            ZStack(alignment: .topLeading) {
                ForEach(0..<Self.letters.count, id: \.self) { i in
                    let hop = reduceMotion ? Squash.rest : Self.jelly(local - Double(i) * 0.09, height: 9)
                    Image("HomeWord\(i)").resizable().interpolation(.high)
                        .frame(width: Self.letters[i].w * Self.wordWidth, height: Self.wordHeight)
                        .scaleEffect(x: hop.x, y: hop.y, anchor: .bottom)
                        .offset(x: Self.letters[i].x * Self.wordWidth, y: -hop.lift)
                }
            }
            .frame(width: Self.wordWidth, height: Self.wordHeight, alignment: .topLeading)
            .offset(y: 82 + float * 0.6)
        }
    }

    struct Squash: Equatable {
        var x: CGFloat, y: CGFloat, lift: CGFloat
        static let rest = Squash(x: 1, y: 1, lift: 0)
    }

    /// Squash and stretch over one bounce, `s` seconds into it (a bounce lasts 1 s).
    static func jelly(_ s: Double, height: Double) -> Squash {
        guard s > 0, s < 1 else { return .rest }
        let keys: [(t: Double, x: Double, y: Double, lift: Double)] = [
            (0, 1, 1, 0), (0.18, 1.1, 0.88, 0), (0.42, 0.93, 1.08, 1), (0.66, 1.04, 0.96, 0), (0.84, 0.99, 1.01, 0), (1, 1, 1, 0)]
        for k in 1..<keys.count where s <= keys[k].t {
            let a = keys[k - 1], b = keys[k]
            let p = (s - a.t) / (b.t - a.t), e = p * p * (3 - 2 * p)
            return Squash(x: CGFloat(a.x + (b.x - a.x) * e), y: CGFloat(a.y + (b.y - a.y) * e),
                          lift: CGFloat((a.lift + (b.lift - a.lift) * e) * height))
        }
        return .rest
    }

    /// A gold "$" coin that pops out of the top of the bag, spins and fades (1.6 s, every 6 s).
    private func coin(t: Double, _ c: (dx: CGFloat, dy: CGFloat, at: Double)) -> some View {
        let since = t - c.at
        let p = since < 0 ? 2 : since.truncatingRemainder(dividingBy: 6) / 1.6
        let live = !reduceMotion && p < 1
        let e = CGFloat(1 - pow(1 - min(1, p), 3))
        let spin = CGFloat(cos(p * 3 * .pi))
        return Text("$")
            .font(CS.inter(15, .heavy))
            .foregroundStyle(Color(fixed: 0xA86A00))
            .frame(width: 28, height: 28)
            .background(RadialGradient(colors: [Color(fixed: 0xFFF3B0), Color(fixed: 0xFFD34D), Color(fixed: 0xF0A818)],
                                       center: UnitPoint(x: 0.35, y: 0.3), startRadius: 1, endRadius: 18), in: Circle())
            .shadow(color: Color(fixed: 0xF0A818, opacity: 0.35), radius: 5, y: 3)
            .scaleEffect(x: spin * (0.4 + 0.6 * e), y: 0.4 + 0.6 * e)
            .offset(x: c.dx * e, y: -70 + c.dy * e)
            .opacity(live ? (p < 0.7 ? 1 : (1 - p) / 0.3) : 0)
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
