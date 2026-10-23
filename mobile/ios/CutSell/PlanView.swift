import SwiftUI

/// Plan · Home (Figma 1137:2736): make videos in batches. "New Mix" opens Mix · Setup, "Bulk" opens the Bulk gallery
/// (the same upload pipeline as Create), "In progress" follows every Mix and Bulk being made (tap → Edits ·
/// Processing showing that group), "This week" shows the week Plan my week built ("Plan my week ›" opens Goal, or Week
/// once a week is built).
struct PlanView: View {
    let selectTab: (MainTab) -> Void

    @EnvironmentObject private var appState: AppState
    @ObservedObject private var mixes = MixCenter.shared
    @ObservedObject private var queue = ProcessingQueue.shared
    @ObservedObject private var thumbnails = ThumbnailStore.shared
    @State private var showingMix = false
    @ObservedObject private var week = PlanWeekStore.shared
    @State private var path: [PlanRoute] = []
    @ObservedObject private var router = AppRouter.shared
    @Environment(\.colorScheme) private var scheme

    enum PlanRoute: Hashable { case goal, week, day(String) }

    private func openFromGuide() {
        guard router.pending == .planWeek else { return }
        router.pending = nil
        path = week.hasWeek ? [.week] : [.goal]
    }

    var body: some View {
        NavigationStack(path: $path) {
            home
                .navigationDestination(for: PlanRoute.self) { route in
                    Group {
                        switch route {
                        case .goal:
                            PlanGoalView(onBuilt: { path = [.week] }).environmentObject(appState)
                        case .week:
                            PlanWeekView(onNewMix: { showingMix = true }, onSettings: { path.append(.goal) },
                                         onOpenDay: { path.append(.day($0)) }).environmentObject(appState)
                        case .day(let day):
                            PlanDayView(day: day).environmentObject(appState)
                        }
                    }
                    // White Indigo (Figma 1209:2 / 70 / 170): the bottom bar stays on Plan's own screens.
                    .safeAreaInset(edge: .bottom) {
                        CSTabBar(current: .plan, select: { tab in if tab == .plan { path = [] } else { selectTab(tab) } })
                    }
                }
        }
        .tint(CS.accent)
        .task { await week.refresh(session: appState.session) }
        // A guide's "Try Plan my week now".
        .onAppear { openFromGuide() }
        .onChange(of: router.pending) { _, _ in openFromGuide() }
        // A reminder was tapped: that day opens, ready to share.
        .onChange(of: week.requestedDay) { _, day in
            guard let day else { return }
            week.requestedDay = nil
            if path.last != .day(day) { path = [.week, .day(day)] }        // already on that day: stay there
        }
        .onAppear {
            if let day = week.requestedDay { week.requestedDay = nil; path = [.week, .day(day)] }
        }
    }

    private var home: some View {
        ZStack {
            CS.wiPage.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    // White Indigo (Figma 1166:107): no Sales Mix / Bulk cards here (they are on Create).
                    WIHeader { EmptyView() }
                    WIScreenTitle(title: "Plan", subtitle: "What’s in progress and what to post.").padding(.top, 2)

                    HStack(alignment: .firstTextBaseline) {
                        Text(L10n.tr("In progress")).font(CS.sectionTitle).foregroundStyle(CS.wiInk)
                        Spacer()
                        Button(L10n.tr("See all in Edits ›")) { openEdits(focus: nil) }
                            .font(CS.inter(13, .semibold)).foregroundStyle(CS.accent)
                            .accessibilityIdentifier("planSeeAll")
                    }
                    .padding(.top, 22)
                    VStack(spacing: 8) {
                        ForEach(mixes.active) { mix in
                            Button { openEdits(focus: mix.mixID) } label: {
                                progressCard(tag: .mix, title: mix.product ?? "Sales Mix",
                                             detail: [mix.brand, "\(mix.ready) of \(mix.total) ready"].compactMap { $0 }.joined(separator: " · "),
                                             ready: mix.ready, total: mix.total,
                                             image: mix.videos.lazy.compactMap { thumbnails.image(for: $0.projectID) }.first)
                            }
                            .buttonStyle(.plain)
                            .accessibilityIdentifier("planProgressMix")
                        }
                        ForEach(queue.activeBulks, id: \.group.id) { bulk in
                            Button { openEdits(focus: bulk.group.id) } label: {
                                progressCard(tag: .bulk, title: "\(bulk.group.total) videos",
                                             detail: "\(bulk.ready) of \(bulk.group.total) ready",
                                             ready: bulk.ready, total: bulk.group.total, image: bulk.thumbnail)
                            }
                            .buttonStyle(.plain)
                            .accessibilityIdentifier("planProgressBulk")
                        }
                        if mixes.active.isEmpty && queue.activeBulks.isEmpty {
                            Text(L10n.tr("Nothing in progress. Start a Sales Mix or a Bulk."))
                                .font(CS.inter(13, .medium)).foregroundStyle(CS.wiSecondary)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal, 14)
                                .frame(maxWidth: .infinity, minHeight: 64)
                                .wiCard()
                                .accessibilityIdentifier("planNothingInProgress")
                        }
                    }
                    .padding(.top, 9)

                    thisWeek.padding(.top, 22)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 20)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .safeAreaInset(edge: .bottom) { CSTabBar(current: .plan, select: selectTab) }
        .task { await mixes.refresh(session: appState.session) }
        .fullScreenCover(isPresented: $showingMix) {
            MixSetupView(onClose: { showingMix = false }, onStarted: { summary in
                showingMix = false
                queue.requestedMix = summary
                openEdits(focus: summary.mixID)
            })
            .environmentObject(appState)
        }
    }

    private func openEdits(focus: String?) {
        queue.focus = focus
        queue.requestedEditsFilter = "processing"
        selectTab(.edits)
    }

    enum Tag { case mix, bulk }

    /// Figma 1166:123 / 132: thumbnail, "SALES MIX" (blue) or "BULK" (white) tag, title, detail, bar and n/total.
    private func progressCard(tag: Tag, title: String, detail: String, ready: Int, total: Int, image: UIImage?) -> some View {
        HStack(alignment: .top, spacing: 10) {
            CS.wiSurface2
                .frame(width: 40, height: 52)
                .overlay { if let image { Image(uiImage: image).resizable().scaledToFill() } }
                .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 8) {
                    Text(L10n.tr(tag == .mix ? "SALES MIX" : "BULK")).font(CS.inter(9, .bold))
                        .foregroundStyle(tag == .mix ? Color(fixed: 0xFFFFFF) : CS.wiInk)
                        .fixedSize()
                        .padding(.horizontal, 8).frame(height: 18)
                        .background(tag == .mix ? CS.accent : CS.wiPage, in: Capsule())
                        .overlay(Capsule().stroke(tag == .mix ? .clear : CS.wiBorder, lineWidth: 1))
                    Text(L10n.tr(title)).font(CS.inter(14, .semibold)).foregroundStyle(CS.wiInk).lineLimit(1)
                }
                Text(L10n.tr(detail)).font(CS.inter(11)).foregroundStyle(CS.wiSecondary).lineLimit(1).padding(.top, 6)
                HStack(spacing: 12) {
                    GeometryReader { proxy in
                        ZStack(alignment: .leading) {
                            Capsule().fill(CS.wiPage)
                            Capsule().fill(CS.accent)
                                .frame(width: proxy.size.width * CGFloat(ready) / CGFloat(max(1, total)))
                        }
                    }
                    .frame(height: 6)
                    Text(verbatim: "\(ready)/\(total)").font(CS.inter(12, .semibold)).foregroundStyle(CS.accent)
                        .fixedSize()
                }
                .padding(.top, 8)
            }
        }
        .padding(.horizontal, 13)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, minHeight: 78, alignment: .leading)
        .wiCard()
    }

    /// "This week": the days of this week with a dot per planned video (its product's color), and today's plan.
    private var thisWeek: some View {
        let calendar = Calendar(identifier: .gregorian)
        let today = Date()
        let weekday = (calendar.component(.weekday, from: today) + 5) % 7          // Monday = 0
        let monday = calendar.date(byAdding: .day, value: -weekday, to: calendar.startOfDay(for: today)) ?? today
        let days = (0..<7).compactMap { calendar.date(byAdding: .day, value: $0, to: monday) }
        let slots = week.state?.week?.slots ?? []
        let todayKey = PlanDay.iso.string(from: today)
        let todays = slots.filter { $0.date == todayKey && $0.postedAt == nil }
        let name = DateFormatter()
        name.locale = L10n.locale
        name.dateFormat = "EEE"
        return VStack(alignment: .leading, spacing: 0) {
            Text(L10n.tr("This week")).font(CS.sectionTitle).foregroundStyle(CS.wiInk)
            // Figma 1166:144-171: 7 days filling the width, 7 pt apart; today blue.
            HStack(spacing: 7) {
                ForEach(Array(days.enumerated()), id: \.offset) { _, day in
                    let isToday = calendar.isDate(day, inSameDayAs: today)
                    let key = PlanDay.iso.string(from: day)
                    Button { if week.hasWeek, slots.contains(where: { $0.date == key }) { path = [.week, .day(key)] } } label: {
                        VStack(spacing: 2) {
                            Text(name.string(from: day).replacingOccurrences(of: ".", with: "").capitalizedFirst).font(CS.inter(11, .semibold))
                                .foregroundStyle(isToday ? Color(fixed: 0xFFFFFF) : CS.wiInk)
                                .lineLimit(1).minimumScaleFactor(0.8)
                            Text(verbatim: "\(calendar.component(.day, from: day))").font(CS.inter(14, .bold))
                                .foregroundStyle(isToday ? Color(fixed: 0xFFFFFF) : CS.wiInk)
                            HStack(spacing: 2) {
                                ForEach(slots.filter { $0.date == key }.prefix(4)) { slot in
                                    Image(week.dot(for: slot)).resizable().frame(width: 6, height: 6)
                                        .opacity(slot.postedAt == nil ? 1 : 0.35)
                                }
                            }
                            .frame(height: 6)
                        }
                        .frame(maxWidth: .infinity).frame(height: 62)
                        .background(isToday ? CS.accent : CS.wiPage, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .stroke(isToday ? .clear : CS.wiBorder, lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.top, 10)
            Button { path = week.hasWeek ? [.week] : [.goal] } label: {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.tr(cardTitle(todays: todays))).font(CS.inter(14, .semibold)).foregroundStyle(CS.wiInk)
                        .accessibilityIdentifier("planWeekCardTitle")
                    Text(L10n.tr(cardBody(slots: slots))).font(CS.inter(11)).foregroundStyle(CS.wiSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(L10n.tr("Plan my week ›")).font(CS.inter(12, .semibold)).foregroundStyle(CS.accent)
                        .padding(.top, 2)
                }
                .padding(15)
                .frame(maxWidth: .infinity, alignment: .leading)
                .wiCard()
            }
            .buttonStyle(.plain)
            .padding(.top, 12)
            .accessibilityIdentifier("planMyWeek")
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("planThisWeek")
    }

    /// "Today: post 1 CeraVe and 1 Dior video" (Figma 1137:2886) from the real plan.
    private func cardTitle(todays: [PlanSlot]) -> String {
        guard week.hasWeek else { return "Plan your week" }
        guard !todays.isEmpty else { return "Nothing left to post today" }
        var counts: [(String, Int)] = []
        for slot in todays {
            let brand = slot.brand ?? "other"
            if let i = counts.firstIndex(where: { $0.0 == brand }) { counts[i].1 += 1 } else { counts.append((brand, 1)) }
        }
        let parts = counts.map { "\($0.1) \($0.0)" }
        let joined = parts.count > 1 ? parts.dropLast().joined(separator: ", ") + " and " + parts.last! : parts[0]
        return "Today: post \(joined) video\(todays.count == 1 ? "" : "s")"
    }

    private func cardBody(slots: [PlanSlot]) -> String {
        guard week.hasWeek else {
            let ready = week.state?.summary.ready ?? 0
            return ready == 0 ? "When your videos are ready, CutSell spreads them over the week so your feed stays varied."
                : "You have \(ready) ready video\(ready == 1 ? "" : "s"). We spread them over the week, between products, so your feed stays varied."
        }
        let days = Set(slots.map(\.date)).count
        return "We spread your \(slots.count) video\(slots.count == 1 ? "" : "s") over \(days) day\(days == 1 ? "" : "s"), between products, so your feed stays varied."
    }
}
