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
    @ObservedObject private var tracker = CutTracker.shared
    @State private var showingMix = false
    @ObservedObject private var week = PlanWeekStore.shared
    @State private var path: [PlanRoute] = []
    /// The day she tapped in the week strip (nil = today).
    @State private var shownDay: String?
    #if DEBUG
    nonisolated(unsafe) static var previewToday: Date?
    #endif
    @ObservedObject private var router = AppRouter.shared
    @Environment(\.colorScheme) private var scheme

    enum PlanRoute: Hashable { case goal, week, day(String) }

    private func openFromGuide() {
        guard router.pending == .planWeek || router.pending == .thisWeek else { return }
        let week = router.pending == .thisWeek
        router.pending = nil
        path = week ? [.week] : [.goal]                // PO 24 Oct: "Plan my week" always opens Plan my week first
    }

    var body: some View {
        NavigationStack(path: $path) {
            home.wiStatusBarCover()
                .navigationDestination(for: PlanRoute.self) { route in
                    Group {
                        switch route {
                        case .goal:
                            PlanMyWeekView(onBuilt: { path = [.week] }).environmentObject(appState)
                        case .week:
                            PlanWeekView(onNewMix: { showingMix = true }, onSettings: nil,
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
                    WIScreenTitle(title: "Plan", subtitle: attention == nil ? "What’s in progress and what to post." : "Plan needs attention")
                        .padding(.top, 2)
                    if let attention {
                        NeedsAttentionSection(summary: attention.summary, rows: attention.rows, onRetry: retry)
                            .padding(.top, 16)
                    }

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

    /// Figma 1233:3243 / 1233:3354: the newest Bulk send where a video really failed (a video without a voice is not
    /// a failure: it goes to Sales Mix from Edits).
    private var attention: (summary: NeedsAttentionSummary, rows: [ProcessingQueue.Row])? {
        let rows = queue.rows(projects: appState.projects, tracker: tracker)
        for group in queue.bulkGroups.sorted(by: { $0.startedAt > $1.startedAt }) {
            let mine = rows.filter { $0.group == group.id }
            let failed = mine.contains { if case .failed(let why) = $0.state { return why != ProcessingQueue.noSpeech }; return false }
            guard failed else { continue }
            // Ready videos leave the list once the whole send is done: the send's size says how many are ready.
            var summary = NeedsAttentionSummary(rows: mine)
            summary.ready = max(summary.ready, group.total - summary.active - summary.queued - summary.failed)
            return (summary, mine)
        }
        return nil
    }

    /// The same Retry as Edits › Processing.
    private func retry(_ row: ProcessingQueue.Row) {
        if let item = row.itemID, queue.canRetry(item) { queue.retry(item); return }
        if let id = row.projectID { Task { _ = await tracker.retry(projectID: id) } }
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

    /// PO 25 Oct: a living Plan. The 7 days (tap one to see it), the chosen day's videos with their picture, brand,
    /// product and time (tap → that day), then the whole week as a strip of large pictures (tap → This week).
    private var thisWeek: some View {
        let calendar = Calendar(identifier: .gregorian)
        var today = Date()
        #if DEBUG
        // Design screenshots only: another "today" (to show the day row on a day other than Monday).
        if let d = Self.previewToday { today = d }
        #endif
        // Option B (PO 26 Oct, to review): the next 7 days from today, today first.
        let start = calendar.startOfDay(for: today)
        let days = (0..<7).compactMap { calendar.date(byAdding: .day, value: $0, to: start) }
        let slots = week.state?.week?.slots ?? []
        let todayKey = PlanDay.iso.string(from: today)
        let upcoming = slots.filter { $0.date >= todayKey && $0.postedAt == nil }
        // Nothing today: the next day with videos opens by itself (she sees what is coming without a tap).
        let chosen = shownDay ?? (slots.contains { $0.date == todayKey } ? todayKey : (upcoming.first?.date ?? todayKey))
        let daySlots = slots.filter { $0.date == chosen }
        let name = DateFormatter()
        name.locale = L10n.locale
        name.dateFormat = "EEE"
        return VStack(alignment: .leading, spacing: 0) {
            Text(L10n.tr("This week")).font(CS.sectionTitle).foregroundStyle(CS.wiInk)
            // Figma 1166:144-171: 7 days filling the width; today blue; the day she taps is outlined.
            HStack(spacing: 7) {
                ForEach(Array(days.enumerated()), id: \.offset) { _, day in
                    let isToday = calendar.isDate(day, inSameDayAs: today)
                    let key = PlanDay.iso.string(from: day)
                    let picked = key == chosen
                    Button { withAnimation(.easeOut(duration: 0.2)) { shownDay = key } } label: {
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
                            .stroke(picked && !isToday ? CS.accent : isToday ? .clear : CS.wiBorder, lineWidth: picked && !isToday ? 2 : 1))
                    }
                    .buttonStyle(.plain)
                    .accessibilityIdentifier("planDay_\(key)")
                }
            }
            .padding(.top, 10)

            dayCard(chosen, slots: daySlots, isToday: chosen == todayKey).padding(.top, 12)

            // PO 26 Oct (A): Coming up starts AFTER the day in the card, so nothing shows twice.
            let after = upcoming.filter { $0.date > chosen }
            if week.hasWeek && !after.isEmpty {
                weekStrip(after).padding(.top, 22)
            } else if week.hasWeek {
                Button(L10n.tr("See this week ›")) { path = [.week] }
                    .font(CS.inter(13, .semibold)).foregroundStyle(CS.accent)
                    .padding(.top, 12)
                    .accessibilityIdentifier("planSeeWeek")
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("planThisWeek")
    }

    /// The chosen day: "Today · Sun 25 · 2 videos to post", each video (picture, brand, product, time, posted), and
    /// "Plan my week ›". Nothing planned yet: what Plan my week does, with its button.
    private func dayCard(_ day: String, slots: [PlanSlot], isToday: Bool) -> some View {
        let left = slots.filter { $0.postedAt == nil }.count
        return VStack(alignment: .leading, spacing: 10) {
            if slots.isEmpty {
                Text(L10n.tr(week.hasWeek ? "Nothing planned for \(isToday ? "today" : PlanDay.short(day))" : "Plan your week"))
                    .font(CS.inter(15, .bold)).foregroundStyle(CS.wiInk)
                    .accessibilityIdentifier("planWeekCardTitle")
                Text(L10n.tr(cardBody(slots: week.state?.week?.slots ?? [])))
                    .font(CS.inter(12)).foregroundStyle(CS.wiSecondary).fixedSize(horizontal: false, vertical: true)
            } else {
                HStack(alignment: .firstTextBaseline) {
                    Text(L10n.tr(isToday ? "Today · \(PlanDay.short(day))" : PlanDay.short(day)))
                        .font(CS.inter(16, .bold)).foregroundStyle(CS.wiInk)
                        .accessibilityIdentifier("planWeekCardTitle")
                    Spacer()
                    Text(L10n.tr(left == 0 ? "All posted ✓" : "\(left) to post")).font(CS.inter(12, .semibold))
                        .foregroundStyle(left == 0 ? CS.wiSuccess : CS.accent)
                }
                ForEach(slots) { slot in
                    Button { path = [.week, .day(day)] } label: { videoRow(slot) }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("planDayVideo")
                }
            }
            Button { path = [.goal] } label: {
                Text(L10n.tr("Plan my week ›")).font(CS.inter(13, .semibold)).foregroundStyle(CS.accent)
            }
            .buttonStyle(.plain)
            .padding(.top, 2)
            .accessibilityIdentifier("planMyWeek")
        }
        .padding(15)
        .frame(maxWidth: .infinity, alignment: .leading)
        .wiCard()
        .animation(.easeOut(duration: 0.2), value: day)
    }

    /// One video of the day: picture, brand dot + brand, product, time; posted ones fade with a check.
    private func videoRow(_ slot: PlanSlot) -> some View {
        HStack(spacing: 12) {
            thumb(slot, width: 44, height: 60, radius: 9)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 5) {
                    Image(week.dot(for: slot)).resizable().frame(width: 7, height: 7)
                    Text(L10n.tr(slot.brand ?? "No brand")).font(CS.inter(13, .semibold)).foregroundStyle(CS.wiInk).lineLimit(1)
                }
                Text(L10n.tr(slot.product ?? "Video")).font(CS.inter(12)).foregroundStyle(CS.wiSecondary).lineLimit(1)
            }
            Spacer(minLength: 6)
            if slot.postedAt != nil {
                Label(L10n.tr("Posted"), systemImage: "checkmark").font(CS.inter(12, .semibold)).foregroundStyle(CS.wiSuccess)
            } else {
                Text(PlanDay.clock(slot.time)).font(CS.inter(13, .bold)).foregroundStyle(CS.accent)
            }
        }
        .padding(8)
        .background(CS.wiPage, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
        .opacity(slot.postedAt == nil ? 1 : 0.6)
    }

    /// "Coming up": every video still to post, as large pictures with day, time and brand, sliding sideways.
    private func weekStrip(_ upcoming: [PlanSlot]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(L10n.tr("Coming up")).font(CS.sectionTitle).foregroundStyle(CS.wiInk)
                Text(verbatim: "\(upcoming.count)").font(CS.inter(13, .bold)).foregroundStyle(CS.accent)
                Spacer()
                Button(L10n.tr("See this week ›")) { path = [.week] }
                    .font(CS.inter(13, .semibold)).foregroundStyle(CS.accent)
                    .accessibilityIdentifier("planSeeWeek")
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(upcoming) { slot in
                        Button { path = [.week, .day(slot.date)] } label: {
                            ZStack(alignment: .bottomLeading) {
                                thumb(slot, width: 112, height: 168, radius: 16)
                                LinearGradient(colors: [.clear, Color(fixed: 0x000000, opacity: 0.6)], startPoint: .center, endPoint: .bottom)
                                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(verbatim: "\(PlanDay.short(slot.date)) · \(PlanDay.clock(slot.time))")
                                        .font(CS.inter(10, .semibold)).foregroundStyle(Color(fixed: 0xFFFFFF, opacity: 0.85))
                                    Text(L10n.tr(slot.brand ?? "No brand")).font(CS.inter(13, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                                        .lineLimit(1)
                                    Text(L10n.tr(slot.product ?? "Video")).font(CS.inter(11)).foregroundStyle(Color(fixed: 0xFFFFFF, opacity: 0.85))
                                        .lineLimit(1)
                                }
                                .padding(9)
                            }
                            .frame(width: 112, height: 168)
                            .overlay(alignment: .topLeading) {
                                Image(week.dot(for: slot)).resizable().frame(width: 9, height: 9).padding(9)
                            }
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("planStripVideo")
                    }
                }
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("planComingUp")
    }

    private func thumb(_ slot: PlanSlot, width: CGFloat, height: CGFloat, radius: CGFloat) -> some View {
        Color(fixed: 0xE9ECF1)
            .frame(width: width, height: height)
            .overlay { if let image = thumbnails.image(for: slot.projectID) { Image(uiImage: image).resizable().scaledToFill() } }
            .clipShape(RoundedRectangle(cornerRadius: radius, style: .continuous))
            .task {
                if let project = appState.projects.first(where: { $0.projectID == slot.projectID }) {
                    await thumbnails.load(project: project, session: appState.session)
                }
            }
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
