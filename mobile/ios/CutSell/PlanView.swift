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

    enum PlanRoute: Hashable { case goal, week, day(String) }

    var body: some View {
        NavigationStack(path: $path) {
            home
                .navigationDestination(for: PlanRoute.self) { route in
                    switch route {
                    case .goal:
                        PlanGoalView(onBuilt: { path = [.week] }).environmentObject(appState)
                    case .week:
                        PlanWeekView(onNewMix: { showingMix = true }, onSettings: { path.append(.goal) }).environmentObject(appState)
                    case .day(let day):
                        PlanDayView(day: day).environmentObject(appState)
                    }
                }
        }
        .tint(CS.cyan)
        .task { await week.refresh(session: appState.session) }
        // A reminder was tapped: that day opens, ready to share.
        .onChange(of: week.requestedDay) { _, day in
            guard let day else { return }
            week.requestedDay = nil
            path = [.week, .day(day)]
        }
        .onAppear {
            if let day = week.requestedDay { week.requestedDay = nil; path = [.week, .day(day)] }
        }
    }

    private var home: some View {
        ZStack {
            CS.base.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    BrandLogo().frame(maxWidth: .infinity).padding(.top, 2)
                    Text("Plan").font(CS.inter(32, .bold)).foregroundStyle(CS.textPrimaryF).padding(.top, 20)
                    Text("Make videos in batches and plan what to post.")
                        .font(CS.inter(14)).foregroundStyle(Color(hex: 0x99A3BD)).padding(.top, 2)

                    HStack(spacing: 8) {
                        bigButton(icon: "⊞", title: "New Mix", subtitle: "Hooks × bodies × CTAs → many videos",
                                  highlighted: true, id: "planNewMix") { showingMix = true }
                        bigButton(icon: "⇪", title: "Bulk", subtitle: "Upload many videos at once",
                                  highlighted: false, id: "planBulk") { queue.requestedCreateMode = .bulk }
                    }
                    .padding(.top, 18)

                    HStack(alignment: .firstTextBaseline) {
                        Text("In progress").font(CS.inter(16, .semibold)).foregroundStyle(.white)
                        Spacer()
                        Button("See all in Edits ›") { openEdits(focus: nil) }
                            .font(CS.inter(12, .semibold)).foregroundStyle(CS.cyan)
                            .accessibilityIdentifier("planSeeAll")
                    }
                    .padding(.top, 18)
                    VStack(spacing: 8) {
                        ForEach(mixes.active) { mix in
                            Button { openEdits(focus: mix.mixID) } label: {
                                progressCard(tag: .mix, title: mix.product ?? "Mix",
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
                            Text("Nothing in progress. Start a Mix or a Bulk.")
                                .font(CS.inter(13)).foregroundStyle(Color(hex: 0x99A3BD))
                                .frame(maxWidth: .infinity, minHeight: 60)
                                .background(Color(hex: 0x0F1726), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                                .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color(hex: 0x33405C), lineWidth: 1))
                                .accessibilityIdentifier("planNothingInProgress")
                        }
                    }
                    .padding(.top, 10)

                    thisWeek.padding(.top, 18)
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

    private func bigButton(icon: String, title: String, subtitle: String, highlighted: Bool, id: String,
                           action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 0) {
                Text(icon).font(CS.inter(18, .bold)).foregroundStyle(.white)
                    .frame(width: 40, height: 40)
                    .background {
                        if highlighted {
                            RoundedRectangle(cornerRadius: 12, style: .continuous).fill(CS.brandFull)
                        } else {
                            RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Color(hex: 0x1F2638))
                        }
                    }
                Text(title).font(CS.inter(16, .bold)).foregroundStyle(.white).padding(.top, 12)
                Text(subtitle).font(CS.inter(11)).foregroundStyle(Color(hex: 0x99A3BD))
                    .fixedSize(horizontal: false, vertical: true).padding(.top, 4)
                Spacer(minLength: 0)
            }
            .padding(14.5)
            .frame(maxWidth: .infinity, minHeight: 132, maxHeight: 132, alignment: .topLeading)
            .background(Color(hex: 0x0F1726), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(highlighted ? Color(hex: 0x19E5FF) : Color(hex: 0x33405C), lineWidth: highlighted ? 1.5 : 1))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(id)
    }

    enum Tag { case mix, bulk }

    private func progressCard(tag: Tag, title: String, detail: String, ready: Int, total: Int, image: UIImage?) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Color(hex: 0x1A2236)
                .frame(width: 40, height: 52)
                .overlay { if let image { Image(uiImage: image).resizable().scaledToFill() } }
                .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 8) {
                    Text(tag == .mix ? "MIX" : "BULK").font(CS.inter(9, .bold)).foregroundStyle(.white)
                        .frame(width: tag == .mix ? 35 : 42, height: 18)
                        .background {
                            if tag == .mix { Capsule().fill(CS.brandFull) } else { Capsule().fill(Color(hex: 0x1F2638)) }
                        }
                    Text(title).font(CS.inter(14, .semibold)).foregroundStyle(.white).lineLimit(1)
                }
                Text(detail).font(CS.inter(11)).foregroundStyle(Color(hex: 0x99A3BD)).lineLimit(1).padding(.top, 6)
                HStack(spacing: 0) {
                    GeometryReader { proxy in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color(hex: 0x242B40))
                            Capsule().fill(CS.brandFull)
                                .frame(width: proxy.size.width * CGFloat(ready) / CGFloat(max(1, total)))
                        }
                    }
                    .frame(height: 6)
                    Text("\(ready)/\(total)").font(CS.inter(12, .semibold)).foregroundStyle(CS.cyan)
                        .frame(width: 50, alignment: .trailing)
                }
                .padding(.top, 8)
            }
        }
        .padding(.horizontal, 13)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, minHeight: 78, alignment: .leading)
        .background(Color(hex: 0x0F1726), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color(hex: 0x33405C), lineWidth: 1))
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
        name.locale = Locale(identifier: "en_US")
        name.dateFormat = "EEE"
        return VStack(alignment: .leading, spacing: 0) {
            Text("This week").font(CS.inter(16, .semibold)).foregroundStyle(.white)
            HStack(spacing: 6) {
                ForEach(Array(days.enumerated()), id: \.offset) { _, day in
                    let isToday = calendar.isDate(day, inSameDayAs: today)
                    let key = PlanDay.iso.string(from: day)
                    Button { if week.hasWeek, slots.contains(where: { $0.date == key }) { path = [.week, .day(key)] } } label: {
                        VStack(spacing: 2) {
                            Text(name.string(from: day)).font(CS.inter(11, .semibold)).foregroundStyle(.white)
                            Text("\(calendar.component(.day, from: day))").font(CS.inter(14, .bold)).foregroundStyle(.white)
                            HStack(spacing: 2) {
                                ForEach(slots.filter { $0.date == key }.prefix(4)) { slot in
                                    Image(week.dot(for: slot)).resizable().frame(width: 6, height: 6)
                                        .opacity(slot.postedAt == nil ? 1 : 0.35)
                                }
                            }
                            .frame(height: 6)
                        }
                        .frame(maxWidth: .infinity).frame(height: 62)
                        .background {
                            if isToday {
                                RoundedRectangle(cornerRadius: 14, style: .continuous).fill(CS.brandFull)
                            } else {
                                RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Color(hex: 0x0F1726))
                            }
                        }
                        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .stroke(isToday ? .clear : Color(hex: 0x33405C), lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.top, 11)
            Button { path = week.hasWeek ? [.week] : [.goal] } label: {
                VStack(alignment: .leading, spacing: 6) {
                    Text(cardTitle(todays: todays)).font(CS.inter(14, .semibold)).foregroundStyle(.white)
                        .accessibilityIdentifier("planWeekCardTitle")
                    Text(cardBody(slots: slots)).font(CS.inter(11)).foregroundStyle(Color(hex: 0x99A3BD))
                        .fixedSize(horizontal: false, vertical: true)
                    Text("Plan my week ›").font(CS.inter(12, .semibold)).foregroundStyle(CS.cyan)
                        .padding(.top, 2)
                }
                .padding(15)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(hex: 0x0F1726), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color(hex: 0x33405C), lineWidth: 1))
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
