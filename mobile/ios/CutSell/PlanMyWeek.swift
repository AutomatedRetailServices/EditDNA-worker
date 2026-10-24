import SwiftUI
import UserNotifications

// MARK: - Plan my week · SIMPLE (Figma 12_Prototype 1258:3101, PO 24 Oct)
//
// One screen: she taps her Ready videos in the order she wants to post them (1, 2, 3…), checks the four settings
// (Per day · Between · Days · Starts) and "Mix products automatically", and "Build plan · N videos" puts each video on a
// day and a time. The plan is written with the server's existing endpoints (no server change): every video goes to its
// day and time with /v1/plan/slots/move (a "pinned" slot), then the posting times / days / count are saved and the week
// is rebuilt, which keeps every pinned slot and adds nothing else.

/// A Ready video she can plan.
struct PlanPick: Identifiable, Hashable {
    let projectID: String
    let brand: String?
    let product: String?
    var id: String { projectID }
    /// The tile's name and the filter chip (Figma: the brand, e.g. "CeraVe").
    var label: String { brand ?? product ?? L10n.tr("No product") }
    /// Same product = same key (for "Mix products automatically").
    var productKey: String { [brand, product].compactMap { $0 }.joined(separator: "·").lowercased() }
}

/// Her four settings, remembered on this iPhone.
struct PlanMyWeekSettings: Codable, Equatable {
    var perDay = 4
    var fromMinutes = 9 * 60
    var toMinutes = 21 * 60
    var days: [Int] = [0, 1, 2, 3, 4, 5]          // Monday = 0 … Saturday = 5
    var startsTomorrow = true
    var start: String? = nil                       // yyyy-MM-dd when not "Tomorrow"
    var mix = false

    static let key = "cutsell.plan.simple"
    static func load() -> PlanMyWeekSettings {
        guard let data = UserDefaults.standard.data(forKey: key),
              let value = try? JSONDecoder().decode(PlanMyWeekSettings.self, from: data) else { return PlanMyWeekSettings() }
        return value
    }
    func save() { if let data = try? JSONEncoder().encode(self) { UserDefaults.standard.set(data, forKey: Self.key) } }

    var startDay: String {
        if !startsTomorrow, let start, start >= PlanDay.iso.string(from: Date()) { return start }
        return PlanGoalView.tomorrow()
    }
}

enum PlanScheduler {
    /// The day's posting times, spread evenly from `from` to `to` (one = `from`), on 5-minute steps, "HH:mm".
    static func times(perDay: Int, from: Int, to: Int) -> [String] {
        let n = max(1, perDay)
        let span = max(0, to - from)
        var out: [String] = []
        for i in 0..<n {
            let raw = n == 1 ? Double(from) : Double(from) + Double(span) * Double(i) / Double(n - 1)
            var minutes = Int((raw / 5).rounded()) * 5
            minutes = min(23 * 60 + 55, max(0, minutes))
            let text = String(format: "%02d:%02d", minutes / 60, minutes % 60)
            if !out.contains(text) { out.append(text) }
        }
        return out
    }

    /// "Mix products automatically": keeps her order, but never the same product twice in a row when another one is left.
    static func mixed(_ picks: [PlanPick]) -> [PlanPick] {
        var left = picks
        var out: [PlanPick] = []
        while !left.isEmpty {
            let previous = out.last?.productKey
            let index = left.firstIndex { $0.productKey != previous || previous == nil } ?? 0
            out.append(left.remove(at: index))
        }
        return out
    }

    /// Each video's day and time: her days from `start`, the day's times in order, skipping a time already taken.
    static func place(_ picks: [PlanPick], settings: PlanMyWeekSettings, taken: Set<String> = [])
        -> [(pick: PlanPick, date: String, time: String)] {
        let ordered = settings.mix ? mixed(picks) : picks
        let times = Self.times(perDay: settings.perDay, from: settings.fromMinutes, to: settings.toMinutes)
        let days = settings.days.isEmpty ? Array(0..<7) : settings.days
        let calendar = Calendar(identifier: .gregorian)
        guard var day = PlanDay.date(settings.startDay) else { return [] }
        var out: [(pick: PlanPick, date: String, time: String)] = []
        var queue = ordered[...]
        var guardDays = 0
        while !queue.isEmpty, guardDays < 400 {
            let weekday = (calendar.component(.weekday, from: day) + 5) % 7
            if days.contains(weekday) {
                let key = PlanDay.iso.string(from: day)
                for time in times where !taken.contains("\(key) \(time)") {
                    guard let pick = queue.first else { break }
                    out.append((pick, key, time))
                    queue = queue.dropFirst()
                }
            }
            day = calendar.date(byAdding: .day, value: 1, to: day) ?? day
            guardDays += 1
        }
        return out
    }

    /// "Mon 26 → Tue 27 · 4 a day · reminders on"
    static func summary(_ placed: [(pick: PlanPick, date: String, time: String)], perDay: Int, reminders: Bool) -> String {
        let range: String
        if let first = placed.first?.date, let last = placed.last?.date {
            range = first == last ? PlanGoalView.dayLabel(first) : "\(PlanGoalView.dayLabel(first)) → \(PlanGoalView.dayLabel(last))"
        } else {
            range = PlanGoalView.dayLabel(PlanGoalView.tomorrow())
        }
        return "\(range) · \(perDay) a day · reminders \(reminders ? "on" : "off")"
    }

    static func clock(minutes: Int) -> String {
        PlanDay.clock(String(format: "%02d:%02d", minutes / 60, minutes % 60))
    }

    /// "Mon – Sat", "Every day", "Mon, Wed, Fri"
    static func daysText(_ days: [Int]) -> String {
        let names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map { L10n.tr($0) }
        let sorted = Array(Set(days)).sorted()
        if sorted.count == 7 { return L10n.tr("Every day") }
        if sorted.count > 2, sorted.last! - sorted.first! == sorted.count - 1 {
            return "\(names[sorted.first!]) – \(names[sorted.last!])"
        }
        return sorted.map { names[$0] }.joined(separator: ", ")
    }
}

struct PlanMyWeekView: View {
    /// "Build plan" done: This week opens.
    let onBuilt: () -> Void
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var store = PlanWeekStore.shared
    @ObservedObject private var catalog = CatalogStore.shared
    @ObservedObject private var thumbnails = ThumbnailStore.shared
    @State private var chosen: [String] = []
    @State private var filter: String?
    @State private var settings = PlanMyWeekSettings.load()
    @State private var editing: Setting?
    @State private var building = false
    @State private var errorMessage: String?

    enum Setting: String, Identifiable { case perDay, between, days, starts; var id: String { rawValue } }

    /// Her Ready videos; the ones she left with "Decide later" first.
    private var videos: [PlanPick] {
        let later = PlanLater.all
        return store.plannable(projects: appState.projects, catalog: catalog).enumerated().sorted { a, b in
            let x = later.firstIndex(of: a.element.projectID) ?? Int.max, y = later.firstIndex(of: b.element.projectID) ?? Int.max
            return x == y ? a.offset < b.offset : x < y
        }.map(\.element)
    }
    /// Already in a plan (not posted yet): grey, can't be tapped.
    private var scheduled: Set<String> {
        Set((store.state?.week?.slots ?? []).filter { $0.postedAt == nil && $0.date >= (store.state?.today ?? "") }.map(\.projectID))
    }
    private var shown: [PlanPick] { videos.filter { filter == nil || $0.label == filter } }
    private var brands: [String] { Array(NSOrderedSet(array: videos.map(\.label))).compactMap { $0 as? String } }
    private var picks: [PlanPick] { chosen.compactMap { id in videos.first { $0.projectID == id } } }
    private var placed: [(pick: PlanPick, date: String, time: String)] {
        PlanScheduler.place(picks, settings: settings, taken: takenTimes)
    }
    private var takenTimes: Set<String> {
        Set((store.state?.week?.slots ?? []).filter { $0.postedAt == nil }.map { "\($0.date) \($0.time)" })
    }
    private var reminders: Bool { store.state?.prefs.remind ?? true }

    var body: some View {
        ZStack(alignment: .bottom) {
            CS.wiPage.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    PlanBackButton(title: "Plan") { dismiss() }.padding(.top, 8)
                    HStack(alignment: .firstTextBaseline) {
                        Text(L10n.tr("Plan my week")).font(CS.inter(28, .bold)).foregroundStyle(CS.wiInk)
                        Spacer()
                        Button(L10n.tr("Select all")) {
                            for v in shown where !scheduled.contains(v.projectID) && !chosen.contains(v.projectID) { chosen.append(v.projectID) }
                        }
                        .font(CS.inter(15, .semibold)).foregroundStyle(CS.accent)
                        .accessibilityIdentifier("pmwSelectAll")
                    }
                    .padding(.top, 12)
                    Text(L10n.tr("Tap videos in the order you want to post."))
                        .font(CS.inter(14)).foregroundStyle(CS.wiSecondary).padding(.top, 4)
                    chips.padding(.top, 14)
                    grid.padding(.top, 12)
                    if videos.isEmpty {
                        Text(L10n.tr("No Ready videos yet. Finished cuts show here."))
                            .font(CS.inter(13, .medium)).foregroundStyle(CS.wiSecondary)
                            .frame(maxWidth: .infinity, minHeight: 120)
                            .accessibilityIdentifier("pmwEmpty")
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 300)
            }
            panel
        }
        .toolbar(.hidden, for: .navigationBar)
        .wiStatusBarCover()
        .task {
            await store.refresh(session: appState.session)
            await catalog.refresh(session: appState.session, projects: appState.projects)
        }
        .onChange(of: settings) { _, value in value.save() }
        .sheet(item: $editing) { which in
            PlanSettingSheet(which: which, settings: $settings)
                .presentationDetents([.height(which == .days ? 360 : 400)])
                .presentationBackground(CS.wiPage)
        }
        .alert(L10n.tr("CutSell"), isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button(L10n.tr("OK"), role: .cancel) {}
        } message: { Text(L10n.tr(errorMessage ?? "")) }
        .accessibilityIdentifier("planMyWeekSimple")
    }

    private var chips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                chip(L10n.tr("All products"), on: filter == nil) { filter = nil }
                ForEach(brands, id: \.self) { name in chip(name, on: filter == name) { filter = name } }
            }
        }
    }

    private func chip(_ title: String, on: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title).font(CS.inter(14, .semibold)).foregroundStyle(on ? Color(fixed: 0xFFFFFF) : CS.wiInk)
                .padding(.horizontal, 14).frame(height: 34)
                .background(on ? CS.wiInk : CS.wiSurface2, in: Capsule())
                .overlay(Capsule().stroke(on ? .clear : CS.wiBorder, lineWidth: 1))
        }
        .buttonStyle(.plain)
    }

    private var grid: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 10), count: 3), spacing: 10) {
            ForEach(shown) { video in tile(video) }
        }
    }

    private func tile(_ video: PlanPick) -> some View {
        let order = chosen.firstIndex(of: video.projectID).map { $0 + 1 }
        let blocked = scheduled.contains(video.projectID)
        return Button {
            guard !blocked else { return }
            if let i = chosen.firstIndex(of: video.projectID) { chosen.remove(at: i) } else { chosen.append(video.projectID) }
        } label: {
            ZStack(alignment: .bottomLeading) {
                Color(fixed: 0xE9ECF1)
                if let image = thumbnails.image(for: video.projectID) { Image(uiImage: image).resizable().scaledToFill() }
                LinearGradient(colors: [.clear, Color(fixed: 0x000000, opacity: 0.55)], startPoint: .center, endPoint: .bottom)
                Text(video.label).font(CS.inter(12, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF)).lineLimit(1)
                    .padding(8)
            }
            .aspectRatio(0.72, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(order != nil ? CS.accent : .clear, lineWidth: 3))
            .overlay(alignment: .topTrailing) {
                Group {
                    if let order {
                        Text(verbatim: "\(order)").font(CS.inter(12, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                            .frame(width: 24, height: 24).background(CS.accent, in: Circle())
                    } else if !blocked {
                        Circle().fill(Color(fixed: 0xFFFFFF, opacity: 0.85)).frame(width: 24, height: 24)
                    }
                }
                .padding(7)
            }
            .overlay { if blocked { Color(fixed: 0xFFFFFF, opacity: 0.6).clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous)) } }
        }
        .buttonStyle(.plain)
        .disabled(blocked)
        .accessibilityIdentifier("pmwVideo")
        .accessibilityValue(order.map { "\($0)" } ?? (blocked ? "planned" : ""))
        .task { if let project = appState.projects.first(where: { $0.projectID == video.projectID }) {
            await thumbnails.load(project: project, session: appState.session) } }
    }

    /// Figma: the fixed panel with the four settings, the switch and the button.
    private var panel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                setting("Per day", "\(settings.perDay) video\(settings.perDay == 1 ? "" : "s")", .perDay).frame(width: 110)
                setting("Between", "\(PlanScheduler.clock(minutes: settings.fromMinutes)) – \(PlanScheduler.clock(minutes: settings.toMinutes))", .between)
            }
            HStack(spacing: 8) {
                setting("Days", PlanScheduler.daysText(settings.days), .days)
                setting("Starts", startsText, .starts)
            }
            HStack {
                Text(L10n.tr("Mix products automatically")).font(CS.inter(14, .medium)).foregroundStyle(CS.wiInk)
                Spacer()
                CSSwitch(isOn: $settings.mix).accessibilityIdentifier("pmwMix")
            }
            .padding(.top, 2)
            Text(L10n.tr(settings.mix ? "On: never the same product twice in a row." : "Off: posts in the order you tapped (1, 2, 3…)."))
                .font(CS.inter(11)).foregroundStyle(CS.wiSecondary).padding(.top, -6)
            Button { Task { await build() } } label: {
                VStack(spacing: 2) {
                    if building { BrandRing(size: 22) } else {
                        Text(L10n.tr("Build plan · \(chosen.count) video\(chosen.count == 1 ? "" : "s")")).font(CS.inter(17, .bold))
                        Text(L10n.tr(PlanScheduler.summary(placed, perDay: settings.perDay, reminders: reminders)))
                            .font(CS.inter(11, .medium)).opacity(0.9)
                            .accessibilityIdentifier("pmwSummary")
                    }
                }
                .foregroundStyle(Color(fixed: 0xFFFFFF))
                .frame(maxWidth: .infinity).frame(height: 56)
                .background(CS.accent.opacity(chosen.isEmpty ? 0.4 : 1), in: Capsule())
            }
            .buttonStyle(.plain)
            .disabled(chosen.isEmpty || building)
            .padding(.top, 4)
            .accessibilityIdentifier("pmwBuild")
        }
        .padding(.horizontal, 20).padding(.top, 16).padding(.bottom, 10)
        .background {
            UnevenRoundedRectangle(topLeadingRadius: 24, topTrailingRadius: 24, style: .continuous)
                .fill(CS.wiPage).ignoresSafeArea(edges: .bottom)
                .shadow(color: Color(fixed: 0x0D1433, opacity: 0.08), radius: 10, y: -2)
        }
    }

    private var startsText: String {
        let day = settings.startDay
        if day == PlanGoalView.tomorrow() { return L10n.tr("Tomorrow") }
        if day == PlanDay.iso.string(from: Date()) { return L10n.tr("Today") }
        return PlanGoalView.dayLabel(day)
    }

    private func setting(_ title: String, _ value: String, _ which: Setting) -> some View {
        Button { editing = which } label: {
            VStack(alignment: .leading, spacing: 3) {
                Text(L10n.tr(title)).font(CS.inter(12, .medium)).foregroundStyle(CS.wiSecondary)
                HStack {
                    Text(L10n.tr(value)).font(CS.inter(16, .semibold)).foregroundStyle(CS.wiInk).lineLimit(1).minimumScaleFactor(0.7)
                    Spacer(minLength: 4)
                    Image(systemName: "chevron.down").font(.system(size: 11, weight: .semibold)).foregroundStyle(CS.wiSecondary)
                }
            }
            .padding(.horizontal, 12).frame(maxWidth: .infinity, alignment: .leading).frame(height: 56)
            .background(CS.wiCard, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("pmwSetting_\(which.rawValue)")
    }

    private func build() async {
        building = true
        defer { building = false }
        do {
            try await store.schedule(placed.map { ($0.pick.projectID, $0.date, $0.time) },
                                     times: PlanScheduler.times(perDay: settings.perDay, from: settings.fromMinutes, to: settings.toMinutes),
                                     days: settings.days, start: settings.startDay, session: appState.session)
            if store.state?.prefs.remind ?? true {
                _ = try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge])
            }
            PostReminders.shared.schedule(store.state)
            PostPrefetch.schedule()
            for id in chosen { PlanLater.remove(id) }
            chosen = []
            onBuilt()
        } catch {
            errorMessage = "Your plan couldn’t be built. Check your connection and try again."
        }
    }
}

/// The selector of one setting (Per day · Between · Days · Starts).
struct PlanSettingSheet: View {
    let which: PlanMyWeekView.Setting
    @Binding var settings: PlanMyWeekSettings
    @Environment(\.dismiss) private var dismiss
    @State private var from = Date()
    @State private var to = Date()
    @State private var start = Date()

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Capsule().fill(Color(fixed: 0xD5D9E2)).frame(width: 40, height: 5).frame(maxWidth: .infinity).padding(.top, 10)
            Text(L10n.tr(title)).font(CS.inter(20, .bold)).foregroundStyle(CS.wiInk)
            switch which {
            case .perDay:
                HStack {
                    round("−", enabled: settings.perDay > 1) { settings.perDay -= 1 }
                        .accessibilityIdentifier("perDayMinus")
                    Spacer()
                    VStack(spacing: 2) {
                        Text(verbatim: "\(settings.perDay)").font(CS.inter(34, .bold)).foregroundStyle(CS.wiInk)
                            .accessibilityIdentifier("perDayValue")
                        Text(L10n.tr("videos a day")).font(CS.inter(12)).foregroundStyle(CS.wiSecondary)
                    }
                    Spacer()
                    round("+", enabled: settings.perDay < 50) { settings.perDay += 1 }
                        .accessibilityIdentifier("perDayPlus")
                }
                .padding(16).wiCard(shadow: false)
                Text(L10n.tr("From 1 to 50 a day. The times are spread evenly inside “Between”."))
                    .font(CS.inter(12)).foregroundStyle(CS.wiSecondary)
            case .between:
                HStack(spacing: 10) {
                    timeBox("From", $from)
                    timeBox("To", $to)
                }
                Text(L10n.tr("\(settings.perDay) a day: \(PlanScheduler.times(perDay: settings.perDay, from: minutes(from), to: max(minutes(from), minutes(to))).prefix(6).map(PlanDay.clock).joined(separator: " · "))\(settings.perDay > 6 ? " …" : "")"))
                    .font(CS.inter(12)).foregroundStyle(CS.wiSecondary)
                    .accessibilityIdentifier("betweenPreview")
            case .days:
                HStack(spacing: 6) {
                    ForEach(0..<7, id: \.self) { d in
                        let on = settings.days.contains(d)
                        Button {
                            if on { if settings.days.count > 1 { settings.days.removeAll { $0 == d } } } else { settings.days = (settings.days + [d]).sorted() }
                        } label: {
                            Text(L10n.tr(["M", "T", "W", "T", "F", "S", "S"][d])).font(CS.inter(14, .bold))
                                .foregroundStyle(on ? Color(fixed: 0xFFFFFF) : CS.wiSecondary)
                                .frame(width: 42, height: 42)
                                .background(on ? CS.accent : CS.wiSurface2, in: Circle())
                                .overlay(Circle().stroke(on ? .clear : CS.wiBorder, lineWidth: 1))
                        }
                        .buttonStyle(.plain)
                        .frame(maxWidth: .infinity)
                        .accessibilityIdentifier("daysPick_\(d)")
                    }
                }
            case .starts:
                DatePicker("", selection: $start, in: Calendar.current.startOfDay(for: Date())..., displayedComponents: .date)
                    .datePickerStyle(.compact).labelsHidden().tint(CS.accent)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button(L10n.tr("Tomorrow")) { settings.startsTomorrow = true; settings.start = nil; dismiss() }
                    .font(CS.inter(14, .semibold)).foregroundStyle(CS.accent)
            }
            Spacer(minLength: 0)
            Button {
                save()
                dismiss()
            } label: {
                Text(L10n.tr("Done")).font(CS.inter(16, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF))
                    .frame(maxWidth: .infinity).frame(height: 50).background(CS.accent, in: Capsule())
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("settingDone")
        }
        .padding(.horizontal, 20).padding(.bottom, 16)
        .environment(\.colorScheme, .light)
        .onAppear {
            from = date(settings.fromMinutes); to = date(settings.toMinutes)
            start = PlanDay.date(settings.startDay) ?? Date()
        }
    }

    private var title: String {
        switch which {
        case .perDay: return "Per day"
        case .between: return "Between"
        case .days: return "Days"
        case .starts: return "Starts"
        }
    }

    private func save() {
        switch which {
        case .between:
            let a = minutes(from), b = minutes(to)
            settings.fromMinutes = min(a, b); settings.toMinutes = max(a, b)
        case .starts:
            let day = PlanDay.iso.string(from: start)
            if day == PlanGoalView.tomorrow() { settings.startsTomorrow = true; settings.start = nil }
            else { settings.startsTomorrow = false; settings.start = day }
        default: break
        }
    }

    private func timeBox(_ label: String, _ value: Binding<Date>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(L10n.tr(label)).font(CS.inter(12, .medium)).foregroundStyle(CS.wiSecondary)
            DatePicker("", selection: value, displayedComponents: .hourAndMinute)
                .datePickerStyle(.wheel).labelsHidden().frame(height: 120).clipped()
        }
        .frame(maxWidth: .infinity)
        .padding(8).wiCard(shadow: false)
    }

    private func round(_ text: String, enabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(verbatim: text).font(CS.inter(22, .bold)).foregroundStyle(text == "+" ? Color(fixed: 0xFFFFFF) : CS.wiInk)
                .frame(width: 44, height: 44)
                .background(text == "+" ? CS.accent : CS.wiSurface2, in: Circle())
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.4)
    }

    private func minutes(_ d: Date) -> Int {
        let c = Calendar.current.dateComponents([.hour, .minute], from: d)
        return (c.hour ?? 9) * 60 + (c.minute ?? 0)
    }
    private func date(_ m: Int) -> Date {
        Calendar.current.date(bySettingHour: m / 60, minute: m % 60, second: 0, of: Date()) ?? Date()
    }
}

// MARK: - This week · a video's sheet (Figma 1257:3329)

struct SlotActionsSheet: View {
    let slot: PlanSlot
    let onSwap: () -> Void
    let onChange: () -> Void
    let onRemove: () -> Void
    @EnvironmentObject private var appState: AppState
    @ObservedObject private var thumbnails = ThumbnailStore.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Capsule().fill(Color(fixed: 0xD5D9E2)).frame(width: 40, height: 5).frame(maxWidth: .infinity).padding(.top, 10)
            Text(verbatim: "\(PlanDay.short(slot.date)) · \(PlanDay.clock(slot.time))")
                .font(CS.inter(14, .medium)).foregroundStyle(CS.wiSecondary).padding(.top, 6)
                .accessibilityIdentifier("slotSheetWhen")
            HStack(spacing: 12) {
                Color(fixed: 0xE9ECF1).frame(width: 40, height: 52)
                    .overlay { if let image = thumbnails.image(for: slot.projectID) { Image(uiImage: image).resizable().scaledToFill() } }
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                VStack(alignment: .leading, spacing: 3) {
                    Text(L10n.tr(slot.product ?? "Video")).font(CS.inter(17, .bold)).foregroundStyle(CS.wiInk).lineLimit(1)
                    Text([slot.brand, ProductDetailView.length(slot.durationSec)].compactMap { $0 }.joined(separator: " · "))
                        .font(CS.inter(13)).foregroundStyle(CS.wiSecondary)
                }
                Spacer()
            }
            .padding(12)
            .wiCard(radius: 14, shadow: false)
            row("arrow.left.arrow.right", "Swap video", id: "slotSwap", action: onSwap)
            row("clock", "Change day or time", id: "slotChange", action: onChange)
            row("trash", "Remove from plan", id: "slotRemove", red: true, action: onRemove)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 20)
        .environment(\.colorScheme, .light)
        .accessibilityIdentifier("slotSheet")
    }

    private func row(_ icon: String, _ title: String, id: String, red: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 14) {
                Image(systemName: icon).font(.system(size: 17, weight: .medium)).frame(width: 24)
                Text(L10n.tr(title)).font(CS.inter(16, .semibold))
                Spacer()
                Image(systemName: "chevron.right").font(.system(size: 13, weight: .semibold)).foregroundStyle(CS.wiSecondary)
            }
            .foregroundStyle(red ? CS.wiError : CS.wiInk)
            .padding(.horizontal, 16).frame(height: 56)
            .background(CS.wiPage, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(id)
    }
}

/// "Swap video": another Ready video takes this one's day and time.
struct SwapVideoSheet: View {
    let slot: PlanSlot
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var store = PlanWeekStore.shared
    @ObservedObject private var catalog = CatalogStore.shared
    @ObservedObject private var thumbnails = ThumbnailStore.shared
    @State private var errorMessage: String?

    private var options: [PlanPick] {
        let planned = Set((store.state?.week?.slots ?? []).map(\.projectID))
        return store.plannable(projects: appState.projects, catalog: catalog).filter { !planned.contains($0.projectID) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Capsule().fill(Color(fixed: 0xD5D9E2)).frame(width: 40, height: 5).frame(maxWidth: .infinity).padding(.top, 10)
            Text(L10n.tr("Swap video")).font(CS.inter(20, .bold)).foregroundStyle(CS.wiInk)
            Text(verbatim: "\(PlanDay.short(slot.date)) · \(PlanDay.clock(slot.time))").font(CS.inter(13)).foregroundStyle(CS.wiSecondary)
            ScrollView {
                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 10), count: 3), spacing: 10) {
                    ForEach(options) { video in
                        Button { Task { await swap(to: video) } } label: {
                            ZStack(alignment: .bottomLeading) {
                                Color(fixed: 0xE9ECF1)
                                if let image = thumbnails.image(for: video.projectID) { Image(uiImage: image).resizable().scaledToFill() }
                                LinearGradient(colors: [.clear, Color(fixed: 0x000000, opacity: 0.55)], startPoint: .center, endPoint: .bottom)
                                Text(video.label).font(CS.inter(12, .bold)).foregroundStyle(Color(fixed: 0xFFFFFF)).lineLimit(1).padding(8)
                            }
                            .aspectRatio(0.72, contentMode: .fit)
                            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("swapOption")
                    }
                }
                if options.isEmpty {
                    Text(L10n.tr("Every Ready video is already in your plan.")).font(CS.inter(13)).foregroundStyle(CS.wiSecondary)
                        .frame(maxWidth: .infinity, minHeight: 80)
                }
            }
            if let errorMessage { Text(L10n.tr(errorMessage)).font(CS.inter(12)).foregroundStyle(CS.wiError) }
        }
        .padding(.horizontal, 20)
        .environment(\.colorScheme, .light)
        .accessibilityIdentifier("swapSheet")
    }

    private func swap(to video: PlanPick) async {
        do {
            try await store.add(projectID: video.projectID, date: slot.date, time: slot.time, replacing: slot.projectID,
                                session: appState.session)
            dismiss()
        } catch { errorMessage = "Couldn’t swap it. Please try again." }
    }
}

/// This week › Settings: the reminders (Plan my week opens from Plan, not from here).
struct PlanWeekSettingsSheet: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var store = PlanWeekStore.shared
    @State private var remind = true

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Capsule().fill(Color(fixed: 0xD5D9E2)).frame(width: 40, height: 5).frame(maxWidth: .infinity).padding(.top, 10)
            Text(L10n.tr("Settings")).font(CS.inter(20, .bold)).foregroundStyle(CS.wiInk)
            HStack {
                Text(L10n.tr("Remind me at each time")).font(CS.inter(15, .semibold)).foregroundStyle(CS.wiInk)
                Spacer()
                CSSwitch(isOn: $remind).accessibilityIdentifier("weekRemind")
            }
            .padding(.horizontal, 15).frame(height: 54).wiCard(radius: 14, shadow: false)
            Text(L10n.tr("A notification at each video’s time, to post it in one tap."))
                .font(CS.inter(12)).foregroundStyle(CS.wiSecondary)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 20)
        .environment(\.colorScheme, .light)
        .onAppear { remind = store.state?.prefs.remind ?? true }
        .onChange(of: remind) { _, on in Task { try? await store.setReminders(on, session: appState.session) } }
    }
}

/// Where she will post each planned video (Export › Add to Plan › Post on). Kept on this iPhone; TikTok when not chosen.
/// CutSell never posts by itself: at the time it reminds her and opens that app with the video in Photos and the caption
/// copied (posting for her needs each network's permission; TikTok's needs the App Store-approved app).
enum PostNetwork: String, CaseIterable, Identifiable {
    case tiktok = "TikTok", instagram = "Instagram", facebook = "Facebook"
    var id: String { rawValue }
    private static let key = "cutsell.plan.networks"

    static func of(_ projectID: String) -> PostNetwork {
        let all = UserDefaults.standard.dictionary(forKey: key) as? [String: String] ?? [:]
        return all[projectID].flatMap(PostNetwork.init(rawValue:)) ?? .tiktok
    }
    static func set(_ network: PostNetwork, for projectID: String) {
        var all = UserDefaults.standard.dictionary(forKey: key) as? [String: String] ?? [:]
        all[projectID] = network.rawValue
        UserDefaults.standard.set(all, forKey: key)
    }
    /// Instagram / Facebook open with their own app (the video is in Photos, the caption copied).
    var appURLs: [String] {
        switch self {
        case .tiktok: return []
        case .instagram: return ["instagram://library", "instagram://app", "https://www.instagram.com"]
        case .facebook: return ["fb://", "https://www.facebook.com"]
        }
    }
}

/// "Decide later" (Export › Add to Plan): ready to schedule, without a day yet. Kept on this iPhone.
enum PlanLater {
    private static let key = "cutsell.plan.later"
    static var all: [String] { UserDefaults.standard.stringArray(forKey: key) ?? [] }
    static func add(_ projectID: String) {
        var list = all.filter { $0 != projectID }; list.insert(projectID, at: 0)
        UserDefaults.standard.set(list, forKey: key)
    }
    static func remove(_ projectID: String) { UserDefaults.standard.set(all.filter { $0 != projectID }, forKey: key) }
}
