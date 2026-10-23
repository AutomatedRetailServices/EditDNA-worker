import SwiftUI

/// Create tab start screen (White Indigo, Figma 12_Prototype 1164:2 "Create v3 ✅ ELEGIDO"): no plan meter, no Script card.
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
            CS.wiPage.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    // 1) Header (Figma 1164:77): only the logo, on the left (~172 pt with its glow). No avatar.
                    BrandLogo(width: 172).padding(.leading, -11.5)
                        .frame(height: 67)
                        .padding(.top, 2)

                    // 2) Greeting, with room under the logo.
                    Text(firstName.isEmpty ? L10n.tr("Hi 👋") : L10n.tr("Hi \(firstName) 👋"))
                        .font(CS.inter(15, .medium))
                        .foregroundStyle(CS.wiSecondary)
                        .padding(.top, 32)
                    Text(L10n.tr("What are we promoting today?"))
                        .font(CS.inter(23, .heavy))
                        .foregroundStyle(CS.wiInk)
                        .lineLimit(1)
                        .minimumScaleFactor(0.75)
                        .padding(.top, 3)

                    // 3) 2 × 2 grid (Figma 1164:40, 1195:115, 1195:103, 1164:47).
                    VStack(spacing: 8) {
                        HStack(spacing: 8) {
                            gridTile(icon: "WIRecord", title: "Record", subtitle: "Camera + teleprompter", id: "hubRecord") {
                                askPromoting(.record, onRecord)
                            }
                            gridTile(icon: "WIMic", title: "Faceless", subtitle: "Just your voice and product", id: "hubFaceless") {
                                // The Faceless flow is not built yet: the existing "Coming soon" notice.
                                askPromoting(.faceless) { voiceOverSoon = true }
                            }
                        }
                        HStack(spacing: 8) {
                            gridTile(icon: "WISalesMix", title: "Sales Mix", subtitle: "Multiple: hooks + bodies + CTAs for the same product",
                                     id: "hubMix") { askPromoting(.salesMix, onMix) }
                            gridTile(icon: "WIPlanCal", title: "Plan my week", subtitle: "Post at your times", id: "hubPlanWeek") {
                                AppRouter.shared.pending = .planWeek
                                selectTab(.plan)
                            }
                        }
                    }
                    .padding(.top, 18)

                    // 4) Upload dropzone, 16 pt under the grid.
                    uploadCard.padding(.top, 16)

                    // 5) Recent
                    HStack(alignment: .firstTextBaseline) {
                        Text(L10n.tr("Recent"))
                            .font(CS.sectionTitle)
                            .foregroundStyle(CS.wiInk)
                        Spacer()
                        Button(L10n.tr("See all")) { selectTab(.edits) }
                            .font(CS.inter(13, .semibold))
                            .foregroundStyle(CS.accent)
                            .accessibilityIdentifier("hubSeeAll")
                    }
                    .padding(.top, 20)

                    if recent.isEmpty {
                        Text(L10n.tr("Your cuts will show up here."))
                            .font(CS.inter(13, .medium))
                            .foregroundStyle(CS.wiSecondary)
                            .frame(maxWidth: .infinity, minHeight: 90)
                            .background(CS.wiCard, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
                            .padding(.top, 12)
                    } else {
                        HStack(spacing: 8) {
                            ForEach(Array(recent.enumerated()), id: \.element.id) { index, project in
                                Button { onOpenProject(project) } label: { recentTile(project, index: index) }
                                    .buttonStyle(.plain)
                            }
                            ForEach(recent.count..<3, id: \.self) { _ in Color.clear.frame(height: 90) }
                        }
                        .padding(.top, 12)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 16)
            }
        }
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

    /// Figma 1205:107 "Upload/Dropzone": light box (#F3F7FF) with a dashed blue border, the round gradient upload
    /// button, the title and line centred, and three blue gradient tiles (Single / Bulk / Combine).
    private var uploadCard: some View {
        VStack(spacing: 0) {
            Button { askPromoting(.upload(.single)) { onUpload(.single) } } label: {
                Image("WIUpload").renderingMode(.template).resizable()
                    .frame(width: 24, height: 24)
                    .foregroundStyle(Color(fixed: 0xFFFFFF))
                    .frame(width: 52, height: 52)
                    .background(CS.accentGradient, in: Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(L10n.tr("Upload videos"))
            .accessibilityIdentifier("hubUploadButton")
            Text(L10n.tr("Upload videos"))
                .font(CS.inter(22, .heavy))
                .foregroundStyle(CS.wiInk)
                .padding(.top, 10)
            Text(L10n.tr("AI cuts silences, mistakes & repeats."))
                .font(CS.inter(13, .medium))
                .foregroundStyle(CS.wiSecondary)
                .multilineTextAlignment(.center)
                .padding(.top, 2)
            HStack(spacing: 8) {
                tile(.single, icon: "WISingle", subtitle: "One video")
                tile(.bulk, icon: "WIBulk", subtitle: "Many at once")
                tile(.combine, icon: "WICombine", subtitle: "Join into one")
            }
            .padding(.top, 12)
        }
        .padding(.top, 16)
        .padding(.horizontal, 14)
        .padding(.bottom, 17)
        .frame(maxWidth: .infinity)
        .background(Color(fixed: 0xF3F7FF), in: RoundedRectangle(cornerRadius: 24, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 24, style: .continuous)
            .strokeBorder(Color(fixed: 0x1769FF, opacity: 0.45), style: StrokeStyle(lineWidth: 1.5, dash: [6, 5])))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("hubUpload")
    }

    /// Figma 1205:115 / 122 / 130: blue gradient tile, radius 18, soft blue shadow, white icon on 20 % white.
    private func tile(_ mode: CreateMode, icon: String, subtitle: String) -> some View {
        Button { askPromoting(.upload(mode)) { onUpload(mode) } } label: {
            VStack(alignment: .leading, spacing: 0) {
                Image(icon).renderingMode(.template).resizable()
                    .frame(width: 18, height: 18)
                    .foregroundStyle(Color(fixed: 0xFFFFFF))
                    .frame(width: 32, height: 32)
                    .background(Color(fixed: 0xFFFFFF, opacity: 0.2), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                Spacer(minLength: 6)
                Text(L10n.tr(mode.title))
                    .font(CS.inter(14, .bold))
                    .foregroundStyle(Color(fixed: 0xFFFFFF))
                    .lineLimit(1).minimumScaleFactor(0.8)
                Text(L10n.tr(subtitle))
                    .font(CS.inter(11, .medium))
                    .foregroundStyle(Color(fixed: 0xFFFFFF, opacity: 0.85))
                    .lineLimit(1).minimumScaleFactor(0.75)
                    .padding(.top, 2)
            }
            .padding(12)
            .frame(maxWidth: .infinity, minHeight: 94, maxHeight: 94, alignment: .topLeading)
            .background(CS.accentGradient, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .shadow(color: Color(fixed: 0x1769FF, opacity: 0.25), radius: 7, x: 0, y: 6)
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("hubMode_\(mode.rawValue)")
    }

    /// Figma 1164:40 "Tile": 171 × 122, radius 20, #F8F9FB, 38 pt #EEF3FF icon box with the blue icon, Bold 15, Medium 11.
    private func gridTile(icon: String, title: String, subtitle: String, id: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 0) {
                Image(icon).renderingMode(.template).resizable()
                    .frame(width: 20, height: 20)
                    .foregroundStyle(CS.accent)
                    .frame(width: 38, height: 38)
                    .background(CS.accentSoft, in: RoundedRectangle(cornerRadius: 11, style: .continuous))
                Text(L10n.tr(title))
                    .font(CS.cardTitle)
                    .foregroundStyle(CS.wiInk)
                    .lineLimit(1).minimumScaleFactor(0.85)
                    .padding(.top, 8)
                Text(L10n.tr(subtitle))
                    .font(CS.inter(11, .medium))
                    .foregroundStyle(CS.wiSecondary)
                    .lineSpacing(1)
                    .lineLimit(2)
                    .minimumScaleFactor(0.85)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 2)
                Spacer(minLength: 0)
            }
            .padding(13)
            .frame(maxWidth: .infinity, minHeight: 122, maxHeight: 122, alignment: .topLeading)
            .background(CS.wiCard, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
            .contentShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(id)
    }

    /// "Recent" (Figma 1164:56-58): thumbnails, 90 high, radius 14.
    private func recentTile(_ project: Project, index: Int) -> some View {
        let isProcessing = Project.processingStates.contains(project.state)
        return RoundedRectangle(cornerRadius: 14, style: .continuous)
            .fill(CS.wiSurface2)          // White Indigo placeholder until the thumbnail loads
            .frame(height: 90)
            .overlay {
                if let image = thumbnails.image(for: project.projectID) {
                    Image(uiImage: image).resizable().scaledToFill()
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .task(id: project.state) { await thumbnails.load(project: project, session: appState.session) }
            .overlay {
                if isProcessing { ProgressView().tint(CS.accent) }
            }
            .accessibilityLabel(project.title)
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
