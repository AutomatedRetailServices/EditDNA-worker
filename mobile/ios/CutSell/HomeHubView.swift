import SwiftUI

/// Create tab start screen (White Indigo, Figma 12_Prototype 1164:2 "Create v3 ✅ ELEGIDO"): no plan meter, no Script card.
struct HomeHubView: View {
    let onUpload: (CreateMode) -> Void
    var onMix: () -> Void = {}
    let onRecord: () -> Void
    let onOpenProject: (Project) -> Void
    let selectTab: (MainTab) -> Void

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
                    // Header (Figma 1164:2): logo left (~172 pt with its glow), round avatar with the initial right.
                    HStack(alignment: .center) {
                        BrandLogo(width: 172).padding(.leading, -11.5)
                        Spacer()
                        Button { selectTab(.you) } label: {
                            Text(verbatim: String(firstName.first ?? "C").uppercased())
                                .font(CS.inter(16, .bold))
                                .foregroundStyle(CS.accent)
                                .frame(width: 40, height: 40)
                                .background(CS.accentSoft, in: Circle())
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("hubAvatar")
                    }
                    .frame(height: 67)
                    .padding(.top, 2)

                    Text(firstName.isEmpty ? L10n.tr("Hi 👋") : L10n.tr("Hi \(firstName) 👋"))
                        .font(CS.inter(15, .medium))
                        .foregroundStyle(CS.wiSecondary)
                        .padding(.top, 6)
                    Text(L10n.tr("What are we selling today?"))
                        .font(CS.inter(32, .heavy))
                        .tracking(CS.screenTitleTracking)
                        .lineSpacing(0)
                        .foregroundStyle(CS.wiInk)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 4)

                    uploadCard.padding(.top, 14)

                    Text(L10n.tr("Create more"))
                        .font(CS.sectionTitle)
                        .foregroundStyle(CS.wiInk)
                        .padding(.top, 18)
                    salesMixCard.padding(.top, 10)
                    HStack(spacing: 8) {
                        smallCard(icon: "WIRecord", title: "Record", subtitle: "Camera + teleprompter", id: "hubRecord", action: onRecord)
                        smallCard(icon: "WIPlanCal", title: "Plan my week", subtitle: "Post at your times", id: "hubPlanWeek") {
                            selectTab(.plan)
                        }
                    }
                    .padding(.top, 8)
                    // "Faceless video" stays hidden until the flow exists.
                    if Self.facelessVideoReady {
                        smallCard(icon: "WIRecord", title: "Faceless video", subtitle: "Product photos or clips + your voice. No camera.",
                                  id: "hubFaceless") { voiceOverSoon = true }
                            .padding(.top, 8)
                    }

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
                    .padding(.top, 18)

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

    /// Figma 1164:92 "Upload/Tiles": blue gradient card, its round upload button (Single) and three white tiles.
    private var uploadCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L10n.tr("Upload videos"))
                        .font(CS.inter(22, .heavy))
                        .foregroundStyle(.white)
                    Text(L10n.tr("AI cuts silences, mistakes & repeats."))
                        .font(CS.inter(13, .medium))
                        .foregroundStyle(Color.white.opacity(0.82))
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 8)
                Button { onUpload(.single) } label: {
                    Image("WIUpload").renderingMode(.template).resizable()
                        .frame(width: 22, height: 22)
                        .foregroundStyle(.white)
                        .frame(width: 42, height: 42)
                        .background(Color.white.opacity(0.18), in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(L10n.tr("Upload videos"))
                .accessibilityIdentifier("hubUploadButton")
            }
            HStack(spacing: 8) {
                tile(.single, icon: "WISingle", subtitle: "One video")
                tile(.bulk, icon: "WIBulk", subtitle: "Many at once")
                tile(.combine, icon: "WICombine", subtitle: "Join into one")
            }
            .padding(.top, 18)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(CS.accentGradient, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("hubUpload")
    }

    private func tile(_ mode: CreateMode, icon: String, subtitle: String) -> some View {
        Button { onUpload(mode) } label: {
            VStack(alignment: .leading, spacing: 0) {
                Image(icon).renderingMode(.template).resizable()
                    .frame(width: 20, height: 20)
                    .foregroundStyle(CS.accent)
                    .frame(width: 36, height: 36)
                    .background(CS.accentSoft, in: RoundedRectangle(cornerRadius: 11, style: .continuous))
                Spacer(minLength: 10)
                Text(L10n.tr(mode.title))
                    .font(CS.inter(14, .bold))
                    .foregroundStyle(CS.wiInk)
                    .lineLimit(1).minimumScaleFactor(0.8)
                Text(L10n.tr(subtitle))
                    .font(CS.inter(11, .medium))
                    .foregroundStyle(CS.wiSecondary)
                    .lineLimit(1).minimumScaleFactor(0.8)
                    .padding(.top, 4)
            }
            .padding(12)
            .frame(maxWidth: .infinity, minHeight: 112, maxHeight: 112, alignment: .topLeading)
            .background(CS.wiPage, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("hubMode_\(mode.rawValue)")
    }

    /// Figma 1164:30 "Sales Mix" (full width): opens Sales Mix · Setup.
    private var salesMixCard: some View {
        Button(action: onMix) {
            HStack(spacing: 14) {
                Image("WISalesMix").renderingMode(.template).resizable()
                    .frame(width: 20, height: 20)
                    .foregroundStyle(.white)
                    .frame(width: 38, height: 38)
                    .background(CS.accent, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                VStack(alignment: .leading, spacing: 4) {
                    Text(L10n.tr("Sales Mix"))
                        .font(CS.cardTitle)
                        .foregroundStyle(CS.wiInk)
                    Text(L10n.tr("Record a few hooks, bodies & CTAs. We mix them into many videos."))
                        .font(CS.inter(11, .medium))
                        .foregroundStyle(CS.wiSecondary)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                Image("WIChevron").renderingMode(.template).resizable()
                    .frame(width: 18, height: 18)
                    .foregroundStyle(CS.accent)
            }
            .padding(.horizontal, 13)
            .padding(.vertical, 18)
            .frame(maxWidth: .infinity, minHeight: 96)
            .background(CS.wiCard, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("hubMix")
    }

    /// Figma 1164:40 / 1164:47: half-width card, white icon box, title, small line.
    private func smallCard(icon: String, title: String, subtitle: String, id: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 0) {
                Image(icon).renderingMode(.template).resizable()
                    .frame(width: 20, height: 20)
                    .foregroundStyle(CS.wiInk)
                    .frame(width: 38, height: 38)
                    .background(CS.wiPage, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
                Spacer(minLength: 8)
                Text(L10n.tr(title))
                    .font(CS.cardTitle)
                    .foregroundStyle(CS.wiInk)
                    .lineLimit(1).minimumScaleFactor(0.85)
                Text(L10n.tr(subtitle))
                    .font(CS.inter(11, .medium))
                    .foregroundStyle(CS.wiSecondary)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                    .padding(.top, 3)
            }
            .padding(13)
            .frame(maxWidth: .infinity, minHeight: 112, maxHeight: 112, alignment: .topLeading)
            .background(CS.wiCard, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(CS.wiBorder, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(id)
    }

    private static let tileColors: [UInt32] = [0x2A3F6B, 0x4A2A6B, 0x1F5A6B]

    /// "Recent" (Figma 1164:56-58): thumbnails, 90 high, radius 14.
    private func recentTile(_ project: Project, index: Int) -> some View {
        let isProcessing = Project.processingStates.contains(project.state)
        return RoundedRectangle(cornerRadius: 14, style: .continuous)
            .fill(LinearGradient(colors: [Color(fixed: Self.tileColors[index % 3]), Color(fixed: 0x0C1730)],
                                 startPoint: .topLeading, endPoint: .bottomTrailing))
            .frame(height: 90)
            .overlay {
                if let image = thumbnails.image(for: project.projectID) {
                    Image(uiImage: image).resizable().scaledToFill()
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .task(id: project.state) { await thumbnails.load(project: project, session: appState.session) }
            .overlay {
                if isProcessing { ProgressView().tint(.white) }
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
