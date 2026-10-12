import SwiftUI

/// Create tab start screen (Figma 03_Create "Create v2/Home Hub", node 1068:22).
/// The plan meter ("7 / 30 cuts · Plan") is left out until plans exist: no invented numbers.
struct HomeHubView: View {
    let onUpload: (CreateMode) -> Void
    let onRecord: () -> Void
    let onOpenProject: (Project) -> Void
    let selectTab: (MainTab) -> Void

    @EnvironmentObject private var appState: AppState
    @ObservedObject private var tracker = CutTracker.shared
    @State private var voiceOverSoon = false

    private var recent: [Project] { Array(appState.projects.prefix(3)) }

    var body: some View {
        ZStack {
            CS.base.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    HStack { BrandLogo(); Spacer() }
                        .padding(.top, 6)

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Create")
                            .font(CS.inter(30, .heavy))
                            .foregroundStyle(CS.textPrimaryF)
                        Text("Turn your raw video into a ready-to-sell cut.")
                            .font(CS.inter(14))
                            .foregroundStyle(CS.textSecondaryF)
                    }
                    .padding(.top, 8)
                    .padding(.bottom, 4)

                    uploadCard
                    rowCard(icon: "HubCamera", title: "Record with CutSell",
                            subtitle: "Camera with clips · 60s · 3m · 10m", id: "hubRecord", action: onRecord)
                    rowCard(icon: "HubMic", title: "Voice Over project",
                            subtitle: "Record your voice over photos or clips", id: "hubVoiceOver") { voiceOverSoon = true }

                    HStack {
                        Text("Recent cuts")
                            .font(CS.inter(15, .bold))
                            .foregroundStyle(CS.textPrimaryF)
                        Spacer()
                        Button("See all in Edits") { selectTab(.edits) }
                            .font(CS.inter(13, .semibold))
                            .foregroundStyle(CS.cyan)
                    }
                    .padding(.top, 10)

                    if recent.isEmpty {
                        Text("Your cuts will show up here.")
                            .font(CS.inter(13))
                            .foregroundStyle(CS.textMutedF)
                            .frame(maxWidth: .infinity, minHeight: 92)
                            .background(CS.surfaceCard, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(CS.borderDefault, lineWidth: 1))
                    } else {
                        HStack(spacing: 10) {
                            ForEach(Array(recent.enumerated()), id: \.element.id) { index, project in
                                Button { onOpenProject(project) } label: { recentTile(project, index: index) }
                                    .buttonStyle(.plain)
                            }
                            ForEach(recent.count..<3, id: \.self) { _ in Color.clear.frame(height: 92) }
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 16)
            }
        }
        .safeAreaInset(edge: .bottom) { CSTabBar(current: .create, select: selectTab) }
        .alert("Voice Over project", isPresented: $voiceOverSoon) {
            Button("OK", role: .cancel) {}
        } message: {
            Text("Coming soon. For now you can record a voice over inside the editor of any cut.")
        }
    }

    private var uploadCard: some View {
        VStack(alignment: .leading, spacing: 4) {
                Image("HubUpload")
                    .resizable()
                    .frame(width: 26, height: 26)
                    .frame(width: 52, height: 52)
                    .background(Color(hex: 0x050814, opacity: 0.28), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                Spacer().frame(height: 12)
                Text("Upload videos")
                    .font(CS.inter(24, .heavy))
                    .foregroundStyle(.white)
                Text("From your gallery. AI cuts it for you.")
                    .font(CS.inter(14))
                    .foregroundStyle(.white)
                HStack(spacing: 6) {
                    ForEach(CreateMode.allCases, id: \.self) { mode in
                        Button { onUpload(mode) } label: {
                            Text(mode.title)
                                .font(CS.inter(12, .semibold))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color(hex: 0x050814, opacity: 0.3), in: Capsule())
                        }
                        .buttonStyle(.plain)
                        .accessibilityIdentifier("hubMode_\(mode.rawValue)")
                    }
                }
                .padding(.top, 12)
            }
            .padding(.horizontal, 22)
            .padding(.vertical, 20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(CS.brandFull, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
            .shadow(color: Color(hex: 0x7C3AED, opacity: 0.55), radius: 25, y: 18)
            .contentShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
            .onTapGesture { onUpload(.single) }
            .accessibilityElement(children: .contain)
            .accessibilityAddTraits(.isButton)
            .accessibilityIdentifier("hubUpload")
    }

    private func rowCard(icon: String, title: String, subtitle: String, id: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 14) {
                Image(icon)
                    .resizable()
                    .frame(width: 24, height: 24)
                    .frame(width: 46, height: 46)
                    .background(CS.surfaceControl, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(CS.borderDefault, lineWidth: 1))
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(CS.inter(16, .bold))
                        .foregroundStyle(CS.textPrimaryF)
                    Text(subtitle)
                        .font(CS.inter(13))
                        .foregroundStyle(CS.textMutedF)
                        .multilineTextAlignment(.leading)
                }
                Spacer(minLength: 0)
                Image("HubChevron").resizable().frame(width: 20, height: 20)
            }
            .padding(16)
            .background(CS.surfaceCard, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(CS.borderDefault, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(id)
    }

    private static let tileColors: [UInt32] = [0x2A3F6B, 0x4A2A6B, 0x1F5A6B]

    private func recentTile(_ project: Project, index: Int) -> some View {
        let isReady = project.state == "draft_ready"
        let isProcessing = Project.processingStates.contains(project.state)
        return RoundedRectangle(cornerRadius: 16, style: .continuous)
            .fill(LinearGradient(colors: [Color(hex: Self.tileColors[index % 3]), CS.surfaceCard],
                                 startPoint: .topLeading, endPoint: .bottomTrailing))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(CS.borderDefault, lineWidth: 1))
            .frame(height: 92)
            .overlay(alignment: .topLeading) {
                Circle().fill(isReady ? CS.ready : CS.draft).frame(width: 8, height: 8).padding(7)
            }
            .overlay {
                if isProcessing { ProgressView().tint(CS.cyan) }
            }
            .overlay(alignment: .bottomLeading) {
                Text(project.title)
                    .font(CS.inter(11, .semibold))
                    .foregroundStyle(.white)
                    .lineLimit(2)
                    .padding(8)
            }
            .accessibilityLabel(project.title)
    }
}
