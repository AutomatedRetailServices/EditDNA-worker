import SwiftUI

/// D-129 mobile UI decision: Edits has exactly these three tabs -- no
/// "Processing" tab/folder. Processing is a transient operational state
/// (still visible under "All" while it runs), never a place a user files
/// work into.
enum EditsTab: String, CaseIterable, Identifiable, Hashable {
    case all = "All", ready = "Ready", drafts = "Drafts"
    var id: String { rawValue }
}

/// Edits tab (Figma 05 Edits, 195:39; empty state 207:92): the saved cuts as a grid with the
/// All / Ready / Drafts filters. Opening a cut keeps the existing processing / editor screens.
struct ProjectsView: View {
    var selectTab: (MainTab) -> Void = { _ in }

    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var notificationCenter: NotificationCenterModel
    @State private var pendingCuts: [PendingCutRecord] = []
    @State private var resumeCut: PendingCutRecord?
    @State private var projectPendingDelete: Project?
    @State private var errorMessage: String?
    @State private var selectedTab: EditsTab = .all
    @State private var showingSearch = false
    @State private var searchText = ""

    private var columns: [GridItem] { Array(repeating: GridItem(.flexible(), spacing: 10), count: 3) }

    private var visibleProjects: [Project] {
        switch selectedTab {
        case .all:
            return appState.projects
        case .ready:
            return appState.projects.filter { $0.state == "draft_ready" }
        case .drafts:
            return appState.projects.filter {
                !Project.processingStates.contains($0.state) && $0.state != "draft_ready"
            }
        }
    }

    private var shownProjects: [Project] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return visibleProjects }
        return visibleProjects.filter { $0.title.localizedCaseInsensitiveContains(query) }
    }

    private func count(_ tab: EditsTab) -> Int {
        switch tab {
        case .all: return appState.projects.count
        case .ready: return appState.projects.filter { $0.state == "draft_ready" }.count
        case .drafts:
            return appState.projects.filter {
                !Project.processingStates.contains($0.state) && $0.state != "draft_ready"
            }.count
        }
    }

    var body: some View {
        NavigationStack {
            ZStack {
                ScreenBackground()
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        header
                        Text("Edits")
                            .font(CS.inter(34, .semibold))
                            .foregroundStyle(.white)
                            .padding(.top, 12)
                        Text("Your saved cuts, ready to review or share.")
                            .font(CS.inter(14))
                            .foregroundStyle(CS.textSecondary)
                            .padding(.top, 2)

                        if showingSearch {
                            TextField("", text: $searchText, prompt: Text("Search edits").foregroundStyle(CS.textMuted))
                                .font(CS.inter(15))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 14)
                                .frame(height: 42)
                                .background(CS.fieldBackground, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                                .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(CS.fieldBorder, lineWidth: 1))
                                .padding(.top, 14)
                                .accessibilityIdentifier("editsSearchField")
                        }

                        HStack(spacing: 0) {
                            ForEach(EditsTab.allCases) { tab in
                                FilterChip(title: "\(tab.rawValue) (\(count(tab)))", isSelected: selectedTab == tab) {
                                    selectedTab = tab
                                }
                                .frame(width: 96)
                                if tab != EditsTab.allCases.last { Spacer(minLength: 8) }
                            }
                        }
                        .padding(.top, 22)
                        .accessibilityIdentifier("editsTabPicker")

                        if !pendingCuts.isEmpty {
                            VStack(spacing: 8) {
                                ForEach(pendingCuts) { pending in
                                    Button { resumeCut = pending } label: { resumeRow(pending) }
                                        .buttonStyle(.plain)
                                }
                            }
                            .padding(.top, 16)
                        }

                        if shownProjects.isEmpty {
                            emptyState
                                .padding(.top, 80)
                        } else {
                            LazyVGrid(columns: columns, spacing: 8) {
                                ForEach(shownProjects) { project in
                                    NavigationLink(value: project) {
                                        EditTile(project: project) { projectPendingDelete = project }
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                            .padding(.top, 10)
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.bottom, 20)
                }
                .refreshable { await refreshAll() }
            }
            .toolbar(.hidden, for: .navigationBar)
            .safeAreaInset(edge: .bottom) { CSTabBar(current: .edits, select: selectTab) }
            .navigationDestination(for: Project.self) { project in
                ProjectDetailView(project: project)
            }
            .sheet(item: $resumeCut) { pending in
                NewCutView(pending: pending)
                    .environmentObject(appState)
                    .onDisappear { Task { await refreshPendingCuts() } }
            }
            .task { await refreshPendingCuts() }
            .confirmationDialog(
                "Delete project?",
                isPresented: Binding(
                    get: { projectPendingDelete != nil },
                    set: { if !$0 { projectPendingDelete = nil } }
                ),
                titleVisibility: .visible
            ) {
                Button("Delete Project", role: .destructive) {
                    guard let project = projectPendingDelete else { return }
                    projectPendingDelete = nil
                    Task { await deleteProject(project) }
                }
                Button("Cancel", role: .cancel) { projectPendingDelete = nil }
            } message: {
                Text("This permanently removes the project, its uploaded media, exports, overlays, timeline assets and feedback.")
            }
            .alert("CutSell", isPresented: Binding(
                get: { errorMessage != nil },
                set: { if !$0 { errorMessage = nil } }
            )) { Button("OK", role: .cancel) {} } message: { Text(errorMessage ?? "") }
        }
        .tint(CS.cyan)
    }

    private var header: some View {
        HStack(spacing: 10) {
            BrandLogo()
            Spacer()
            CircleIconButton(size: 32, action: {
                showingSearch.toggle()
                if !showingSearch { searchText = "" }
            }) {
                Image("IconSearch").resizable().frame(width: 16, height: 16)
            }
            .accessibilityLabel("Search")
            Menu {
                Button("New cut") { selectTab(.create) }
                Button("Refresh") { Task { await refreshAll() } }
            } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 34, height: 34)
                    .background(Color(hex: 0x050A17), in: Circle())
                    .overlay(Circle().stroke(CS.circleButtonBorder, lineWidth: 1))
            }
            .accessibilityLabel("More")
        }
        .padding(.top, 8)
    }

    private func resumeRow(_ pending: PendingCutRecord) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "arrow.clockwise.icloud")
                .foregroundStyle(CS.cyan)
            VStack(alignment: .leading, spacing: 3) {
                Text(pending.title.isEmpty ? "Interrupted Cut" : pending.title)
                    .font(CS.inter(14, .semibold))
                    .foregroundStyle(.white)
                Text("\(pending.videos.count) clip\(pending.videos.count == 1 ? "" : "s") • Resume upload")
                    .font(CS.inter(12))
                    .foregroundStyle(CS.textSecondary)
            }
            Spacer()
            Image(systemName: "chevron.right").font(.caption).foregroundStyle(CS.textMuted)
        }
        .padding(12)
        .background(CS.card, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(CS.tileBorder, lineWidth: 1))
    }

    private var emptyState: some View {
        HStack(spacing: 20) {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(hex: 0x050A17))
                .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color(hex: 0x1A8CFF), lineWidth: 1.5))
                .overlay(Image("EmptyPlay").resizable().frame(width: 44, height: 44))
                .frame(width: 90, height: 160)
            VStack(alignment: .leading, spacing: 8) {
                Text(emptyTitle)
                    .font(CS.inter(19, .semibold))
                    .foregroundStyle(.white)
                Text(emptyMessage)
                    .font(CS.inter(13))
                    .foregroundStyle(Color(hex: 0x99B2E0))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(20)
        .frame(maxWidth: .infinity)
        .frame(minHeight: 220)
        .background(Color(hex: 0x040812, opacity: 0.96), in: RoundedRectangle(cornerRadius: 24, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 24, style: .continuous).stroke(Color(hex: 0x14478C), lineWidth: 1))
        .padding(.horizontal, 10)
    }

    private var emptyTitle: String {
        if !searchText.isEmpty { return "No matches" }
        switch selectedTab {
        case .all: return "No edits yet"
        case .ready: return "No ready edits"
        case .drafts: return "No drafts"
        }
    }

    private var emptyMessage: String {
        if !searchText.isEmpty { return "Try another name." }
        return "Your finished and draft edits will appear here."
    }

    @MainActor
    private func refreshAll() async {
        do {
            try await appState.refreshProjects()
            if let session = appState.session {
                await notificationCenter.refresh(session: session, surfaceLocalAlerts: false)
            }
            await refreshPendingCuts()
        } catch { errorMessage = error.localizedDescription }
    }

    @MainActor
    private func refreshPendingCuts() async {
        pendingCuts = await PendingCutStore.shared.list()
    }

    @MainActor
    private func deleteProject(_ project: Project) async {
        do {
            try await appState.deleteProject(project)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

/// One saved cut in the Edits grid (Figma tile: status dot, "•••" menu, radius 12).
struct EditTile: View {
    let project: Project
    let onDelete: () -> Void

    private var isReady: Bool { project.state == "draft_ready" }
    private var isProcessing: Bool { Project.processingStates.contains(project.state) }

    var body: some View {
        RoundedRectangle(cornerRadius: 12, style: .continuous)
            .fill(Color(hex: 0x050A17))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(CS.tileBorder, lineWidth: 1))
            .aspectRatio(108.0 / 96.0, contentMode: .fit)
            .overlay(alignment: .topLeading) {
                Circle()
                    .fill(isReady ? CS.ready : CS.draft)
                    .frame(width: 10, height: 10)
                    .padding(10)
            }
            .overlay(alignment: .topTrailing) {
                Menu {
                    Button("Delete", role: .destructive, action: onDelete)
                } label: {
                    Image(systemName: "ellipsis")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(width: 34, height: 28)
                }
                .accessibilityLabel("Options for \(project.title)")
            }
            .overlay {
                if isProcessing {
                    ProgressView().tint(CS.cyan)
                }
            }
            .overlay(alignment: .bottomLeading) {
                Text(project.title)
                    .font(CS.inter(11, .semibold))
                    .foregroundStyle(.white)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                    .padding(.horizontal, 9)
                    .padding(.bottom, 8)
            }
            .contentShape(Rectangle())
            .accessibilityElement(children: .combine)
            .accessibilityLabel("\(project.title), \(isReady ? "ready" : isProcessing ? "processing" : "draft")")
    }
}

struct ProjectDetailView: View {
    let project: Project
    /// "Edits" on the Processing screen (from the Home Hub it switches to the Edits tab).
    var onOpenEdits: (() -> Void)?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Group {
            if project.state == Project.uploadNotFinishedState {
                // No video reached the server yet, so there is no cut and no draft to edit.
                UploadNotFinishedView(project: project)
            } else if Project.processingStates.contains(project.state) {
                // Figma Processing → Ready, following the cut's real state on the server.
                CutProgressView(
                    projectID: project.projectID, fileName: project.title, thumbnail: nil,
                    onClose: { dismiss() }, onOpenEdits: { if let open = onOpenEdits { open() } else { dismiss() } }
                )
                .onAppear { CutTracker.shared.follow(project: project) }
            } else {
                DraftEditorView(project: project)
            }
        }
        .navigationTitle(project.title)
        .navigationBarTitleDisplayMode(.inline)
    }
}

/// A project whose video never finished uploading (state "created"): there is nothing to edit yet.
/// Offers to resume the saved upload instead of opening an editor that can only fail.
struct UploadNotFinishedView: View {
    let project: Project
    @EnvironmentObject private var appState: AppState
    @State private var pending: PendingCutRecord?
    @State private var checked = false
    @State private var resuming: PendingCutRecord?

    var body: some View {
        ZStack {
            ScreenBackground()
            VStack(spacing: 14) {
                Text("!")
                    .font(CS.inter(46, .bold))
                    .foregroundStyle(CS.alert)
                Text("This video didn’t finish uploading")
                    .font(CS.inter(22, .bold))
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                Text(pending != nil
                     ? "Your video is saved on this iPhone. Resume the upload and CutSell will make the cut."
                     : (checked ? "The video isn’t on this iPhone anymore. Create a new cut from Create." : ""))
                    .font(CS.inter(15))
                    .foregroundStyle(CS.textSecondary)
                    .multilineTextAlignment(.center)
                if let pending {
                    PrimaryButton(title: "Resume upload", height: 54, cornerRadius: 16, fontSize: 16) {
                        resuming = pending
                    }
                    .frame(width: 270)
                    .padding(.top, 20)
                    .accessibilityIdentifier("resumeUploadButton")
                }
            }
            .padding(.horizontal, 28)
        }
        .task {
            pending = await PendingCutStore.shared.list().first { $0.projectID == project.projectID }
            checked = true
        }
        .sheet(item: $resuming, onDismiss: {
            Task {
                pending = await PendingCutStore.shared.list().first { $0.projectID == project.projectID }
                try? await appState.refreshProjects()
            }
        }) { record in
            NewCutView(pending: record)
                .environmentObject(appState)
        }
    }
}
