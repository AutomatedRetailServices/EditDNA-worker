import SwiftUI

/// Edits filters (Figma 1134:2683 / 1134:2885): Processing (N) · Ready (N) · Drafts (N) · All (N).
/// Build 24 (Product Owner, 18 Oct): "Processing" is shown only while something is being made; a
/// cut leaves it the moment it is ready and is in Ready at once.
enum EditsTab: String, CaseIterable, Identifiable, Hashable {
    case processing = "Processing", ready = "Ready", drafts = "Drafts", all = "All"
    var id: String { rawValue }

    /// Figma chip widths (12 pt SF semibold, height 38).
    var chipWidth: CGFloat {
        switch self {
        case .processing: return 100
        case .ready, .drafts: return 84
        case .all: return 66
        }
    }
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
    @ObservedObject private var queue = ProcessingQueue.shared
    @ObservedObject private var tracker = CutTracker.shared

    private var columns: [GridItem] { Array(repeating: GridItem(.flexible(), spacing: 10), count: 3) }

    private var visibleProjects: [Project] {
        switch selectedTab {
        case .all, .processing:
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

    /// The Processing rows, in order: this iPhone's uploads and cuts, cuts still processing on the
    /// server, and uploads that stopped (FAILED · Retry continues where they stopped).
    private var processingRows: [ProcessingQueue.Row] {
        var rows = queue.rows(projects: appState.projects, tracker: tracker)
        let listed = Set(rows.compactMap(\.projectID))
        for pending in pendingCuts where !listed.contains(pending.projectID) {
            rows.append(ProcessingQueue.Row(id: "pending-" + pending.projectID, itemID: nil, projectID: pending.projectID,
                                            name: pending.title.isEmpty ? "Interrupted Cut" : pending.title,
                                            duration: pending.videos.reduce(0) { $0 + $1.duration },
                                            thumbnail: ThumbnailStore.shared.image(for: pending.projectID),
                                            state: .failed("Upload paused")))
        }
        return rows
    }

    private var visibleTabs: [EditsTab] {
        processingRows.isEmpty ? [.all, .ready, .drafts] : EditsTab.allCases
    }

    private func count(_ tab: EditsTab) -> Int {
        switch tab {
        case .processing: return processingRows.count
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
                        Text(selectedTab == .processing ? "Cuts being edited now. You can leave the app."
                                                        : "Your saved cuts, ready to review or share.")
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

                        if visibleTabs.contains(.processing) {
                            // Figma 1134:2817-2823: 100 / 84 / 84 / 66 pt chips, 6 pt apart.
                            HStack(spacing: 6) {
                                ForEach(visibleTabs) { tab in
                                    EditsFilterChip(title: "\(tab.rawValue) (\(count(tab)))", isSelected: selectedTab == tab) {
                                        selectedTab = tab
                                    }
                                    .frame(width: tab.chipWidth)
                                    .accessibilityIdentifier("editsFilter\(tab.rawValue)")
                                }
                                Spacer(minLength: 0)
                            }
                            .padding(.top, 22)
                            // .contain keeps each chip's own id (editsFilterProcessing…); without it the
                            // stack's id replaced the chips' ids and "Processing" could not be found.
                            .accessibilityElement(children: .contain)
                            .accessibilityIdentifier("editsTabPicker")
                        } else {
                            HStack(spacing: 0) {
                                ForEach(visibleTabs) { tab in
                                    FilterChip(title: "\(tab.rawValue) (\(count(tab)))", isSelected: selectedTab == tab) {
                                        selectedTab = tab
                                    }
                                    .frame(width: 96)
                                    .accessibilityIdentifier("editsFilter\(tab.rawValue)")
                                    if tab != visibleTabs.last { Spacer(minLength: 8) }
                                }
                            }
                            .padding(.top, 22)
                            // .contain keeps each chip's own id (editsFilterProcessing…); without it the
                            // stack's id replaced the chips' ids and "Processing" could not be found.
                            .accessibilityElement(children: .contain)
                            .accessibilityIdentifier("editsTabPicker")
                        }

                        if selectedTab == .processing {
                            processingList
                        } else {
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
                    }
                    .padding(.horizontal, 20)
                    .padding(.bottom, 20)
                }
                .refreshable { await refreshAll() }
            // Cuts still being made: follow them on the server until they are Ready
            // (the card then turns green and gets its thumbnail).
            .task(id: appState.projects.filter { Project.processingStates.contains($0.state) }.map(\.id)) {
                for project in appState.projects where Project.processingStates.contains(project.state) {
                    CutTracker.shared.follow(project: project)
                }
            }
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
            // "Go to Edits" (Bulk sheet / Processing screen) opens on Processing.
            .onAppear { takeRequestedFilter() }
            .onChange(of: queue.requestedEditsFilter) { _, _ in takeRequestedFilter() }
            // The last cut finished: Processing disappears, Ready shows it.
            .onChange(of: processingRows.isEmpty) { _, empty in
                if empty && selectedTab == .processing { selectedTab = .ready }
            }
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

    private func takeRequestedFilter() {
        guard queue.requestedEditsFilter == "processing" else { return }
        queue.requestedEditsFilter = nil
        selectedTab = .processing
        Task { await refreshPendingCuts() }
    }

    /// Figma 1134:2825-2884: "In order", the rows, and the note.
    private var processingList: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                Text("In order").font(CS.inter(16, .semibold)).foregroundStyle(.white)
                Spacer()
                Text("Each one moves to Ready when it’s done").font(CS.inter(11, .medium)).foregroundStyle(Color(hex: 0x99A3BD))
            }
            .padding(.top, 16)
            VStack(spacing: 8) {
                ForEach(Array(processingRows.enumerated()), id: \.element.id) { index, row in
                    processingRow(index + 1, row)
                }
            }
            .padding(.top, 12)
            Text("Editing keeps going on our servers even if you close the app. We’ll notify you as each cut is ready.")
                .font(CS.inter(12))
                .foregroundStyle(Color(hex: 0x99A3BD))
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 15)
                .padding(.vertical, 10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(hex: 0x0F1726), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color(hex: 0x33405C), lineWidth: 1))
                .padding(.top, 18)
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("editsProcessingList")
    }

    @ViewBuilder
    private func processingRow(_ order: Int, _ row: ProcessingQueue.Row) -> some View {
        let retry: () -> Void = {
            if let item = row.itemID {
                queue.retry(item)
            } else if let pending = pendingCuts.first(where: { $0.projectID == row.projectID }) {
                resumeCut = pending
            }
        }
        if let id = row.projectID, let project = appState.projects.first(where: { $0.projectID == id }),
           pendingCuts.first(where: { $0.projectID == id }) == nil {
            NavigationLink(value: project) { QueueRowView(order: order, row: row, onRetry: retry) }
                .buttonStyle(.plain)
        } else {
            QueueRowView(order: order, row: row, onRetry: retry)
        }
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
        case .all, .processing: return "No edits yet"
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
        // Build 23: a cut that already started from its voice is not "Resume upload" (its video
        // uploads by itself in the background).
        pendingCuts = await PendingCutStore.shared.list().filter { $0.jobID == nil }
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
    @EnvironmentObject private var appState: AppState
    @ObservedObject private var thumbnails = ThumbnailStore.shared

    private var isReady: Bool { project.state == "draft_ready" }
    private var isProcessing: Bool { Project.processingStates.contains(project.state) }

    var body: some View {
        RoundedRectangle(cornerRadius: 12, style: .continuous)
            .fill(Color(hex: 0x050A17))
            .aspectRatio(108.0 / 96.0, contentMode: .fit)
            .overlay {
                if let image = thumbnails.image(for: project.projectID) {
                    Image(uiImage: image).resizable().scaledToFill()
                        .overlay(LinearGradient(colors: [.clear, .black.opacity(0.65)], startPoint: .center, endPoint: .bottom))
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(CS.tileBorder, lineWidth: 1))
            .task(id: project.state) {
                await thumbnails.load(project: project, session: appState.session)
                if project.state == "draft_ready" {
                    await EditorPrewarm.prewarm(projectID: project.projectID, session: appState.session)
                }
            }
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
                Text(CutNaming.readable(project: project))
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
            pending = await PendingCutStore.shared.list().first { $0.projectID == project.projectID && $0.jobID == nil }
            checked = true
        }
        .sheet(item: $resuming, onDismiss: {
            Task {
                pending = await PendingCutStore.shared.list().first { $0.projectID == project.projectID && $0.jobID == nil }
                try? await appState.refreshProjects()
            }
        }) { record in
            NewCutView(pending: record)
                .environmentObject(appState)
        }
    }
}


/// Edits filter chip as drawn in Figma 1134:2817 (38 pt high, 12 pt SF semibold; selected = cyan→blue
/// gradient with a cyan stroke; others dark with a blue stroke).
struct EditsFilterChip: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(CS.sf(12, .semibold))
                .foregroundStyle(Color(hex: 0xF0F7FF))
                .lineLimit(1)
                .minimumScaleFactor(0.85)
                .frame(maxWidth: .infinity)
                .frame(height: 38)
                .background {
                    if isSelected {
                        Capsule().fill(LinearGradient(colors: [Color(hex: 0x0DCCFF), Color(hex: 0x1459FF)],
                                                      startPoint: .leading, endPoint: .trailing))
                    } else {
                        Capsule().fill(Color(hex: 0x03060F))
                    }
                }
                .overlay(Capsule().stroke(isSelected ? Color(hex: 0x0DE5FF) : Color(hex: 0x1F4794), lineWidth: 1.3))
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}
