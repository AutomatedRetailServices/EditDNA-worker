import SwiftUI
import Combine

/// Edits filters (Figma 1134:2683 / 1134:2885): Processing (N) · Ready (N) · Drafts (N) · All (N).
/// Build 24 (Product Owner, 18 Oct): "Processing" is shown only while something is being made; a
/// cut leaves it the moment it is ready and is in Ready at once.
enum EditsTab: String, CaseIterable, Identifiable, Hashable {
    case processing = "Processing", ready = "Ready", products = "Products", drafts = "Drafts", all = "All"
    var id: String { rawValue }

    /// Figma chip widths (12 pt SF semibold, height 38).
    var chipWidth: CGFloat {
        switch self {
        case .processing: return 100
        case .ready, .drafts: return 84
        case .products: return 80
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
    @ObservedObject private var catalog = CatalogStore.shared
    @ObservedObject private var mixCenter = MixCenter.shared
    @State private var assigning: AssignRequest?
    @State private var mixRoute: ProductRoute?

    private var columns: [GridItem] { Array(repeating: GridItem(.flexible(), spacing: 10), count: 3) }

    private var visibleProjects: [Project] {
        switch selectedTab {
        case .all, .processing, .products:
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
        // Mix: its videos are one group (MixGroupRow), not one row each; its upload project is not a cut.
        let mixVideos = mixCenter.activeProjectIDs
        var rows = queue.rows(projects: appState.projects.filter { !mixVideos.contains($0.projectID) }, tracker: tracker)
            .filter { row in row.projectID.map { !mixVideos.contains($0) && !mixCenter.mixIDs.contains($0) } ?? true }
        let listed = Set(rows.compactMap(\.projectID))
        // A record whose upload is running in CutSell right now is not "paused" (its queue row already
        // shows Uploading); only a really interrupted upload is listed as Failed with Retry.
        for pending in pendingCuts where !listed.contains(pending.projectID)
            && !ActiveUploads.shared.contains(pending.projectID) && !mixCenter.mixIDs.contains(pending.projectID) {
            rows.append(ProcessingQueue.Row(id: "pending-" + pending.projectID, itemID: nil, projectID: pending.projectID,
                                            name: pending.title.isEmpty ? "Interrupted Cut" : pending.title,
                                            duration: pending.videos.reduce(0) { $0 + $1.duration },
                                            thumbnail: ThumbnailStore.shared.image(for: pending.projectID),
                                            state: .failed("Upload paused")))
        }
        return rows
    }

    /// Plan · "In progress": only the rows of the group tapped there (a Bulk send); a Mix shows only its group row.
    private var focusedRows: [ProcessingQueue.Row] {
        guard let focus = queue.focus else { return processingRows }
        if mixCenter.mixIDs.contains(focus) { return [] }
        let ids = Set(queue.items.filter { $0.group == focus }.map(\.id))
        return processingRows.filter { row in row.itemID.map { ids.contains($0) } ?? false }
    }

    /// Figma 1135:2686: Processing · Ready · Products · Drafts · All, sliding sideways. Processing only while
    /// something is being made.
    private var visibleTabs: [EditsTab] {
        processingRows.isEmpty && mixCenter.active.isEmpty ? [.ready, .products, .drafts, .all] : EditsTab.allCases
    }

    private func chipTitle(_ tab: EditsTab) -> String {
        tab == .products ? "Products" : "\(tab.rawValue) (\(count(tab)))"
    }

    private func count(_ tab: EditsTab) -> Int {
        switch tab {
        case .processing: return processingRows.count + mixCenter.active.count
        case .products: return catalog.overview.brands.count
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
                        Text(L10n.tr("Edits"))
                            .font(CS.inter(34, .semibold))
                            .foregroundStyle(.white)
                            .padding(.top, 12)
                        Text(L10n.tr(selectedTab == .processing ? "Cuts being edited now. You can leave the app."
                             : selectedTab == .products ? "Your cuts, organized by brand and product."
                             : "Your saved cuts, ready to review or share."))
                            .font(CS.inter(14))
                            .foregroundStyle(CS.textSecondary)
                            .padding(.top, 2)

                        if showingSearch {
                            TextField("", text: $searchText, prompt: Text(L10n.tr("Search edits")).foregroundStyle(CS.textMuted))
                                .font(CS.inter(15))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 14)
                                .frame(height: 42)
                                .background(CS.fieldBackground, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                                .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(CS.fieldBorder, lineWidth: 1))
                                .padding(.top, 14)
                                .accessibilityIdentifier("editsSearchField")
                        }

                        // Figma 1135:2823-2830: 38 pt chips, 6 pt apart, sliding sideways.
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 6) {
                                ForEach(visibleTabs) { tab in
                                    EditsFilterChip(title: chipTitle(tab), isSelected: selectedTab == tab) {
                                        selectedTab = tab
                                    }
                                    .frame(minWidth: tab.chipWidth)  // longer words (Spanish) widen the chip
                                    .accessibilityIdentifier("editsFilter\(tab.rawValue)")
                                }
                            }
                            // .contain keeps each chip's own id (editsFilterProcessing…).
                            .accessibilityElement(children: .contain)
                            .accessibilityIdentifier("editsTabPicker")
                        }
                        .scrollClipDisabled()
                        .padding(.top, 22)

                        if selectedTab == .processing {
                            processingList
                        } else if selectedTab == .products {
                            ProductsList(store: catalog, projects: appState.projects) { ids, suggestion in
                                assigning = AssignRequest(projectIDs: ids, suggestion: suggestion)
                            }
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
                                    // The ••• menu sits OVER the link (not inside it): a tap on ••• opens the menu, never the cut.
                                    ZStack(alignment: .topTrailing) {
                                        NavigationLink(value: project) {
                                            EditTile(project: project, showsMenu: false) { projectPendingDelete = project }
                                        }
                                        .buttonStyle(.plain)
                                        EditTileMenu(project: project) { projectPendingDelete = project }
                                    }
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
                for project in appState.projects where Project.processingStates.contains(project.state)
                    && !mixCenter.activeProjectIDs.contains(project.projectID) {
                    CutTracker.shared.follow(project: project)
                }
            }
            }
            .toolbar(.hidden, for: .navigationBar)
            .safeAreaInset(edge: .bottom) { CSTabBar(current: .edits, select: selectTab) }
            .navigationDestination(for: Project.self) { project in
                ProjectDetailView(project: project)
            }
            .navigationDestination(for: ProductRoute.self) { route in
                ProductDetailView(route: route).environmentObject(appState)
            }
            .navigationDestination(item: $mixRoute) { route in
                ProductDetailView(route: route).environmentObject(appState)
            }
            .task { await mixCenter.refresh(session: appState.session) }
            // A Mix video just got made: it joins Ready (and its group moves on).
            .onReceive(NotificationCenter.default.publisher(for: .mixProgress)) { _ in
                Task { try? await appState.refreshProjects() }
            }
            // Mix · Results: "Make videos" opens Processing; a finished Mix opens its videos.
            .onChange(of: queue.requestedMix) { _, mix in
                guard let mix else { return }
                queue.requestedMix = nil
                selectedTab = .processing
                if let product = mix.productID, !mix.isActive { mixRoute = ProductRoute(productID: product, mixID: mix.mixID) }
            }
            .sheet(item: $assigning, onDismiss: { Task { await refreshCatalog() } }) { request in
                AssignProductSheet(request: request)
                    .environmentObject(appState)
                    .presentationDetents([.height(560), .large])
                    .presentationBackground(Color(hex: 0x0D121F))
            }
            .task(id: selectedTab) { if selectedTab == .products { await refreshCatalog() } }
            .sheet(item: $resumeCut) { pending in
                NewCutView(pending: pending)
                    .environmentObject(appState)
                    .onDisappear { Task { await refreshPendingCuts() } }
            }
            .task { await refreshPendingCuts() }
            // Build 24: the paused-upload list follows the queue (a cut that started from its voice or
            // finished uploading leaves it right away instead of staying as a stale "Failed").
            .onReceive(queue.$items.dropFirst()) { _ in Task { await refreshPendingCuts() } }
            .onReceive(tracker.$cuts.dropFirst()) { _ in Task { await refreshPendingCuts() } }
            // "Go to Edits" (Bulk sheet / Processing screen) opens on Processing.
            .onAppear { takeRequestedFilter() }
            .onChange(of: queue.requestedEditsFilter) { _, _ in takeRequestedFilter() }
            // The last cut finished: Processing disappears, Ready shows it.
            .onChange(of: processingRows.isEmpty && mixCenter.active.isEmpty) { _, empty in
                if empty && selectedTab == .processing { selectedTab = .ready }
            }
            .confirmationDialog(
                L10n.tr("Delete project?"),
                isPresented: Binding(
                    get: { projectPendingDelete != nil },
                    set: { if !$0 { projectPendingDelete = nil } }
                ),
                titleVisibility: .visible
            ) {
                Button(L10n.tr("Delete Project"), role: .destructive) {
                    guard let project = projectPendingDelete else { return }
                    projectPendingDelete = nil
                    Task { await deleteProject(project) }
                }
                Button(L10n.tr("Cancel"), role: .cancel) { projectPendingDelete = nil }
            } message: {
                Text(L10n.tr("This permanently removes the project, its uploaded media, exports, overlays, timeline assets and feedback."))
            }
            .alert(L10n.tr("CutSell"), isPresented: Binding(
                get: { errorMessage != nil },
                set: { if !$0 { errorMessage = nil } }
            )) { Button(L10n.tr("OK"), role: .cancel) {} } message: { Text(L10n.tr(errorMessage ?? "")) }
        }
        .tint(CS.cyan)
    }

    private func refreshCatalog() async {
        await catalog.refresh(session: appState.session, projects: appState.projects)
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
                Text(L10n.tr("In order")).font(CS.inter(16, .semibold)).foregroundStyle(Color.ink)
                Spacer()
                Text(L10n.tr("Each one moves to Ready when it’s done")).font(CS.inter(11, .medium)).foregroundStyle(Color(hex: 0x99A3BD))
            }
            .padding(.top, 16)
            if let focus = queue.focus {
                HStack {
                    Text(L10n.tr(mixCenter.mixes.contains { $0.mixID == focus } ? "Showing this Mix" : "Showing this Bulk"))
                        .font(CS.inter(12, .semibold)).foregroundStyle(Color(hex: 0x99A3BD))
                    Spacer()
                    Button(L10n.tr("Show all")) { queue.focus = nil }
                        .font(CS.inter(12, .semibold)).foregroundStyle(CS.cyan)
                        .accessibilityIdentifier("processingShowAll")
                }
                .padding(.top, 10)
                .accessibilityIdentifier("processingFocus")
            }
            VStack(spacing: 8) {
                ForEach(mixCenter.active.filter { queue.focus == nil || queue.focus == $0.mixID }) { mix in
                    MixGroupRow(mix: mix, onRetry: { Task { await mixCenter.retry(mix, session: appState.session) } }) {
                        if let product = mix.productID { mixRoute = ProductRoute(productID: product, mixID: mix.mixID) }
                    }
                }
                ForEach(Array(focusedRows.enumerated()), id: \.element.id) { index, row in
                    processingRow(index + 1, row)
                }
            }
            .padding(.top, 12)
            Text(L10n.tr("Editing keeps going on our servers even if you close the app. We’ll notify you as each cut is ready."))
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
            .accessibilityLabel(L10n.tr("Search"))
            Menu {
                Button(L10n.tr("New cut")) { selectTab(.create) }
                Button(L10n.tr("Refresh")) { Task { await refreshAll() } }
            } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundStyle(Color.ink)
                    .frame(width: 34, height: 34)
                    .background(Color(hex: 0x050A17), in: Circle())
                    .overlay(Circle().stroke(CS.circleButtonBorder, lineWidth: 1))
            }
            .accessibilityLabel(L10n.tr("More"))
        }
        .padding(.top, 8)
    }

    private func resumeRow(_ pending: PendingCutRecord) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "arrow.clockwise.icloud")
                .foregroundStyle(CS.cyan)
            VStack(alignment: .leading, spacing: 3) {
                Text(L10n.tr(pending.title.isEmpty ? "Interrupted Cut" : pending.title))
                    .font(CS.inter(14, .semibold))
                    .foregroundStyle(Color.ink)
                Text(L10n.tr("\(pending.videos.count) clip\(pending.videos.count == 1 ? "" : "s") • Resume upload"))
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
                Text(L10n.tr(emptyTitle))
                    .font(CS.inter(19, .semibold))
                    .foregroundStyle(Color.ink)
                Text(L10n.tr(emptyMessage))
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
        case .products: return "No products yet"
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
            await refreshCatalog()
            await mixCenter.refresh(session: appState.session)
            await PlanWeekStore.shared.refresh(session: appState.session)   // the ••• menu knows her personal videos
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
    var showsMenu: Bool = true
    let onDelete: () -> Void
    @EnvironmentObject private var appState: AppState
    @ObservedObject private var thumbnails = ThumbnailStore.shared
    @ObservedObject private var plan = PlanWeekStore.shared

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
                if showsMenu { EditTileMenu(project: project, onDelete: onDelete) }
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
            // .contain (not .combine): the ••• menu stays its own button for VoiceOver and the UI tests
            .accessibilityElement(children: .contain)
            .accessibilityLabel(L10n.tr("\(project.title), \(isReady ? "ready" : isProcessing ? "processing" : "draft")"))
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
                Text(L10n.tr("!"))
                    .font(CS.inter(46, .bold))
                    .foregroundStyle(CS.alert)
                Text(L10n.tr("This video didn’t finish uploading"))
                    .font(CS.inter(22, .bold))
                    .foregroundStyle(Color.ink)
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
            Text(L10n.tr(title))
                .font(CS.sf(12, .semibold))
                .foregroundStyle(Color(hex: 0xF0F7FF))
                .lineLimit(1)
                .minimumScaleFactor(0.85)
                .padding(.horizontal, 12)
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


/// The "•••" of an Edits tile: Personal · don't plan / Use in Plan (Ready cuts), Delete.
struct EditTileMenu: View {
    let project: Project
    let onDelete: () -> Void
    @EnvironmentObject private var appState: AppState
    @ObservedObject private var plan = PlanWeekStore.shared

    var body: some View {
        Menu {
            if project.state == "draft_ready" {
                let personal = plan.isPersonal(project.projectID)
                Button(personal ? "Use in Plan" : "Personal · don't plan") {
                    Task {
                        do { try await plan.setPersonal(project.projectID, !personal, session: appState.session) }
                        catch { plan.lastError = error.localizedDescription }
                    }
                }
            }
            Button(L10n.tr("Delete"), role: .destructive, action: onDelete)
        } label: {
            Image(systemName: "ellipsis")
                .font(.system(size: 13, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 34, height: 28)
                .contentShape(Rectangle())
        }
        .accessibilityLabel(L10n.tr("Options for \(project.title)"))
        .accessibilityIdentifier(project.state == "draft_ready" ? "editTileMenuReady" : "editTileMenu")
    }
}
