import SwiftUI

struct ProjectsView: View {
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var notificationCenter: NotificationCenterModel
    @State private var showingNewCut = false
    @State private var showingNotifications = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            Group {
                if appState.projects.isEmpty {
                    ContentUnavailableView(
                        "No cuts yet",
                        systemImage: "scissors",
                        description: Text("Upload raw product footage and CutSell will build your first draft.")
                    )
                } else {
                    List(appState.projects) { project in
                        NavigationLink(value: project) {
                            VStack(alignment: .leading, spacing: 5) {
                                Text(project.title).font(.headline)
                                HStack {
                                    Text(project.state.replacingOccurrences(of: "_", with: " ").capitalized)
                                    if let count = project.sourceCount { Text("• \(count) clip\(count == 1 ? "" : "s")") }
                                }
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                    .refreshable {
                        do {
                            try await appState.refreshProjects()
                            if let session = appState.session {
                                await notificationCenter.refresh(session: session, surfaceLocalAlerts: false)
                            }
                        } catch { errorMessage = error.localizedDescription }
                    }
                }
            }
            .navigationTitle("CutSell")
            .toolbar {
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button { showingNotifications = true } label: {
                        ZStack(alignment: .topTrailing) {
                            Image(systemName: "bell")
                            if notificationCenter.unreadCount > 0 {
                                Text("\(min(notificationCenter.unreadCount, 9))")
                                    .font(.system(size: 9, weight: .bold))
                                    .foregroundStyle(.white)
                                    .frame(width: 15, height: 15)
                                    .background(.red, in: Circle())
                                    .offset(x: 7, y: -7)
                            }
                        }
                    }
                    Button { showingNewCut = true } label: { Image(systemName: "plus") }
                }
            }
            .navigationDestination(for: Project.self) { project in
                ProjectDetailView(project: project)
            }
            .sheet(isPresented: $showingNewCut) {
                NewCutView()
                    .environmentObject(appState)
            }
            .sheet(isPresented: $showingNotifications) {
                NotificationInboxView()
                    .environmentObject(notificationCenter)
            }
            .alert("CutSell", isPresented: Binding(
                get: { errorMessage != nil },
                set: { if !$0 { errorMessage = nil } }
            )) { Button("OK", role: .cancel) {} } message: { Text(errorMessage ?? "") }
        }
    }
}

private struct NotificationInboxView: View {
    @EnvironmentObject private var notificationCenter: NotificationCenterModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if notificationCenter.items.isEmpty {
                    ContentUnavailableView("No updates", systemImage: "bell", description: Text("Draft and export updates will appear here."))
                } else {
                    List(notificationCenter.items) { item in
                        HStack(alignment: .top, spacing: 12) {
                            Image(systemName: icon(item.kind))
                                .foregroundStyle(item.kind.contains("failed") ? .red : .accentColor)
                                .frame(width: 24)
                            VStack(alignment: .leading, spacing: 4) {
                                Text(title(item.kind)).font(.headline)
                                Text(item.projectID).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .navigationTitle("Updates")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Mark read") { notificationCenter.markAllRead() }
                        .disabled(notificationCenter.unreadCount == 0)
                }
            }
            .onAppear { notificationCenter.markAllRead() }
        }
    }

    private func title(_ kind: String) -> String {
        switch kind {
        case "draft_ready": return "Draft ready"
        case "render_finished": return "Export ready"
        case "processing_failed": return "Cut needs another try"
        case "render_failed": return "Export failed"
        default: return kind.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    private func icon(_ kind: String) -> String {
        switch kind {
        case "draft_ready": return "wand.and.stars"
        case "render_finished": return "checkmark.circle"
        case "processing_failed", "render_failed": return "exclamationmark.triangle"
        default: return "bell"
        }
    }
}

struct ProjectDetailView: View {
    let project: Project

    var body: some View {
        Group {
            switch project.state {
            case "processing", "uploaded", "preparing", "transcribing", "analyzing", "composing":
                ProcessingView(project: project)
            default:
                DraftEditorView(project: project)
            }
        }
        .navigationTitle(project.title)
        .navigationBarTitleDisplayMode(.inline)
    }
}
