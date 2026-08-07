import SwiftUI

struct ProjectsView: View {
    @EnvironmentObject private var appState: AppState
    @State private var showingNewCut = false
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
                        do { try await appState.refreshProjects() }
                        catch { errorMessage = error.localizedDescription }
                    }
                }
            }
            .navigationTitle("CutSell")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
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
            .alert("CutSell", isPresented: Binding(
                get: { errorMessage != nil },
                set: { if !$0 { errorMessage = nil } }
            )) { Button("OK", role: .cancel) {} } message: { Text(errorMessage ?? "") }
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
