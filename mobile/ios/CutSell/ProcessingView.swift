import SwiftUI

struct ProcessingView: View {
    @EnvironmentObject private var appState: AppState
    let project: Project
    @State private var current: Project
    @State private var job: JobStatus?
    @State private var errorMessage: String?
    @State private var openDraft = false

    init(project: Project) {
        self.project = project
        _current = State(initialValue: project)
    }

    var body: some View {
        VStack(spacing: 22) {
            Spacer()
            Image(systemName: "wand.and.stars")
                .font(.system(size: 48))

            Text(stageTitle)
                .font(.title2.bold())

            ProgressView(value: Double(job?.progress ?? fallbackProgress) / 100)
                .padding(.horizontal, 36)

            Text("\(job?.progress ?? fallbackProgress)%")
                .font(.headline.monospacedDigit())

            Text("You can leave this screen. CutSell keeps processing on the server.")
                .font(.footnote)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 36)

            if let errorMessage {
                Text(errorMessage).foregroundStyle(.red).font(.footnote)
                Button("Try again") { Task { await pollOnce() } }
            }

            Spacer()
        }
        .task { await pollLoop() }
        .navigationDestination(isPresented: $openDraft) {
            DraftEditorView(project: current)
        }
    }

    private var fallbackProgress: Int {
        switch current.state {
        case "preparing": return 5
        case "uploaded": return 10
        case "transcribing": return 30
        case "analyzing": return 60
        case "composing": return 85
        case "draft_ready": return 100
        default: return 5
        }
    }

    private var stageTitle: String {
        let state = job?.state ?? current.state
        switch state {
        case "preparing": return "Preparing footage"
        case "uploaded": return "Waiting for CutSell"
        case "transcribing": return "Listening to your footage"
        case "analyzing": return "Watching and choosing best takes"
        case "composing": return "Building your sales edit"
        case "draft_ready", "finished": return "Your draft is ready"
        case "failed": return "This cut needs another try"
        default: return state.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    private func pollLoop() async {
        while !Task.isCancelled && !openDraft {
            await pollOnce()
            if openDraft { break }
            try? await Task.sleep(for: .seconds(2))
        }
    }

    @MainActor
    private func pollOnce() async {
        guard let session = appState.session else { return }
        do {
            let refreshed: Project = try await APIClient.shared.request(
                "/v1/projects/\(project.projectID)",
                query: [URLQueryItem(name: "user_id", value: session.userID)]
            )
            current = refreshed
            if let jobID = refreshed.latestJobID {
                let status: JobStatus = try await APIClient.shared.request("/v1/jobs/\(jobID)")
                job = status
                if status.state == "finished" || refreshed.state == "draft_ready" {
                    try? await appState.refreshProjects()
                    openDraft = true
                }
                if status.state == "failed" { errorMessage = status.error ?? "Processing failed" }
            } else if refreshed.state == "draft_ready" || refreshed.state == "finished" {
                openDraft = true
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
