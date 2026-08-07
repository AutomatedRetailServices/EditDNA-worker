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
        case "preparing": 5
        case "uploaded": 10
        case "transcribing": 30
        case "analyzing": 60
        case "composing": 85
        case "draft_ready": 100
        default: 5
        }
    }

    private var stageTitle: String {
        let state = job?.state ?? current.state
        switch state {
        case "preparing": "Preparing footage"
        case "uploaded": "Waiting for CutSell"
        case "transcribing": "Listening to your footage"
        case "analyzing": "Watching and choosing best takes"
        case "composing": "Building your sales edit"
        case "draft_ready", "finished": "Your draft is ready"
        case "failed": "This cut needs another try"
        default: state.replacingOccurrences(of: "_", with: " ").capitalized
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
