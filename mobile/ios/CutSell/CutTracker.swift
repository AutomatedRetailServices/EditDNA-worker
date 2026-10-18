import AVFoundation
import SwiftUI
import UIKit
import UserNotifications

/// Follows every cut that is being made, from the server's REAL state, until it is ready.
/// It keeps polling while you use other tabs ("Continue in background") and tells you when the
/// cut is ready (a banner inside the app, or an iPhone notification if CutSell is in the background).
@MainActor
final class CutTracker: ObservableObject {
    static let shared = CutTracker()

    struct Cut: Identifiable, Equatable {
        let id: String              // project id
        var title: String
        var fileName: String
        var thumbnail: UIImage?
        var footageSeconds: Double?
        var jobID: String?
        var projectState: String = "uploaded"
        var jobState: String?
        var jobProgress: Int?
        var error: String?
        var isReady: Bool { projectState == "draft_ready" || jobState == "finished" }
        var isFailed: Bool { jobState == "failed" || jobState == "canceled" || projectState == "failed" || gone }
        /// The server kept answering 403/404 (project deleted or not ours): stop asking.
        var gone = false
        var clientErrors = 0
    }

    @Published private(set) var cuts: [String: Cut] = [:]
    /// A cut that became ready while you were somewhere else in the app (shows a banner).
    @Published var readyBanner: String?
    /// The project whose Processing / Ready screen is on screen (no banner for it).
    var visibleProjectID: String?

    private var loop: Task<Void, Never>?
    private let pollSeconds: UInt64 = 3

    /// Called right after the cut started (upload done, job created).
    func track(projectID: String, jobID: String?, title: String, fileName: String,
               thumbnail: UIImage?, footageSeconds: Double?) {
        var cut = cuts[projectID] ?? Cut(id: projectID, title: title, fileName: fileName)
        cut.jobID = jobID ?? cut.jobID
        cut.thumbnail = thumbnail ?? cut.thumbnail
        cut.footageSeconds = footageSeconds ?? cut.footageSeconds
        if !fileName.isEmpty { cut.fileName = fileName }
        cuts[projectID] = cut
        startLoop()
    }

    /// For a project opened from Edits / Recent cuts (no local video information).
    func follow(project: Project) {
        if cuts[project.projectID] == nil {
            cuts[project.projectID] = Cut(
                id: project.projectID, title: project.title, fileName: project.title,
                jobID: project.latestJobID, projectState: project.state
            )
        }
        startLoop()
    }

    func cut(_ projectID: String) -> Cut? { cuts[projectID] }

    // MARK: Polling

    private func startLoop() {
        guard loop == nil else { return }
        loop = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let active = self.cuts.values.filter { !$0.isReady && !$0.isFailed }
                if active.isEmpty { break }
                for cut in active { await self.poll(cut.id) }
                try? await Task.sleep(nanoseconds: self.pollSeconds * 1_000_000_000)
            }
            self?.loop = nil
        }
    }

    private func poll(_ projectID: String) async {
        guard let session = await APIClient.shared.session, var cut = cuts[projectID] else { return }
        let wasReady = cut.isReady
        do {
            let project: Project = try await APIClient.shared.request(
                "/v1/projects/\(projectID)",
                query: [URLQueryItem(name: "user_id", value: session.userID)]
            )
            cut.projectState = project.state
            cut.jobID = project.latestJobID ?? cut.jobID
            if let jobID = cut.jobID {
                let status: JobStatus = try await APIClient.shared.request("/v1/jobs/\(jobID)")
                cut.jobState = status.state
                cut.jobProgress = status.progress
                cut.error = status.state == "failed" ? (status.error ?? "processing_failed") : nil
            }
            cut.clientErrors = 0
            cut.error = cut.isFailed ? (cut.error ?? "processing_failed") : nil
        } catch APIError.http(let code, _) where (400..<500).contains(code) && code != 429 {
            cut.clientErrors += 1
            if cut.clientErrors >= 3 { cut.gone = true; cut.error = "not_found" }
        } catch {
            // A temporary network problem: try again on the next round.
        }
        cuts[projectID] = cut
        if cut.isReady && !wasReady { announceReady(cut) }
    }

    private func announceReady(_ cut: Cut) {
        CutSellDiagnostics.log("cut_ready_seen", ["project_id": cut.id])
        if UIApplication.shared.applicationState != .active {
            let content = UNMutableNotificationContent()
            content.title = "Your cut is ready"
            content.body = "Open CutSell to review and export it."
            content.sound = .default
            let request = UNNotificationRequest(identifier: "cut-ready-\(cut.id)", content: content, trigger: nil)
            UNUserNotificationCenter.current().add(request)
        } else if visibleProjectID != cut.id {
            readyBanner = cut.id
        }
        Task { try? await AppStateBridge.refreshProjects?() }
    }

    // MARK: Progress shown on the Processing screen (from the server's real state)

    struct Stage: Equatable {
        let step: Int          // 1...8
        let title: String
        let percent: Int
    }

    static let stepCount = 8

    /// Upload is step 1 (its real %), then the server's job: 0-100 % of the job maps to 20-100 %.
    static func uploadStage(progress: Double, status: String) -> Stage {
        let starting = status.contains("Starting AI edit") || status.contains("Creating project")
        if starting && progress >= 0.999 {
            return Stage(step: 2, title: "Starting your cut", percent: 20)
        }
        return Stage(step: 1, title: "Uploading your video", percent: Int((min(1, max(0, progress)) * 19).rounded()))
    }

    static func serverStage(_ cut: Cut) -> Stage {
        if cut.isReady { return Stage(step: 8, title: "Your cut is ready", percent: 100) }
        let state = cut.jobState == "queued" ? "queued" : cut.projectState
        let job = cut.jobProgress
        func scaled(_ fallback: Int) -> Int { 20 + Int(Double(job ?? fallback) * 0.8) }
        switch state {
        // Build 23: one honest name for the server's work, with what it is doing now.
        case "queued", "uploaded", "preparing", "created":
            return Stage(step: 3, title: "Cutting your video… starting", percent: scaled(2))
        case "transcribing":
            return Stage(step: 4, title: "Cutting your video… listening", percent: scaled(12))
        case "analyzing":
            let p = job ?? 30
            return p < 50
                ? Stage(step: 5, title: "Cutting your video… choosing the best takes", percent: scaled(30))
                : Stage(step: 6, title: "Cutting your video… building the edit", percent: scaled(55))
        case "composing":
            return Stage(step: 7, title: "Cutting your video… almost done", percent: scaled(84))
        default:
            return Stage(step: 3, title: "Cutting your video… starting", percent: scaled(2))
        }
    }

    // MARK: Helpers

    /// First frame of a local video, for the 9:16 preview.
    nonisolated static func firstFrame(of url: URL) async -> UIImage? {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 360, height: 640)
        guard let image = try? await generator.image(at: .zero).image else { return nil }
        return UIImage(cgImage: image)
    }
}

/// Lets the tracker ask the app to refresh the Edits list without owning AppState.
@MainActor
enum AppStateBridge {
    static var refreshProjects: (() async throws -> Void)?
}
