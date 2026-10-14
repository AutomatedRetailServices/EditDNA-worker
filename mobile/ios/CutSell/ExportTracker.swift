import CryptoKit
import Foundation
import UIKit
import UserNotifications

/// Follows every export the creator started, app-wide, so leaving the editor does not stop it:
/// polls the job every second, keeps the real % for the editor, records how long each stage took
/// (logged as CUTSELLEXPORT lines for the export measurements), and tells the creator when the
/// video is ready if they are no longer on that cut's editor (in-app banner, or a notification
/// when CutSell is in the background).
@MainActor
final class ExportTracker: ObservableObject {
    static let shared = ExportTracker()

    @Published private(set) var jobs: [String: JobStatus] = [:]
    /// A finished export for a cut whose editor is not on screen (shown as a banner).
    @Published var readyBanner: String?
    /// The project whose editor is on screen.
    var visibleProjectID: String?

    struct Timing {
        let projectID: String
        let title: String
        let started: Date
        var marks: [(label: String, seconds: Double)] = []
        var lastState = ""
        var lastProgress = -1
    }
    private var timings: [String: Timing] = [:]
    private var loops: [String: Task<Void, Never>] = [:]

    /// The last finished export of each cut and what it was made from. Exporting again with
    /// nothing changed gives that video back at once instead of rendering it again (its link
    /// lasts an hour; after 50 minutes it is rendered again).
    private var finished: [String: (key: String, status: JobStatus, at: Date)] = [:]
    private var jobKeys: [String: (projectID: String, key: String)] = [:]

    static func contentKey(draft: JSONValue, sources: [JSONValue]) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = (try? encoder.encode(JSONValue.object(["draft": draft, "sources": .array(sources)]))) ?? Data()
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    func reusable(projectID: String, key: String) -> JobStatus? {
        guard let last = finished[projectID], last.key == key, Date().timeIntervalSince(last.at) < 50 * 60 else { return nil }
        return last.status
    }

    func remember(jobID: String, projectID: String, key: String) { jobKeys[jobID] = (projectID, key) }

    func follow(jobID: String, projectID: String, title: String, preparedSeconds: Double, submittedSeconds: Double) {
        guard loops[jobID] == nil else { return }
        var timing = Timing(projectID: projectID, title: title, started: Date().addingTimeInterval(-submittedSeconds))
        timing.marks.append(("prepare", preparedSeconds))
        timing.marks.append(("submitted", submittedSeconds))
        timings[jobID] = timing
        // A little time to finish polling if the creator leaves the app right away.
        let background = UIApplication.shared.beginBackgroundTask(withName: "cutsell-export") {}
        loops[jobID] = Task { @MainActor [weak self] in
            defer { UIApplication.shared.endBackgroundTask(background) }
            await self?.poll(jobID: jobID)
        }
    }

    private func poll(jobID: String) async {
        var failures = 0
        while !Task.isCancelled {
            do {
                let status: JobStatus = try await APIClient.shared.request("/v1/jobs/\(jobID)")
                failures = 0
                jobs[jobID] = status
                record(jobID: jobID, status: status)
                if ["finished", "failed", "canceled"].contains(status.state) {
                    if status.state == "finished" {
                        if let known = jobKeys[jobID] { finished[known.projectID] = (known.key, status, Date()) }
                        announce(jobID: jobID)
                    }
                    break
                }
            } catch {
                failures += 1
                if failures >= 20 { break }
            }
            try? await Task.sleep(for: .seconds(1))
        }
        loops[jobID] = nil
    }

    private func record(jobID: String, status: JobStatus) {
        guard var timing = timings[jobID] else { return }
        let progress = status.progress ?? -1
        if status.state != timing.lastState || progress != timing.lastProgress {
            let seconds = Date().timeIntervalSince(timing.started)
            timing.marks.append(("\(status.state)@\(max(0, progress))", seconds))
            timing.lastState = status.state
            timing.lastProgress = progress
            NSLog("CUTSELLEXPORT %@ %@ %.1f", jobID, "\(status.state)@\(max(0, progress))", seconds)
            if ["finished", "failed", "canceled"].contains(status.state) {
                let line = timing.marks.map { String(format: "%@=%.1f", $0.label, $0.seconds) }.joined(separator: " ")
                NSLog("CUTSELLEXPORT summary %@", line)
            }
        }
        timings[jobID] = timing
    }

    /// Seconds from tapping Export to each recorded stage (for the probe).
    func marks(jobID: String) -> [(label: String, seconds: Double)] { timings[jobID]?.marks ?? [] }

    private func announce(jobID: String) {
        guard let timing = timings[jobID] else { return }
        if UIApplication.shared.applicationState != .active {
            let content = UNMutableNotificationContent()
            content.title = "Your video is ready"
            content.body = timing.title.isEmpty ? "Open CutSell to save or share it." : "\(timing.title) is ready to save or share."
            content.sound = .default
            UNUserNotificationCenter.current().add(
                UNNotificationRequest(identifier: "export-ready-\(jobID)", content: content, trigger: nil)
            )
        } else if visibleProjectID != timing.projectID {
            readyBanner = timing.title.isEmpty ? "Your video is ready" : "\(timing.title) is ready"
        }
    }
}
