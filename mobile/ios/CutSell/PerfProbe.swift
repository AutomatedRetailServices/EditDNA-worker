import Foundation
import SwiftUI

#if DEBUG
/// Debug builds only (`-CutSellScreen perf_probe`): measures the app's own code paths against the live
/// server with a synthetic test video (no personal data), so the same steps can be timed before and
/// after a change. Every step is timed with the app's real classes: the multipart upload, the editor
/// load, the preview build and each editor action. Results go to Documents/perf.json and the console.
///
/// For each action two times are taken:
/// - visible_ms: until the editor's state shows the change (what the person sees),
/// - saved_ms: until the server confirmed it.
@MainActor
final class PerfProbe: ObservableObject {
    @Published var lines: [String] = []
    private var results: [[String: String]] = []

    private func record(_ step: String, _ fields: [String: String]) {
        var row = fields
        row["step"] = step
        results.append(row)
        let text = "CUTSELL_PERF " + row.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " ")
        print(text)
        lines.append(text)
        write()
    }

    private func write() {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("perf.json")
        if let data = try? JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: url)
        }
    }

    private static func ms(_ duration: Duration) -> String {
        let c = duration.components
        return String(Int(Double(c.seconds) * 1000 + Double(c.attoseconds) / 1e15))
    }

    private func time(_ step: String, extra: [String: String] = [:], _ work: () async throws -> Void) async rethrows {
        let clock = ContinuousClock()
        let start = clock.now
        try await work()
        var fields = extra
        fields["ms"] = Self.ms(clock.now - start)
        record(step, fields)
    }

    /// Runs `action`; `visible` is checked every 5 ms on the main actor until it turns true.
    private func timeAction(_ step: String, visible: @escaping @MainActor () -> Bool, _ action: @escaping @MainActor () async -> Void) async {
        let clock = ContinuousClock()
        let start = clock.now
        var visibleAt: ContinuousClock.Instant?
        let task = Task { @MainActor in await action() }
        while visibleAt == nil {
            if visible() { visibleAt = clock.now; break }
            if clock.now - start > .seconds(30) { break }
            try? await Task.sleep(for: .milliseconds(5))
        }
        await task.value
        let saved = clock.now
        if visibleAt == nil, visible() { visibleAt = saved }
        record(step, [
            "visible_ms": visibleAt.map { Self.ms($0 - start) } ?? "never",
            "saved_ms": Self.ms(saved - start),
        ])
    }

    func run(appState: AppState) async {
        let env = ProcessInfo.processInfo.environment
        guard let token = env["CUTSELL_PROBE_TOKEN"], let user = env["CUTSELL_PROBE_USER"],
              let api = env["CUTSELL_PROBE_API"], let videoPath = env["CUTSELL_PROBE_VIDEO"] else {
            record("setup", ["error": "missing_env"]); return
        }
        UserDefaults.standard.set(api, forKey: "cutsell.api.baseURL")
        let session = CutSellSession(userID: user, accessToken: token)
        await APIClient.shared.setSession(session)
        appState.session = session
        let videoURL = URL(fileURLWithPath: videoPath)

        do {
            try await time("server_ping") { let _: JSONValue = try await APIClient.shared.request("/v1/healthz") }
            try await time("server_ping_2") { let _: JSONValue = try await APIClient.shared.request("/v1/healthz") }

            var project: Project?
            try await time("create_project") { project = try await appState.createProject(title: "Perf probe") }
            guard let project else { return }

            let size = ((try? FileManager.default.attributesOfItem(atPath: videoPath))?[.size] as? NSNumber)?.int64Value ?? 0
            var sourceURI = ""
            let clock = ContinuousClock()
            let uploadStart = clock.now
            let result = try await MultipartUploadManager.shared.upload(
                fileURL: videoURL, projectID: project.projectID, session: session, contentType: "video/mp4"
            ) { _ in }
            sourceURI = result.sourceURI
            let uploadElapsed = clock.now - uploadStart
            let seconds = Double(uploadElapsed.components.seconds) + Double(uploadElapsed.components.attoseconds) / 1e18
            record("upload", [
                "ms": Self.ms(uploadElapsed), "mb": String(format: "%.1f", Double(size) / 1_048_576),
                "mb_per_s": String(format: "%.2f", Double(size) / 1_048_576 / max(0.001, seconds)),
            ])

            struct Body: Encodable {
                let project_id: String; let user_id: String; let sources: [SourceInput]
                let language_hint: String?; let audio_overlap: Bool
            }
            var jobID = ""
            try await time("submit_cut") {
                let response: FlowBSubmitResponse = try await APIClient.shared.request("/v1/flow-b/jobs", method: "POST", body: Body(
                    project_id: project.projectID, user_id: user,
                    sources: [SourceInput(originalName: "perf.mp4", uri: sourceURI, sourceOrder: 0, durationSec: 60)],
                    language_hint: nil, audio_overlap: false
                ))
                jobID = response.jobID
            }
            try await time("server_cut_wait") {
                for _ in 0..<300 {
                    let status: JobStatus = try await APIClient.shared.request("/v1/jobs/\(jobID)")
                    if ["finished", "failed", "canceled"].contains(status.state) { break }
                    try await Task.sleep(for: .seconds(2))
                }
            }

            let model = DraftEditorViewModel(project: project, session: session)
            await time("editor_load") { await model.load() }
            record("editor_info", ["clips": String(model.selectedClips.count), "duration_s": String(format: "%.1f", model.timelineDuration)])
            let playback = DraftPlaybackController()
            await time("preview_build") { await playback.rebuild(from: model) }

            // Actions, each timed until visible and until saved.
            if let clip = model.selectedClips.first, let id = clip["clip_id"]?.stringValue {
                let start = clip["start"]?.doubleValue ?? 0
                let end = clip["end"]?.doubleValue ?? start
                if let at = model.safeSplitTime(clipID: id, near: (start + end) / 2) {
                    let before = model.selectedClips.count
                    await timeAction("action_split", visible: { model.selectedClips.count > before }) {
                        await model.split(clipID: id, at: at)
                    }
                    await time("preview_rebuild_after_split") { await playback.rebuild(from: model) }
                }
            }
            if let id = model.selectedClips.first?["clip_id"]?.stringValue {
                await timeAction("action_caption_text", visible: {
                    model.selectedClips.first?["caption_text"]?.stringValue == "Perf probe caption"
                }) { await model.editCaption(clipID: id, text: "Perf probe caption") }
            }
            await timeAction("action_caption_style", visible: { model.captionPreset == "box" }) {
                await model.setCaptionSettings(preset: "box")
            }
            if let id = model.selectedClips.first?["clip_id"]?.stringValue {
                await timeAction("action_volume", visible: {
                    (model.selectedClips.first?["audio_volume"]?.doubleValue ?? 1) < 0.6
                }) { await model.setAudio(clipID: id, volume: 0.5) }
            }
            if model.selectedClips.count > 1, let id = model.selectedClips.last?["clip_id"]?.stringValue {
                let before = model.selectedClips.count
                await timeAction("action_delete", visible: { model.selectedClips.count < before }) {
                    await model.remove(clipID: id)
                }
                let afterDelete = model.selectedClips.count
                await timeAction("action_undo", visible: { model.selectedClips.count > afterDelete }) {
                    _ = await model.undo()
                }
            }
            record("done", [:])
        } catch {
            record("error", ["error": String(describing: type(of: error)), "detail": String(error.localizedDescription.prefix(160))])
        }
    }
}

struct PerfProbeScreen: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var probe = PerfProbe()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(probe.lines.enumerated()), id: \.offset) { _, line in
                    Text(line).font(.system(size: 10, design: .monospaced)).foregroundStyle(.white)
                }
            }
            .padding()
        }
        .background(Color.black)
        .task { await probe.run(appState: appState) }
    }
}
#endif
