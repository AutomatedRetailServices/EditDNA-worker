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
        print("CUTSELL_PERF begin=\(step)")
        let clock = ContinuousClock()
        let start = clock.now
        var visibleAt: ContinuousClock.Instant?
        let finished = PerfFlag()
        let task = Task { @MainActor in await action(); finished.value = true }
        while visibleAt == nil {
            if visible() { visibleAt = clock.now; break }
            if clock.now - start > .seconds(30) { break }
            try? await Task.sleep(for: .milliseconds(5))
        }
        // Wait for the server's answer, at most 60 s (a step that never finishes is reported, not hung on).
        while !finished.value && clock.now - start < .seconds(60) {
            try? await Task.sleep(for: .milliseconds(10))
        }
        if !finished.value {
            record(step, ["visible_ms": visibleAt.map { Self.ms($0 - start) } ?? "never", "saved_ms": "timeout_60s"])
            task.cancel()
            return
        }
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
              let api = env["CUTSELL_PROBE_API"] else {
            record("setup", ["error": "missing_env"]); return
        }
        let videoPath = env["CUTSELL_PROBE_VIDEO"] ?? ""
        UserDefaults.standard.set(api, forKey: "cutsell.api.baseURL")
        let session = CutSellSession(userID: user, accessToken: token)
        await APIClient.shared.setSession(session)
        appState.session = session
        let videoURL = URL(fileURLWithPath: videoPath)

        do {
            try await time("server_ping") { let _: JSONValue = try await APIClient.shared.request("/v1/healthz") }
            try await time("server_ping_2") { let _: JSONValue = try await APIClient.shared.request("/v1/healthz") }

            var project: Project?
            if let existing = env["CUTSELL_PROBE_PROJECT"] {
                // The cut was made outside the app (the Simulator cannot run iOS background uploads);
                // only the editor steps are timed here.
                project = Project(projectID: existing, userID: user, title: "Perf probe", state: "draft_ready",
                                  latestJobID: nil, sourceCount: 1, updatedAt: nil)
            } else {
                try await time("create_project") { project = try await appState.createProject(title: "Perf probe") }
            }
            guard let project else { return }
            if env["CUTSELL_PROBE_PROJECT"] == nil {

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

            }

            // 1) Open the app -> Edits list. Before: the list waits for the server. After: the list
            //    kept on the phone shows at once, the server's copy replaces it.
            await time("edits_list_server") { try? await appState.refreshProjects() }
            await time("edits_list_shown") {
                appState.projects = []
                if let cached = ProjectListCache.load(userID: user) { appState.projects = cached } // AFTER-ONLY
                if appState.projects.isEmpty { try? await appState.refreshProjects() }
            }

            // 2) Tap a cut -> editor ready (UI on screen) and first frame of the video.
            for pass in ["first_open", "reopen"] {
                let model = DraftEditorViewModel(project: project, session: session)
                let playback = DraftPlaybackController()
                let clock = ContinuousClock()
                let tapped = clock.now
                let loading = Task { @MainActor in await model.load() }
                var uiAt: ContinuousClock.Instant?
                while uiAt == nil, clock.now - tapped < .seconds(60) {
                    if model.snapshot != nil && !model.isLoading { uiAt = clock.now; break }
                    try? await Task.sleep(for: .milliseconds(5))
                }
                await playback.rebuild(from: model)
                let frameAt = clock.now
                await loading.value
                if !playback.isReady { await playback.rebuild(from: model) }   // fresh links arrived
                record("editor_\(pass)", [
                    "ui_ms": uiAt.map { Self.ms($0 - tapped) } ?? "never",
                    "first_frame_ms": Self.ms(frameAt - tapped),
                    "player_ready": String(playback.isReady),
                    "local_video": String(playback.sourcesAreLocal(model.snapshot)), // AFTER-ONLY
                ])

                // 3) Tap play -> sound starts (the player is really playing and time moves).
                playback.seek(to: 0)
                let playTap = clock.now
                playback.togglePlayback()
                var playingAt: ContinuousClock.Instant?
                while playingAt == nil, clock.now - playTap < .seconds(30) {
                    if playback.player.timeControlStatus == .playing, playback.player.currentTime().seconds > 0.03 { playingAt = clock.now }
                    try? await Task.sleep(for: .milliseconds(5))
                }
                record("play_\(pass)", ["ms": playingAt.map { Self.ms($0 - playTap) } ?? "never"])
                playback.pause()

                // 4) End of the cut: the player must stop there (build 11 kept playing the original).
                if pass == "first_open", playback.duration > 2 {
                    playback.seek(to: playback.duration - 1.0)
                    playback.togglePlayback()
                    try? await Task.sleep(for: .seconds(4))
                    let position = playback.player.currentTime().seconds
                    record("end_of_cut", [
                        "cut_s": String(format: "%.2f", playback.duration),
                        "player_s_after_3s_past_end": String(format: "%.2f", position),
                        "overrun_s": String(format: "%.2f", max(0, position - playback.duration)),
                        "still_playing": String(playback.player.rate != 0),
                        "timeline_s": String(format: "%.2f", playback.currentTime),
                    ])
                    playback.pause()
                }

                if pass == "reopen" {
                    // 5) Each edit: on screen (visible_ms) and confirmed by the server (saved_ms).
                    if let clip = model.selectedClips.first, let id = clip["clip_id"]?.stringValue {
                        let start = clip["start"]?.doubleValue ?? 0
                        let end = clip["end"]?.doubleValue ?? start
                        if let at = model.safeSplitTime(clipID: id, near: (start + end) / 2) {
                            let before = model.selectedClips.count
                            await timeAction("action_split", visible: { model.selectedClips.count > before }) {
                                await model.split(clipID: id, at: at)
                            }
                        }
                    }
                    if let id = model.selectedClips.first?["clip_id"]?.stringValue {
                        await timeAction("action_caption_text", visible: {
                            model.selectedClips.first?["caption_text"]?.stringValue == "Perf probe caption"
                        }) { await model.editCaption(clipID: id, text: "Perf probe caption") }
                    }
                    if let id = model.selectedClips.first?["clip_id"]?.stringValue {
                        let was = model.clipIsMirrored(id)
                        await timeAction("action_mirror", visible: { model.clipIsMirrored(id) != was }) {
                            _ = await model.toggleMirror(clipID: id)
                        }
                    }
                    await timeAction("action_caption_style", visible: { model.captionPreset == "box" }) {
                        await model.setCaptionSettings(preset: "box")
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
                    record("errors", ["message": model.errorMessage ?? "none"])
                }
                // Give the background download a moment before reopening (it is what a second open uses).
                if pass == "first_open" {
                    try? await Task.sleep(for: .seconds(8))
                    for _ in 0..<60 where !playback.sourcesAreLocal(model.snapshot) { try? await Task.sleep(for: .milliseconds(500)) } // AFTER-ONLY
                }
            }
            record("done", [:])
        } catch {
            record("error", ["error": String(describing: type(of: error)), "detail": String(error.localizedDescription.prefix(160))])
        }
    }
}

@MainActor
final class PerfFlag {
    var value = false
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
