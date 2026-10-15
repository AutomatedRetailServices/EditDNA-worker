#if DEBUG
import AVFoundation
import SwiftUI

/// Editor probe "v5" (Debug only, real cut on the live server): every Editor v5 option, end to end.
/// Two overlays on layers (Float and Stacked, Soft edge, sound on), a voice-over after the 3 · 2 · 1
/// countdown (moved, split, trimmed, Lower video sound), Speed 2x on a clip, the CutSell sounds Tap,
/// Good, Wrong and Typing (stretched past 3 s), Undo / Redo, then a full Export. Every step logs
/// one CUTSELLV5 line.
@MainActor
enum EditorV5Probe {
    static func log(_ text: String) { NSLog("CUTSELLV5 %@", text) }

    static func run(model: DraftEditorViewModel, playback: DraftPlaybackController,
                    select: @escaping (EditorSelection?) -> Void,
                    open: @escaping (EditorPanelKind) -> Void,
                    toast: @escaping (String) -> Void,
                    countdown: @escaping (Int?) -> Void) async {
        let wait = EditorMediaProbe.wait
        let until = EditorMediaProbe.until
        await wait(4)
        let startDuration = model.timelineDuration
        log(String(format: "start clips=%d duration=%.2f", model.selectedClips.count, startDuration))

        // 1. Overlay panel: Photo · Video.
        open(.overlayAdd)
        await wait(2.5)
        guard let photo = EditorMediaProbe.makePhoto(), let video = await EditorMediaProbe.makeVideo(seconds: 4) else {
            log("could not make test media"); return
        }
        let before = model.mediaOverlays.count
        model.addOverlay(fileURL: photo, at: 0.5)
        await until(60) { model.pendingOverlays.isEmpty && model.mediaOverlays.count > before }
        model.addOverlay(fileURL: video, at: 1.0)
        await wait(0.5)
        log("video_overlay_new_layer=\(model.lastAddedOverlayNewLayer)")
        await until(90) { model.pendingOverlays.isEmpty && model.mediaOverlays.count > before + 1 }
        let layers = model.overlaysOnScreen.map { "\(model.overlayTitle($0.id)):\($0.style.layer)" }.joined(separator: ",")
        log("overlays=\(model.mediaOverlays.count) layers=\(layers) rows=\(TimelineV4.lowerRows(for: model).count)")
        await wait(2)
        if let videoOverlay = model.overlaysOnScreen.first(where: { $0.kind == .video }) {
            select(.overlay(videoOverlay.id, isText: false))
            await wait(2)
            await model.setOverlayStyle(id: videoOverlay.id) { $0.layout = .stacked }?.value
            await wait(2.5)
            await model.setOverlayStyle(id: videoOverlay.id) { $0.edge = .soft; $0.muted = false }?.value
            let style = model.overlayStyle(videoOverlay.id)
            log("video_overlay layout=\(style.layout.rawValue) edge=\(style.edge.rawValue) muted=\(style.muted) layer=\(style.layer)")
            playback.seek(to: 1.5)
            await wait(3)
        }
        if let photoOverlay = model.overlaysOnScreen.first(where: { $0.kind == .photo }) {
            select(.overlay(photoOverlay.id, isText: false))
            await model.setOverlayStyle(id: photoOverlay.id) { $0.layout = .float; $0.rotation = 12 }?.value
            log("photo_overlay layout=\(model.overlayStyle(photoOverlay.id).layout.rawValue) rotation=\(model.overlayStyle(photoOverlay.id).rotation)")
            await wait(2.5)
        }
        select(nil)

        // 2. Voice-over: 3 · 2 · 1, recorded take saved, then move · split · trim · Lower video sound.
        for value in [3, 2, 1] { countdown(value); await wait(1) }
        countdown(nil)
        if let take = EditorV4View.sampleVoiceM4A(seconds: 6) {
            let voicesBefore = model.voiceClips.count
            let outcome = await model.addRecordedVoiceOver(fileURL: take, at: 1, enhanced: true)
            log("voice_saved outcome=\(outcome) shown_pending=\(model.pendingVoiceOvers.count)")
            await until(45) { model.pendingVoiceOvers.isEmpty && model.voiceClips.count > voicesBefore }
            if let voice = model.voiceClips.last {
                log(String(format: "voice_on_draft start=%.2f length=%.2f clean=%@", voice.start, voice.length, voice.clean ? "yes" : "no"))
                select(.voiceOver(voice.id))
                await wait(2.5)
                await model.moveAudioClip(id: voice.id, to: voice.start + 1)?.value
                let moved = model.voiceClips.first { $0.id == voice.id }
                log(String(format: "voice_moved start=%.2f", moved?.start ?? -1))
                if let moved {
                    _ = model.splitAudioClip(id: moved.id, at: moved.start + moved.length / 2)
                    await wait(1.5)
                    log("voice_split pieces=\(model.voiceClips.count)")
                    if let last = model.voiceClips.last {
                        await model.trimAudioClip(id: last.id, edge: .end, to: last.end - 0.5)?.value
                        log(String(format: "voice_trimmed end=%.2f", model.voiceClips.last?.end ?? -1))
                    }
                    if let first = model.voiceClips.first {
                        await model.updateAudioClip(id: first.id) { $0.duckVideo = true }.value
                        log("lower_video_sound=\(model.voiceClips.first?.duckVideo == true)")
                    }
                }
                await wait(2.5)
            }
        }
        select(nil)

        // 3. Speed 2x on the first clip.
        if let first = model.selectedClips.first?["clip_id"]?.stringValue {
            select(.video(first))
            open(.speed)
            await wait(2)
            let durationBefore = model.timelineDuration
            await model.setClipSpeed(clipID: first, speed: 2).value
            log(String(format: "speed_2x clip=%@ duration %.2f -> %.2f", first, durationBefore, model.timelineDuration))
            await wait(3)
            open(.speed)
        }
        select(nil)

        // 4. CutSell sounds: Tap, Good, Wrong, Typing (stretched past 3 s -> the 9 s typing).
        open(.audio)
        await SfxLibrary.shared.prepare()
        log("sounds_ready=\(SfxLibrary.shared.ready.sorted().joined(separator: ",")) failed=\(SfxLibrary.shared.failed)")
        await wait(2)
        let tap = model.addSound(sfxID: "tap", at: 0.3)
        _ = model.addSound(sfxID: "good", at: 1.2)
        _ = model.addSound(sfxID: "wrong", at: 2.1)
        let typing = model.addSound(sfxID: "typing", at: 2.6)
        await wait(2)
        if let item = model.audioClips.first(where: { $0.id == typing }) {
            await model.trimAudioClip(id: typing, edge: .end, to: item.start + 4.5)?.value
        }
        let sounds = model.soundClips.map { "\($0.sfxID ?? "?")@\(String(format: "%.1f", $0.start))+\(String(format: "%.1f", $0.length))" }
        log("sounds=\(sounds.joined(separator: ",")) tap_present=\(model.audioClips.contains { $0.id == tap })")
        await wait(2.5)
        open(.audio)

        // 5. Undo / Redo of the last change.
        let count = model.audioClips.count
        _ = await model.undo()
        let undone = model.audioClips.first { $0.id == typing }?.sfxID ?? "gone"
        _ = await model.redo()
        log("undo_redo audio_clips=\(count) typing_after_undo=\(undone) typing_after_redo=\(model.audioClips.first { $0.id == typing }?.sfxID ?? "gone")")

        // 6. Volume panel, then Export with everything.
        open(.volume)
        await wait(2)
        open(.volume)
        let started = Date()
        toast("Exporting…")
        let exporting = Task { await model.export() }
        await until(300) {
            if let job = model.exportJob { return ["finished", "failed", "canceled"].contains(job.state) }
            return false
        }
        _ = await exporting.value
        let job = model.exportJob
        log(String(format: "export state=%@ total=%.1f", job?.state ?? "none", Date().timeIntervalSince(started)))
        if let error = job?.error { log("export error=\(error)") }
        if let link = job?.result?["download_url"]?.stringValue, let url = URL(string: link),
           let pair = try? await URLSession.shared.download(from: url) {
            let copy = FileManager.default.temporaryDirectory.appendingPathComponent("probe-export-\(UUID().uuidString.prefix(6)).mp4")
            try? FileManager.default.moveItem(at: pair.0, to: copy)
            let seconds = await EditorMediaInfo.duration(of: copy)
            log(String(format: "export_mp4 duration=%.2f expected=%.2f", seconds, model.timelineDuration))
        }
        log("done error=\(model.errorMessage ?? "none")")
        toast(job?.state == "finished" ? "Export ready" : "Export: \(job?.state ?? "none")")
    }
}
#endif
