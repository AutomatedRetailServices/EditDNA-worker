#if DEBUG
import AVFoundation
import SwiftUI
import UIKit

/// Editor probe "media" (Debug only, real cut on the live server): the newest photo and video in
/// the Simulator's Photos (put there by the test) go in as a photo overlay, a video overlay and a
/// full-screen B-roll clip; then a clip, a voice-over and an overlay are moved (with Undo / Redo),
/// and the whole cut is exported. Every step logs one CUTSELLMEDIA line.
@MainActor
enum EditorMediaProbe {
    static func log(_ text: String) { NSLog("CUTSELLMEDIA %@", text) }

    static func wait(_ seconds: Double) async { try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000)) }

    static func run(model: DraftEditorViewModel, select: @escaping (EditorSelection?) -> Void) async {
        await wait(4)
        // Test media made here (a photo and a 4 s video), as if picked in the gallery: the
        // Simulator's Photos permission prompt would cover the screen.
        guard let photo = makePhoto(), let stillVideo = await makeVideo(seconds: 4) else {
            log("could not make test media"); return
        }
        // A video with sound and movement, like one from a phone: a copy of this cut's own test video.
        var video = stillVideo
        if let link = model.playbackURL(forSource: nil),
           let pair = try? await URLSession.shared.download(from: link) {
            let copy = FileManager.default.temporaryDirectory.appendingPathComponent("probe-phone-video-\(UUID().uuidString.prefix(6)).mp4")
            try? FileManager.default.moveItem(at: pair.0, to: copy)
            video = copy
        }
        log("test_media video=\(video.lastPathComponent)")
        // 1. Photo overlay.
        var started = Date()
        let photoBefore = model.mediaOverlays.count
        model.addOverlay(fileURL: photo, at: 0.5)
        await wait(0.3)
        log(String(format: "photo_overlay shown_after=%.2f pending=%d", Date().timeIntervalSince(started), model.pendingOverlays.count))
        await until(60) { model.pendingOverlays.isEmpty && model.mediaOverlays.count > photoBefore }
        log(String(format: "photo_overlay uploaded_after=%.1f overlays=%d", Date().timeIntervalSince(started), model.mediaOverlays.count))
        // 2. Video overlay.
        started = Date()
        let overlaysBefore = model.mediaOverlays.count
        model.addOverlay(fileURL: video, at: 2.0)
        await until(90) { model.pendingOverlays.isEmpty && model.mediaOverlays.count > overlaysBefore }
        log(String(format: "video_overlay uploaded_after=%.1f overlays=%d", Date().timeIntervalSince(started), model.mediaOverlays.count))
        // Move / resize the photo, move the video overlay later.
        if let first = model.mediaOverlays.first?["overlay_id"]?.stringValue {
            select(.overlay(first, isText: false))
            await model.setOverlayFrame(id: first, x: 0.3, y: 0.25, width: 0.5).value
            log("photo_overlay moved x=0.3 y=0.25 width=0.5 saved=\(model.mediaOverlays.first?["x"]?.doubleValue == 0.3)")
        }
        await wait(2)
        if let second = model.mediaOverlays.last, let id = second["overlay_id"]?.stringValue,
           let start = second["start"]?.doubleValue, let end = second["end"]?.doubleValue {
            await model.setOverlayTime(id: id, start: start + 1, end: end + 1).value
            log(String(format: "video_overlay moved_to=%.2f", model.mediaOverlays.last?["start"]?.doubleValue ?? -1))
        }
        await wait(2)
        // 3. Full-screen B-roll at 2 s (main track).
        let before = model.selectedClips.count
        started = Date()
        model.addBrollClip(fileURL: video, at: 2.0)
        await until(90) { model.pendingBroll == nil && model.selectedClips.count > before }
        let ids = model.selectedClips.compactMap { $0["clip_id"]?.stringValue }
        log(String(format: "broll clips=%d->%d after=%.1f order=%@", before, model.selectedClips.count,
                   Date().timeIntervalSince(started), ids.joined(separator: ",")))
        await wait(2)
        // 4. Move a clip (lift the last one to the front), Undo, Redo.
        if ids.count > 1 {
            let moved = [ids[ids.count - 1]] + ids.dropLast()
            let revision = model.snapshot?.revision ?? 0
            await model.reorderNow(clipIDs: moved).value
            let now = model.selectedClips.compactMap { $0["clip_id"]?.stringValue }
            log("clip_moved first=\(now.first ?? "") saves=\((model.snapshot?.revision ?? 0) - revision)")
            await wait(2)
            _ = await model.undo()
            log("clip_undo first=\(model.selectedClips.first?["clip_id"]?.stringValue ?? "")")
            await wait(1.5)
            _ = await model.redo()
            log("clip_redo first=\(model.selectedClips.first?["clip_id"]?.stringValue ?? "")")
        }
        await wait(2)
        // 5. Move the voice-over 1.5 s later, Undo, Redo.
        if let voice = model.timelineComposition?.voiceOverPlacements.first {
            await model.moveVoiceOver(id: voice.placementID, to: voice.timelineStartSec + 1.5)
            let after = model.timelineComposition?.voiceOverPlacements.first { $0.placementID == voice.placementID }?.timelineStartSec ?? -1
            log(String(format: "voice_moved %.2f->%.2f", voice.timelineStartSec, after))
            await wait(1.5)
            _ = await model.undo()
            let undone = model.timelineComposition?.voiceOverPlacements.first { $0.placementID == voice.placementID }?.timelineStartSec ?? -1
            log(String(format: "voice_undo %.2f", undone))
            await wait(1.5)
            _ = await model.redo()
            let redone = model.timelineComposition?.voiceOverPlacements.first { $0.placementID == voice.placementID }?.timelineStartSec ?? -1
            log(String(format: "voice_redo %.2f", redone))
        }
        await wait(2)
        // 6. Export with everything.
        started = Date()
        let exporting = Task { await model.export() }
        await until(900) { ["finished", "failed", "canceled"].contains(model.exportJob?.state ?? "") }
        _ = await exporting.value
        log(String(format: "export state=%@ total=%.1f", model.exportJob?.state ?? "none", Date().timeIntervalSince(started)))
        if let reason = model.exportJob?.error { log("export error=\(reason.prefix(400))") }
        if let link = model.exportJob?.result?["download_url"]?.stringValue, let url = URL(string: link) {
            let downloadStart = Date()
            if let pair = try? await URLSession.shared.download(from: url) {
                let size = (try? pair.0.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                log(String(format: "export download_to_phone=%.1f bytes=%d", Date().timeIntervalSince(downloadStart), size))
            }
        }
        log("done error=\(model.errorMessage ?? "none")")
    }

    static func makePhoto() -> URL? {
        let size = CGSize(width: 900, height: 600)
        let image = UIGraphicsImageRenderer(size: size).image { context in
            let colors = [UIColor(red: 0.1, green: 0.9, blue: 1, alpha: 1).cgColor, UIColor(red: 1, green: 0.13, blue: 0.84, alpha: 1).cgColor]
            if let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: colors as CFArray, locations: [0, 1]) {
                context.cgContext.drawLinearGradient(gradient, start: .zero, end: CGPoint(x: size.width, y: size.height), options: [])
            }
            let text = "50% OFF" as NSString
            text.draw(at: CGPoint(x: 220, y: 230), withAttributes: [.font: UIFont.boldSystemFont(ofSize: 110), .foregroundColor: UIColor.white])
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("probe-photo-\(UUID().uuidString.prefix(6)).jpg")
        guard let data = image.jpegData(compressionQuality: 0.9), (try? data.write(to: url)) != nil else { return nil }
        return url
    }

    /// A short 360x640 H.264 video whose colour changes every half second.
    static func makeVideo(seconds: Int) async -> URL? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("probe-video-\(UUID().uuidString.prefix(6)).mp4")
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return nil }
        let width = 360, height = 640, fps: Int32 = 30
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
        writer.add(input)
        guard writer.startWriting() else { return nil }
        writer.startSession(atSourceTime: .zero)
        let palette: [(UInt8, UInt8, UInt8)] = [(255, 80, 40), (40, 200, 255), (120, 60, 240), (40, 220, 120)]
        for frame in 0..<(Int(fps) * seconds) {
            while !input.isReadyForMoreMediaData { try? await Task.sleep(nanoseconds: 5_000_000) }
            guard let pool = adaptor.pixelBufferPool else { return nil }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
            guard let buffer else { return nil }
            CVPixelBufferLockBaseAddress(buffer, [])
            let color = palette[(frame / 15) % palette.count]
            if let base = CVPixelBufferGetBaseAddress(buffer) {
                // One row of the colour, copied to every row (fast even in a Debug build).
                let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
                var row = [UInt8](repeating: 255, count: width * 4)
                for x in 0..<width { row[x * 4] = color.2; row[x * 4 + 1] = color.1; row[x * 4 + 2] = color.0 }
                row.withUnsafeBytes { source in
                    for y in 0..<height { memcpy(base + y * rowBytes, source.baseAddress!, width * 4) }
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: fps))
        }
        input.markAsFinished()
        await writer.finishWriting()
        return writer.status == .completed ? url : nil
    }

    static func until(_ seconds: Double, _ condition: () -> Bool) async {
        var waited = 0.0
        while !condition(), waited < seconds { await wait(0.5); waited += 0.5 }
    }
}
#endif
