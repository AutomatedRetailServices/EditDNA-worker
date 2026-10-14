#if DEBUG
import Photos
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

    static func newest(_ type: PHAssetMediaType) -> PHAsset? {
        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        options.fetchLimit = 1
        return PHAsset.fetchAssets(with: type, options: options).firstObject
    }

    static func photoFile(_ asset: PHAsset) async -> URL? {
        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = true
        options.deliveryMode = .highQualityFormat
        let data: Data? = await withCheckedContinuation { continuation in
            PHImageManager.default().requestImageDataAndOrientation(for: asset, options: options) { data, _, _, _ in
                continuation.resume(returning: data)
            }
        }
        guard let data, let image = UIImage(data: data), let jpeg = image.jpegData(compressionQuality: 0.9) else { return nil }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("probe-photo-\(UUID().uuidString.prefix(6)).jpg")
        try? jpeg.write(to: url)
        return url
    }

    static func run(model: DraftEditorViewModel, select: @escaping (EditorSelection?) -> Void) async {
        await wait(4)
        let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        log("photos=\(status.rawValue)")
        guard let photoAsset = newest(.image), let videoAsset = newest(.video),
              let photo = await photoFile(photoAsset),
              let video = try? await GalleryExport.export(videoAsset, name: nil).url else {
            log("no media in Photos"); return
        }
        // 1. Photo overlay.
        var started = Date()
        model.addOverlay(fileURL: photo, at: 0.5)
        await wait(0.3)
        log(String(format: "photo_overlay shown_after=%.2f pending=%d", Date().timeIntervalSince(started), model.pendingOverlays.count))
        await until(40) { model.pendingOverlays.isEmpty }
        log(String(format: "photo_overlay uploaded_after=%.1f overlays=%d", Date().timeIntervalSince(started), model.mediaOverlays.count))
        // 2. Video overlay.
        started = Date()
        model.addOverlay(fileURL: video, at: 2.0)
        await until(60) { model.pendingOverlays.isEmpty }
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
        if let link = model.exportJob?.result?["download_url"]?.stringValue, let url = URL(string: link) {
            let downloadStart = Date()
            if let pair = try? await URLSession.shared.download(from: url) {
                let size = (try? pair.0.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                log(String(format: "export download_to_phone=%.1f bytes=%d", Date().timeIntervalSince(downloadStart), size))
            }
        }
        log("done error=\(model.errorMessage ?? "none")")
    }

    static func until(_ seconds: Double, _ condition: () -> Bool) async {
        var waited = 0.0
        while !condition(), waited < seconds { await wait(0.5); waited += 0.5 }
    }
}
#endif
