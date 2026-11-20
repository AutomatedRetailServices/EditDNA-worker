import Foundation
import Photos

/// PO 20 Nov 2026: "el video que se graba desde la cámara del app no se guarda en el teléfono en automático".
/// Every clip recorded with the CutSell camera (talking or B-roll) is also saved to Photos, and the cut remembers it
/// as the original (OriginalLinks), so "Restoring video…" works later like for a gallery video.
/// No permission (or a refusal) never blocks the recording: the clip just is not copied.
enum CameraOriginals {
    static func keep(_ url: URL) async {
        let status = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        let allowed: Bool
        switch status {
        case .authorized, .limited: allowed = true
        case .notDetermined: allowed = await PHPhotoLibrary.requestAuthorization(for: .addOnly) == .authorized
        default: allowed = false
        }
        guard allowed, FileManager.default.fileExists(atPath: url.path) else {
            CutSellDiagnostics.log("camera_saved_to_photos", ["ok": "no", "why": allowed ? "no_file" : "no_permission"])
            return
        }
        var localID: String?
        do {
            try await PHPhotoLibrary.shared().performChanges {
                let request = PHAssetCreationRequest.forAsset()
                let options = PHAssetResourceCreationOptions()
                options.shouldMoveFile = false            // the app still uploads and edits its own copy
                request.addResource(with: .video, fileURL: url, options: options)
                localID = request.placeholderForCreatedAsset?.localIdentifier
            }
            OriginalLinks.note(file: url, photosID: localID)
            CutSellDiagnostics.log("camera_saved_to_photos", ["ok": "yes"])
        } catch {
            CutSellDiagnostics.log("camera_saved_to_photos", ["ok": "no", "why": "save_failed"])
        }
    }
}
