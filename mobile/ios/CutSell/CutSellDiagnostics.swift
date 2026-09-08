import AVFoundation
import CoreMedia
import Foundation
import UIKit
import os

/// D-133 (docs/CUTSELL_DECISIONS.md D-133) -- bounded, diagnostics-only
/// instrumentation added for ONE controlled real-device ingestion QA pass.
/// This file NEVER logs media content, transcript text, or user-identifying
/// data beyond the device model / iOS version this task's own directive
/// explicitly requires be captured. Every call site this file's functions
/// are wired into (see CameraCaptureView/NewCutView/MultipartUploadManager/
/// ProcessingView/DraftPlaybackView/FinishedExportActionsView) is a single
/// additive log call placed after an EXISTING operation already completes --
/// no control flow, timing, retry, upload, or product behavior is changed by
/// this file or its call sites. Uses `os.Logger` (structured, on-device only
/// unless the user has Console/sysdiagnose access -- no network transport of
/// its own) rather than the raw `print`/`NSLog` this app previously had none
/// of at all (D-130's own finding).
enum CutSellDiagnostics {
    private static let logger = Logger(subsystem: "ai.cutsell.app", category: "device-qa")

    /// Logs one bounded, structured diagnostic event. `fields` values must
    /// never be media content, transcript text, or free-form user text --
    /// every call site in this codebase passes only the bounded scalar
    /// fields D-133 names (device/OS identifiers, media technical metadata,
    /// job ids, durations, booleans).
    static func log(_ event: String, _ fields: [String: String] = [:]) {
        let pairs = fields.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " ")
        logger.log("\(event, privacy: .public) \(pairs, privacy: .public)")
    }

    /// Real hardware identifier (e.g. "iPhone15,2"), not the generic
    /// UIDevice.model ("iPhone") -- required by D-133's device-model field.
    static var deviceModel: String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let machineMirror = Mirror(reflecting: systemInfo.machine)
        let identifier = machineMirror.children.reduce(into: "") { partial, element in
            guard let value = element.value as? Int8, value != 0 else { return }
            partial.append(Character(UnicodeScalar(UInt8(value))))
        }
        return identifier.isEmpty ? UIDevice.current.model : identifier
    }

    static var iosVersion: String { UIDevice.current.systemVersion }
}

/// D-133 -- pure, read-only technical metadata snapshot for one media file.
/// Every field here was confirmed mechanically available via AVFoundation
/// (D-133's own "Media metadata gap" investigation); this struct/its
/// builder never changes VideoPreparation.swift's existing transcode
/// decision -- it is called ADDITIONALLY alongside it, never instead of it.
struct MediaDiagnosticsSnapshot {
    let codec: String?
    let container: String
    let durationSeconds: Double
    let width: Int
    let height: Int
    let fps: Double?
    let orientationDegrees: Int
    let mirrored: Bool?
    let fileSizeBytes: Int64
    let hasAudioTrack: Bool
    let audioSampleRate: Double?

    var logFields: [String: String] {
        var fields: [String: String] = [
            "container": container,
            "duration_s": String(format: "%.2f", durationSeconds),
            "width": String(width),
            "height": String(height),
            "orientation_deg": String(orientationDegrees),
            "file_size_bytes": String(fileSizeBytes),
            "has_audio_track": String(hasAudioTrack),
        ]
        if let codec { fields["codec"] = codec }
        if let fps { fields["fps"] = String(format: "%.2f", fps) }
        if let mirrored { fields["mirrored"] = String(mirrored) }
        if let audioSampleRate { fields["audio_sample_rate"] = String(format: "%.0f", audioSampleRate) }
        return fields
    }
}

enum MediaDiagnostics {
    /// `mirroredHint` is a heuristic (front camera == mirrored during
    /// capture preview), not a value read from the file itself -- no
    /// per-track "was this mirrored when recorded" flag exists in
    /// AVFoundation's public API for a delivered movie file. Recorded
    /// honestly as a hint, never presented as a measured fact.
    static func capture(fileURL: URL, mirroredHint: Bool? = nil) async -> MediaDiagnosticsSnapshot? {
        let asset = AVURLAsset(url: fileURL)
        guard let videoTrack = (try? await asset.loadTracks(withMediaType: .video))?.first else {
            return nil
        }

        let duration = (try? await asset.load(.duration).seconds) ?? 0
        let naturalSize = (try? await videoTrack.load(.naturalSize)) ?? .zero
        let transform = (try? await videoTrack.load(.preferredTransform)) ?? .identity
        let displaySize = naturalSize.applying(transform)
        let width = Int(abs(displaySize.width))
        let height = Int(abs(displaySize.height))
        let nominalFrameRate = try? await videoTrack.load(.nominalFrameRate)
        let orientationDegrees = rotationDegrees(from: transform)

        var codec: String?
        if let descriptions = try? await videoTrack.load(.formatDescriptions), let first = descriptions.first {
            codec = fourCCString(CMFormatDescriptionGetMediaSubType(first))
        }

        let audioTracks = (try? await asset.loadTracks(withMediaType: .audio)) ?? []
        var audioSampleRate: Double?
        if let audioTrack = audioTracks.first,
           let descriptions = try? await audioTrack.load(.formatDescriptions),
           let first = descriptions.first,
           let basicDescription = CMAudioFormatDescriptionGetStreamBasicDescription(first)?.pointee {
            audioSampleRate = basicDescription.mSampleRate
        }

        let attributes = try? FileManager.default.attributesOfItem(atPath: fileURL.path)
        let fileSize = (attributes?[.size] as? NSNumber)?.int64Value ?? 0

        return MediaDiagnosticsSnapshot(
            codec: codec,
            container: fileURL.pathExtension.lowercased(),
            durationSeconds: duration,
            width: width,
            height: height,
            fps: nominalFrameRate.map { Double($0) }.flatMap { $0 > 0 ? $0 : nil },
            orientationDegrees: orientationDegrees,
            mirrored: mirroredHint,
            fileSizeBytes: fileSize,
            hasAudioTrack: !audioTracks.isEmpty,
            audioSampleRate: audioSampleRate
        )
    }

    /// Canonical quadrant-rotation mapping for the four `preferredTransform`
    /// values real iPhone recordings actually produce (0/90/180/270). A
    /// transform outside these four exact matrices is reported as 0 rather
    /// than guessed via a float comparison.
    private static func rotationDegrees(from transform: CGAffineTransform) -> Int {
        switch (transform.a, transform.b, transform.c, transform.d) {
        case (0, 1, -1, 0): return 90
        case (0, -1, 1, 0): return 270
        case (-1, 0, 0, -1): return 180
        default: return 0
        }
    }

    private static func fourCCString(_ code: FourCharCode) -> String {
        let bytes: [UInt8] = [
            UInt8((code >> 24) & 0xff),
            UInt8((code >> 16) & 0xff),
            UInt8((code >> 8) & 0xff),
            UInt8(code & 0xff),
        ]
        return String(bytes: bytes, encoding: .ascii)?.trimmingCharacters(in: .whitespaces) ?? String(code)
    }
}
