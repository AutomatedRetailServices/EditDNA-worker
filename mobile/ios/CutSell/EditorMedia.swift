import AVFoundation
import SwiftUI
import UIKit
import UniformTypeIdentifiers

// MARK: - Overlay media kept on this iPhone
//
// A photo or video picked for an overlay is copied here first, so it shows on the video at once
// and survives a failed upload (it is uploaded again from this copy). Once uploaded, the server's
// `uri` points to the same file, so the preview keeps using the local copy (no download).

enum OverlayMediaStore {
    static var folder: URL {
        let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CutSell/OverlayMedia", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private static var indexURL: URL { folder.appendingPathComponent("index.json") }

    /// uri -> local file name (read from disk once, then kept in memory).
    private static let lock = NSLock()
    private static var cached: [String: String]?
    private static func index() -> [String: String] {
        lock.lock(); defer { lock.unlock() }
        if let cached { return cached }
        let map = (try? Data(contentsOf: indexURL)).flatMap { try? JSONDecoder().decode([String: String].self, from: $0) } ?? [:]
        cached = map
        return map
    }

    /// Copies the picked file; returns the local file name (nil if the phone can't keep it).
    static func keep(_ file: URL, id: String) -> String? {
        let ext = file.pathExtension.isEmpty ? "jpg" : file.pathExtension.lowercased()
        let name = "\(id).\(ext)"
        let target = folder.appendingPathComponent(name)
        try? FileManager.default.removeItem(at: target)
        do { try FileManager.default.copyItem(at: file, to: target) } catch { return nil }
        return name
    }

    static func fileURL(_ name: String) -> URL { folder.appendingPathComponent(name) }

    static func link(uri: String, fileName: String) {
        var map = index()
        map[uri] = fileName
        lock.lock(); cached = map; lock.unlock()
        if let data = try? JSONEncoder().encode(map) { try? data.write(to: indexURL, options: .atomic) }
    }

    /// The local copy for an uploaded overlay, if this iPhone has it.
    static func localURL(forURI uri: String) -> URL? {
        guard let name = index()[uri] else { return nil }
        let url = fileURL(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }
}

enum EditorMediaKind: String { case photo, video }

enum EditorMediaInfo {
    static func kind(of url: URL) -> EditorMediaKind {
        let type = UTType(filenameExtension: url.pathExtension.lowercased())
        return (type?.conforms(to: .movie) ?? false) || (type?.conforms(to: .video) ?? false) ? .video : .photo
    }

    /// Width / height as shown (rotation applied); 9/16 when unknown.
    static func aspect(of url: URL, kind: EditorMediaKind) async -> Double {
        switch kind {
        case .photo:
            guard let image = UIImage(contentsOfFile: url.path), image.size.height > 0 else { return 1 }
            return image.size.width / image.size.height
        case .video:
            guard let track = (try? await AVURLAsset(url: url).loadTracks(withMediaType: .video))?.first,
                  let loaded = try? await track.load(.naturalSize, .preferredTransform) else { return 9.0 / 16 }
            let shown = loaded.0.applying(loaded.1)
            let w = abs(shown.width), h = abs(shown.height)
            return h > 0 ? w / h : 9.0 / 16
        }
    }

    static func duration(of url: URL) async -> Double {
        guard let time = try? await AVURLAsset(url: url).load(.duration) else { return 0 }
        let seconds = CMTimeGetSeconds(time)
        return seconds.isFinite ? max(0, seconds) : 0
    }

    static func contentType(for url: URL) -> String {
        UTType(filenameExtension: url.pathExtension.lowercased())?.preferredMIMEType ?? "application/octet-stream"
    }
}

/// An overlay picked a moment ago that is still uploading (shown on the video and on the overlay
/// row at once, with its upload %); on failure it stays here and is uploaded again.
struct PendingOverlay: Identifiable, Equatable {
    let id: String
    let fileName: String
    let kind: EditorMediaKind
    var start: Double
    var end: Double
    var sourceDuration: Double?
    var x: Double = 0.5
    var y: Double = 0.5
    var width: Double = 0.4
    var progress: Double = 0
    var failed = false
    var attempts = 0

    var fileURL: URL { OverlayMediaStore.fileURL(fileName) }
}

/// A full-screen B-roll video being added to the main track (uploading).
struct PendingBroll: Equatable {
    let id: String
    var progress: Double
    var failed = false
}

// MARK: - Upload with progress (single PUT or POST form)

final class UploadProgressDelegate: NSObject, URLSessionTaskDelegate {
    let onProgress: @Sendable (Double) -> Void
    init(onProgress: @escaping @Sendable (Double) -> Void) { self.onProgress = onProgress }
    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64,
                    totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        guard totalBytesExpectedToSend > 0 else { return }
        onProgress(Double(totalBytesSent) / Double(totalBytesExpectedToSend))
    }
}

extension OverlayUploadManager {
    /// Same upload as `upload(fileURL:projectID:session:)`, reporting the % sent.
    func upload(fileURL: URL, projectID: String, session: CutSellSession,
                progress: @escaping @Sendable (Double) -> Void) async throws -> OverlayPresignResponse {
        let values = try fileURL.resourceValues(forKeys: [.fileSizeKey])
        guard let size = values.fileSize, size > 0 else { throw OverlayUploadError.invalidFile }
        struct Body: Encodable {
            let project_id: String
            let user_id: String
            let original_name: String
            let content_type: String
            let size_bytes: Int
        }
        let presign: OverlayPresignResponse = try await APIClient.shared.request(
            "/v1/overlays/uploads/presign", method: "POST",
            body: Body(project_id: projectID, user_id: session.userID, original_name: fileURL.lastPathComponent,
                       content_type: EditorMediaInfo.contentType(for: fileURL), size_bytes: size)
        )
        guard let uploadURL = URL(string: presign.uploadURL) else { throw OverlayUploadError.invalidUploadURL }
        let delegate = UploadProgressDelegate(onProgress: progress)
        if DirectPutUpload.isPut(presign.method) {
            var request = URLRequest(url: uploadURL)
            request.httpMethod = "PUT"
            request.setValue(presign.contentType, forHTTPHeaderField: "Content-Type")
            for (key, value) in presign.headers ?? [:] { request.setValue(value, forHTTPHeaderField: key) }
            let (_, response) = try await URLSession.shared.upload(for: request, fromFile: fileURL, delegate: delegate)
            let status = (response as? HTTPURLResponse)?.statusCode ?? -1
            guard (200..<300).contains(status) else { throw OverlayUploadError.uploadFailed(status) }
            progress(1)
            return presign
        }
        // Presigned POST form: fields first, the file last.
        let boundary = "CutSellOverlay-\(UUID().uuidString)"
        var request = URLRequest(url: uploadURL)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        for key in presign.fields.keys.sorted() {
            body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(key)\"\r\n\r\n\(presign.fields[key] ?? "")\r\n".utf8))
        }
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"\(fileURL.lastPathComponent)\"\r\nContent-Type: \(presign.contentType)\r\n\r\n".utf8))
        body.append(try Data(contentsOf: fileURL))
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))
        let (_, response) = try await URLSession.shared.upload(for: request, from: body, delegate: delegate)
        let status = (response as? HTTPURLResponse)?.statusCode ?? -1
        guard (200..<300).contains(status) else { throw OverlayUploadError.uploadFailed(status) }
        progress(1)
        return presign
    }
}

// MARK: - Overlay media on screen

/// The overlay's real picture: the photo, or the video's frame at `seconds` of it.
struct OverlayMediaPicture: View {
    let url: URL?
    let kind: EditorMediaKind
    var seconds: Double = 0
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Color.black.opacity(0.2)
            if let image { Image(uiImage: image).resizable().scaledToFill() }
        }
        .task(id: "\(url?.path ?? "")|\(kind == .video ? Int((seconds * 5).rounded()) : 0)") {
            guard let url else { return }
            switch kind {
            case .photo:
                image = await Task.detached(priority: .userInitiated) { UIImage(contentsOfFile: url.path) }.value
            case .video:
                if let frame = await ClipFrameCache.shared.frame(url: url, at: seconds) { image = frame }
            }
        }
    }
}

// MARK: - Magnets for dragging on the timeline

enum TimelineSnap {
    /// Points within this distance of a magnet snap to it.
    static let distancePoints: CGFloat = 10

    /// `value` snapped to the nearest magnet within `distancePoints` (at `pixelsPerSecond`).
    static func snap(_ value: Double, to magnets: [Double], pixelsPerSecond: CGFloat) -> (value: Double, snapped: Double?) {
        let limit = Double(distancePoints / max(1, pixelsPerSecond))
        guard let nearest = magnets.min(by: { abs($0 - value) < abs($1 - value) }), abs(nearest - value) <= limit else {
            return (value, nil)
        }
        return (nearest, nearest)
    }

    /// Start for an item of `length` dragged to `proposed`: its start or its end sticks to a magnet.
    static func snapStart(_ proposed: Double, length: Double, magnets: [Double], pixelsPerSecond: CGFloat) -> (start: Double, snapped: Double?) {
        let byStart = snap(proposed, to: magnets, pixelsPerSecond: pixelsPerSecond)
        let byEnd = snap(proposed + length, to: magnets, pixelsPerSecond: pixelsPerSecond)
        switch (byStart.snapped, byEnd.snapped) {
        case (let s?, let e?):
            return abs(s - proposed) <= abs(e - (proposed + length)) ? (s, s) : (e - length, e)
        case (let s?, nil): return (s, s)
        case (nil, let e?): return (e - length, e)
        default: return (proposed, nil)
        }
    }

}

/// Where a lifted clip lands among the others: the index whose slot centre is nearest the finger.
enum ClipReorder {
    /// `lengths` of the clips in order; the lifted one (`from`) is moved so its centre sits at `center`
    /// (timeline seconds). Returns the new order of indexes.
    static func order(lengths: [Double], from: Int, center: Double) -> [Int] {
        var rest = Array(lengths.indices)
        rest.remove(at: from)
        var best = 0
        var bestDistance = Double.greatestFiniteMagnitude
        for slot in 0...rest.count {
            var cursor = 0.0
            for index in rest.prefix(slot) { cursor += lengths[index] }
            let slotCenter = cursor + lengths[from] / 2
            let distance = abs(slotCenter - center)
            if distance < bestDistance { bestDistance = distance; best = slot }
        }
        var out = rest
        out.insert(from, at: best)
        return out
    }
}

extension DraftEditorViewModel {
    /// Every photo / video overlay to draw on the video: those on the draft (from the copy kept on
    /// this iPhone, else the server's link) and those still uploading.
    var overlaysOnScreen: [OverlayOnScreen] {
        var out: [OverlayOnScreen] = mediaOverlays.compactMap { item in
            guard let id = item["overlay_id"]?.stringValue else { return nil }
            let uri = item["uri"]?.stringValue ?? ""
            let kind = EditorMediaKind(rawValue: item["kind"]?.stringValue ?? "photo") ?? .photo
            let url = OverlayMediaStore.localURL(forURI: uri)
                ?? (item["preview_url"]?.stringValue ?? item["playback_url"]?.stringValue).flatMap(URL.init(string:))
            let start = item["start"]?.doubleValue ?? 0
            return OverlayOnScreen(
                id: id, kind: kind, url: url,
                x: item["x"]?.doubleValue ?? 0.5, y: item["y"]?.doubleValue ?? 0.5,
                width: item["width"]?.doubleValue ?? 0.4,
                start: start, end: item["end"]?.doubleValue ?? start + 1,
                sourceStart: item["source_start"]?.doubleValue ?? 0
            )
        }
        for item in pendingOverlays {
            out.append(OverlayOnScreen(id: item.id, kind: item.kind, url: item.fileURL, x: item.x, y: item.y,
                                       width: item.width, start: item.start, end: item.end, sourceStart: 0,
                                       uploading: item.failed ? nil : item.progress, failed: item.failed))
        }
        return out
    }
}

extension DraftEditorViewModel {
    /// An export is being prepared or rendered right now.
    var isExporting: Bool {
        if exportPreparing != nil { return true }
        guard let job = exportJob else { return false }
        return !["finished", "failed", "canceled"].contains(job.state)
    }
}
