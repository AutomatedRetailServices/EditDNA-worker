import Foundation
import SwiftUI

enum TimelineAssetRole: String, Codable, CaseIterable {
    case primarySource = "PRIMARY_SOURCE"
    case supplementalBroll = "SUPPLEMENTAL_BROLL"
    case voiceOver = "VOICE_OVER"
}

enum TimelineMediaKind: String, Codable {
    case video = "VIDEO"
    case audio = "AUDIO"
}

enum TimelineAssetQualificationStatus: String, Codable {
    case uploaded = "UPLOADED"
    case qualifying = "QUALIFYING"
    case ready = "READY"
    case rejected = "REJECTED"
    case failed = "FAILED"
    case deleted = "DELETED"
}

struct TimelineMediaAsset: Codable, Identifiable, Hashable {
    let assetID: String
    let role: TimelineAssetRole
    let mediaKind: TimelineMediaKind
    let durationSec: Double
    let hasAudio: Bool
    let qualificationStatus: TimelineAssetQualificationStatus
    let replacesAssetID: String?
    let createdAt: String?

    var id: String { assetID }
    var isReady: Bool { qualificationStatus == .ready }

    enum CodingKeys: String, CodingKey {
        case assetID = "asset_id"
        case role
        case mediaKind = "media_kind"
        case durationSec = "duration_sec"
        case hasAudio = "has_audio"
        case qualificationStatus = "qualification_status"
        case replacesAssetID = "replaces_asset_id"
        case createdAt = "created_at"
    }
}

struct TimelineAssetListResponse: Codable, Hashable {
    let projectID: String
    let assets: [TimelineMediaAsset]

    enum CodingKeys: String, CodingKey {
        case projectID = "project_id"
        case assets
    }
}

struct TimelineAssetLibrary: Hashable {
    let projectID: String
    let assets: [TimelineMediaAsset]

    var readyBroll: [TimelineMediaAsset] {
        assets.filter { $0.isReady && $0.role == .supplementalBroll && $0.mediaKind == .video }
    }

    var readyVoiceOvers: [TimelineMediaAsset] {
        assets.filter { $0.isReady && $0.role == .voiceOver && $0.mediaKind == .audio }
    }

    var readyPrimarySources: [TimelineMediaAsset] {
        assets.filter { $0.isReady && $0.role == .primarySource && $0.mediaKind == .video }
    }
}

enum TimelineAssetRegistryClient {
    static func list(
        projectID: String,
        userID: String,
        api: APIClient = .shared
    ) async throws -> TimelineAssetLibrary {
        let response: TimelineAssetListResponse = try await api.request(
            "/v1/projects/\(projectID)/timeline-assets",
            query: [URLQueryItem(name: "user_id", value: userID)]
        )
        guard response.projectID == projectID else { throw TimelineAssetRegistryError.projectMismatch }
        return TimelineAssetLibrary(projectID: response.projectID, assets: response.assets)
    }
}

enum TimelineAssetRegistryError: LocalizedError {
    case projectMismatch

    var errorDescription: String? {
        switch self {
        case .projectMismatch: "CutSell returned timeline assets for a different project."
        }
    }
}

struct TimelineFrame: Identifiable, Hashable {
    let time: Double
    let url: URL
    var id: String { "\(time)-\(url.absoluteString)" }
}

struct SourceTimelineAssets: Hashable {
    let sourceAssetID: String
    let frames: [TimelineFrame]
    let waveformURL: URL?
}

/// Preview-only filmstrip/waveform material embedded in a draft snapshot.
/// This is intentionally distinct from D-279's editable asset registry.
enum SourcePreviewAssetCatalog {
    static func build(from snapshot: DraftSnapshot?) -> [String: SourceTimelineAssets] {
        guard let snapshot else { return [:] }
        var output: [String: SourceTimelineAssets] = [:]
        for source in snapshot.sources {
            guard let object = source.objectValue,
                  let sourceID = object["source_asset_id"]?.stringValue else { continue }
            let assets = object["timeline_assets"]?.objectValue
            let frames = assets?["filmstrip"]?.arrayValue?.compactMap { item -> TimelineFrame? in
                guard let frame = item.objectValue,
                      let time = frame["time"]?.doubleValue,
                      let rawURL = frame["download_url"]?.stringValue,
                      let url = URL(string: rawURL) else { return nil }
                return TimelineFrame(time: time, url: url)
            } ?? []
            let waveformURL = assets?["waveform_download_url"]?.stringValue.flatMap(URL.init(string:))
            output[sourceID] = SourceTimelineAssets(
                sourceAssetID: sourceID,
                frames: frames,
                waveformURL: waveformURL
            )
        }
        return output
    }
}

@MainActor
final class WaveformLoader: ObservableObject {
    @Published private(set) var peaks: [Double] = []
    @Published private(set) var isLoading = false

    private var loadedURL: URL?

    func load(_ url: URL?) async {
        guard let url else {
            peaks = []
            loadedURL = nil
            return
        }
        if loadedURL == url, !peaks.isEmpty { return }
        isLoading = true
        defer { isLoading = false }
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                peaks = []
                return
            }
            struct Payload: Decodable { let peaks: [Double] }
            let payload = try JSONDecoder().decode(Payload.self, from: data)
            peaks = payload.peaks.map { min(1, max(0, $0)) }
            loadedURL = url
        } catch {
            peaks = []
        }
    }
}

struct WaveformView: View {
    let peaks: [Double]

    var body: some View {
        GeometryReader { proxy in
            Canvas { context, size in
                guard !peaks.isEmpty, size.width > 0, size.height > 0 else { return }
                let visibleCount = min(peaks.count, max(1, Int(size.width / 2)))
                let stride = max(1, peaks.count / visibleCount)
                let samples = Swift.stride(from: 0, to: peaks.count, by: stride).map { peaks[$0] }
                let step = size.width / CGFloat(max(1, samples.count))
                let mid = size.height / 2
                var path = Path()
                for (index, value) in samples.enumerated() {
                    let x = CGFloat(index) * step + step / 2
                    let half = max(1, CGFloat(value) * mid)
                    path.move(to: CGPoint(x: x, y: mid - half))
                    path.addLine(to: CGPoint(x: x, y: mid + half))
                }
                context.stroke(path, with: .foreground, lineWidth: 1.25)
            }
        }
        .accessibilityHidden(true)
    }
}
