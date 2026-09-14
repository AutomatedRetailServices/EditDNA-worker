import Foundation
import SwiftUI

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

enum TimelineAssetCatalog {
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
