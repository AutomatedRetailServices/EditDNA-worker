import SwiftUI

struct VisualTimelineView: View {
    @ObservedObject var model: DraftEditorViewModel
    @State private var selectedClipID: String?
    @State private var zoom: CGFloat = 1.0

    private var assetCatalog: [String: SourceTimelineAssets] {
        TimelineAssetCatalog.build(from: model.snapshot)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Timeline").font(.headline)
                Spacer()
                Text("\(model.selectedClips.count) clips")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            ScrollView(.horizontal) {
                HStack(spacing: 4) {
                    ForEach(Array(model.selectedClips.enumerated()), id: \.offset) { index, clip in
                        TimelineClipCell(
                            index: index,
                            clip: clip,
                            assets: assetCatalog[clip["source_asset_id"]?.stringValue ?? ""],
                            zoom: zoom,
                            selected: selectedClipID == clip["clip_id"]?.stringValue
                        )
                        .onTapGesture {
                            selectedClipID = clip["clip_id"]?.stringValue
                        }
                    }
                }
                .padding(.vertical, 4)
            }
            .scrollIndicators(.hidden)
            .simultaneousGesture(
                MagnifyGesture()
                    .onChanged { value in
                        zoom = min(2.5, max(0.65, value.magnification))
                    }
            )

            if let selectedClipID,
               let clip = model.selectedClips.first(where: { $0["clip_id"]?.stringValue == selectedClipID }) {
                TimelineClipInspector(model: model, clip: clip, assets: assetCatalog[clip["source_asset_id"]?.stringValue ?? ""])
            } else {
                Text("Tap a clip to trim, split, edit captions or adjust audio.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal)
        .onChange(of: model.selectedClips.count) { _, _ in
            if let selectedClipID,
               !model.selectedClips.contains(where: { $0["clip_id"]?.stringValue == selectedClipID }) {
                self.selectedClipID = nil
            }
        }
    }
}

private struct TimelineClipCell: View {
    let index: Int
    let clip: [String: JSONValue]
    let assets: SourceTimelineAssets?
    let zoom: CGFloat
    let selected: Bool

    private var start: Double { clip["start"]?.doubleValue ?? 0 }
    private var end: Double { clip["end"]?.doubleValue ?? start + 1 }
    private var duration: Double { max(0.15, end - start) }
    private var width: CGFloat { min(340, max(92, CGFloat(duration) * 38 * zoom)) }

    private var frames: [TimelineFrame] {
        guard let assets else { return [] }
        let inRange = assets.frames.filter { $0.time >= start && $0.time <= end }
        if inRange.isEmpty { return Array(assets.frames.prefix(4)) }
        let maxFrames = max(2, min(8, Int(width / 44)))
        guard inRange.count > maxFrames else { return inRange }
        let stride = max(1, inRange.count / maxFrames)
        return Swift.stride(from: 0, to: inRange.count, by: stride).prefix(maxFrames).map { inRange[$0] }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            ZStack {
                RoundedRectangle(cornerRadius: 9)
                    .fill(.secondary.opacity(0.12))
                if frames.isEmpty {
                    Image(systemName: "video")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                } else {
                    HStack(spacing: 1) {
                        ForEach(frames) { frame in
                            AsyncImage(url: frame.url) { image in
                                image.resizable().scaledToFill()
                            } placeholder: {
                                Rectangle().fill(.secondary.opacity(0.15))
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .clipped()
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 9))
                }
            }
            .frame(width: width, height: 72)
            .overlay {
                RoundedRectangle(cornerRadius: 9)
                    .stroke(selected ? Color.accentColor : Color.clear, lineWidth: 3)
            }

            HStack(spacing: 6) {
                Text("#\(index + 1)").fontWeight(.semibold)
                if let role = clip["semantic_role"]?.stringValue {
                    Text(role.capitalized)
                }
                Text(String(format: "%.1fs", duration))
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            .frame(width: width, alignment: .leading)
        }
    }
}

private struct TimelineClipInspector: View {
    @ObservedObject var model: DraftEditorViewModel
    let clip: [String: JSONValue]
    let assets: SourceTimelineAssets?

    @StateObject private var waveform = WaveformLoader()
    @State private var trimStart: Double = 0
    @State private var trimEnd: Double = 0
    @State private var caption: String = ""
    @State private var splitTime: Double?

    private var clipID: String { clip["clip_id"]?.stringValue ?? "" }
    private var start: Double { clip["start"]?.doubleValue ?? 0 }
    private var end: Double { clip["end"]?.doubleValue ?? start + 1 }

    private var safeSplitTimes: [Double] {
        let words = clip["words"]?.arrayValue?.compactMap(\.objectValue) ?? []
        guard words.count >= 2 else { return [] }
        return zip(words, words.dropFirst()).compactMap { left, right in
            guard let leftEnd = left["end"]?.doubleValue,
                  let rightStart = right["start"]?.doubleValue else { return nil }
            let candidate = (leftEnd + rightStart) / 2
            guard candidate - start >= 0.15, end - candidate >= 0.15 else { return nil }
            return candidate
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !waveform.peaks.isEmpty {
                WaveformView(peaks: waveform.peaks)
                    .frame(height: 42)
                    .foregroundStyle(Color.accentColor)
            }

            TextField("Caption", text: $caption, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...3)
                .onSubmit { Task { await model.editCaption(clipID: clipID, text: caption) } }

            if end - start >= 0.30 {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Trim")
                        Spacer()
                        Text(String(format: "%.2f – %.2f", trimStart, trimEnd))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    .font(.caption)

                    Slider(value: $trimStart, in: start...max(start, trimEnd - 0.15), step: 0.01)
                    Slider(value: $trimEnd, in: min(end, trimStart + 0.15)...end, step: 0.01)
                    Button("Apply trim") {
                        Task { await model.trim(clipID: clipID, start: trimStart, end: trimEnd) }
                    }
                    .buttonStyle(.bordered)
                    .disabled(trimEnd - trimStart < 0.15 || (abs(trimStart - start) < 0.005 && abs(trimEnd - end) < 0.005))
                }
            }

            HStack {
                if let splitTime {
                    Button {
                        Task { await model.split(clipID: clipID, at: splitTime) }
                    } label: {
                        Label("Split", systemImage: "scissors")
                    }
                    .buttonStyle(.bordered)
                }

                Menu {
                    Button("Mute") { Task { await model.setAudio(clipID: clipID, muted: true) } }
                    Button("Unmute") { Task { await model.setAudio(clipID: clipID, muted: false) } }
                    Button("Volume 50%") { Task { await model.setAudio(clipID: clipID, volume: 0.5) } }
                    Button("Volume 100%") { Task { await model.setAudio(clipID: clipID, volume: 1.0) } }
                } label: {
                    Label("Audio", systemImage: "waveform")
                }
                .buttonStyle(.bordered)

                Spacer()

                Button(role: .destructive) {
                    Task { await model.remove(clipID: clipID) }
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(12)
        .background(.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
        .task(id: assets?.waveformURL) {
            await waveform.load(assets?.waveformURL)
        }
        .onAppear { resetFields() }
        .onChange(of: clipID) { _, _ in resetFields() }
        .onChange(of: start) { _, _ in resetFields() }
        .onChange(of: end) { _, _ in resetFields() }
    }

    private func resetFields() {
        trimStart = start
        trimEnd = end
        caption = clip["caption_text"]?.stringValue ?? clip["text"]?.stringValue ?? ""
        if let middle = safeSplitTimes.min(by: { abs($0 - ((start + end) / 2)) < abs($1 - ((start + end) / 2)) }) {
            splitTime = middle
        } else {
            splitTime = nil
        }
    }
}
