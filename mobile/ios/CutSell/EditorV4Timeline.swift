import SwiftUI

/// Editor v4 timeline (Figma 1083:2559): the cursor stays in the middle and the tracks slide under
/// it (by themselves while the video plays, or with a finger); two fingers zoom. Video track with
/// real frames, the clip number and the clip's caption inside it; voice-over track; overlay track.
struct TimelineV4: View {
    @ObservedObject var model: DraftEditorViewModel
    @ObservedObject var playback: DraftPlaybackController
    @Binding var selection: EditorSelection?
    @Binding var pixelsPerSecond: CGFloat
    let onAddVoice: () -> Void
    let onAddOverlay: () -> Void

    @State private var dragStartTime: Double?
    @State private var zoomStart: CGFloat?

    static let minZoom: CGFloat = 6
    static let maxZoom: CGFloat = 240

    private var duration: Double { max(0.1, playback.duration > 0 ? playback.duration : model.timelineDuration) }

    var body: some View {
        GeometryReader { proxy in
            let width = proxy.size.width
            let center = width / 2
            let now = playback.currentTime
            let x: (Double) -> CGFloat = { seconds in center + CGFloat(seconds - now) * pixelsPerSecond }
            ZStack(alignment: .topLeading) {
                // Time + ruler (y 0...14)
                ruler(width: width, x: x)
                Text("\(EditorTime.precise(now)) / \(EditorTime.precise(duration))")
                    .font(CS.sf(11, .semibold))
                    .monospacedDigit()
                    .foregroundStyle(Color(hex: 0xF7FAFF))
                    .padding(.horizontal, 4)
                    .background(Color(hex: 0x050814))
                    .offset(x: 12, y: -2)
                    .accessibilityIdentifier("timelineTime")

                videoTrack(width: width, x: x).offset(y: 20)
                voiceTrack(width: width, x: x).offset(y: 84)
                overlayTrack(width: width, x: x).offset(y: 118)

                // Cursor fixed in the middle.
                VStack(spacing: 0) {
                    Circle()
                        .fill(LinearGradient(colors: [CS.cyan, Color(hex: 0x1769FF), Color(hex: 0x7C3AED), Color(hex: 0xFF21D7)], startPoint: .leading, endPoint: .trailing))
                        .frame(width: 12, height: 12)
                    RoundedRectangle(cornerRadius: 1).fill(Color(hex: 0xF7FAFF)).frame(width: 2, height: 132)
                }
                .offset(x: center - 6, y: 10)
                .allowsHitTesting(false)
            }
            .frame(width: width, height: 146, alignment: .topLeading)
            .clipped()
            .contentShape(Rectangle())
            .gesture(scrub)
            .simultaneousGesture(zoom)
        }
        .frame(height: 146)
        .accessibilityIdentifier("timelineV4")
    }

    // MARK: Gestures

    /// One finger slides the timeline under the fixed cursor.
    private var scrub: some Gesture {
        DragGesture(minimumDistance: 6)
            .onChanged { value in
                if dragStartTime == nil {
                    dragStartTime = playback.currentTime
                    playback.pause()
                }
                let target = (dragStartTime ?? 0) - Double(value.translation.width / pixelsPerSecond)
                playback.seek(to: max(0, min(duration, target)))
            }
            .onEnded { _ in dragStartTime = nil }
    }

    /// Two fingers zoom in and out on the same timeline.
    private var zoom: some Gesture {
        MagnificationGesture()
            .onChanged { scale in
                if zoomStart == nil { zoomStart = pixelsPerSecond }
                pixelsPerSecond = max(Self.minZoom, min(Self.maxZoom, (zoomStart ?? 30) * scale))
            }
            .onEnded { _ in zoomStart = nil }
    }

    // MARK: Ruler

    private func ruler(width: CGFloat, x: @escaping (Double) -> CGFloat) -> some View {
        let step = Self.labelStep(pixelsPerSecond)
        let first = max(0, Int((playback.currentTime - Double(width / 2 / pixelsPerSecond)) / step) - 1)
        let last = Int((playback.currentTime + Double(width / 2 / pixelsPerSecond)) / step) + 1
        return ZStack(alignment: .topLeading) {
            ForEach(first...max(first, last), id: \.self) { index in
                let seconds = Double(index) * step
                let px = x(seconds)
                if seconds <= duration + 0.01, px > 112, px < width - 6 {
                    Text(Self.label(seconds))
                        .font(CS.sf(9, .medium))
                        .foregroundStyle(Color(hex: 0x6F86AE))
                        .fixedSize()
                        .offset(x: px - 6, y: 0)
                }
            }
        }
        .frame(width: width, height: 14, alignment: .topLeading)
    }

    /// Seconds between ruler labels: about one label every 60 points.
    static func labelStep(_ pixelsPerSecond: CGFloat) -> Double {
        for step in [0.5, 1, 2, 5, 10, 15, 30, 60] where Double(pixelsPerSecond) * step >= 55 { return step }
        return 120
    }

    static func label(_ seconds: Double) -> String {
        if seconds < 60 {
            return seconds == seconds.rounded() ? "\(Int(seconds))s" : String(format: "%.1fs", seconds)
        }
        return EditorTime.clock(seconds)
    }

    // MARK: Video track (frames + number + caption inside each clip)

    private func videoTrack(width: CGFloat, x: @escaping (Double) -> CGFloat) -> some View {
        let windows = EditorTracks.clipWindows(model)
        let clips = model.selectedClips
        return ZStack(alignment: .topLeading) {
            ForEach(Array(windows.enumerated()), id: \.element.clipID) { index, window in
                let left = x(window.start)
                let clipWidth = max(4, CGFloat(window.end - window.start) * pixelsPerSecond - 3)
                if left + clipWidth > -20, left < width + 20 {
                    let clip = clips.first { $0["clip_id"]?.stringValue == window.clipID }
                    let caption = model.captionsEnabled
                        ? (clip?["caption_text"]?.stringValue ?? clip?["text"]?.stringValue ?? "")
                        : ""
                    ClipBlockV4(
                        url: model.playbackURL(forSource: window.sourceID),
                        sourceStart: window.sourceStart,
                        length: window.end - window.start,
                        number: index + 1,
                        caption: caption,
                        width: clipWidth,
                        isSelected: selection == .video(window.clipID)
                    )
                    .offset(x: left)
                    .onTapGesture {
                        selection = selection == .video(window.clipID) ? nil : .video(window.clipID)
                    }
                }
            }
        }
        .frame(width: width, height: 58, alignment: .topLeading)
        .accessibilityIdentifier("videoTrack")
    }

    // MARK: Voice-over track (always visible)

    private func voiceTrack(width: CGFloat, x: @escaping (Double) -> CGFloat) -> some View {
        let placements = model.timelineComposition?.voiceOverPlacements ?? []
        return ZStack(alignment: .topLeading) {
            if placements.isEmpty {
                Button(action: onAddVoice) {
                    Text("+  Voice over")
                        .font(CS.sf(11, .semibold))
                        .foregroundStyle(Color(hex: 0xB89CFF))
                        .padding(.leading, 9)
                        .frame(width: width - 32, height: 28, alignment: .leading)
                        .background(Color(hex: 0x0A1326), in: RoundedRectangle(cornerRadius: 8))
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(style: StrokeStyle(lineWidth: 1, dash: [4, 3])).foregroundStyle(Color(hex: 0x7C3AED, opacity: 0.8)))
                }
                .buttonStyle(.plain)
                .offset(x: 16)
                .accessibilityIdentifier("voiceTrackEmpty")
            }
            ForEach(Array(placements.enumerated()), id: \.element.id) { index, placement in
                let left = x(placement.timelineStartSec)
                let blockWidth = max(8, CGFloat(placement.timelineEndSec - placement.timelineStartSec) * pixelsPerSecond)
                if left + blockWidth > -20, left < width + 20 {
                    VoiceBlockV4(
                        title: "Voice \(index + 1)",
                        assetID: placement.assetID,
                        width: blockWidth,
                        isSelected: selection == .voiceOver(placement.placementID)
                    )
                    .offset(x: left)
                    .onTapGesture {
                        selection = selection == .voiceOver(placement.placementID) ? nil : .voiceOver(placement.placementID)
                    }
                }
            }
        }
        .frame(width: width, height: 28, alignment: .topLeading)
        .accessibilityIdentifier("voiceTrack")
    }

    // MARK: Overlay track (text, photo, B-roll)

    private struct OverlayItem: Identifiable {
        let id: String
        let start: Double
        let end: Double
        let title: String
        let kind: Kind
        enum Kind { case text, media, broll }
    }

    private var overlayItems: [OverlayItem] {
        var items: [OverlayItem] = []
        for item in model.textOverlays {
            let start = item["start"]?.doubleValue ?? 0
            items.append(OverlayItem(id: item["overlay_id"]?.stringValue ?? UUID().uuidString, start: start,
                                     end: item["end"]?.doubleValue ?? start + 1, title: item["text"]?.stringValue ?? "Text", kind: .text))
        }
        for item in model.mediaOverlays {
            let start = item["start"]?.doubleValue ?? 0
            items.append(OverlayItem(id: item["overlay_id"]?.stringValue ?? UUID().uuidString, start: start,
                                     end: item["end"]?.doubleValue ?? start + 1,
                                     title: (item["kind"]?.stringValue ?? "photo").capitalized, kind: .media))
        }
        for placement in model.timelineComposition?.brollPlacements ?? [] {
            items.append(OverlayItem(id: placement.placementID, start: placement.timelineStartSec,
                                     end: placement.timelineEndSec, title: "B-roll", kind: .broll))
        }
        return items
    }

    private func isSelected(_ item: OverlayItem) -> Bool {
        switch item.kind {
        case .text: return selection == .overlay(item.id, isText: true)
        case .media: return selection == .overlay(item.id, isText: false)
        case .broll: return selection == .broll(item.id)
        }
    }

    private func select(_ item: OverlayItem) {
        let target: EditorSelection
        switch item.kind {
        case .text: target = .overlay(item.id, isText: true)
        case .media: target = .overlay(item.id, isText: false)
        case .broll: target = .broll(item.id)
        }
        selection = selection == target ? nil : target
    }

    private static func fill(_ kind: OverlayItem.Kind) -> AnyShapeStyle {
        switch kind {
        case .media: return AnyShapeStyle(Color(hex: 0xF0F2F7))
        case .broll:
            return AnyShapeStyle(LinearGradient(colors: [Color(hex: 0xD61FFF, opacity: 0.7), Color(hex: 0x1A6BFF, opacity: 0.6)],
                                                startPoint: .leading, endPoint: .trailing))
        case .text:
            return AnyShapeStyle(LinearGradient(colors: [Color(hex: 0x0A5CDB), Color(hex: 0x1A1F5C)], startPoint: .leading, endPoint: .trailing))
        }
    }

    private func overlayTrack(width: CGFloat, x: @escaping (Double) -> CGFloat) -> some View {
        let items = overlayItems
        return ZStack(alignment: .topLeading) {
            if items.isEmpty {
                Button(action: onAddOverlay) {
                    Text("+  Text · photo · B-roll")
                        .font(CS.sf(11, .semibold))
                        .foregroundStyle(CS.cyan)
                        .padding(.leading, 9)
                        .frame(width: width - 32, height: 24, alignment: .leading)
                        .background(Color(hex: 0x0A1326), in: RoundedRectangle(cornerRadius: 8))
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(style: StrokeStyle(lineWidth: 1, dash: [4, 3])).foregroundStyle(Color(hex: 0x6F86AE)))
                }
                .buttonStyle(.plain)
                .offset(x: 16)
                .accessibilityIdentifier("overlayTrackEmpty")
            } else {
                ForEach(items) { item in
                    let left = x(item.start)
                    let itemWidth = max(10, CGFloat(item.end - item.start) * pixelsPerSecond - 2)
                    if left + itemWidth > -20, left < width + 20 {
                        Text(item.title)
                            .font(CS.sf(10, .semibold))
                            .foregroundStyle(item.kind == .media ? Color(hex: 0xD10F1A) : .white)
                            .lineLimit(1)
                            .padding(.horizontal, 6)
                            .frame(width: itemWidth, height: 24, alignment: .leading)
                            .background(Self.fill(item.kind), in: RoundedRectangle(cornerRadius: 7))
                            .overlay(RoundedRectangle(cornerRadius: 7).stroke(isSelected(item) ? CS.cyan : .clear, lineWidth: 2))
                            .offset(x: left)
                            .onTapGesture { select(item) }
                            .accessibilityIdentifier(item.kind == .broll ? "brollBlock" : "overlayBlock")
                    }
                }
                Button(action: onAddOverlay) {
                    Image(systemName: "plus").font(.system(size: 12, weight: .heavy)).foregroundStyle(Color(hex: 0x0B1020))
                        .frame(width: 24, height: 24).background(Color(hex: 0xE6E9EF), in: RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                .offset(x: width - 34)
                .accessibilityLabel("Add text, photo or B-roll")
            }
        }
        .frame(width: width, height: 24, alignment: .topLeading)
    }
}

/// One clip of the video track: real frames, its number, and its caption inside it.
private struct ClipBlockV4: View {
    let url: URL?
    let sourceStart: Double
    let length: Double
    let number: Int
    let caption: String
    let width: CGFloat
    let isSelected: Bool

    var body: some View {
        let tiles = max(1, min(12, Int((width / 34).rounded(.up))))
        let tileWidth = width / CGFloat(tiles)
        ZStack(alignment: .topLeading) {
            HStack(spacing: 0) {
                ForEach(0..<tiles, id: \.self) { index in
                    ClipFrameImage(url: url, seconds: sourceStart + 0.2 + length * Double(index) / Double(tiles))
                        .frame(width: tileWidth, height: 38)
                        .clipped()
                }
            }
            .frame(width: width, height: 38, alignment: .leading)
            VStack(spacing: 0) {
                Spacer(minLength: 0)
                Text(caption)
                    .font(CS.sf(10, .semibold))
                    .foregroundStyle(Color(hex: 0xF7FAFF))
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .padding(.horizontal, 6)
                    .frame(width: width, height: 20, alignment: .leading)
                    .background(Color(hex: 0x050814))
            }
            Text("\(number)")
                .font(CS.sf(9, .bold))
                .foregroundStyle(Color(hex: 0xF7FAFF))
                .frame(width: 16, height: 16)
                .background(Color(hex: 0x050814), in: Circle())
                .offset(x: 4, y: 4)
        }
        .frame(width: width, height: 58)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay {
            if isSelected {
                ZStack {
                    RoundedRectangle(cornerRadius: 8).stroke(CS.cyan, lineWidth: 2)
                    HStack {
                        RoundedRectangle(cornerRadius: 4).fill(Color(hex: 0xC7CFDB)).frame(width: 9)
                        Spacer()
                        RoundedRectangle(cornerRadius: 4).fill(Color(hex: 0xC7CFDB)).frame(width: 9)
                    }
                }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Clip \(number)")
    }
}

/// One voice-over on its track: violet block with the real waveform of the recording.
struct VoiceBlockV4: View {
    let title: String
    let assetID: String
    let width: CGFloat
    let isSelected: Bool
    @State private var samples: [Float] = []

    var body: some View {
        let bars = max(4, Int((width - 56) / 3.7))
        ZStack(alignment: .leading) {
            RoundedRectangle(cornerRadius: 8).fill(Color(hex: 0x7C3AED))
            HStack(spacing: 0) {
                Text("🎙 \(title)")
                    .font(CS.sf(9, .semibold))
                    .foregroundStyle(Color(hex: 0xF7FAFF))
                    .lineLimit(1)
                    .fixedSize()
                    .padding(.leading, 5)
                HStack(alignment: .center, spacing: 1.7) {
                    ForEach(0..<bars, id: \.self) { index in
                        RoundedRectangle(cornerRadius: 1)
                            .fill(CS.cyan)
                            .frame(width: 2, height: max(3, 20 * CGFloat(level(index, of: bars))))
                    }
                }
                .padding(.leading, 6)
                Spacer(minLength: 0)
            }
            .clipped()
        }
        .frame(width: width, height: 28)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(isSelected ? CS.cyan : Color(hex: 0x7C3AED), lineWidth: isSelected ? 2 : 1))
        .task(id: assetID) {
            if let url = VoiceOverStore.url(for: assetID) { samples = await Waveform.samples(url: url, count: 90) }
        }
        .accessibilityIdentifier("voiceBlock")
    }

    /// Real loudness when the recording is on this iPhone; otherwise an even placeholder pattern.
    private func level(_ index: Int, of count: Int) -> Float {
        if !samples.isEmpty {
            let position = Int(Float(index) / Float(max(1, count)) * Float(samples.count))
            return samples[min(samples.count - 1, position)]
        }
        let seed = abs((assetID.hashValue &+ index &* 7919) % 100)
        return 0.25 + Float(seed) / 140
    }
}

extension EditorTime {
    /// "00:12.4"
    static func precise(_ seconds: Double) -> String { tenths(seconds) }
}
