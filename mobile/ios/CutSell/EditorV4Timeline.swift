import SwiftUI

/// Editor v4 timeline (Figma 1083:2559, approved 13 Oct 2026). Like CapCut, it starts at the left:
/// a column of row icons (speaker = mute the video's sound · mic · layers) and right after it the
/// fixed white cursor. The tracks slide under the cursor by themselves while the video plays, or
/// with a finger; two fingers zoom. Rows: video (frames, the clip's caption on top, its audio at
/// the bottom, Auto Zoom ◇ keyframes), voice-over, overlay. "+" only on rows that have something.
struct TimelineV4: View {
    @ObservedObject var model: DraftEditorViewModel
    @ObservedObject var playback: DraftPlaybackController
    @Binding var selection: EditorSelection?
    @Binding var pixelsPerSecond: CGFloat
    let onAddVideo: () -> Void
    let onAddVoice: () -> Void
    let onAddOverlay: () -> Void
    let onToggleVideoSound: () -> Void

    @State private var dragStartTime: Double?
    @State private var zoomStart: CGFloat?
    @State private var keyframeDrag: (id: String, dx: CGFloat)?
    /// A clip edge being dragged (trim by the edges).
    @State private var trimDrag: TrimDragState?

    static let height: CGFloat = 162
    static let minZoom: CGFloat = 6
    static let maxZoom: CGFloat = 240
    /// Where the cursor stands (right after the icon column), as in the Figma.
    static let cursorX: CGFloat = 54

    // Rows, measured from the top of the timeline area (Figma y 615).
    private static let videoBandY: CGFloat = 37
    private static let videoY: CGFloat = 40
    private static let voiceBandY: CGFloat = 90
    private static let voiceY: CGFloat = 93
    private static let overlayBandY: CGFloat = 126
    private static let overlayY: CGFloat = 129

    private var duration: Double { max(0.1, playback.duration > 0 ? playback.duration : model.timelineDuration) }

    var body: some View {
        GeometryReader { proxy in
            let width = proxy.size.width
            let now = playback.currentTime
            let pps = pixelsPerSecond
            // The tracks slide under the fixed cursor until the end of the cut reaches the right side;
            // from there the cut stays in view and the cursor walks to its end (never an empty timeline).
            let scroll = Self.scrollTime(now: now, duration: duration, width: width, pixelsPerSecond: pps)
            let cursorAt = Self.cursorX + CGFloat(now - scroll) * pps
            let x: (Double) -> CGFloat = { seconds in Self.cursorX + CGFloat(seconds - scroll) * pps }
            ZStack(alignment: .topLeading) {
                Color(hex: 0x081226)
                timeLabel(now).offset(x: 16, y: 6)
                ruler(width: width, x: x).offset(y: 24)
                // Row bands: only from the cursor to the right.
                band(width: width, y: Self.videoBandY, height: 56)
                band(width: width, y: Self.voiceBandY, height: 36)
                band(width: width, y: Self.overlayBandY, height: 36)

                ZStack(alignment: .topLeading) {
                    videoTrack(width: width, x: x).offset(y: Self.videoY)
                    voiceTrack(width: width, x: x).offset(y: Self.voiceY)
                    overlayTrack(width: width, x: x).offset(y: Self.overlayY)
                    keyframes(width: width, x: x).offset(y: Self.videoY)
                }
                .frame(width: width, height: Self.height, alignment: .topLeading)
                .mask(alignment: .topLeading) {
                    Rectangle().frame(width: max(0, width - Self.cursorX + 2), height: Self.height).offset(x: Self.cursorX - 2)
                }

                // Add video: always there, small, at the right end of the video row.
                plusButton(action: onAddVideo, label: "Add video")
                    .offset(x: width - 34, y: Self.videoY + 13)

                rowIcons

                // The fixed cursor: from the top of the video track down to the overlay row.
                RoundedRectangle(cornerRadius: 1).fill(Color.white)
                    .frame(width: 2, height: 119)
                    .offset(x: cursorAt - 1, y: Self.videoBandY + 1)
                    .allowsHitTesting(false)
            }
            .frame(width: width, height: Self.height, alignment: .topLeading)
            .clipped()
            .contentShape(Rectangle())
            .gesture(scrub)
            .simultaneousGesture(zoom)
        }
        .frame(height: Self.height)
        .overlay(alignment: .top) { Rectangle().fill(CS.borderDefault).frame(height: 1).offset(y: -1) }
        .accessibilityIdentifier("timelineV4")
    }

    /// Time shown at the cursor's resting place: the playhead, but never so far that the cut's end
    /// leaves the right part of the timeline (the room before the video "+").
    static func scrollTime(now: Double, duration: Double, width: CGFloat, pixelsPerSecond: CGFloat) -> Double {
        let visible = Double(max(40, width - cursorX - 40) / pixelsPerSecond)
        return min(now, max(0, duration - visible))
    }

    private func band(width: CGFloat, y: CGFloat, height: CGFloat) -> some View {
        Rectangle().fill(Color(hex: 0x050814))
            .frame(width: max(0, width - Self.cursorX), height: height)
            .offset(x: Self.cursorX, y: y)
            .allowsHitTesting(false)
    }

    private func timeLabel(_ now: Double) -> some View {
        HStack(spacing: 4) {
            Text(EditorTime.clock(now)).font(.system(size: 12, weight: .semibold)).foregroundStyle(CS.textPrimaryF)
            Text("/ \(EditorTime.clock(duration))").font(.system(size: 12, weight: .medium)).foregroundStyle(CS.textMutedF)
        }
        .monospacedDigit()
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("timelineTime")
    }

    private func plusButton(action: @escaping () -> Void, label: String) -> some View {
        Button(action: action) {
            Text("+").font(.system(size: 18, weight: .medium)).foregroundStyle(Color(hex: 0x050814))
                .frame(width: 24, height: 24)
                .background(Color.white, in: RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }

    // MARK: Row icons (column at the left edge)

    private var rowIcons: some View {
        let muted = model.videoTrackVolume <= 0.001
        return ZStack(alignment: .topLeading) {
            rowIcon("Ed4RowSpeaker", label: muted ? "Turn the video sound on" : "Mute the video sound", action: onToggleVideoSound)
                .overlay {
                    if muted {
                        Rectangle().fill(CS.error).frame(width: 2, height: 24).rotationEffect(.degrees(-45)).allowsHitTesting(false)
                    }
                }
                .offset(x: 12, y: 49)
                .accessibilityIdentifier("rowVideoSound")
            rowIcon("Ed4RowMic", label: "Voice-over", action: onAddVoice)
                .offset(x: 12, y: 92)
            rowIcon("Ed4RowLayers", label: "Overlay", action: onAddOverlay)
                .offset(x: 12, y: 128)
        }
    }

    private func rowIcon(_ image: String, label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(image).resizable().frame(width: 18, height: 18)
                .frame(width: 32, height: 32)
                .background(CS.surfaceCard, in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(CS.borderDefault, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }

    // MARK: Gestures

    /// One finger slides the tracks under the fixed cursor.
    private var scrub: some Gesture {
        DragGesture(minimumDistance: 6)
            .onChanged { value in
                guard keyframeDrag == nil else { return }
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

    // MARK: Ruler: "00:00 • 00:02 • 00:04" sliding with the tracks

    private func ruler(width: CGFloat, x: @escaping (Double) -> CGFloat) -> some View {
        let step = Self.labelStep(pixelsPerSecond)
        let now = Self.scrollTime(now: playback.currentTime, duration: duration, width: width, pixelsPerSecond: pixelsPerSecond)
        let first = max(0, Int((now - Double(Self.cursorX / pixelsPerSecond)) / step) - 1)
        let last = Int((now + Double(width / pixelsPerSecond)) / step) + 1
        return ZStack(alignment: .topLeading) {
            ForEach(first...max(first, last), id: \.self) { index in
                let seconds = Double(index) * step
                let px = x(seconds)
                if seconds <= duration + 0.01, px > 38 + 12, px < width - 14 {
                    Text(EditorTime.clock(seconds))
                        .font(.system(size: 10, weight: .medium)).monospacedDigit()
                        .foregroundStyle(CS.textMutedF)
                        .fixedSize()
                        .position(x: px, y: 6)
                }
                let mid = x(seconds + step / 2)
                if seconds + step / 2 <= duration, mid > 44, mid < width - 4 {
                    Text("•").font(.system(size: 8, weight: .bold)).foregroundStyle(CS.textMutedF)
                        .position(x: mid, y: 6)
                }
            }
        }
        .frame(width: width, height: 14, alignment: .topLeading)
        .allowsHitTesting(false)
    }

    /// Seconds between ruler labels: about one label every 60 points (2 s at the default zoom).
    static func labelStep(_ pixelsPerSecond: CGFloat) -> Double {
        for step in [0.5, 1, 2, 5, 10, 15, 30, 60] where Double(pixelsPerSecond) * step >= 55 { return step }
        return 120
    }

    static func label(_ seconds: Double) -> String { EditorTime.clock(seconds) }

    // MARK: Video track (frames · caption on top · audio at the bottom)

    private func videoTrack(width: CGFloat, x: @escaping (Double) -> CGFloat) -> some View {
        let clips = model.selectedClips
        let windows = Self.windows(clips: clips, trim: trimDrag)
        return ZStack(alignment: .topLeading) {
            ForEach(Array(windows.enumerated()), id: \.element.clipID) { index, window in
                // While the START edge is dragged its right side stays put and the left side
                // follows the finger; the clips after it move when the finger lifts (ripple).
                let shift = Self.startDragShift(window: window, trim: trimDrag)
                let left = x(window.start) + CGFloat(shift) * pixelsPerSecond
                let clipWidth = max(4, CGFloat(window.end - window.start) * pixelsPerSecond - 2)
                if left + clipWidth > -40, left < width + 40 {
                    let clip = clips.first { $0["clip_id"]?.stringValue == window.clipID }
                    let isSelected = selection == .video(window.clipID)
                    let caption = model.captionsEnabled ? Self.caption(of: clip, window: window, trim: trimDrag, model: model) : ""
                    ClipBlockV4(
                        url: model.previewVideoURL(forSource: window.sourceID),
                        sourceStart: window.sourceStart,
                        length: window.end - window.start,
                        number: index + 1,
                        caption: caption,
                        speech: trimDrag?.clipID == window.clipID ? trimDrag?.speech ?? [] : Self.speech(of: clip),
                        muted: clip?["audio_muted"]?.boolValue == true,
                        width: clipWidth,
                        isSelected: isSelected
                    )
                    .offset(x: left)
                    .onTapGesture {
                        selection = selection == .video(window.clipID) ? nil : .video(window.clipID)
                    }
                    if isSelected {
                        trimHandle(.start, clipID: window.clipID, index: index)
                            .offset(x: left - TrimHandleV4.outside)
                        trimHandle(.end, clipID: window.clipID, index: index)
                            .offset(x: left + clipWidth - TrimHandleV4.width + TrimHandleV4.outside)
                    }
                }
            }
        }
        .frame(width: width, height: 50, alignment: .topLeading)
        .accessibilityIdentifier("videoTrack")
    }

    /// The cut's clips one after the other (no gaps), with the clip being trimmed at its new size.
    static func windows(clips: [[String: JSONValue]], trim: TrimDragState?) -> [EditorTracks.ClipWindow] {
        var cursor = 0.0
        return clips.compactMap { clip in
            let id = clip["clip_id"]?.stringValue
            var start = clip["start"]?.doubleValue ?? 0
            var end = clip["end"]?.doubleValue ?? start
            if let trim, trim.clipID == id {
                if trim.edge == .start { start = trim.value } else { end = trim.value }
            }
            let length = max(0, end - start)
            defer { cursor += length }
            guard let id, length > 0 else { return nil }
            return EditorTracks.ClipWindow(clipID: id, start: cursor, end: cursor + length, sourceStart: start,
                              sourceID: clip["source_asset_id"]?.stringValue)
        }
    }

    /// Seconds the trimmed clip is drawn to the left while its start edge is dragged outward
    /// (to the right when shortened), so its right side does not move under the finger.
    static func startDragShift(window: EditorTracks.ClipWindow, trim: TrimDragState?) -> Double {
        guard let trim, trim.clipID == window.clipID, trim.edge == .start else { return 0 }
        return trim.value - trim.original
    }

    /// The caption on the clip: the words heard in its (new) window, unless it was written by hand.
    static func caption(of clip: [String: JSONValue]?, window: EditorTracks.ClipWindow, trim: TrimDragState?,
                        model: DraftEditorViewModel) -> String {
        guard let clip else { return "" }
        if let trim, trim.clipID == window.clipID, !TrimMath.captionEditedByHand(clip) {
            return trim.caption
        }
        return clip["caption_text"]?.stringValue ?? clip["text"]?.stringValue ?? ""
    }

    // MARK: Trim handles (white, on the selected clip's edges)

    private func trimHandle(_ edge: TrimEdge, clipID: String, index: Int) -> some View {
        TrimHandleV4(edge: edge, active: trimDrag?.clipID == clipID && trimDrag?.edge == edge,
                     blocked: trimDrag?.clipID == clipID && trimDrag?.edge == edge && trimDrag?.atNeighbor == true)
            .highPriorityGesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in trimChanged(edge: edge, clipID: clipID, index: index, dx: value.translation.width) }
                    .onEnded { _ in trimEnded() }
            )
            .accessibilityLabel(edge == .start ? "Trim the start of the clip" : "Trim the end of the clip")
            .accessibilityIdentifier(edge == .start ? "trimHandleStart" : "trimHandleEnd")
    }

    private func trimChanged(edge: TrimEdge, clipID: String, index: Int, dx: CGFloat) {
        let clips = model.selectedClips
        guard index < clips.count, clips[index]["clip_id"]?.stringValue == clipID else { return }
        if trimDrag == nil {
            playback.pause()
            let clip = clips[index]
            let original = edge == .start ? (clip["start"]?.doubleValue ?? 0) : (clip["end"]?.doubleValue ?? 0)
            let sourceID = clip["source_asset_id"]?.stringValue
            let pool = model.displayDraft?.objectValue.map { TrimMath.wordPool($0, sourceID: sourceID) } ?? []
            trimDrag = TrimDragState(
                clipID: clipID, edge: edge, original: original, value: original,
                limits: TrimMath.limits(clips: clips, index: index, edge: edge,
                                        sourceDuration: model.sourceDuration(forSource: sourceID)),
                otherEdge: edge == .start ? (clip["end"]?.doubleValue ?? 0) : (clip["start"]?.doubleValue ?? 0),
                sourceID: sourceID, pool: pool,
                mirrored: clip["mirror"]?.boolValue == true || model.displayDraft?["mirror"]?.boolValue == true
            )
        }
        guard var drag = trimDrag else { return }
        let resolved = TrimMath.resolve(proposed: drag.original + Double(dx / pixelsPerSecond), edge: edge, limits: drag.limits)
        if resolved.atNeighbor && !drag.atNeighbor { TrimHaptics.neighbor() }
        if resolved.passed && !drag.passed { TrimHaptics.passed() }
        if resolved.atSourceEdge && !drag.atSourceEdge { TrimHaptics.sourceEdge() }
        drag.value = resolved.value
        drag.atNeighbor = resolved.atNeighbor
        drag.passed = resolved.passed
        drag.atSourceEdge = resolved.atSourceEdge
        trimDrag = drag
        // The exact frame at the handle and "+0,4 s" on the video.
        let added = edge == .end ? drag.value - drag.original : drag.original - drag.value
        model.trimScrub = TrimScrub(sourceID: drag.sourceID,
                                    sourceTime: edge == .end ? max(drag.otherEdge, drag.value - 1.0 / 30) : drag.value,
                                    delta: added, mirrored: drag.mirrored)
    }

    private func trimEnded() {
        guard let drag = trimDrag else { return }
        model.trimScrub = nil
        // One save when the finger lifts (the new size is on screen at once, ripple included).
        if abs(drag.value - drag.original) >= 0.001 {
            model.trimClip(clipID: drag.clipID, edge: drag.edge, to: drag.value)
        }
        trimDrag = nil
    }

    /// The spoken parts of a clip (source seconds), for its audio strip.
    static func speech(of clip: [String: JSONValue]?) -> [ClosedRange<Double>] {
        let words = clip?["words"]?.arrayValue?.compactMap(\.objectValue) ?? []
        return words.compactMap { word in
            guard let start = word["start"]?.doubleValue, let end = word["end"]?.doubleValue, end > start else { return nil }
            return start...end
        }
    }

    // MARK: Auto Zoom ◇ keyframes (tap = select · drag = move · Delete removes)

    private func keyframes(width: CGFloat, x: @escaping (Double) -> CGFloat) -> some View {
        let marks = model.zoomMarks
        let on = model.autoZoomEnabled
        return ZStack(alignment: .topLeading) {
            ForEach(marks) { mark in
                let dx = keyframeDrag?.id == mark.id ? keyframeDrag?.dx ?? 0 : 0
                let px = x(mark.time) + dx
                if px > Self.cursorX - 8, px < width + 8 {
                    ZoomDiamond(isSelected: selection == .zoom(mark.id), dimmed: !on)
                        .frame(width: 26, height: 26)
                        .contentShape(Rectangle())
                        .position(x: px, y: 25)
                        .onTapGesture { selection = selection == .zoom(mark.id) ? nil : .zoom(mark.id) }
                        .gesture(DragGesture(minimumDistance: 4)
                            .onChanged { value in
                                playback.pause()
                                selection = .zoom(mark.id)
                                keyframeDrag = (mark.id, value.translation.width)
                            }
                            .onEnded { value in
                                keyframeDrag = nil
                                let target = mark.time + Double(value.translation.width / pixelsPerSecond)
                                Task { await model.moveZoomKeyframe(id: mark.id, to: max(0, min(duration - 0.1, target))) }
                            })
                        .accessibilityLabel("Zoom at \(EditorTime.tenths(mark.time))")
                        .accessibilityIdentifier("zoomKeyframe")
                }
            }
        }
        .frame(width: width, height: 50, alignment: .topLeading)
    }

    // MARK: Voice-over track

    private func voiceTrack(width: CGFloat, x: @escaping (Double) -> CGFloat) -> some View {
        let placements = model.timelineComposition?.voiceOverPlacements ?? []
        let pending = model.pendingVoiceOvers
        let lastEnd = (placements.map(\.timelineEndSec) + pending.map(\.end)).max()
        let labels = Self.voiceLabels(
            placements.enumerated().map { index, placement in
                VoiceLabelInput(key: placement.placementID, title: "Voiceover \(index + 1)",
                                left: x(placement.timelineStartSec),
                                width: max(8, CGFloat(placement.timelineEndSec - placement.timelineStartSec) * pixelsPerSecond),
                                status: .saved)
            } + pending.enumerated().map { index, item in
                VoiceLabelInput(key: "pending_\(item.id)", title: "Voiceover \(placements.count + index + 1)",
                                left: x(item.start), width: max(8, CGFloat(item.duration) * pixelsPerSecond),
                                status: item.failed ? .failed : .uploading)
            }
        )
        return ZStack(alignment: .topLeading) {
            ForEach(Array(placements.enumerated()), id: \.element.id) { index, placement in
                let left = x(placement.timelineStartSec)
                let blockWidth = max(8, CGFloat(placement.timelineEndSec - placement.timelineStartSec) * pixelsPerSecond)
                if left + blockWidth > 0, left < width + 20 {
                    VoiceBlockV4(
                        title: "Voiceover \(index + 1)",
                        assetID: placement.assetID,
                        width: blockWidth,
                        isSelected: selection == .voiceOver(placement.placementID),
                        showsLabel: false
                    )
                    .offset(x: left)
                    .onTapGesture {
                        selection = selection == .voiceOver(placement.placementID) ? nil : .voiceOver(placement.placementID)
                    }
                }
            }
            // Saved a moment ago: on the row at once while it uploads (tap to retry if it failed).
            ForEach(Array(pending.enumerated()), id: \.element.id) { index, item in
                let left = x(item.start)
                let blockWidth = max(8, CGFloat(item.duration) * pixelsPerSecond)
                if left + blockWidth > 0, left < width + 20 {
                    VoiceBlockV4(
                        title: "Voiceover \(placements.count + index + 1)",
                        assetID: item.id,
                        width: blockWidth,
                        isSelected: false,
                        localURL: PendingVoiceOverStore.fileURL(item),
                        status: item.failed ? .failed : .uploading,
                        showsLabel: false
                    )
                    .offset(x: left)
                    .onTapGesture { if item.failed { model.retryPendingVoiceOver(id: item.id) } }
                    .accessibilityLabel(item.failed ? "Voice-over not uploaded. Tap to try again." : "Voice-over uploading")
                }
            }
            // Labels on their own layer, above every block: a short voice's label runs past its
            // end and the next voice's label moves over so no label is ever cut or covered.
            ForEach(labels, id: \.key) { label in
                if label.x + label.width > 0, label.x < width + 20 {
                    VoiceLabelV4(title: label.title, status: label.status, onTab: !label.inside)
                        .offset(x: label.x, y: 5)
                }
            }
            // "+" only when the row already has a voice: right after the last one and its label.
            if let lastEnd {
                let labelEnd = labels.map { $0.x + $0.width }.max() ?? 0
                let px = max(Self.cursorX + 4, x(lastEnd) + 6, labelEnd + 8)
                if px < width - 24 {
                    plusButton(action: onAddVoice, label: "Add voice-over")
                        .offset(x: px, y: 3)
                }
            }
        }
        .frame(width: width, height: 30, alignment: .topLeading)
        .accessibilityIdentifier("voiceTrack")
    }

    struct VoiceLabelInput {
        let key: String
        let title: String
        let left: CGFloat
        let width: CGFloat
        let status: VoiceBlockV4.Status
    }

    struct VoiceLabelPlacement {
        let key: String
        let title: String
        let x: CGFloat
        let width: CGFloat
        let status: VoiceBlockV4.Status
        /// Fits inside its own block (otherwise it sits on its own blue tab).
        let inside: Bool
    }

    /// Where each "Voiceover N" label goes: at its block's start, or just after the previous
    /// label when that one ran long. Never on top of another label, never cut.
    static func voiceLabels(_ inputs: [VoiceLabelInput]) -> [VoiceLabelPlacement] {
        var previousEnd = -CGFloat.greatestFiniteMagnitude
        return inputs.sorted { $0.left < $1.left }.map { input in
            let width = VoiceLabelV4.width(title: input.title, status: input.status)
            let x = max(input.left + 7, previousEnd + 6)
            previousEnd = x + width
            let inside = x + width <= input.left + input.width - 4
            return VoiceLabelPlacement(key: input.key, title: input.title, x: x, width: width,
                                       status: input.status, inside: inside)
        }
    }

    // MARK: Overlay track (text, photo, B-roll); empty = no "+" (added from the toolbar)

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
            RoundedRectangle(cornerRadius: 7).fill(CS.surfaceControl)
                .frame(width: max(0, width - Self.cursorX), height: 30)
                .offset(x: Self.cursorX)
                .accessibilityIdentifier(items.isEmpty ? "overlayTrackEmpty" : "overlayTrack")
            ForEach(items) { item in
                let left = x(item.start)
                let itemWidth = max(10, CGFloat(item.end - item.start) * pixelsPerSecond - 2)
                if left + itemWidth > 0, left < width + 20 {
                    Text(item.title)
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(item.kind == .media ? Color(hex: 0xD10F1A) : .white)
                        .lineLimit(1)
                        .padding(.horizontal, 7)
                        .frame(width: itemWidth, height: 30, alignment: .leading)
                        .background(Self.fill(item.kind), in: RoundedRectangle(cornerRadius: 7))
                        .overlay(RoundedRectangle(cornerRadius: 7).stroke(isSelected(item) ? CS.cyan : .clear, lineWidth: 2))
                        .offset(x: left)
                        .onTapGesture { select(item) }
                        .accessibilityIdentifier(item.kind == .broll ? "brollBlock" : "overlayBlock")
                }
            }
            // "+" once the row has something: right after the last item (Figma 1026:3400).
            if let lastEnd = items.map(\.end).max() {
                let px = max(Self.cursorX + 4, x(lastEnd) + 6)
                if px < width - 24 {
                    plusButton(action: onAddOverlay, label: "Add text, photo or B-roll")
                        .offset(x: px, y: 3)
                }
            }
        }
        .frame(width: width, height: 30, alignment: .topLeading)
    }
}

/// ◇ with a dot in the centre (the Figma keyframe), cyan when selected.
struct ZoomDiamond: View {
    let isSelected: Bool
    var dimmed = false

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 2)
                .fill(isSelected ? CS.cyan : Color(hex: 0x050814))
                .frame(width: 11, height: 11)
                .overlay(RoundedRectangle(cornerRadius: 2).stroke(isSelected ? Color.white : CS.cyan, lineWidth: 1.5))
                .rotationEffect(.degrees(45))
            Circle().fill(isSelected ? Color(hex: 0x050814) : CS.cyan).frame(width: 3.5, height: 3.5)
        }
        .opacity(dimmed ? 0.45 : 1)
        .shadow(color: .black.opacity(0.5), radius: 1.5)
    }
}

/// One clip of the video track (Figma: 50 high, white 1 pt border, 42-wide frames, the caption on
/// top and the clip's audio at the bottom).
private struct ClipBlockV4: View {
    let url: URL?
    let sourceStart: Double
    let length: Double
    let number: Int
    let caption: String
    let speech: [ClosedRange<Double>]
    let muted: Bool
    let width: CGFloat
    let isSelected: Bool

    var body: some View {
        let tiles = max(1, min(16, Int((width / 42).rounded(.up))))
        ZStack(alignment: .topLeading) {
            HStack(spacing: 0) {
                ForEach(0..<tiles, id: \.self) { index in
                    ClipFrameImage(url: url, seconds: sourceStart + 0.2 + length * Double(index) / Double(tiles))
                        .frame(width: 42, height: 50)
                        .clipped()
                }
            }
            .frame(width: width, height: 50, alignment: .leading)
            .clipped()
            if !caption.isEmpty {
                Text(caption)
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .padding(.horizontal, 5)
                    .frame(width: width, height: 15, alignment: .leading)
                    .background(Color.black.opacity(0.45))
            }
            audioStrip
                .frame(width: width, height: 10)
                .background(Color.black.opacity(0.35))
                .offset(y: 40)
        }
        .frame(width: width, height: 50)
        .clipShape(RoundedRectangle(cornerRadius: 7))
        .overlay(RoundedRectangle(cornerRadius: 7).stroke(isSelected ? CS.cyan : Color.white, lineWidth: isSelected ? 2 : 1))
        .overlay {
            if isSelected {
                HStack {
                    RoundedRectangle(cornerRadius: 3).fill(Color(hex: 0xC7CFDB)).frame(width: 7)
                    Spacer()
                    RoundedRectangle(cornerRadius: 3).fill(Color(hex: 0xC7CFDB)).frame(width: 7)
                }
                .padding(.vertical, 1)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Clip \(number)")
    }

    /// Cyan bars where the clip has speech (from the words' times), low where it is quiet.
    private var audioStrip: some View {
        let bars = max(1, Int((width - 4) / 3))
        return HStack(alignment: .bottom, spacing: 1) {
            ForEach(0..<bars, id: \.self) { index in
                RoundedRectangle(cornerRadius: 0.5)
                    .fill(muted ? Color(hex: 0x6F86AE) : CS.cyan)
                    .frame(width: 2, height: level(index, of: bars))
            }
        }
        .padding(.leading, 3)
        .frame(width: width, height: 10, alignment: .bottomLeading)
    }

    private func level(_ index: Int, of count: Int) -> CGFloat {
        let t = sourceStart + length * (Double(index) + 0.5) / Double(count)
        let spoken = speech.isEmpty || speech.contains { $0.contains(t) }
        guard spoken else { return 2 }
        let seed = abs((number &* 131 &+ index &* 7919) % 7)
        return CGFloat(3 + seed)
    }
}

/// "Voiceover N" on the mic row: inside its block, or on its own blue tab when the block is short.
struct VoiceLabelV4: View {
    let title: String
    let status: VoiceBlockV4.Status
    let onTab: Bool

    /// Room the label takes (text at 10 pt semibold, plus the upload / retry mark and the tab).
    static func width(title: String, status: VoiceBlockV4.Status) -> CGFloat {
        CGFloat(title.count) * 6.2 + (status == .saved ? 0 : 13) + 12
    }

    var body: some View {
        HStack(spacing: 4) {
            if status == .uploading {
                ProgressView().controlSize(.mini).tint(.white)
            } else if status == .failed {
                Image(systemName: "arrow.clockwise").font(.system(size: 9, weight: .bold)).foregroundStyle(.white)
            }
            Text(title)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.white)
                .lineLimit(1)
                .fixedSize()
        }
        .padding(.horizontal, onTab ? 6 : 0)
        .frame(height: 20)
        .background {
            if onTab { Capsule().fill(Color(hex: 0x1769FF)).overlay(Capsule().stroke(CS.cyan, lineWidth: 1)) }
        }
        .fixedSize()
        .allowsHitTesting(false)
        .accessibilityIdentifier("voiceLabel")
    }
}

/// State of one clip edge while it is dragged.
struct TrimDragState {
    let clipID: String
    let edge: TrimEdge
    /// The edge's place in the original before the drag (seconds).
    let original: Double
    var value: Double
    let limits: TrimMath.Limits
    /// The clip's other edge (unchanged by this drag).
    let otherEdge: Double
    let sourceID: String?
    /// Every word heard in this original (for the caption and the audio strip while dragging).
    let pool: [[String: JSONValue]]
    let mirrored: Bool
    var atNeighbor = false
    var passed = false
    var atSourceEdge = false

    private var window: (Double, Double) { edge == .start ? (value, otherEdge) : (otherEdge, value) }
    var caption: String { TrimMath.text(of: TrimMath.words(in: pool, start: window.0, end: window.1)) }
    /// The speech of the new window, so the clip's audio strip grows with it.
    var speech: [ClosedRange<Double>] {
        TrimMath.words(in: pool, start: window.0, end: window.1).compactMap { word in
            guard let s = word["start"]?.doubleValue, let e = word["end"]?.doubleValue, e > s else { return nil }
            return s...e
        }
    }
}

/// White handle on a selected clip's edge (CapCut-like): outward brings back hidden material,
/// inward shortens the clip.
struct TrimHandleV4: View {
    static let width: CGFloat = 14
    /// How far it stands outside the clip, so it is easy to grab.
    static let outside: CGFloat = 6
    let edge: TrimEdge
    let active: Bool
    let blocked: Bool

    var body: some View {
        UnevenRoundedRectangle(
            topLeadingRadius: edge == .start ? 6 : 1, bottomLeadingRadius: edge == .start ? 6 : 1,
            bottomTrailingRadius: edge == .end ? 6 : 1, topTrailingRadius: edge == .end ? 6 : 1)
            .fill(blocked ? Color(hex: 0xFFD60A) : Color.white)
            .frame(width: Self.width, height: 50)
            .overlay(Capsule().fill(Color(hex: 0x050814).opacity(0.7)).frame(width: 2.5, height: 16))
            .shadow(color: .black.opacity(active ? 0.5 : 0.3), radius: active ? 4 : 2)
            .scaleEffect(y: active ? 1.06 : 1)
            // A bigger area to grab than what is drawn.
            .contentShape(Rectangle().inset(by: -10))
    }
}

/// One voice-over on its track (Figma: blue #1769FF block, cyan border, "Voiceover 1" and the
/// recording's real waveform in cyan). The label is never cut: on a short block it runs past the
/// block's end on its own blue tab.
struct VoiceBlockV4: View {
    enum Status { case saved, uploading, failed }

    let title: String
    let assetID: String
    let width: CGFloat
    let isSelected: Bool
    var localURL: URL? = nil
    var status: Status = .saved
    /// The timeline draws the labels on their own layer (so they are never covered).
    var showsLabel: Bool = true
    @State private var samples: [Float] = []

    /// Room the label needs inside the block (text + paddings).
    static let labelRoom: CGFloat = 74

    var body: some View {
        let room = Self.labelRoom + (status == .saved ? 0 : 14)
        let labelInside = width >= room
        let waveStart: CGFloat = labelInside ? room - 6 : 4
        let bars = max(2, Int((width - waveStart - 4) / 3.4))
        ZStack(alignment: .leading) {
            RoundedRectangle(cornerRadius: 7).fill(Color(hex: 0x1769FF).opacity(status == .saved ? 1 : 0.7))
            HStack(alignment: .center, spacing: 1.4) {
                ForEach(0..<bars, id: \.self) { index in
                    RoundedRectangle(cornerRadius: 1)
                        .fill(CS.cyan)
                        .frame(width: 2, height: max(3, 17 * CGFloat(level(index, of: bars))))
                }
            }
            .padding(.leading, waveStart)
            .frame(width: width, alignment: .leading)
            .clipped()
        }
        .frame(width: width, height: 30)
        .clipShape(RoundedRectangle(cornerRadius: 7))
        .overlay(RoundedRectangle(cornerRadius: 7).stroke(borderColor, style: borderStyle))
        .overlay(alignment: .leading) { if showsLabel { label(inside: labelInside) } }
        .task(id: assetID) {
            if let url = localURL ?? VoiceOverStore.url(for: assetID) { samples = await Waveform.samples(url: url, count: 90) }
        }
        .accessibilityIdentifier(status == .saved ? "voiceBlock" : "voiceBlockPending")
    }

    private var borderColor: Color {
        if isSelected { return .white }
        return status == .failed ? CS.error : CS.cyan
    }

    private var borderStyle: StrokeStyle {
        StrokeStyle(lineWidth: isSelected ? 2 : 1, dash: status == .uploading ? [4, 3] : [])
    }

    private func label(inside: Bool) -> some View {
        HStack(spacing: 4) {
            if status == .uploading {
                ProgressView().controlSize(.mini).tint(.white)
            } else if status == .failed {
                Image(systemName: "arrow.clockwise").font(.system(size: 9, weight: .bold)).foregroundStyle(.white)
            }
            Text(title)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.white)
                .lineLimit(1)
                .fixedSize()
        }
        .padding(.horizontal, inside ? 0 : 6)
        .frame(height: 20)
        .background {
            if !inside { Capsule().fill(Color(hex: 0x1769FF)).overlay(Capsule().stroke(CS.cyan, lineWidth: 1)) }
        }
        .padding(.leading, 7)
        .fixedSize()
        .allowsHitTesting(false)
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
