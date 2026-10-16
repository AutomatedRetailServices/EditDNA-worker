import SwiftUI
import UIKit

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
    /// A panel is open under the timeline: show fewer rows (the rest slide up and down).
    var compact = false

    @State private var dragStartTime: Double?
    /// The rows under the video row slide up and down when there are more than fit.
    @State private var rowScroll: CGFloat = 0
    @State private var rowScrollStart: CGFloat?
    @State private var dragAxisVertical: Bool?
    /// A voice-over or sound edge being dragged (trim).
    @State private var audioTrim: AudioTrimState?
    @State private var zoomStart: CGFloat?
    @State private var keyframeDrag: (id: String, dx: CGFloat)?
    /// A clip edge being dragged (trim by the edges).
    @State private var trimDrag: TrimDragState?
    /// A clip lifted with a long press and dragged to another place (reorder, CapCut-like).
    @State private var liftDrag: LiftDragState?
    /// A voice-over or an overlay held and dragged sideways (new start time).
    @State private var moveDrag: MoveDragState?
    /// An overlay edge being dragged (trim).
    @State private var overlayTrim: OverlayTrimState?

    static let height: CGFloat = 162
    static let rowHeight: CGFloat = 36
    static let lowerTop: CGFloat = 90

    /// Timeline height for `rows` rows under the video row (2 to 3 shown; more slide).
    static func height(rows: Int, compact: Bool) -> CGFloat {
        lowerTop + CGFloat(max(2, min(rows, compact ? 2 : 3))) * rowHeight
    }

    /// The rows under the video: voice-over, sounds (when there are any), then one row per
    /// overlay layer (the higher number is drawn on top of the video).
    enum LowerRow: Hashable { case voice, sounds, layer(Int) }

    static func lowerRows(for model: DraftEditorViewModel) -> [LowerRow] {
        var rows: [LowerRow] = [.voice]
        if !model.soundClips.isEmpty || model.pendingVoiceOvers.contains(where: \.isMusic) { rows.append(.sounds) }
        let top = max(0, model.topOverlayLayer)
        rows += (0...top).map { LowerRow.layer($0) }
        return rows
    }

    static func height(for model: DraftEditorViewModel, compact: Bool) -> CGFloat {
        height(rows: lowerRows(for: model).count, compact: compact)
    }

    static let minZoom: CGFloat = 6
    static let maxZoom: CGFloat = 240
    /// Where the cursor stands (right after the icon column), as in the Figma.
    static let cursorX: CGFloat = 54

    // Rows, measured from the top of the timeline area (Figma y 615).
    private static let videoBandY: CGFloat = 37
    private static let videoY: CGFloat = 40

    private var duration: Double { max(0.1, playback.duration > 0 ? playback.duration : model.timelineDuration) }

    private var rows: [LowerRow] { Self.lowerRows(for: model) }
    private var timelineHeight: CGFloat { Self.height(rows: rows.count, compact: compact) }
    private var maxRowScroll: CGFloat {
        max(0, CGFloat(rows.count) * Self.rowHeight - (timelineHeight - Self.lowerTop))
    }

    var body: some View {
        GeometryReader { proxy in
            let width = proxy.size.width
            let now = playback.currentTime
            let pps = pixelsPerSecond
            let height = timelineHeight
            let scrollY = min(rowScroll, maxRowScroll)
            // The tracks slide under the fixed cursor until the end of the cut reaches the right side;
            // from there the cut stays in view and the cursor walks to its end (never an empty timeline).
            let scroll = Self.scrollTime(now: now, duration: duration, width: width, pixelsPerSecond: pps)
            let cursorAt = Self.cursorX + CGFloat(now - scroll) * pps
            let x: (Double) -> CGFloat = { seconds in Self.cursorX + CGFloat(seconds - scroll) * pps }
            ZStack(alignment: .topLeading) {
                Color(hex: 0x081226)
                timeLabel(now).offset(x: 16, y: 6)
                ruler(width: width, x: x).offset(y: 24)
                band(width: width, y: Self.videoBandY, height: 56)

                // Rows under the video row (voice · sounds · overlay layers), sliding up and down.
                ZStack(alignment: .topLeading) {
                    ForEach(Array(rows.enumerated()), id: \.element) { index, row in
                        let y = CGFloat(index) * Self.rowHeight
                        band(width: width, y: y, height: Self.rowHeight)
                        lowerRow(row, width: width, x: x)
                            .offset(y: y + 3)
                    }
                }
                .frame(width: width, height: CGFloat(rows.count) * Self.rowHeight, alignment: .topLeading)
                .offset(y: Self.lowerTop - scrollY)
                .mask(alignment: .topLeading) {
                    Rectangle().frame(width: max(0, width - Self.cursorX + 2), height: max(0, height - Self.lowerTop))
                        .offset(x: Self.cursorX - 2, y: Self.lowerTop)
                }

                ZStack(alignment: .topLeading) {
                    videoTrack(width: width, x: x).offset(y: Self.videoY)
                    keyframes(width: width, x: x).offset(y: Self.videoY)
                }
                .frame(width: width, height: height, alignment: .topLeading)
                .mask(alignment: .topLeading) {
                    Rectangle().frame(width: max(0, width - Self.cursorX + 2), height: height).offset(x: Self.cursorX - 2)
                }

                // Add video: always there, small, at the right end of the video row.
                plusButton(action: onAddVideo, label: "Add video")
                    .offset(x: width - 34, y: Self.videoY + 13)

                rowIcons(scrollY: scrollY, height: height)

                // The fixed cursor: from the top of the video track down to the last row shown.
                RoundedRectangle(cornerRadius: 1).fill(Color.white)
                    .frame(width: 2, height: height - Self.videoBandY - 6)
                    .offset(x: cursorAt - 1, y: Self.videoBandY + 1)
                    .allowsHitTesting(false)
            }
            .frame(width: width, height: height, alignment: .topLeading)
            .clipped()
            .contentShape(Rectangle())
            .gesture(scrub)
            .simultaneousGesture(zoom)
        }
        .frame(height: timelineHeight)
        .overlay(alignment: .top) { Rectangle().fill(CS.borderDefault).frame(height: 1).offset(y: -1) }
        .animation(.easeOut(duration: 0.2), value: rows.count)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("timelineV4")
    }

    @ViewBuilder
    private func lowerRow(_ row: LowerRow, width: CGFloat, x: @escaping (Double) -> CGFloat) -> some View {
        switch row {
        case .voice: voiceTrack(width: width, x: x)
        case .sounds: soundTrack(width: width, x: x)
        case .layer(let layer): overlayTrack(layer: layer, width: width, x: x)
        }
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

    private func rowIcons(scrollY: CGFloat, height: CGFloat) -> some View {
        let muted = model.videoTrackVolume <= 0.001
        let layers = rows.filter { if case .layer = $0 { return true } else { return false } }.count
        return ZStack(alignment: .topLeading) {
            rowIcon("Ed4RowSpeaker", label: muted ? "Turn the video sound on" : "Video sound", action: onToggleVideoSound)
                .overlay {
                    if muted {
                        Rectangle().fill(CS.error).frame(width: 2, height: 24).rotationEffect(.degrees(-45)).allowsHitTesting(false)
                    }
                }
                .offset(x: 12, y: 49)
                .accessibilityIdentifier("rowVideoSound")
                .accessibilityValue(muted ? "off" : "on")
            ZStack(alignment: .topLeading) {
                ForEach(Array(rows.enumerated()), id: \.element) { index, row in
                    Group {
                        switch row {
                        case .voice:
                            rowIcon("Ed4RowMic", label: "Voice-over", action: onAddVoice)
                        case .sounds:
                            Image(systemName: "music.note").font(.system(size: 15, weight: .semibold))
                                .foregroundStyle(LinearGradient(colors: [CS.cyan, Color(hex: 0x7C3AED), Color(hex: 0xFF21D7)],
                                                                startPoint: .topLeading, endPoint: .bottomTrailing))
                                .frame(width: 32, height: 32)
                                .background(CS.surfaceCard, in: RoundedRectangle(cornerRadius: 8))
                                .overlay(RoundedRectangle(cornerRadius: 8).stroke(CS.borderDefault, lineWidth: 1))
                                .accessibilityLabel("Sounds")
                        case .layer(let layer):
                            rowIcon("Ed4RowLayers", label: "Overlay layer \(layer + 1)", action: onAddOverlay)
                                .overlay(alignment: .topTrailing) {
                                    if layers > 1 { LayerBadge(number: layer + 1).offset(x: 6, y: -6) }
                                }
                        }
                    }
                    .offset(x: 12, y: CGFloat(index) * Self.rowHeight + 2)
                }
            }
            .offset(y: Self.lowerTop - scrollY)
            .mask(alignment: .topLeading) {
                Rectangle().frame(width: 60, height: max(0, height - Self.lowerTop)).offset(y: Self.lowerTop)
            }
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
                if dragAxisVertical == nil {
                    // Up / down over the rows under the video slides them (when there are more).
                    dragAxisVertical = abs(value.translation.height) > abs(value.translation.width)
                        && value.startLocation.y > Self.lowerTop && maxRowScroll > 0
                }
                if dragAxisVertical == true {
                    if rowScrollStart == nil { rowScrollStart = rowScroll }
                    rowScroll = max(0, min(maxRowScroll, (rowScrollStart ?? 0) - value.translation.height))
                    return
                }
                if dragStartTime == nil {
                    dragStartTime = playback.currentTime
                    playback.pause()
                }
                let target = (dragStartTime ?? 0) - Double(value.translation.width / pixelsPerSecond)
                playback.seek(to: max(0, min(duration, target)))
            }
            .onEnded { _ in
                #if DEBUG
                NSLog("CUTSELLB17 scrub_end t=%.2f vertical=%@", playback.currentTime, dragAxisVertical == true ? "yes" : "no")
                #endif
                dragStartTime = nil
                rowScrollStart = nil
                dragAxisVertical = nil
            }
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
        let windows = Self.windows(clips: Self.liftOrder(clips, lift: liftDrag), trim: trimDrag)
        return ZStack(alignment: .topLeading) {
            ForEach(Array(windows.enumerated()), id: \.element.clipID) { index, window in
                // While the START edge is dragged its right side stays put and the left side
                // follows the finger; the clips after it move when the finger lifts (ripple).
                let shift = Self.startDragShift(window: window, trim: trimDrag)
                let lifted = liftDrag?.clipID == window.clipID
                // A lifted clip follows the finger; the others already stand in their new places.
                let left = lifted ? x(liftDrag?.originalStart ?? window.start) + (liftDrag?.dx ?? 0)
                                  : x(window.start) + CGFloat(shift / window.speed) * pixelsPerSecond
                let clipWidth = max(4, CGFloat(window.end - window.start) * pixelsPerSecond - 2)
                if left + clipWidth > -40, left < width + 40 {
                    let clip = clips.first { $0["clip_id"]?.stringValue == window.clipID }
                    let isSelected = selection == .video(window.clipID)
                    let caption = model.captionsEnabled ? Self.caption(of: clip, window: window, trim: trimDrag, model: model) : ""
                    ClipBlockV4(
                        url: model.previewVideoURL(forSource: window.sourceID),
                        sourceStart: window.sourceStart,
                        length: window.sourceLength,
                        number: index + 1,
                        caption: caption,
                        speech: trimDrag?.clipID == window.clipID ? trimDrag?.speech ?? [] : Self.speech(of: clip),
                        muted: model.clipIsSilent(window.clipID),
                        width: clipWidth,
                        isSelected: isSelected
                    )
                    .overlay(alignment: .topTrailing) {
                        if model.clipIsSilent(window.clipID) { MuteBadge().padding(.top, 17).padding(.trailing, 4) }
                    }
                    .scaleEffect(lifted ? 1.06 : 1)
                    .shadow(color: .black.opacity(lifted ? 0.6 : 0), radius: lifted ? 8 : 0, y: lifted ? 4 : 0)
                    .zIndex(lifted ? 2 : 0)
                    .offset(x: left, y: lifted ? -4 : 0)
                    .onTapGesture {
                        selection = selection == .video(window.clipID) ? nil : .video(window.clipID)
                    }
                    .overlay(alignment: .bottomTrailing) {
                        if window.speed != 1 {
                            Text(ClipSpeed.label(window.speed))
                                .font(.system(size: 9, weight: .bold)).foregroundStyle(.white)
                                .padding(.horizontal, 4).padding(.vertical, 1)
                                .background(Color(hex: 0x1769FF), in: RoundedRectangle(cornerRadius: 3))
                                .padding(.trailing, 4).padding(.bottom, 12)
                                .allowsHitTesting(false)
                        }
                    }
                    .gesture(liftGesture(clipID: window.clipID, windowStart: window.start))
                    if isSelected && !lifted {
                        trimHandle(.start, clipID: window.clipID, index: index)
                            .offset(x: left - TrimHandleV4.outside)
                        trimHandle(.end, clipID: window.clipID, index: index)
                            .offset(x: left + clipWidth - TrimHandleV4.width + TrimHandleV4.outside)
                    }
                }
            }
        }
        .frame(width: width, height: 50, alignment: .topLeading)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("videoTrack")
    }

    // MARK: Lift a clip and drop it somewhere else (hold, then drag)

    private func liftGesture(clipID: String, windowStart: Double) -> some Gesture {
        LongPressGesture(minimumDuration: 0.3)
            .sequenced(before: DragGesture(minimumDistance: 0))
            .onChanged { value in
                guard case .second(true, let drag) = value else { return }
                if liftDrag == nil {
                    playback.pause()
                    SnapFeedback.lift()
                    let ids = model.selectedClips.compactMap { $0["clip_id"]?.stringValue }
                    liftDrag = LiftDragState(clipID: clipID, originalStart: windowStart, originalOrder: ids, order: ids)
                }
                guard var lift = liftDrag else { return }
                lift.dx = drag?.translation.width ?? 0
                let clips = model.selectedClips
                let lengths = clips.map { max(0, ($0["end"]?.doubleValue ?? 0) - ($0["start"]?.doubleValue ?? 0)) / ClipSpeed.of($0) }
                if let from = clips.firstIndex(where: { $0["clip_id"]?.stringValue == clipID }) {
                    let center = lift.originalStart + lengths[from] / 2 + Double(lift.dx / pixelsPerSecond)
                    let order = ClipReorder.order(lengths: lengths, from: from, center: center)
                        .compactMap { clips[$0]["clip_id"]?.stringValue }
                    if order != lift.order { SnapFeedback.tick() }
                    lift.order = order
                }
                liftDrag = lift
            }
            .onEnded { _ in
                guard let lift = liftDrag else { return }
                if lift.order != lift.originalOrder {
                    // One save when the finger lifts (the new order is on screen at once).
                    model.reorderNow(clipIDs: lift.order)
                }
                liftDrag = nil
            }
    }

    /// The clips in the order the lifted clip makes (the original order when nothing is lifted).
    static func liftOrder(_ clips: [[String: JSONValue]], lift: LiftDragState?) -> [[String: JSONValue]] {
        guard let lift else { return clips }
        return lift.order.compactMap { id in clips.first { $0["clip_id"]?.stringValue == id } }
    }

    // MARK: Move a voice-over or an overlay sideways (hold, then drag), with magnets

    /// Where things stick: the cursor, the clip joins and the other items' edges.
    private func magnets(excluding id: String) -> [Double] {
        var points: [Double] = [playback.currentTime, 0]
        for window in EditorTracks.clipWindows(model) { points.append(window.start); points.append(window.end) }
        for clip in model.audioClips where clip.id != id { points.append(clip.start); points.append(clip.end) }
        for item in model.overlaysOnScreen where item.id != id { points.append(item.start); points.append(item.end) }
        return points
    }

    /// Hold, then drag (an item that is not selected: a plain swipe still slides the timeline).
    private func moveGesture(kind: MoveDragState.Kind, id: String, start: Double, length: Double) -> some Gesture {
        LongPressGesture(minimumDuration: 0.3)
            .sequenced(before: DragGesture(minimumDistance: 0, coordinateSpace: .global))
            .onChanged { value in
                guard case .second(true, let drag) = value else { return }
                moveChanged(kind: kind, id: id, start: start, length: length, dx: drag?.translation.width ?? 0)
            }
            .onEnded { _ in moveEnded() }
    }

    /// Build 18: a SELECTED overlay, sound or voice moves with a plain drag (like CapCut); no hold.
    private func directMoveGesture(kind: MoveDragState.Kind, id: String, start: Double, length: Double) -> some Gesture {
        DragGesture(minimumDistance: 6, coordinateSpace: .global)
            .onChanged { drag in moveChanged(kind: kind, id: id, start: start, length: length, dx: drag.translation.width) }
            .onEnded { _ in moveEnded() }
    }

    /// Move for an item on the timeline: plain drag when selected, hold-and-drag otherwise.
    private func itemMove(kind: MoveDragState.Kind, id: String, start: Double, length: Double, selected: Bool) -> AnyGesture<Void> {
        selected
            ? AnyGesture(directMoveGesture(kind: kind, id: id, start: start, length: length).map { _ in () })
            : AnyGesture(moveGesture(kind: kind, id: id, start: start, length: length).map { _ in () })
    }

    private func moveChanged(kind: MoveDragState.Kind, id: String, start: Double, length: Double, dx: CGFloat) {
        if moveDrag == nil {
            playback.pause()
            SnapFeedback.lift()
            moveDrag = MoveDragState(kind: kind, id: id, originalStart: start, length: length, start: start)
            switch kind {
            case .voice: selection = .voiceOver(id)
            case .sound: selection = .sound(id)
            case .overlay: selection = .overlay(id, isText: false)
            }
        }
        guard var move = moveDrag else { return }
        let proposed = max(0, move.originalStart + Double(dx / pixelsPerSecond))
        let snapped = TimelineSnap.snapStart(proposed, length: length, magnets: magnets(excluding: id),
                                             pixelsPerSecond: pixelsPerSecond)
        if snapped.snapped != nil && snapped.snapped != move.snappedTo { SnapFeedback.tick() }
        move.snappedTo = snapped.snapped
        move.start = max(0, snapped.start)
        moveDrag = move
    }

    private func moveEnded() {
        guard let move = moveDrag else { return }
        moveDrag = nil
        guard abs(move.start - move.originalStart) >= 0.01 else { return }
        CutSellDiagnostics.log("timeline_item_moved", ["kind": "\(move.kind)", "from": String(format: "%.2f", move.originalStart),
                                                       "to": String(format: "%.2f", move.start)])
        switch move.kind {
        case .voice, .sound:
            // One save when the finger lifts (voices go to the first free place).
            model.moveAudioClip(id: move.id, to: move.start)
        case .overlay:
            let total = max(move.length, model.timelineDuration)
            let begin = min(move.start, max(0, total - move.length))
            if model.pendingOverlays.contains(where: { $0.id == move.id }) {
                model.updatePendingOverlay(id: move.id, start: begin, end: begin + move.length)
            } else {
                model.setOverlayTime(id: move.id, start: begin, end: begin + move.length)
            }
        }
    }

    /// Start shown for an item while it is dragged (its saved start otherwise).
    private func shownStart(_ id: String, _ start: Double) -> Double {
        moveDrag?.id == id ? moveDrag?.start ?? start : start
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
            let speed = ClipSpeed.of(clip)
            let length = max(0, end - start) / speed
            defer { cursor += length }
            guard let id, length > 0 else { return nil }
            return EditorTracks.ClipWindow(clipID: id, start: cursor, end: cursor + length, sourceStart: start,
                              sourceID: clip["source_asset_id"]?.stringValue, speed: speed)
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
        // The finger moves on the played timeline; a 2x clip covers twice as much of its original.
        let speed = ClipSpeed.of(clips[index])
        let resolved = TrimMath.resolve(proposed: drag.original + Double(dx / pixelsPerSecond) * speed, edge: edge, limits: drag.limits)
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
        let marks = model.videoKeyframeMarks
        let on = model.autoZoomEnabled
        return ZStack(alignment: .topLeading) {
            ForEach(marks) { mark in
                diamond(mark, dimmed: mark.isAI && !on, x: x, width: width, y: 25)
            }
        }
        .frame(width: width, height: 50, alignment: .topLeading)
    }

    /// One ◇ (tap = select, the toolbar's Delete removes it; drag = move it in its clip / overlay).
    @ViewBuilder
    private func diamond(_ mark: KeyframeMark, dimmed: Bool, x: @escaping (Double) -> CGFloat, width: CGFloat,
                         y: CGFloat) -> some View {
        let dx = keyframeDrag?.id == mark.id ? keyframeDrag?.dx ?? 0 : 0
        let px = x(mark.time) + dx
        if px > Self.cursorX - 8, px < width + 8 {
            ZoomDiamond(isSelected: selection == .zoom(mark.id), dimmed: dimmed)
                .frame(width: 26, height: 26)
                .contentShape(Rectangle())
                .position(x: px, y: y)
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
                        model.moveKeyframe(id: mark.id, to: max(0, min(duration - 0.01, target)))
                    })
                .accessibilityElement()
                .accessibilityLabel("Keyframe at \(EditorTime.tenths(mark.time))")
                .accessibilityValue(String(format: "%.2f", mark.time))
                .accessibilityIdentifier(mark.isAI ? "zoomKeyframe" : "userKeyframe")
        }
    }

    // MARK: Voice-over track (draft voice clips, plus recordings still uploading)

    private func audioTrimRange(_ clip: AudioClipItem) -> (start: Double, end: Double) {
        if let trim = audioTrim, trim.id == clip.id { return (trim.start, trim.end) }
        let start = shownStart(clip.id, clip.start)
        return (start, start + clip.length)
    }

    private func voiceTrack(width: CGFloat, x: @escaping (Double) -> CGFloat) -> some View {
        let voices = model.voiceClips.sorted { $0.start < $1.start }
        let pending = model.pendingVoiceOvers.filter { !$0.isMusic }
        let lastEnd = (voices.map(\.end) + pending.map(\.end)).max()
        let labels = Self.voiceLabels(
            voices.enumerated().map { index, clip in
                let range = audioTrimRange(clip)
                return VoiceLabelInput(key: clip.id, title: "Voiceover \(index + 1)", left: x(range.start),
                                       width: max(8, CGFloat(range.end - range.start) * pixelsPerSecond), status: .saved)
            } + pending.enumerated().map { index, item in
                VoiceLabelInput(key: "pending_\(item.id)", title: "Voiceover \(voices.count + index + 1)",
                                left: x(item.start), width: max(8, CGFloat(item.duration) * pixelsPerSecond),
                                status: item.failed ? .failed : .uploading)
            }
        )
        return ZStack(alignment: .topLeading) {
            ForEach(Array(voices.enumerated()), id: \.element.id) { index, clip in
                let range = audioTrimRange(clip)
                let left = x(range.start)
                let blockWidth = max(8, CGFloat(range.end - range.start) * pixelsPerSecond)
                let isSelected = selection == .voiceOver(clip.id)
                if left + blockWidth > -20, left < width + 20 {
                    VoiceBlockV4(
                        title: "Voiceover \(index + 1)",
                        assetID: clip.assetID ?? clip.id,
                        width: blockWidth,
                        isSelected: isSelected,
                        showsLabel: false
                    )
                    .overlay(alignment: .trailing) {
                        if clip.muted || clip.volume <= 0.001 { MuteBadge().padding(.trailing, 3) }
                    }
                    .scaleEffect(moveDrag?.id == clip.id ? 1.05 : 1)
                    .offset(x: left, y: moveDrag?.id == clip.id ? -2 : 0)
                    .onTapGesture { selection = isSelected ? nil : .voiceOver(clip.id) }
                    .gesture(itemMove(kind: .voice, id: clip.id, start: clip.start, length: clip.length, selected: isSelected))
                    if isSelected && moveDrag == nil {
                        audioTrimHandle(.start, clip: clip).offset(x: left - TrimHandleV4.outside)
                        audioTrimHandle(.end, clip: clip).offset(x: left + blockWidth - TrimHandleV4.width + TrimHandleV4.outside)
                    }
                }
            }
            // Saved a moment ago: on the row at once while it uploads (tap to retry if it failed).
            ForEach(Array(pending.enumerated()), id: \.element.id) { index, item in
                let left = x(item.start)
                let blockWidth = max(8, CGFloat(item.duration) * pixelsPerSecond)
                if left + blockWidth > 0, left < width + 20 {
                    VoiceBlockV4(
                        title: "Voiceover \(voices.count + index + 1)",
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
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("voiceTrack")
    }

    // MARK: Sounds row: CutSell sounds and music from Files

    private func soundTrack(width: CGFloat, x: @escaping (Double) -> CGFloat) -> some View {
        let sounds = model.soundClips.sorted { $0.start < $1.start }
        let pending = model.pendingVoiceOvers.filter(\.isMusic)
        return ZStack(alignment: .topLeading) {
            ForEach(sounds) { clip in
                let range = audioTrimRange(clip)
                let left = x(range.start)
                let blockWidth = max(10, CGFloat(range.end - range.start) * pixelsPerSecond)
                let isSelected = selection == .sound(clip.id)
                if left + blockWidth > -20, left < width + 20 {
                    SoundBlockV5(title: model.title(of: clip), isMusic: clip.kind == .music, width: blockWidth, isSelected: isSelected,
                                 muted: clip.muted || clip.volume <= 0.001, start: range.start)
                        .scaleEffect(moveDrag?.id == clip.id ? 1.05 : 1)
                        .offset(x: left, y: moveDrag?.id == clip.id ? -2 : 0)
                        .onTapGesture { selection = isSelected ? nil : .sound(clip.id) }
                        .gesture(itemMove(kind: .sound, id: clip.id, start: clip.start, length: clip.length, selected: isSelected))
                    if isSelected && moveDrag == nil {
                        audioTrimHandle(.start, clip: clip).offset(x: left - TrimHandleV4.outside)
                        audioTrimHandle(.end, clip: clip).offset(x: left + blockWidth - TrimHandleV4.width + TrimHandleV4.outside)
                    }
                }
            }
            ForEach(pending) { item in
                SoundBlockV5(title: item.failed ? "Music · tap to retry" : "Music…", isMusic: true,
                             width: max(10, CGFloat(item.duration) * pixelsPerSecond), isSelected: false)
                    .opacity(0.7)
                    .offset(x: x(item.start))
                    .onTapGesture { if item.failed { model.retryPendingVoiceOver(id: item.id) } }
            }
        }
        .frame(width: width, height: 30, alignment: .topLeading)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("soundTrack")
    }

    /// White handles on a selected voice-over or sound: shorter or longer (never past its recording).
    private func audioTrimHandle(_ edge: TrimEdge, clip: AudioClipItem) -> some View {
        TrimHandleV4(edge: edge, active: audioTrim?.id == clip.id && audioTrim?.edge == edge, blocked: audioTrim?.atLimit == true && audioTrim?.edge == edge, height: 30)
            .highPriorityGesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        if audioTrim == nil {
                            playback.pause()
                            audioTrim = AudioTrimState(id: clip.id, edge: edge, start: clip.start, end: clip.end,
                                                       limits: model.audioTrimLimits(clip, edge: edge))
                        }
                        guard var trim = audioTrim else { return }
                        let delta = Double(value.translation.width / pixelsPerSecond)
                        let proposed = TimelineSnap.snap((edge == .start ? clip.start : clip.end) + delta,
                                                         to: magnets(excluding: clip.id), pixelsPerSecond: pixelsPerSecond).value
                        let bounded = min(trim.limits.upperBound, max(trim.limits.lowerBound, proposed))
                        let atLimit = bounded != proposed
                        if atLimit && !trim.atLimit { TrimHaptics.neighbor() }
                        trim.atLimit = atLimit
                        if edge == .start { trim.start = bounded } else { trim.end = bounded }
                        audioTrim = trim
                    }
                    .onEnded { _ in
                        guard let trim = audioTrim else { return }
                        audioTrim = nil
                        let value = trim.edge == .start ? trim.start : trim.end
                        let original = trim.edge == .start ? clip.start : clip.end
                        if abs(value - original) >= 0.01 { model.trimAudioClip(id: clip.id, edge: trim.edge, to: value) }
                    }
            )
            .accessibilityIdentifier(edge == .start ? "audioTrimStart" : "audioTrimEnd")
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
            // Inside its block only at its usual place; a label moved over sits on its own tab so it
            // stays readable over the next voice's waveform.
            let inside = x <= input.left + 7.5 && x + width <= input.left + input.width - 4
            return VoiceLabelPlacement(key: input.key, title: input.title, x: x, width: width,
                                       status: input.status, inside: inside)
        }
    }

    // MARK: Overlay rows (one per layer; photos and videos over the video)

    private func overlayTrack(layer: Int, width: CGFloat, x: @escaping (Double) -> CGFloat) -> some View {
        let items = model.overlaysOnScreen.filter { $0.style.layer == layer }
        // Older text overlays and B-roll placements (made before this editor) stay on the first row.
        let legacy = layer == 0 ? legacyOverlayItems : []
        let isTop = layer == max(0, model.topOverlayLayer)
        let ends = items.map(\.end) + legacy.map(\.end)
        return ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 7).fill(CS.surfaceControl)
                .frame(width: max(0, width - Self.cursorX), height: 30)
                .offset(x: Self.cursorX)
                .accessibilityIdentifier(items.isEmpty && legacy.isEmpty ? "overlayTrackEmpty" : "overlayTrack")
            ForEach(legacy) { item in
                let left = x(item.start)
                let itemWidth = max(10, CGFloat(item.end - item.start) * pixelsPerSecond - 2)
                if left + itemWidth > 0, left < width + 20 {
                    Text(item.title)
                        .font(.system(size: 10, weight: .semibold)).foregroundStyle(.white).lineLimit(1)
                        .padding(.horizontal, 7)
                        .frame(width: itemWidth, height: 30, alignment: .leading)
                        .background(Color(hex: 0x1A3A7A), in: RoundedRectangle(cornerRadius: 7))
                        .overlay(RoundedRectangle(cornerRadius: 7).stroke(selection == item.selection ? CS.cyan : .clear, lineWidth: 2))
                        .offset(x: left)
                        .onTapGesture { selection = selection == item.selection ? nil : item.selection }
                }
            }
            ForEach(items) { item in
                let start = overlayTrim?.id == item.id ? overlayTrim?.start ?? item.start : shownStart(item.id, item.start)
                let end = overlayTrim?.id == item.id ? overlayTrim?.end ?? item.end : start + (item.end - item.start)
                let left = x(start)
                let itemWidth = max(10, CGFloat(end - start) * pixelsPerSecond - 2)
                let isSelected = selection == .overlay(item.id, isText: false)
                if left + itemWidth > -30, left < width + 30 {
                    OverlayBlockV4(item: item, title: model.overlayTitle(item.id), width: itemWidth, isSelected: isSelected)
                        .scaleEffect(moveDrag?.id == item.id ? 1.05 : 1)
                        .offset(x: left, y: moveDrag?.id == item.id ? -2 : 0)
                        .onTapGesture {
                            if item.failed { model.retryPendingOverlay(id: item.id); return }
                            selection = isSelected ? nil : .overlay(item.id, isText: false)
                        }
                        .gesture(itemMove(kind: .overlay, id: item.id, start: item.start, length: item.end - item.start, selected: isSelected))
                    if isSelected && moveDrag == nil {
                        overlayTrimHandle(.start, item: item)
                            .offset(x: left - TrimHandleV4.outside)
                        overlayTrimHandle(.end, item: item)
                            .offset(x: left + itemWidth - TrimHandleV4.width + TrimHandleV4.outside)
                    }
                }
            }
            // Keyframe ◇ of the overlays on this row, over their blocks.
            ForEach(items) { item in
                ForEach(model.overlayKeyframeMarks(overlayID: item.id)) { mark in
                    diamond(mark, dimmed: false, x: x, width: width, y: 15)
                }
            }
            // A video overlay whose sound is off shows the mute icon.
            ForEach(items.filter { $0.kind == .video && $0.style.muted }) { item in
                let left = x(shownStart(item.id, item.start))
                if left > Self.cursorX - 20, left < width {
                    MuteBadge().offset(x: left + 4, y: 17).allowsHitTesting(false)
                }
            }
            // "+" on the top row once there is something: the next overlay can go on a new layer.
            if isTop, let lastEnd = ends.max() {
                let px = max(Self.cursorX + 4, x(lastEnd) + 6)
                if px < width - 24 {
                    plusButton(action: onAddOverlay, label: "Add a photo or video")
                        .offset(x: px, y: 3)
                }
            }
        }
        .frame(width: width, height: 30, alignment: .topLeading)
    }

    private struct LegacyOverlayItem: Identifiable {
        let id: String
        let start: Double
        let end: Double
        let title: String
        let selection: EditorSelection
    }

    private var legacyOverlayItems: [LegacyOverlayItem] {
        var items: [LegacyOverlayItem] = []
        for item in model.textOverlays {
            let id = item["overlay_id"]?.stringValue ?? UUID().uuidString
            let start = item["start"]?.doubleValue ?? 0
            items.append(LegacyOverlayItem(id: id, start: start, end: item["end"]?.doubleValue ?? start + 1,
                                           title: item["text"]?.stringValue ?? "", selection: .overlay(id, isText: true)))
        }
        for placement in model.timelineComposition?.brollPlacements ?? [] {
            items.append(LegacyOverlayItem(id: placement.placementID, start: placement.timelineStartSec,
                                           end: placement.timelineEndSec, title: "Video", selection: .broll(placement.placementID)))
        }
        return items
    }

    /// White handles on a selected overlay's edges: shorter or longer on screen.
    private func overlayTrimHandle(_ edge: TrimEdge, item: OverlayOnScreen) -> some View {
        TrimHandleV4(edge: edge, active: overlayTrim?.id == item.id && overlayTrim?.edge == edge, blocked: false, height: 30)
            .highPriorityGesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        if overlayTrim == nil {
                            playback.pause()
                            overlayTrim = OverlayTrimState(id: item.id, edge: edge, start: item.start, end: item.end,
                                                           originalStart: item.start, originalEnd: item.end)
                        }
                        guard var trim = overlayTrim else { return }
                        let delta = Double(value.translation.width / pixelsPerSecond)
                        let total = max(item.end, model.timelineDuration)
                        let points = magnets(excluding: item.id)
                        if edge == .start {
                            let proposed = TimelineSnap.snap(max(0, trim.originalStart + delta), to: points, pixelsPerSecond: pixelsPerSecond)
                            // A video overlay cannot start before its own first frame.
                            let earliest = item.kind == .video ? max(0, trim.originalStart - item.sourceStart) : 0
                            trim.start = min(max(earliest, proposed.value), trim.end - 0.2)
                        } else {
                            let proposed = TimelineSnap.snap(trim.originalEnd + delta, to: points, pixelsPerSecond: pixelsPerSecond)
                            trim.end = max(trim.start + 0.2, min(total, proposed.value))
                        }
                        overlayTrim = trim
                    }
                    .onEnded { _ in
                        guard let trim = overlayTrim else { return }
                        overlayTrim = nil
                        let sourceStart = item.kind == .video ? item.sourceStart + (trim.start - trim.originalStart) : nil
                        if model.pendingOverlays.contains(where: { $0.id == item.id }) {
                            model.updatePendingOverlay(id: item.id, start: trim.start, end: trim.end)
                        } else {
                            model.setOverlayTime(id: item.id, start: trim.start, end: trim.end, sourceStart: sourceStart)
                        }
                    }
            )
            .accessibilityIdentifier(edge == .start ? "overlayTrimStart" : "overlayTrimEnd")
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

/// A clip held and dragged to another place in the cut.
struct LiftDragState {
    let clipID: String
    /// Its place on the timeline when it was lifted (seconds).
    let originalStart: Double
    let originalOrder: [String]
    var order: [String]
    var dx: CGFloat = 0
}

/// A voice-over or an overlay held and dragged sideways.
struct MoveDragState {
    enum Kind { case voice, sound, overlay }
    let kind: Kind
    let id: String
    let originalStart: Double
    let length: Double
    var start: Double
    var snappedTo: Double?
}

/// A voice-over / sound edge being dragged (timeline seconds).
struct AudioTrimState {
    let id: String
    let edge: TrimEdge
    var start: Double
    var end: Double
    let limits: ClosedRange<Double>
    var atLimit = false
}

/// The tornasol number on a layer row's icon (Figma 1125:1191).
struct LayerBadge: View {
    let number: Int
    var body: some View {
        Text("\(number)")
            .font(.system(size: 9, weight: .semibold)).foregroundStyle(.white)
            .frame(width: 14, height: 14)
            .background(LinearGradient(colors: [CS.cyan, Color(hex: 0x1769FF), Color(hex: 0x7C3AED), Color(hex: 0xFF21D7)],
                                       startPoint: .leading, endPoint: .trailing), in: Circle())
            .allowsHitTesting(false)
            .accessibilityIdentifier("layerBadge")
    }
}

/// One CutSell sound or music clip on the sounds row.
struct SoundBlockV5: View {
    let title: String
    let isMusic: Bool
    let width: CGFloat
    let isSelected: Bool
    var muted = false
    /// Where it starts on the timeline (read by the UI tests).
    var start: Double = 0

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: muted ? "speaker.slash.fill" : (isMusic ? "music.note" : "waveform"))
                .font(.system(size: 9, weight: .bold))
            Text(title).font(.system(size: 10, weight: .semibold)).lineLimit(1)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 6)
        .frame(width: width, height: 30, alignment: .leading)
        .background(LinearGradient(colors: [Color(hex: 0x7C3AED, opacity: 0.75), Color(hex: 0xFF21D7, opacity: 0.55)],
                                   startPoint: .leading, endPoint: .trailing), in: RoundedRectangle(cornerRadius: 7))
        .clipShape(RoundedRectangle(cornerRadius: 7))
        .overlay(RoundedRectangle(cornerRadius: 7).stroke(isSelected ? Color.white : Color(hex: 0xFF21D7, opacity: 0.6),
                                                          lineWidth: isSelected ? 2 : 1))
        // A short sound (Tap is 0.6 s) is still easy to touch: the touch area is at least 32 wide.
        .frame(width: max(width, 32), alignment: .leading)
        .contentShape(Rectangle())
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(muted ? "\(title), muted" : title)
        .accessibilityValue(String(format: "%.2f-%.2f", start, Double(width)))
        .accessibilityIdentifier("soundBlock")
    }
}

/// The small mute icon on a silent clip of the timeline.
struct MuteBadge: View {
    var body: some View {
        Image(systemName: "speaker.slash.fill")
            .font(.system(size: 8, weight: .bold)).foregroundStyle(.white)
            .frame(width: 15, height: 15)
            .background(CS.error.opacity(0.9), in: Circle())
            .accessibilityIdentifier("muteBadge")
    }
}

/// An overlay edge being dragged.
struct OverlayTrimState {
    let id: String
    let edge: TrimEdge
    var start: Double
    var end: Double
    let originalStart: Double
    let originalEnd: Double
}

/// A light knock when something sticks to a magnet, a firmer one when a clip is lifted.
enum SnapFeedback {
    @MainActor static func tick() { UIImpactFeedbackGenerator(style: .light).impactOccurred(intensity: 0.7) }
    @MainActor static func lift() { UIImpactFeedbackGenerator(style: .medium).impactOccurred() }
}

/// One photo / video overlay on the overlay row: its real picture(s), the % while it uploads,
/// "retry" when the upload failed (Figma 1026:3400).
struct OverlayBlockV4: View {
    let item: OverlayOnScreen
    var title: String = ""
    let width: CGFloat
    let isSelected: Bool

    var body: some View {
        let tiles = max(1, min(12, Int((width / 30).rounded(.up))))
        ZStack(alignment: .leading) {
            HStack(spacing: 0) {
                ForEach(0..<tiles, id: \.self) { index in
                    OverlayMediaPicture(url: item.url, kind: item.kind,
                                        seconds: item.sourceStart + (item.end - item.start) * Double(index) / Double(tiles))
                        .frame(width: 30, height: 30).clipped()
                }
            }
            .frame(width: width, height: 30, alignment: .leading)
            .clipped()
            if !title.isEmpty, item.uploading == nil, !item.failed {
                LinearGradient(colors: [CS.cyan.opacity(0.45), Color(hex: 0x1769FF, opacity: 0.45),
                                        Color(hex: 0x7C3AED, opacity: 0.45), Color(hex: 0xFF21D7, opacity: 0.45)],
                               startPoint: .leading, endPoint: .trailing)
                Text(title).font(.system(size: 12, weight: .semibold)).foregroundStyle(CS.textPrimaryF)
                    .lineLimit(1).padding(.leading, 8)
            }
            if let progress = item.uploading {
                Color.black.opacity(0.45)
                Text("\(Int((progress * 100).rounded()))%")
                    .font(.system(size: 10, weight: .bold)).monospacedDigit().foregroundStyle(.white)
                    .padding(.leading, 6)
            } else if item.failed {
                CS.error.opacity(0.55)
                Image(systemName: "arrow.clockwise").font(.system(size: 11, weight: .bold)).foregroundStyle(.white)
                    .padding(.leading, 6)
            }
        }
        .frame(width: width, height: 30)
        .clipShape(RoundedRectangle(cornerRadius: 7))
        .overlay(RoundedRectangle(cornerRadius: 7).stroke(isSelected ? Color.white : CS.cyan.opacity(0.7), lineWidth: isSelected ? 2 : 1))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(item.kind == .video ? "Video overlay" : "Photo overlay")
        .accessibilityIdentifier("overlayBlock")
        .accessibilityValue(String(format: "%@ %.2f-%.2f", item.kind == .video ? "video" : "photo", item.start, item.end))
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
    var height: CGFloat = 50

    var body: some View {
        UnevenRoundedRectangle(
            topLeadingRadius: edge == .start ? 6 : 1, bottomLeadingRadius: edge == .start ? 6 : 1,
            bottomTrailingRadius: edge == .end ? 6 : 1, topTrailingRadius: edge == .end ? 6 : 1)
            .fill(blocked ? Color(hex: 0xFFD60A) : Color.white)
            .frame(width: Self.width, height: height)
            .overlay(Capsule().fill(Color(hex: 0x050814).opacity(0.7)).frame(width: 2.5, height: min(16, height * 0.45)))
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
