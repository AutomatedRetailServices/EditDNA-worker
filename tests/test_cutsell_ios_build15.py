"""Build 15 fixes on the iPhone (reported on build 14): one editor header, Processing ring and cut
name, real overlays (gallery straight away, upload %, retry, move/pinch, row, trim/move/delete,
no text), full-screen B-roll from the video row "+", drag and drop with magnets, frames that never
vanish, and a faster, followed Export."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IOS = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (IOS / name).read_text()


def body(source: str, signature: str) -> str:
    return source.split(signature, 1)[1].split("\n    }\n", 1)[0]


# 1. One header ------------------------------------------------------------------------------------

def test_editor_has_no_second_system_bar():
    editor = read("DraftEditorView.swift")
    assert ".toolbar(.hidden, for: .navigationBar)" in editor
    assert ".navigationTitle(project.title)" not in editor
    assert ".toolbar(.visible, for: .navigationBar)" not in read("ProcessingScreen.swift")


# Processing ring and name ----------------------------------------------------------------------

def test_processing_ring_is_continuous_gradient_turning_with_glow():
    ring = read("ProcessingScreen.swift").split("struct BrandRing: View {", 1)[1].split("\nstruct StageDots", 1)[0]
    for hex_ in ("0x19E6FF", "0x1769FF", "0x7C3AED", "0xFF21D7"):
        assert hex_ in ring
    assert ring.count("0x19E6FF") == 2                     # closes on the first colour: no seam
    assert "dash:" not in ring                              # one continuous circle, no gaps
    assert "static let lineWidth: CGFloat = 4.5" in ring
    assert "static let secondsPerTurn = 2.5" in ring and "TimelineView(.animation" in ring
    assert ".blur(radius: 14 * scale)" in ring and "0.6" in ring and ".blendMode(.plusLighter)" in ring
    assert "accessibilityReduceMotion" in ring and "sin(time" in ring
    assert ring.count(".rotationEffect(.degrees(angle))") == 2   # ring and glow turn together


def angle(t: float) -> float:
    return (t % 2.5) / 2.5 * 360


def test_ring_turns_once_every_two_and_a_half_seconds_linearly():
    assert angle(0) == 0 and angle(1.25) == 180 and abs(angle(2.5)) < 1e-9
    steps = [angle(i * 0.1) for i in range(25)]
    assert all(abs((b - a) - 14.4) < 1e-6 for a, b in zip(steps, steps[1:]))


def test_same_ring_on_loading_screens_with_the_logo():
    assert "BrandRing(size: 84)" in read("DraftEditorView.swift")
    assert "BrandRing(size: 72)" in read("CutSellApp.swift")


def test_processing_shows_the_cut_name_not_the_file_name():
    screen = read("ProcessingScreen.swift")
    assert "fileName: CutName.display(cut?.title)" in screen
    assert "Text(CutName.display(cut.title))" in screen
    names = screen.split("enum CutName {", 1)[1].split("\n}\n", 1)[0]
    assert 'lower.hasPrefix("copy_")' in names and "[0-9A-Fa-f]{8}" in names


# 2. Overlay -----------------------------------------------------------------------------------------

def test_overlay_opens_the_gallery_straight_away_and_has_no_text():
    v4 = read("EditorV4View.swift")
    # Editor v5 (Figma 1119:670): Overlay opens "Add overlay · From your Photos" with Photo and
    # Video, each opening the phone's gallery filtered; no old "Add to the video" menu, no text.
    assert "case .overlay: openOverlayPanel()" in v4
    assert '"Add to the video"' not in v4 and "confirmationDialog" not in v4
    assert "addTextOverlay" not in v4 and '"Add text"' not in v4
    assert "onAddOverlay: { openOverlayPanel() }" in v4
    assert "videosOnly: !galleryPhotos, photosOnly: galleryPhotos" in v4


def test_picked_media_uploads_with_progress_and_retries_without_losing_it():
    vm = read("DraftEditorViewModel.swift")
    add = body(vm, "func addOverlay(fileURL: URL, at start: Double)")
    assert "OverlayMediaStore.keep(fileURL, id: id)" in add       # kept on the phone first
    assert "pendingOverlays.append(pending)" in add and "uploadPendingOverlay(id: id)" in add
    upload = body(vm, "private func uploadPendingOverlay(id: String)")
    assert "for attempt in 0..<PendingVoiceOverStore.attemptsPerRound" in upload
    assert "pendingOverlays[index].failed = true" in upload
    finish = vm.split("private func finishPendingOverlay(", 1)[1].split("\n    }\n", 1)[0]
    assert "setOverlayProgress(item.id, value)" in finish
    # One save carrying the overlay with its layer, layout and edge (one Undo removes it).
    assert "current.style.write(into: &overlay)" in finish and 'object["media_overlays"] = .array(items + [.object(entry)])' in finish
    assert "OverlayMediaStore.link(uri: uploaded.uri, fileName: item.fileName)" in finish
    media = read("EditorMedia.swift")
    assert "didSendBodyData" in media and "applicationSupportDirectory" in media


def test_overlay_is_real_media_moved_with_one_finger_and_resized_with_two():
    v4 = read("EditorV4View.swift")
    view = v4.split("private struct MediaOverlayOnVideo: View {", 1)[1].split("\n}\n", 1)[0]
    assert "OverlayMediaPicture(url: overlay.url" in view
    assert "let h = w / CGFloat(max(0.1, aspect ?? 1))" in view   # its real shape, like the export
    assert "DragGesture(minimumDistance: 2)" in view and "MagnificationGesture()" in view
    assert '"Tap to retry"' in view and "overlayUploadPercent" in view
    assert "model.setOverlayFrame(id: overlay.id, x: x, y: y, width: width)" in v4
    assert '"Photo"' not in v4                                   # no test card any more


def test_overlay_row_has_real_thumbnails_trim_move_delete():
    tl = read("EditorV4Timeline.swift")
    assert "struct OverlayBlockV4: View" in tl and "OverlayMediaPicture(url: item.url" in tl
    assert "overlayTrimHandle(.start, item: item)" in tl and "overlayTrimHandle(.end, item: item)" in tl
    assert "moveGesture(kind: .overlay" in tl
    assert "model.setOverlayTime(id: item.id, start: trim.start, end: trim.end, sourceStart: sourceStart)" in tl
    assert "await model.removeMediaOverlay(id: id)" in read("EditorV4View.swift")


def fit_overlays(clips, overlays):
    total = sum(max(0, c["end"] - c["start"]) for c in clips)
    out = []
    for item in overlays:
        if item["start"] >= total - 0.05:
            continue
        out.append(dict(item, end=min(item["end"], total)))
    return out


def test_overlays_never_run_past_the_cut():
    clips = [{"start": 0, "end": 4}, {"start": 10, "end": 12}]
    assert fit_overlays(clips, [{"start": 5, "end": 9}, {"start": 1, "end": 2}, {"start": 6, "end": 7}]) == [
        {"start": 5, "end": 6}, {"start": 1, "end": 2}]
    vm = read("DraftEditorViewModel.swift")
    assert "let edited = Self.fitOverlaysToCut(built)" in vm


def test_export_server_draws_overlays_where_the_phone_shows_them():
    v4 = read("EditorV4View.swift")
    # Centre x/y and width as a share of the frame: the same numbers the export uses
    # (overlay=x=W*x-w/2:y=H*y-h/2, scale=width*1080:-1).
    assert "frame.width * CGFloat(overlay.x)" in v4 and "frame.height * CGFloat(overlay.y)" in v4
    assert "frame.width * CGFloat(overlay.width)" in v4


# 3. B-roll -----------------------------------------------------------------------------------------

def test_broll_comes_from_the_video_row_plus_into_the_main_track():
    v4 = read("EditorV4View.swift")
    assert "onAddVideo: { playback.pause(); sheet = .brollGallery }" in v4
    assert "model.addBrollClip(fileURL: url, at: playback.currentTime)" in v4
    assert "BrollView(" not in v4
    vm = read("DraftEditorViewModel.swift")
    insert = vm.split("static func insertBroll(", 1)[1].split("\n    }\n", 1)[0]
    assert 'object["broll_sources"]' in insert and "clips.replaceSubrange(index...index, with: [left, clip, right])" in insert
    assert '"text": .string(""), "caption_text": .string("")' in insert
    assert "MultipartUploadManager.shared.upload(" in vm


def insert_broll(clips, length, at):
    cursor = 0.0
    for i, c in enumerate(clips):
        n = c["end"] - c["start"]
        if at <= cursor + 0.05:
            return clips[:i] + [{"broll": True, "len": length}] + clips[i:]
        if at < cursor + n - 0.05:
            cut = c["start"] + (at - cursor)
            return clips[:i] + [dict(c, end=cut), {"broll": True, "len": length}, dict(c, start=cut)] + clips[i + 1:]
        cursor += n
    return clips + [{"broll": True, "len": length}]


def test_broll_lands_at_the_cursor():
    clips = [{"start": 0, "end": 4}, {"start": 10, "end": 12}]
    assert insert_broll(clips, 3, 0)[0] == {"broll": True, "len": 3}
    assert insert_broll(clips, 3, 4)[1] == {"broll": True, "len": 3}           # between clips
    middle = insert_broll(clips, 3, 1.5)                                         # inside the first: split
    assert middle[0] == {"start": 0, "end": 1.5} and middle[2] == {"start": 1.5, "end": 4}
    assert insert_broll(clips, 3, 99)[-1] == {"broll": True, "len": 3}


def test_no_technical_text_shown():
    for name in ("EditorV4View.swift", "EditorV2View.swift"):
        assert "main video is registered" not in read(name)
    vm = read("DraftEditorViewModel.swift")
    assert "primary source asset is registered" not in vm
    assert "static func plainWords(" in vm
    assert "Text(DraftEditorViewModel.plainWords(model.errorMessage))" in read("EditorV4View.swift")


# 4. Drag and drop ------------------------------------------------------------------------------------

def reorder(lengths, frm, center):
    rest = [i for i in range(len(lengths)) if i != frm]
    best, best_d = 0, float("inf")
    for slot in range(len(rest) + 1):
        c = sum(lengths[i] for i in rest[:slot]) + lengths[frm] / 2
        if abs(c - center) < best_d:
            best, best_d = slot, abs(c - center)
    return rest[:best] + [frm] + rest[best:]


def test_lifted_clip_lands_where_the_finger_is_and_the_others_make_room():
    lengths = [2.0, 3.0, 1.0]
    assert reorder(lengths, 0, 1.0) == [0, 1, 2]        # not moved
    assert reorder(lengths, 0, 4.5) == [1, 0, 2]        # dragged over the second clip
    assert reorder(lengths, 2, 0.2) == [2, 0, 1]        # last to first
    media = read("EditorMedia.swift")
    assert "static func order(lengths: [Double], from: Int, center: Double) -> [Int]" in media


def snap(value, magnets, pps, distance=10):
    limit = distance / pps
    nearest = min(magnets, key=lambda m: abs(m - value))
    return nearest if abs(nearest - value) <= limit else value


def test_soft_magnets_to_cursor_and_edges():
    assert snap(3.2, [3.0, 8.0], pps=30) == 3.0        # 6 pt away: sticks
    assert snap(3.5, [3.0, 8.0], pps=30) == 3.5        # 15 pt away: free
    tl = read("EditorV4Timeline.swift")
    magnets = tl.split("private func magnets(excluding id: String)", 1)[1].split("\n    }\n", 1)[0]
    assert "playback.currentTime" in magnets and "clipWindows(model)" in magnets
    assert "model.audioClips" in magnets and "overlaysOnScreen" in magnets
    assert "SnapFeedback.tick()" in tl and "SnapFeedback.lift()" in tl
    assert "LongPressGesture(minimumDuration: 0.3)" in tl


def test_one_save_on_release_and_undo_redo_cover_moves():
    tl = read("EditorV4Timeline.swift")
    lift = tl.split("private func liftGesture(", 1)[1].split("\n    }\n", 1)[0]
    assert lift.count("model.reorderNow(") == 1 and ".onEnded" in lift
    move = tl.split("private func moveGesture(", 1)[1].split("\n    }\n", 1)[0]
    # Voice-overs and sounds are draft clips (Editor v5): one save, the draft's own Undo / Redo.
    assert "model.moveAudioClip(id: move.id, to: move.start)" in move and "model.setOverlayTime(" in move
    v5 = read("EditorV5Model.swift")
    move_audio = v5.split("func moveAudioClip(id: String, to start: Double)", 1)[1].split("\n    }\n", 1)[0]
    assert "PendingVoiceOverStore.freeStart(begin, duration: item.length, taken: others)" in move_audio
    assert "return updateAudioClip(id: id)" in move_audio
    vm = read("DraftEditorViewModel.swift")
    assert "perform(predict: change) { [weak self] in _ = await self?.saveChange(change) }" in vm
    assert "return perform(predict: predict)" in body(vm, "func reorderNow(clipIDs: [String]) -> Task<Void, Never>")


# 5. Frames ------------------------------------------------------------------------------------------

def test_frames_are_kept_and_retried():
    frames = read("ClipFrames.swift")
    assert "if !Task.isCancelled { generators[url.path] = nil }" in frames
    assert "for attempt in 0..<4" in frames


# 6. Export ------------------------------------------------------------------------------------------

def test_export_starts_on_tap_waits_for_edits_and_is_followed_app_wide():
    v4 = read("EditorV4View.swift")
    assert "if !model.isExporting { Task { await model.export() } }" in v4
    vm = read("DraftEditorViewModel.swift")
    export = body(vm, "func export() async")
    assert "_ = await actionChain?.value" in export and "pendingOverlays" in export and "pendingVoiceOvers" in export
    assert "ExportTracker.shared.follow(" in export and "ExportTracker.shared.reusable(" in export
    tracker = read("ExportTracker.swift")
    assert "UNMutableNotificationContent()" in tracker and "beginBackgroundTask" in tracker
    assert 'NSLog("CUTSELLEXPORT' in tracker
    sheet = read("EditorV2Sheets.swift").split("struct ExportSheet: View {", 1)[1]
    assert "exportPercent" in sheet and "You can leave this screen" in sheet


def test_probes_cover_overlay_broll_moves_export_and_ring():
    probe = read("EditorMediaProbe.swift")
    for step in ("photo_overlay", "video_overlay", "broll", "clip_moved", "voice_moved", "export state", "makeVideo"):
        assert step in probe
    workflow = (ROOT / ".github" / "workflows" / "cutsell-ios-editor-probe.yml").read_text()
    assert 'SIMCTL_CHILD_CUTSELL_PROBE_STATE="media"' in workflow
    assert "processing_ring_1.png" in workflow and "processing_ring_2.png" in workflow
    assert 'cp "$RUNNER_TEMP"/probe/*.png probe/' in workflow
    assert "*.mp4 probe/" not in workflow
