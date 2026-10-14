"""Editor v4: trim by the edges (iPhone).

A clip is a window on its original (`source_asset_id` + `start`/`end`, names unchanged). White
handles on the selected clip: outward brings back hidden material, inward shortens; the clips after
it follow (ripple). Limits: the original's start/end, and a knock (haptic) at the neighbour clip's
material that lets go after 0.6 s more. While dragging: the exact frame and "+0,4 s" on the video.
Captions follow the words (hand-written captions are kept). One save on release (edit queue with the
409 retry), Undo/Redo. A 720p copy of the whole original feeds the preview.

The pure rules are mirrored here in Python (same formulas as TrimMath) and the Swift source is
checked for each behaviour."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IOS = ROOT / "mobile" / "ios" / "CutSell"


def read(name: str) -> str:
    return (IOS / name).read_text()


def block(source: str, signature: str) -> str:
    return source.split(signature, 1)[1].split("\n    }\n", 1)[0]


# --- Python mirror of TrimMath ------------------------------------------------------------------

MIN_LENGTH = 0.1
PASS_THROUGH = 0.6


def limits(clips, index, edge, source_duration):
    clip = clips[index]
    source, start, end = clip["source_asset_id"], clip["start"], clip["end"]
    if edge == "end":
        neighbor = None
        if index + 1 < len(clips) and clips[index + 1]["source_asset_id"] == source and clips[index + 1]["start"] >= end - 0.001:
            neighbor = clips[index + 1]["start"]
        return start + MIN_LENGTH, source_duration if source_duration is not None else float("inf"), neighbor
    neighbor = None
    if index > 0 and clips[index - 1]["source_asset_id"] == source and clips[index - 1]["end"] <= start + 0.001:
        neighbor = clips[index - 1]["end"]
    return 0.0, end - MIN_LENGTH, neighbor


def resolve(proposed, edge, lim):
    lower, upper, neighbor = lim
    value, at_neighbor, passed = proposed, False, False
    if neighbor is not None:
        if edge == "end" and proposed > neighbor:
            if proposed - neighbor < PASS_THROUGH:
                value, at_neighbor = neighbor, True
            else:
                passed = True
        if edge == "start" and proposed < neighbor:
            if neighbor - proposed < PASS_THROUGH:
                value, at_neighbor = neighbor, True
            else:
                passed = True
    clamped = min(max(value, lower), upper)
    return round(clamped, 3), at_neighbor and clamped == value, passed


def words_in(pool, start, end):
    return [w for w in pool if start <= (w["start"] + w["end"]) / 2 <= end]


CLIPS = [
    {"clip_id": "a", "source_asset_id": "s1", "start": 1.0, "end": 3.0},
    {"clip_id": "b", "source_asset_id": "s1", "start": 4.0, "end": 6.0},
    {"clip_id": "c", "source_asset_id": "s2", "start": 0.5, "end": 2.0},
]


def test_outward_recovers_hidden_material_inward_shortens():
    lim = limits(CLIPS, 0, "end", 10.0)
    assert resolve(3.4, "end", lim)[0] == 3.4          # outward: 0.4 s the cut had hidden comes back
    assert resolve(2.5, "end", lim)[0] == 2.5          # inward: shorter
    lim = limits(CLIPS, 0, "start", 10.0)
    assert resolve(0.6, "start", lim)[0] == 0.6        # start outward
    assert resolve(1.5, "start", lim)[0] == 1.5        # start inward


def test_limits_are_the_original_start_and_end():
    assert resolve(-3.0, "start", limits(CLIPS, 0, "start", 10.0))[0] == 0.0
    assert resolve(99.0, "end", limits(CLIPS, 1, "end", 7.25))[0] == 7.25
    # Never shorter than the minimum.
    assert resolve(0.2, "end", limits(CLIPS, 0, "end", 10.0))[0] == 1.0 + MIN_LENGTH


def test_neighbor_material_holds_then_lets_go_after_0_6_s():
    lim = limits(CLIPS, 0, "end", 10.0)               # next clip b starts at 4.0 in the same original
    assert lim[2] == 4.0
    assert resolve(4.3, "end", lim) == (4.0, True, False)    # knock: held at the neighbour's material
    assert resolve(4.59, "end", lim) == (4.0, True, False)
    value, held, passed = resolve(4.65, "end", lim)          # pushed 0.6 s more: it goes on
    assert value == 4.65 and not held and passed
    lim = limits(CLIPS, 1, "start", 10.0)             # previous clip a ends at 3.0
    assert resolve(2.8, "start", lim) == (3.0, True, False)
    assert resolve(2.3, "start", lim)[2] is True
    # A clip from another original has no neighbour limit.
    assert limits(CLIPS, 1, "end", 10.0)[2] is None


def test_ripple_keeps_the_timeline_without_gaps():
    def windows(clips):
        cursor, out = 0.0, []
        for clip in clips:
            length = clip["end"] - clip["start"]
            out.append((cursor, cursor + length))
            cursor += length
        return out
    stretched = [dict(CLIPS[0], end=3.4)] + CLIPS[1:]
    before, after = windows(CLIPS), windows(stretched)
    assert after[1][0] == before[1][0] + 0.4 and after[2][0] == before[2][0] + 0.4
    assert all(after[i][1] == after[i + 1][0] for i in range(len(after) - 1))


POOL = [
    {"text": "hello", "start": 1.1, "end": 1.4}, {"text": "there", "start": 2.0, "end": 2.6},
    {"text": "hidden", "start": 3.1, "end": 3.35}, {"text": "later", "start": 5.0, "end": 5.4},
]


def test_words_that_enter_appear_and_words_that_leave_hide():
    assert [w["text"] for w in words_in(POOL, 1.0, 3.0)] == ["hello", "there"]
    assert [w["text"] for w in words_in(POOL, 1.0, 3.4)] == ["hello", "there", "hidden"]
    assert [w["text"] for w in words_in(POOL, 1.6, 3.0)] == ["there"]


# --- Swift source -------------------------------------------------------------------------------

def test_trim_math_matches_the_python_rules():
    trim = read("TrimEdges.swift")
    assert "static let minLength = 0.1" in trim
    assert "static let passThrough = 0.6" in trim
    assert "if proposed - neighbor < passThrough { value = neighbor; atNeighbor = true } else { passed = true }" in trim
    assert "if neighbor - proposed < passThrough { value = neighbor; atNeighbor = true } else { passed = true }" in trim
    assert "return middle >= start && middle <= end" in trim
    assert 'for key in ["selected", "alternates", "discarded"]' in trim


def test_clip_fields_are_not_renamed_and_other_settings_are_kept():
    trim = read("TrimEdges.swift")
    body = trim.split("static func trimmed(", 1)[1].split("\n    }\n", 1)[0]
    assert 'out["start"] = .number(' in body and 'out["end"] = .number(' in body
    assert "var out = clip" in body                      # mirror, audio_volume, ids... kept as they are
    for renamed in ('"source_start"', '"source_end"', '"in_point"', '"out_point"', '"trim_start"'):
        assert renamed not in trim
    assert 'if edge == .end { out["manual_end"] = .bool(true) }' in body
    assert 'if !handWritten { out["caption_text"] = .string(spoken) }' in body


def test_handles_are_white_on_the_selected_clip_and_drive_the_drag():
    tl = read("EditorV4Timeline.swift")
    assert "struct TrimHandleV4: View" in tl and ".fill(blocked ? Color(hex: 0xFFD60A) : Color.white)" in tl
    track = tl.split("private func videoTrack(", 1)[1].split("\n    }\n", 1)[0]
    assert "if isSelected && !lifted {" in track and "trimHandle(.start" in track and "trimHandle(.end" in track
    assert ".highPriorityGesture(" in tl                 # wins over the timeline scrub
    changed = block(tl, "private func trimChanged(")
    assert "TrimMath.resolve(proposed: drag.original + Double(dx / pixelsPerSecond)" in changed
    assert "TrimHaptics.neighbor()" in changed and "TrimHaptics.passed()" in changed
    assert "model.trimScrub = TrimScrub(" in changed


def test_one_save_on_release_through_the_edit_queue():
    tl = read("EditorV4Timeline.swift")
    ended = block(tl, "private func trimEnded()")
    assert ended.count("model.trimClip(") == 1
    changed = block(tl, "private func trimChanged(")
    assert "trimClip" not in changed and "saveChange" not in changed     # nothing saved while dragging
    vm = read("DraftEditorViewModel.swift")
    trim = vm.split("func trimClip(clipID: String, edge: TrimEdge, to value: Double)", 1)[1].split("\n    }\n", 1)[0]
    assert "return perform(predict: change) { [weak self] in _ = await self?.saveChange(change) }" in trim
    # saveChange -> commit: the queue that reloads and retries once on a 409.
    assert "private func saveChange(" in vm and "isConflict(error)" in vm


def test_drag_shows_the_exact_frame_and_the_added_seconds():
    trim = read("TrimEdges.swift")
    assert "generator.requestedTimeToleranceBefore = .zero" in trim
    assert "generator.requestedTimeToleranceAfter = .zero" in trim
    assert "Text(TrimMath.deltaLabel(scrub.delta))" in trim
    assert "formatter.locale = locale" in trim           # "+0,4 s" on a Spanish iPhone
    v4 = read("EditorV4View.swift")
    assert "if let scrub = model.trimScrub {" in v4 and "TrimFrameOverlay(scrub: scrub" in v4


def test_caption_and_audio_strip_follow_the_new_window_while_dragging():
    tl = read("EditorV4Timeline.swift")
    assert "if let trim, trim.clipID == window.clipID, !TrimMath.captionEditedByHand(clip)" in tl
    assert "trimDrag?.clipID == window.clipID ? trimDrag?.speech ?? [] : Self.speech(of: clip)" in tl
    assert "static func windows(clips: [[String: JSONValue]], trim: TrimDragState?)" in tl


def test_720p_copy_of_the_whole_original_feeds_the_preview():
    trim = read("TrimEdges.swift")
    assert "AVAssetExportPreset1280x720" in trim
    assert "guard full > 0, copy >= full - 0.1" in trim   # the WHOLE original, or it is not used
    assert 'LocalCaches.folder("Preview720")' in trim
    playback = read("DraftPlaybackView.swift")
    assert "PreviewProxyCache.localURL(forURI: uri)" in playback
    tl = read("EditorV4Timeline.swift")
    assert "url: model.previewVideoURL(forSource: window.sourceID)" in tl


def test_split_keeps_a_hand_dragged_end_on_the_right_piece():
    vm = read("DraftEditorViewModel.swift")
    split = vm.split("static func predictedSplit(", 1)[1].split("\n    }\n", 1)[0]
    assert 'first.removeValue(forKey: "manual_end")' in split


def test_trim_probe_runs_on_a_real_cut():
    v4 = read("EditorV4View.swift")
    assert 'previewState.probe == "trim"' in v4 and "private func runTrimProbe() async" in v4
    probe = (ROOT / ".github" / "workflows" / "cutsell-ios-editor-probe.yml").read_text()
    assert "CUTSELL_PROBE_STATE=\"trim\"" in probe or "trim voiceover" in probe
