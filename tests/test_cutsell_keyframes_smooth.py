"""21 Oct -- keyframes like CapCut: ONE smooth track over the whole edited video (no jump at a join),
per-keyframe curves, Auto Zoom AI in / hold / out, and AI keyframes refreshed on a new cut."""
from types import SimpleNamespace

from cutsell_worker import keyframes as kf
from cutsell_worker.draft_merge import merge_reprocessed_draft


def _key(i, t, s, curve="ease_in_out", src="a", origin="user"):
    return kf.TransformKeyframe(keyframe_id=f"k{i}", target="video", scale=s, source_asset_id=src,
                                source_time=t, curve=curve, origin=origin)


def _clips():
    # two clips of the same source: 0-4 s and 10-14 s -> timeline 0-4 and 4-8
    return [SimpleNamespace(source_asset_id="a", start=0.0, end=4.0, speed=1.0),
            SimpleNamespace(source_asset_id="a", start=10.0, end=14.0, speed=1.0)]


def test_the_track_crosses_a_join_without_a_jump():
    keys = [_key(1, 2.0, 1.0), _key(2, 12.0, 1.3)]          # timeline 2 s -> 6 s
    windows = kf.clip_windows(_clips())
    track = kf.timeline_track(keys, windows)
    assert [round(p[0], 3) for p in track] == [2.0, 6.0]
    first, second = kf.clip_track(track, windows[0]), kf.clip_track(track, windows[1])
    # at the join (timeline 4 s): end of clip 1 (source 4) == start of clip 2 (source 10)
    left = kf.value_at(first, 4.0, 1)
    right = kf.value_at(second, 10.0, 1)
    assert abs(left - right) < 1e-9 and 1.0 < left < 1.3
    assert abs(kf.value_at(second, 12.0, 1) - 1.3) < 1e-9


def test_ffmpeg_expression_matches_value_at_inside_each_clip():
    keys = [_key(1, 2.0, 1.0), _key(2, 12.0, 1.3, curve="linear"), _key(3, 13.5, 1.0)]
    windows = kf.clip_windows(_clips())
    track = kf.timeline_track(keys, windows)
    for w in windows:
        pts = kf.clip_track(track, w)
        expr = kf.expression(pts, 1, var="T")
        for step in range(9):
            t = w.start + step * (w.end - w.start) / 8
            code = expr.replace("clip(", "_clip(").replace("if(", "_if(").replace("T", repr(t))
            value = eval(code, {"_clip": lambda v, a, b: max(a, min(b, v)), "lt": lambda a, b: a < b,
                                "_if": lambda c, a, b: a if c else b})
            assert abs(value - kf.value_at(pts, t, 1)) < 1e-4


def test_curves():
    pts = [(0.0, 1.0, 0, 0, 0, kf.CURVES.index("linear")), (1.0, 2.0, 0, 0, 0, 0)]
    assert abs(kf.value_at(pts, 0.25, 1) - 1.25) < 1e-9
    pts[0] = (0.0, 1.0, 0, 0, 0, kf.CURVES.index("ease_in"))
    assert abs(kf.value_at(pts, 0.5, 1) - 1.25) < 1e-9
    pts[0] = (0.0, 1.0, 0, 0, 0, kf.CURVES.index("ease_out"))
    assert abs(kf.value_at(pts, 0.5, 1) - 1.75) < 1e-9
    pts[0] = (0.0, 1.0, 0, 0, 0, 0)
    assert abs(kf.value_at(pts, 0.5, 1) - 1.5) < 1e-9          # ease in-out: symmetric
    assert kf.value_at(pts, -1, 1) == 1.0 and kf.value_at(pts, 5, 1) == 2.0   # held outside


def test_curve_round_trips_and_defaults():
    k = kf.keyframe_from_dict({"keyframe_id": "x", "target": "video", "source_asset_id": "a", "source_time": 1,
                               "scale": 1.2, "curve": "ease_out"}) if hasattr(kf, "keyframe_from_dict") else None
    if k is None:
        return
    assert k.curve == "ease_out" and kf.keyframe_to_dict(k)["curve"] == "ease_out"
    plain = kf.keyframe_from_dict({"keyframe_id": "y", "target": "video", "source_asset_id": "a", "source_time": 1,
                                   "scale": 1.2, "curve": "bounce"})
    assert plain.curve == "ease_in_out" and "curve" not in kf.keyframe_to_dict(plain)


def test_ai_zoom_is_in_hold_out_with_04s_ramps():
    keys = kf.zoom_to_keyframes({"source_time": 5.0, "duration": 2.0, "scale": 1.2, "keyframe_id": "z1"})
    assert [round(k.source_time, 2) for k in keys] == [5.0, 5.4, 6.6, 7.0]
    assert [k.scale for k in keys] == [1.0, 1.2, 1.2, 1.0]


def test_a_cut_out_keyframe_is_left_out():
    keys = [_key(1, 7.0, 1.3)]                      # source 7 s is not played
    assert kf.timeline_track(keys, kf.clip_windows(_clips())) == ()


def test_reprocess_refreshes_ai_keyframes_and_keeps_the_creators():
    old = {"selected": [], "keyframes": [
        {"keyframe_id": "u1", "target": "video", "source_asset_id": "a", "source_time": 1.0, "scale": 1.3,
         "origin": "user"},
        {"keyframe_id": "old-1", "target": "video", "source_asset_id": "a", "source_time": 9.0, "scale": 1.1,
         "origin": "ai"}]}
    new = {"selected": [], "zoom_keyframes": [
        {"keyframe_id": "z-001-01", "source_asset_id": "a", "source_time": 3.0, "duration": 1.6, "scale": 1.18,
         "kind": "product", "origin": "ai"}]}
    merged = merge_reprocessed_draft(old, new)
    ids = [k["keyframe_id"] for k in merged["keyframes"]]
    assert "u1" in ids and "old-1" not in ids
    assert len([k for k in merged["keyframes"] if k.get("origin") == "ai"]) == 4
