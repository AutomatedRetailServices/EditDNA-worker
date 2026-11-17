"""Smart zoom (PO 16 Nov 2026): the AI picks how each zoom moves (fast / slow / reveal) in the word-fix call it
already makes, and the zoom goes to her face or the product (OpenCV face finder, no paid AI)."""
from types import SimpleNamespace

import pytest

from cutsell_worker import keyframes, serde, simple_engine_adapter as A, zoom_framing as F
from cutsell_worker.contracts import ZoomKeyframe
from cutsell_worker.simple_engine import engine as E

SOURCE = SimpleNamespace(source_order=1, source_asset_id="s1")
SPLITS = [{"start": 0.0, "end": 30.0, "visible": True}]


def _moment(s, kind, style="", e=0.0, w="x"):
    return {"s": s, "kind": kind, "style": style, "e": e, "w": w}


def test_the_ai_answer_carries_the_move_and_bad_moves_fall_back_to_the_kind():
    words = [{"w": "mira", "s": 0.0, "e": 0.3}, {"w": "esto", "s": 0.4, "e": 0.7}, {"w": "ya.", "s": 0.8, "e": 1.0}]
    splits = [{"start": 0.0, "end": 1.2, "visible": True}]
    answer = '{"fix": [], "zoom": [[0, "look", "fast"], [1, "benefit", "spin"], [2, "price"]]}'
    _, _, moments, _ = E.fix_words_and_moments(words, splits, lambda p: (answer, {}))
    assert moments == [{"i": 0, "kind": "look", "style": "fast"}, {"i": 1, "kind": "benefit", "style": ""},
                       {"i": 2, "kind": "price", "style": ""}]


def test_phrase_end_stops_at_the_sentence_mark_or_a_pause():
    words = [{"w": "esto", "s": 1.0, "e": 1.3}, {"w": "te", "s": 1.4, "e": 1.5}, {"w": "levanta.", "s": 1.6, "e": 2.1},
             {"w": "Y", "s": 2.2, "e": 2.3}]
    assert E.phrase_end(words, 0) == 2.1
    words[1]["s"] = 2.0          # a pause after the first word
    assert E.phrase_end(words, 0) == 1.3


def test_three_moves():
    keys = A._zoom_keyframes(SOURCE, [_moment(2.0, "price", "fast"), _moment(8.0, "benefit", "slow", e=10.8),
                                      _moment(16.0, "product", "reveal")], SPLITS)
    fast, slow, reveal = keys
    assert (fast.style, fast.ramp_in, fast.ramp_out, fast.duration) == ("fast", 0.3, 0.8, 2.0)
    assert fast.source_time + fast.ramp_in == pytest.approx(2.0)         # it lands ON the word (PO: precise)
    assert fast.scale == 1.25                                           # price 1.20 + punch, capped at 125 %
    assert slow.style == "slow" and slow.duration == pytest.approx(10.8 - 7.85 + 0.2, abs=1e-3)
    assert slow.ramp_in > 1.5 and slow.ramp_out == 0.9                  # keeps moving in while she talks
    assert reveal.style == "reveal" and reveal.ramp_in == 0.5 and reveal.ramp_out >= 1.5
    # PO 16 Nov: never "de golpe" -- every zoom eases in and eases back out
    assert all(k.ramp_in >= 0.3 and k.ramp_out >= 0.8 for k in keys)
    assert all((k.x, k.y) == (0.0, 0.0) for k in keys)                  # no face info: centred, as before


def test_no_move_given_uses_the_kind_and_zooms_never_overlap():
    keys = A._zoom_keyframes(SOURCE, [_moment(2.0, "hook", e=5.6), _moment(5.2, "cta")], SPLITS)
    assert keys[0].style == "slow"
    assert len(keys) == 2 and keys[1].source_time >= keys[0].source_time + keys[0].duration + A.ZOOM_GAP_SEC
    assert keys[1].style == "fast"


def test_a_short_part_still_zooms_smoothly_both_ways():
    keys = A._zoom_keyframes(SOURCE, [_moment(1.0, "price", "fast")], [{"start": 0.0, "end": 2.0, "visible": True}])
    k = keys[0]
    assert k.ramp_in + k.ramp_out <= k.duration + 1e-6 and k.ramp_out > k.ramp_in   # shorter, never a jump out


def test_a_slow_zoom_never_runs_into_a_join():
    keys = A._zoom_keyframes(SOURCE, [_moment(1.0, "benefit", "slow", e=9.0)],
                             [{"start": 0.0, "end": 3.0, "visible": True}])
    assert keys[0].source_time + keys[0].duration <= 3.0 - A.ZOOM_EDGE_SEC + 1e-6


FACE = (0.5, 0.3, 0.2, 0.15)         # a selfie: her face high in the picture


def test_face_moments_go_to_her_face_and_keep_the_whole_head():
    x, y = F.pan_for("hook", FACE, 1.2)
    assert x == 0.0 and y > 0                                           # the picture moves down: face towards 40 %
    top = 0.5 + (FACE[1] - FACE[3] / 2 - F.HAIR_ROOM * FACE[3] - 0.5) * 1.2 + y
    assert top >= F.FACE_MARGIN - 1e-9                                  # never a cut head
    assert abs(y) <= (1.2 - 1) / 2 + 1e-9                               # never an empty edge


def test_product_moments_go_below_her_face_but_keep_the_face_in():
    face = (0.5, 0.35, 0.2, 0.15)
    x, y = F.pan_for("price", face, 1.25)
    assert y < 0                                                        # towards the product in her hand
    top = 0.5 + (face[1] - face[3] / 2 - F.HAIR_ROOM * face[3] - 0.5) * 1.25 + y
    assert top >= F.FACE_MARGIN - 1e-9
    assert F.pan_for("price", None, 1.25) == (0.0, 0.0)                 # no face: centred
    assert F.pan_for("hook", (0.1, 0.5, 0.2, 0.2), 1.2)[0] > 0          # face on the left: picture moves right


def test_face_finder_is_used_and_logged():
    log = []
    keys = A._zoom_keyframes(SOURCE, [_moment(2.0, "hook", "fast")], SPLITS, log, face_at=lambda t: FACE)
    assert keys[0].y > 0 and log[0]["target"] == "face" and log[0]["style"] == "fast"
    keys = A._zoom_keyframes(SOURCE, [_moment(2.0, "hook", "fast")], SPLITS, face_at=lambda t: 1 / 0)
    assert (keys[0].x, keys[0].y) == (0.0, 0.0)                          # a failing finder never breaks the cut


def test_keyframes_use_the_ramps_and_the_pan_and_old_zooms_stay_the_same():
    old = keyframes.zoom_to_keyframes({"source_time": 1.0, "duration": 1.5, "scale": 1.12})
    assert [k.source_time for k in old] == [1.0, 1.4, 2.1, 2.5] and all(k.x == k.y == 0 for k in old)
    new = keyframes.zoom_to_keyframes({"source_time": 1.0, "duration": 1.0, "scale": 1.2, "ramp_in": 0.08,
                                       "ramp_out": 0.25, "x": 0.5, "y": -0.05})
    assert [k.source_time for k in new] == [1.0, 1.08, 1.75, 2.0]
    assert [k.y for k in new] == [0.0, -0.05, -0.05, 0.0] and new[1].x == 0.1     # pan clamped to the zoom
    assert keyframes.zoom_ramps(1.0, 0.8, 0.8) == pytest.approx((0.5, 0.5))         # never past each other


def test_draft_round_trip_keeps_the_new_fields():
    z = ZoomKeyframe(keyframe_id="z1", source_asset_id="s1", source_time=1.0, duration=2.0, scale=1.2, x=0.05,
                     y=-0.08, ramp_in=1.4, ramp_out=0.35, style="slow")
    back = serde._zoom_keyframe_from_dict(serde._json_safe(z))
    assert back == z
    assert serde._zoom_keyframe_from_dict({"source_time": 1, "duration": 1, "scale": 1.1}).style == ""


def test_render_plan_draws_the_aim_and_speed_of_a_draft_never_opened_in_the_editor(tmp_path):
    from cutsell_worker.render_plan import build_render_plan
    draft = serde.draft_from_dict({
        "project_id": "p", "selected": [{"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 6.0}],
        "auto_zoom_enabled": True,
        "zoom_keyframes": [{"keyframe_id": "z1", "source_asset_id": "s1", "source_time": 1.0, "duration": 1.0,
                            "scale": 1.2, "x": 0.0, "y": 0.06, "ramp_in": 0.08, "ramp_out": 0.25, "style": "fast"}],
    })
    seg = build_render_plan(draft, {"s1": str(tmp_path / "a.mp4")})[0]
    assert seg.zoom_keyframes == ()
    ys = [p[3] for p in seg.transform_keyframes]
    assert max(ys) == pytest.approx(0.06) and [round(p[0], 3) for p in seg.transform_keyframes][:2] == [1.0, 1.08]


def test_render_plan_keeps_old_centred_zooms_on_their_old_path(tmp_path):
    from cutsell_worker.render_plan import build_render_plan
    draft = serde.draft_from_dict({
        "project_id": "p", "selected": [{"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 6.0}],
        "auto_zoom_enabled": True,
        "zoom_keyframes": [{"keyframe_id": "z1", "source_asset_id": "s1", "source_time": 1.0, "duration": 1.5,
                            "scale": 1.12}],
    })
    seg = build_render_plan(draft, {"s1": str(tmp_path / "a.mp4")})[0]
    assert seg.zoom_keyframes == ((1.0, 1.5, 1.12),) and seg.transform_keyframes == ()


def test_one_odd_look_is_not_taken_for_her_face(monkeypatch):
    looks = iter([None, (0.7, 0.55, 0.4, 0.25), (0.5, 0.3, 0.3, 0.17), (0.52, 0.31, 0.3, 0.17), None])
    monkeypatch.setattr(F, "grab_gray", lambda path, t: (b"x", 1, 1))
    monkeypatch.setattr(F, "find_face", lambda *a: next(looks))
    assert F.face_at("v.mp4", 5.0) == pytest.approx((0.51, 0.305, 0.3, 0.17))
    lonely = iter([None, None, (0.5, 0.3, 0.3, 0.17), None, None])
    monkeypatch.setattr(F, "find_face", lambda *a: next(lonely))
    assert F.face_at("v.mp4", 5.0) is None                               # one look alone: centred, as before
