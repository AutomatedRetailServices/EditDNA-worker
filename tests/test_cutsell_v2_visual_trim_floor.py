from dataclasses import replace
from types import SimpleNamespace

import pytest

from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy
from cutsell_worker.render_plan import RenderSegment, build_render_plan, _coalesce_contiguous_segments
from cutsell_worker.render import tighten_trailing_silence


def draft(source="src", basis="selected_neighbors_inside_high_confidence_audience_demonstration"):
    clip = DraftClip("demo", "src", 0, 10, 20, "pour water", "pour water")
    return DraftTimeline("cutsell.v1", "p", EditStrategy.STORYTELLING, (clip,), (), (), {
        "editorial_engine_v2": {"status": "frozen_boundary_verified_pending_post_render_review"},
        "editorial_engine_v2_continuity_restoration": [{
            "left_clip_id": "demo", "source_asset_id": source,
            "gap_start": 12, "gap_end": 20, "basis": basis,
        }],
    })


def test_plan_propagates_authorized_visual_floor_without_caption_dependency():
    plan = build_render_plan(draft(), {"src": "unused"})
    assert plan[0].trailing_trim_floor == 20


@pytest.mark.parametrize("source,basis", [
    ("other", "selected_neighbors_inside_high_confidence_audience_demonstration"),
    ("src", "selected_neighbors_inside_high_confidence_audience_region"),
])
def test_unrelated_or_speech_only_evidence_does_not_lock_render(source, basis):
    assert build_render_plan(draft(source, basis), {"src": "unused"})[0].trailing_trim_floor is None


def test_legacy_draft_does_not_consume_v2_visual_lock():
    d = draft()
    diagnostics = dict(d.diagnostics)
    diagnostics.pop("editorial_engine_v2")
    assert build_render_plan(replace(d, diagnostics=diagnostics), {"src": "unused"})[0].trailing_trim_floor is None


@pytest.mark.parametrize("floor,expected", [(None, 12.04), (16, 16), (20, 20)])
def test_renderer_never_trims_through_visual_floor(monkeypatch, floor, expected):
    monkeypatch.setattr("cutsell_worker.render.probe_media", lambda _: SimpleNamespace(has_audio=True))
    monkeypatch.setattr("cutsell_worker.render.subprocess.run", lambda *a, **k: SimpleNamespace(
        returncode=0, stderr="silence_start: 2\nsilence_end: 10"))
    segment = RenderSegment("demo", "src", "unused", 10, 20, trailing_trim_floor=floor)
    assert tighten_trailing_silence(segment).end == expected


def test_coalescing_preserves_latest_visual_floor():
    left = RenderSegment("a", "src", "unused", 0, 10, trailing_trim_floor=8)
    right = RenderSegment("b", "src", "unused", 10, 20, trailing_trim_floor=18)
    assert _coalesce_contiguous_segments((left, right))[0].trailing_trim_floor == 18


def test_split_fragment_retains_source_bound_parent_visual_floor():
    d = draft()
    piece = replace(d.selected[0], clip_id="physical-child", start=15,
                    parent_semantic_clip_id="demo")
    assert build_render_plan(replace(d, selected=(piece,)), {"src": "unused"})[0].trailing_trim_floor == 20


@pytest.mark.parametrize("floor", [float("nan"), float("inf"), 21, 9])
def test_invalid_or_outside_floor_is_not_attached(floor):
    d = draft()
    d.diagnostics["editorial_engine_v2_continuity_restoration"][0]["gap_end"] = floor
    assert build_render_plan(d, {"src": "unused"})[0].trailing_trim_floor is None
