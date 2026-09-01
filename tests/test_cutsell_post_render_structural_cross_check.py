"""check_render_plan_covers_edit_plan (D-025) -- the one real, implemented
piece of PostRenderWatchListenQC this cycle. See that module's docstring
for exactly what it does and does not check (the render PLAN artifact, not
decoded MP4 bytes -- the perceptual checks remain an honest, undone gap).
"""
from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.post_render_watch_listen_qc import (
    STRUCTURAL_SEGMENT_MISSING,
    STRUCTURAL_SEGMENT_TRUNCATED,
    check_render_plan_covers_edit_plan,
)
from cutsell_worker.render_plan import RenderSegment


def _plan_with_keep(clips):
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=clips, alternates=(), discarded=(),
        diagnostics={"final_story_coherence_validation": {"freeze_blocked": False}},
    )
    return build_canonical_edit_plan(draft)


def _clip(clip_id, start, end, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text="x", caption_text="x", selected=True,
    )


def _segment(clip_id, start, end, source="src"):
    return RenderSegment(clip_id=clip_id, source_asset_id=source, source_path="/tmp/x.mp4", start=start, end=end)


def test_passes_when_every_keep_clip_is_fully_covered():
    plan = _plan_with_keep((_clip("a", 0.0, 3.0), _clip("b", 3.0, 6.0)))
    segments = (_segment("a", 0.0, 3.0), _segment("b", 3.0, 6.0))

    result = check_render_plan_covers_edit_plan(segments, plan)

    assert result.status == "PASS"
    assert result.findings == ()


def test_passes_when_coalescing_merged_two_keep_clips_into_one_segment():
    # render_plan.py's own coalescing merges two touching segments into one
    # (dropping the second's clip_id) -- this check must not false-positive
    # on that legitimate, existing behavior.
    plan = _plan_with_keep((_clip("a", 0.0, 3.0), _clip("b", 3.0, 6.0)))
    coalesced = (_segment("a", 0.0, 6.0),)  # covers both a's and b's ranges

    result = check_render_plan_covers_edit_plan(coalesced, plan)

    assert result.status == "PASS"


def test_flags_missing_segment_when_a_keep_clip_has_no_covering_segment_at_all():
    plan = _plan_with_keep((_clip("a", 0.0, 3.0), _clip("b", 10.0, 13.0)))
    segments = (_segment("a", 0.0, 3.0),)  # "b" entirely absent

    result = check_render_plan_covers_edit_plan(segments, plan)

    assert result.status == "FAIL"
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.kind == STRUCTURAL_SEGMENT_MISSING
    assert finding.detail["clip_id"] == "b"
    assert finding.routes_to == "SelectionFreeze"


def test_flags_truncated_when_a_segment_overlaps_but_does_not_fully_cover_the_clip():
    plan = _plan_with_keep((_clip("a", 0.0, 5.0),))
    segments = (_segment("a", 0.0, 3.0),)  # shrunk -- only covers half of it

    result = check_render_plan_covers_edit_plan(segments, plan)

    assert result.status == "FAIL"
    assert result.findings[0].kind == STRUCTURAL_SEGMENT_TRUNCATED


def test_does_not_flag_segments_from_a_different_source():
    plan = _plan_with_keep((_clip("a", 0.0, 3.0, source="src1"),))
    segments = (_segment("a", 0.0, 3.0, source="src2"),)  # same id, wrong source

    result = check_render_plan_covers_edit_plan(segments, plan)

    assert result.status == "FAIL"
    assert result.findings[0].kind == STRUCTURAL_SEGMENT_MISSING
