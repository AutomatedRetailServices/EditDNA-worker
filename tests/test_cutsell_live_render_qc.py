"""Live PostRenderWatchListenQC + bounded physical repair wiring (D-030).

Proves the REAL execution-path order the canonical directive requires:

    Render actual MP4 -> PostRenderWatchListenQC on that actual file
    -> PASS / targeted Boundary repair + re-render / invalidate + route upstream

Real ffmpeg rendering (`render.render_preview`) and the real, deterministic
structural checks (`check_render_plan_covers_edit_plan`,
`check_render_sequence_matches_edit_plan`, `check_no_duplicate_render_
segments`) are exercised for real, unmocked, throughout. The physical-defect
DETECTION layer (`run_post_render_media_qc`) is injected/scripted for the
repair-loop tests (3-5) rather than engineering a natural ffmpeg-detected
defect at an exact segment boundary -- that detector's own correctness is
already exhaustively covered against synthetic fixtures in
tests/test_cutsell_post_render_media_qc.py; this file's job is the
ORCHESTRATION/wiring around it (does a physical failure actually trigger a
real re-render with a repaired segment, does the loop actually bound itself,
does a semantic mismatch actually skip Boundary and block delivery), which
is exactly the risk introduced by this cycle's live wiring.

Skipped, not failed, if ffmpeg is absent from the runner.
"""
import shutil
import subprocess

import pytest

from cutsell_worker import live_render_qc
from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.post_render_watch_listen_qc import (
    LINGERING_ACCIDENTAL_SILENCE,
    PostRenderFinding,
    PostRenderQCResult,
    STRUCTURAL_DUPLICATE_SEGMENT,
    STRUCTURAL_SEGMENT_MISSING,
    STRUCTURAL_SEQUENCE_MISMATCH,
)
from cutsell_worker.render_plan import RenderSegment, build_render_plan

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("live_render_qc")


@pytest.fixture(scope="module")
def source_video(media_dir):
    # One continuous 8s source: no natural silence/freeze, so any physical
    # finding used below is deliberately injected, not accidental.
    path = str(media_dir / "source.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=8",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-y", path,
    ])
    return path


def _clip(clip_id, start, end, text, *, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=True,
    )


def _draft(clips):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="live-qc-test", strategy=EditStrategy.STORYTELLING,
        selected=tuple(clips), alternates=(), discarded=(), diagnostics={},
    )


def _segment(clip, source_path):
    return RenderSegment(
        clip_id=clip.clip_id, source_asset_id=clip.source_asset_id, source_path=source_path,
        start=clip.start, end=clip.end,
    )


@pytest.fixture
def two_clip_draft_and_segments(source_video):
    clip_a = _clip("a", 0.0, 3.0, "first idea")
    clip_b = _clip("b", 3.0, 6.0, "second idea")
    draft = _draft([clip_a, clip_b])
    segments = (_segment(clip_a, source_video), _segment(clip_b, source_video))
    return draft, segments


# ---------------------------------------------------------------------------
# 1. Render output actually invokes PostRenderWatchListenQC
# ---------------------------------------------------------------------------

def test_render_actually_invokes_post_render_watch_listen_qc(monkeypatch, tmp_path, two_clip_draft_and_segments):
    draft, segments = two_clip_draft_and_segments
    calls = []
    real_qc = live_render_qc.run_post_render_media_qc

    def spy_qc(media_path, **kwargs):
        calls.append(media_path)
        return real_qc(media_path, **kwargs)

    monkeypatch.setattr(live_render_qc, "run_post_render_media_qc", spy_qc)

    output = str(tmp_path / "out.mp4")
    result = live_render_qc.render_with_post_render_qc(draft, segments, output)

    assert result.status == "PASS"
    assert len(calls) == 1
    assert calls[0] == output


# ---------------------------------------------------------------------------
# 2. Clean render passes directly
# ---------------------------------------------------------------------------

def test_clean_render_passes_directly(tmp_path, two_clip_draft_and_segments):
    draft, segments = two_clip_draft_and_segments
    output = str(tmp_path / "out.mp4")

    result = live_render_qc.render_with_post_render_qc(draft, segments, output)

    assert result.status == "PASS"
    assert result.output_path == output
    assert len(result.attempts) == 1
    assert result.attempts[0].status == "PASS"
    assert result.attempts[0].findings == ()


# ---------------------------------------------------------------------------
# 3/4. Physical failure triggers Boundary repair; repaired render is QC'd again
# ---------------------------------------------------------------------------

def test_physical_failure_triggers_boundary_repair_and_is_re_qcd(monkeypatch, tmp_path, two_clip_draft_and_segments):
    draft, segments = two_clip_draft_and_segments
    output = str(tmp_path / "out.mp4")

    defect = PostRenderFinding(
        kind=LINGERING_ACCIDENTAL_SILENCE, start=2.7, end=3.0,
        detail={"duration_sec": 0.3}, routes_to="BoundaryEngine",
    )
    qc_calls = []
    render_calls = []
    real_render = live_render_qc.render_preview

    def fake_qc(media_path, **kwargs):
        qc_calls.append(media_path)
        if len(qc_calls) == 1:
            return PostRenderQCResult(status="FAIL", findings=(defect,))
        return PostRenderQCResult(status="PASS", findings=())

    def spy_render(segs, out, **kwargs):
        render_calls.append(tuple(segs))
        return real_render(segs, out, **kwargs)

    monkeypatch.setattr(live_render_qc, "run_post_render_media_qc", fake_qc)
    monkeypatch.setattr(live_render_qc, "render_preview", spy_render)

    result = live_render_qc.render_with_post_render_qc(draft, segments, output, max_attempts=3)

    assert result.status == "PASS"
    assert len(qc_calls) == 2  # re-QC'd after the repair -- not just once
    assert len(render_calls) == 2  # actually re-rendered
    assert len(result.attempts) == 2
    assert result.attempts[0].status == "PHYSICAL_FAIL_REPAIRED"
    assert result.attempts[0].repair_applied["clip_id"] == "a"
    assert result.attempts[1].status == "PASS"
    # The re-render actually used a trimmed segment, not the original.
    assert render_calls[1][0].end < render_calls[0][0].end == 3.0


# ---------------------------------------------------------------------------
# 5. Repair loop is bounded
# ---------------------------------------------------------------------------

def test_bounded_physical_repair_loop_never_spins_past_max_attempts(monkeypatch, tmp_path, two_clip_draft_and_segments):
    draft, segments = two_clip_draft_and_segments
    output = str(tmp_path / "out.mp4")

    def always_fails(media_path, **kwargs):
        return PostRenderQCResult(status="FAIL", findings=(
            PostRenderFinding(
                kind=LINGERING_ACCIDENTAL_SILENCE, start=2.7, end=3.0,
                detail={}, routes_to="BoundaryEngine",
            ),
        ))

    monkeypatch.setattr(live_render_qc, "run_post_render_media_qc", always_fails)

    result = live_render_qc.render_with_post_render_qc(draft, segments, output, max_attempts=3)

    assert result.status == "NEEDS_HUMAN_REVIEW"
    assert result.output_path is None
    assert len(result.attempts) == 3  # bounded -- never spins further
    assert all(a.repair_requested for a in result.attempts)


# ---------------------------------------------------------------------------
# 6/7. Semantic mismatch does NOT route to Boundary and prevents delivery
# ---------------------------------------------------------------------------

def test_semantic_mismatch_never_routes_to_boundary_and_blocks_delivery(monkeypatch, tmp_path, two_clip_draft_and_segments):
    draft, segments = two_clip_draft_and_segments
    # A duplicated clip_id is a real structural/semantic mismatch -- caught
    # before any media decode, so it must never reach the physical repair path.
    broken_segments = (segments[0], segments[0])
    output = str(tmp_path / "out.mp4")

    repair_calls = []
    monkeypatch.setattr(
        live_render_qc, "repair_segment_for_finding",
        lambda *a, **k: repair_calls.append(1) or None,
    )
    render_calls = []
    real_render = live_render_qc.render_preview
    monkeypatch.setattr(
        live_render_qc, "render_preview",
        lambda segs, out, **kw: render_calls.append(1) or real_render(segs, out, **kw),
    )

    result = live_render_qc.render_with_post_render_qc(draft, broken_segments, output)

    assert result.status == "SEMANTIC_MISMATCH_INVALIDATED"
    assert result.output_path is None
    assert repair_calls == []  # Boundary was never asked to "fix" this
    assert len(render_calls) == 1  # never re-rendered/retried either
    assert result.attempts[0].status == "SEMANTIC_MISMATCH"
    assert result.attempts[0].repair_requested is False


# ---------------------------------------------------------------------------
# 8. Composite component absence fails QC
# ---------------------------------------------------------------------------

def test_composite_component_absence_fails_qc(tmp_path, two_clip_draft_and_segments):
    draft, segments = two_clip_draft_and_segments
    # Segment for clip "b" is missing entirely from the render -- exactly
    # the "accepted Composite piece absent" / dropped-KEEP-clip shape.
    missing_segment_render = (segments[0],)
    output = str(tmp_path / "out.mp4")

    result = live_render_qc.render_with_post_render_qc(draft, missing_segment_render, output)

    assert result.status == "SEMANTIC_MISMATCH_INVALIDATED"
    assert result.output_path is None
    kinds = [f["kind"] for f in result.attempts[0].findings]
    assert STRUCTURAL_SEGMENT_MISSING in kinds


# ---------------------------------------------------------------------------
# 9. Frozen plan/render sequence mismatch fails QC
# ---------------------------------------------------------------------------

def test_frozen_plan_render_sequence_mismatch_fails_qc(tmp_path, two_clip_draft_and_segments):
    draft, segments = two_clip_draft_and_segments
    reversed_segments = (segments[1], segments[0])  # rendered out of frozen order
    output = str(tmp_path / "out.mp4")

    result = live_render_qc.render_with_post_render_qc(draft, reversed_segments, output)

    assert result.status == "SEMANTIC_MISMATCH_INVALIDATED"
    kinds = [f["kind"] for f in result.attempts[0].findings]
    assert STRUCTURAL_SEQUENCE_MISMATCH in kinds


# ---------------------------------------------------------------------------
# 10. Final PASS output corresponds to the exact frozen plan id/version/hash
# ---------------------------------------------------------------------------

def test_final_pass_output_identifies_the_exact_frozen_plan(tmp_path, two_clip_draft_and_segments):
    draft, segments = two_clip_draft_and_segments
    output = str(tmp_path / "out.mp4")

    result = live_render_qc.render_with_post_render_qc(draft, segments, output)

    assert result.status == "PASS"
    expected_plan = build_canonical_edit_plan(draft)
    assert result.plan_id == expected_plan.plan_id
    assert result.semantic_hash == expected_plan.semantic_hash
    assert all(a.plan_id == result.plan_id for a in result.attempts)
    assert all(a.semantic_hash == result.semantic_hash for a in result.attempts)


def test_final_pass_preserves_a_repair_loop_derived_plan_version(tmp_path, two_clip_draft_and_segments):
    # A draft that already carries a repair-loop-derived plan_version (e.g.
    # 2, from an earlier D-026 repair) must have that version preserved in
    # the live-render QC's own observability, not silently reset to 1.
    draft, segments = two_clip_draft_and_segments
    from dataclasses import replace
    versioned_draft = replace(draft, diagnostics={"canonical_edit_plan": {"plan_id": "plan_carried_over", "plan_version": 2}})
    output = str(tmp_path / "out.mp4")

    result = live_render_qc.render_with_post_render_qc(versioned_draft, segments, output)

    assert result.status == "PASS"
    assert result.plan_id == "plan_carried_over"
    assert result.plan_version == 2


# ---------------------------------------------------------------------------
# D-036: legitimate physical fragments (a Boundary split of one semantic
# clip into multiple physical pieces) survive build_render_plan and pass
# the live QC service end-to-end -- the exact shape that incorrectly
# invalidated RAW 33415661351 before this fix.
# ---------------------------------------------------------------------------

def test_fragment_provenance_survives_build_render_plan_into_render_segments(source_video):
    from dataclasses import replace as _replace
    clip_a = _replace(
        _clip("a", 0.0, 1.0, "first idea"),
        render_fragment_id="a__f0", parent_semantic_clip_id="a",
        fragment_index=0, fragment_count=2, boundary_reason="remove_micro_visual_reset_word_gap",
    )
    clip_b = _replace(
        _clip("a", 1.4, 2.0, "second idea"),
        render_fragment_id="a__f1", parent_semantic_clip_id="a",
        fragment_index=1, fragment_count=2, boundary_reason="remove_micro_visual_reset_word_gap",
    )
    draft = _draft([clip_a, clip_b])

    plan = build_render_plan(draft, {"src": source_video})

    assert [seg.render_fragment_id for seg in plan] == ["a__f0", "a__f1"]
    assert all(seg.parent_semantic_clip_id == "a" for seg in plan)
    assert [seg.fragment_index for seg in plan] == [0, 1]
    assert all(seg.fragment_count == 2 for seg in plan)
    assert all(seg.boundary_reason == "remove_micro_visual_reset_word_gap" for seg in plan)


def test_legitimate_physical_fragments_pass_the_live_qc_service_end_to_end(tmp_path, source_video):
    from dataclasses import replace as _replace
    clip_a = _replace(
        _clip("a", 0.0, 1.0, "first idea"),
        render_fragment_id="a__f0", parent_semantic_clip_id="a", fragment_index=0, fragment_count=2,
    )
    clip_b = _replace(
        _clip("a", 1.4, 3.0, "first idea continued"),
        render_fragment_id="a__f1", parent_semantic_clip_id="a", fragment_index=1, fragment_count=2,
    )
    draft = _draft([clip_a, clip_b])
    plan = build_render_plan(draft, {"src": source_video})
    output = str(tmp_path / "out.mp4")

    result = live_render_qc.render_with_post_render_qc(draft, plan, output)

    assert result.status == "PASS"
    assert result.output_path == output
    assert len(result.attempts) == 1
    assert result.attempts[0].status == "PASS"


def test_re_split_reordering_of_one_semantic_clip_still_fails_qc(tmp_path, source_video):
    # Same parent, same fragment ids as a legitimate split -- but the render
    # order was corrupted (fragment 1 before fragment 0). A real reordering
    # bug must still fail, not be waved through just because provenance is
    # present.
    from dataclasses import replace as _replace
    clip_a_second = _replace(
        _clip("a", 1.4, 3.0, "first idea continued"),
        render_fragment_id="a__f1", parent_semantic_clip_id="a", fragment_index=1, fragment_count=2,
    )
    clip_a_first = _replace(
        _clip("a", 0.0, 1.0, "first idea"),
        render_fragment_id="a__f0", parent_semantic_clip_id="a", fragment_index=0, fragment_count=2,
    )
    draft = _draft([clip_a_second, clip_a_first])  # reversed render order
    plan = build_render_plan(draft, {"src": source_video})
    output = str(tmp_path / "out.mp4")

    result = live_render_qc.render_with_post_render_qc(draft, plan, output)

    assert result.status == "SEMANTIC_MISMATCH_INVALIDATED"
    kinds = [f["kind"] for f in result.attempts[0].findings]
    assert STRUCTURAL_DUPLICATE_SEGMENT in kinds or STRUCTURAL_SEQUENCE_MISMATCH in kinds


def test_composite_pieces_remain_distinguishable_from_physical_fragments(tmp_path, source_video):
    # A CompositeResolver-accepted composite realizes one Idea from TWO
    # distinct semantic clips (their own clip_id each, no shared parent) --
    # this must never be mistaken for one clip's physical fragments, and
    # must not interfere with a genuinely fragmented clip elsewhere in the
    # same render.
    from dataclasses import replace as _replace
    composite_left = _clip("composite_left", 0.0, 1.0, "first half of the idea")
    composite_right = _clip("composite_right", 1.0, 2.0, "second half of the idea")
    fragmented_a = _replace(
        _clip("a", 3.0, 4.0, "an unrelated idea"),
        render_fragment_id="a__f0", parent_semantic_clip_id="a", fragment_index=0, fragment_count=2,
    )
    fragmented_b = _replace(
        _clip("a", 4.4, 5.4, "continues"),
        render_fragment_id="a__f1", parent_semantic_clip_id="a", fragment_index=1, fragment_count=2,
    )
    draft = _draft([composite_left, composite_right, fragmented_a, fragmented_b])
    plan = build_render_plan(draft, {"src": source_video})
    output = str(tmp_path / "out.mp4")

    result = live_render_qc.render_with_post_render_qc(draft, plan, output)

    assert result.status == "PASS"


def test_fragment_ids_survive_into_qc_finding_diagnostics_on_failure(tmp_path, source_video):
    # When a REAL duplicate is caught, the finding's own detail must still
    # carry the fragment/parent identity -- not just the bare clip_id -- so
    # a reviewer reading render_attempt diagnostics can tell exactly which
    # physical piece collided.
    from dataclasses import replace as _replace
    clip_a = _replace(
        _clip("a", 0.0, 1.0, "first idea"),
        render_fragment_id="a__f0", parent_semantic_clip_id="a", fragment_index=0, fragment_count=2,
    )
    clip_a_dup = _replace(
        _clip("a", 0.0, 1.0, "first idea"),
        render_fragment_id="a__f0", parent_semantic_clip_id="a", fragment_index=0, fragment_count=2,
    )
    draft = _draft([clip_a, clip_a_dup])
    plan = build_render_plan(draft, {"src": source_video})
    output = str(tmp_path / "out.mp4")

    result = live_render_qc.render_with_post_render_qc(draft, plan, output)

    assert result.status == "SEMANTIC_MISMATCH_INVALIDATED"
    finding = result.attempts[0].findings[0]
    assert finding["detail"]["render_fragment_id"] == "a__f0"


def test_rendered_sequence_reconstructed_from_fragments_equals_frozen_plan(tmp_path, source_video):
    # The render segments' clip_id order, once fragments of one parent are
    # accounted for, must reconstruct EXACTLY the frozen CanonicalEditPlan's
    # keep_sequence -- no missing, extra, or reordered semantic content.
    from dataclasses import replace as _replace
    clip_a = _replace(
        _clip("a", 0.0, 1.0, "first idea"),
        render_fragment_id="a__f0", parent_semantic_clip_id="a", fragment_index=0, fragment_count=2,
    )
    clip_a2 = _replace(
        _clip("a", 1.4, 2.0, "first idea continued"),
        render_fragment_id="a__f1", parent_semantic_clip_id="a", fragment_index=1, fragment_count=2,
    )
    clip_b = _clip("b", 3.0, 4.0, "second idea")
    draft = _draft([clip_a, clip_a2, clip_b])
    plan = build_render_plan(draft, {"src": source_video})

    edit_plan = build_canonical_edit_plan(draft)
    assert [c.clip_id for c in edit_plan.keep_sequence] == ["a", "a", "b"]

    output = str(tmp_path / "out.mp4")
    result = live_render_qc.render_with_post_render_qc(draft, plan, output)

    assert result.status == "PASS"
    assert [seg.clip_id for seg in plan] == [c.clip_id for c in edit_plan.keep_sequence]
