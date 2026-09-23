"""D-288 -- bounded post-render PERCEPTUAL repair cycle.

Audit finding this closes (`docs/CUTSELL_DECISIONS.md` D-288): a perceptual
finding's `routes_to` label was read in exactly one place in the whole
codebase (a summary counter) and never actually routed to a real repair
authority. `cutsell_worker/perceptual_repair_cycle.py` is the real, bounded
connection: finding -> owning authority -> safe repair -> render -> technical
QC -> Watch+Listen (re-review).

Dependency-injected throughout (the render/technical-QC and perceptual-
review callables are fakes the test controls), except for the ONE real
authority this module is required to reuse rather than reimplement:
`live_boundary_repair.repair_segment_for_finding`, monkeypatched here at
the point `perceptual_repair_cycle` imports it -- same established pattern
this repo already uses for `live_render_qc.py`'s own tests
(`monkeypatch.setattr(live_render_qc, "repair_segment_for_finding", ...)`)
-- so no real ffmpeg/media decode is needed. No Video00 fact/id anywhere
below.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cutsell_worker import perceptual_repair_cycle as cycle
from cutsell_worker.live_boundary_repair import SegmentRepairAttempt
from cutsell_worker.perceptual_watch_listen import (
    EVALUATED_FAIL,
    EVALUATED_PASS,
    ERROR,
    UNCERTAIN,
    ROUTE_BEST_TAKE,
    ROUTE_BOUNDARY,
    CapabilityReport,
    PerceptualFinding,
    PerceptualReview,
    GATE_MODE_STATE_MACHINE_V1,
    overall_status,
)
from cutsell_worker.render_plan import RenderSegment


def _segment(clip_id="clip-1", start=0.0, end=3.0):
    return RenderSegment(clip_id=clip_id, source_asset_id="src-1", source_path="/tmp/x.mp4", start=start, end=end)


def _fake_windows(segments):
    # Deterministic, non-ffmpeg stand-in for `live_boundary_repair.
    # segment_output_windows` -- this module never needs the real
    # trailing-silence-tightening math, only SOME windows to hand the
    # perceptual-review callable.
    cursor = 0.0
    out = []
    for seg in segments:
        duration = seg.end - seg.start
        out.append((cursor, cursor + duration))
        cursor += duration
    return out


def _review(*capabilities) -> PerceptualReview:
    return PerceptualReview(status=overall_status(capabilities), gate_mode=GATE_MODE_STATE_MACHINE_V1, capabilities=tuple(capabilities))


def _pass_review() -> PerceptualReview:
    return _review(CapabilityReport("cap1", EVALUATED_PASS, "mp4_measured"))


def _physical_fail_review() -> PerceptualReview:
    finding = PerceptualFinding("cap1", "PERCEPTUAL_INTERIOR_DEAD_AIR", 0.5, 1.5, "FAIL", ROUTE_BOUNDARY)
    return _review(CapabilityReport("cap1", EVALUATED_FAIL, "mp4_measured", (finding,)))


def _semantic_fail_review() -> PerceptualReview:
    finding = PerceptualFinding("cap2", "PERCEPTUAL_REPEATED_AUDIENCE_CONTENT", 4.0, 6.0, "FAIL", ROUTE_BEST_TAKE)
    return _review(CapabilityReport("cap2", EVALUATED_FAIL, "transcript_derived", (finding,)))


def _error_review() -> PerceptualReview:
    return _review(CapabilityReport("cap3", ERROR, "mp4_measured", note="decode failed"))


def _uncertain_review() -> PerceptualReview:
    return _review(CapabilityReport("cap1", UNCERTAIN, "mp4_measured"))


def _sequence(*reviews):
    it = iter(reviews)

    def _fn(output_path, draft, segments, windows):
        return next(it)

    return _fn


def _always_pass_qc(*, output_path="repaired.mp4"):
    def _fn(draft, segments, out_path):
        return SimpleNamespace(status="PASS", output_path=output_path)

    return _fn


# =============================================================================
# Immediate PASS / HUMAN_REVIEW_REQUIRED -- no repair needed or attempted
# =============================================================================

def test_system_pass_returns_immediately_no_repair_attempted():
    result = cycle.run_perceptual_repair_cycle(
        draft=object(), segments=(_segment(),), output_path="out.mp4",
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_pass_review()),
        output_windows=_fake_windows,
    )
    assert result.status == cycle.STATUS_PASS
    assert result.attempts == ()
    assert result.repair_loop_kind == "perceptual_post_render"


def test_uncertain_finding_never_auto_repaired_goes_to_human_review():
    result = cycle.run_perceptual_repair_cycle(
        draft=object(), segments=(_segment(),), output_path="out.mp4",
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_uncertain_review()),
        output_windows=_fake_windows,
    )
    assert result.status == cycle.STATUS_NEEDS_HUMAN_REVIEW
    assert result.attempts == ()


# =============================================================================
# Physical (BoundaryEngine-routed) FAIL -- reuses the EXISTING repair authority
# =============================================================================

def test_physical_finding_is_repaired_reverified_and_reaches_pass(monkeypatch):
    repaired_segments = (_segment(end=1.4),)

    def fake_repair(segments, finding):
        attempt = SegmentRepairAttempt(
            segment_index=0, clip_id="clip-1", finding_kind=finding.kind, edge="trailing",
            original_start=0.0, original_end=3.0, repaired_start=0.0, repaired_end=1.4,
            trim_sec=1.6, reason="trimmed_trailing_edge_by_1.600s_for_test",
        )
        return repaired_segments, attempt

    monkeypatch.setattr(cycle, "repair_segment_for_finding", fake_repair)

    result = cycle.run_perceptual_repair_cycle(
        draft=object(), segments=(_segment(),), output_path="out.mp4",
        render_and_technical_qc=_always_pass_qc(output_path="repaired-out.mp4"),
        perceptual_review=_sequence(_physical_fail_review(), _pass_review()),
        output_windows=_fake_windows,
    )
    assert result.status == cycle.STATUS_PASS
    assert len(result.attempts) == 1
    assert result.attempts[0].repaired is True
    assert result.attempts[0].repair_kind == cycle.REPAIR_KIND_PHYSICAL_BOUNDARY
    assert result.final_segments == repaired_segments
    assert result.output_path == "repaired-out.mp4"


def test_no_safe_repair_stays_blocked_with_explicit_reason(monkeypatch):
    """The hard floor / "no safe repair" refusal is INHERITED from
    `repair_segment_for_finding` -- this module never re-implements it or
    second-guesses a `None` result."""
    monkeypatch.setattr(cycle, "repair_segment_for_finding", lambda segments, finding: None)

    result = cycle.run_perceptual_repair_cycle(
        draft=object(), segments=(_segment(),), output_path="out.mp4",
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_physical_fail_review()),
        output_windows=_fake_windows,
    )
    assert result.status == cycle.STATUS_BLOCKED
    assert result.attempts[0].repaired is False
    assert result.attempts[0].reason == "no_safe_repair_within_hard_floor"


def test_repair_that_breaks_technical_qc_stays_blocked_never_delivered(monkeypatch):
    def fake_repair(segments, finding):
        attempt = SegmentRepairAttempt(
            segment_index=0, clip_id="clip-1", finding_kind=finding.kind, edge="trailing",
            original_start=0.0, original_end=3.0, repaired_start=0.0, repaired_end=1.4,
            trim_sec=1.6, reason="trimmed_trailing_edge_by_1.600s_for_test",
        )
        return (_segment(end=1.4),), attempt

    monkeypatch.setattr(cycle, "repair_segment_for_finding", fake_repair)

    def failing_qc(draft, segments, out_path):
        return SimpleNamespace(status="SEMANTIC_MISMATCH_INVALIDATED", output_path=None)

    result = cycle.run_perceptual_repair_cycle(
        draft=object(), segments=(_segment(),), output_path="out.mp4",
        render_and_technical_qc=failing_qc,
        perceptual_review=_sequence(_physical_fail_review()),
        output_windows=_fake_windows,
    )
    assert result.status == cycle.STATUS_BLOCKED
    assert result.attempts[0].reason == "repair_broke_technical_qc"
    # never silently "delivered" -- output_path stays the ORIGINAL, unrepaired file
    assert result.output_path == "out.mp4"


def test_repair_loop_is_bounded_never_spins_forever(monkeypatch):
    """A finding that keeps getting "repaired" but the re-review keeps
    finding it BLOCKED again must stop at max_attempts, never loop
    indefinitely."""
    call_count = {"n": 0}

    def fake_repair(segments, finding):
        call_count["n"] += 1
        attempt = SegmentRepairAttempt(
            segment_index=0, clip_id="clip-1", finding_kind=finding.kind, edge="trailing",
            original_start=0.0, original_end=3.0, repaired_start=0.0, repaired_end=1.4,
            trim_sec=0.1, reason="trimmed",
        )
        return (_segment(end=1.4),), attempt

    monkeypatch.setattr(cycle, "repair_segment_for_finding", fake_repair)

    result = cycle.run_perceptual_repair_cycle(
        draft=object(), segments=(_segment(),), output_path="out.mp4",
        render_and_technical_qc=_always_pass_qc(),
        perceptual_review=_sequence(
            _physical_fail_review(), _physical_fail_review(), _physical_fail_review(),
        ),
        output_windows=_fake_windows,
        max_attempts=2,
    )
    assert result.status == cycle.STATUS_BLOCKED
    assert result.attempts[-1].reason == "max_attempts_exhausted"
    assert call_count["n"] == 2  # never exceeds max_attempts


# =============================================================================
# Semantic (BestTakeResolver-routed) FAIL -- NO automatic repair, ever
# =============================================================================

def test_semantic_finding_never_auto_repaired_no_content_removed(monkeypatch):
    repair_calls = []
    monkeypatch.setattr(cycle, "repair_segment_for_finding", lambda *a: repair_calls.append(1) or None)

    result = cycle.run_perceptual_repair_cycle(
        draft=object(), segments=(_segment(), _segment(clip_id="clip-2", start=4.0, end=6.0)),
        output_path="out.mp4",
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_semantic_fail_review()),
        output_windows=_fake_windows,
    )
    assert result.status == cycle.STATUS_NEEDS_SEMANTIC_REVIEW
    assert result.attempts[0].repair_kind == cycle.REPAIR_KIND_NO_AUTOMATIC_SEMANTIC
    assert result.attempts[0].repaired is False
    # repair_segment_for_finding (the physical-only authority) is never
    # even consulted for a semantic finding.
    assert repair_calls == []
    # the segments handed back are UNCHANGED -- nothing was silently trimmed.
    assert len(result.final_segments) == 2


def test_error_capability_with_no_findings_stays_blocked_never_fabricates_a_target():
    result = cycle.run_perceptual_repair_cycle(
        draft=object(), segments=(_segment(),), output_path="out.mp4",
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_error_review()),
        output_windows=_fake_windows,
    )
    assert result.status == cycle.STATUS_BLOCKED
    assert result.attempts[0].repair_kind == cycle.REPAIR_KIND_ERROR_CAPABILITY
    assert result.attempts[0].finding_kind is None


# =============================================================================
# build_semantic_repair_plan -- the "return to Selection, new plan version,
# refreeze" contract
# =============================================================================

def test_semantic_repair_plan_increments_version_never_reuses_v1():
    from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
    from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION

    def _clip(clip_id, start, end, text):
        return DraftClip(
            clip_id=clip_id, source_asset_id="src", source_order=0,
            start=start, end=end, text=text, caption_text=text, selected=True,
        )

    draft_v1 = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(_clip("a", 0.0, 2.0, "one idea"), _clip("b", 3.0, 5.0, "one idea restated")),
        alternates=(), discarded=(), diagnostics={"take_judge_groups": []},
    )
    plan_v1 = build_canonical_edit_plan(draft_v1)
    assert plan_v1.plan_version == 1

    # The caller (human/PO-authorized) supplies the corrected draft -- this
    # function never decides which clip to drop.
    draft_v2 = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(_clip("a", 0.0, 2.0, "one idea"),),
        alternates=(), discarded=(_clip("b", 3.0, 5.0, "one idea restated"),),
        diagnostics={"take_judge_groups": []},
    )
    plan_v2 = cycle.build_semantic_repair_plan(draft_v2, plan_v1)
    assert plan_v2.plan_version == 2
    assert plan_v2.plan_id != plan_v1.plan_id  # different content -> different plan identity
    assert len(plan_v2.keep_sequence) == 1
