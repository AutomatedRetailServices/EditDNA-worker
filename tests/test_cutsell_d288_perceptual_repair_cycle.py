"""D-288 -- bounded post-render PERCEPTUAL repair cycle (still DISCONNECTED
from every live caller -- see the module's own docstring).

Audit finding this closes (`docs/CUTSELL_DECISIONS.md` D-288): a perceptual
finding's `routes_to` label was read in exactly one place in the whole
codebase (a summary counter) and never actually routed to a real repair
authority. `cutsell_worker/perceptual_repair_cycle.py` is the real, bounded
connection: finding -> owning authority -> safe repair -> render -> technical
QC -> Watch+Listen (re-review).

Correction (this file, second pass, findings 3/4/6): a real-code review
found `live_boundary_repair.repair_segment_for_finding` does not itself
check word boundaries and can, at a shared cut, attribute a finding to the
WRONG neighboring segment. This module now enforces THREE independent
preconditions above that function -- exact/unambiguous segment identity,
an explicit allowlist of confirmed (not merely "FAIL"-labeled) defect
kinds, and a caller-supplied real word floor -- before ever accepting its
repair. This file proves all three, plus finding 4 (using the segments AS
RENDERED, including the renderer's own trims, on every re-review).

Dependency-injected throughout (the render/technical-QC and perceptual-
review callables are fakes the test controls), except for the ONE real
authority this module is required to reuse rather than reimplement:
`live_boundary_repair.repair_segment_for_finding`, monkeypatched here at
the point `perceptual_repair_cycle` imports it -- same established pattern
this repo already uses for `live_render_qc.py`'s own tests. No real ffmpeg/
media decode is needed. No Video00 fact/id anywhere below.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cutsell_worker import perceptual_repair_cycle as cycle
from cutsell_worker.live_boundary_repair import SegmentRepairAttempt
from cutsell_worker.perceptual_watch_listen import (
    DEAD_AIR_FAIL_SEC,
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


def _passthrough_segments_as_rendered(segments, trims):
    return tuple(segments), 0


def _review(*capabilities) -> PerceptualReview:
    return PerceptualReview(status=overall_status(capabilities), gate_mode=GATE_MODE_STATE_MACHINE_V1, capabilities=tuple(capabilities))


def _pass_review() -> PerceptualReview:
    return _review(CapabilityReport("cap1", EVALUATED_PASS, "mp4_measured"))


def _confirmed_dead_air_finding(start=0.5, end=0.5 + DEAD_AIR_FAIL_SEC):
    return PerceptualFinding(
        "cap1", "PERCEPTUAL_INTERIOR_DEAD_AIR", start, end, "FAIL", ROUTE_BOUNDARY,
        detail={"duration_sec": DEAD_AIR_FAIL_SEC},
    )


def _physical_fail_review(finding=None) -> PerceptualReview:
    finding = finding or _confirmed_dead_air_finding()
    return _review(CapabilityReport("cap1", EVALUATED_FAIL, "mp4_measured", (finding,)))


def _unconfirmed_reset_candidate_finding(start=2.7, end=3.05):
    # A "candidate" motion event with NO measured pause nearby -- exactly
    # the shape finding 3 says must never authorize a repair on its own,
    # even though it is routed to BoundaryEngine within the generic
    # EDGE_DEBRIS_WINDOW_SEC.
    return PerceptualFinding(
        "cap2", "PERCEPTUAL_RESET_DEBRIS_AT_EDGE", start, end, "FAIL", ROUTE_BOUNDARY,
        detail={"measured_pause_nearby": False},
    )


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
        return SimpleNamespace(status="PASS", output_path=output_path, attempts=())

    return _fn


def _run(*, segments=(_segment(),), word_floor_by_clip_id=None, **kwargs):
    defaults = dict(
        draft=object(), segments=segments, output_path="out.mp4",
        output_windows=_fake_windows, segments_as_rendered=_passthrough_segments_as_rendered,
        word_floor_by_clip_id=word_floor_by_clip_id if word_floor_by_clip_id is not None else {
            seg.clip_id: (seg.start, seg.end) for seg in segments
        },
    )
    defaults.update(kwargs)
    return cycle.run_perceptual_repair_cycle(**defaults)


# =============================================================================
# Immediate PASS / HUMAN_REVIEW_REQUIRED -- no repair needed or attempted
# =============================================================================

def test_system_pass_returns_immediately_no_repair_attempted():
    result = _run(
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_pass_review()),
    )
    assert result.status == cycle.STATUS_PASS
    assert result.attempts == ()
    assert result.repair_loop_kind == "perceptual_post_render"


def test_uncertain_finding_never_auto_repaired_goes_to_human_review():
    result = _run(
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_uncertain_review()),
    )
    assert result.status == cycle.STATUS_NEEDS_HUMAN_REVIEW
    assert result.attempts == ()


# =============================================================================
# Physical (BoundaryEngine-routed), CONFIRMED FAIL -- reuses the EXISTING
# repair authority, gated by the three D-288 preconditions
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

    result = _run(
        # Word floor at 1.4 -- exactly at the repaired trailing edge, so
        # the repair does not cross it.
        word_floor_by_clip_id={"clip-1": (0.0, 1.4)},
        render_and_technical_qc=_always_pass_qc(output_path="repaired-out.mp4"),
        perceptual_review=_sequence(_physical_fail_review(), _pass_review()),
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

    result = _run(
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_physical_fail_review()),
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
        return SimpleNamespace(status="SEMANTIC_MISMATCH_INVALIDATED", output_path=None, attempts=())

    result = _run(
        word_floor_by_clip_id={"clip-1": (0.0, 1.4)},
        render_and_technical_qc=failing_qc,
        perceptual_review=_sequence(_physical_fail_review()),
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

    result = _run(
        word_floor_by_clip_id={"clip-1": (0.0, 1.4)},
        render_and_technical_qc=_always_pass_qc(),
        perceptual_review=_sequence(
            _physical_fail_review(), _physical_fail_review(), _physical_fail_review(),
        ),
        max_attempts=2,
    )
    assert result.status == cycle.STATUS_BLOCKED
    assert result.attempts[-1].reason == "max_attempts_exhausted"
    assert call_count["n"] == 2  # never exceeds max_attempts


# =============================================================================
# Finding 3.a: a bare "candidate"/routing-window match never authorizes a repair
# =============================================================================

def test_unconfirmed_candidate_motion_finding_never_authorizes_a_repair(monkeypatch):
    """A reset/break `severity=FAIL` finding with NO measured pause nearby
    (a bare kinematic "candidate", D-149) must never reach `repair_
    segment_for_finding` at all -- a generic 0.35s edge-routing window is
    routing, never repair authorization."""
    repair_calls = []
    monkeypatch.setattr(cycle, "repair_segment_for_finding", lambda *a: repair_calls.append(1) or None)

    result = _run(
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_physical_fail_review(_unconfirmed_reset_candidate_finding())),
    )
    assert result.status == cycle.STATUS_BLOCKED
    assert result.attempts[0].reason == "finding_not_confirmed_repairable_defect"
    assert repair_calls == []


def test_a_dead_air_finding_below_the_fail_floor_is_never_confirmed():
    short_finding = PerceptualFinding(
        "cap1", "PERCEPTUAL_INTERIOR_DEAD_AIR", 0.5, 0.9, "FAIL", ROUTE_BOUNDARY,
        detail={"duration_sec": 0.4},  # below DEAD_AIR_FAIL_SEC
    )
    assert cycle._is_confirmed_repairable_defect(short_finding) is False


def test_a_speech_energy_at_cut_finding_is_never_in_the_repairable_allowlist():
    finding = PerceptualFinding("cap1", "PERCEPTUAL_SPEECH_ENERGY_AT_CUT", 1.0, 1.06, "FAIL", ROUTE_BOUNDARY, detail={})
    assert cycle._is_confirmed_repairable_defect(finding) is False


# =============================================================================
# Finding 3.b: exact segment identity -- never the neighboring clip at a
# shared cut boundary
# =============================================================================

def test_shared_cut_boundary_ambiguity_refuses_rather_than_guessing(monkeypatch):
    """Two adjacent segments sharing a cut at t=3.0: a finding sitting
    right at that instant could equally be segment 0's trailing edge or
    segment 1's leading edge. This module must refuse rather than let
    `repair_segment_for_finding`'s own "first match wins" loop silently
    pick the WRONG (earlier) clip."""
    repair_calls = []
    monkeypatch.setattr(cycle, "repair_segment_for_finding", lambda *a: repair_calls.append(1) or None)

    segments = (_segment(clip_id="clip-A", start=0.0, end=3.0), _segment(clip_id="clip-B", start=3.0, end=6.0))
    # Sits exactly on the shared boundary -- ambiguous under the SAME
    # tolerance repair_segment_for_finding itself uses.
    finding = PerceptualFinding("cap1", "PERCEPTUAL_INTERIOR_DEAD_AIR", 2.9, 3.1, "FAIL", ROUTE_BOUNDARY, detail={"duration_sec": DEAD_AIR_FAIL_SEC})

    result = _run(
        segments=segments,
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_physical_fail_review(finding)),
    )
    assert result.status == cycle.STATUS_BLOCKED
    assert result.attempts[0].reason == "ambiguous_or_no_matching_segment_edge"
    # The real repair authority was never even consulted -- this module
    # refuses BEFORE delegating, it never lets that function guess.
    assert repair_calls == []


def test_repair_never_touches_the_neighboring_clip(monkeypatch):
    """Direct proof of the exact failure mode named by the correction:
    a finding whose coordinates are unambiguously segment 1's own leading
    edge must never result in segment 0 (the neighbor) being trimmed."""
    segments = (_segment(clip_id="clip-A", start=0.0, end=3.0), _segment(clip_id="clip-B", start=3.0, end=6.0))
    # Clearly inside segment 1's own leading edge, far from segment 0's
    # trailing edge tolerance window.
    finding = PerceptualFinding("cap1", "PERCEPTUAL_INTERIOR_DEAD_AIR", 3.05, 3.05 + DEAD_AIR_FAIL_SEC, "FAIL", ROUTE_BOUNDARY, detail={"duration_sec": DEAD_AIR_FAIL_SEC})

    repaired_segments = (segments[0], _segment(clip_id="clip-B", start=3.05 + DEAD_AIR_FAIL_SEC, end=6.0))

    def fake_repair(segs, f):
        attempt = SegmentRepairAttempt(
            segment_index=1, clip_id="clip-B", finding_kind=f.kind, edge="leading",
            original_start=3.0, original_end=6.0, repaired_start=3.05 + DEAD_AIR_FAIL_SEC, repaired_end=6.0,
            trim_sec=DEAD_AIR_FAIL_SEC, reason="trimmed_leading_edge_for_test",
        )
        return repaired_segments, attempt

    monkeypatch.setattr(cycle, "repair_segment_for_finding", fake_repair)

    result = _run(
        segments=segments,
        word_floor_by_clip_id={"clip-A": (0.0, 3.0), "clip-B": (3.05 + DEAD_AIR_FAIL_SEC, 6.0)},
        render_and_technical_qc=_always_pass_qc(),
        perceptual_review=_sequence(_physical_fail_review(finding), _pass_review()),
    )
    assert result.status == cycle.STATUS_PASS
    # clip-A (the neighbor) is byte-identical to its original -- untouched.
    assert result.final_segments[0] == segments[0]
    assert result.final_segments[1].clip_id == "clip-B"


def test_repair_target_disagreeing_with_disambiguation_is_refused(monkeypatch):
    """A synthetic worst case: `repair_segment_for_finding` returns a
    repair for a DIFFERENT clip than this module's own independent
    disambiguation identified. Neither guess is trusted over the other;
    the repair is refused."""
    segments = (_segment(clip_id="clip-A", start=0.0, end=3.0),)
    finding = _confirmed_dead_air_finding()  # unambiguously clip-A's trailing edge

    def fake_repair(segs, f):
        # Deliberately returns a bogus attempt naming a DIFFERENT clip_id
        # than the one that was actually matched -- simulates a defect or
        # disagreement in the shared authority.
        attempt = SegmentRepairAttempt(
            segment_index=0, clip_id="clip-DIFFERENT", finding_kind=f.kind, edge="trailing",
            original_start=0.0, original_end=3.0, repaired_start=0.0, repaired_end=1.4,
            trim_sec=1.6, reason="trimmed",
        )
        return (_segment(clip_id="clip-DIFFERENT", end=1.4),), attempt

    monkeypatch.setattr(cycle, "repair_segment_for_finding", fake_repair)

    result = _run(
        segments=segments,
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_physical_fail_review(finding)),
    )
    assert result.status == cycle.STATUS_BLOCKED
    assert result.attempts[0].reason == "repair_target_disagrees_with_disambiguated_segment"


# =============================================================================
# Finding 3.c: protected speech boundaries, independently re-verified
# =============================================================================

def test_missing_word_floor_evidence_refuses_the_repair(monkeypatch):
    monkeypatch.setattr(cycle, "repair_segment_for_finding", lambda segs, f: (
        (_segment(end=1.4),),
        SegmentRepairAttempt(
            segment_index=0, clip_id="clip-1", finding_kind=f.kind, edge="trailing",
            original_start=0.0, original_end=3.0, repaired_start=0.0, repaired_end=1.4,
            trim_sec=1.6, reason="trimmed",
        ),
    ))
    result = _run(
        word_floor_by_clip_id={},  # nothing supplied for clip-1
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_physical_fail_review()),
    )
    assert result.status == cycle.STATUS_BLOCKED
    assert result.attempts[0].reason == "no_word_floor_evidence_supplied"


def test_a_repair_that_would_cross_the_last_aligned_word_is_refused(monkeypatch):
    """The exact scenario the correction names: `repair_segment_for_
    finding`'s leading-edge branch has NO measured-silence floor at all,
    so a bogus repair that would cut INTO real speech must be caught
    here, independently, even though the shared authority itself allowed
    it through."""
    monkeypatch.setattr(cycle, "repair_segment_for_finding", lambda segs, f: (
        (_segment(end=1.0),),  # trims to 1.0 -- but the last real word ends at 1.4
        SegmentRepairAttempt(
            segment_index=0, clip_id="clip-1", finding_kind=f.kind, edge="trailing",
            original_start=0.0, original_end=3.0, repaired_start=0.0, repaired_end=1.0,
            trim_sec=2.0, reason="trimmed",
        ),
    ))
    result = _run(
        word_floor_by_clip_id={"clip-1": (0.0, 1.4)},  # last aligned word ends at 1.4
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_physical_fail_review()),
    )
    assert result.status == cycle.STATUS_BLOCKED
    assert result.attempts[0].reason == "repair_would_cross_protected_speech_boundary"


def test_a_leading_edge_repair_that_would_cross_the_first_aligned_word_is_refused(monkeypatch):
    monkeypatch.setattr(cycle, "repair_segment_for_finding", lambda segs, f: (
        (_segment(start=0.6),),
        SegmentRepairAttempt(
            segment_index=0, clip_id="clip-1", finding_kind=f.kind, edge="leading",
            original_start=0.0, original_end=3.0, repaired_start=0.6, repaired_end=3.0,
            trim_sec=0.6, reason="trimmed",
        ),
    ))
    result = _run(
        word_floor_by_clip_id={"clip-1": (0.3, 3.0)},  # first aligned word starts at 0.3
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_physical_fail_review()),
    )
    assert result.status == cycle.STATUS_BLOCKED
    assert result.attempts[0].reason == "repair_would_cross_protected_speech_boundary"


def test_a_repair_that_exactly_meets_the_word_floor_is_accepted(monkeypatch):
    monkeypatch.setattr(cycle, "repair_segment_for_finding", lambda segs, f: (
        (_segment(end=1.4),),
        SegmentRepairAttempt(
            segment_index=0, clip_id="clip-1", finding_kind=f.kind, edge="trailing",
            original_start=0.0, original_end=3.0, repaired_start=0.0, repaired_end=1.4,
            trim_sec=1.6, reason="trimmed",
        ),
    ))
    result = _run(
        word_floor_by_clip_id={"clip-1": (0.0, 1.4)},  # exactly at the repaired edge
        render_and_technical_qc=_always_pass_qc(),
        perceptual_review=_sequence(_physical_fail_review(), _pass_review()),
    )
    assert result.status == cycle.STATUS_PASS


# =============================================================================
# Finding 4: segments AS RENDERED (renderer trims folded in before re-review)
# =============================================================================

def test_renderer_trims_are_folded_in_before_the_next_re_review(monkeypatch):
    monkeypatch.setattr(cycle, "repair_segment_for_finding", lambda segs, f: (
        (_segment(end=1.4),),
        SegmentRepairAttempt(
            segment_index=0, clip_id="clip-1", finding_kind=f.kind, edge="trailing",
            original_start=0.0, original_end=3.0, repaired_start=0.0, repaired_end=1.4,
            trim_sec=1.6, reason="trimmed",
        ),
    ))

    renderer_further_trimmed = (_segment(end=1.2),)  # the RENDERER additionally shortened it to 1.2

    def fake_segments_as_rendered(segments, trims):
        assert trims == ({"clip_id": "clip-1", "tightened_end": 1.2},)
        return renderer_further_trimmed, 1

    def fake_qc(draft, segments, out_path):
        attempt = SimpleNamespace(renderer_trailing_trims=({"clip_id": "clip-1", "tightened_end": 1.2},))
        return SimpleNamespace(status="PASS", output_path="repaired.mp4", attempts=(attempt,))

    seen_segments_in_review = []

    def _review_seeing(output_path, draft, segments, windows):
        seen_segments_in_review.append(segments)
        return _pass_review() if seen_segments_in_review[1:] else _physical_fail_review()

    result = cycle.run_perceptual_repair_cycle(
        draft=object(), segments=(_segment(),), output_path="out.mp4",
        render_and_technical_qc=fake_qc,
        perceptual_review=_review_seeing,
        output_windows=_fake_windows,
        segments_as_rendered=fake_segments_as_rendered,
        word_floor_by_clip_id={"clip-1": (0.0, 1.2)},
    )
    assert result.status == cycle.STATUS_PASS
    # The SECOND review call (the re-review after repair) must have seen
    # the RENDERER-adjusted segments (end=1.2), never the pre-render
    # repair-only segments (end=1.4).
    assert seen_segments_in_review[1] == renderer_further_trimmed
    assert result.final_segments == renderer_further_trimmed


# =============================================================================
# Semantic (BestTakeResolver-routed) FAIL -- NO automatic repair, ever
# =============================================================================

def test_semantic_finding_never_auto_repaired_no_content_removed(monkeypatch):
    repair_calls = []
    monkeypatch.setattr(cycle, "repair_segment_for_finding", lambda *a: repair_calls.append(1) or None)

    segments = (_segment(), _segment(clip_id="clip-2", start=4.0, end=6.0))
    result = _run(
        segments=segments,
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_semantic_fail_review()),
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
    result = _run(
        render_and_technical_qc=lambda *a: (_ for _ in ()).throw(AssertionError("must not render")),
        perceptual_review=_sequence(_error_review()),
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
