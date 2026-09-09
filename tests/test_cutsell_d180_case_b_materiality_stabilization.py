"""D-180 -- CASE-B fast-path condition-4 materiality stabilization.

BOUNDED REFINEMENT (docs/CUTSELL_DECISIONS.md D-180, post D-179 forensic).
`_case_b_fast_path_conflict`'s condition 4 originally compared RAW
`delivery_event_count` alone (D-179 proved this is EVIDENCE, not
MATERIALITY -- a raw provider-perceived count difference can be entirely
non-material). This suite proves the ONE authorized change:

- condition 4 becomes 4a (raw count asymmetry, unchanged) AND 4b (the
  same asymmetry must ALSO be corroborated by D-097's own existing,
  already-tested `d097_would_be_counted` materiality doctrine -- no new
  confidence/duration/density cutoff of this module's own invention);
- a genuinely material/severe DELIVERY performance difference still
  bypasses (the mechanism stays actionable, never neutered);
- a large raw count gap that is NOT materially corroborated (D-179's own
  gynecologist shape) no longer bypasses;
- D-150's `ABSTAIN_CONFLICT` firewall is untouched -- the fast path never
  even runs, so condition 4 is never reached, so D-180 cannot manufacture
  a semantic winner where D-150 correctly abstains (pimples);
- the refinement never rewrites conditions 1-3, `_semantic_best_take`'s
  Steps 6-9, the terminal tie-break, Family Formation, D-150, Boundary,
  Pacing, or the Renderer;
- introduces no new feature flag and no new threshold family (D-167's V2
  severity vocabulary is deliberately NOT reused -- it is flag-gated,
  while this gate runs unconditionally; using it would make condition 4's
  behavior silently depend on an unrelated flag).
"""
from __future__ import annotations

import inspect

import pytest

from cutsell_worker.case_b_performance_evidence import (
    CaseBEvent,
    CaseBPerformanceEvidence,
    build_case_b_performance_evidence,
)
from cutsell_worker.contracts import (
    CandidateTake, ProcessingRequest, RankedTake, SemanticLabel, SemanticRole, Word,
)
from cutsell_worker.pipeline import (
    _WINNER_PATH_DELIVERYSCORE_PATH,
    _WINNER_PATH_SEMANTIC_FAST_PATH,
    _case_b_condition4_diagnostics,
    _case_b_fast_path_conflict,
    _material_delivery_event_count,
    _meaning_sufficient_member_ids,
    _semantic_best_take,
    build_flow_b_draft,
)
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.semantic_authority_observability import AUTHORITY_ABSTAIN_CONFLICT
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _words(text: str, start: float, end: float):
    tokens = text.split()
    step = (end - start) / len(tokens)
    return tuple(Word(t, start + i * step, start + (i + 1) * step) for i, t in enumerate(tokens))


def _take(clip_id, start, end, text, *, source="src", words=None, signals=None, complete_idea=True):
    return CandidateTake(
        clip_id, source, 0, start, end, text,
        words=words if words is not None else _words(text, start, end),
        signals=signals, complete_idea=complete_idea,
    )


def _event(kind, start, end, *, source="src", confidence=0.9, description="candidate"):
    return TemporalEvent(source, start, end, kind, confidence, description)


def _context(source_id, events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id=source_id, summary="", dominant_style="", creator_intent="",
            events=tuple(events),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def ranked(*pairs):
    return tuple(RankedTake(clip_id, score, "watch_listen_baseline") for clip_id, score in pairs)


def _evidence_map(members, context):
    return {member.clip_id: build_case_b_performance_evidence(member, context) for member in members}


def _synthetic_evidence(candidate_id: str, events: tuple[CaseBEvent, ...]) -> CaseBPerformanceEvidence:
    """A hand-built evidence record, for exercising `_material_delivery_
    event_count`/`_case_b_condition4_diagnostics` directly without going
    through the real geometry/confidence-floor replay -- used only where a
    real fixture would be more indirection than signal."""
    return CaseBPerformanceEvidence(
        candidate_id=candidate_id, source_asset_id="src", delivery_available=True,
        delivery_start=0.0, delivery_end=10.0, delivery_span_duration=10.0,
        delivery_events=events, delivery_event_count=len(events),
        delivery_event_duration_total=sum(e.duration for e in events),
        count_by_kind={}, duration_by_kind={}, event_density=None,
    )


def _synthetic_event(kind: str, *, would_be_counted: bool) -> CaseBEvent:
    return CaseBEvent(
        kind=kind, start=1.0, end=1.5, confidence=0.9, duration=0.5,
        evidence_source="test", straddle=False, mediasignal_fields=(),
        d097_geometrically_inside=would_be_counted, d097_meets_confidence_floor=would_be_counted,
        d097_would_be_counted=would_be_counted,
    )


# === Items 1-13: `_case_b_fast_path_conflict` condition 4a/4b matrix ========

def test_01_equal_event_count_no_conflict():
    """Item 1: equal raw counts -- condition 4a itself never fires."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [
        _event("body_reset_candidate", 1.0, 1.5, confidence=0.9),
        _event("hand_motion_reset_candidate", 5.0, 5.5, confidence=0.9),
    ])
    evidence = _evidence_map((a, b), context)
    assert evidence["A"].delivery_event_count == evidence["B"].delivery_event_count == 1
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_02_small_gap_zero_confidence_events_none_materiality_no_conflict():
    """Item 2: a small raw count gap where the extra events carry
    confidence far below D-097's own floor (0.88/0.76) -- NONE
    materiality, no conflict under D-180 (would have conflicted pre-D-180
    on raw count alone)."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [
        _event("body_reset_candidate", 1.0, 1.5, confidence=0.2),
        _event("hand_motion_reset_candidate", 2.0, 2.3, confidence=0.3),
    ])
    evidence = _evidence_map((a, b), context)
    assert evidence["A"].delivery_event_count == 2  # raw count gap present (4a)
    assert _material_delivery_event_count(evidence["A"]) == 0  # not materially corroborated
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_03_small_gap_near_floor_low_confidence_mild_no_conflict():
    """Item 3: events just under D-097's confidence floor -- MILD/
    unconfirmed, not material, no conflict."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [
        _event("body_reset_candidate", 1.0, 1.5, confidence=0.87),  # just under 0.88 RESET floor
    ])
    evidence = _evidence_map((a, b), context)
    assert evidence["A"].delivery_event_count == 1
    assert _material_delivery_event_count(evidence["A"]) == 0
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_04_large_raw_gap_still_not_material_no_conflict():
    """Item 4 (the D-179 key case): a LARGE raw count gap (6 extra events)
    that is entirely low-confidence/edge -- still not materially
    corroborated. Raw count alone would have bypassed pre-D-180; D-180
    requires materiality too."""
    a = _take("A", 0.0, 10.0, "one two three four five six seven eight",
              words=_words("one two three four five six seven eight", 0.3, 9.7))
    b = _take("B", 10.0, 14.0, "nine ten eleven twelve", words=_words("nine ten eleven twelve", 10.2, 13.8))
    context = _context("src", [
        _event("hand_motion_reset_candidate", 1.0, 1.2, confidence=0.4),
        _event("hand_motion_reset_candidate", 2.0, 2.2, confidence=0.5),
        _event("hand_motion_reset_candidate", 3.0, 3.2, confidence=0.3),
        _event("body_reset_candidate", 4.0, 4.2, confidence=0.6),
        _event("body_reset_candidate", 5.0, 5.2, confidence=0.2),
        _event("facial_expression_shift_candidate", 6.0, 6.2, confidence=0.5),  # BREAK floor 0.76
    ])
    evidence = _evidence_map((a, b), context)
    assert evidence["A"].delivery_event_count == 6
    assert evidence["B"].delivery_event_count == 0
    assert _material_delivery_event_count(evidence["A"]) == 0
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_05_material_gap_conflict_actionable():
    """Item 5: a modest raw gap where events ARE materially corroborated
    (high confidence, well interior) -- 4a AND 4b both hold, conflict
    fires exactly as pre-D-180 (this shape is unaffected by D-180)."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [
        _event("body_reset_candidate", 1.0, 1.5, confidence=0.92),
        _event("hand_motion_reset_candidate", 2.0, 2.3, confidence=0.95),
    ])
    evidence = _evidence_map((a, b), context)
    assert _material_delivery_event_count(evidence["A"]) == 2
    basis = _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence)
    assert basis is not None
    assert basis["semantic_fast_path_candidate_material_event_count"] == 2
    assert basis["deliveryscore_top_candidate_material_event_count"] == 0


def test_06_severe_gap_conflict_actionable():
    """Item 6: a severe DELIVERY impairment (many high-confidence reset/
    break events) -- clearly actionable under D-180."""
    a = _take("A", 0.0, 10.0, "one two three four five six seven eight",
              words=_words("one two three four five six seven eight", 0.3, 9.7))
    b = _take("B", 10.0, 14.0, "nine ten eleven twelve", words=_words("nine ten eleven twelve", 10.2, 13.8))
    context = _context("src", [
        _event("hand_motion_reset_candidate", 1.0, 1.2, confidence=0.93),
        _event("hand_motion_reset_candidate", 2.0, 2.2, confidence=0.94),
        _event("body_reset_candidate", 4.0, 4.2, confidence=0.9),
        _event("facial_expression_shift_candidate", 6.0, 6.2, confidence=0.85),
    ])
    evidence = _evidence_map((a, b), context)
    assert _material_delivery_event_count(evidence["A"]) == 4
    basis = _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence)
    assert basis is not None


def test_07_entry_only_events_no_conflict():
    """Item 7: ENTRY-only events never enter case_b evidence at all
    (structural D-115/D-122 zone filter, unmodified by D-180)."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 2.0, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [_event("camera_disengagement_candidate", 0.1, 0.5, confidence=0.95)])
    evidence = _evidence_map((a, b), context)
    assert evidence["A"].delivery_event_count == 0
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_08_exit_only_events_no_conflict():
    """Item 8: EXIT-only events never enter case_b evidence either."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 6.5))
    context = _context("src", [_event("hand_motion_reset_candidate", 6.7, 7.2, confidence=0.95)])
    evidence = _evidence_map((a, b), context)
    assert evidence["B"].delivery_event_count == 0
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_09_case_c_ambiguous_straddle_not_geometrically_inside_no_conflict():
    """Item 9: a CASE-C ambiguous both-edge-straddle-shaped event sitting
    right at D-097's own edge margin (`_CLEANLINESS_EDGE_MARGIN_SEC` =
    0.35s) -- geometrically NOT interior, so `d097_would_be_counted` is
    False regardless of confidence. Ambiguous CASE-C fails open (never
    counted toward materiality), never toward conflict."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    # Event at [0.0, 0.2] is inside A's take span but well within the 0.35s
    # edge margin -- excluded by `_interior_events` geometry, not by this
    # module's own invention.
    context = _context("src", [_event("body_reset_candidate", 0.0, 0.2, confidence=0.99)])
    evidence = _evidence_map((a, b), context)
    assert _material_delivery_event_count(evidence["A"]) == 0
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_10_missing_materiality_evidence_fails_open():
    """Item 10: evidence missing for one competitor -- fails open (no
    conflict), unchanged pre/post D-180 (condition check happens before
    4a/4b are ever evaluated)."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    context = _context("src", [_event("body_reset_candidate", 1.0, 1.5, confidence=0.95)])
    evidence = _evidence_map((a,), context)  # "B" never computed
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_11_conflicting_materiality_evidence_tied_no_conflict():
    """Item 11: both sides have the SAME material event count (a genuine
    tie under materiality, even though raw counts might differ) -- no
    conflict, never a coin-flip."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [
        _event("body_reset_candidate", 1.0, 1.5, confidence=0.9),   # material, A
        _event("body_reset_candidate", 0.5, 0.7, confidence=0.3),   # non-material extra, A (raw gap only)
        _event("hand_motion_reset_candidate", 5.0, 5.5, confidence=0.9),  # material, B
    ])
    evidence = _evidence_map((a, b), context)
    assert evidence["A"].delivery_event_count == 2  # raw gap: 2 vs 1
    assert _material_delivery_event_count(evidence["A"]) == 1
    assert _material_delivery_event_count(evidence["B"]) == 1
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_12_ordinary_expressive_motion_not_material_no_conflict():
    """Item 12: an event kind outside both `_RESET_KINDS` and `_BREAK_
    KINDS` (never meets any confidence floor by construction) -- ordinary
    motion, not material."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [_event("camera_disengagement_candidate", 1.0, 1.5, confidence=0.99)])
    # camera_disengagement_candidate is a BREAK kind (0.76 floor) but is
    # zoned ENTRY/EXIT by D-115 in practice; here it lands inside A's
    # DELIVERY span to exercise the "high confidence but wrong kind
    # bucket" shape generically -- use a genuinely non-reset/non-break
    # synthetic kind instead for an unambiguous "ordinary motion" case.
    ordinary = _synthetic_event("ordinary_expressive_motion", would_be_counted=False)
    ev_a = _synthetic_evidence("A", (ordinary,))
    ev_b = _synthetic_evidence("B", ())
    evidence = {"A": ev_a, "B": ev_b}
    assert _material_delivery_event_count(ev_a) == 0
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_13_breaking_character_material_conflict_actionable():
    """Item 13: a genuine breaking-character DELIVERY defect (facial
    expression shift, high confidence, interior) -- BREAK_KINDS floor
    (0.76) met, material, actionable."""
    a = _take("A", 0.0, 6.0, "one two three four five six",
              words=_words("one two three four five six", 0.3, 5.7))
    b = _take("B", 6.0, 10.0, "seven eight nine ten", words=_words("seven eight nine ten", 6.2, 9.8))
    context = _context("src", [_event("facial_expression_shift_candidate", 3.0, 3.3, confidence=0.8)])
    evidence = _evidence_map((a, b), context)
    assert _material_delivery_event_count(evidence["A"]) == 1
    basis = _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence)
    assert basis is not None


# === Items 14-15: D-179 abstract replay + real-material control ============

def test_14_d179_abstract_gynecologist_replay_no_bypass_post_d180():
    """Item 14 (D-179 abstract replay, generic non-Video00 fixture):
    Candidate B is the decisive semantic winner (confidence 0.95). B has
    MORE raw DELIVERY events than A (the DeliveryScore-preferred
    alternative) -- pre-D-180 shape. But B's extra events are low-
    confidence/marginal (never meeting D-097's own reset/break floor) --
    existing evidence shows the difference is NOT material. PRE-D-180 this
    would have bypassed B's decisive winner; POST-D-180, no bypass -- B
    survives its own single_semantic_winner fast path."""
    candidate_a = _take("candidate_a", 0.0, 5.0, "topic one two three four",
                         words=_words("topic one two three four", 0.3, 4.7))
    candidate_b = _take("candidate_b", 5.0, 10.0, "topic five six seven eight",
                         words=_words("topic five six seven eight", 5.3, 9.7))
    r = ranked(("candidate_b", 0.40), ("candidate_a", 0.90))  # DeliveryScorer prefers A
    context = _context("src", [
        _event("hand_motion_reset_candidate", 6.0, 6.2, confidence=0.5),
        _event("hand_motion_reset_candidate", 7.0, 7.2, confidence=0.4),
        _event("body_reset_candidate", 8.0, 8.2, confidence=0.6),
    ])  # 3 marginal events for B, none material
    decisions = {"candidate_b": ("winner", 0.95), "candidate_a": ("alternate", 0.8)}
    evidence = _evidence_map((candidate_a, candidate_b), context)
    assert evidence["candidate_b"].delivery_event_count == 3  # raw gap present
    assert _material_delivery_event_count(evidence["candidate_b"]) == 0  # not material

    before = _semantic_best_take((candidate_a, candidate_b), decisions, "candidate_a", r)
    assert before == ("candidate_b", "candidate_b", "single_semantic_winner")

    after = _semantic_best_take(
        (candidate_a, candidate_b), decisions, "candidate_a", r, case_b_evidence_by_id=evidence,
    )
    assert after == ("candidate_b", "candidate_b", "single_semantic_winner")  # NO bypass post-D-180


def test_15_real_material_control_bypass_remains_actionable():
    """Item 15 (real-material control): SAME shape as item 14, but B's
    extra events are genuinely material (high confidence, interior) --
    the bypass mechanism remains actionable and falls through to the
    existing DeliveryScorer ladder step, exactly as pre-D-180."""
    candidate_a = _take("candidate_a", 0.0, 5.0, "topic one two three four",
                         words=_words("topic one two three four", 0.3, 4.7))
    candidate_b = _take("candidate_b", 5.0, 10.0, "topic five six seven eight",
                         words=_words("topic five six seven eight", 5.3, 9.7))
    r = ranked(("candidate_b", 0.40), ("candidate_a", 0.90))
    context = _context("src", [
        _event("hand_motion_reset_candidate", 6.0, 6.2, confidence=0.93),
        _event("hand_motion_reset_candidate", 7.0, 7.2, confidence=0.94),
    ])  # 2 material events for B
    decisions = {"candidate_b": ("winner", 0.95), "candidate_a": ("alternate", 0.8)}
    evidence = _evidence_map((candidate_a, candidate_b), context)
    assert _material_delivery_event_count(evidence["candidate_b"]) == 2

    after = _semantic_best_take(
        (candidate_a, candidate_b), decisions, "candidate_a", r, case_b_evidence_by_id=evidence,
    )
    selected, preferred, reason = after
    assert reason != "single_semantic_winner"
    assert reason == "delivery_tie_break_among_survivors"
    assert selected == "candidate_a"  # the EXISTING DeliveryScorer path decided, never CASE B directly


# === Item 16: D-150 pimples firewall =========================================

def test_16_d150_complete_context_conflict_preserved_no_manufactured_winner():
    """Item 16: a genuine D-150 `ABSTAIN_CONFLICT` (D-147's own
    COMPLETE_CONTEXT_CONFLICT state) still abstains -- the fast path never
    runs at all when `semantic_comparative_authority` is ABSTAIN_CONFLICT,
    so condition 4 (pre- or post-D-180) is never even reached. D-180's
    materiality refinement cannot manufacture a semantic winner here,
    exactly like pre-D-180."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    r = ranked(("A", 0.40), ("B", 0.90))
    # Strongly material evidence favoring A -- if condition 4 were somehow
    # reached and could pick a winner directly, this is the shape that
    # would tempt it. It must not matter: ABSTAIN_CONFLICT skips the fast
    # path entirely, before case_b evidence is even consulted.
    context = _context("src", [
        _event("body_reset_candidate", 1.0, 1.5, confidence=0.95),
        _event("hand_motion_reset_candidate", 2.0, 2.3, confidence=0.95),
    ])
    decisions = {"A": ("winner", 0.95), "B": ("alternate", 0.8)}
    evidence = _evidence_map((a, b), context)

    result = _semantic_best_take(
        (a, b), decisions, "B", r,
        case_b_evidence_by_id=evidence,
        semantic_comparative_authority=AUTHORITY_ABSTAIN_CONFLICT,
    )
    selected, preferred, reason = result
    assert reason != "single_semantic_winner"
    # Never a fabricated CASE-B-direct pick -- the general ladder decides.
    assert reason == "delivery_tie_break_among_survivors"
    assert selected == "B"


# === Items 17-20: determinism / independence ================================

def test_17_determinism_same_inputs_twice():
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [
        _event("body_reset_candidate", 1.0, 1.5, confidence=0.92),
        _event("hand_motion_reset_candidate", 2.0, 2.3, confidence=0.95),
    ])
    evidence = _evidence_map((a, b), context)
    r1 = _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence)
    r2 = _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence)
    assert r1 == r2


def test_18_candidate_ordering_reversed_same_result():
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [
        _event("body_reset_candidate", 1.0, 1.5, confidence=0.92),
        _event("hand_motion_reset_candidate", 2.0, 2.3, confidence=0.95),
    ])
    evidence_ab = _evidence_map((a, b), context)
    evidence_ba = _evidence_map((b, a), context)
    r1 = _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence_ab)
    r2 = _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence_ba)
    assert r1 == r2


def test_19_clip_ids_changed_same_result():
    a = _take("clip_x1", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("clip_y2", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [
        _event("body_reset_candidate", 1.0, 1.5, confidence=0.92),
        _event("hand_motion_reset_candidate", 2.0, 2.3, confidence=0.95),
    ])
    evidence = _evidence_map((a, b), context)
    basis = _case_b_fast_path_conflict("clip_x1", "clip_y2", {"clip_x1", "clip_y2"}, evidence)
    assert basis is not None
    assert basis["semantic_fast_path_candidate_material_event_count"] == 2
    assert basis["deliveryscore_top_candidate_material_event_count"] == 0


def test_20_family_group_id_independent_same_result():
    """`_case_b_fast_path_conflict`/`_material_delivery_event_count` never
    reference a family/group id -- the same evidence produces the same
    verdict no matter which family it is evaluated under."""
    source = inspect.getsource(_case_b_fast_path_conflict)
    source += inspect.getsource(_material_delivery_event_count)
    assert "group_id" not in source and "family_id" not in source and "gid" not in source


# === Items 21-28: `_case_b_condition4_diagnostics` reason ladder ============

def test_21_diagnostics_not_evaluated_when_no_candidate():
    diag = _case_b_condition4_diagnostics(None, "B", {"A", "B"}, {"A": object(), "B": object()})
    assert diag["case_b_condition4_reason"] == "no_semantic_fast_path_candidate_or_no_evidence"
    assert diag["case_b_materiality_state"] == "NOT_EVALUATED"


def test_22_diagnostics_no_evidence_at_all():
    diag = _case_b_condition4_diagnostics("A", "B", {"A", "B"}, None)
    assert diag["case_b_condition4_reason"] == "no_semantic_fast_path_candidate_or_no_evidence"


def test_23_diagnostics_already_agrees_reason():
    diag = _case_b_condition4_diagnostics("A", "A", {"A", "B"}, {"A": object()})
    assert diag["case_b_condition4_reason"] == "deliveryscore_already_agrees_with_semantic_winner"


def test_24_diagnostics_alt_meaning_insufficient_reason():
    diag = _case_b_condition4_diagnostics("A", "B", {"A"}, {"A": object(), "B": object()})
    assert diag["case_b_condition4_reason"] == "alternative_meaning_insufficient"


def test_25_diagnostics_evidence_missing_reason():
    diag = _case_b_condition4_diagnostics("A", "B", {"A", "B"}, {"A": object()})
    assert diag["case_b_condition4_reason"] == "evidence_missing_for_one_or_both_candidates"


def test_26_diagnostics_no_count_difference_reason():
    ev_a = _synthetic_evidence("A", (_synthetic_event("body_reset_candidate", would_be_counted=True),))
    ev_b = _synthetic_evidence("B", (_synthetic_event("body_reset_candidate", would_be_counted=True),))
    diag = _case_b_condition4_diagnostics("A", "B", {"A", "B"}, {"A": ev_a, "B": ev_b})
    assert diag["case_b_count_difference_present"] is False
    assert diag["case_b_materiality_state"] == "NO_COUNT_DIFFERENCE"
    assert diag["case_b_condition4_reason"] == "no_raw_count_asymmetry"


def test_27_diagnostics_material_confirmed_reason():
    ev_a = _synthetic_evidence("A", (
        _synthetic_event("body_reset_candidate", would_be_counted=True),
        _synthetic_event("hand_motion_reset_candidate", would_be_counted=True),
    ))
    ev_b = _synthetic_evidence("B", ())
    diag = _case_b_condition4_diagnostics("A", "B", {"A", "B"}, {"A": ev_a, "B": ev_b})
    assert diag["case_b_count_difference_present"] is True
    assert diag["case_b_materiality_evidence_available"] is True
    assert diag["case_b_materiality_state"] == "MATERIAL"
    assert diag["case_b_condition4_actionable"] is True
    assert diag["case_b_condition4_reason"] == "material_count_difference_confirmed"
    assert diag["case_b_materiality_source"] == "d097_would_be_counted"


def test_28_diagnostics_not_material_reason():
    ev_a = _synthetic_evidence("A", (
        _synthetic_event("body_reset_candidate", would_be_counted=False),
        _synthetic_event("hand_motion_reset_candidate", would_be_counted=False),
    ))
    ev_b = _synthetic_evidence("B", ())
    diag = _case_b_condition4_diagnostics("A", "B", {"A", "B"}, {"A": ev_a, "B": ev_b})
    assert diag["case_b_count_difference_present"] is True
    assert diag["case_b_materiality_state"] == "NOT_MATERIAL"
    assert diag["case_b_condition4_actionable"] is False
    assert diag["case_b_condition4_reason"] == "raw_count_difference_not_materially_corroborated"


# === Items 29-34: no-new-threshold / no-flag / no-double-count / wiring ======

def test_29_no_new_threshold_or_score_field_introduced():
    """D-180's core principle: RAW EVENT COUNT IS EVIDENCE, NOT
    MATERIALITY -- but the fix must introduce no NEW threshold family
    either. `CaseBPerformanceEvidence`/`CaseBEvent` gain no new numeric
    cutoff fields; D-180 reuses only the pre-existing `d097_would_be_
    counted` boolean."""
    forbidden = {"materiality_score", "severity_score", "case_b_threshold", "d180_threshold", "weight"}
    fields = {f.lower() for f in CaseBPerformanceEvidence.__dataclass_fields__}
    fields |= {f.lower() for f in CaseBEvent.__dataclass_fields__}
    assert not (fields & forbidden), f"unexpected new threshold-like field(s): {fields & forbidden}"


def test_30_no_provider_or_network_reference_in_d180_helpers():
    forbidden = ("requests", "openai", "google.generativeai", "genai", "gemini", "modal", "runpod")
    for fn in (_material_delivery_event_count, _case_b_fast_path_conflict, _case_b_condition4_diagnostics):
        source = inspect.getsource(fn).lower()
        for needle in forbidden:
            assert needle not in source, f"{needle!r} unexpectedly referenced in {fn.__name__}"


def test_31_no_zone_usability_v2_double_counting():
    """D-180 deliberately does NOT consume D-167's V2 severity vocabulary
    (flag-gated; this gate runs unconditionally) -- guards against ever
    double-counting raw event count + V2 severity as independent votes."""
    for fn in (_material_delivery_event_count, _case_b_fast_path_conflict, _case_b_condition4_diagnostics):
        source = inspect.getsource(fn)
        assert "zone_usability_v2" not in source
        assert "ZoneUsabilityV2" not in source


def test_32_no_new_feature_flag_for_condition4_refinement():
    """No new env-var-gated flag guards `_material_delivery_event_count`/
    the 4b check -- it is an unconditional correctness refinement, exactly
    as D-180 mandates ('prefer NO new feature flag')."""
    for fn in (_material_delivery_event_count, _case_b_fast_path_conflict):
        source = inspect.getsource(fn)
        assert "os.environ" not in source and "getenv" not in source


def test_33_pipeline_wiring_carries_new_condition4_diagnostics_keys():
    """Item 34-equivalent: the new D-180 diagnostics keys are threaded
    end-to-end through `build_flow_b_draft`'s `take_judge_groups` row,
    alongside D-122/D-123's own pre-existing keys, without changing the
    pre-existing winner/membership outcome (same fixture D-122/D-123's own
    pipeline-wiring tests used)."""
    from cutsell_worker.contracts import MediaSignals
    weak = CandidateTake(
        clip_id="weak", source_asset_id="src", source_order=0, start=1.0, end=3.0,
        text="this serum changed my skin",
        signals=MediaSignals(source_asset_id="src", start=1.0, end=3.0, audio_quality=0.3, eye_contact=0.2),
    )
    strong = CandidateTake(
        clip_id="strong", source_asset_id="src", source_order=0, start=4.0, end=6.0,
        text="this serum changed my skin",
        signals=MediaSignals(source_asset_id="src", start=4.0, end=6.0, audio_quality=0.95, eye_contact=0.95),
    )
    request = ProcessingRequest(project_id="project-1", user_id="user-1", sources=())
    labels = (
        SemanticLabel(weak.clip_id, SemanticRole.PROOF, 0.9),
        SemanticLabel(strong.clip_id, SemanticRole.PROOF, 0.9),
    )
    result = build_flow_b_draft(request, (weak, strong), labels)
    assert result.state.value == "draft_ready"
    assert [clip.clip_id for clip in result.draft.selected] == [strong.clip_id]

    groups = (result.draft.diagnostics or {}).get("take_judge_groups") or []
    assert len(groups) == 1
    row = groups[0]
    for key in (
        "case_b_count_difference_present", "case_b_materiality_evidence_available",
        "case_b_materiality_state", "case_b_materiality_source",
        "case_b_condition4_actionable", "case_b_condition4_reason",
    ):
        assert key in row
    assert row["case_b_materiality_source"] == "d097_would_be_counted"
    # No conflict evidence exists for this fixture -- condition 4 never
    # even reaches a count-difference evaluation.
    assert row["case_b_condition4_reason"] == "no_semantic_fast_path_candidate_or_no_evidence"


def test_34_d123_conditions_1_2_3_unchanged_regression():
    """Regression confirmation: D-123's OTHER three conditions (evidence
    exists; local_selected != preferred; alt is meaning-sufficient) are
    untouched by D-180 -- only condition 4 gained the materiality
    sub-check. Mirrors `test_cutsell_d123_case_b_fast_path_gate.py`'s own
    Section B assertions verbatim, run again here as a same-file guard."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [_event("body_reset_candidate", 1.0, 1.5, confidence=0.9)])
    evidence = _evidence_map((a, b), context)
    # Condition 1 (no evidence supplied).
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, None) is None
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, {}) is None
    # Condition 2 (already agrees).
    assert _case_b_fast_path_conflict("A", "A", {"A", "B"}, evidence) is None
    # Condition 3 (alt meaning-insufficient).
    assert _case_b_fast_path_conflict("A", "B", {"A"}, evidence) is None
