"""D-123 -- BestTake CASE B performance-aware fast-path gate.

BOUNDED BEHAVIOR CHANGE (docs/CUTSELL_DECISIONS.md D-123, post D-121/D-122;
docs/CUTSELL_BESTTAKE_CASE_B_FORENSIC_D121.md). This suite proves the ONE
authorized change: D-122's CASE B evidence may prevent `_semantic_best_
take`'s `single_semantic_winner` fast path from returning BEFORE
performance is consulted, when ALL FOUR core-rule conditions hold. It
never:

- introduces a new CASE B score, weight, or threshold (`_case_b_fast_path_
  conflict` compares only D-122's already-computed `delivery_event_count`);
- picks a winner directly from CASE B evidence (a bypass falls through to
  the SAME general ladder `_semantic_best_take` already had -- the ladder,
  not CASE B, decides);
- changes `take_judge.rank_takes`/`score_take`, `MediaSignals`, D-097
  cleanliness evidence, Boundary, or render-plan behavior;
- overrides the deterministic safety veto (`_single_winner_safety_veto`
  still gates first, unconditionally).

Also proves the new observability fields (`winner_path_before/after`,
`semantic_fast_path_bypassed`, `bypass_reason`, `case_b_conflict_present`,
`case_b_conflict_basis`, `meaning_sufficient_candidates`, `final_winner`)
are wired into `take_judge_groups` diagnostics, and that
`deterministic_best_take_authority.py`'s DETERMINISTIC_OVERRIDE annotation
additively upgrades `winner_path_after`/`final_winner` too.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from cutsell_worker.case_b_performance_evidence import build_case_b_performance_evidence
from cutsell_worker.contracts import (
    CandidateTake, DraftClip, EditStrategy, MediaSignals, ProcessingRequest,
    RankedTake, SCHEMA_VERSION as CONTRACTS_SCHEMA_VERSION, SemanticLabel, SemanticRole, Word,
)
from cutsell_worker.deterministic_best_take_authority import apply_deterministic_best_take_authority
from cutsell_worker.pipeline import (
    _WINNER_PATH_DELIVERYSCORE_PATH,
    _WINNER_PATH_SEMANTIC_FAST_PATH,
    _case_b_fast_path_conflict,
    _meaning_sufficient_member_ids,
    _semantic_best_take,
    _single_semantic_winner_candidate,
    _winner_path_from_reason,
    build_flow_b_draft,
)
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.take_judge import rank_takes, score_take
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


# === Section A: _meaning_sufficient_member_ids ===============================

def test_meaning_sufficient_excludes_nothing_by_default():
    a = _take("A", 0.0, 4.0, "one two three four")
    b = _take("B", 4.0, 8.0, "five six seven eight")
    assert _meaning_sufficient_member_ids((a, b), None) == {"A", "B"}


def test_meaning_sufficient_excludes_delete_recommended():
    a = _take("A", 0.0, 4.0, "one two three four")
    b = _take("B", 4.0, 8.0, "five six seven eight")
    assert _meaning_sufficient_member_ids((a, b), {"A": True}) == {"B"}


def test_meaning_sufficient_excludes_explicitly_incomplete():
    a = _take("A", 0.0, 4.0, "one two three four", complete_idea=False)
    b = _take("B", 4.0, 8.0, "five six seven eight", complete_idea=True)
    assert _meaning_sufficient_member_ids((a, b), None) == {"B"}


def test_meaning_sufficient_never_excludes_unset_completeness():
    # WHEN UNCERTAIN, KEEP: complete_idea=None (unset/unknown) never excludes.
    a = _take("A", 0.0, 4.0, "one two three four", complete_idea=None)
    b = _take("B", 4.0, 8.0, "five six seven eight", complete_idea=True)
    assert _meaning_sufficient_member_ids((a, b), None) == {"A", "B"}


# === Section B: _case_b_fast_path_conflict (the CORE RULE) ===================

def test_conflict_fires_when_all_four_conditions_hold():
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [
        _event("body_reset_candidate", 1.0, 1.5),
        _event("hand_motion_reset_candidate", 2.0, 2.3),
    ])  # both land in A's DELIVERY span; B has none
    evidence = _evidence_map((a, b), context)
    basis = _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence)
    assert basis is not None
    assert basis["semantic_fast_path_candidate"] == "A"
    assert basis["deliveryscore_top_candidate"] == "B"
    assert basis["semantic_fast_path_candidate_delivery_event_count"] == 2
    assert basis["deliveryscore_top_candidate_delivery_event_count"] == 0


def test_no_conflict_when_no_case_b_evidence_supplied():
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, None) is None
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, {}) is None


def test_no_conflict_when_semantic_winner_already_matches_deliveryscore_top():
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [_event("body_reset_candidate", 1.0, 1.5)])
    evidence = _evidence_map((a, b), context)
    # local_selected_clip_id == preferred_id ("A") -- nothing to bypass for.
    assert _case_b_fast_path_conflict("A", "A", {"A", "B"}, evidence) is None


def test_no_conflict_when_deliveryscore_top_is_meaning_insufficient():
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [_event("body_reset_candidate", 1.0, 1.5)])
    evidence = _evidence_map((a, b), context)
    # "B" is NOT in the meaning-sufficient set -- e.g. it is incomplete.
    assert _case_b_fast_path_conflict("A", "B", {"A"}, evidence) is None


def test_no_conflict_when_case_b_evidence_tied():
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [
        _event("body_reset_candidate", 1.0, 1.5),
        _event("body_reset_candidate", 5.0, 5.5),
    ])  # one event each -- a genuine tie, never a guessed cutoff
    evidence = _evidence_map((a, b), context)
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_no_conflict_when_evidence_favors_the_semantic_winner_instead():
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    # B (the DeliveryScorer top) has MORE events, not fewer -- asymmetry
    # favors A, the semantic winner: never a basis to bypass A's own fast path.
    context = _context("src", [
        _event("body_reset_candidate", 5.0, 5.5),
        _event("hand_motion_reset_candidate", 6.0, 6.3),
    ])
    evidence = _evidence_map((a, b), context)
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_no_conflict_when_evidence_missing_for_either_side():
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    context = _context("src", [_event("body_reset_candidate", 1.0, 1.5)])
    evidence = _evidence_map((a,), context)  # "B" never computed
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_no_conflict_when_difference_is_entry_only_events():
    # ENTRY-only events never enter case_b evidence at all (D-115/D-122
    # DELIVERY-zone filter) -- both counts are naturally 0 vs 0.
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 2.0, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [_event("camera_disengagement_candidate", 0.1, 0.5)])  # ENTRY-only for A
    evidence = _evidence_map((a, b), context)
    assert evidence["A"].delivery_event_count == 0
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


def test_no_conflict_when_difference_is_exit_only_events():
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 6.5))
    context = _context("src", [_event("hand_motion_reset_candidate", 6.7, 7.2)])  # EXIT-only for B
    evidence = _evidence_map((a, b), context)
    assert evidence["B"].delivery_event_count == 0
    assert _case_b_fast_path_conflict("A", "B", {"A", "B"}, evidence) is None


# === Section C: _semantic_best_take gate integration =========================

def test_pimples_shaped_fixture_bypasses_and_deliveryscore_path_wins():
    """Mirrors D-118's exact real shape (docs/CUTSELL_BESTTAKE_CASE_B_
    FORENSIC_D121.md Section 5): the labelled "winner" (confidence 0.95)
    carries MORE local-failure/reset evidence than the labelled
    "alternate" DeliveryScorer prefers. D-123 gates the early exit; the
    EXISTING ladder (never CASE B) then picks the winner."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    r = ranked(("A", 0.40), ("B", 0.90))  # DeliveryScorer strongly prefers B
    context = _context("src", [
        _event("body_reset_candidate", 1.0, 1.5, confidence=0.9),
        _event("hand_motion_reset_candidate", 2.0, 2.3, confidence=0.9),
    ])
    decisions = {"A": ("winner", 0.95), "B": ("alternate", 0.8)}
    evidence = _evidence_map((a, b), context)

    before = _semantic_best_take((a, b), decisions, "B", r)
    assert before == ("A", "A", "single_semantic_winner")

    after = _semantic_best_take((a, b), decisions, "B", r, case_b_evidence_by_id=evidence)
    selected, preferred, reason = after
    assert reason != "single_semantic_winner"
    assert reason == "delivery_tie_break_among_survivors"  # the SAME existing ladder step
    assert selected == "B"  # the EXISTING DeliveryScorer path decided, never CASE B directly

    before_path, before_consulted = _winner_path_from_reason(before[2])
    after_path, after_consulted = _winner_path_from_reason(reason)
    assert before_path == _WINNER_PATH_SEMANTIC_FAST_PATH and before_consulted is False
    assert after_path == _WINNER_PATH_DELIVERYSCORE_PATH and after_consulted is True


def test_no_new_case_b_numeric_score_field_exists():
    from cutsell_worker.case_b_performance_evidence import CaseBPerformanceEvidence
    forbidden = {"score", "case_b_score", "weight", "threshold"}
    fields = {f.lower() for f in CaseBPerformanceEvidence.__dataclass_fields__}
    assert not (fields & forbidden), f"unexpected numeric-score-like field(s): {fields & forbidden}"


def test_deterministic_safety_veto_still_gates_before_case_b_is_even_consulted():
    """A vetoed label (D-101/D-103) must fall through to the same ladder
    regardless of whether CASE B evidence exists or what it says -- the
    veto is never weakened or bypassed by D-123."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    r = ranked(("A", 0.40), ("B", 0.90))
    context = _context("src", [_event("body_reset_candidate", 5.0, 5.5)])  # favors A, not B
    decisions = {"A": ("winner", 0.95), "B": ("alternate", 0.8)}
    evidence = _evidence_map((a, b), context)

    # A itself carries a D-081 semantic-delete-recommended flag -> vetoed.
    without_case_b = _semantic_best_take(
        (a, b), decisions, "B", r, semantic_delete_recommended={"A": True},
    )
    with_case_b = _semantic_best_take(
        (a, b), decisions, "B", r, semantic_delete_recommended={"A": True},
        case_b_evidence_by_id=evidence,
    )
    assert without_case_b == with_case_b
    assert without_case_b[2] != "single_semantic_winner"


# === Section D: no-behavior-change / D-097 double-counting proofs ============

def test_rank_takes_and_score_take_unaffected_by_the_gate():
    signals = MediaSignals(source_asset_id="src", start=0.0, end=6.0, visual_fumble=0.2)
    takes = (
        _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0), signals=signals),
        _take("c2", 7.0, 12.0, "four five six", words=_words("four five six", 7.5, 11.5)),
    )
    context = _context("src", [_event("body_reset_candidate", 4.0, 4.5)])
    before_scores = [score_take(t) for t in takes]
    before_ranked = rank_takes(takes)

    evidence = _evidence_map(takes, context)
    _semantic_best_take(
        takes, {"c1": ("winner", 0.95), "c2": ("alternate", 0.8)}, "c2",
        ranked(("c1", 0.4), ("c2", 0.9)), case_b_evidence_by_id=evidence,
    )

    after_scores = [score_take(t) for t in takes]
    after_ranked = rank_takes(takes)
    assert before_scores == after_scores
    assert before_ranked == after_ranked


def test_mediasignals_object_identity_unchanged_by_the_gate():
    signals = MediaSignals(source_asset_id="src", start=0.0, end=4.0, visual_fumble=0.2)
    a = _take("A", 0.0, 4.0, "one two three four", signals=signals,
              words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    context = _context("src", [_event("body_reset_candidate", 1.0, 1.5)])
    evidence = _evidence_map((a, b), context)
    _semantic_best_take(
        (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.8)}, "B",
        ranked(("A", 0.4), ("B", 0.9)), case_b_evidence_by_id=evidence,
    )
    assert a.signals is signals  # never mutated


def test_no_provider_or_network_reference_in_the_new_gate_helpers():
    forbidden = ("requests", "openai", "google.generativeai", "genai", "gemini", "modal", "runpod")
    for fn in (_meaning_sufficient_member_ids, _case_b_fast_path_conflict):
        source = inspect.getsource(fn).lower()
        for needle in forbidden:
            assert needle not in source, f"{needle!r} unexpectedly referenced in {fn.__name__}"


def test_case_b_gate_diagnostics_keys_never_referenced_by_render_plan_modules():
    repo_root = Path(__file__).resolve().parents[1] / "cutsell_worker"
    forbidden_keys = (
        "winner_path_before", "winner_path_after", "semantic_fast_path_bypassed",
        "bypass_reason", "case_b_conflict_present", "case_b_conflict_basis",
        "meaning_sufficient_candidates",
    )
    for name in ("render_plan.py", "canonical_edit_plan.py"):
        text = (repo_root / name).read_text()
        for key in forbidden_keys:
            assert key not in text, f"{key!r} unexpectedly referenced in {name} -- D-123 must stay bounded to BestTake"


# === Section E: pipeline-level wiring =========================================

def test_flow_b_pipeline_carries_the_new_d123_diagnostics_keys():
    # Same weak/strong fixture D-122's own pipeline-wiring test used --
    # proves the new keys are threaded end-to-end without changing the
    # pre-existing winner/membership outcome.
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
    assert [clip.clip_id for clip in result.draft.alternates] == [weak.clip_id]

    groups = (result.draft.diagnostics or {}).get("take_judge_groups") or []
    assert len(groups) == 1
    row = groups[0]
    # D-122 keys still present (regression preservation).
    for key in (
        "winner_path", "performance_consulted_before_winner",
        "deliveryscore_top_candidate", "semantic_fast_path_candidate", "case_b_evidence",
    ):
        assert key in row
    # D-123 keys present and internally consistent.
    for key in (
        "winner_path_before", "winner_path_after", "semantic_fast_path_bypassed",
        "bypass_reason", "case_b_conflict_present", "case_b_conflict_basis",
        "meaning_sufficient_candidates", "final_winner",
    ):
        assert key in row
    assert row["final_winner"] == row["selected_clip_id"]
    # No conflict evidence exists for this fixture (D-122's own weak/strong
    # pair carries no whole-video multimodal events) -- no bypass expected.
    assert row["case_b_conflict_present"] is False
    assert row["semantic_fast_path_bypassed"] is False
    assert row["bypass_reason"] is None


def test_unrelated_family_unchanged_when_another_family_conflicts():
    # NEGATIVE CONTROL: a second, wholly unrelated family in the SAME
    # pipeline call must never be affected by a bypass decided in another
    # family -- the gate is per-family, never global.
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
    singleton = CandidateTake(
        clip_id="solo", source_asset_id="src", source_order=1, start=10.0, end=13.0,
        text="completely unrelated single idea about pricing",
    )
    request = ProcessingRequest(project_id="project-1", user_id="user-1", sources=())
    labels = (
        SemanticLabel(weak.clip_id, SemanticRole.PROOF, 0.9),
        SemanticLabel(strong.clip_id, SemanticRole.PROOF, 0.9),
        SemanticLabel(singleton.clip_id, SemanticRole.CTA, 0.9),
    )
    result = build_flow_b_draft(request, (weak, strong, singleton), labels)
    selected_ids = {clip.clip_id for clip in result.draft.selected}
    assert "strong" in selected_ids
    assert "solo" in selected_ids
    assert "weak" not in selected_ids


# === Section F: deterministic_best_take_authority additive upgrade ==========

def test_deterministic_override_also_upgrades_winner_path_after_and_final_winner():
    winner = DraftClip(
        clip_id="winner", source_asset_id="src", source_order=0, start=0.0, end=4.0,
        text="one two three four", caption_text="one two three four",
        semantic_role=SemanticRole.STORY, selected=False,
    )
    loser = DraftClip(
        clip_id="loser", source_asset_id="src", source_order=1, start=4.0, end=8.0,
        text="one two three four five", caption_text="one two three four five",
        semantic_role=SemanticRole.STORY, selected=True,
    )
    from cutsell_worker.contracts import DraftTimeline
    take_judge_groups = [{
        "group_id": "tg_1",
        "selected_clip_id": "loser",
        "winner_path": "OTHER_EXISTING_PATH",
        "performance_consulted_before_winner": False,
        "winner_path_before": "OTHER_EXISTING_PATH",
        "winner_path_after": "OTHER_EXISTING_PATH",
        "final_winner": "loser",
        "ranked": [
            {"clip_id": "winner", "score": 0.95, "reason": "watch_listen_baseline"},
            {"clip_id": "loser", "score": 0.10, "reason": "watch_listen_baseline"},
        ],
    }]
    draft = DraftTimeline(
        schema_version=CONTRACTS_SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=(loser,), alternates=(), discarded=(winner,),
        diagnostics={"take_judge_groups": take_judge_groups},
    )
    result_draft = apply_deterministic_best_take_authority(draft, swap_enabled=False)
    row = result_draft.diagnostics["take_judge_groups"][0]
    assert row["winner_path"] == "DETERMINISTIC_OVERRIDE"
    assert row["winner_path_after"] == "DETERMINISTIC_OVERRIDE"
    assert row["final_winner"] == "winner"
    # winner_path_before is the semantic-only counterfactual -- untouched.
    assert row["winner_path_before"] == "OTHER_EXISTING_PATH"
    assert {clip.clip_id for clip in result_draft.selected} == {"winner"}
