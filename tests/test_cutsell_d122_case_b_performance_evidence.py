"""D-122 -- BestTake CASE B performance-evidence projection.

ADVISORY / DIAGNOSTICS ONLY (docs/CUTSELL_DECISIONS.md D-122;
docs/CUTSELL_BESTTAKE_CASE_B_FORENSIC_D121.md). This suite proves:

- the new `case_b_performance_evidence.py` module is a pure, additive
  re-projection of D-115's already-tested DELIVERY-zone classification --
  ENTRY/EXIT-only events are excluded, a straddling DELIVERY-overlap event
  is included, factual aggregates (counts/durations/density) are correct,
  and no default-only signal is fabricated when a candidate has no words;
- the D-121-confirmed MediaSignals/D-097 double-counting risk is now
  INSPECTABLE per event (provenance mapping, D-097-vs-D-115 window
  disagreement), computed by calling D-097's own real private helpers
  directly -- never a re-typed copy of its margin/confidence constants;
- `pipeline.py`'s two new diagnostics-only helpers
  (`_winner_path_from_reason`, `_single_semantic_winner_candidate`)
  correctly classify the already-existing `semantic_best_take_reason`
  vocabulary without changing it;
- `_semantic_best_take`, `take_judge.rank_takes`/`score_take`, and
  `apply_post_freeze_boundary_pass` are BYTE-IDENTICAL before and after
  CASE B evidence is computed -- this module changes no ranking, no
  winner, no membership, no Boundary result;
- the pimples-shaped CASE 1-6 fixtures the task specified.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cutsell_worker.audio_silence import AUDIO_SILENCE_EVENT_KIND
from cutsell_worker.boundary_engine_pass import apply_post_freeze_boundary_pass
from cutsell_worker.case_b_performance_evidence import (
    MEDIASIGNALS_PROVENANCE,
    MEDIASIGNALS_PROVENANCE_PRODUCER,
    SCHEMA_VERSION,
    build_case_b_performance_evidence,
    case_b_performance_evidence_diagnostics,
)
from cutsell_worker.contracts import (
    CandidateTake, DraftClip, DraftTimeline, EditStrategy, JobState,
    MediaSignals, ProcessingRequest, ProcessingResult, RankedTake,
    SCHEMA_VERSION as CONTRACTS_SCHEMA_VERSION, SemanticLabel, SemanticRole, Word,
)
from cutsell_worker.pipeline import (
    _WINNER_PATH_DELIVERYSCORE_PATH,
    _WINNER_PATH_OTHER_EXISTING_PATH,
    _WINNER_PATH_SEMANTIC_FAST_PATH,
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


# === Section A: case_b_performance_evidence.py module tests =================

# --- 1: DELIVERY event projected ---------------------------------------------

def test_delivery_event_is_projected_with_full_fields():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event("body_reset_candidate", 3.0, 3.5, confidence=0.9)])
    evidence = build_case_b_performance_evidence(take, context)
    assert evidence.delivery_available is True
    assert evidence.delivery_start == pytest.approx(2.0)
    assert evidence.delivery_end == pytest.approx(5.0)
    assert evidence.delivery_event_count == 1
    row = evidence.delivery_events[0]
    assert row.kind == "body_reset_candidate"
    assert row.start == pytest.approx(3.0) and row.end == pytest.approx(3.5)
    assert row.duration == pytest.approx(0.5)
    assert row.confidence == pytest.approx(0.9)
    assert row.evidence_source == "local_performance"
    assert row.straddle is False


# --- 2/3: ENTRY-only / EXIT-only events excluded -----------------------------

def test_entry_only_event_excluded_from_case_b():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event("camera_disengagement_candidate", 0.1, 0.5)])
    evidence = build_case_b_performance_evidence(take, context)
    assert evidence.delivery_event_count == 0
    assert evidence.delivery_events == ()


def test_exit_only_event_excluded_from_case_b():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event("hand_motion_reset_candidate", 5.4, 5.9)])
    evidence = build_case_b_performance_evidence(take, context)
    assert evidence.delivery_event_count == 0
    assert evidence.delivery_events == ()


# --- 4: cross-boundary DELIVERY-overlap included -----------------------------

def test_cross_boundary_straddle_event_included_as_delivery():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    # Starts before DELIVERY, ends inside it -- D-115 classifies any overlap
    # as DELIVERY; CASE B must include it, flagged as a straddle.
    context = _context("src", [_event("facial_expression_shift_candidate", 1.5, 2.5)])
    evidence = build_case_b_performance_evidence(take, context)
    assert evidence.delivery_event_count == 1
    row = evidence.delivery_events[0]
    assert row.straddle is True


# --- 5/6: count/duration by kind ---------------------------------------------

def test_count_and_duration_by_kind_correct_across_multiple_events():
    take = _take("c1", 0.0, 10.0, "one two three four", words=_words("one two three four", 1.0, 9.0))
    context = _context("src", [
        _event("body_reset_candidate", 2.0, 2.5),
        _event("body_reset_candidate", 4.0, 4.8),
        _event("hand_motion_reset_candidate", 6.0, 6.2),
    ])
    evidence = build_case_b_performance_evidence(take, context)
    assert evidence.delivery_event_count == 3
    assert evidence.count_by_kind == {"body_reset_candidate": 2, "hand_motion_reset_candidate": 1}
    assert evidence.duration_by_kind["body_reset_candidate"] == pytest.approx(0.5 + 0.8)
    assert evidence.duration_by_kind["hand_motion_reset_candidate"] == pytest.approx(0.2)
    assert evidence.delivery_event_duration_total == pytest.approx(0.5 + 0.8 + 0.2)


# --- 7: delivery duration correct --------------------------------------------

def test_delivery_span_duration_correct():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    evidence = build_case_b_performance_evidence(take, None)
    assert evidence.delivery_span_duration == pytest.approx(3.0)


# --- 8: event density mathematically correct ---------------------------------

def test_event_density_is_duration_over_delivery_span():
    take = _take("c1", 0.0, 10.0, "one two three four", words=_words("one two three four", 0.0, 10.0))
    context = _context("src", [_event("body_reset_candidate", 2.0, 3.0)])  # 1.0s duration
    evidence = build_case_b_performance_evidence(take, context)
    assert evidence.delivery_span_duration == pytest.approx(10.0)
    assert evidence.event_density == pytest.approx(1.0 / 10.0)


def test_event_density_none_when_no_delivery_span():
    take = _take("c1", 0.0, 6.0, "unused", words=())
    evidence = build_case_b_performance_evidence(take, None)
    assert evidence.event_density is None


# --- 9: no words -> evidence unavailable safely ------------------------------

def test_no_words_delivery_unavailable_and_no_crash():
    take = _take("c1", 0.0, 6.0, "unused", words=())
    context = _context("src", [_event("body_reset_candidate", 3.0, 3.5)])
    evidence = build_case_b_performance_evidence(take, context)
    assert evidence.delivery_available is False
    assert evidence.delivery_start is None and evidence.delivery_end is None
    assert evidence.delivery_span_duration is None
    # No delivery span -> D-115 classifies the event UNKNOWN, never DELIVERY:
    # CASE B must not fabricate a DELIVERY event without a real span.
    assert evidence.delivery_event_count == 0
    diag = case_b_performance_evidence_diagnostics(evidence)
    assert diag["delivery_available"] is False
    assert diag["schema_version"] == SCHEMA_VERSION


# --- 10: MediaSignals provenance mapping correct -----------------------------

def test_mediasignals_provenance_mapping_matches_local_performance_buckets():
    # Read directly off local_performance.py's own body_n/face_n/disengage_n
    # bucket assignment -- never guessed from field names.
    assert MEDIASIGNALS_PROVENANCE["body_reset_candidate"] == ("visual_fumble", "gesture_naturalness")
    assert MEDIASIGNALS_PROVENANCE["hand_motion_reset_candidate"] == ("visual_fumble", "gesture_naturalness")
    assert MEDIASIGNALS_PROVENANCE["facial_expression_shift_candidate"] == ("visual_fumble", "expression_naturalness")
    assert MEDIASIGNALS_PROVENANCE["camera_disengagement_candidate"] == ("distraction_risk",)
    assert MEDIASIGNALS_PROVENANCE_PRODUCER == "local_performance.apply_local_performance_to_takes"


def test_delivery_event_row_carries_its_mediasignal_fields():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event("camera_disengagement_candidate", 3.0, 3.3)])
    evidence = build_case_b_performance_evidence(take, context)
    assert evidence.delivery_events[0].mediasignal_fields == ("distraction_risk",)


# --- 11: D-097 cleanliness overlap mapping correct ---------------------------

def test_d097_would_be_counted_true_for_strong_reset_inside_interior_window():
    # take.start=0, take.end=10 -> D-097 interior window is [0.35, 9.65].
    # A body_reset_candidate at confidence>=0.88 (the _RESET_KINDS floor)
    # sitting well inside that window must show d097_would_be_counted=True.
    take = _take("c1", 0.0, 10.0, "one two three four", words=_words("one two three four", 0.5, 9.5))
    context = _context("src", [_event("body_reset_candidate", 4.0, 4.5, confidence=0.90)])
    evidence = build_case_b_performance_evidence(take, context)
    row = evidence.delivery_events[0]
    assert row.d097_geometrically_inside is True
    assert row.d097_meets_confidence_floor is True
    assert row.d097_would_be_counted is True


def test_d097_would_be_counted_false_below_confidence_floor():
    take = _take("c1", 0.0, 10.0, "one two three four", words=_words("one two three four", 0.5, 9.5))
    # Same geometry as above, but confidence below D-097's own 0.88 reset floor.
    context = _context("src", [_event("body_reset_candidate", 4.0, 4.5, confidence=0.60)])
    evidence = build_case_b_performance_evidence(take, context)
    row = evidence.delivery_events[0]
    assert row.d097_geometrically_inside is True
    assert row.d097_meets_confidence_floor is False
    assert row.d097_would_be_counted is False


# --- 12 / CASE 5 / CASE 6: D-097 vs D-115 window disagreement ----------------

def test_case5_delivery_event_outside_d097_interior_margin():
    # take.start=0.0 -> D-097's interior window starts at 0.35s. Put the
    # event's own words such that DELIVERY starts at 0.1s (word envelope),
    # and the event sits at 0.15-0.25s: inside D-115 DELIVERY, but BEFORE
    # D-097's own 0.35s margin -- geometrically outside D-097's window.
    take = _take("c1", 0.0, 10.0, "one two three", words=_words("one two three", 0.1, 9.9))
    context = _context("src", [_event("facial_expression_shift_candidate", 0.15, 0.25, confidence=0.9)])
    evidence = build_case_b_performance_evidence(take, context)
    row = evidence.delivery_events[0]
    assert row.d097_geometrically_inside is False
    assert row.d097_would_be_counted is False


def test_case6_delivery_event_counted_by_both_d115_and_d097():
    take = _take("c1", 0.0, 10.0, "one two three four", words=_words("one two three four", 0.5, 9.5))
    context = _context("src", [_event("hand_motion_reset_candidate", 5.0, 5.3, confidence=0.95)])
    evidence = build_case_b_performance_evidence(take, context)
    row = evidence.delivery_events[0]
    # D-115: this event is classified DELIVERY (it is in `evidence.delivery_
    # events` at all -- CASE B never includes ENTRY/EXIT-only events).
    # D-097: well inside the interior window at a confidence above the reset
    # floor -- both consumers see it, and CASE B makes both visible.
    assert row.d097_geometrically_inside is True
    assert row.d097_meets_confidence_floor is True
    assert row.d097_would_be_counted is True


# --- 13: no default-only fake evidence ---------------------------------------

def test_audio_silence_excluded_from_case_b_unlike_general_d115_evidence():
    # D-115's own general evidence includes audio_silence_interval
    # (positioned_performance_evidence.POSITIONED_EVENT_KINDS); CASE B is
    # scoped to only the four D-114 local-performance kinds this task named
    # -- audio_silence_interval must never appear in delivery_events.
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event(AUDIO_SILENCE_EVENT_KIND, 3.0, 3.5, confidence=1.0)])
    evidence = build_case_b_performance_evidence(take, context)
    assert evidence.delivery_event_count == 0


def test_unknown_event_kind_never_gains_fake_case_b_evidence():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event("framing_quality_candidate", 3.0, 3.5)])
    evidence = build_case_b_performance_evidence(take, context)
    assert evidence.delivery_event_count == 0


def test_events_from_a_different_source_excluded():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0), source="src_a")
    context = _context("src_b", [_event("body_reset_candidate", 3.0, 3.5, source="src_b")])
    evidence = build_case_b_performance_evidence(take, context)
    assert evidence.delivery_event_count == 0


def test_diagnostics_row_is_json_safe_and_bounded():
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event("body_reset_candidate", 3.0, 3.5, confidence=0.9)])
    evidence = build_case_b_performance_evidence(take, context)
    diag = case_b_performance_evidence_diagnostics(evidence)
    assert diag["candidate_id"] == "c1"
    assert diag["source_asset_id"] == "src"
    assert isinstance(diag["delivery_events"], list) and len(diag["delivery_events"]) == 1
    event_row = diag["delivery_events"][0]
    assert event_row["kind"] == "body_reset_candidate"
    assert event_row["mediasignal_fields"] == ["visual_fumble", "gesture_naturalness"]
    # d097_would_be_counted is always exactly the AND of the other two flags
    # -- never a separately-invented threshold.
    assert event_row["d097_would_be_counted"] == (event_row["d097_geometrically_inside"] and event_row["d097_meets_confidence_floor"])
    assert event_row["d097_would_be_counted"] is True  # confidence 0.9 >= the reused 0.88 reset floor, well inside the window


# === Section B: pipeline.py helper unit tests ================================

def test_winner_path_semantic_fast_path_never_consults_performance():
    path, consulted = _winner_path_from_reason("single_semantic_winner")
    assert path == _WINNER_PATH_SEMANTIC_FAST_PATH
    assert consulted is False


@pytest.mark.parametrize("reason", ["delivery_tie_break_among_survivors", "local_fallback"])
def test_winner_path_deliveryscore_path_consults_performance(reason):
    path, consulted = _winner_path_from_reason(reason)
    assert path == _WINNER_PATH_DELIVERYSCORE_PATH
    assert consulted is True


@pytest.mark.parametrize("reason", [
    "critical_coverage_dominance", "unresolved_unique_fact_asymmetry",
    "unresolved_contradiction", "single_member_no_contest",
    "single_bts_unusable", "no_usable_realization",
])
def test_winner_path_other_existing_path_for_meaning_driven_reasons(reason):
    path, consulted = _winner_path_from_reason(reason)
    assert path == _WINNER_PATH_OTHER_EXISTING_PATH
    assert consulted is False


def test_single_semantic_winner_candidate_decisive():
    a = _take("A", 0.0, 4.0, "one two three four")
    b = _take("B", 4.0, 8.0, "five six seven eight")
    winner = _single_semantic_winner_candidate((a, b), {"A": ("alternate", 0.8), "B": ("winner", 0.95)})
    assert winner == "B"


def test_single_semantic_winner_candidate_non_decisive_returns_none():
    a = _take("A", 0.0, 4.0, "one two three four")
    b = _take("B", 4.0, 8.0, "five six seven eight")
    # Two "winner" labels -- not decisive.
    assert _single_semantic_winner_candidate((a, b), {"A": ("winner", 0.9), "B": ("winner", 0.9)}) is None
    # Below the confidence floor -- not decisive.
    assert _single_semantic_winner_candidate((a, b), {"A": ("winner", 0.5), "B": ("alternate", 0.5)}) is None


# === Section C: no-behavior-change proofs ====================================

def test_semantic_best_take_decision_unaffected_by_case_b_computation():
    a = _take("A", 0.0, 4.0, "one two three four", complete_idea=True)
    b = _take("B", 4.0, 8.0, "five six seven eight nine", complete_idea=True)
    r = ranked(("A", 0.55), ("B", 0.70))
    context = _context("src", [_event("body_reset_candidate", 1.0, 1.5)])

    before = _semantic_best_take((a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.8)}, "B", r)
    # CASE B evidence computed for side effects only, exactly like D-115's
    # own module -- never read by _semantic_best_take.
    build_case_b_performance_evidence(a, context)
    build_case_b_performance_evidence(b, context)
    after = _semantic_best_take((a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.8)}, "B", r)

    assert before == after
    assert before[0] == "A"  # single_semantic_winner fast path, unchanged


def test_rank_takes_and_score_take_unaffected_by_case_b_computation():
    signals = MediaSignals(source_asset_id="src", start=0.0, end=6.0, visual_fumble=0.2)
    takes = (
        _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0), signals=signals),
        _take("c2", 7.0, 12.0, "four five six", words=_words("four five six", 7.5, 11.5)),
    )
    context = _context("src", [_event("body_reset_candidate", 4.0, 4.5)])

    before_scores = [score_take(take) for take in takes]
    before_ranked = rank_takes(takes)

    for take in takes:
        build_case_b_performance_evidence(take, context)

    after_scores = [score_take(take) for take in takes]
    after_ranked = rank_takes(takes)

    assert before_scores == after_scores
    assert before_ranked == after_ranked


def test_boundary_pass_unaffected_by_case_b_diagnostics_key():
    words = _words("one two three four", 10.5, 15.5)
    clip = DraftClip(
        clip_id="c1", source_asset_id="src", source_order=0, start=10.0, end=16.0,
        text="one two three four", caption_text="one two three four", words=words,
        semantic_role=SemanticRole.STORY, selected=True,
    )
    base_diagnostics = {
        "whole_video_context": {"sources": [{
            "source_asset_id": "src",
            "events": [{"kind": AUDIO_SILENCE_EVENT_KIND, "start": 9.4, "end": 10.9, "confidence": 1.0}],
        }]},
    }
    context = _context("src", [_event(AUDIO_SILENCE_EVENT_KIND, 9.4, 10.9, confidence=1.0)])
    take = _take("c1", 10.0, 16.0, "one two three four", words=words)
    case_b_row = case_b_performance_evidence_diagnostics(build_case_b_performance_evidence(take, context))

    def _draft(diagnostics):
        return DraftTimeline(
            schema_version=CONTRACTS_SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
            selected=(clip,), alternates=(), discarded=(), diagnostics=diagnostics,
        )

    def _result(draft):
        return ProcessingResult(
            schema_version=CONTRACTS_SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY,
            draft=draft, stage_status={},
        )

    result_without = apply_post_freeze_boundary_pass(_result(_draft(dict(base_diagnostics))))
    result_with = apply_post_freeze_boundary_pass(_result(_draft({
        **base_diagnostics,
        "take_judge_groups": [{"group_id": "tg_x", "case_b_evidence": {"c1": case_b_row}}],
    })))
    assert result_without.draft.selected == result_with.draft.selected


def test_case_b_module_never_imports_a_provider_call():
    source = Path(__file__).resolve().parents[1] / "cutsell_worker" / "case_b_performance_evidence.py"
    text = source.read_text()
    forbidden = ("requests", "openai", "google.generativeai", "genai", "gemini", "modal", "runpod")
    lowered = text.lower()
    for needle in forbidden:
        assert needle not in lowered, f"{needle!r} unexpectedly referenced in case_b_performance_evidence.py"


def test_case_b_diagnostics_keys_never_referenced_by_render_plan_modules():
    repo_root = Path(__file__).resolve().parents[1] / "cutsell_worker"
    forbidden_keys = ("case_b_evidence", "winner_path", "performance_consulted_before_winner")
    for name in ("render_plan.py", "canonical_edit_plan.py"):
        text = (repo_root / name).read_text()
        for key in forbidden_keys:
            assert key not in text, f"{key!r} unexpectedly referenced in {name} -- D-122 must stay diagnostics-only"


# === Section D: pipeline-level wiring + pimples-shaped CASE 1-6 fixtures ====

def test_flow_b_pipeline_winner_and_membership_unchanged_with_case_b_diagnostics():
    # Reproduces the existing, already-passing weak/strong retry-family
    # pattern (test_cutsell_clean_worker_m0_m1.py) verbatim, byte-identically,
    # to prove D-122's additive wiring changes neither the winner nor the
    # membership this pipeline already produced before D-122 existed.
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
    # D-122 additive fields present, factual, and consistent with the
    # UNCHANGED existing fields on the same row.
    assert row["selected_clip_id"] == "strong"
    assert row["local_selected_clip_id"] == "strong"
    assert row["deliveryscore_top_candidate"] == "strong"
    assert row["winner_path"] in (_WINNER_PATH_DELIVERYSCORE_PATH, _WINNER_PATH_OTHER_EXISTING_PATH)
    assert "case_b_evidence" in row
    assert set(row["case_b_evidence"].keys()) == {"weak", "strong"}


def test_case1_semantic_winner_bypasses_more_delivery_evidence_no_behavior_change():
    """CASE 1: two meaning-sufficient competitors, semantic winner A,
    DeliveryScorer winner B, A has MORE DELIVERY-overlap reset evidence.
    Expected: diagnostics show the conflict; actual winner remains exactly
    the pre-D-122 winner (A, via the unmodified single_semantic_winner
    fast path)."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    r = ranked(("A", 0.40), ("B", 0.90))  # DeliveryScorer strongly prefers B
    context = _context("src", [
        _event("body_reset_candidate", 1.0, 1.5, confidence=0.9),
        _event("hand_motion_reset_candidate", 2.0, 2.3, confidence=0.9),
    ])  # both fall inside A's DELIVERY span; B has none

    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.8)}, "B", r,
    )
    assert selected == "A"  # UNCHANGED: fast path never reads `r`
    assert reason == "single_semantic_winner"

    winner_path, consulted = _winner_path_from_reason(reason)
    assert winner_path == _WINNER_PATH_SEMANTIC_FAST_PATH
    assert consulted is False

    evidence_a = build_case_b_performance_evidence(a, context)
    evidence_b = build_case_b_performance_evidence(b, context)
    assert evidence_a.delivery_event_count == 2
    assert evidence_b.delivery_event_count == 0
    # The conflict is visible (A won; A also has the worse CASE B evidence)
    # but this test asserts NOTHING changes the winner -- that is D-121's
    # own documented open design question, not resolved here.
    assert _single_semantic_winner_candidate((a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.8)}) == "A"


def test_case2_semantic_winner_also_has_cleaner_case_b_evidence_no_conflict():
    """CASE 2: semantic winner A also has cleaner CASE B evidence.
    Expected: no behavior change (trivially true -- D-122 changes nothing),
    and no conflict between the two views."""
    a = _take("A", 0.0, 4.0, "one two three four", words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 8.0, "five six seven eight", words=_words("five six seven eight", 4.2, 7.8))
    r = ranked(("A", 0.90), ("B", 0.40))
    context = _context("src", [_event("body_reset_candidate", 4.5, 5.0, confidence=0.9)])  # only inside B

    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.8)}, "A", r,
    )
    assert selected == "A"
    assert reason == "single_semantic_winner"
    evidence_a = build_case_b_performance_evidence(a, context)
    evidence_b = build_case_b_performance_evidence(b, context)
    assert evidence_a.delivery_event_count == 0
    assert evidence_b.delivery_event_count == 1


def test_case3_cleaner_performance_but_insufficient_meaning_never_implied_as_winner():
    """CASE 3: one candidate has cleaner CASE B evidence but is an
    EXPLICITLY incomplete attempt. Expected: the existing D-082 meaning-
    sufficiency ladder still excludes it (unaffected by D-122); CASE B
    evidence is visible but never implies it should have won."""
    a = _take("A", 0.0, 4.0, "one two three four", complete_idea=False,
              words=_words("one two three four", 0.2, 3.8))
    b = _take("B", 4.0, 9.0, "five six seven eight nine ten", complete_idea=True,
              words=_words("five six seven eight nine ten", 4.2, 8.8))
    r = ranked(("A", 0.85), ("B", 0.30))  # A "looks" better on delivery
    context = _context("src", [_event("body_reset_candidate", 5.0, 5.5, confidence=0.9)])  # only inside B

    # No decisive semantic winner -> falls through the existing D-082 ladder.
    selected, preferred, reason = _semantic_best_take(
        (a, b), {}, "A", r,
    )
    assert selected == "B"  # meaning-sufficiency (complete_idea) still wins, unchanged by D-122
    evidence_a = build_case_b_performance_evidence(a, context)
    evidence_b = build_case_b_performance_evidence(b, context)
    assert evidence_a.delivery_event_count == 0
    assert evidence_b.delivery_event_count == 1  # B has the worse CASE B evidence AND still correctly wins


def test_case4_same_event_already_reflected_in_mediasignals_visible():
    """CASE 4: the same physical event is already represented in
    MediaSignals (via local_performance's own real bucket assignment).
    Expected: the double-counting/provenance marker is visible."""
    take = _take("c1", 0.0, 6.0, "one two three", words=_words("one two three", 2.0, 5.0))
    context = _context("src", [_event("hand_motion_reset_candidate", 3.0, 3.3, confidence=0.9)])
    evidence = build_case_b_performance_evidence(take, context)
    row = evidence.delivery_events[0]
    assert row.mediasignal_fields == ("visual_fumble", "gesture_naturalness")


# === Section E: deterministic_best_take_authority DETERMINISTIC_OVERRIDE ====

def test_deterministic_override_annotation_only_touches_changed_groups():
    from dataclasses import replace as dataclass_replace

    from cutsell_worker.deterministic_best_take_authority import apply_deterministic_best_take_authority

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
    take_judge_groups = [{
        "group_id": "tg_1",
        "selected_clip_id": "loser",
        "winner_path": "OTHER_EXISTING_PATH",
        "performance_consulted_before_winner": False,
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
    groups_after = result_draft.diagnostics["take_judge_groups"]
    assert groups_after[0]["winner_path"] == "DETERMINISTIC_OVERRIDE"
    assert groups_after[0]["performance_consulted_before_winner"] is True
    # The actual selection change is the SAME as before D-122 -- only the
    # diagnostic annotation is new.
    assert {clip.clip_id for clip in result_draft.selected} == {"winner"}

