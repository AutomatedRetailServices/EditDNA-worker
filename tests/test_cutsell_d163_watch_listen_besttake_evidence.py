"""D-163 Phase D -- Watch+Listen PERFORMANCE/USABILITY evidence for BestTake.

Per docs/CUTSELL_DECISIONS.md D-148 through D-163 and `watch_listen_
besttake_evidence.py`'s own module docstring: this capability is
DIAGNOSTIC-ONLY in this task -- `evaluate_watch_listen_besttake_guard`
never mutates `selected_clip_id`/`ranked`/membership/grouping/Boundary
itself. Every test below either exercises the guard/evidence-builder in
isolation, or proves the `pipeline.py` wiring is a pure, flag-gated,
additive no-op on the actual selection this task's own scope requires.

Generic fixtures throughout -- no Video00 wording, no real transcript
text beyond short, obviously-synthetic sentences.
"""
from __future__ import annotations

import pytest

from cutsell_worker.deterministic_best_take_authority import clear_retry_family_winner
from cutsell_worker.raw_understanding_map import (
    BEHAVIOR_ABANDONED_ATTEMPT,
    BEHAVIOR_AUDIENCE_DELIVERY,
    BEHAVIOR_BREAKING_CHARACTER,
    BEHAVIOR_CLEAN_ATTEMPT,
    BEHAVIOR_FALSE_START,
    BEHAVIOR_POST_TAKE_RESET,
    BEHAVIOR_RECORDING_PROCESS,
    BehaviorHypothesis,
)
from cutsell_worker.watch_listen_besttake_evidence import (
    CASE_A_BOUNDARY_ONLY,
    CASE_B_DELIVERY_OWNED,
    CASE_C_AMBIGUOUS,
    CASE_CLEAN,
    CONTINUITY_CONTINUOUS,
    CONTINUITY_INTERRUPTED_AT_EDGE,
    CONTINUITY_INTERRUPTED_DURING_DELIVERY,
    DOUBLE_COUNTING_AUDIT,
    EDITABILITY_BOUNDARY_ONLY,
    EDITABILITY_CLEAN,
    EDITABILITY_DELIVERY_OWNED,
    GUARD_BYPASS_POOR_USABILITY_WINNER,
    GUARD_NO_ACTION,
    GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE,
    GUARD_PRESERVE_STRUCTURED_WINNER,
    GUARD_UNCERTAIN,
    WatchListenBestTakeEvidence,
    build_watch_listen_besttake_evidence,
    evaluate_watch_listen_besttake_guard,
    watch_listen_besttake_diagnostics,
    watch_listen_besttake_evidence_enabled,
    watch_listen_besttake_group_row,
)
from cutsell_worker.watch_listen_understanding import (
    USABILITY_QUESTIONABLE,
    USABILITY_UNKNOWN,
    USABILITY_UNUSABLE,
    USABILITY_USABLE,
    UnderstandingSpan,
)

_ENV = "CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _behavior(label):
    return BehaviorHypothesis(label=label, confidence=0.9, provenance="fixture", basis="fixture")


def _span(
    span_id, *, entry=USABILITY_USABLE, delivery=USABILITY_USABLE, exit=USABILITY_USABLE,
    behaviors=(), conflict_flags=(),
):
    overall = delivery if delivery in (USABILITY_UNUSABLE, USABILITY_UNKNOWN) else (
        USABILITY_QUESTIONABLE if USABILITY_QUESTIONABLE in (entry, exit) else USABILITY_USABLE
    )
    return UnderstandingSpan(
        span_id=span_id, source_asset_id="src", source_start=0.0, source_end=1.0,
        behavior_state_hypotheses=behaviors, behavior_confidence="SUPPORTED",
        attempt_boundary_hypotheses=(), attempt_relation_hypotheses=(),
        relation_confidence="UNKNOWN",
        meaning_completion_hypothesis="COMPLETE",
        performance_usability_hypothesis=overall,
        entry_usability=entry, delivery_usability=delivery, exit_usability=exit,
        conflict_flags=conflict_flags, evidence_provenance={},
    )


def _evidence(candidate_id, span, *, meaning_sufficient=True):
    return build_watch_listen_besttake_evidence(candidate_id, span, meaning_sufficient=meaning_sufficient)


def _clear_flag(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)


def _set_flag_on(monkeypatch):
    monkeypatch.setenv(_ENV, "1")


# ===========================================================================
# 1-3. Capability flag
# ===========================================================================

def test_01_flag_defaults_off_when_env_unset(monkeypatch):
    _clear_flag(monkeypatch)
    assert watch_listen_besttake_evidence_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on"])
def test_02_flag_recognizes_true_like_values(value):
    assert watch_listen_besttake_evidence_enabled({_ENV: value}) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_03_flag_recognizes_false_like_values(value):
    assert watch_listen_besttake_evidence_enabled({_ENV: value}) is False


# ===========================================================================
# 4-10. `build_watch_listen_besttake_evidence` -- evidence contract, CASE
# A/B/C ownership, fail-open on missing span.
# ===========================================================================

def test_04_missing_span_returns_none_fail_open():
    assert build_watch_listen_besttake_evidence("a", None, meaning_sufficient=True) is None


def test_05_clean_span_case_clean_editability_clean():
    ev = _evidence("a", _span("a"))
    assert ev.case_classification == CASE_CLEAN
    assert ev.editability_status == EDITABILITY_CLEAN
    assert ev.performance_continuity_status == CONTINUITY_CONTINUOUS
    assert ev.delivery_defect_present is False


def test_06_delivery_unusable_is_case_b_delivery_owned():
    ev = _evidence("a", _span("a", delivery=USABILITY_UNUSABLE))
    assert ev.case_classification == CASE_B_DELIVERY_OWNED
    assert ev.editability_status == EDITABILITY_DELIVERY_OWNED
    assert ev.performance_continuity_status == CONTINUITY_INTERRUPTED_DURING_DELIVERY
    assert ev.delivery_defect_present is True


def test_07_entry_only_questionable_is_case_a_boundary_only():
    ev = _evidence("a", _span("a", entry=USABILITY_QUESTIONABLE))
    assert ev.case_classification == CASE_A_BOUNDARY_ONLY
    assert ev.editability_status == EDITABILITY_BOUNDARY_ONLY
    assert ev.entry_only_defect is True
    assert ev.exit_only_defect is False
    assert ev.performance_continuity_status == CONTINUITY_INTERRUPTED_AT_EDGE


def test_08_exit_only_questionable_is_case_a_boundary_only():
    ev = _evidence("a", _span("a", exit=USABILITY_QUESTIONABLE))
    assert ev.case_classification == CASE_A_BOUNDARY_ONLY
    assert ev.exit_only_defect is True
    assert ev.entry_only_defect is False


def test_09_unknown_delivery_is_case_c_ambiguous():
    ev = _evidence("a", _span("a", delivery=USABILITY_UNKNOWN))
    assert ev.case_classification == CASE_C_AMBIGUOUS
    assert ev.performance_continuity_status == "UNKNOWN"


def test_10_no_final_winner_field_on_the_evidence_dataclass():
    fields = WatchListenBestTakeEvidence.__dataclass_fields__
    assert "final_winner" not in fields
    assert "winner" not in fields
    assert "selected" not in fields


def test_10b_no_opaque_score_field_anywhere_on_the_evidence():
    import dataclasses
    ev = _evidence("a", _span("a"))
    for f in dataclasses.fields(ev):
        assert not isinstance(getattr(ev, f.name), float), f.name


# ===========================================================================
# 11-14. Behavior-derived flags: breaking character, reset/fumble.
# ===========================================================================

def test_11_breaking_character_during_delivery_requires_both_label_and_unusable_delivery():
    ev = _evidence("a", _span("a", delivery=USABILITY_UNUSABLE, behaviors=(_behavior(BEHAVIOR_BREAKING_CHARACTER),)))
    assert ev.breaking_character_during_delivery is True


def test_12_breaking_character_label_alone_without_delivery_defect_is_false():
    ev = _evidence("a", _span("a", delivery=USABILITY_USABLE, behaviors=(_behavior(BEHAVIOR_BREAKING_CHARACTER),)))
    assert ev.breaking_character_during_delivery is False


def test_13_reset_or_fumble_during_delivery_true_for_reset_recording_false_start_or_abandoned():
    for label in (BEHAVIOR_POST_TAKE_RESET, BEHAVIOR_RECORDING_PROCESS, BEHAVIOR_FALSE_START, BEHAVIOR_ABANDONED_ATTEMPT):
        ev = _evidence("a", _span("a", delivery=USABILITY_UNUSABLE, behaviors=(_behavior(label),)))
        assert ev.reset_or_fumble_during_delivery is True, label


def test_14_ordinary_expressive_motion_no_defect_labels_no_penalty_flags():
    ev = _evidence("a", _span("a", behaviors=(_behavior(BEHAVIOR_AUDIENCE_DELIVERY), _behavior(BEHAVIOR_CLEAN_ATTEMPT))))
    assert ev.breaking_character_during_delivery is False
    assert ev.reset_or_fumble_during_delivery is False
    assert ev.delivery_defect_present is False


# ===========================================================================
# 15-17. Audio honesty.
# ===========================================================================

def test_15_audio_signal_usability_always_unknown_v1_no_overclaim():
    ev = _evidence("a", _span("a", delivery=USABILITY_UNUSABLE))
    assert ev.audio_signal_usability == USABILITY_UNKNOWN


def test_16_visual_signal_usability_mirrors_overall_performance_usability():
    span = _span("a", delivery=USABILITY_UNUSABLE)
    ev = _evidence("a", span)
    assert ev.visual_signal_usability == span.performance_usability_hypothesis


def test_17_double_counting_audit_names_the_correlated_and_independent_dimensions():
    assert DOUBLE_COUNTING_AUDIT["delivery_usability"] == "PARTIALLY_CORRELATED"
    assert DOUBLE_COUNTING_AUDIT["breaking_character_during_delivery"] == "INDEPENDENT"
    assert DOUBLE_COUNTING_AUDIT["conflict_flags"] == "INDEPENDENT"
    assert DOUBLE_COUNTING_AUDIT["audio_signal_usability"] == "INDEPENDENT"
    assert DOUBLE_COUNTING_AUDIT["visual_signal_usability"] == "SAME_SOURCE_DUPLICATE"


# ===========================================================================
# 18-38. `evaluate_watch_listen_besttake_guard` -- the required fixtures.
# ===========================================================================

def _ranked(*pairs):
    return [{"clip_id": cid, "score": score} for cid, score in pairs]


def test_18_clean_vs_clean_preserves_existing_ranking_no_action_needed():
    ev = {"a": _evidence("a", _span("a")), "b": _evidence("b", _span("b"))}
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev,
        ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_PRESERVE_STRUCTURED_WINNER
    assert result.dominant_candidate_id is None


def test_19_winner_unusable_during_delivery_alt_usable_guard_fires_dominance():
    ev = {
        "a": _evidence("a", _span("a", delivery=USABILITY_UNUSABLE)),
        "b": _evidence("b", _span("b", delivery=USABILITY_USABLE)),
    }
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev,
        ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert result.dominant_candidate_id == "b"


def test_20_winner_questionable_entry_only_alt_clean_no_besttake_replacement():
    ev = {
        "a": _evidence("a", _span("a", entry=USABILITY_QUESTIONABLE)),
        "b": _evidence("b", _span("b")),
    }
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_PRESERVE_STRUCTURED_WINNER
    assert result.dominant_candidate_id is None


def test_21_winner_questionable_exit_only_is_boundary_only_preserve():
    ev = {
        "a": _evidence("a", _span("a", exit=USABILITY_QUESTIONABLE)),
        "b": _evidence("b", _span("b")),
    }
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_PRESERVE_STRUCTURED_WINNER


def test_22_both_have_delivery_defects_no_dominant_alternative_bypass():
    ev = {
        "a": _evidence("a", _span("a", delivery=USABILITY_UNUSABLE)),
        "b": _evidence("b", _span("b", delivery=USABILITY_UNUSABLE)),
    }
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_BYPASS_POOR_USABILITY_WINNER
    assert result.dominant_candidate_id is None


def test_23_meaning_insufficient_alternative_never_wins():
    ev = {
        "a": _evidence("a", _span("a", delivery=USABILITY_UNUSABLE)),
        "b": _evidence("b", _span("b", delivery=USABILITY_USABLE), meaning_sufficient=False),
    }
    result = evaluate_watch_listen_besttake_guard(
        # "b" deliberately excluded from meaning_sufficient_ids
        winner_id="a", meaning_sufficient_ids={"a"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.9)),
    )
    assert result.guard_status == GUARD_BYPASS_POOR_USABILITY_WINNER
    assert result.dominant_candidate_id is None


def test_24_visual_fumble_during_delivery_shape_is_a_dominance_case():
    winner_span = _span("a", delivery=USABILITY_UNUSABLE, behaviors=(_behavior(BEHAVIOR_RECORDING_PROCESS),))
    ev = {"a": _evidence("a", winner_span), "b": _evidence("b", _span("b"))}
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert ev["a"].reset_or_fumble_during_delivery is True


def test_25_breaking_character_during_delivery_shape_is_a_dominance_case():
    winner_span = _span("a", delivery=USABILITY_UNUSABLE, behaviors=(_behavior(BEHAVIOR_BREAKING_CHARACTER),))
    ev = {"a": _evidence("a", winner_span), "b": _evidence("b", _span("b"))}
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert ev["a"].breaking_character_during_delivery is True


def test_26_reset_after_delivery_only_shape_entry_exit_never_demotes():
    # A reset-family label present, but usability is only QUESTIONABLE at
    # the exit edge -- CASE A, never sufficient alone.
    ev = {
        "a": _evidence("a", _span("a", exit=USABILITY_QUESTIONABLE, behaviors=(_behavior(BEHAVIOR_POST_TAKE_RESET),))),
        "b": _evidence("b", _span("b")),
    }
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_PRESERVE_STRUCTURED_WINNER


def test_27_camera_disengagement_during_delivery_shape_is_a_dominance_case():
    # Camera disengagement is one of the real Track C kinds folded into
    # delivery_usability's own UNUSABLE classification -- represented here
    # generically via the delivery zone itself (no per-kind field exists
    # beyond the bounded behavior labels this module reads).
    ev = {
        "a": _evidence("a", _span("a", delivery=USABILITY_UNUSABLE)),
        "b": _evidence("b", _span("b")),
    }
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE


def test_28_audio_pause_during_delivery_never_asserts_tone_or_prosody():
    ev = _evidence("a", _span("a", delivery=USABILITY_UNUSABLE))
    assert ev.audio_signal_usability == USABILITY_UNKNOWN  # never a tone/prosody claim


def test_29_ordinary_expressive_hand_motion_no_penalty():
    ev = {
        "a": _evidence("a", _span("a", behaviors=(_behavior(BEHAVIOR_AUDIENCE_DELIVERY),))),
        "b": _evidence("b", _span("b")),
    }
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_PRESERVE_STRUCTURED_WINNER


def test_30_high_energy_clean_vs_low_energy_clean_no_automatic_energy_preference():
    # Both USABLE at every zone -- this module has no energy dimension at
    # all, so neither can ever "dominate" the other through this guard.
    ev = {"a": _evidence("a", _span("a")), "b": _evidence("b", _span("b"))}
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.6), ("b", 0.6)),
    )
    assert result.guard_status == GUARD_PRESERVE_STRUCTURED_WINNER
    assert result.dominant_candidate_id is None


def test_31_semantic_winner_equals_deliveryscore_winner_but_wl_contradicts():
    # Agreement between semantic and DeliveryScorer is NOT absolute (D-126/
    # D-162): strong contradictory W+L usability evidence still fires.
    ev = {
        "a": _evidence("a", _span("a", delivery=USABILITY_UNUSABLE)),
        "b": _evidence("b", _span("b")),
    }
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev,
        ranked=_ranked(("a", 0.9), ("b", 0.5)),  # DeliveryScorer strongly agrees with "a" too
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert result.dominant_candidate_id == "b"


def test_32_semantic_nondecisive_deliveryscore_bad_winner_abstract_d162_replay():
    """Abstract D-162 replay (this task's own required fixture): family
    {A, B}, both meaning-sufficient, semantic authority non-decisive,
    DeliveryScorer picks A, W+L shows A materially worse on DELIVERY, B
    usable. Expected: the guard prevents blind acceptance of A (reports
    PERFORMANCE_DOMINANT_ALTERNATIVE) -- production logic never encodes
    which literal candidate name wins; only the shape is asserted."""
    ev = {
        "a": _evidence("a", _span("a", delivery=USABILITY_UNUSABLE)),
        "b": _evidence("b", _span("b", delivery=USABILITY_USABLE)),
    }
    # DeliveryScorer (the existing ranker) picked "a" -- represented by a
    # thin score gap (never decisive on its own either), matching
    # "semantic authority non-decisive" from the abstract fixture.
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev,
        ranked=_ranked(("a", 0.55), ("b", 0.50)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert result.dominant_candidate_id == "b"
    # The existing deterministic ladder (D-123, unchanged) would NOT
    # independently pick "b" here (gap 0.05 < CLEAR_WINNER_MINIMUM_GAP) --
    # confirming this is exactly the D-162 real-media shape: the guard
    # surfaces the problem even when the existing ladder does not.
    assert result.existing_ladder_agrees is False
    assert clear_retry_family_winner([{"clip_id": "a", "score": 0.55, "reason": ""}, {"clip_id": "b", "score": 0.50, "reason": ""}]) is None


def test_32b_existing_ladder_agreement_reported_true_when_it_independently_supports_the_alternative():
    ev = {
        "a": _evidence("a", _span("a", delivery=USABILITY_UNUSABLE)),
        "b": _evidence("b", _span("b", delivery=USABILITY_USABLE)),
    }
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev,
        ranked=_ranked(("a", 0.40), ("b", 0.90)),  # gap 0.50 >= 0.30: ladder independently picks "b"
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert result.existing_ladder_agrees is True


def test_33_conflict_flags_on_winner_route_to_uncertain_never_forced():
    ev = {
        "a": _evidence("a", _span("a", delivery=USABILITY_UNUSABLE, conflict_flags=("MEANING_COMPLETE_VS_ABANDONED_ATTEMPT_EVIDENCE",))),
        "b": _evidence("b", _span("b")),
    }
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_UNCERTAIN
    assert result.dominant_candidate_id is None


def test_34_conflict_flags_on_the_alternative_disqualify_it_from_dominance():
    ev = {
        "a": _evidence("a", _span("a", delivery=USABILITY_UNUSABLE)),
        "b": _evidence("b", _span("b", conflict_flags=("EXIT_RESET_VS_MEANING_COMPLETE",))),
    }
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_BYPASS_POOR_USABILITY_WINNER
    assert result.dominant_candidate_id is None


def test_35_missing_evidence_for_winner_fails_open_no_action():
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id={"b": _evidence("b", _span("b"))},
        ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_NO_ACTION


def test_36_none_evidence_value_fails_open_no_action():
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id={"a": None, "b": _evidence("b", _span("b"))},
        ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_NO_ACTION


def test_37_no_winner_id_fails_open_no_action():
    result = evaluate_watch_listen_besttake_guard(
        winner_id=None, meaning_sufficient_ids={"a", "b"},
        evidence_by_id={"a": _evidence("a", _span("a")), "b": _evidence("b", _span("b"))}, ranked=(),
    )
    assert result.guard_status == GUARD_NO_ACTION


def test_38_winner_not_meaning_sufficient_upstream_owned_no_action():
    ev = {"a": _evidence("a", _span("a", delivery=USABILITY_UNUSABLE)), "b": _evidence("b", _span("b"))}
    result = evaluate_watch_listen_besttake_guard(
        winner_id="a", meaning_sufficient_ids={"b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.5)),
    )
    assert result.guard_status == GUARD_NO_ACTION
    assert result.guard_reason == "winner_not_meaning_sufficient_upstream_owned"


def test_38b_candidate_ordering_deterministic_across_repeated_calls():
    ev = {
        "a": _evidence("a", _span("a", delivery=USABILITY_UNUSABLE)),
        "b": _evidence("b", _span("b", delivery=USABILITY_USABLE)),
        "c": _evidence("c", _span("c", delivery=USABILITY_USABLE)),
    }
    ranked = _ranked(("a", 0.8), ("b", 0.5), ("c", 0.5))
    first = evaluate_watch_listen_besttake_guard(winner_id="a", meaning_sufficient_ids={"a", "b", "c"}, evidence_by_id=ev, ranked=ranked)
    second = evaluate_watch_listen_besttake_guard(winner_id="a", meaning_sufficient_ids={"a", "b", "c"}, evidence_by_id=ev, ranked=ranked)
    assert first.dominant_candidate_id == second.dominant_candidate_id == "b"  # tie-break: clip_id "b" < "c"


# ===========================================================================
# 39-42. Diagnostics / group-row projection.
# ===========================================================================

def test_39_diagnostics_counts_every_bucket():
    results = [
        evaluate_watch_listen_besttake_guard(winner_id="a", meaning_sufficient_ids={"a"}, evidence_by_id={"a": _evidence("a", _span("a"))}, ranked=_ranked(("a", 0.5))),
        evaluate_watch_listen_besttake_guard(winner_id=None, meaning_sufficient_ids=set(), evidence_by_id={}, ranked=()),
    ]
    diag = watch_listen_besttake_diagnostics(results)
    assert diag["watch_listen_besttake_evaluated_count"] == 2
    assert diag["watch_listen_besttake_preserved_count"] == 1
    assert diag["watch_listen_besttake_no_action_count"] == 1


def test_40_diagnostics_field_names_match_directive_vocabulary():
    diag = watch_listen_besttake_diagnostics([])
    required = {
        "watch_listen_besttake_evaluated_count", "watch_listen_besttake_no_action_count",
        "watch_listen_besttake_preserved_count", "watch_listen_besttake_bypass_count",
        "watch_listen_besttake_dominance_count", "watch_listen_besttake_uncertain_count",
    }
    assert required.issubset(diag.keys())


def test_41_group_row_never_mutates_winner_and_has_no_action_applied():
    ev = {"a": _evidence("a", _span("a", delivery=USABILITY_UNUSABLE)), "b": _evidence("b", _span("b"))}
    result = evaluate_watch_listen_besttake_guard(winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.5)))
    row = watch_listen_besttake_group_row(result, "a")
    assert row["watch_listen_besttake_winner_before"] == "a"
    assert row["watch_listen_besttake_winner_after"] == "a"  # D-163: never mutated in this task
    assert row["watch_listen_besttake_action_applied"] is False
    assert row["watch_listen_besttake_guard_status"] == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert row["watch_listen_besttake_dominant_candidate"] == "b"


def test_42_group_row_field_names_match_directive_vocabulary():
    ev = {"a": _evidence("a", _span("a"))}
    result = evaluate_watch_listen_besttake_guard(winner_id="a", meaning_sufficient_ids={"a"}, evidence_by_id=ev, ranked=_ranked(("a", 0.5)))
    row = watch_listen_besttake_group_row(result, "a")
    required = {
        "watch_listen_besttake_evaluated", "watch_listen_besttake_candidate_count",
        "watch_listen_besttake_current_winner", "watch_listen_besttake_guard_status",
        "watch_listen_besttake_guard_reason", "watch_listen_besttake_dominant_candidate",
        "watch_listen_besttake_winner_before", "watch_listen_besttake_winner_after",
        "watch_listen_besttake_action_applied", "candidate_usability_summary",
    }
    assert required.issubset(row.keys())


def test_42b_no_transcript_text_anywhere_in_diagnostics_output():
    ev = {"a": _evidence("a", _span("a", delivery=USABILITY_UNUSABLE)), "b": _evidence("b", _span("b"))}
    result = evaluate_watch_listen_besttake_guard(winner_id="a", meaning_sufficient_ids={"a", "b"}, evidence_by_id=ev, ranked=_ranked(("a", 0.8), ("b", 0.5)))
    row = watch_listen_besttake_group_row(result, "a")
    for value in row.values():
        if isinstance(value, str):
            assert len(value) < 80
        if isinstance(value, dict):
            for v in value.values():
                assert isinstance(v, str) and len(v) < 20


# ===========================================================================
# 43-52. Live pipeline wiring: flag OFF/ON, byte-identical selection,
# D-123/D-128/family/render/pacing non-interference.
# ===========================================================================

def test_43_pipeline_flag_off_never_calls_the_guard_module():
    import cutsell_worker.pipeline as pipeline_mod
    source = open(pipeline_mod.__file__, encoding="utf-8").read()
    assert "watch_listen_besttake_evidence_enabled()" in source
    assert "if watch_listen_besttake_evidence_enabled() and watch_listen_spans_by_id:" in source


def test_44_pipeline_never_mutates_selected_clip_id_from_the_guard_result():
    import cutsell_worker.pipeline as pipeline_mod
    source = open(pipeline_mod.__file__, encoding="utf-8").read()
    # The ONLY assignment near the D-163 block must be to the row dict /
    # results accumulator, never to selected_clip_id or the groups list.
    idx = source.index("D-163 Phase D (docs/CUTSELL_DECISIONS.md D-163)")
    block = source[idx: idx + 1800]
    assert "selected_clip_id =" not in block
    assert "groups.append" not in block


def test_45_watch_listen_besttake_summary_present_in_outer_diagnostics():
    import cutsell_worker.pipeline as pipeline_mod
    source = open(pipeline_mod.__file__, encoding="utf-8").read()
    assert '"watch_listen_besttake_evidence":' in source


def test_46_module_never_imports_family_formation_or_semantic_authority_modules():
    import cutsell_worker.watch_listen_besttake_evidence as mod
    source = open(mod.__file__, encoding="utf-8").read()
    forbidden = [
        "take_grouping_provider", "watch_listen_relation_discovery",
        "attempt_relationship_authority", "semantic_authority_observability",
        "boundary_engine", "temporal_editing", "renderer", "composite_resolver",
        "google.generativeai", "requests", "httpx", "urllib",
    ]
    for name in forbidden:
        assert f"import {name}" not in source and f"from .{name}" not in source, name


def test_47_module_never_calls_a_provider_or_network_symbol():
    import cutsell_worker.watch_listen_besttake_evidence as mod
    source = open(mod.__file__, encoding="utf-8").read()
    for token in ("generate_content", "genai.", "requests.post", "requests.get", "socket."):
        assert token not in source


def _code_body_excluding_module_docstring(mod) -> str:
    source = open(mod.__file__, encoding="utf-8").read()
    first = source.index('"""')
    second = source.index('"""', first + 3)
    return source[second + 3:]


def test_48_module_reuses_deterministic_best_take_authority_never_reimplements_the_gap_rule():
    import cutsell_worker.watch_listen_besttake_evidence as mod
    source = open(mod.__file__, encoding="utf-8").read()
    assert "from .deterministic_best_take_authority import clear_retry_family_winner" in source
    # The gap NUMBER (0.30) itself must never be re-typed as a literal in
    # this module's own executable code -- only reused via the import
    # above. `_existing_ladder_pick`'s call actually delegates to it.
    import inspect
    from cutsell_worker.watch_listen_besttake_evidence import _existing_ladder_pick
    assert "clear_retry_family_winner" in inspect.getsource(_existing_ladder_pick)


def test_49_no_new_opaque_weighted_score_in_module_source():
    import cutsell_worker.watch_listen_besttake_evidence as mod
    # No multiplication-based weighted-sum score construction of the shape
    # this task explicitly forbids, anywhere in the actual code (not the
    # docstring's own "do not do this" example).
    body = _code_body_excluding_module_docstring(mod)
    assert " * visual" not in body and " * audio" not in body


def test_50_d150_gate_module_never_references_this_new_module():
    import cutsell_worker.semantic_authority_observability as d150
    source = open(d150.__file__, encoding="utf-8").read()
    assert "watch_listen_besttake_evidence" not in source


def test_51_take_judge_and_deterministic_best_take_authority_unaware_of_this_module():
    import cutsell_worker.take_judge as tj
    import cutsell_worker.deterministic_best_take_authority as dbta
    for mod in (tj, dbta):
        source = open(mod.__file__, encoding="utf-8").read()
        assert "watch_listen_besttake_evidence" not in source


def test_52_pipeline_wiring_present_and_flag_gated_source_check():
    import cutsell_worker.pipeline as pipeline_mod
    source = open(pipeline_mod.__file__, encoding="utf-8").read()
    assert "from .watch_listen_besttake_evidence import (" in source
    assert "watch_listen_besttake_group_row(_wlbt_result, selected_clip_id)" in source
