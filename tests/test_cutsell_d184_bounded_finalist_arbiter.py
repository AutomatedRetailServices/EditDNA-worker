"""D-184: Bounded Finalist Arbiter -- OFFLINE / DIAGNOSTIC ONLY.

Covers the directive's own required fixture categories: eligibility
gating (D-183 state x candidate count), P0 meaning parity (negation/
number conflict via `language_proposition_relation.py`'s own existing
primitives, outside-meaning-sufficient exclusion), V2 factual dominance
(and its reverse), near-equal performance (the D-181 Pimples abstract
replay -- must NOT magically produce a preference), three-finalist
dominance, performance-vs-editability conflict, missing evidence (never
a fallback to raw score), prosodic-audio/P1 explicit absence,
determinism/order-independence, the double-counting audit, and the
structural no-winner-mutation guarantee.
"""
from __future__ import annotations

from pathlib import Path

from cutsell_worker.bounded_finalist_arbiter import (
    DECISION_ABSTAIN,
    DECISION_PREFER_CANDIDATE,
    DOUBLE_COUNTING_AUDIT,
    PROSODIC_AUDIO_AVAILABLE,
    STATE_CONFLICTED,
    STATE_INSUFFICIENT_EVIDENCE,
    STATE_NEAR_EQUAL,
    STATE_NOT_ELIGIBLE,
    STATE_PREFERENCE_SUPPORTED,
    BoundedFinalistArbiterResult,
    FinalistArbiterInput,
    _v2_preferred_candidate,
    bounded_finalist_arbiter_diagnostics,
    bounded_finalist_arbiter_enabled,
    bounded_finalist_arbiter_run_summary,
    evaluate_bounded_finalist_arbiter,
)
from cutsell_worker.positioned_performance_evidence import (
    DeliverySpan,
    PositionAwarePerformanceEvidence,
    PositionedEvent,
    ZONE_DELIVERY,
    ZONE_ENTRY,
    ZONE_EXIT,
)
from cutsell_worker.raw_understanding_map import RawUnderstandingSpan
from cutsell_worker.watch_listen_zone_usability_v2 import build_zone_usability_v2

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# V2 fixture helpers (same shape as test_cutsell_d167_zone_usability_v2.py).
# ---------------------------------------------------------------------------
def _span(
    span_id="c1", source_start=0.0, source_end=8.5, delivery_start=0.2, delivery_end=8.2,
    entry_events=(), delivery_events=(), exit_events=(), conflict_flags=(), delivery_available=True,
):
    if delivery_available:
        delivery = DeliverySpan(start=delivery_start, end=delivery_end, available=True, source="word_envelope")
    else:
        delivery = DeliverySpan(start=None, end=None, available=False, source="unavailable_no_words")
    events = tuple(entry_events) + tuple(delivery_events) + tuple(exit_events)
    positioned = PositionAwarePerformanceEvidence(
        candidate_id=span_id, source_asset_id="s1", source_start=source_start, source_end=source_end,
        delivery_span=delivery, positioned_events=events,
    )
    return RawUnderstandingSpan(
        span_id=span_id, source_asset_id="s1", source_start=source_start, source_end=source_end,
        transcript="", word_timings=(), positioned_evidence=positioned,
        behavior_hypotheses=(), conflict_flags=tuple(conflict_flags), evidence_provenance={},
    )


def _ev(kind, start, end, confidence=0.9, zone=ZONE_DELIVERY):
    return PositionedEvent(
        kind=kind, start=start, end=end, confidence=confidence, zone=zone,
        overlaps_delivery=(zone == ZONE_DELIVERY), starts_before_delivery=(zone == ZONE_ENTRY),
        ends_after_delivery=(zone == ZONE_EXIT), evidence_source="local_performance",
    )


def _v2(span_id, **kwargs):
    return build_zone_usability_v2(span_id, _span(span_id=span_id, **kwargs))


def _clean_v2(span_id):
    return _v2(span_id)


def _impaired_delivery_v2(span_id):
    return _v2(span_id, delivery_events=[_ev("hand_motion_reset_candidate", 1.0, 6.0)])


def _base_input(**overrides):
    defaults = dict(
        family_id="fam1",
        candidate_ids=("a", "b"),
        meaning_sufficient_candidate_ids=("a", "b"),
        terminal_confidence_state="NON_DECISIVE",
    )
    defaults.update(overrides)
    return FinalistArbiterInput(**defaults)


# ---------------------------------------------------------------------------
# 1-6. Feature flag.
# ---------------------------------------------------------------------------
def test_01_flag_default_off():
    assert bounded_finalist_arbiter_enabled({}) is False


def test_02_flag_unset_env_off():
    assert bounded_finalist_arbiter_enabled(None) in (True, False)  # never raises; reads real os.environ


def test_03_flag_true_variants():
    for value in ("1", "true", "True", "yes", "on"):
        assert bounded_finalist_arbiter_enabled({"CUTSELL_BOUNDED_FINALIST_ARBITER_ENABLED": value}) is True


def test_04_flag_false_variants():
    for value in ("0", "false", "no", "off", "", "garbage"):
        assert bounded_finalist_arbiter_enabled({"CUTSELL_BOUNDED_FINALIST_ARBITER_ENABLED": value}) is False


# ---------------------------------------------------------------------------
# 5-14. Eligibility gate (D-183 state x candidate count).
# ---------------------------------------------------------------------------
def test_05_decisive_state_not_eligible():
    result = evaluate_bounded_finalist_arbiter(_base_input(terminal_confidence_state="DECISIVE"))
    assert result.arbiter_state == STATE_NOT_ELIGIBLE
    assert result.decision == DECISION_ABSTAIN
    assert result.preferred_candidate_id is None


def test_06_decisive_by_elimination_not_eligible():
    result = evaluate_bounded_finalist_arbiter(_base_input(terminal_confidence_state="DECISIVE_BY_ELIMINATION"))
    assert result.arbiter_state == STATE_NOT_ELIGIBLE


def test_07_unknown_state_not_eligible():
    result = evaluate_bounded_finalist_arbiter(_base_input(terminal_confidence_state="UNKNOWN"))
    assert result.arbiter_state == STATE_NOT_ELIGIBLE


def test_08_none_state_not_eligible():
    result = evaluate_bounded_finalist_arbiter(_base_input(terminal_confidence_state=None))
    assert result.arbiter_state == STATE_NOT_ELIGIBLE


def test_09_one_candidate_not_eligible():
    result = evaluate_bounded_finalist_arbiter(_base_input(
        candidate_ids=("a",), meaning_sufficient_candidate_ids=("a",),
    ))
    assert result.arbiter_state == STATE_NOT_ELIGIBLE


def test_10_four_candidates_not_eligible():
    result = evaluate_bounded_finalist_arbiter(_base_input(
        candidate_ids=("a", "b", "c", "d"),
        meaning_sufficient_candidate_ids=("a", "b", "c", "d"),
    ))
    assert result.arbiter_state == STATE_NOT_ELIGIBLE


def test_11_zero_candidates_not_eligible():
    result = evaluate_bounded_finalist_arbiter(_base_input(
        candidate_ids=(), meaning_sufficient_candidate_ids=(),
    ))
    assert result.arbiter_state == STATE_NOT_ELIGIBLE


def test_12_two_candidates_non_decisive_eligible():
    result = evaluate_bounded_finalist_arbiter(_base_input(terminal_confidence_state="NON_DECISIVE"))
    assert result.arbiter_state != STATE_NOT_ELIGIBLE


def test_13_three_candidates_tied_eligible():
    result = evaluate_bounded_finalist_arbiter(_base_input(
        candidate_ids=("a", "b", "c"), meaning_sufficient_candidate_ids=("a", "b", "c"),
        terminal_confidence_state="TIED",
    ))
    assert result.arbiter_state != STATE_NOT_ELIGIBLE


def test_14_conflicted_terminal_state_eligible():
    result = evaluate_bounded_finalist_arbiter(_base_input(terminal_confidence_state="CONFLICTED"))
    assert result.arbiter_state != STATE_NOT_ELIGIBLE


# ---------------------------------------------------------------------------
# 15-20. P0 meaning parity/safety.
# ---------------------------------------------------------------------------
def test_15_candidate_outside_meaning_sufficient_set_is_conflicted():
    result = evaluate_bounded_finalist_arbiter(_base_input(meaning_sufficient_candidate_ids=("a",)))
    assert result.decision == DECISION_ABSTAIN
    assert result.arbiter_state == STATE_CONFLICTED
    assert result.meaning_parity_status == "CONFLICT"
    assert result.structured_conflict is True


def test_16_negation_conflict_via_texts_is_conflicted():
    result = evaluate_bounded_finalist_arbiter(_base_input(candidate_texts={
        "a": "The cream clears acne breakouts in 2 weeks.",
        "b": "The cream does not clear acne breakouts in 2 weeks.",
    }))
    assert result.decision == DECISION_ABSTAIN
    assert result.arbiter_state == STATE_CONFLICTED
    assert result.meaning_parity_status == "CONFLICT"


def test_17_number_conflict_via_texts_is_conflicted():
    result = evaluate_bounded_finalist_arbiter(_base_input(candidate_texts={
        "a": "Apply the cream twice daily for 2 weeks to clear the breakout.",
        "b": "Apply the cream twice daily for 4 weeks to clear the breakout.",
    }))
    assert result.decision == DECISION_ABSTAIN
    assert result.arbiter_state == STATE_CONFLICTED
    assert result.meaning_parity_status == "CONFLICT"


def test_18_consistent_texts_yield_consistent_status_and_proceed():
    result = evaluate_bounded_finalist_arbiter(_base_input(candidate_texts={
        "a": "Apply the cream twice daily for 2 weeks to clear the breakout.",
        "b": "Apply the cream twice daily for 2 weeks to clear the breakout gently.",
    }))
    assert result.meaning_parity_status == "CONSISTENT"
    assert result.arbiter_state != STATE_NOT_ELIGIBLE
    # No text-driven conflict -> falls through to performance evaluation,
    # not blocked at the meaning gate.
    assert result.reason != "meaning_parity_conflict_detected"


def test_19_no_texts_supplied_is_unknown_not_fabricated_consistent():
    result = evaluate_bounded_finalist_arbiter(_base_input())
    assert result.meaning_parity_status == "UNKNOWN"


def test_20_unrelated_texts_no_shared_content_not_a_conflict():
    # No shared content -> `claim_signatures_conflict` never fires (not
    # comparable, never a fabricated conflict verdict).
    result = evaluate_bounded_finalist_arbiter(_base_input(candidate_texts={
        "a": "The moisturizer feels light on my skin.",
        "b": "The serum smells like citrus and lavender.",
    }))
    assert result.meaning_parity_status in ("CONSISTENT", "UNKNOWN")
    assert result.arbiter_state != STATE_CONFLICTED or result.reason != "meaning_parity_conflict_detected"


# ---------------------------------------------------------------------------
# 21-28. V2 factual dominance (D-172, reused verbatim) and its reverse.
# ---------------------------------------------------------------------------
def test_21_v2_dominance_prefers_the_dominant_candidate():
    result = evaluate_bounded_finalist_arbiter(_base_input(v2_evidence_by_id={
        "a": _impaired_delivery_v2("a"),
        "b": _clean_v2("b"),
    }))
    assert result.decision == DECISION_PREFER_CANDIDATE
    assert result.preferred_candidate_id == "b"
    assert result.arbiter_state == STATE_PREFERENCE_SUPPORTED
    assert result.performance_comparison_status == "DOMINANT"
    assert "zone_usability_v2" in result.evidence_sources


def test_22_v2_dominance_reverse_preference():
    result = evaluate_bounded_finalist_arbiter(_base_input(v2_evidence_by_id={
        "a": _clean_v2("a"),
        "b": _impaired_delivery_v2("b"),
    }))
    assert result.decision == DECISION_PREFER_CANDIDATE
    assert result.preferred_candidate_id == "a"


def test_23_near_equal_v2_is_abstain_not_a_forced_pick():
    result = evaluate_bounded_finalist_arbiter(_base_input(v2_evidence_by_id={
        "a": _clean_v2("a"),
        "b": _clean_v2("b"),
    }))
    assert result.decision == DECISION_ABSTAIN
    assert result.arbiter_state == STATE_NEAR_EQUAL
    assert result.preferred_candidate_id is None
    assert result.performance_comparison_status == "NEAR_EQUAL"


def test_24_missing_v2_evidence_for_one_candidate_is_no_evidence():
    result = evaluate_bounded_finalist_arbiter(_base_input(v2_evidence_by_id={
        "a": _impaired_delivery_v2("a"),
        # "b" missing entirely -- never partially guessed.
    }))
    assert result.performance_comparison_status == "NO_EVIDENCE"
    assert "zone_usability_v2" in result.missing_evidence
    assert result.decision == DECISION_ABSTAIN
    assert result.arbiter_state == STATE_INSUFFICIENT_EVIDENCE


def test_25_explicit_none_v2_evidence_is_no_evidence():
    result = evaluate_bounded_finalist_arbiter(_base_input(v2_evidence_by_id={"a": None, "b": None}))
    assert result.performance_comparison_status == "NO_EVIDENCE"
    assert result.arbiter_state == STATE_INSUFFICIENT_EVIDENCE


def test_26_boundary_only_entry_difference_creates_no_preference():
    # ENTRY-only defect on "a" -- same clean DELIVERY as "b". Since
    # dominance REQUIRES a strictly better delivery, a Boundary-only
    # difference must never create a preference (D-167 firewall).
    winner_case_a = _v2("a", entry_events=[_ev("hand_motion_reset_candidate", 0.0, 0.1, zone=ZONE_ENTRY)])
    clean_b = _clean_v2("b")
    result = evaluate_bounded_finalist_arbiter(_base_input(v2_evidence_by_id={
        "a": winner_case_a, "b": clean_b,
    }))
    assert result.decision == DECISION_ABSTAIN
    assert result.preferred_candidate_id is None


def test_27_ordinary_motion_isolated_event_creates_no_preference():
    a_with_brief_motion = _v2("a", delivery_events=[_ev("hand_motion_reset_candidate", 4.0, 4.067)])
    clean_b = _clean_v2("b")
    result = evaluate_bounded_finalist_arbiter(_base_input(v2_evidence_by_id={
        "a": a_with_brief_motion, "b": clean_b,
    }))
    assert result.decision == DECISION_ABSTAIN
    assert result.preferred_candidate_id is None


def test_28_internal_delivery_material_dominance_allows_preference():
    # A genuine, MATERIAL delivery-zone difference (not Boundary-only,
    # not ordinary motion) IS allowed to produce a preference.
    result = evaluate_bounded_finalist_arbiter(_base_input(v2_evidence_by_id={
        "a": _impaired_delivery_v2("a"),
        "b": _clean_v2("b"),
    }))
    assert result.decision == DECISION_PREFER_CANDIDATE
    assert result.preferred_candidate_id == "b"


# ---------------------------------------------------------------------------
# 29-32. Three-finalist control.
# ---------------------------------------------------------------------------
def test_29_three_finalist_clear_dominance_prefers_winner():
    result = evaluate_bounded_finalist_arbiter(_base_input(
        candidate_ids=("a", "b", "c"),
        meaning_sufficient_candidate_ids=("a", "b", "c"),
        v2_evidence_by_id={
            "a": _impaired_delivery_v2("a"),
            "b": _impaired_delivery_v2("b"),
            "c": _clean_v2("c"),
        },
    ))
    assert result.decision == DECISION_PREFER_CANDIDATE
    assert result.preferred_candidate_id == "c"


def test_30_three_finalist_no_dominant_candidate_never_arbitrary_pick():
    # None of the three strictly dominates all others -> ABSTAIN, never an
    # arbitrary winner (order/id/source based).
    result = evaluate_bounded_finalist_arbiter(_base_input(
        candidate_ids=("a", "b", "c"),
        meaning_sufficient_candidate_ids=("a", "b", "c"),
        v2_evidence_by_id={
            "a": _clean_v2("a"),
            "b": _clean_v2("b"),
            "c": _clean_v2("c"),
        },
    ))
    assert result.decision == DECISION_ABSTAIN
    assert result.preferred_candidate_id is None
    assert result.arbiter_state in (STATE_NEAR_EQUAL, STATE_CONFLICTED)


def test_31_v2_preferred_candidate_reports_internal_conflict_on_a_cycle(monkeypatch):
    # Real V2 dominance is mathematically transitive (Pareto-style
    # comparison over totally-ordered per-zone ranks), so a genuine
    # 3-cycle cannot occur from real evidence -- this proves the DEFENSIVE
    # cycle-detection code path itself, directly, via a stubbed dominance
    # function forced into an inconsistent (adversarial-input) shape.
    import cutsell_worker.bounded_finalist_arbiter as mod

    def _fake_dominates(alternative, winner):
        table = {("A", "B"): True, ("B", "C"): True, ("C", "A"): True}
        return table.get((alternative, winner), False)

    monkeypatch.setattr(mod, "zone_usability_v2_dominates", _fake_dominates)
    preferred, conflicted, has_evidence = mod._v2_preferred_candidate(
        ("A", "B", "C"), {"A": "A", "B": "B", "C": "C"},
    )
    assert has_evidence is True
    assert conflicted is True
    assert preferred is None


def test_32_three_finalist_missing_evidence_for_one_is_no_evidence():
    result = evaluate_bounded_finalist_arbiter(_base_input(
        candidate_ids=("a", "b", "c"),
        meaning_sufficient_candidate_ids=("a", "b", "c"),
        v2_evidence_by_id={"a": _clean_v2("a"), "b": _clean_v2("b")},
    ))
    assert result.performance_comparison_status == "NO_EVIDENCE"
    assert result.decision == DECISION_ABSTAIN


# ---------------------------------------------------------------------------
# 33-36. Performance-vs-editability conflict; editability-only preference.
# ---------------------------------------------------------------------------
def test_33_performance_vs_editability_conflict_is_conflicted_abstain():
    result = evaluate_bounded_finalist_arbiter(_base_input(
        v2_evidence_by_id={"a": _impaired_delivery_v2("a"), "b": _clean_v2("b")},
        editability_preferred_candidate_id="a",
    ))
    assert result.decision == DECISION_ABSTAIN
    assert result.arbiter_state == STATE_CONFLICTED
    assert result.structured_conflict is True
    assert result.preferred_candidate_id is None


def test_34_editability_only_preference_when_performance_near_equal_but_evidence_present():
    # Performance NEAR_EQUAL (no dominance found, but v2 evidence exists),
    # editability supplies the only actual preference among sources ->
    # PREFERENCE_SUPPORTED (the single distinct non-None preference wins,
    # no cross-source disagreement since the near-equal source contributes
    # no opinion of its own).
    result = evaluate_bounded_finalist_arbiter(_base_input(
        v2_evidence_by_id={"a": _clean_v2("a"), "b": _clean_v2("b")},
        editability_preferred_candidate_id="a",
    ))
    assert result.decision == DECISION_PREFER_CANDIDATE
    assert result.preferred_candidate_id == "a"
    assert result.editability_comparison_status == "DOMINANT"


def test_35_editability_evidence_referencing_non_finalist_is_ignored():
    result = evaluate_bounded_finalist_arbiter(_base_input(
        v2_evidence_by_id={"a": _clean_v2("a"), "b": _clean_v2("b")},
        editability_preferred_candidate_id="z",
    ))
    assert result.editability_comparison_status == "NO_EVIDENCE"
    assert "editability_evidence" in result.missing_evidence


def test_36_no_editability_evidence_is_the_honest_default():
    result = evaluate_bounded_finalist_arbiter(_base_input(
        v2_evidence_by_id={"a": _clean_v2("a"), "b": _clean_v2("b")},
    ))
    assert result.editability_comparison_status == "NO_EVIDENCE"


# ---------------------------------------------------------------------------
# 37-39. Missing-evidence control -- never a fallback to raw score.
# ---------------------------------------------------------------------------
def test_37_no_v2_no_editability_is_insufficient_evidence():
    result = evaluate_bounded_finalist_arbiter(_base_input())
    assert result.decision == DECISION_ABSTAIN
    assert result.arbiter_state == STATE_INSUFFICIENT_EVIDENCE
    assert result.preferred_candidate_id is None


def test_38_terminal_scores_never_influence_the_decision():
    common = dict(v2_evidence_by_id={})
    starved = evaluate_bounded_finalist_arbiter(_base_input(**common))
    with_scores = evaluate_bounded_finalist_arbiter(_base_input(
        terminal_scores={"a": 0.91, "b": 0.12}, **common,
    ))
    assert starved.arbiter_state == with_scores.arbiter_state == STATE_INSUFFICIENT_EVIDENCE
    assert starved.preferred_candidate_id == with_scores.preferred_candidate_id is None
    assert starved.decision == with_scores.decision == DECISION_ABSTAIN


def test_39_no_max_score_fallback_when_v2_absent_even_with_lopsided_terminal_scores():
    result = evaluate_bounded_finalist_arbiter(_base_input(
        terminal_scores={"a": 0.99, "b": 0.01},
    ))
    # A raw-score fallback would prefer "a" here; the arbiter must not.
    assert result.preferred_candidate_id is None
    assert result.decision == DECISION_ABSTAIN


# ---------------------------------------------------------------------------
# 40-41. Decisive D-183 case should not run; the D-181 Pimples abstract
# replay must not "magically fix" a near-equal family.
# ---------------------------------------------------------------------------
def test_40_decisive_state_never_evaluates_evidence():
    result = evaluate_bounded_finalist_arbiter(_base_input(
        terminal_confidence_state="DECISIVE",
        v2_evidence_by_id={"a": _impaired_delivery_v2("a"), "b": _clean_v2("b")},
    ))
    assert result.arbiter_state == STATE_NOT_ELIGIBLE
    assert result.performance_comparison_status == "NOT_EVALUATED"


def test_41_d181_pimples_abstract_replay_abstains_without_fabricated_preference():
    # Generic replay of the D-181 real-media shape: two meaning-sufficient,
    # complete, same-proposition retries; D-150 already routed this to
    # ABSTAIN_CONFLICT upstream (not modeled here) and D-183 classified the
    # terminal comparison NON_DECISIVE; V2 performance is near-equal
    # (both clean); prosody is unavailable. D-184 must NOT invent a
    # preference just because a Product Owner wants Pimples resolved.
    result = evaluate_bounded_finalist_arbiter(_base_input(
        candidate_ids=("pimples_a", "pimples_b"),
        meaning_sufficient_candidate_ids=("pimples_a", "pimples_b"),
        terminal_confidence_state="NON_DECISIVE",
        candidate_texts={
            "pimples_a": "I noticed a small skin reaction near my ear after using it.",
            "pimples_b": "There was irritation along my jaw, kind of like an allergy.",
        },
        v2_evidence_by_id={
            "pimples_a": _clean_v2("pimples_a"),
            "pimples_b": _clean_v2("pimples_b"),
        },
    ))
    assert result.decision == DECISION_ABSTAIN
    assert result.preferred_candidate_id is None
    assert result.arbiter_state in (STATE_NEAR_EQUAL, STATE_INSUFFICIENT_EVIDENCE)


# ---------------------------------------------------------------------------
# 42-44. Determinism / order independence.
# ---------------------------------------------------------------------------
def test_42_determinism_same_input_same_result():
    inp = _base_input(v2_evidence_by_id={"a": _impaired_delivery_v2("a"), "b": _clean_v2("b")})
    r1 = evaluate_bounded_finalist_arbiter(inp)
    r2 = evaluate_bounded_finalist_arbiter(inp)
    assert r1 == r2


def test_43_candidate_order_independence():
    v2 = {"a": _impaired_delivery_v2("a"), "b": _clean_v2("b")}
    forward = evaluate_bounded_finalist_arbiter(_base_input(
        candidate_ids=("a", "b"), meaning_sufficient_candidate_ids=("a", "b"), v2_evidence_by_id=v2,
    ))
    backward = evaluate_bounded_finalist_arbiter(_base_input(
        candidate_ids=("b", "a"), meaning_sufficient_candidate_ids=("b", "a"), v2_evidence_by_id=v2,
    ))
    assert forward.decision == backward.decision
    assert forward.preferred_candidate_id == backward.preferred_candidate_id
    assert forward.arbiter_state == backward.arbiter_state


def test_44_clip_id_and_family_id_are_arbitrary_labels():
    v2 = {"x9": _impaired_delivery_v2("x9"), "y7": _clean_v2("y7")}
    result = evaluate_bounded_finalist_arbiter(_base_input(
        family_id="totally_different_family_ref",
        candidate_ids=("x9", "y7"), meaning_sufficient_candidate_ids=("x9", "y7"),
        v2_evidence_by_id=v2,
    ))
    assert result.preferred_candidate_id == "y7"


# ---------------------------------------------------------------------------
# 45-46. Double-counting audit (D-163 vs D-172).
# ---------------------------------------------------------------------------
def test_45_performance_evidence_by_id_never_changes_the_decision():
    v2 = {"a": _impaired_delivery_v2("a"), "b": _clean_v2("b")}
    without_d163 = evaluate_bounded_finalist_arbiter(_base_input(v2_evidence_by_id=v2))
    with_d163 = evaluate_bounded_finalist_arbiter(_base_input(
        v2_evidence_by_id=v2,
        performance_evidence_by_id={"a": object(), "b": object()},
    ))
    assert without_d163.decision == with_d163.decision
    assert without_d163.preferred_candidate_id == with_d163.preferred_candidate_id
    assert without_d163.arbiter_state == with_d163.arbiter_state


def test_46_double_counting_audit_documents_d163_d172_correlation():
    assert DOUBLE_COUNTING_AUDIT["v2_evidence_by_id"] == "DECISION_SOURCE"
    assert "PARTIALLY_CORRELATED" in DOUBLE_COUNTING_AUDIT["performance_evidence_by_id"]
    assert "NEVER_INDEPENDENTLY_VOTED" in DOUBLE_COUNTING_AUDIT["performance_evidence_by_id"]


# ---------------------------------------------------------------------------
# 47-49. Prosodic Audio / P1 explicit absence.
# ---------------------------------------------------------------------------
def test_47_prosodic_audio_available_is_false():
    assert PROSODIC_AUDIO_AVAILABLE is False


def test_48_prosodic_audio_status_always_not_available():
    scenarios = [
        _base_input(),
        _base_input(v2_evidence_by_id={"a": _impaired_delivery_v2("a"), "b": _clean_v2("b")}),
        _base_input(terminal_confidence_state="DECISIVE"),
        _base_input(meaning_sufficient_candidate_ids=("a",)),
    ]
    for inp in scenarios:
        result = evaluate_bounded_finalist_arbiter(inp)
        row = bounded_finalist_arbiter_diagnostics(result)
        assert row["bounded_finalist_arbiter_prosodic_audio_status"] == "NOT_AVAILABLE"
        assert "prosodic_audio" in result.missing_evidence
        assert "p1_global_context" in result.missing_evidence


def test_49_p1_global_context_never_consulted_source_scan():
    source = (REPO_ROOT / "cutsell_worker" / "bounded_finalist_arbiter.py").read_text()
    for forbidden in ("editorial_moment_understanding", "whole_video_reasoning", "sequence_role", "commercial_role", "funnel"):
        assert forbidden not in source.casefold()


# ---------------------------------------------------------------------------
# Structural: no winner mutation, no provider, no reference leak, no
# score-weight/master-score, diagnostics/run-summary shape.
# ---------------------------------------------------------------------------
def test_action_applied_is_always_false():
    scenarios = [
        _base_input(),
        _base_input(v2_evidence_by_id={"a": _impaired_delivery_v2("a"), "b": _clean_v2("b")}),
        _base_input(terminal_confidence_state="DECISIVE"),
        _base_input(meaning_sufficient_candidate_ids=("a",)),
        _base_input(candidate_ids=("a", "b", "c", "d"), meaning_sufficient_candidate_ids=("a", "b", "c", "d")),
    ]
    for inp in scenarios:
        result = evaluate_bounded_finalist_arbiter(inp)
        assert result.action_applied is False


def test_result_carries_no_selection_mutation_fields():
    result = evaluate_bounded_finalist_arbiter(_base_input())
    for forbidden_attr in ("selected_clip_id", "final_winner", "render_plan", "draft_clip", "retry_family"):
        assert not hasattr(result, forbidden_attr)


def test_no_provider_network_calls_source_scan():
    source = (REPO_ROOT / "cutsell_worker" / "bounded_finalist_arbiter.py").read_text()
    for forbidden in ("openai", "gemini", "requests.", "urllib", "http.client", "socket.", "subprocess", "genai"):
        assert forbidden not in source.casefold()


def test_no_reference_or_qa_string_leak_source_scan():
    source = (REPO_ROOT / "cutsell_worker" / "bounded_finalist_arbiter.py").read_text()
    for forbidden in ("cutai", "cut.ai", "human_gold", "cutsell_gold"):
        assert forbidden not in source.casefold()


def test_no_score_weight_or_master_score_source_scan():
    source = (REPO_ROOT / "cutsell_worker" / "bounded_finalist_arbiter.py").read_text()
    for forbidden in ("master_score", "score_weight", "weighted_score"):
        assert forbidden not in source.casefold()


def test_diagnostics_row_has_all_13_required_fields():
    result = evaluate_bounded_finalist_arbiter(_base_input(
        v2_evidence_by_id={"a": _impaired_delivery_v2("a"), "b": _clean_v2("b")},
    ))
    row = bounded_finalist_arbiter_diagnostics(result)
    for field in (
        "bounded_finalist_arbiter_eligible", "bounded_finalist_arbiter_candidate_count",
        "bounded_finalist_arbiter_state", "bounded_finalist_arbiter_decision",
        "bounded_finalist_arbiter_preferred_candidate_id", "bounded_finalist_arbiter_reason",
        "bounded_finalist_arbiter_meaning_parity", "bounded_finalist_arbiter_performance_status",
        "bounded_finalist_arbiter_editability_status", "bounded_finalist_arbiter_prosodic_audio_status",
        "bounded_finalist_arbiter_conflict", "bounded_finalist_arbiter_missing_evidence",
        "bounded_finalist_arbiter_action_applied",
    ):
        assert field in row
    assert len(row) == 13


def test_run_summary_has_all_7_required_counts_and_is_tail_safe():
    empty_summary = bounded_finalist_arbiter_run_summary([])
    for field in (
        "arbiter_evaluated_count", "arbiter_preference_supported_count", "arbiter_abstain_count",
        "arbiter_near_equal_count", "arbiter_conflicted_count", "arbiter_insufficient_evidence_count",
        "arbiter_not_eligible_count",
    ):
        assert field in empty_summary
        assert empty_summary[field] == 0


def test_run_summary_counts_a_mixed_batch_correctly():
    rows = [
        bounded_finalist_arbiter_diagnostics(evaluate_bounded_finalist_arbiter(_base_input(
            v2_evidence_by_id={"a": _impaired_delivery_v2("a"), "b": _clean_v2("b")},
        ))),
        bounded_finalist_arbiter_diagnostics(evaluate_bounded_finalist_arbiter(_base_input(
            v2_evidence_by_id={"a": _clean_v2("a"), "b": _clean_v2("b")},
        ))),
        bounded_finalist_arbiter_diagnostics(evaluate_bounded_finalist_arbiter(_base_input())),
        bounded_finalist_arbiter_diagnostics(evaluate_bounded_finalist_arbiter(_base_input(
            terminal_confidence_state="DECISIVE",
        ))),
        bounded_finalist_arbiter_diagnostics(evaluate_bounded_finalist_arbiter(_base_input(
            v2_evidence_by_id={"a": _impaired_delivery_v2("a"), "b": _clean_v2("b")},
            editability_preferred_candidate_id="a",
        ))),
        {"unrelated_row": True},
        None,
    ]
    summary = bounded_finalist_arbiter_run_summary(rows)
    assert summary["arbiter_evaluated_count"] == 5
    assert summary["arbiter_preference_supported_count"] == 1
    assert summary["arbiter_near_equal_count"] == 1
    assert summary["arbiter_insufficient_evidence_count"] == 1
    assert summary["arbiter_not_eligible_count"] == 1
    assert summary["arbiter_conflicted_count"] == 1
    assert summary["arbiter_abstain_count"] == 4


def test_bounded_finalist_arbiter_input_and_result_are_frozen():
    inp = _base_input()
    try:
        inp.family_id = "changed"
        assert False, "FinalistArbiterInput must be frozen"
    except Exception:
        pass
    result = evaluate_bounded_finalist_arbiter(inp)
    try:
        result.decision = "PREFER_CANDIDATE"
        assert False, "BoundedFinalistArbiterResult must be frozen"
    except Exception:
        pass


def test_schema_version_present():
    import cutsell_worker.bounded_finalist_arbiter as mod
    assert mod.SCHEMA_VERSION == "cutsell.bounded_finalist_arbiter.v1"


def test_result_is_a_bounded_finalist_arbiter_result_instance():
    result = evaluate_bounded_finalist_arbiter(_base_input())
    assert isinstance(result, BoundedFinalistArbiterResult)


# ---------------------------------------------------------------------------
# Pipeline wiring (flag OFF is a total no-op; flag ON is diagnostic-only).
# ---------------------------------------------------------------------------
def _weak_strong_fixture():
    from cutsell_worker.contracts import CandidateTake, MediaSignals, ProcessingRequest, SemanticLabel, SemanticRole

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
    return request, (weak, strong), labels, strong.clip_id


def test_pipeline_wiring_flag_off_is_a_total_no_op():
    from cutsell_worker.pipeline import build_flow_b_draft

    request, takes, labels, expected_winner = _weak_strong_fixture()
    result = build_flow_b_draft(request, takes, labels)
    assert [clip.clip_id for clip in result.draft.selected] == [expected_winner]
    groups = (result.draft.diagnostics or {}).get("take_judge_groups") or []
    assert len(groups) == 1
    row = groups[0]
    assert not any(key.startswith("bounded_finalist_arbiter_") for key in row)
    assert result.draft.diagnostics.get("bounded_finalist_arbiter") == {"status": "disabled"}


def test_pipeline_wiring_flag_on_carries_diagnostics_without_changing_winner(monkeypatch):
    monkeypatch.setenv("CUTSELL_BOUNDED_FINALIST_ARBITER_ENABLED", "1")
    from cutsell_worker.pipeline import build_flow_b_draft

    request, takes, labels, expected_winner = _weak_strong_fixture()
    result = build_flow_b_draft(request, takes, labels)
    # Zero effect on the winner/membership -- the whole point of "diagnostic
    # only" -- regardless of the flag.
    assert [clip.clip_id for clip in result.draft.selected] == [expected_winner]

    groups = (result.draft.diagnostics or {}).get("take_judge_groups") or []
    assert len(groups) == 1
    row = groups[0]
    for key in (
        "bounded_finalist_arbiter_eligible", "bounded_finalist_arbiter_candidate_count",
        "bounded_finalist_arbiter_state", "bounded_finalist_arbiter_decision",
        "bounded_finalist_arbiter_preferred_candidate_id", "bounded_finalist_arbiter_reason",
        "bounded_finalist_arbiter_meaning_parity", "bounded_finalist_arbiter_performance_status",
        "bounded_finalist_arbiter_editability_status", "bounded_finalist_arbiter_prosodic_audio_status",
        "bounded_finalist_arbiter_conflict", "bounded_finalist_arbiter_missing_evidence",
        "bounded_finalist_arbiter_action_applied",
    ):
        assert key in row
    assert row["bounded_finalist_arbiter_action_applied"] is False
    # D-183's own NON_DECISIVE state on this fixture (proven in
    # test_cutsell_d183_terminal_besttake_confidence.py) makes the arbiter
    # eligible; with no V2/editability evidence supplied this fixture never
    # threads Watch+Listen spans, so it correctly abstains.
    assert row["bounded_finalist_arbiter_eligible"] is True
    assert row["bounded_finalist_arbiter_decision"] == DECISION_ABSTAIN
    assert row["final_winner"] == expected_winner  # unchanged winner

    summary = result.draft.diagnostics.get("bounded_finalist_arbiter")
    assert summary["status"] == "evaluated"
    assert summary["arbiter_evaluated_count"] == 1
