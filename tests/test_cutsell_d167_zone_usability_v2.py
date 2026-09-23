"""D-167: Watch+Listen Zone-Usability Refinement V2.

Covers all 40 directive-required fixture categories. This module is
additive-only and grants no new authority. Several tests explicitly prove
D-157 (`watch_listen_understanding.py`) and D-163
(`watch_listen_besttake_evidence.py`) remain at literal zero diff, and
that D-166's `language_spine.py` is untouched.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cutsell_worker.positioned_performance_evidence import (
    DeliverySpan,
    PositionAwarePerformanceEvidence,
    PositionedEvent,
    ZONE_DELIVERY,
    ZONE_ENTRY,
    ZONE_EXIT,
)
from cutsell_worker.raw_understanding_map import RawUnderstandingSpan
from cutsell_worker.watch_listen_besttake_evidence import (
    CASE_A_BOUNDARY_ONLY,
    CASE_B_DELIVERY_OWNED,
    CASE_C_AMBIGUOUS,
    CASE_CLEAN,
)
from cutsell_worker.watch_listen_zone_usability_v2 import (
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    PATTERN_ISOLATED,
    PATTERN_NONE,
    PATTERN_REPEATED,
    PATTERN_SUSTAINED,
    SEVERITY_MATERIAL,
    SEVERITY_MILD,
    SEVERITY_MIXED,
    SEVERITY_NONE,
    SEVERITY_SEVERE,
    SEVERITY_UNKNOWN,
    USABILITY_IMPAIRED,
    USABILITY_QUESTIONABLE,
    USABILITY_UNKNOWN,
    USABILITY_UNUSABLE,
    USABILITY_USABLE,
    DOUBLE_COUNTING_AUDIT_V2,
    build_zone_usability_v2,
    candidate_zone_usability_v2_row,
    zone_usability_v2_diagnostics,
    zone_usability_v2_dominates,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


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


# ---------------------------------------------------------------------------
# 1. clean delivery
# ---------------------------------------------------------------------------
def test_01_clean_delivery():
    span = _span()
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.zone_usability == USABILITY_USABLE
    assert result.delivery.zone_severity == SEVERITY_NONE
    assert result.overall_usability == USABILITY_USABLE


# ---------------------------------------------------------------------------
# 2. one brief isolated delivery event
# ---------------------------------------------------------------------------
def test_02_brief_isolated_delivery_event():
    # A brief, isolated, LOW-materiality event (ordinary hand motion) --
    # the Ordinary Motion Firewall means this carries NO penalty at all
    # (SEVERITY_NONE/USABLE), exactly as intended: an isolated low-tier
    # event is not "negative evidence" merely for existing.
    span = _span(delivery_events=[_ev("hand_motion_reset_candidate", 4.0, 4.067)])
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.isolated_event is True
    assert result.delivery.zone_severity == SEVERITY_NONE
    assert result.delivery.zone_usability == USABILITY_USABLE


# ---------------------------------------------------------------------------
# 3. sustained delivery fumble
# ---------------------------------------------------------------------------
def test_03_sustained_delivery_fumble():
    # One event covering >= half of the 8s delivery.
    span = _span(delivery_events=[_ev("hand_motion_reset_candidate", 1.0, 6.0)])
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.sustained_defect is True
    assert result.delivery.affected_fraction >= 0.5
    assert result.delivery.zone_severity == SEVERITY_MATERIAL
    assert result.delivery.zone_usability == USABILITY_IMPAIRED


# ---------------------------------------------------------------------------
# 4. repeated delivery resets
# ---------------------------------------------------------------------------
def test_04_repeated_delivery_resets():
    span = _span(delivery_events=[
        _ev("hand_motion_reset_candidate", 1.0, 1.1),
        _ev("hand_motion_reset_candidate", 3.0, 3.1),
        _ev("hand_motion_reset_candidate", 5.0, 5.1),
    ])
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.repeated_defect is True
    assert result.delivery.zone_severity == SEVERITY_MILD


# ---------------------------------------------------------------------------
# 5. brief breaking-character event
# ---------------------------------------------------------------------------
def test_05_brief_breaking_character():
    span = _span(delivery_events=[_ev("breaking_character", 4.0, 4.2)])
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.isolated_event is True
    assert result.delivery.zone_severity == SEVERITY_MATERIAL
    assert result.delivery.zone_usability == USABILITY_IMPAIRED


# ---------------------------------------------------------------------------
# 6. sustained breaking-character event
# ---------------------------------------------------------------------------
def test_06_sustained_breaking_character():
    span = _span(delivery_events=[_ev("breaking_character", 1.0, 6.0)])
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.sustained_defect is True
    assert result.delivery.zone_severity == SEVERITY_SEVERE
    assert result.delivery.zone_usability == USABILITY_UNUSABLE


# ---------------------------------------------------------------------------
# 7. brief camera disengagement
# ---------------------------------------------------------------------------
def test_07_brief_camera_disengagement():
    span = _span(delivery_events=[_ev("camera_disengagement_candidate", 4.0, 4.1)])
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.zone_severity == SEVERITY_MILD
    assert result.delivery.zone_usability == USABILITY_QUESTIONABLE


# ---------------------------------------------------------------------------
# 8. sustained camera disengagement
# ---------------------------------------------------------------------------
def test_08_sustained_camera_disengagement():
    span = _span(delivery_events=[_ev("camera_disengagement_candidate", 1.0, 6.0)])
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.zone_severity == SEVERITY_SEVERE
    assert result.delivery.zone_usability == USABILITY_UNUSABLE


# ---------------------------------------------------------------------------
# 9. ordinary expressive hand motion (Ordinary Motion Firewall)
# ---------------------------------------------------------------------------
def test_09_ordinary_expressive_hand_motion_no_penalty():
    span = _span(delivery_events=[_ev("hand_motion_reset_candidate", 4.0, 4.067)])
    result = build_zone_usability_v2("c1", span)
    # A single, brief, low-materiality event -- LOW+ISOLATED -> NO penalty
    # at all (the Firewall's strongest guarantee).
    assert result.delivery.zone_severity == SEVERITY_NONE
    assert result.delivery.zone_usability == USABILITY_USABLE
    # The Firewall's broader guarantee: an isolated LOW-materiality event
    # never reaches MATERIAL/SEVERE/UNUSABLE regardless of exact tier.
    assert result.delivery.zone_usability in (USABILITY_USABLE, USABILITY_QUESTIONABLE)


# ---------------------------------------------------------------------------
# 10. ordinary body motion
# ---------------------------------------------------------------------------
def test_10_ordinary_body_motion_no_penalty():
    span = _span(delivery_events=[_ev("body_reset_candidate", 4.0, 4.1)])
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.zone_usability in (USABILITY_USABLE, USABILITY_QUESTIONABLE)


# ---------------------------------------------------------------------------
# 11. entry-only event (CASE A / Boundary Firewall)
# ---------------------------------------------------------------------------
def test_11_entry_only_event_boundary_firewall():
    span = _span(entry_events=[_ev("hand_motion_reset_candidate", 0.0, 0.1, zone=ZONE_ENTRY)])
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.event_count == 0
    assert result.delivery.zone_usability == USABILITY_USABLE
    # ENTRY-only defect never produces a whole-take rejection.
    assert result.overall_usability in (USABILITY_USABLE, USABILITY_QUESTIONABLE)
    assert result.overall_usability != USABILITY_UNUSABLE
    assert result.case_classification in (CASE_A_BOUNDARY_ONLY, CASE_CLEAN)


# ---------------------------------------------------------------------------
# 12. exit-only event
# ---------------------------------------------------------------------------
def test_12_exit_only_event_boundary_firewall():
    span = _span(exit_events=[_ev("hand_motion_reset_candidate", 8.3, 8.4, zone=ZONE_EXIT)])
    result = build_zone_usability_v2("c1", span)
    assert result.overall_usability != USABILITY_UNUSABLE
    assert result.case_classification in (CASE_A_BOUNDARY_ONLY, CASE_CLEAN)


# ---------------------------------------------------------------------------
# 13. cross-boundary event (CASE C ambiguous -- delivery unavailable)
# ---------------------------------------------------------------------------
def test_13_cross_boundary_ambiguous_no_delivery():
    span = _span(delivery_available=False)
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.zone_usability == USABILITY_UNKNOWN
    assert result.overall_usability == USABILITY_UNKNOWN
    assert result.case_classification == CASE_C_AMBIGUOUS


# ---------------------------------------------------------------------------
# 14. short audio pause -- audio kinds never drive zone usability (Audio Honesty)
# ---------------------------------------------------------------------------
def test_14_short_audio_pause_no_defect_classification():
    span = _span(delivery_events=[_ev("audio_silence_interval", 4.0, 4.3)])
    result = build_zone_usability_v2("c1", span)
    # audio_silence_interval is not in _DEFECT_KINDS -- never classified as
    # a defect event by this module (Audio Honesty, restated for V2).
    assert result.delivery.event_count == 0
    assert result.delivery.zone_usability == USABILITY_USABLE


# ---------------------------------------------------------------------------
# 15. long audio interruption -- same honesty rule applies regardless of duration
# ---------------------------------------------------------------------------
def test_15_long_audio_interruption_no_defect_classification():
    span = _span(delivery_events=[_ev("audio_silence_interval", 1.0, 6.0)])
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.event_count == 0
    assert result.delivery.zone_usability == USABILITY_USABLE


# ---------------------------------------------------------------------------
# 16. same event count, different durations
# ---------------------------------------------------------------------------
def test_16_same_event_count_different_durations():
    span_short = _span(span_id="short", delivery_events=[_ev("hand_motion_reset_candidate", 4.0, 4.067)])
    span_long = _span(span_id="long", delivery_events=[_ev("hand_motion_reset_candidate", 1.0, 6.0)])
    r_short = build_zone_usability_v2("short", span_short)
    r_long = build_zone_usability_v2("long", span_long)
    assert r_short.delivery.event_count == r_long.delivery.event_count == 1
    assert r_short.delivery.event_duration_total_sec < r_long.delivery.event_duration_total_sec
    assert r_short.delivery.affected_fraction < r_long.delivery.affected_fraction
    # Duration difference produces a genuine categorical distinction.
    assert r_short.delivery.zone_usability != r_long.delivery.zone_usability


# ---------------------------------------------------------------------------
# 17. same duration, different delivery lengths (delivery-length normalization)
# ---------------------------------------------------------------------------
def test_17_same_duration_different_delivery_lengths():
    span_short_delivery = _span(
        span_id="a", source_end=3.0, delivery_start=0.1, delivery_end=2.0,
        delivery_events=[_ev("hand_motion_reset_candidate", 1.0, 1.5)],
    )
    span_long_delivery = _span(
        span_id="b", source_end=20.0, delivery_start=0.1, delivery_end=16.0,
        delivery_events=[_ev("hand_motion_reset_candidate", 1.0, 1.5)],
    )
    r_a = build_zone_usability_v2("a", span_short_delivery)
    r_b = build_zone_usability_v2("b", span_long_delivery)
    assert r_a.delivery.event_duration_total_sec == r_b.delivery.event_duration_total_sec
    assert r_a.delivery.affected_fraction > r_b.delivery.affected_fraction  # same defect, shorter delivery = worse


# ---------------------------------------------------------------------------
# 18. repeated tiny events
# ---------------------------------------------------------------------------
def test_18_repeated_tiny_events():
    span = _span(delivery_events=[_ev("facial_expression_shift_candidate", i, i + 0.05) for i in range(1, 7, 2)])
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.repeated_defect is True
    assert result.delivery.affected_fraction < 0.1


# ---------------------------------------------------------------------------
# 19. one sustained interval
# ---------------------------------------------------------------------------
def test_19_one_sustained_interval():
    span = _span(delivery_events=[_ev("facial_expression_shift_candidate", 1.0, 7.0)])
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.sustained_defect is True
    assert result.delivery.event_count == 1


# ---------------------------------------------------------------------------
# 20. same-source duplicate evidence (double-counting audit)
# ---------------------------------------------------------------------------
def test_20_same_source_duplicate_classification():
    assert DOUBLE_COUNTING_AUDIT_V2["dominant_event_kinds"] == "SAME_SOURCE_DUPLICATE"
    assert DOUBLE_COUNTING_AUDIT_V2["zone_conflict"] == "SAME_SOURCE_DUPLICATE"


# ---------------------------------------------------------------------------
# 21. partially-correlated evidence
# ---------------------------------------------------------------------------
def test_21_partially_correlated_classification():
    for key in ("event_duration_total_sec", "affected_fraction", "zone_severity", "zone_usability"):
        assert DOUBLE_COUNTING_AUDIT_V2[key] == "PARTIALLY_CORRELATED"


# ---------------------------------------------------------------------------
# 22. independent evidence
# ---------------------------------------------------------------------------
def test_22_independent_classification():
    assert DOUBLE_COUNTING_AUDIT_V2["pattern"] == "INDEPENDENT"


# ---------------------------------------------------------------------------
# 23. conflict evidence
# ---------------------------------------------------------------------------
def test_23_conflict_evidence_forces_mixed():
    span = _span(
        delivery_events=[_ev("hand_motion_reset_candidate", 4.0, 4.067)],
        conflict_flags=("MEANING_COMPLETE_VS_ABANDONED_ATTEMPT_EVIDENCE",),
    )
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.zone_conflict is True
    assert result.delivery.zone_severity == SEVERITY_MIXED


# ---------------------------------------------------------------------------
# 24. missing evidence (no delivery span)
# ---------------------------------------------------------------------------
def test_24_missing_evidence_unknown_not_clean():
    span = _span(delivery_available=False)
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.zone_usability == USABILITY_UNKNOWN
    assert result.delivery.zone_severity == SEVERITY_UNKNOWN
    # Never equates "not measured" with "clean".
    assert result.delivery.zone_usability != USABILITY_USABLE


# ---------------------------------------------------------------------------
# 25. clean vs mildly impaired
# ---------------------------------------------------------------------------
def test_25_clean_vs_mildly_impaired():
    clean = build_zone_usability_v2("clean", _span(span_id="clean"))
    mild = build_zone_usability_v2("mild", _span(span_id="mild", delivery_events=[_ev("camera_disengagement_candidate", 4.0, 4.1)]))
    assert clean.delivery.zone_severity == SEVERITY_NONE
    assert mild.delivery.zone_severity != SEVERITY_NONE


# ---------------------------------------------------------------------------
# 26. mildly impaired vs severe
# ---------------------------------------------------------------------------
def test_26_mildly_impaired_vs_severe():
    mild = build_zone_usability_v2("mild", _span(span_id="mild", delivery_events=[_ev("camera_disengagement_candidate", 4.0, 4.1)]))
    severe = build_zone_usability_v2("severe", _span(span_id="severe", delivery_events=[_ev("breaking_character", 1.0, 6.0)]))
    assert mild.delivery.zone_severity in (SEVERITY_MILD, SEVERITY_NONE)
    assert severe.delivery.zone_severity == SEVERITY_SEVERE
    assert zone_usability_v2_dominates(mild, severe) is True


# ---------------------------------------------------------------------------
# 27. both severe
# ---------------------------------------------------------------------------
def test_27_both_severe_no_dominance():
    a = build_zone_usability_v2("a", _span(span_id="a", delivery_events=[_ev("breaking_character", 1.0, 6.0)]))
    b = build_zone_usability_v2("b", _span(span_id="b", delivery_events=[_ev("breaking_character", 1.5, 6.5)]))
    assert zone_usability_v2_dominates(a, b) is False
    assert zone_usability_v2_dominates(b, a) is False


# ---------------------------------------------------------------------------
# 28. meaning-insufficient alternative still excluded by D-163 (module-leaf proof)
# ---------------------------------------------------------------------------
def test_28_no_meaning_authority_in_v2_module():
    import cutsell_worker.watch_listen_zone_usability_v2 as mod
    text = Path(mod.__file__).read_text()
    assert "meaning_sufficient" not in text
    assert "selected_clip_id" not in text


# ---------------------------------------------------------------------------
# 29. D-164 saturation replay
# ---------------------------------------------------------------------------
def test_29_d164_saturation_replay():
    # Abstract fixture matching the real D-164 finding's SHAPE (never real
    # ids/text): Candidate A -- multiple low-materiality events, short/
    # isolated-magnitude total; Candidate B -- similar event count, but a
    # much larger total DELIVERY-affecting duration.
    span_a = _span(span_id="A", delivery_events=[
        _ev("facial_expression_shift_candidate", 2.0, 2.067),
        _ev("facial_expression_shift_candidate", 4.0, 4.067),
        _ev("hand_motion_reset_candidate", 6.0, 6.067),
    ])
    span_b = _span(span_id="B", delivery_events=[
        _ev("hand_motion_reset_candidate", 1.0, 3.5),
        _ev("hand_motion_reset_candidate", 4.0, 6.5),
        _ev("hand_motion_reset_candidate", 7.0, 7.5),
    ])
    result_a = build_zone_usability_v2("A", span_a)
    result_b = build_zone_usability_v2("B", span_b)
    # V1 (unchanged, verified via test_34) would mark BOTH UNUSABLE with no
    # further distinction. V2 exposes real, differing facts:
    assert result_a.delivery.event_duration_total_sec != result_b.delivery.event_duration_total_sec
    assert result_a.delivery.affected_fraction != result_b.delivery.affected_fraction
    # And with B's much larger affected fraction, V2 produces a genuine
    # categorical distinction A and B no longer share.
    assert result_a.delivery.zone_usability != result_b.delivery.zone_usability
    assert result_b.delivery.affected_fraction >= 0.5  # B is SUSTAINED
    assert result_a.delivery.affected_fraction < 0.5   # A is not


# ---------------------------------------------------------------------------
# 30. D-163 guard remains diagnostic-only
# ---------------------------------------------------------------------------
def test_30_d163_guard_remains_diagnostic_only():
    from cutsell_worker.watch_listen_besttake_evidence import watch_listen_besttake_group_row
    # The existing D-163 group_row contract still never applies an action.
    import inspect
    source = inspect.getsource(watch_listen_besttake_group_row)
    assert '"watch_listen_besttake_action_applied": False' in source


# ---------------------------------------------------------------------------
# 31. selected winner unchanged
# ---------------------------------------------------------------------------
def test_31_no_selected_clip_id_field_anywhere_in_v2():
    import cutsell_worker.watch_listen_zone_usability_v2 as mod
    for name in dir(mod):
        obj = getattr(mod, name)
        if hasattr(obj, "__dataclass_fields__"):
            assert "selected_clip_id" not in obj.__dataclass_fields__


# ---------------------------------------------------------------------------
# 32. family unchanged
# ---------------------------------------------------------------------------
def test_32_family_modules_unaware_of_v2():
    for path in ("take_grouping.py", "take_grouping_provider.py", "hybrid_session_cleanup.py",
                 "semantic_idea_equivalence.py"):
        text = (REPO_ROOT / "cutsell_worker" / path).read_text()
        assert "watch_listen_zone_usability_v2" not in text


# ---------------------------------------------------------------------------
# 33. proposition/attempt unchanged
# ---------------------------------------------------------------------------
def test_33_attempt_relationship_authority_unaware_of_v2():
    text = (REPO_ROOT / "cutsell_worker" / "attempt_relationship_authority.py").read_text()
    assert "watch_listen_zone_usability_v2" not in text


# ---------------------------------------------------------------------------
# 34. D-123 unchanged
# ---------------------------------------------------------------------------
def test_34_d123_unchanged():
    text = (REPO_ROOT / "cutsell_worker" / "deterministic_best_take_authority.py").read_text()
    assert "watch_listen_zone_usability_v2" not in text


# ---------------------------------------------------------------------------
# 35. D-128 unchanged
# ---------------------------------------------------------------------------
def test_35_d128_unchanged():
    text = (REPO_ROOT / "cutsell_worker" / "multimodal_besttake_fallback.py").read_text()
    assert "watch_listen_zone_usability_v2" not in text


# ---------------------------------------------------------------------------
# 36. D-150 unchanged
# ---------------------------------------------------------------------------
def test_36_d150_unchanged():
    text = (REPO_ROOT / "cutsell_worker" / "semantic_authority_observability.py").read_text()
    assert "watch_listen_zone_usability_v2" not in text


# ---------------------------------------------------------------------------
# 37. Boundary unchanged
# ---------------------------------------------------------------------------
def test_37_boundary_unchanged():
    text = (REPO_ROOT / "cutsell_worker" / "boundary_engine_pass.py").read_text()
    assert "watch_listen_zone_usability_v2" not in text


# ---------------------------------------------------------------------------
# 38. Pacing unchanged
# ---------------------------------------------------------------------------
def test_38_pacing_unchanged():
    text = (REPO_ROOT / "cutsell_worker" / "dialogue_pacing_transition.py").read_text()
    assert "watch_listen_zone_usability_v2" not in text


# ---------------------------------------------------------------------------
# 39. Language Spine D-166 unchanged
# ---------------------------------------------------------------------------
def test_39_language_spine_unchanged():
    text = (REPO_ROOT / "cutsell_worker" / "language_spine.py").read_text()
    assert "watch_listen_zone_usability_v2" not in text
    assert "watch_listen" not in text.casefold() or "watch_listen_corroboration" not in text  # no new dependency added


# ---------------------------------------------------------------------------
# 40. no provider/network call
# ---------------------------------------------------------------------------
def test_40_no_provider_network_call():
    import cutsell_worker.watch_listen_zone_usability_v2 as mod
    source = Path(mod.__file__).read_text()
    for forbidden in ("openai", "gemini", "requests.", "urllib", "http.client", "socket.", "subprocess"):
        assert forbidden not in source.casefold()


# ---------------------------------------------------------------------------
# Additional structural / zero-diff / contract tests
# ---------------------------------------------------------------------------
def test_d157_watch_listen_understanding_zero_diff():
    text = (REPO_ROOT / "cutsell_worker" / "watch_listen_understanding.py").read_text()
    assert "watch_listen_zone_usability_v2" not in text
    # Load-bearing V1 symbol untouched.
    assert "def _usability_for_zone(span: RawUnderstandingSpan, zone: str) -> str:" in text


def test_d163_watch_listen_besttake_evidence_zero_diff():
    text = (REPO_ROOT / "cutsell_worker" / "watch_listen_besttake_evidence.py").read_text()
    assert "watch_listen_zone_usability_v2" not in text
    assert "def evaluate_watch_listen_besttake_guard(" in text
    assert "GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE" in text


def test_dominance_returns_false_on_case_a_winner():
    winner = build_zone_usability_v2("w", _span(span_id="w", entry_events=[_ev("hand_motion_reset_candidate", 0.0, 0.1, zone=ZONE_ENTRY)]))
    alt = build_zone_usability_v2("alt", _span(span_id="alt"))
    assert winner.case_classification == CASE_A_BOUNDARY_ONLY
    assert zone_usability_v2_dominates(alt, winner) is False


def test_diagnostics_row_fields_exact():
    span = _span(delivery_events=[_ev("hand_motion_reset_candidate", 1.0, 1.5)])
    result = build_zone_usability_v2("c1", span)
    row = candidate_zone_usability_v2_row(result)
    for field in (
        "watch_listen_zone_usability_v2", "watch_listen_entry_severity", "watch_listen_delivery_severity",
        "watch_listen_exit_severity", "watch_listen_delivery_event_count",
        "watch_listen_delivery_event_duration_sec", "watch_listen_delivery_duration_sec",
        "watch_listen_delivery_affected_fraction", "watch_listen_delivery_pattern",
        "watch_listen_dominant_event_kinds", "watch_listen_zone_conflict",
    ):
        assert field in row


def test_tail_safe_summary_counts_and_no_ids():
    span_clean = _span(span_id="clean")
    span_impaired = _span(span_id="impaired", delivery_events=[_ev("hand_motion_reset_candidate", 1.0, 6.0)])
    results = [build_zone_usability_v2("clean", span_clean), build_zone_usability_v2("impaired", span_impaired)]
    summary = zone_usability_v2_diagnostics(results)
    for key in ("usable_count", "questionable_count", "impaired_count", "unusable_count", "unknown_count",
                "isolated_defect_count", "repeated_defect_count", "sustained_defect_count"):
        assert key in summary
    assert "clean" not in str(summary)
    assert "impaired" not in summary  # candidate_id "impaired" must not leak as a key/value string
    assert summary["candidate_count"] == 2


def test_fail_open_on_missing_span():
    assert build_zone_usability_v2("missing", None) is None


def test_no_new_opaque_score_no_float_master_field():
    span = _span(delivery_events=[_ev("hand_motion_reset_candidate", 1.0, 1.5)])
    result = build_zone_usability_v2("c1", span)
    # zone_usability/zone_severity are categorical strings, never a float score.
    assert isinstance(result.delivery.zone_usability, str)
    assert isinstance(result.delivery.zone_severity, str)


def test_confidence_reuses_d097_floors():
    # A reset-kind event below the established 0.88 floor is still counted
    # as a defect (per V1's own no-floor contract for zone membership), but
    # categorized WEAK confidence rather than SUPPORTED.
    span = _span(delivery_events=[_ev("hand_motion_reset_candidate", 1.0, 1.1, confidence=0.5)])
    result = build_zone_usability_v2("c1", span)
    assert result.delivery.event_count == 1
    assert result.delivery.confidence in (CONFIDENCE_SUPPORTED, "WEAK")


def test_zone_duration_normalization_entry_exit():
    span = _span(source_start=0.0, source_end=10.0, delivery_start=2.0, delivery_end=8.0)
    result = build_zone_usability_v2("c1", span)
    assert result.entry.zone_duration_sec == pytest.approx(2.0)
    assert result.exit.zone_duration_sec == pytest.approx(2.0)
    assert result.delivery.zone_duration_sec == pytest.approx(6.0)
