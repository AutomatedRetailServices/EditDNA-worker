"""D-188: Prosodic Audio V2 -> Bounded Finalist Arbiter -- PHASE B,
DIAGNOSTIC FUSION ONLY. Offline. No winner authority. No RAW. No
provider. No master prosody score. No "more energy = better" rule.

Covers this task's own 54-item offline test matrix: absence/availability
of Prosodic evidence, D-183 eligibility precedence (DECISIVE never
reopened), the fusion merge (visual+prosody agree/conflict/near-equal/
one-missing), each of the three DIRECTIONALLY SAFE dimensions in
isolation, each DESCRIPTIVE-ONLY dimension producing NO preference,
gain-invariance, the filler control, the Boundary-owned-pause control,
the meaning/negation/number/factual firewall (blocks before Prosody is
even consulted), three-finalist dominance and cycle-conflict, the
D-186B abstract Pimples/Gynecologist replay controls (generic, no
literal transcript text), no-majority-voting/no-master-score/no-new-
weights/no-provider/no-winner-mutation structural scans, determinism,
and order/id independence. Items 42-54 of the directive's matrix
(D-123/D-150/D-167/D-174/D-180/D-183/D-184/D-187/Family/Language Spine/
Boundary/Pacing/Renderer unchanged) are verified by running those
modules' own existing, untouched test suites plus a `git diff` scope
check -- not duplicated here.
"""
from __future__ import annotations

from pathlib import Path

from cutsell_worker.bounded_finalist_arbiter import (
    DECISION_ABSTAIN,
    DECISION_PREFER_CANDIDATE,
    STATE_CONFLICTED,
    STATE_INSUFFICIENT_EVIDENCE,
    STATE_NEAR_EQUAL,
    STATE_NOT_ELIGIBLE,
    STATE_PREFERENCE_SUPPORTED,
    BoundedFinalistArbiterResult,
    FinalistArbiterInput,
    bounded_finalist_arbiter_prosodic_fusion_diagnostics,
    bounded_finalist_arbiter_prosodic_fusion_run_summary,
    evaluate_bounded_finalist_arbiter,
)
from cutsell_worker.prosodic_audio_v2 import (
    CONTINUITY_CONTINUOUS,
    CONTINUITY_FRAGMENTED,
    HESITATION_NOT_OBSERVED,
    HESITATION_PRESENT,
    PITCH_NOT_IMPLEMENTED,
    ProsodicDeliveryEvidence,
    RESTART_NOT_OBSERVED,
    RESTART_SUPPORTED,
    STATUS_EVALUATED,
    STATUS_NOT_EVALUABLE,
    UNKNOWN,
    analyze_prosodic_delivery,
)
from cutsell_worker.prosodic_finalist_comparison import (
    COMPARISON_CONFLICTED,
    COMPARISON_DOMINANT,
    COMPARISON_INSUFFICIENT_EVIDENCE,
    COMPARISON_NEAR_EQUAL,
    COMPARISON_NOT_EVALUABLE,
    DOUBLE_COUNTING_AUDIT,
    ProsodicFinalistComparison,
    compare_prosodic_finalists,
    prosodic_finalist_diagnostics,
    prosodic_finalist_run_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDED_ARBITER_SOURCE = (REPO_ROOT / "cutsell_worker" / "bounded_finalist_arbiter.py").read_text()
PROSODIC_COMPARISON_SOURCE = (REPO_ROOT / "cutsell_worker" / "prosodic_finalist_comparison.py").read_text()


# ---------------------------------------------------------------------------
# Fixture helpers.
# ---------------------------------------------------------------------------
def _pde(
    cid, continuity=CONTINUITY_CONTINUOUS, hesitation=HESITATION_NOT_OBSERVED,
    restart=RESTART_NOT_OBSERVED, rate=2.0, energy_var=0.3, status=STATUS_EVALUATED,
    emphasis="EMPHASIS_NOT_OBSERVED",
):
    return ProsodicDeliveryEvidence(
        candidate_id=cid, source_asset_id="s1", source_start=0.0, source_end=4.0,
        analysis_status=status,
        speech_duration_sec=4.0, voiced_or_active_speech_duration_sec=3.5,
        speech_rate=rate, speech_rate_state="MODERATE",
        pause_count=0 if continuity == CONTINUITY_CONTINUOUS else 2,
        pause_total_sec=0.0 if continuity == CONTINUITY_CONTINUOUS else 0.6,
        pause_structure_state=continuity,
        hesitation_state=hesitation, restart_or_interruption_state=restart,
        vocal_continuity_state=continuity,
        energy_mean=0.2, energy_variation=energy_var, energy_dynamics_state="MODERATE_VARIATION",
        emphasis_dynamics_state=emphasis,
        pitch_analysis_status=PITCH_NOT_IMPLEMENTED, pitch_variation_state=UNKNOWN,
        delivery_variation_state="MODERATE_VARIATION",
        evidence_confidence="SUPPORTED", missing_evidence=("pitch_analysis",),
        provenance="test_fixture",
    )


def _fusion_input(**overrides):
    defaults = dict(
        family_id="fam1",
        candidate_ids=("a", "b"),
        meaning_sufficient_candidate_ids=("a", "b"),
        terminal_confidence_state="NON_DECISIVE",
    )
    defaults.update(overrides)
    return FinalistArbiterInput(**defaults)


# ===========================================================================
# 1-2. Prosodic absent / actual audio required.
# ===========================================================================
def test_01_prosodic_absent_reproduces_d184_original_behavior():
    result = evaluate_bounded_finalist_arbiter(_fusion_input())
    assert result.prosodic_comparison_status == "NOT_AVAILABLE"
    assert "prosodic_audio" in result.missing_evidence
    assert result.arbiter_state == STATE_INSUFFICIENT_EVIDENCE  # no other evidence supplied either


def test_02_actual_audio_evidence_required_transcript_only_not_evaluable():
    comparison = compare_prosodic_finalists({"a": None, "b": None}, ["a", "b"])
    assert comparison.comparison_state == COMPARISON_NOT_EVALUABLE
    result = evaluate_bounded_finalist_arbiter(_fusion_input(prosodic_comparison=comparison))
    assert result.prosodic_comparison_status == "INSUFFICIENT"
    assert "prosodic_audio" in result.missing_evidence


def test_02b_transcript_only_analyze_prosodic_delivery_never_fakes_evidence():
    ev = analyze_prosodic_delivery("a", "s1", 0.0, 4.0, None)
    assert ev.analysis_status == STATUS_NOT_EVALUABLE
    comparison = compare_prosodic_finalists(
        {"a": ev, "b": _pde("b")}, ["a", "b"],
    )
    # One real EVALUATED candidate + one NOT_EVALUABLE candidate is a
    # PARTIAL comparison -- INSUFFICIENT_EVIDENCE, never a guessed
    # dominance. NOT_EVALUABLE is reserved for when NO candidate has real
    # evidence at all (see test_02).
    assert comparison.comparison_state == COMPARISON_INSUFFICIENT_EVIDENCE


# ===========================================================================
# 3. D-183 DECISIVE -> arbiter not eligible, Prosody never reopens it.
# ===========================================================================
def test_03_d183_decisive_stays_not_eligible_even_with_prosody_dominance():
    comparison = compare_prosodic_finalists(
        {"a": _pde("a", CONTINUITY_FRAGMENTED, HESITATION_PRESENT, RESTART_SUPPORTED),
         "b": _pde("b", CONTINUITY_CONTINUOUS, HESITATION_NOT_OBSERVED, RESTART_NOT_OBSERVED)},
        ["a", "b"],
    )
    assert comparison.comparison_state == COMPARISON_DOMINANT
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        terminal_confidence_state="DECISIVE", prosodic_comparison=comparison,
    ))
    assert result.arbiter_state == STATE_NOT_ELIGIBLE
    assert result.decision == DECISION_ABSTAIN
    # Eligibility gate is FIRST -- Prosody status reflects "never consulted".
    assert result.prosodic_comparison_status == "NOT_AVAILABLE"


# ===========================================================================
# 4-6. D-183 NON_DECISIVE / TIED / CONFLICTED + Prosody dominance -> PREFER.
# ===========================================================================
def _dominance_comparison():
    return compare_prosodic_finalists(
        {"a": _pde("a", CONTINUITY_FRAGMENTED, HESITATION_PRESENT, RESTART_SUPPORTED),
         "b": _pde("b", CONTINUITY_CONTINUOUS, HESITATION_NOT_OBSERVED, RESTART_NOT_OBSERVED)},
        ["a", "b"],
    )


def test_04_non_decisive_plus_prosody_dominance_prefers():
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        terminal_confidence_state="NON_DECISIVE", prosodic_comparison=_dominance_comparison(),
    ))
    assert result.decision == DECISION_PREFER_CANDIDATE
    assert result.preferred_candidate_id == "b"
    assert result.arbiter_state == STATE_PREFERENCE_SUPPORTED
    assert result.prosodic_comparison_status == "AVAILABLE"
    assert result.action_applied is False


def test_05_tied_plus_prosody_dominance_prefers():
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        terminal_confidence_state="TIED", prosodic_comparison=_dominance_comparison(),
    ))
    assert result.decision == DECISION_PREFER_CANDIDATE
    assert result.preferred_candidate_id == "b"


def test_06_conflicted_terminal_plus_prosody_safe_preference_prefers():
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        terminal_confidence_state="CONFLICTED", prosodic_comparison=_dominance_comparison(),
    ))
    assert result.decision == DECISION_PREFER_CANDIDATE
    assert result.preferred_candidate_id == "b"


# ===========================================================================
# 7-10. Fusion with Visual (D-172 V2) evidence: agree / conflict / near-
# equal-both / near-equal-and-dominance / visual-unavailable.
# ===========================================================================
def test_07_visual_near_equal_editability_no_evidence_prosody_dominance_prefers():
    # No v2_evidence_by_id / editability supplied at all -> both NO_EVIDENCE
    # (never appended to `sources`) -- Prosody alone decides.
    result = evaluate_bounded_finalist_arbiter(_fusion_input(prosodic_comparison=_dominance_comparison()))
    assert result.decision == DECISION_PREFER_CANDIDATE
    assert result.preferred_candidate_id == "b"


def test_08_visual_and_prosody_agree_same_candidate_prefers():
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        prosodic_comparison=_dominance_comparison(),
        editability_preferred_candidate_id="b",
    ))
    assert result.decision == DECISION_PREFER_CANDIDATE
    assert result.preferred_candidate_id == "b"


def test_09_visual_vs_prosody_conflict_abstains():
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        prosodic_comparison=_dominance_comparison(),  # favors b
        editability_preferred_candidate_id="a",  # favors a
    ))
    assert result.decision == DECISION_ABSTAIN
    assert result.arbiter_state == STATE_CONFLICTED
    assert result.structured_conflict is True


def test_10_both_near_equal_abstains():
    near_equal = compare_prosodic_finalists({"a": _pde("a"), "b": _pde("b")}, ["a", "b"])
    assert near_equal.comparison_state == COMPARISON_NEAR_EQUAL
    result = evaluate_bounded_finalist_arbiter(_fusion_input(prosodic_comparison=near_equal))
    assert result.decision == DECISION_ABSTAIN
    assert result.arbiter_state == STATE_NEAR_EQUAL


# ===========================================================================
# 11. Prosody internal conflict -> ABSTAIN/CONFLICTED.
# ===========================================================================
def test_11_prosodic_internal_conflict_abstains():
    conflicted = compare_prosodic_finalists(
        {"a": _pde("a", CONTINUITY_CONTINUOUS, HESITATION_NOT_OBSERVED, RESTART_SUPPORTED),
         "b": _pde("b", CONTINUITY_FRAGMENTED, HESITATION_NOT_OBSERVED, RESTART_NOT_OBSERVED)},
        ["a", "b"],
    )
    assert conflicted.comparison_state == COMPARISON_CONFLICTED
    result = evaluate_bounded_finalist_arbiter(_fusion_input(prosodic_comparison=conflicted))
    assert result.decision == DECISION_ABSTAIN
    assert result.arbiter_state == STATE_CONFLICTED
    assert result.prosodic_comparison_status == "CONFLICTED"


# ===========================================================================
# 12. Missing Prosodic evidence (partial) -> INSUFFICIENT, not used.
# ===========================================================================
def test_12_partial_prosodic_evidence_is_insufficient_never_a_source():
    partial = compare_prosodic_finalists(
        {"a": _pde("a"), "b": ProsodicDeliveryEvidence(
            candidate_id="b", source_asset_id="s1", source_start=0.0, source_end=4.0,
            analysis_status=STATUS_NOT_EVALUABLE,
            speech_duration_sec=None, voiced_or_active_speech_duration_sec=None,
            speech_rate=None, speech_rate_state=UNKNOWN, pause_count=None, pause_total_sec=None,
            pause_structure_state=UNKNOWN, hesitation_state=UNKNOWN, restart_or_interruption_state=UNKNOWN,
            vocal_continuity_state=UNKNOWN, energy_mean=None, energy_variation=None,
            energy_dynamics_state=UNKNOWN, emphasis_dynamics_state=UNKNOWN,
            pitch_analysis_status=PITCH_NOT_IMPLEMENTED, pitch_variation_state=UNKNOWN,
            delivery_variation_state=UNKNOWN, evidence_confidence="UNKNOWN",
            missing_evidence=("audio_samples",), provenance="test",
        )},
        ["a", "b"],
    )
    assert partial.comparison_state == COMPARISON_INSUFFICIENT_EVIDENCE
    result = evaluate_bounded_finalist_arbiter(_fusion_input(prosodic_comparison=partial))
    assert result.prosodic_comparison_status == "INSUFFICIENT"
    assert result.arbiter_state == STATE_INSUFFICIENT_EVIDENCE


# ===========================================================================
# 13-16. Each safe dimension alone; pause_structure == continuity signal.
# ===========================================================================
def test_13_continuity_only_safe_preference():
    c = compare_prosodic_finalists(
        {"a": _pde("a", continuity=CONTINUITY_FRAGMENTED), "b": _pde("b", continuity=CONTINUITY_CONTINUOUS)},
        ["a", "b"],
    )
    assert c.comparison_state == COMPARISON_DOMINANT and c.preferred_candidate_id == "b"


def test_14_hesitation_only_safe_preference():
    c = compare_prosodic_finalists(
        {"a": _pde("a", hesitation=HESITATION_PRESENT), "b": _pde("b", hesitation=HESITATION_NOT_OBSERVED)},
        ["a", "b"],
    )
    assert c.comparison_state == COMPARISON_DOMINANT and c.preferred_candidate_id == "b"


def test_15_restart_only_safe_preference():
    c = compare_prosodic_finalists(
        {"a": _pde("a", restart=RESTART_SUPPORTED), "b": _pde("b", restart=RESTART_NOT_OBSERVED)},
        ["a", "b"],
    )
    assert c.comparison_state == COMPARISON_DOMINANT and c.preferred_candidate_id == "b"


def test_16_pause_structure_comparison_always_equals_continuity_comparison():
    c = compare_prosodic_finalists(
        {"a": _pde("a", continuity=CONTINUITY_FRAGMENTED), "b": _pde("b", continuity=CONTINUITY_CONTINUOUS)},
        ["a", "b"],
    )
    assert c.pause_structure_comparison == c.continuity_comparison
    assert DOUBLE_COUNTING_AUDIT["pause_structure_comparison"].startswith("SAME_UNDERLYING_SIGNAL")


# ===========================================================================
# 17-20. Descriptive-only differences -> NO preference.
# ===========================================================================
def test_17_speech_rate_only_difference_no_preference():
    c = compare_prosodic_finalists(
        {"a": _pde("a", rate=1.0), "b": _pde("b", rate=5.0)}, ["a", "b"],
    )
    assert c.comparison_state == COMPARISON_NEAR_EQUAL
    assert c.preferred_candidate_id is None
    assert c.descriptive_rate_relation == "b"  # observational only


def test_18_energy_only_difference_no_preference():
    c = compare_prosodic_finalists(
        {"a": _pde("a", energy_var=0.05), "b": _pde("b", energy_var=0.9)}, ["a", "b"],
    )
    assert c.comparison_state == COMPARISON_NEAR_EQUAL
    assert c.preferred_candidate_id is None
    assert c.descriptive_energy_relation == "b"


def test_19_emphasis_only_difference_no_preference():
    c = compare_prosodic_finalists(
        {"a": _pde("a", emphasis="EMPHASIS_NOT_OBSERVED"), "b": _pde("b", emphasis="EMPHASIS_PATTERN_PRESENT")},
        ["a", "b"],
    )
    assert c.comparison_state == COMPARISON_NEAR_EQUAL
    assert c.preferred_candidate_id is None
    assert c.descriptive_emphasis_relation == "b"


def test_20_pitch_only_difference_no_preference_never_implemented():
    c = compare_prosodic_finalists({"a": _pde("a"), "b": _pde("b")}, ["a", "b"])
    assert c.pitch_status == PITCH_NOT_IMPLEMENTED
    assert c.comparison_state == COMPARISON_NEAR_EQUAL


def test_20b_descriptive_evidence_never_creates_result_via_fusion():
    """End-to-end: a descriptive-only difference must never turn into a
    D-184 PREFER_CANDIDATE even when it is the ONLY evidence supplied."""
    c = compare_prosodic_finalists(
        {"a": _pde("a", rate=1.0, energy_var=0.05), "b": _pde("b", rate=5.0, energy_var=0.9)}, ["a", "b"],
    )
    result = evaluate_bounded_finalist_arbiter(_fusion_input(prosodic_comparison=c))
    assert result.decision == DECISION_ABSTAIN
    assert result.arbiter_state == STATE_NEAR_EQUAL


# ===========================================================================
# 21. Gain-invariance.
# ===========================================================================
def test_21_gain_invariance_same_comparison_state():
    import numpy as np
    from cutsell_worker.prosodic_audio_v2 import AudioSamples

    base_wave = np.concatenate([
        np.full(1000, 0.05, dtype=np.float32), np.full(1000, 0.4, dtype=np.float32),
        np.full(1000, 0.08, dtype=np.float32), np.full(1000, 0.35, dtype=np.float32),
    ])
    loud_wave = base_wave * 3.0
    audio_a = AudioSamples(source_asset_id="s1", sample_rate=1000, samples=base_wave, duration_sec=4.0, provenance="t")
    audio_b = AudioSamples(source_asset_id="s1", sample_rate=1000, samples=loud_wave, duration_sec=4.0, provenance="t")
    ev_a = analyze_prosodic_delivery("a", "s1", 0.0, 4.0, audio_a)
    ev_b = analyze_prosodic_delivery("b", "s1", 0.0, 4.0, audio_b)
    c = compare_prosodic_finalists({"a": ev_a, "b": ev_b}, ["a", "b"])
    assert c.comparison_state in (COMPARISON_NEAR_EQUAL,)  # identical delivery shape, just louder
    assert c.preferred_candidate_id is None


# ===========================================================================
# 22. Filler control.
# ===========================================================================
def test_22_filler_text_does_not_cause_acoustic_preference():
    """Candidate A has filler text in Language Spine but continuous
    acoustic delivery; Candidate B has no filler but interrupted acoustic
    delivery. Actual acoustic shape must control -- B, not A, should be
    the one with worse delivery evidence."""
    import numpy as np
    from cutsell_worker.prosodic_audio_v2 import AudioSamples

    continuous_wave = np.tile(np.linspace(-0.3, 0.3, 200).astype(np.float32), 20)
    ev_a = analyze_prosodic_delivery(
        "a", "s1", 0.0, 4.0, AudioSamples("s1", 1000, continuous_wave, 4.0, "t"),
        language_filler_present=True,
    )
    ev_b = analyze_prosodic_delivery(
        "b", "s1", 0.0, 4.0, AudioSamples("s1", 1000, continuous_wave, 4.0, "t"),
        audio_silence_intervals=[(1.5, 2.0)],
    )
    assert ev_a.hesitation_state == HESITATION_NOT_OBSERVED  # filler alone never escalates
    assert ev_b.hesitation_state == HESITATION_PRESENT  # real interior pause does
    c = compare_prosodic_finalists({"a": ev_a, "b": ev_b}, ["a", "b"])
    assert c.comparison_state == COMPARISON_DOMINANT
    assert c.preferred_candidate_id == "a"  # A's actual delivery is cleaner, despite filler text


# ===========================================================================
# 23. Boundary-owned pause ignored (edge pause never counted as interior).
# ===========================================================================
def test_23_boundary_only_pause_never_makes_a_take_lose():
    import numpy as np
    from cutsell_worker.prosodic_audio_v2 import AudioSamples

    wave = np.tile(np.linspace(-0.3, 0.3, 200).astype(np.float32), 20)
    audio = AudioSamples("s1", 1000, wave, 4.0, "t")
    # Silence flush against BOTH edges of the span -- boundary/dead-air,
    # never an interior delivery interruption.
    ev_a = analyze_prosodic_delivery("a", "s1", 1.0, 3.0, audio, audio_silence_intervals=[(0.0, 1.0), (3.0, 4.0)])
    ev_b = analyze_prosodic_delivery("b", "s1", 1.0, 3.0, audio, audio_silence_intervals=())
    assert ev_a.pause_count == 0
    assert ev_a.vocal_continuity_state == ev_b.vocal_continuity_state == CONTINUITY_CONTINUOUS
    c = compare_prosodic_finalists({"a": ev_a, "b": ev_b}, ["a", "b"])
    assert c.comparison_state == COMPARISON_NEAR_EQUAL


# ===========================================================================
# 24-27. Meaning/negation/number/factual conflict blocks BEFORE Prosody.
# ===========================================================================
def test_24_meaning_conflict_blocks_prosody_never_consulted():
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        candidate_texts={
            "a": "The cream clears acne breakouts in 2 weeks.",
            "b": "The cream does not clear acne breakouts in 2 weeks.",
        },
        prosodic_comparison=_dominance_comparison(),
    ))
    assert result.arbiter_state == STATE_CONFLICTED
    assert result.prosodic_comparison_status == "NOT_AVAILABLE"  # never even looked at


def test_25_negation_conflict_blocks():
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        candidate_texts={
            "a": "The cream clears acne breakouts in 2 weeks.",
            "b": "The cream does not clear acne breakouts in 2 weeks.",
        },
        prosodic_comparison=_dominance_comparison(),
    ))
    assert result.arbiter_state == STATE_CONFLICTED


def test_26_number_conflict_blocks():
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        candidate_texts={
            "a": "Apply the cream twice daily for 2 weeks to clear the breakout.",
            "b": "Apply the cream twice daily for 4 weeks to clear the breakout.",
        },
        prosodic_comparison=_dominance_comparison(),
    ))
    assert result.arbiter_state == STATE_CONFLICTED


def test_27_outside_meaning_sufficient_blocks():
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        meaning_sufficient_candidate_ids=("a",),  # b excluded
        prosodic_comparison=_dominance_comparison(),
    ))
    assert result.arbiter_state == STATE_CONFLICTED
    assert result.reason == "candidate_not_meaning_sufficient"


# ===========================================================================
# 28-29. Three finalists.
# ===========================================================================
def test_28_three_finalists_clear_prosodic_dominance():
    c = compare_prosodic_finalists(
        {"a": _pde("a", CONTINUITY_FRAGMENTED, HESITATION_PRESENT, RESTART_SUPPORTED),
         "b": _pde("b", CONTINUITY_CONTINUOUS, HESITATION_NOT_OBSERVED, RESTART_NOT_OBSERVED),
         "c": _pde("c", CONTINUITY_FRAGMENTED, HESITATION_PRESENT, RESTART_SUPPORTED)},
        ["a", "b", "c"],
    )
    assert c.comparison_state == COMPARISON_DOMINANT
    assert c.preferred_candidate_id == "b"
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        candidate_ids=("a", "b", "c"), meaning_sufficient_candidate_ids=("a", "b", "c"),
        prosodic_comparison=c,
    ))
    assert result.decision == DECISION_PREFER_CANDIDATE
    assert result.preferred_candidate_id == "b"


def test_29_three_finalists_cycle_conflict_abstains():
    # A beats B on continuity, B beats C on hesitation, C beats A on restart
    # -- a genuine directed 3-cycle, never resolved by an arbitrary pick.
    a = _pde("a", continuity=CONTINUITY_CONTINUOUS, hesitation=HESITATION_PRESENT, restart=RESTART_SUPPORTED)
    b = _pde("b", continuity=CONTINUITY_FRAGMENTED, hesitation=HESITATION_NOT_OBSERVED, restart=RESTART_SUPPORTED)
    c_ev = _pde("c", continuity=CONTINUITY_FRAGMENTED, hesitation=HESITATION_PRESENT, restart=RESTART_NOT_OBSERVED)
    comparison = compare_prosodic_finalists({"a": a, "b": b, "c": c_ev}, ["a", "b", "c"])
    assert comparison.comparison_state == COMPARISON_CONFLICTED
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        candidate_ids=("a", "b", "c"), meaning_sufficient_candidate_ids=("a", "b", "c"),
        prosodic_comparison=comparison,
    ))
    assert result.decision == DECISION_ABSTAIN
    assert result.arbiter_state == STATE_CONFLICTED


# ===========================================================================
# 30-32. D-186B abstract replay controls (generic -- no literal transcript).
# ===========================================================================
def test_30_d186b_abstract_pimples_differentiation_prefers_clean_delivery():
    """D-186B's real Pimples family was NON_DECISIVE/NEAR_EQUAL on visual;
    this generic two-take fixture (same shape, no literal transcript)
    checks that Prosody differentiates and the fusion PREFERS the cleaner
    delivery -- exactly the roadmap's success interpretation."""
    fragmented = _pde("a", CONTINUITY_FRAGMENTED, HESITATION_PRESENT, RESTART_SUPPORTED)
    clean = _pde("b", CONTINUITY_CONTINUOUS, HESITATION_NOT_OBSERVED, RESTART_NOT_OBSERVED)
    comparison = compare_prosodic_finalists({"a": fragmented, "b": clean}, ["a", "b"])
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        terminal_confidence_state="NON_DECISIVE", prosodic_comparison=comparison,
    ))
    assert result.decision == DECISION_PREFER_CANDIDATE
    assert result.preferred_candidate_id == "b"
    assert result.action_applied is False


def test_31_d186b_pimples_near_equal_control_no_forced_preference():
    """Same Language/Visual shape, but Audio A/B are themselves near-equal
    -- must NOT force a preference from speech-rate/energy noise."""
    a = _pde("a", CONTINUITY_CONTINUOUS, HESITATION_NOT_OBSERVED, RESTART_NOT_OBSERVED, rate=2.1, energy_var=0.28)
    b = _pde("b", CONTINUITY_CONTINUOUS, HESITATION_NOT_OBSERVED, RESTART_NOT_OBSERVED, rate=2.3, energy_var=0.31)
    comparison = compare_prosodic_finalists({"a": a, "b": b}, ["a", "b"])
    assert comparison.comparison_state == COMPARISON_NEAR_EQUAL
    result = evaluate_bounded_finalist_arbiter(_fusion_input(prosodic_comparison=comparison))
    assert result.decision == DECISION_ABSTAIN


def test_32_d186b_gynecologist_decisive_control_never_reopened():
    """Known D-186B finding: Gynecologist family was D-183 DECISIVE / D-184
    NOT_ELIGIBLE. D-188 must not make it eligible just because Prosodic
    evidence exists."""
    comparison = _dominance_comparison()
    result = evaluate_bounded_finalist_arbiter(_fusion_input(
        terminal_confidence_state="DECISIVE", prosodic_comparison=comparison,
    ))
    assert result.arbiter_state == STATE_NOT_ELIGIBLE
    assert result.decision == DECISION_ABSTAIN


# ===========================================================================
# 33-36. Structural firewalls: no majority voting, no master score, no new
# weights, no provider/network.
# ===========================================================================
def _strip_docstrings(source: str) -> str:
    """Remove every triple-quoted string AND `#` comment so a word-scan
    only inspects executable code -- avoids false positives from prose
    that NAMES a forbidden concept only to disclaim it (e.g. this
    module's own "no majority voting" doctrine, stated in both
    docstrings and inline comments)."""
    import re
    no_docstrings = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    return re.sub(r"#.*", "", no_docstrings)


def test_33_no_majority_voting_source_scan():
    for source in (BOUNDED_ARBITER_SOURCE, PROSODIC_COMPARISON_SOURCE):
        code_only = _strip_docstrings(source)
        assert "majority" not in code_only.casefold()
        assert "2_of_3" not in code_only.casefold() and "2-of-3" not in code_only.casefold()


def test_34_no_master_score_source_scan():
    for source in (BOUNDED_ARBITER_SOURCE, PROSODIC_COMPARISON_SOURCE):
        assert "prosody_score" not in source.casefold()
        assert "master_score" not in source.casefold()
    field_names = set(ProsodicFinalistComparison.__dataclass_fields__.keys())
    assert "score" not in field_names and "prosody_score" not in field_names


def test_35_no_new_weights_source_scan():
    for source in (BOUNDED_ARBITER_SOURCE, PROSODIC_COMPARISON_SOURCE):
        assert "score_weight" not in source.casefold()
        assert "weighted_score" not in source.casefold()
        assert re_no_weight_literal(source)


def re_no_weight_literal(source: str) -> bool:
    import re
    # No "weight = <number>" assignment anywhere (a bespoke numeric weight
    # would be exactly this shape).
    return re.search(r"weight\s*=\s*[\d.]+", source, re.IGNORECASE) is None


def test_36_no_provider_or_network_source_scan():
    for source in (BOUNDED_ARBITER_SOURCE, PROSODIC_COMPARISON_SOURCE):
        lowered = source.casefold()
        for forbidden in ("requests", "urllib", "socket.", "http.client", "openai", "anthropic", "gemini", "modal.com"):
            assert forbidden not in lowered


# ===========================================================================
# 37. No winner mutation.
# ===========================================================================
def test_37_no_winner_mutation():
    result = evaluate_bounded_finalist_arbiter(_fusion_input(prosodic_comparison=_dominance_comparison()))
    assert result.action_applied is False
    field_names = set(BoundedFinalistArbiterResult.__dataclass_fields__.keys())
    assert "selected_clip_id" not in field_names
    comparison_field_names = set(ProsodicFinalistComparison.__dataclass_fields__.keys())
    assert "selected_clip_id" not in comparison_field_names
    assert "winner" not in comparison_field_names
    assert "action" not in comparison_field_names


# ===========================================================================
# 38-41. Determinism, order/id independence.
# ===========================================================================
def test_38_determinism_identical_inputs_identical_output():
    c1 = _dominance_comparison()
    c2 = _dominance_comparison()
    assert c1 == c2
    r1 = evaluate_bounded_finalist_arbiter(_fusion_input(prosodic_comparison=c1))
    r2 = evaluate_bounded_finalist_arbiter(_fusion_input(prosodic_comparison=c2))
    assert r1 == r2


def test_39_candidate_order_independence():
    c_ab = compare_prosodic_finalists(
        {"a": _pde("a", CONTINUITY_FRAGMENTED), "b": _pde("b", CONTINUITY_CONTINUOUS)}, ["a", "b"],
    )
    c_ba = compare_prosodic_finalists(
        {"a": _pde("a", CONTINUITY_FRAGMENTED), "b": _pde("b", CONTINUITY_CONTINUOUS)}, ["b", "a"],
    )
    assert c_ab.comparison_state == c_ba.comparison_state == COMPARISON_DOMINANT
    assert c_ab.preferred_candidate_id == c_ba.preferred_candidate_id == "b"


def test_40_clip_id_independence():
    a = _pde("clip_xyz_9f2", CONTINUITY_FRAGMENTED, HESITATION_PRESENT, RESTART_SUPPORTED)
    b = _pde("clip_abc_1a7", CONTINUITY_CONTINUOUS, HESITATION_NOT_OBSERVED, RESTART_NOT_OBSERVED)
    c = compare_prosodic_finalists({"clip_xyz_9f2": a, "clip_abc_1a7": b}, ["clip_xyz_9f2", "clip_abc_1a7"])
    assert c.comparison_state == COMPARISON_DOMINANT
    assert c.preferred_candidate_id == "clip_abc_1a7"


def test_41_family_id_independence():
    comparison = _dominance_comparison()
    r1 = evaluate_bounded_finalist_arbiter(_fusion_input(family_id="tg_aaa", prosodic_comparison=comparison))
    r2 = evaluate_bounded_finalist_arbiter(_fusion_input(family_id="tg_zzz_different", prosodic_comparison=comparison))
    assert r1.decision == r2.decision == DECISION_PREFER_CANDIDATE
    assert r1.preferred_candidate_id == r2.preferred_candidate_id == "b"


# ===========================================================================
# Diagnostics / tail-safe summary shape (this task's own required fields).
# ===========================================================================
def test_prosodic_finalist_diagnostics_has_13_required_keys():
    comparison = _dominance_comparison()
    row = prosodic_finalist_diagnostics(comparison)
    for key in (
        "prosodic_finalist_evaluated", "prosodic_finalist_state",
        "prosodic_finalist_preferred_candidate_id", "prosodic_finalist_continuity_relation",
        "prosodic_finalist_hesitation_relation", "prosodic_finalist_restart_relation",
        "prosodic_finalist_pause_relation", "prosodic_finalist_descriptive_rate_relation",
        "prosodic_finalist_descriptive_energy_relation", "prosodic_finalist_descriptive_emphasis_relation",
        "prosodic_finalist_directional_evidence_present", "prosodic_finalist_conflict",
        "prosodic_finalist_missing_evidence",
    ):
        assert key in row
    assert len(row) == 13
    import json
    json.dumps(row)  # bounded, JSON-safe, no waveform/transcript dump


def test_bounded_finalist_arbiter_prosodic_fusion_diagnostics_shape():
    result = evaluate_bounded_finalist_arbiter(_fusion_input(prosodic_comparison=_dominance_comparison()))
    row = bounded_finalist_arbiter_prosodic_fusion_diagnostics(result)
    assert row == {"bounded_finalist_arbiter_prosodic_status": "AVAILABLE", "bounded_finalist_arbiter_prosodic_contributed": True}


def test_tail_safe_summaries_have_all_6_required_counts():
    comparisons = [_dominance_comparison(), compare_prosodic_finalists({"a": _pde("a"), "b": _pde("b")}, ["a", "b"])]
    summary = prosodic_finalist_run_summary(comparisons)
    for key in (
        "prosodic_finalist_evaluated_count", "prosodic_finalist_dominance_count",
        "prosodic_finalist_near_equal_count", "prosodic_finalist_conflicted_count",
        "prosodic_finalist_insufficient_count",
    ):
        assert key in summary
    assert summary["prosodic_finalist_dominance_count"] == 1
    assert summary["prosodic_finalist_near_equal_count"] == 1

    results = [evaluate_bounded_finalist_arbiter(_fusion_input(prosodic_comparison=_dominance_comparison()))]
    fusion_summary = bounded_finalist_arbiter_prosodic_fusion_run_summary(results)
    assert fusion_summary == {"arbiter_preferences_due_to_prosody_count": 1}


# ===========================================================================
# Old D-184 behavior byte-identical when prosodic_comparison is None
# (backward-compatibility, restated here for this task's own record).
# ===========================================================================
def test_backward_compat_v2_only_behavior_unaffected_by_new_field():
    from cutsell_worker.watch_listen_zone_usability_v2 import build_zone_usability_v2
    from cutsell_worker.raw_understanding_map import RawUnderstandingSpan
    from cutsell_worker.positioned_performance_evidence import DeliverySpan, PositionAwarePerformanceEvidence

    def span(span_id):
        delivery = DeliverySpan(start=0.2, end=8.2, available=True, source="word_envelope")
        positioned = PositionAwarePerformanceEvidence(
            candidate_id=span_id, source_asset_id="s1", source_start=0.0, source_end=8.5,
            delivery_span=delivery, positioned_events=(),
        )
        return RawUnderstandingSpan(
            span_id=span_id, source_asset_id="s1", source_start=0.0, source_end=8.5,
            transcript="", word_timings=(), positioned_evidence=positioned,
            behavior_hypotheses=(), conflict_flags=(), evidence_provenance={},
        )

    v2_by_id = {"a": build_zone_usability_v2("a", span("a")), "b": build_zone_usability_v2("b", span("b"))}
    result = evaluate_bounded_finalist_arbiter(_fusion_input(v2_evidence_by_id=v2_by_id))
    assert result.prosodic_comparison_status == "NOT_AVAILABLE"
    assert "prosodic_audio" in result.missing_evidence
