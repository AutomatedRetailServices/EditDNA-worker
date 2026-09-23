"""Offline characterization of a pause-only BestTake false-positive risk.

The same 0.71 s interior silence can be an intentional rhetorical beat or
an accidental hesitation. The current acoustic features cannot tell which.
This test records the actual opt-in authority path; it is NOT a claim that
the clean candidate should win either editorial interpretation.
"""

from dataclasses import replace

import numpy as np
import pytest

from cutsell_worker.bounded_finalist_arbiter import (
    FinalistArbiterInput,
    evaluate_bounded_finalist_arbiter,
)
from cutsell_worker.bounded_finalist_authority import evaluate_bounded_finalist_authority
from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.prosodic_audio_v2 import AudioSamples, RESTART_SUPPORTED, analyze_prosodic_delivery
from cutsell_worker.prosodic_finalist_comparison import (
    compare_prosodic_finalists,
    prosodic_finalist_diagnostics,
)
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.watch_listen_besttake_v2_evidence import build_candidate_zone_usability_v2
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def test_one_natural_pause_and_ordinary_gesture_cannot_authorize_winner_mutation():
    """One pause supplies every alleged acoustic-quality difference.

    The generic words remain complete, fluent and identical; no recording
    error is supplied. A short hand movement is not independent evidence
    of a failed line. These are synthetic inputs, not a Video00 replay.
    """
    sr = 1000
    samples = np.full(8 * sr, 0.2, dtype=np.float32)
    samples[1500:2210] = 0.0
    audio = AudioSamples("source", sr, samples, 8.0, "synthetic")
    pause = (1.5, 2.21)
    words_a = (Word("Here", 0.1, 0.5), Word("is", 0.6, 0.9),
               Word("the", 2.3, 2.6), Word("result", 2.7, 3.6))
    words_b = tuple(Word(w.text, w.start + 4.0, w.end + 4.0) for w in words_a)
    a = CandidateTake("natural_pause", "source", 0, 0.0, 4.0, "Here is the result.", words=words_a)
    b = CandidateTake("continuous", "source", 0, 4.0, 8.0, "Here is the result.", words=words_b)
    gesture = TemporalEvent("source", 1.7, 1.9, "hand_motion_reset_candidate", 0.9,
                            "ordinary gesture while speaking")
    context = WholeVideoContext(
        sources=(SourceVideoContext("source", "", "", "", events=(gesture,)),),
        status=ProviderStatus("whole_video_analysis", True, True, "ok"),
    )

    acoustic = {
        t.clip_id: analyze_prosodic_delivery(
            t.clip_id, "source", t.start, t.end, audio, words=t.words,
            audio_silence_intervals=(pause,),
        )
        for t in (a, b)
    }
    assert acoustic[a.clip_id].pause_count == 1
    assert acoustic[a.clip_id].pause_total_sec == 0.71
    assert acoustic[b.clip_id].pause_count == 0
    comparison = compare_prosodic_finalists(acoustic, (a.clip_id, b.clip_id))
    assert comparison.comparison_state == "INSUFFICIENT_EVIDENCE"
    assert comparison.preferred_candidate_id is None
    assert not comparison.directional_evidence_present
    assert "independent_in_span_disruption_evidence" in comparison.missing_evidence
    assert (comparison.continuity_comparison, comparison.hesitation_comparison,
            comparison.restart_comparison, comparison.pause_structure_comparison) == (b.clip_id,) * 4

    visual = {t.clip_id: build_candidate_zone_usability_v2(t, context) for t in (a, b)}
    arbiter = evaluate_bounded_finalist_arbiter(FinalistArbiterInput(
        family_id="generic_family", candidate_ids=(a.clip_id, b.clip_id),
        meaning_sufficient_candidate_ids=(a.clip_id, b.clip_id),
        terminal_confidence_state="CONFLICTED",
        candidate_texts={t.clip_id: t.text for t in (a, b)},
        v2_evidence_by_id=visual,
        prosodic_comparison=comparison,
    ))
    assert arbiter.meaning_parity_status == "CONSISTENT"
    assert arbiter.performance_comparison_status == "NEAR_EQUAL"
    assert arbiter.evidence_sources == ("zone_usability_v2",)
    # Desired safety boundary: these three Prosodic votes share one pause
    # and V2 found no independent performance defect. They cannot by
    # themselves certify that a natural spoken beat is a failed take.
    assert arbiter.arbiter_state == "NEAR_EQUAL"
    assert arbiter.preferred_candidate_id is None

    authority = evaluate_bounded_finalist_authority(
        enabled=True, winner_before=a.clip_id, candidate_ids=(a.clip_id, b.clip_id),
        terminal_confidence_state="CONFLICTED", arbiter_result=arbiter,
    )
    assert authority.authority_state == "NO_SUPPORTED_PREFERENCE"
    assert authority.winner_after == a.clip_id


def _audio_pair(pauses=(), *, language_restart=False):
    """Generic synthetic audio, not a recorded Video00 candidate."""
    sr = 1000
    samples = np.full(8 * sr, 0.2, dtype=np.float32)
    for start, end in pauses:
        samples[round(start * sr):round(end * sr)] = 0.0
    audio = AudioSamples("source", sr, samples, 8.0, "synthetic")
    return {
        cid: analyze_prosodic_delivery(
            cid, "source", start, start + 4.0, audio,
            audio_silence_intervals=pauses,
            language_restart_evidence=language_restart if cid == "a" else False,
        )
        for cid, start in (("a", 0.0), ("b", 4.0))
    }


@pytest.mark.parametrize("pauses", [((1.5, 2.21),), ((0.7, 1.5), (2.2, 3.0)), ((0.8, 3.2),)])
def test_pause_count_or_length_cannot_supply_missing_error_proof(pauses):
    evidence = _audio_pair(pauses)
    result = compare_prosodic_finalists(evidence, ("a", "b"))
    assert evidence["a"].pause_count == len(pauses)
    assert result.comparison_state == "INSUFFICIENT_EVIDENCE"
    assert result.preferred_candidate_id is None
    assert not result.conflict_present
    row = prosodic_finalist_diagnostics(result)
    assert not row["prosodic_finalist_directional_evidence_present"]
    assert "independent_in_span_disruption_evidence" in row["prosodic_finalist_missing_evidence"]


def test_language_restart_flag_does_not_localize_an_error_inside_a_clean_retry():
    evidence = _audio_pair(language_restart=True)
    assert evidence["a"].pause_count == 0
    assert evidence["a"].restart_or_interruption_state == RESTART_SUPPORTED
    result = compare_prosodic_finalists(evidence, ("a", "b"))
    assert result.restart_comparison == "b"
    assert result.comparison_state == "INSUFFICIENT_EVIDENCE"
    assert result.preferred_candidate_id is None


def test_equal_known_descriptors_remain_near_equal():
    result = compare_prosodic_finalists(_audio_pair(), ("a", "b"))
    assert result.comparison_state == "NEAR_EQUAL"
    assert result.preferred_candidate_id is None
    assert not result.directional_evidence_present


def test_unknown_member_does_not_become_near_equal_in_a_three_candidate_set():
    evidence = _audio_pair()
    evidence["c"] = replace(evidence["a"], candidate_id="c", hesitation_state="UNKNOWN")
    result = compare_prosodic_finalists(evidence, ("a", "b", "c"))
    assert result.hesitation_comparison == "UNKNOWN"
    assert result.comparison_state == "INSUFFICIENT_EVIDENCE"
    assert result.preferred_candidate_id is None


def test_absent_acoustic_input_stays_not_evaluable():
    result = compare_prosodic_finalists({"a": None, "b": None}, ("a", "b"))
    assert result.comparison_state == "NOT_EVALUABLE"
    assert result.preferred_candidate_id is None


def test_order_and_candidate_labels_cannot_create_a_preference():
    evidence = _audio_pair(((1.5, 2.21),))
    for names in (("a", "b"), ("later", "earlier")):
        renamed = {name: replace(evidence[old], candidate_id=name)
                   for name, old in zip(names, ("a", "b"))}
        for ids in (names, tuple(reversed(names))):
            result = compare_prosodic_finalists(renamed, ids)
            assert result.comparison_state == "INSUFFICIENT_EVIDENCE"
            assert result.preferred_candidate_id is None


def _independent_fumble_input():
    """An explicitly supplied, source-local verbal fumble, NOT inferred
    from a pause or an ordinary hand movement. This validates existing V2
    consumption, not a new automatic fumble detector."""
    takes = tuple(CandidateTake(
        cid, "source", 0, start, start + 4.0, "The package contains 12 items.",
        words=(Word("The", start + 0.1, start + 0.3),
               Word("package", start + 0.4, start + 0.8),
               Word("contains", start + 1.0, start + 1.5),
               Word("12", start + 2.0, start + 2.5),
               Word("items", start + 3.0, start + 3.8)),
    ) for cid, start in (("a", 0.0), ("b", 4.0)))
    context = WholeVideoContext(
        sources=(SourceVideoContext("source", "", "", "", events=(
            TemporalEvent("source", 4.5, 7.5, "verbal_fumble", 0.95,
                          "independently supplied in-speech production error"),
        )),), status=ProviderStatus("whole_video_analysis", True, True, "ok"),
    )
    return FinalistArbiterInput(
        family_id="generic", candidate_ids=("a", "b"),
        meaning_sufficient_candidate_ids=("a", "b"), terminal_confidence_state="CONFLICTED",
        candidate_texts={take.clip_id: take.text for take in takes},
        v2_evidence_by_id={take.clip_id: build_candidate_zone_usability_v2(take, context) for take in takes},
        # Raw descriptors favor b (no pause), independent error is in b.
        prosodic_comparison=compare_prosodic_finalists(_audio_pair(((1.5, 2.21),)), ("a", "b")),
    )


def test_independent_delivery_evidence_still_can_support_a_winner_change():
    result = evaluate_bounded_finalist_arbiter(_independent_fumble_input())
    assert result.meaning_parity_status == "CONSISTENT"
    assert result.performance_comparison_status == "DOMINANT"
    assert result.arbiter_state == "PREFERENCE_SUPPORTED"
    assert result.preferred_candidate_id == "a"
    assert "prosodic_delivery" not in result.evidence_sources
    authority = evaluate_bounded_finalist_authority(
        enabled=True, winner_before="b", candidate_ids=("a", "b"),
        terminal_confidence_state="CONFLICTED", arbiter_result=result,
    )
    assert authority.winner_after == "a"
    assert authority.authority_applied


@pytest.mark.parametrize("other", ["The package does not contain 12 items.",
                                  "The package contains 20 items."])
def test_independent_delivery_preference_never_overrides_meaning_conflict(other):
    data = _independent_fumble_input()
    result = evaluate_bounded_finalist_arbiter(replace(
        data, candidate_texts={"a": data.candidate_texts["a"], "b": other},
    ))
    assert result.arbiter_state == "CONFLICTED"
    assert result.preferred_candidate_id is None


def test_default_opt_in_flags_are_unchanged():
    from cutsell_worker.bounded_finalist_arbiter import bounded_finalist_arbiter_enabled
    from cutsell_worker.bounded_finalist_authority import bounded_finalist_arbiter_authority_enabled
    from cutsell_worker.prosodic_finalist_comparison import prosodic_finalist_arbiter_diagnostics_enabled

    assert not bounded_finalist_arbiter_enabled({})
    assert not bounded_finalist_arbiter_authority_enabled({})
    assert not prosodic_finalist_arbiter_diagnostics_enabled({})
