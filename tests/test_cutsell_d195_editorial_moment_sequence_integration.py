"""D-195: P1 Editorial Moment & Sequence Understanding -- Phase B tests.

Generic, abstract fixtures only -- NO Video00 literal text/spans/ids.
Covers the directive's 60-item offline test matrix: canonical-evidence
mapping, mapping firewalls, sequence classification via real adapters,
no-recomputation/no-authority structural guarantees, determinism, and
pipeline-level default-off parity / flag-on immutability.
"""
from __future__ import annotations

import ast
import pathlib
import time

import pytest

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.editorial_moment_sequence import (
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    MOMENT_ROLE_ABANDONED_ATTEMPT,
    MOMENT_ROLE_BREAKING_CHARACTER,
    MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
    MOMENT_ROLE_CONTINUATION,
    MOMENT_ROLE_CORRECTION,
    MOMENT_ROLE_FALSE_START,
    MOMENT_ROLE_NEW_AUDIENCE_BEAT,
    MOMENT_ROLE_RECORDING_PROCESS,
    MOMENT_ROLE_RETRY,
    SEQUENCE_KIND_BLOOPER_SERIES,
    SEQUENCE_KIND_CLEAN_DELIVERY_SEQUENCE,
    SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE,
    SEQUENCE_KIND_RETRY_SERIES,
    EARLIER_SOURCE_REDUNDANCY_NOT_EVALUATED,
)
from cutsell_worker.editorial_moment_sequence_integration import (
    CAPABILITY_AVAILABLE,
    CAPABILITY_NOT_EVALUABLE,
    CAPABILITY_PARTIAL,
    EditorialMomentUnderstanding,
    build_editorial_moment_understanding_for_source,
    build_editorial_moment_understanding_for_sources,
    build_editorial_moments_for_source,
    build_editorial_sequences_for_moments,
    editorial_moment_sequence_diagnostics_enabled,
    editorial_moment_understanding_diagnostics,
    editorial_moment_understanding_run_summary,
)
from cutsell_worker.language_proposition_relation import PropositionCandidate, ClaimSignature
from cutsell_worker.language_utterance_attempt import (
    ATTEMPT_CLEAN,
    ATTEMPT_CORRECTION,
    LanguageAttempt,
    MEANING_COMPLETE,
)
from cutsell_worker.raw_understanding_map import (
    BEHAVIOR_ABANDONED_ATTEMPT,
    BEHAVIOR_BREAKING_CHARACTER,
    BEHAVIOR_CLEAN_ATTEMPT,
    BEHAVIOR_FALSE_START,
    BEHAVIOR_RECORDING_PROCESS,
    BehaviorHypothesis,
    MAP_STATUS_COMPLETE_EXISTING_EVIDENCE,
    MAP_STATUS_FAILED,
    RawUnderstandingMap,
)
from cutsell_worker.watch_listen_understanding import (
    AttemptRelationHypothesis,
    RELATION_COMPLEMENTARY,
    RELATION_CONTINUATION,
    RELATION_CORRECTION,
    RELATION_NEW_AUDIENCE_BEAT,
    RELATION_RETRY,
    RELATION_UNCERTAIN,
    CONFIDENCE_SUPPORTED as WL_CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN as WL_CONFIDENCE_UNKNOWN,
    MEANING_COMPLETE as WL_MEANING_COMPLETE,
    MEANING_INCOMPLETE as WL_MEANING_INCOMPLETE,
    USABILITY_UNUSABLE,
    USABILITY_USABLE,
    UnderstandingSpan,
    WatchListenUnderstanding,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER_DIR = REPO_ROOT / "cutsell_worker"
MODULE_PATH = WORKER_DIR / "editorial_moment_sequence_integration.py"


def _code_only_source() -> str:
    src = MODULE_PATH.read_text()
    tree = ast.parse(src)
    docstring_node = tree.body[0] if tree.body and isinstance(tree.body[0], ast.Expr) else None
    if docstring_node is None:
        return src
    lines = src.splitlines(keepends=True)
    return "".join(lines[docstring_node.end_lineno:])


# ---------------------------------------------------------------------------
# Generic fixture factories.
# ---------------------------------------------------------------------------
def _take(clip_id, order, start, end, text="generic abstract statement"):
    return CandidateTake(
        clip_id=clip_id, source_asset_id="src1", source_order=order, start=start, end=end, text=text,
    )


def _behavior(label, confidence=0.8):
    return BehaviorHypothesis(label=label, confidence=confidence, provenance="VISUAL_SIGNAL", basis="generic")


def _span(
    span_id, start, end, *, behavior_labels=(), meaning=WL_MEANING_COMPLETE, relation=None,
    delivery_usability=USABILITY_USABLE, exit_usability=USABILITY_USABLE,
):
    hyps = tuple(_behavior(label) for label in behavior_labels)
    rel = (
        (AttemptRelationHypothesis(relation, WL_CONFIDENCE_SUPPORTED, "x", None, ()),)
        if relation else ()
    )
    return UnderstandingSpan(
        span_id=span_id, source_asset_id="src1", source_start=start, source_end=end,
        behavior_state_hypotheses=hyps, behavior_confidence=WL_CONFIDENCE_SUPPORTED,
        attempt_boundary_hypotheses=(), attempt_relation_hypotheses=rel,
        relation_confidence=WL_CONFIDENCE_SUPPORTED, meaning_completion_hypothesis=meaning,
        performance_usability_hypothesis=USABILITY_USABLE, entry_usability=USABILITY_USABLE,
        delivery_usability=delivery_usability, exit_usability=exit_usability,
        conflict_flags=(), evidence_provenance={},
    )


def _wlu(*spans):
    return WatchListenUnderstanding(source_asset_id="src1", understanding_spans=tuple(spans))


def _two_take_retry_fixture():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    spans = [
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_ABANDONED_ATTEMPT], meaning=WL_MEANING_INCOMPLETE),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], meaning=WL_MEANING_COMPLETE, relation=RELATION_RETRY),
    ]
    return takes, _wlu(*spans)


# ---------------------------------------------------------------------------
# 1-2: flag.
# ---------------------------------------------------------------------------
def test_01_flag_default_off():
    assert editorial_moment_sequence_diagnostics_enabled({}) is False


def test_02_flag_on_diagnostics():
    assert editorial_moment_sequence_diagnostics_enabled(
        {"CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED": "1"}
    ) is True


# ---------------------------------------------------------------------------
# 3-4: RawUnderstanding / WatchListen mapping.
# ---------------------------------------------------------------------------
def test_03_raw_understanding_map_failed_status_yields_not_evaluable():
    takes, wlu = _two_take_retry_fixture()
    raw_map = RawUnderstandingMap(
        source_asset_id="src1", source_duration=2.0, source_timeline_origin="source_relative_seconds",
        transcript="", word_timings=(), track_status={"raw_understanding_map_status": MAP_STATUS_FAILED},
    )
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu,
        raw_understanding_map=raw_map,
    )
    assert u.capability_status == CAPABILITY_NOT_EVALUABLE
    assert "RAW_UNDERSTANDING_MAP_FAILED" in u.missing_evidence


def test_04_watch_listen_understanding_absent_is_not_evaluable():
    takes, _ = _two_take_retry_fixture()
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=None,
    )
    assert u.capability_status == CAPABILITY_NOT_EVALUABLE
    assert u.moment_count == 0


def test_04b_watch_listen_understanding_present_yields_available():
    takes, wlu = _two_take_retry_fixture()
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu,
    )
    assert u.capability_status == CAPABILITY_AVAILABLE
    assert u.moment_count == 2


# ---------------------------------------------------------------------------
# 5-6: LanguageAttempt / PropositionCandidate mapping.
# ---------------------------------------------------------------------------
def test_05_real_language_attempt_preferred_over_fallback():
    takes, wlu = _two_take_retry_fixture()
    real_attempt_1 = LanguageAttempt(
        source_asset_id="src1", attempt_id="c1", utterance_ids=("c1",), source_start=0.0, source_end=1.0,
        text_raw="x", text_normalized="x", attempt_state=ATTEMPT_CLEAN, meaning_completion=MEANING_COMPLETE,
        restart_evidence=False, correction_evidence=False, continuation_evidence=False,
        recording_process_evidence=False, confidence=CONFIDENCE_SUPPORTED, provenance="TEST_REAL_ATTEMPT",
    )
    real_attempt_2 = LanguageAttempt(
        source_asset_id="src1", attempt_id="c2", utterance_ids=("c2",), source_start=1.0, source_end=2.0,
        text_raw="x", text_normalized="x", attempt_state=ATTEMPT_CORRECTION, meaning_completion=MEANING_COMPLETE,
        restart_evidence=True, correction_evidence=True, continuation_evidence=False,
        recording_process_evidence=False, confidence=CONFIDENCE_SUPPORTED, provenance="TEST_REAL_ATTEMPT",
    )
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu,
        language_attempts_by_span_id={"c1": real_attempt_1, "c2": real_attempt_2},
    )
    assert "LANGUAGE_ATTEMPT_NOT_SUPPLIED" not in u.missing_evidence
    # Real attempt states win over the fallback derivation (which, from
    # c1's own ABANDONED_ATTEMPT behavior evidence alone, would have
    # classified it FALSE_START/ABANDONED_ATTEMPT instead of CLEAN).
    assert u.moments[0].moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY
    assert u.moments[1].moment_role == MOMENT_ROLE_CORRECTION


def test_06_proposition_candidate_id_mapping():
    takes = [_take("c1", 0, 0.0, 1.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]))
    sig = ClaimSignature(frozenset(), False, frozenset(), "NONE", "", "abc")
    prop = PropositionCandidate(
        source_asset_id="src1", proposition_candidate_id="prop_1", attempt_ids=("c1",),
        source_start=0.0, source_end=1.0, text_raw="x", text_normalized="x", claim_signature=sig,
        meaning_completion=MEANING_COMPLETE, editorial_slot_evidence="OTHER", confidence=CONFIDENCE_SUPPORTED,
        provenance="TEST",
    )
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu,
        proposition_candidates=(prop,),
    )
    assert "PROPOSITION_CANDIDATE_NOT_SUPPLIED" not in u.missing_evidence
    assert u.moments[0].proposition_candidate_ids == ("prop_1",)


# ---------------------------------------------------------------------------
# 7-10: RelationEvidence-style (D-157 relation hypothesis) mapping.
# ---------------------------------------------------------------------------
def test_07_retry_mapping():
    takes, wlu = _two_take_retry_fixture()
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.moments[1].moment_role == MOMENT_ROLE_RETRY


def test_08_correction_mapping():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_CORRECTION),
    )
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.moments[1].moment_role == MOMENT_ROLE_CORRECTION


def test_09_continuation_mapping():
    takes = [_take("c1", 0, 0.0, 1.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], meaning=WL_MEANING_INCOMPLETE))
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.moments[0].moment_role == MOMENT_ROLE_CONTINUATION


def test_10_new_beat_mapping():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_NEW_AUDIENCE_BEAT),
    )
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.moments[1].moment_role == MOMENT_ROLE_NEW_AUDIENCE_BEAT


# ---------------------------------------------------------------------------
# 11-15: behavior mapping.
# ---------------------------------------------------------------------------
def test_11_recording_process_behavior_mapping():
    takes = [_take("c1", 0, 0.0, 1.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_RECORDING_PROCESS]))
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.moments[0].moment_role == MOMENT_ROLE_RECORDING_PROCESS


def test_12_false_start_mapping():
    takes = [_take("c1", 0, 0.0, 1.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_FALSE_START]))
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.moments[0].moment_role == MOMENT_ROLE_FALSE_START


def test_13_abandoned_attempt_mapping():
    takes = [_take("c1", 0, 0.0, 5.0, text="a fairly long abandoned attempt with many words in it")]
    wlu = _wlu(_span("c1", 0.0, 5.0, behavior_labels=[BEHAVIOR_ABANDONED_ATTEMPT], meaning=WL_MEANING_INCOMPLETE))
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.moments[0].moment_role == MOMENT_ROLE_ABANDONED_ATTEMPT


def test_14_breaking_character_mapping():
    takes = [_take("c1", 0, 0.0, 1.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_BREAKING_CHARACTER]))
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.moments[0].moment_role == MOMENT_ROLE_BREAKING_CHARACTER


def test_15_clean_audience_delivery_mapping():
    takes = [_take("c1", 0, 0.0, 1.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]))
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.moments[0].moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY


# ---------------------------------------------------------------------------
# 16-21: missing evidence.
# ---------------------------------------------------------------------------
def test_16_missing_language_attempt_uses_fallback():
    takes, wlu = _two_take_retry_fixture()
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert "LANGUAGE_ATTEMPT_NOT_SUPPLIED" in u.missing_evidence
    assert u.moment_count == 2  # still produces real moments via the fallback adapter


def test_17_missing_proposition_candidate():
    takes, wlu = _two_take_retry_fixture()
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert "PROPOSITION_CANDIDATE_NOT_SUPPLIED" in u.missing_evidence


def test_18_missing_relation_evidence():
    takes, wlu = _two_take_retry_fixture()
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert "RELATION_EVIDENCE_NOT_SUPPLIED" in u.missing_evidence


def test_19_missing_behavior_evidence():
    takes = [_take("c1", 0, 0.0, 1.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0, behavior_labels=[]))
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert "BEHAVIOR_HYPOTHESIS" not in u.moments[0].provenance


def test_20_optional_visual_missing():
    takes = [_take("c1", 0, 0.0, 1.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], exit_usability=USABILITY_USABLE))
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert "VISUAL_CORROBORATION" not in u.moments[0].provenance


def test_21_optional_prosodic_missing():
    takes, wlu = _two_take_retry_fixture()
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert "PROSODIC_EVIDENCE_NOT_SUPPLIED" in u.missing_evidence
    assert all("PROSODIC_CORROBORATION" not in m.provenance for m in u.moments)


# ---------------------------------------------------------------------------
# 22-24: prosodic / visual corroboration and the ordinary-motion firewall.
# ---------------------------------------------------------------------------
class _FakeProsodic:
    def __init__(self, vocal_continuity_state):
        self.vocal_continuity_state = vocal_continuity_state


def test_22_prosodic_continuous_corroboration():
    takes = [_take("c1", 0, 0.0, 1.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]))
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu,
        prosodic_evidence_by_span_id={"c1": _FakeProsodic("CONTINUOUS")},
    )
    assert "PROSODIC_CORROBORATION" in u.moments[0].provenance
    assert not u.moments[0].conflict_flags
    assert u.moments[0].moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY


def test_23_prosodic_interrupted_corroboration_flags_conflict():
    takes = [_take("c1", 0, 0.0, 1.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]))
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu,
        prosodic_evidence_by_span_id={"c1": _FakeProsodic("FRAGMENTED")},
    )
    assert u.moments[0].conflict_flags
    assert u.confidence == CONFIDENCE_MIXED


def test_24_ordinary_motion_does_not_break_role():
    """A usable delivery zone (no defect kinds) never downgrades or
    overrides a clean role -- exit_usability stays USABLE."""
    takes = [_take("c1", 0, 0.0, 1.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], exit_usability=USABILITY_USABLE))
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.moments[0].moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY
    assert not u.moments[0].conflict_flags


# ---------------------------------------------------------------------------
# 25-33: sequence classification through the real adapter path.
# ---------------------------------------------------------------------------
def test_25_retry_series():
    takes, wlu = _two_take_retry_fixture()
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.sequence_hypotheses[0].sequence_kind == SEQUENCE_KIND_RETRY_SERIES


def test_26_blooper_series():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0), _take("c3", 2, 2.0, 3.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_BREAKING_CHARACTER]),
        _span("c3", 2.0, 3.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_RETRY),
    )
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.sequence_hypotheses[0].sequence_kind == SEQUENCE_KIND_BLOOPER_SERIES


def test_27_clean_delivery_sequence():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
    )
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.sequence_hypotheses[0].sequence_kind == SEQUENCE_KIND_CLEAN_DELIVERY_SEQUENCE


def test_28_preassembled_final_sequence():
    # RELATION_COMPLEMENTARY (unlike RELATION_RETRY/RELATION_NEW_AUDIENCE_BEAT)
    # never overrides a clean attempt's moment_role away from
    # CLEAN_AUDIENCE_DELIVERY in D-194's own classifier, while still
    # counting as forward-progression-compatible evidence at the
    # sequence level -- exactly the real-pipeline shape this fixture
    # needs to prove PREASSEMBLED_FINAL_SEQUENCE end to end.
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0), _take("c3", 2, 2.0, 3.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_COMPLEMENTARY),
        _span("c3", 2.0, 3.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT], relation=RELATION_COMPLEMENTARY),
    )
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert all(m.moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY for m in u.moments)
    assert u.sequence_hypotheses[0].sequence_kind == SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE


def test_29_clean_adjacent_takes_not_automatically_final():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0), _take("c3", 2, 2.0, 3.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c3", 2.0, 3.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
    )
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.sequence_hypotheses[0].sequence_kind != SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE


def test_30_chronology_reversal_same_classification():
    def _build(offset):
        takes = [_take("c1", 0, offset, offset + 1.0), _take("c2", 1, offset + 1.0, offset + 2.0)]
        wlu = _wlu(
            _span("c1", offset, offset + 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
            _span("c2", offset + 1.0, offset + 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        )
        return build_editorial_moment_understanding_for_source(
            source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu,
        )

    early = _build(0.0)
    late = _build(500.0)
    assert early.sequence_hypotheses[0].sequence_kind == late.sequence_hypotheses[0].sequence_kind


def test_31_jump_cut_unavailable():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
    )
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    from cutsell_worker.editorial_moment_sequence import CONTINUITY_NOT_AVAILABLE
    assert u.sequence_hypotheses[0].continuity_status == CONTINUITY_NOT_AVAILABLE


def test_32_sequence_conflict_propagates():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(
        _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("c2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
    )
    u = build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu,
        prosodic_evidence_by_span_id={"c1": _FakeProsodic("FRAGMENTED")},
    )
    assert u.sequence_hypotheses[0].conflict_flags
    assert u.sequence_hypotheses[0].confidence == CONFIDENCE_MIXED


def test_33_no_distant_redundancy():
    takes, wlu = _two_take_retry_fixture()
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert all(s.earlier_source_redundancy_status == EARLIER_SOURCE_REDUNDANCY_NOT_EVALUATED for s in u.sequence_hypotheses)


# ---------------------------------------------------------------------------
# 34-37: timing / determinism.
# ---------------------------------------------------------------------------
def test_34_exact_source_timing():
    takes = [_take("c1", 0, 12.5, 17.25)]
    wlu = _wlu(_span("c1", 12.5, 17.25, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]))
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert u.moments[0].source_start == 12.5
    assert u.moments[0].source_end == 17.25


def test_35_deterministic_moment_ids():
    takes, wlu = _two_take_retry_fixture()
    u1 = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    u2 = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert [m.editorial_moment_id for m in u1.moments] == [m.editorial_moment_id for m in u2.moments]


def test_36_deterministic_sequence_ids():
    takes, wlu = _two_take_retry_fixture()
    u1 = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    u2 = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert [s.sequence_id for s in u1.sequence_hypotheses] == [s.sequence_id for s in u2.sequence_hypotheses]


def test_37_deterministic_ordering_regardless_of_input_order():
    takes_fwd, wlu = _two_take_retry_fixture()
    takes_rev = tuple(reversed(takes_fwd))
    u_fwd = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes_fwd, watch_listen_understanding=wlu)
    u_rev = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes_rev, watch_listen_understanding=wlu)
    assert [m.editorial_moment_id for m in u_fwd.moments] == [m.editorial_moment_id for m in u_rev.moments]


# ---------------------------------------------------------------------------
# 38-43: no recomputation / no provider (structural, docstring-stripped).
# ---------------------------------------------------------------------------
def test_38_no_transcript_dump():
    takes = [_take("c1", 0, 0.0, 1.0, text="a very specific real transcript sentence")]
    wlu = _wlu(_span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]))
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    d = editorial_moment_understanding_diagnostics(u)
    assert "a very specific real transcript sentence" not in str(d)


@pytest.mark.parametrize("forbidden", ["openai", "requests.", "urllib", "subprocess", "socket"])
def test_39_no_provider_call(forbidden):
    assert forbidden not in _code_only_source().lower()


def test_40_no_asr_rerun():
    tree = ast.parse(MODULE_PATH.read_text())
    imported = {n.module.rsplit(".", 1)[-1] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "asr" not in imported


def test_41_no_visual_rerun():
    tree = ast.parse(MODULE_PATH.read_text())
    imported = {n.module.rsplit(".", 1)[-1] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "local_performance" not in imported


def test_42_no_prosodic_decode():
    tree = ast.parse(MODULE_PATH.read_text())
    imported = {n.module.rsplit(".", 1)[-1] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "prosodic_audio_v2" not in imported


def test_43_no_audio_silence_rerun():
    tree = ast.parse(MODULE_PATH.read_text())
    imported = {n.module.rsplit(".", 1)[-1] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "audio_silence" not in imported


# ---------------------------------------------------------------------------
# 44-51: no authority mutation / no future-scope reach.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("forbidden", [
    "take_group_id", "family_complete_context", "selected_clip_id", "_semantic_best_take",
    "bounded_finalist_authority", "bounded_finalist_arbiter", "winner_after",
    "boundary_engine_pass", "BoundaryEngine", "dialogue_pacing_transition",
    "render_plan", "RenderSegment", "canonical_edit_plan",
])
def test_44_to_50_no_authority_mutation_strings(forbidden):
    assert forbidden not in _code_only_source()


def test_51_no_p2_whole_video_import():
    tree = ast.parse(MODULE_PATH.read_text())
    imported = {n.module.rsplit(".", 1)[-1] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "whole_video_openai" not in imported


@pytest.mark.parametrize("forbidden", ["commercial", "sales_funnel", "funnel", "cta_score"])
def test_52_53_no_commercial_or_funnel_fields(forbidden):
    assert forbidden not in _code_only_source().lower()


# ---------------------------------------------------------------------------
# 54-55: pipeline-level default-off parity / flag-on immutability.
# ---------------------------------------------------------------------------
def _pipeline_fixture():
    from cutsell_worker.contracts import MediaSignals, ProcessingRequest, SemanticLabel, SemanticRole

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


def test_54_default_off_byte_equivalent(monkeypatch):
    from cutsell_worker.pipeline import build_flow_b_draft

    monkeypatch.delenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", raising=False)
    request, takes, labels, expected_winner = _pipeline_fixture()
    result = build_flow_b_draft(request, takes, labels)
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]
    assert result.draft.diagnostics["editorial_moment_sequence"] == {"status": "disabled"}


def test_55_flag_on_winner_immutability(monkeypatch):
    from cutsell_worker.pipeline import build_flow_b_draft

    request, takes, labels, expected_winner = _pipeline_fixture()
    monkeypatch.delenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", raising=False)
    off_result = build_flow_b_draft(request, takes, labels)

    monkeypatch.setenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", "1")
    on_result = build_flow_b_draft(request, takes, labels)

    assert [c.clip_id for c in on_result.draft.selected] == [c.clip_id for c in off_result.draft.selected] == [expected_winner]
    assert on_result.draft.diagnostics["editorial_moment_sequence"]["status"] in ("evaluated",)
    assert off_result.draft.diagnostics["editorial_moment_sequence"] == {"status": "disabled"}


# ---------------------------------------------------------------------------
# 56-60: runtime / capability / missing-evidence / local-only / no-global-search.
# ---------------------------------------------------------------------------
def test_56_pipeline_runtime_lightweight():
    takes = [_take(f"c{i}", i, float(i), float(i) + 1.0) for i in range(20)]
    wlu = _wlu(*[_span(f"c{i}", float(i), float(i) + 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]) for i in range(20)])
    start = time.monotonic()
    build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0  # no invented target -- just a sanity ceiling for a synthetic 20-moment source


def test_57_capability_status_values():
    takes, wlu = _two_take_retry_fixture()
    available = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    assert available.capability_status == CAPABILITY_AVAILABLE
    not_evaluable = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=None)
    assert not_evaluable.capability_status == CAPABILITY_NOT_EVALUABLE

    partial_takes = takes + [_take("c3", 2, 2.0, 3.0)]  # c3 has no matching span
    partial = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=partial_takes, watch_listen_understanding=wlu)
    assert partial.capability_status == CAPABILITY_PARTIAL


def test_58_missing_evidence_diagnostics_shape():
    takes, wlu = _two_take_retry_fixture()
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    d = editorial_moment_understanding_diagnostics(u)
    assert "missing_evidence" in d
    assert isinstance(d["missing_evidence"], list)


def test_59_local_only_grouping_rejects_cross_source():
    m1 = build_editorial_moments_for_source(
        source_asset_id="src1", takes_for_source=[_take("c1", 0, 0.0, 1.0)],
        understanding_spans_by_id={"c1": _span("c1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT])},
    )[0]
    with pytest.raises(ValueError):
        # classify_editorial_sequence itself enforces the single-source
        # constraint -- build_editorial_sequences_for_moments never
        # bypasses it.
        from cutsell_worker.editorial_moment_sequence import classify_editorial_sequence
        other = m1[0]._replace(source_asset_id="other") if hasattr(m1[0], "_replace") else None
        if other is None:
            import dataclasses
            other = dataclasses.replace(m1[0], source_asset_id="other")
        classify_editorial_sequence([m1[0], other])


def test_60_no_global_search_across_sources():
    takes_a = [_take("a1", 0, 0.0, 1.0), _take("a2", 1, 1.0, 2.0)]
    wlu_a = _wlu(
        _span("a1", 0.0, 1.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
        _span("a2", 1.0, 2.0, behavior_labels=[BEHAVIOR_CLEAN_ATTEMPT]),
    )

    class _Take(CandidateTake):
        pass

    takes_b = [
        CandidateTake(clip_id="b1", source_asset_id="src2", source_order=0, start=0.0, end=1.0, text="x"),
    ]
    wlu_b = WatchListenUnderstanding(
        source_asset_id="src2",
        understanding_spans=(UnderstandingSpan(
            span_id="b1", source_asset_id="src2", source_start=0.0, source_end=1.0,
            behavior_state_hypotheses=(_behavior(BEHAVIOR_CLEAN_ATTEMPT),), behavior_confidence=WL_CONFIDENCE_SUPPORTED,
            attempt_boundary_hypotheses=(), attempt_relation_hypotheses=(), relation_confidence=WL_CONFIDENCE_UNKNOWN,
            meaning_completion_hypothesis=WL_MEANING_COMPLETE, performance_usability_hypothesis=USABILITY_USABLE,
            entry_usability=USABILITY_USABLE, delivery_usability=USABILITY_USABLE, exit_usability=USABILITY_USABLE,
            conflict_flags=(), evidence_provenance={},
        ),),
    )
    understandings = build_editorial_moment_understanding_for_sources(
        sources=["src1", "src2"], takes=takes_a + takes_b, watch_listen_understandings=(wlu_a, wlu_b),
    )
    src1_result = next(u for u in understandings if u.source_asset_id == "src1")
    src2_result = next(u for u in understandings if u.source_asset_id == "src2")
    # src1's sequence never includes b1, and vice versa -- each source's
    # moments/sequences are built strictly from its own evidence.
    assert all(m.source_asset_id == "src1" for m in src1_result.moments)
    assert all(m.source_asset_id == "src2" for m in src2_result.moments)
    assert "b1" not in [m.source_span_id for m in src1_result.moments]


# ---------------------------------------------------------------------------
# Run summary shape.
# ---------------------------------------------------------------------------
def test_run_summary_shape_and_missing_counts():
    takes, wlu = _two_take_retry_fixture()
    u = build_editorial_moment_understanding_for_source(source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu)
    summary = editorial_moment_understanding_run_summary([u])
    for key in (
        "p1_editorial_moment_status", "p1_missing_language_count", "p1_missing_behavior_count",
        "p1_missing_relation_count", "editorial_moment_count", "editorial_sequence_count",
    ):
        assert key in summary
    assert summary["p1_missing_language_count"] == 1
