"""D-194: P1 Editorial Moment & Sequence Understanding -- Phase A tests.

Generic, abstract fixtures only -- NO Video00 literal text/spans/ids
(CLAUDE.md's binding rule, and D-193's own generic-fixture precedent).
Covers the directive's 60-item fixture matrix plus the D-193 contract
replay (A-D) and the structural no-authority-mutation guarantees.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from cutsell_worker.editorial_moment_sequence import (
    ALLOWED_AUDIENCE_DELIVERY_STATUSES,
    ALLOWED_CONTINUITY_STATUSES,
    ALLOWED_INTERNAL_REDUNDANCY_STATUSES,
    ALLOWED_MOMENT_ROLES,
    ALLOWED_PROGRESSION_STATUSES,
    ALLOWED_RECORDING_PROCESS_STATUSES,
    ALLOWED_SEQUENCE_KINDS,
    AUDIENCE_DELIVERY_NOT_SUPPORTED,
    AUDIENCE_DELIVERY_PARTIAL,
    AUDIENCE_DELIVERY_SUPPORTED,
    AUDIENCE_DELIVERY_UNCERTAIN,
    CONTINUITY_NOT_AVAILABLE,
    CONTINUITY_NO_TRANSITION_EVIDENCE,
    CONTINUITY_TRANSITION_EVIDENCE_PRESENT,
    EARLIER_SOURCE_REDUNDANCY_NOT_EVALUATED,
    INTERNAL_REDUNDANCY_NOT_EVALUATED,
    INTERNAL_REDUNDANCY_PRESENT,
    MOMENT_ROLE_ABANDONED_ATTEMPT,
    MOMENT_ROLE_BREAKING_CHARACTER,
    MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
    MOMENT_ROLE_CONTINUATION,
    MOMENT_ROLE_CORRECTION,
    MOMENT_ROLE_FALSE_START,
    MOMENT_ROLE_NEW_AUDIENCE_BEAT,
    MOMENT_ROLE_POST_TAKE_RESET,
    MOMENT_ROLE_PRE_TAKE_SETUP,
    MOMENT_ROLE_RECORDING_PROCESS,
    MOMENT_ROLE_RETRY,
    MOMENT_ROLE_UNCERTAIN,
    PROGRESSION_FORWARD_PROGRESS,
    PROGRESSION_MIXED,
    PROGRESSION_UNKNOWN,
    RECORDING_PROCESS_ABSENT,
    RECORDING_PROCESS_PRESENT,
    SEQUENCE_KIND_BLOOPER_SERIES,
    SEQUENCE_KIND_CLEAN_DELIVERY_SEQUENCE,
    SEQUENCE_KIND_MIXED,
    SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE,
    SEQUENCE_KIND_RECORDING_PROCESS_SEQUENCE,
    SEQUENCE_KIND_RETRY_SERIES,
    SEQUENCE_KIND_UNCERTAIN,
    EditorialMoment,
    classify_editorial_moment,
    classify_editorial_sequence,
    editorial_moment_diagnostics,
    editorial_moment_sequence_run_summary,
    editorial_sequence_diagnostics,
)
from cutsell_worker.language_proposition_relation import (
    RELATION_COMPLEMENTARY,
    RELATION_CONTINUATION,
    RELATION_CORRECTION,
    RELATION_DISTINCT_PROPOSITION,
    RELATION_NEW_AUDIENCE_BEAT,
    RELATION_RETRY,
    RELATION_UNCERTAIN,
)
from cutsell_worker.language_utterance_attempt import (
    ATTEMPT_ABANDONED,
    ATTEMPT_CLEAN,
    ATTEMPT_CONTINUATION,
    ATTEMPT_CORRECTION,
    ATTEMPT_FALSE_START,
    ATTEMPT_RECORDING_PROCESS,
    ATTEMPT_UNCERTAIN,
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
    LanguageAttempt,
    MEANING_COMPLETE,
    MEANING_INCOMPLETE,
    MEANING_UNCERTAIN,
)
from cutsell_worker.raw_understanding_map import (
    BEHAVIOR_BREAKING_CHARACTER,
    BEHAVIOR_POST_TAKE_RESET,
    BEHAVIOR_PRE_TAKE_SETUP,
    BehaviorHypothesis,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER_DIR = REPO_ROOT / "cutsell_worker"
MODULE_PATH = WORKER_DIR / "editorial_moment_sequence.py"


def _code_only_source() -> str:
    """Returns the module's source with its own module docstring stripped
    -- the docstring legitimately NAMES every authority module it does NOT
    touch (module-leaf precedent set by every sibling D-15x/D-16x/D-19x
    module), so structural no-mutation/no-import checks must inspect real
    code, never prose."""
    src = MODULE_PATH.read_text()
    tree = ast.parse(src)
    docstring_node = tree.body[0] if tree.body and isinstance(tree.body[0], ast.Expr) else None
    if docstring_node is None:
        return src
    lines = src.splitlines(keepends=True)
    return "".join(lines[docstring_node.end_lineno:])


# ---------------------------------------------------------------------------
# Generic fixture factories -- abstract "attempt N" content, no real transcript.
# ---------------------------------------------------------------------------
def _attempt(
    *,
    source_asset_id: str = "src",
    attempt_id: str = "att_1",
    start: float = 0.0,
    end: float = 2.0,
    state: str = ATTEMPT_CLEAN,
    meaning: str = MEANING_COMPLETE,
    confidence: str = CONFIDENCE_SUPPORTED,
    text: str = "generic abstract statement one",
) -> LanguageAttempt:
    return LanguageAttempt(
        source_asset_id=source_asset_id,
        attempt_id=attempt_id,
        utterance_ids=(f"{attempt_id}_u0",),
        source_start=start,
        source_end=end,
        text_raw=text,
        text_normalized=text.lower(),
        attempt_state=state,
        meaning_completion=meaning,
        restart_evidence=False,
        correction_evidence=False,
        continuation_evidence=False,
        recording_process_evidence=(state == ATTEMPT_RECORDING_PROCESS),
        confidence=confidence,
        provenance="ATTEMPT_RECONSTRUCTION",
    )


def _behavior(label: str, *, confidence: float = 0.8, provenance: str = "VISUAL_SIGNAL", basis: str = "generic") -> BehaviorHypothesis:
    return BehaviorHypothesis(label=label, confidence=confidence, provenance=provenance, basis=basis)


class _FakeProsodic:
    """Minimal stand-in for ProsodicDeliveryEvidence -- only the one field
    classify_editorial_moment reads via getattr, per module docstring."""
    def __init__(self, vocal_continuity_state: str):
        self.vocal_continuity_state = vocal_continuity_state


# ---------------------------------------------------------------------------
# 1-20: moment classifier fixture matrix.
# ---------------------------------------------------------------------------
def test_01_single_clean_audience_delivery():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE))
    assert m.moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY
    assert m.audience_delivery_status == AUDIENCE_DELIVERY_SUPPORTED
    assert not m.conflict_flags


def test_02_clean_attempt_but_incomplete_meaning():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN, meaning=MEANING_INCOMPLETE))
    assert m.moment_role == MOMENT_ROLE_UNCERTAIN
    assert m.audience_delivery_status == AUDIENCE_DELIVERY_UNCERTAIN


def test_03_pre_take_setup():
    m = classify_editorial_moment(
        _attempt(state=ATTEMPT_CLEAN),
        behavior_hypotheses=(_behavior(BEHAVIOR_PRE_TAKE_SETUP),),
    )
    assert m.moment_role == MOMENT_ROLE_PRE_TAKE_SETUP


def test_04_recording_process_marker():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_RECORDING_PROCESS, meaning=MEANING_UNCERTAIN))
    assert m.moment_role == MOMENT_ROLE_RECORDING_PROCESS
    assert m.recording_process_status == RECORDING_PROCESS_PRESENT


def test_05_false_start():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_FALSE_START, meaning=MEANING_INCOMPLETE))
    assert m.moment_role == MOMENT_ROLE_FALSE_START


def test_06_abandoned_attempt():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_ABANDONED, meaning=MEANING_INCOMPLETE))
    assert m.moment_role == MOMENT_ROLE_ABANDONED_ATTEMPT


def test_07_retry():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN), relation_to_predecessor=RELATION_RETRY)
    assert m.moment_role == MOMENT_ROLE_RETRY


def test_08_correction():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CORRECTION))
    assert m.moment_role == MOMENT_ROLE_CORRECTION


def test_09_continuation():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CONTINUATION))
    assert m.moment_role == MOMENT_ROLE_CONTINUATION


def test_10_post_take_reset():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN), behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET),))
    assert m.moment_role == MOMENT_ROLE_POST_TAKE_RESET


def test_11_breaking_character():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN), behavior_hypotheses=(_behavior(BEHAVIOR_BREAKING_CHARACTER),))
    assert m.moment_role == MOMENT_ROLE_BREAKING_CHARACTER


def test_12_new_audience_beat():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN), relation_to_predecessor=RELATION_NEW_AUDIENCE_BEAT)
    assert m.moment_role == MOMENT_ROLE_NEW_AUDIENCE_BEAT


def test_13_conflicting_behavior_and_language_evidence():
    m = classify_editorial_moment(
        _attempt(state=ATTEMPT_CONTINUATION),
        behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET),),
    )
    # Structural role stays language-derived; conflict is recorded, never
    # silently resolved, and confidence is forced MIXED.
    assert m.moment_role == MOMENT_ROLE_CONTINUATION
    assert "CONTINUATION_STATE_VS_RESET_OR_BREAK_BEHAVIOR_EVIDENCE" in m.conflict_flags
    assert m.confidence == CONFIDENCE_MIXED


def test_14_missing_behavior_evidence():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN))
    assert m.moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY  # behavior_hypotheses=() default


def test_15_missing_language_attempt_raises_typeerror_without_one():
    with pytest.raises(TypeError):
        classify_editorial_moment()  # type: ignore[call-arg]


def test_16_missing_proposition_candidate_ids_defaults_empty():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN))
    assert m.proposition_candidate_ids == ()


def test_17_optional_prosodic_absent():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN))
    assert "PROSODIC_CORROBORATION" not in m.provenance


def test_18_optional_visual_absent():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN))
    assert "VISUAL_CORROBORATION" not in m.provenance


def test_19_prosodic_continuous_corroboration_no_conflict():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN), prosodic_evidence=_FakeProsodic("CONTINUOUS"))
    assert m.moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY
    assert "PROSODIC_CORROBORATION" in m.provenance
    assert not m.conflict_flags


def test_20_prosodic_interrupted_corroboration_flags_conflict():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN), prosodic_evidence=_FakeProsodic("FRAGMENTED"))
    assert m.moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY  # role never downgraded by prosody alone
    assert "PROSODIC_FRAGMENTED_CONTINUITY_VS_CLEAN_AUDIENCE_DELIVERY_STRUCTURE" in m.conflict_flags
    assert m.confidence == CONFIDENCE_MIXED


def test_20b_visual_reset_corroboration_flags_conflict():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN), visual_reset_present=True)
    assert m.moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY
    assert "VISUAL_RESET_EVIDENCE_VS_CLEAN_AUDIENCE_DELIVERY_STRUCTURE" in m.conflict_flags


def test_20c_good_prosody_alone_never_creates_clean_delivery():
    """Quality-vs-structure firewall: a non-clean attempt_state stays its
    own role even with maximally clean prosodic corroboration."""
    m = classify_editorial_moment(_attempt(state=ATTEMPT_ABANDONED, meaning=MEANING_INCOMPLETE), prosodic_evidence=_FakeProsodic("CONTINUOUS"))
    assert m.moment_role == MOMENT_ROLE_ABANDONED_ATTEMPT


def test_20d_good_visual_alone_never_creates_clean_delivery():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_RECORDING_PROCESS, meaning=MEANING_UNCERTAIN), visual_reset_present=False)
    assert m.moment_role == MOMENT_ROLE_RECORDING_PROCESS


def _clean_moment(idx: int, *, source_asset_id: str = "src", start: float = 0.0, end: float = 2.0) -> EditorialMoment:
    return classify_editorial_moment(
        _attempt(source_asset_id=source_asset_id, attempt_id=f"att_{idx}", start=start, end=end, state=ATTEMPT_CLEAN),
        local_sequence_position=idx,
    )


def _role_moment(idx: int, role_state: str, *, meaning: str = MEANING_INCOMPLETE, start: float = 0.0, end: float = 2.0, behavior=()) -> EditorialMoment:
    return classify_editorial_moment(
        _attempt(attempt_id=f"att_{idx}", start=start, end=end, state=role_state, meaning=meaning),
        behavior_hypotheses=behavior,
        local_sequence_position=idx,
    )


# ---------------------------------------------------------------------------
# 21-42: sequence classifier fixture matrix.
# ---------------------------------------------------------------------------
def test_21_multiple_clean_raw_takes_not_preassembled():
    moments = [_clean_moment(0, start=0.0, end=2.0), _clean_moment(1, start=2.1, end=4.0), _clean_moment(2, start=4.1, end=6.0)]
    seq = classify_editorial_sequence(moments)  # no relation_candidates supplied
    assert seq.sequence_kind == SEQUENCE_KIND_CLEAN_DELIVERY_SEQUENCE
    assert seq.sequence_kind != SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE


def test_22_simple_retry_series():
    moments = [
        _role_moment(0, ATTEMPT_ABANDONED, meaning=MEANING_INCOMPLETE, start=0.0, end=1.0),
        classify_editorial_moment(_attempt(attempt_id="att_1", start=1.0, end=2.0, state=ATTEMPT_CLEAN), relation_to_predecessor=RELATION_RETRY, local_sequence_position=1),
        _clean_moment(2, start=2.0, end=3.0),
    ]
    seq = classify_editorial_sequence(moments, relation_candidates=(RELATION_RETRY, RELATION_CONTINUATION))
    assert seq.sequence_kind == SEQUENCE_KIND_RETRY_SERIES


def test_23_blooper_series():
    moments = [
        _clean_moment(0, start=0.0, end=1.0),
        _role_moment(1, ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, start=1.0, end=1.5, behavior=(_behavior(BEHAVIOR_BREAKING_CHARACTER),)),
        _clean_moment(2, start=2.0, end=3.0),
    ]
    seq = classify_editorial_sequence(moments)
    assert seq.sequence_kind == SEQUENCE_KIND_BLOOPER_SERIES


def test_24_clean_delivery_sequence():
    moments = [_clean_moment(0, start=0.0, end=1.0), _clean_moment(1, start=1.0, end=2.0)]
    seq = classify_editorial_sequence(moments)
    assert seq.sequence_kind == SEQUENCE_KIND_CLEAN_DELIVERY_SEQUENCE


def test_25_genuine_preassembled_final_sequence():
    moments = [_clean_moment(0, start=0.0, end=1.0), _clean_moment(1, start=1.0, end=2.0), _clean_moment(2, start=2.0, end=3.0)]
    seq = classify_editorial_sequence(moments, relation_candidates=(RELATION_NEW_AUDIENCE_BEAT, RELATION_COMPLEMENTARY))
    assert seq.sequence_kind == SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE
    assert seq.proposition_progression_status == PROGRESSION_FORWARD_PROGRESS
    assert seq.confidence == CONFIDENCE_SUPPORTED


def test_26_false_positive_final_sequence_firewall_no_relation_evidence():
    moments = [_clean_moment(0, start=0.0, end=1.0), _clean_moment(1, start=1.0, end=2.0), _clean_moment(2, start=2.0, end=3.0)]
    seq = classify_editorial_sequence(moments)  # relation_candidates=() default
    assert seq.sequence_kind != SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE
    assert seq.proposition_progression_status == PROGRESSION_UNKNOWN


def test_27_chronology_only_cannot_classify_final():
    early = [_clean_moment(0, start=0.0, end=1.0), _clean_moment(1, start=1.0, end=2.0)]
    late = [_clean_moment(2, start=100.0, end=101.0), _clean_moment(3, start=101.0, end=102.0)]
    seq_early = classify_editorial_sequence(early)
    seq_late = classify_editorial_sequence(late)
    assert seq_early.sequence_kind == seq_late.sequence_kind == SEQUENCE_KIND_CLEAN_DELIVERY_SEQUENCE


def test_28_jump_cut_only_cannot_classify_final():
    moments = [_clean_moment(0, start=0.0, end=1.0), _clean_moment(1, start=1.0, end=2.0)]
    seq_with_cut = classify_editorial_sequence(moments, jump_cut_evidence=True)
    seq_without = classify_editorial_sequence(moments, jump_cut_evidence=None)
    assert seq_with_cut.sequence_kind == seq_without.sequence_kind == SEQUENCE_KIND_CLEAN_DELIVERY_SEQUENCE
    assert seq_with_cut.continuity_status == CONTINUITY_TRANSITION_EVIDENCE_PRESENT
    assert seq_without.continuity_status == CONTINUITY_NOT_AVAILABLE


def test_29_forward_proposition_progression():
    moments = [_clean_moment(0, start=0.0, end=1.0), _clean_moment(1, start=1.0, end=2.0)]
    seq = classify_editorial_sequence(moments, relation_candidates=(RELATION_DISTINCT_PROPOSITION,))
    assert seq.proposition_progression_status == PROGRESSION_FORWARD_PROGRESS


def test_30_repeated_proposition_not_progression():
    moments = [_clean_moment(0, start=0.0, end=1.0), _clean_moment(1, start=1.0, end=2.0)]
    seq = classify_editorial_sequence(moments, relation_candidates=(RELATION_UNCERTAIN,))
    assert seq.proposition_progression_status == PROGRESSION_UNKNOWN
    assert seq.sequence_kind != SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE


def test_31_correction_sequence_not_final():
    moments = [_clean_moment(0, start=0.0, end=1.0), _clean_moment(1, start=1.0, end=2.0)]
    seq = classify_editorial_sequence(moments, relation_candidates=(RELATION_CORRECTION,))
    assert seq.sequence_kind == SEQUENCE_KIND_RETRY_SERIES
    assert seq.sequence_kind != SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE


def test_32_retry_sequence_not_final():
    moments = [_clean_moment(0, start=0.0, end=1.0), _clean_moment(1, start=1.0, end=2.0)]
    seq = classify_editorial_sequence(moments, relation_candidates=(RELATION_RETRY,))
    assert seq.sequence_kind == SEQUENCE_KIND_RETRY_SERIES


def test_33_recording_process_interruption_prevents_final_classification():
    moments = [
        _clean_moment(0, start=0.0, end=1.0),
        _role_moment(1, ATTEMPT_RECORDING_PROCESS, meaning=MEANING_UNCERTAIN, start=1.0, end=1.5),
        _clean_moment(2, start=2.0, end=3.0),
    ]
    seq = classify_editorial_sequence(moments, relation_candidates=(RELATION_COMPLEMENTARY, RELATION_COMPLEMENTARY))
    assert seq.sequence_kind == SEQUENCE_KIND_RECORDING_PROCESS_SEQUENCE
    assert seq.sequence_kind != SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE


def test_34_incomplete_proposition_prevents_strong_final_classification():
    moments = [
        _clean_moment(0, start=0.0, end=1.0),
        _role_moment(1, ATTEMPT_CLEAN, meaning=MEANING_INCOMPLETE, start=1.0, end=2.0),
    ]
    assert moments[1].moment_role == MOMENT_ROLE_UNCERTAIN
    seq = classify_editorial_sequence(moments, relation_candidates=(RELATION_CONTINUATION,))
    assert seq.sequence_kind != SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE


def test_35_sequence_containing_one_bad_failed_moment():
    moments = [
        _clean_moment(0, start=0.0, end=1.0),
        _role_moment(1, ATTEMPT_ABANDONED, meaning=MEANING_INCOMPLETE, start=1.0, end=1.5),
        _clean_moment(2, start=2.0, end=3.0),
    ]
    seq = classify_editorial_sequence(moments)
    assert seq.sequence_kind == SEQUENCE_KIND_RETRY_SERIES
    assert seq.sequence_kind != SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE


def test_36_two_moment_sequence():
    moments = [_clean_moment(0, start=0.0, end=1.0), _clean_moment(1, start=1.0, end=2.0)]
    seq = classify_editorial_sequence(moments)
    assert len(seq.moment_ids) == 2


def test_37_five_moment_sequence():
    moments = [_clean_moment(i, start=float(i), end=float(i) + 1.0) for i in range(5)]
    seq = classify_editorial_sequence(moments)
    assert len(seq.moment_ids) == 5


def test_38_exact_source_timing_preserved():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN, start=12.5, end=17.25))
    assert m.source_start == 12.5
    assert m.source_end == 17.25


def test_39_sequence_span_exact_union():
    moments = [_clean_moment(0, start=3.0, end=5.0), _clean_moment(1, start=5.0, end=9.0)]
    seq = classify_editorial_sequence(moments)
    assert seq.source_start == 3.0
    assert seq.source_end == 9.0


def test_40_deterministic_ids():
    a1 = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN))
    a2 = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN))
    assert a1.editorial_moment_id == a2.editorial_moment_id
    seq1 = classify_editorial_sequence([_clean_moment(0), _clean_moment(1, start=2.0, end=3.0)])
    seq2 = classify_editorial_sequence([_clean_moment(0), _clean_moment(1, start=2.0, end=3.0)])
    assert seq1.sequence_id == seq2.sequence_id


def test_41_deterministic_order():
    m0 = _clean_moment(0, start=0.0, end=1.0)
    m1 = _clean_moment(1, start=1.0, end=2.0)
    seq_forward = classify_editorial_sequence([m0, m1])
    seq_reversed = classify_editorial_sequence([m1, m0])
    assert seq_forward.sequence_id == seq_reversed.sequence_id
    assert seq_forward.moment_ids == seq_reversed.moment_ids  # both re-sorted by source position


def test_42_candidate_source_id_independence_where_semantics_same():
    a = classify_editorial_moment(_attempt(attempt_id="att_alpha", state=ATTEMPT_CLEAN))
    b = classify_editorial_moment(_attempt(attempt_id="att_beta", state=ATTEMPT_CLEAN))
    # Different attempt_id -> different minted id (content+timing+role AND
    # membership-anchored), but SAME moment_role/status semantics.
    assert a.editorial_moment_id != b.editorial_moment_id
    assert a.moment_role == b.moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY


# ---------------------------------------------------------------------------
# 43-46: confidence / provenance / conflict contracts.
# ---------------------------------------------------------------------------
def test_43_confidence_categorical_only():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN))
    assert m.confidence in {CONFIDENCE_SUPPORTED, CONFIDENCE_WEAK, CONFIDENCE_MIXED, CONFIDENCE_UNKNOWN}
    assert not isinstance(m.confidence, (int, float))


def test_44_provenance_retained():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN), behavior_hypotheses=(_behavior(BEHAVIOR_PRE_TAKE_SETUP),))
    assert "LANGUAGE_ATTEMPT_STATE" in m.provenance
    assert "BEHAVIOR_HYPOTHESIS" in m.provenance


def test_45_conflict_retained():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CONTINUATION), behavior_hypotheses=(_behavior(BEHAVIOR_BREAKING_CHARACTER),))
    assert len(m.conflict_flags) == 1


def test_46_no_family_mutation():
    src = _code_only_source()
    for forbidden in ("take_group_id", "family_complete_context", "semantic_idea_id ="):
        assert forbidden not in src


def test_47_no_besttake_mutation():
    src = _code_only_source()
    for forbidden in ("selected_clip_id", "_semantic_best_take", "DeterministicBestTakeAuthority"):
        assert forbidden not in src


def test_48_no_bounded_finalist_authority_mutation():
    src = _code_only_source()
    for forbidden in ("bounded_finalist_authority", "bounded_finalist_arbiter", "winner_after"):
        assert forbidden not in src


def test_49_no_ordering_mutation():
    src = _code_only_source()
    assert "ordering_authority" not in src and "final_sibling_grouping.reorder" not in src


def test_50_no_boundary_mutation():
    src = _code_only_source()
    assert "boundary_engine_pass" not in src and "BoundaryEngine" not in src


def test_51_no_pacing_mutation():
    src = _code_only_source()
    assert "dialogue_pacing_transition" not in src


def test_52_no_render_mutation():
    src = _code_only_source()
    for forbidden in ("render_plan", "RenderSegment", "canonical_edit_plan"):
        assert forbidden not in src


def test_53_no_provider_network_call():
    src = _code_only_source()
    for forbidden in ("openai", "requests.", "urllib", "subprocess", "socket"):
        assert forbidden not in src.lower()


def test_54_no_commercial_role_fields():
    fields = {f.name for f in _dataclass_fields("EditorialMoment")} | {f.name for f in _dataclass_fields("EditorialSequenceHypothesis")}
    assert not any("commercial" in f.lower() or "sales" in f.lower() or "funnel" in f.lower() for f in fields)


def test_55_no_funnel_fields():
    src = _code_only_source()
    assert "funnel" not in src.lower() and "cta_score" not in src.lower()


def test_56_no_master_score():
    src = _code_only_source()
    assert "master_score" not in src.lower()
    # confidence fields are always the categorical vocabulary, never numeric.
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN))
    assert isinstance(m.confidence, str)


def test_57_no_automatic_supersession_or_deletion():
    seq = classify_editorial_sequence([_clean_moment(0), _clean_moment(1, start=2.0, end=3.0)], internal_redundancy_status=INTERNAL_REDUNDANCY_PRESENT)
    assert seq.internal_redundancy_status == INTERNAL_REDUNDANCY_PRESENT
    assert seq.earlier_source_redundancy_status == EARLIER_SOURCE_REDUNDANCY_NOT_EVALUATED
    # No deletion side effect is even possible: the function is pure and
    # returns a new frozen object, never mutates its inputs.


def test_58_no_chronology_winner_rule():
    early = classify_editorial_sequence([_clean_moment(0, start=0.0, end=1.0), _clean_moment(1, start=1.0, end=2.0)])
    late = classify_editorial_sequence([_clean_moment(2, start=500.0, end=501.0), _clean_moment(3, start=501.0, end=502.0)])
    assert early.sequence_kind == late.sequence_kind  # position never decides kind


def test_59_no_clean_equals_final_rule():
    moments = [_clean_moment(0, start=0.0, end=1.0), _clean_moment(1, start=1.0, end=2.0), _clean_moment(2, start=2.0, end=3.0)]
    seq = classify_editorial_sequence(moments)  # clean, but no relation evidence
    assert seq.sequence_kind != SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE


def test_60_whole_video_distant_redundancy_stays_not_evaluated():
    seq = classify_editorial_sequence([_clean_moment(0), _clean_moment(1, start=2.0, end=3.0)])
    assert seq.earlier_source_redundancy_status == EARLIER_SOURCE_REDUNDANCY_NOT_EVALUATED


# ---------------------------------------------------------------------------
# D-193 contract replay (A-D).
# ---------------------------------------------------------------------------
def test_d193_contract_a_several_clean_takes_not_preassembled():
    moments = [_clean_moment(i, start=float(i) * 2, end=float(i) * 2 + 1.5) for i in range(3)]
    seq = classify_editorial_sequence(moments)
    assert seq.sequence_kind != SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE


def test_d193_contract_b_failed_then_retry_then_clean():
    moments = [
        _role_moment(0, ATTEMPT_ABANDONED, meaning=MEANING_INCOMPLETE, start=0.0, end=1.0),
        classify_editorial_moment(_attempt(attempt_id="att_1", start=1.0, end=2.0, state=ATTEMPT_CLEAN), relation_to_predecessor=RELATION_RETRY, local_sequence_position=1),
        _clean_moment(2, start=2.0, end=3.0),
    ]
    seq = classify_editorial_sequence(moments, relation_candidates=(RELATION_RETRY, RELATION_CONTINUATION))
    assert seq.sequence_kind in {SEQUENCE_KIND_RETRY_SERIES, SEQUENCE_KIND_BLOOPER_SERIES}


def test_d193_contract_c_forward_progressing_clean_moments_supported():
    moments = [_clean_moment(i, start=float(i), end=float(i) + 1.0) for i in range(3)]
    seq = classify_editorial_sequence(moments, relation_candidates=(RELATION_NEW_AUDIENCE_BEAT, RELATION_COMPLEMENTARY))
    assert seq.sequence_kind == SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE
    assert seq.confidence in {CONFIDENCE_SUPPORTED}


def test_d193_contract_d_final_like_with_retry_relation_not_supported():
    moments = [_clean_moment(0, start=0.0, end=1.0), _clean_moment(1, start=1.0, end=2.0)]
    seq = classify_editorial_sequence(moments, relation_candidates=(RELATION_RETRY,))
    assert seq.sequence_kind != SEQUENCE_KIND_PREASSEMBLED_FINAL_SEQUENCE


# ---------------------------------------------------------------------------
# Diagnostics / run summary shape.
# ---------------------------------------------------------------------------
def test_diagnostics_moment_shape():
    # D-196's own required per-moment trace field list.
    d = editorial_moment_diagnostics(_clean_moment(0))
    for key in (
        "editorial_moment_id", "source_asset_id", "source_start", "source_end",
        "attempt_ids", "proposition_candidate_ids", "moment_role", "audience_delivery_status",
        "recording_process_status", "completion_status", "confidence", "conflict", "provenance",
    ):
        assert key in d
    assert isinstance(d["attempt_ids"], list)
    assert isinstance(d["proposition_candidate_ids"], list)
    assert isinstance(d["provenance"], list)


def test_diagnostics_sequence_shape():
    # D-196's own required per-sequence trace field list.
    seq = classify_editorial_sequence([_clean_moment(0), _clean_moment(1, start=2.0, end=3.0)])
    d = editorial_sequence_diagnostics(seq)
    for key in (
        "sequence_id", "source_asset_id", "source_start", "source_end", "moment_ids", "moment_count",
        "sequence_kind", "sequence_completeness", "proposition_progression_status",
        "internal_redundancy_status", "continuity_status", "earlier_source_redundancy_status",
        "confidence", "conflict", "provenance",
    ):
        assert key in d
    assert isinstance(d["moment_ids"], list)
    assert isinstance(d["provenance"], list)


def test_run_summary_counts():
    moments = [_clean_moment(0), _role_moment(1, ATTEMPT_ABANDONED, meaning=MEANING_INCOMPLETE, start=2.0, end=2.5)]
    seqs = [classify_editorial_sequence(moments)]
    summary = editorial_moment_sequence_run_summary(moments, seqs)
    assert summary["editorial_moment_count"] == 2
    assert summary["clean_audience_delivery_count"] == 1
    assert summary["abandoned_attempt_count"] == 1
    assert summary["editorial_sequence_count"] == 1
    assert summary["retry_series_count"] == 1


def test_no_transcript_dump_in_diagnostics():
    m = classify_editorial_moment(_attempt(state=ATTEMPT_CLEAN, text="a very specific real transcript sentence"))
    d = editorial_moment_diagnostics(m)
    assert "a very specific real transcript sentence" not in str(d)


# ---------------------------------------------------------------------------
# Vocabulary completeness / allowed-set sanity.
# ---------------------------------------------------------------------------
def test_allowed_vocabularies_are_frozensets_of_str():
    for vocab in (
        ALLOWED_MOMENT_ROLES, ALLOWED_SEQUENCE_KINDS, ALLOWED_AUDIENCE_DELIVERY_STATUSES,
        ALLOWED_RECORDING_PROCESS_STATUSES, ALLOWED_PROGRESSION_STATUSES,
        ALLOWED_INTERNAL_REDUNDANCY_STATUSES, ALLOWED_CONTINUITY_STATUSES,
    ):
        assert isinstance(vocab, frozenset)
        assert all(isinstance(v, str) for v in vocab)


def test_sequence_requires_two_or_more_moments():
    with pytest.raises(ValueError):
        classify_editorial_sequence([_clean_moment(0)])


def test_sequence_requires_same_source_asset_id():
    with pytest.raises(ValueError):
        classify_editorial_sequence([
            _clean_moment(0, source_asset_id="src_a"),
            _clean_moment(1, source_asset_id="src_b", start=2.0, end=3.0),
        ])


def test_internal_redundancy_default_not_evaluated():
    seq = classify_editorial_sequence([_clean_moment(0), _clean_moment(1, start=2.0, end=3.0)])
    assert seq.internal_redundancy_status == INTERNAL_REDUNDANCY_NOT_EVALUATED


def test_internal_redundancy_rejects_unknown_value():
    with pytest.raises(ValueError):
        classify_editorial_sequence([_clean_moment(0), _clean_moment(1, start=2.0, end=3.0)], internal_redundancy_status="NOT_A_REAL_STATUS")


# ---------------------------------------------------------------------------
# Module-leaf structural guarantees: this module (D-194's own pure
# classifiers) is imported by NOTHING in production, and imports NOTHING
# from any authority module. `pipeline.py` is deliberately EXCLUDED from
# this list as of D-195 (docs/CUTSELL_DECISIONS.md D-195), which
# explicitly authorizes wiring a NEW, SEPARATE integration module
# (`editorial_moment_sequence_integration.py`) into `pipeline.py`'s own
# diagnostics -- see `test_cutsell_d195_editorial_moment_sequence_
# integration.py`'s own module-leaf tests for the D-195-era version of
# this guarantee (pipeline.py may reference the INTEGRATION module;
# D-194's own classifiers module must still never be imported directly,
# and no authority/Family/BestTake/Boundary/Pacing/render module may
# ever reference either one).
# ---------------------------------------------------------------------------
_PRODUCTION_MODULES = (
    "flow_b.py", "bounded_finalist_arbiter.py", "bounded_finalist_authority.py",
    "composite_resolver.py", "realization_resolver.py", "boundary_engine_pass.py",
    "dialogue_pacing_transition.py", "take_grouping.py", "take_grouping_provider.py",
    "deterministic_best_take_authority.py", "take_judge.py", "semantic_authority_observability.py",
    "whole_video_openai.py",
)


@pytest.mark.parametrize("module_name", _PRODUCTION_MODULES)
def test_not_imported_by_any_production_module(module_name: str):
    path = WORKER_DIR / module_name
    if not path.exists():
        pytest.skip(f"{module_name} not present in this checkout")
    src = path.read_text()
    assert "editorial_moment_sequence" not in src


def test_pipeline_never_imports_d194_classifiers_directly():
    """D-195 wires a SEPARATE integration module into pipeline.py -- this
    module's own pure classifiers (editorial_moment_sequence.py) are never
    imported directly by pipeline.py; the one-hop indirection through
    editorial_moment_sequence_integration.py is deliberate (module-leaf
    isolation)."""
    tree = ast.parse((WORKER_DIR / "pipeline.py").read_text())
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.rsplit(".", 1)[-1])
    assert "editorial_moment_sequence" not in imported_modules
    assert "editorial_moment_sequence_integration" in imported_modules


def test_module_imports_nothing_from_authority_modules():
    tree = ast.parse((WORKER_DIR / "editorial_moment_sequence.py").read_text())
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.rsplit(".", 1)[-1])
    forbidden = {
        "pipeline", "flow_b", "bounded_finalist_arbiter", "bounded_finalist_authority",
        "composite_resolver", "realization_resolver", "boundary_engine_pass",
        "dialogue_pacing_transition", "take_grouping", "take_grouping_provider",
        "deterministic_best_take_authority", "take_judge", "semantic_authority_observability",
        "whole_video_openai",
    }
    assert not (imported_modules & forbidden)


def test_module_has_no_feature_flag():
    src = _code_only_source()
    assert "os.environ" not in src and "CUTSELL_" not in src


def _dataclass_fields(class_name: str):
    import dataclasses
    import cutsell_worker.editorial_moment_sequence as mod
    cls = getattr(mod, class_name)
    return dataclasses.fields(cls)
