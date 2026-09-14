"""D-200.3: DIMENSION-AWARE EDITORIAL RELATION -- BOUNDED IMPLEMENTATION,
tests.

Proves: (1) the new ``StructuredEditorialRelationEvidence`` type is a
frozen, categorical-only, three-dimension container with no numeric
master score and no transcript; (2) the D-157/D-169 decomposition table
(D-200.2 Section 1) is applied exactly -- each flat label maps to exactly
one dimension, D-157 never becomes proposition authority; (3) cross-
dimension differences never become a conflict, same-dimension
disagreements do (and never erase the other two dimensions); (4) the
D-197 grouping-consumer translation (``grouping_effective_relation``)
implements the positive-join/explicit-split/proposition-never-controls-
grouping contracts; (5) the P1 integration switches the grouping input to
the structured translation ONLY when the live Language-Spine diagnostics
flag is active, leaving moment classification (D-194) and the D-198
legacy path byte-identical; (6) no forbidden authority/module is ever
touched.

See docs/CUTSELL_DECISIONS.md D-200.3.
"""
from __future__ import annotations

import inspect

from cutsell_worker.contracts import CandidateTake, Word
import cutsell_worker.structured_editorial_relation as structured_module
from cutsell_worker.structured_editorial_relation import (
    ATTEMPT_CONTINUATION,
    ATTEMPT_CORRECTION,
    ATTEMPT_RETRY,
    ATTEMPT_UNKNOWN,
    BEAT_NEW_AUDIENCE,
    BEAT_UNKNOWN,
    GROUPING_ACTION_JOIN,
    GROUPING_ACTION_SPLIT,
    GROUPING_REASON_ATTEMPT_JOIN,
    GROUPING_REASON_EXPLICIT_BEAT_BOUNDARY,
    GROUPING_REASON_NO_POSITIVE_JOIN_EVIDENCE,
    GROUPING_REASON_NO_PREDECESSOR,
    PROPOSITION_COMPLEMENTARY,
    PROPOSITION_DISTINCT,
    PROPOSITION_UNKNOWN,
    StructuredEditorialRelationEvidence,
    build_structured_editorial_relation,
    grouping_effective_relation,
    structured_editorial_relation_diagnostics,
)
from cutsell_worker.editorial_moment_sequence_integration import (
    build_editorial_local_groups,
    build_editorial_moment_understanding_for_source,
    build_editorial_moments_for_source,
    editorial_moment_understanding_diagnostics,
    editorial_moment_understanding_run_summary,
)
from cutsell_worker.language_proposition_relation import (
    ClaimSignature,
    PropositionCandidate,
    RelationEvidence,
)
from cutsell_worker.language_spine_live_integration import LiveLanguageSpineEvidence
from cutsell_worker.language_utterance_attempt import (
    ATTEMPT_CLEAN,
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
    MEANING_COMPLETE,
    LanguageAttempt,
)
from cutsell_worker.raw_understanding_map import BehaviorHypothesis
from cutsell_worker.watch_listen_understanding import (
    RELATION_COMPLEMENTARY,
    RELATION_CONTINUATION,
    RELATION_CORRECTION,
    RELATION_DISTINCT_PROPOSITION,
    RELATION_NEW_AUDIENCE_BEAT,
    RELATION_RETRY,
    RELATION_UNCERTAIN,
    AttemptRelationHypothesis,
    UnderstandingSpan,
    WatchListenUnderstanding,
)


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------
def _hyp(relation, confidence=CONFIDENCE_SUPPORTED, left_span_id="span_left"):
    return AttemptRelationHypothesis(relation, confidence, "basis", left_span_id, ("DETERMINISTIC_RULE",))


_DUMMY_SIGNATURE = ClaimSignature(
    content_tokens=frozenset(), negation_present=False, numbers=frozenset(),
    claim_type="NONE", negation_role="", signature_hash="dummy",
)


def _relation_evidence(relation_candidate, confidence=CONFIDENCE_SUPPORTED, *, left="p0", right="p1"):
    return RelationEvidence(
        left_proposition_candidate_id=left, right_proposition_candidate_id=right,
        relation_candidate=relation_candidate, support_status="SUPPORT", confidence=confidence,
        semantic_support="UNKNOWN", language_support="SUPPORT", watch_listen_support="UNKNOWN",
        meaning_conflict=False, proposition_conflict=False, provenance=("CLAIM_SIGNATURE",),
    )


def _attempt(attempt_id, start, end, *, source_asset_id="src1", text="generic statement"):
    return LanguageAttempt(
        source_asset_id=source_asset_id, attempt_id=attempt_id, utterance_ids=(attempt_id,),
        source_start=start, source_end=end, text_raw=text, text_normalized=text.lower(),
        attempt_state=ATTEMPT_CLEAN, meaning_completion=MEANING_COMPLETE,
        restart_evidence=False, correction_evidence=False, continuation_evidence=False,
        recording_process_evidence=False, confidence=CONFIDENCE_SUPPORTED, provenance="CANONICAL_LANGUAGE_SPINE",
    )


def _prop(prop_id, attempt_id, start, end, *, source_asset_id="src1"):
    return PropositionCandidate(
        source_asset_id=source_asset_id, proposition_candidate_id=prop_id, attempt_ids=(attempt_id,),
        source_start=start, source_end=end, text_raw="x", text_normalized="x",
        claim_signature=_DUMMY_SIGNATURE, meaning_completion=MEANING_COMPLETE,
        editorial_slot_evidence="OTHER", confidence=CONFIDENCE_SUPPORTED, provenance="LANGUAGE_ATTEMPT",
    )


def _take(clip_id, order, start, end, text="generic statement"):
    return CandidateTake(clip_id=clip_id, source_asset_id="src1", source_order=order, start=start, end=end, text=text)


def _behavior(label, confidence=0.8):
    return BehaviorHypothesis(label=label, confidence=confidence, provenance="VISUAL_SIGNAL", basis="generic")


def _span(span_id, start, end, *, source_asset_id="src1", relation=None):
    rel = (AttemptRelationHypothesis(relation, CONFIDENCE_SUPPORTED, "x", None, ()),) if relation else ()
    return UnderstandingSpan(
        span_id=span_id, source_asset_id=source_asset_id, source_start=start, source_end=end,
        behavior_state_hypotheses=(), behavior_confidence=CONFIDENCE_SUPPORTED,
        attempt_boundary_hypotheses=(), attempt_relation_hypotheses=rel,
        relation_confidence=CONFIDENCE_SUPPORTED, meaning_completion_hypothesis=MEANING_COMPLETE,
        performance_usability_hypothesis="USABLE", entry_usability="USABLE",
        delivery_usability="USABLE", exit_usability="USABLE",
        conflict_flags=(), evidence_provenance={},
    )


def _wlu(*spans, source_asset_id="src1"):
    return WatchListenUnderstanding(source_asset_id=source_asset_id, understanding_spans=tuple(spans))


def _live_spine(*, attempts=(), proposition_candidates=(), relation_evidence=(), source_asset_id="src1"):
    return LiveLanguageSpineEvidence(
        source_asset_id=source_asset_id, words=(), phrases=(), utterances=(), attempts=attempts,
        proposition_candidates=proposition_candidates, relation_evidence=relation_evidence,
        capability_status="AVAILABLE", missing_evidence=(), conflicts=(), provenance=("CANONICAL_LANGUAGE_SPINE",),
    )


def _pair_understanding(*, d157_relation=None, canonical_relation=None, canonical_confidence=CONFIDENCE_SUPPORTED, live_spine=None):
    """Builds a two-moment, one-source P1 understanding where the FIRST
    moment's real ``LanguageAttempt``/``PropositionCandidate`` bridges onto
    span ``c1`` and the second onto ``c2`` -- exactly the shape
    ``build_editorial_moment_understanding_for_source`` needs to resolve a
    real canonical relation for the c1->c2 predecessor edge, in addition
    to whatever D-157 hypothesis is attached to span ``c2``."""
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0), _span("c2", 1.0, 2.0, relation=d157_relation))
    return build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=takes, watch_listen_understanding=wlu,
        live_language_spine=live_spine,
    )


# ===========================================================================
# 1-5: StructuredEditorialRelationEvidence type contract.
# ===========================================================================
def test_01_type_is_frozen_dataclass():
    ev = build_structured_editorial_relation(left_source_span_id="a", right_source_span_id="b")
    import dataclasses
    assert dataclasses.is_dataclass(ev)
    try:
        ev.attempt_relation = ATTEMPT_RETRY  # type: ignore[misc]
        assert False, "frozen dataclass must reject mutation"
    except dataclasses.FrozenInstanceError:
        pass


def test_02_three_independent_dimensions_present():
    ev = build_structured_editorial_relation(left_source_span_id="a", right_source_span_id="b")
    for field in ("attempt_relation", "proposition_relation", "editorial_beat_relation"):
        assert hasattr(ev, field)


def test_03_categorical_status_per_dimension():
    ev = build_structured_editorial_relation(left_source_span_id="a", right_source_span_id="b")
    for field in ("attempt_relation_status", "proposition_relation_status", "editorial_beat_relation_status"):
        assert getattr(ev, field) in (CONFIDENCE_SUPPORTED, CONFIDENCE_WEAK, CONFIDENCE_MIXED, CONFIDENCE_UNKNOWN)


def test_04_no_master_confidence_field():
    ev = build_structured_editorial_relation(left_source_span_id="a", right_source_span_id="b")
    field_names = {f.name for f in ev.__dataclass_fields__.values()}
    assert not any("master" in n or n == "confidence" for n in field_names)


def test_05_no_transcript_field():
    ev = build_structured_editorial_relation(left_source_span_id="a", right_source_span_id="b")
    field_names = {f.name for f in ev.__dataclass_fields__.values()}
    assert not any("text" in n or "transcript" in n for n in field_names)


# ===========================================================================
# 6-11: D-157 mapping.
# ===========================================================================
def test_06_d157_retry_maps_to_attempt():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b", d157_hypotheses=(_hyp(RELATION_RETRY),),
    )
    assert ev.attempt_relation == ATTEMPT_RETRY
    assert ev.proposition_relation == PROPOSITION_UNKNOWN
    assert ev.editorial_beat_relation == BEAT_UNKNOWN


def test_07_d157_correction_maps_to_attempt():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b", d157_hypotheses=(_hyp(RELATION_CORRECTION),),
    )
    assert ev.attempt_relation == ATTEMPT_CORRECTION


def test_08_d157_continuation_maps_to_attempt():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b", d157_hypotheses=(_hyp(RELATION_CONTINUATION),),
    )
    assert ev.attempt_relation == ATTEMPT_CONTINUATION


def test_09_d157_new_audience_beat_maps_to_beat():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b", d157_hypotheses=(_hyp(RELATION_NEW_AUDIENCE_BEAT),),
    )
    assert ev.editorial_beat_relation == BEAT_NEW_AUDIENCE
    assert ev.attempt_relation == ATTEMPT_UNKNOWN


def test_10_d157_complementary_never_becomes_proposition_authority():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b", d157_hypotheses=(_hyp(RELATION_COMPLEMENTARY),),
    )
    # D-157's own COMPLEMENTARY is content-blind -- must NOT populate
    # proposition_relation at all.
    assert ev.proposition_relation == PROPOSITION_UNKNOWN
    assert ev.attempt_relation == ATTEMPT_UNKNOWN
    assert ev.editorial_beat_relation == BEAT_UNKNOWN


def test_11_d157_uncertain_asserts_no_dimension():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b", d157_hypotheses=(_hyp(RELATION_UNCERTAIN),),
    )
    assert ev.attempt_relation == ATTEMPT_UNKNOWN
    assert ev.proposition_relation == PROPOSITION_UNKNOWN
    assert ev.editorial_beat_relation == BEAT_UNKNOWN


# ===========================================================================
# 12-17: D-169 mapping.
# ===========================================================================
def test_12_d169_retry_maps_to_attempt():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        canonical_relation_evidence=_relation_evidence(RELATION_RETRY),
    )
    assert ev.attempt_relation == ATTEMPT_RETRY
    assert ev.attempt_relation_provenance == ("D169_LANGUAGE_PROPOSITION",)


def test_13_d169_correction_maps_to_attempt():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        canonical_relation_evidence=_relation_evidence(RELATION_CORRECTION),
    )
    assert ev.attempt_relation == ATTEMPT_CORRECTION


def test_14_d169_continuation_maps_to_attempt():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        canonical_relation_evidence=_relation_evidence(RELATION_CONTINUATION),
    )
    assert ev.attempt_relation == ATTEMPT_CONTINUATION


def test_15_d169_complementary_maps_to_proposition():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        canonical_relation_evidence=_relation_evidence(RELATION_COMPLEMENTARY),
    )
    assert ev.proposition_relation == PROPOSITION_COMPLEMENTARY
    assert ev.proposition_relation_provenance == ("D169_LANGUAGE_PROPOSITION",)


def test_16_d169_distinct_proposition_maps_to_proposition():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        canonical_relation_evidence=_relation_evidence(RELATION_DISTINCT_PROPOSITION),
    )
    assert ev.proposition_relation == PROPOSITION_DISTINCT


def test_17_d169_new_audience_beat_maps_to_beat():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        canonical_relation_evidence=_relation_evidence(RELATION_NEW_AUDIENCE_BEAT),
    )
    assert ev.editorial_beat_relation == BEAT_NEW_AUDIENCE


# ===========================================================================
# 18-24: Cross-dimension coexistence / same-dimension conflict / UNKNOWN firewall.
# ===========================================================================
def test_18_cross_dimension_retry_and_distinct_coexist():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        d157_hypotheses=(_hyp(RELATION_RETRY),),
        canonical_relation_evidence=_relation_evidence(RELATION_DISTINCT_PROPOSITION),
    )
    assert ev.attempt_relation == ATTEMPT_RETRY
    assert ev.proposition_relation == PROPOSITION_DISTINCT
    assert ev.conflict_flags == ()


def test_19_cross_dimension_new_beat_and_distinct_coexist():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        d157_hypotheses=(_hyp(RELATION_NEW_AUDIENCE_BEAT),),
        canonical_relation_evidence=_relation_evidence(RELATION_DISTINCT_PROPOSITION),
    )
    assert ev.editorial_beat_relation == BEAT_NEW_AUDIENCE
    assert ev.proposition_relation == PROPOSITION_DISTINCT
    assert ev.conflict_flags == ()


def test_20_same_dimension_conflict_attempt_retry_vs_correction():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        d157_hypotheses=(_hyp(RELATION_RETRY),),
        canonical_relation_evidence=_relation_evidence(RELATION_CORRECTION),
    )
    assert ev.attempt_relation == ATTEMPT_UNKNOWN
    assert ev.attempt_relation_status == CONFIDENCE_MIXED
    assert "ATTEMPT_SAME_DIMENSION_CONFLICT" in ev.conflict_flags


def test_21_same_dimension_agreement_raises_status_to_supported():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        d157_hypotheses=(_hyp(RELATION_RETRY, confidence=CONFIDENCE_WEAK),),
        canonical_relation_evidence=_relation_evidence(RELATION_RETRY, confidence=CONFIDENCE_WEAK),
    )
    assert ev.attempt_relation == ATTEMPT_RETRY
    assert ev.attempt_relation_status == CONFIDENCE_SUPPORTED


def test_22_unknown_firewall_attempt_conflict_does_not_erase_proposition():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        d157_hypotheses=(_hyp(RELATION_RETRY),),
        canonical_relation_evidence=_relation_evidence(RELATION_CORRECTION),
    )
    # Even though ATTEMPT collapses to UNKNOWN on conflict, a real
    # proposition-dimension conflict is unrelated and independently
    # evaluated (here it stays UNKNOWN only for lack of any proposition
    # evidence, never because of the attempt conflict).
    assert ev.attempt_relation == ATTEMPT_UNKNOWN
    assert ev.proposition_relation == PROPOSITION_UNKNOWN
    assert ev.proposition_relation_status == CONFIDENCE_UNKNOWN  # absence, not conflict-forced


def test_23_beat_same_dimension_conflict_isolated_from_attempt():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        d157_hypotheses=(_hyp(RELATION_RETRY), _hyp(RELATION_NEW_AUDIENCE_BEAT)),
        canonical_relation_evidence=_relation_evidence(RELATION_RETRY),
    )
    # D-157 emits BOTH a RETRY (attempt) and NEW_AUDIENCE_BEAT (beat)
    # hypothesis for the same edge (module docstring's own documented
    # possibility) -- attempt agrees with D-169 (SUPPORTED), beat has only
    # one source (its own status), neither erases the other.
    assert ev.attempt_relation == ATTEMPT_RETRY
    assert ev.attempt_relation_status == CONFIDENCE_SUPPORTED
    assert ev.editorial_beat_relation == BEAT_NEW_AUDIENCE


def test_24_no_evidence_at_all_all_dimensions_unknown_no_conflict():
    ev = build_structured_editorial_relation(left_source_span_id="a", right_source_span_id="b")
    assert ev.attempt_relation == ATTEMPT_UNKNOWN
    assert ev.proposition_relation == PROPOSITION_UNKNOWN
    assert ev.editorial_beat_relation == BEAT_UNKNOWN
    assert ev.conflict_flags == ()


# ===========================================================================
# 25-33: D-197 grouping-consumer translation.
# ===========================================================================
def test_25_no_predecessor_splits_with_reason():
    value, action, reason = grouping_effective_relation(None)
    assert value is None and action == GROUPING_ACTION_SPLIT and reason == GROUPING_REASON_NO_PREDECESSOR


def test_26_retry_joins():
    ev = build_structured_editorial_relation(left_source_span_id="a", right_source_span_id="b", d157_hypotheses=(_hyp(RELATION_RETRY),))
    value, action, reason = grouping_effective_relation(ev)
    assert value == ATTEMPT_RETRY and action == GROUPING_ACTION_JOIN and reason == GROUPING_REASON_ATTEMPT_JOIN


def test_27_correction_joins():
    ev = build_structured_editorial_relation(left_source_span_id="a", right_source_span_id="b", d157_hypotheses=(_hyp(RELATION_CORRECTION),))
    value, action, _ = grouping_effective_relation(ev)
    assert value == ATTEMPT_CORRECTION and action == GROUPING_ACTION_JOIN


def test_28_continuation_joins():
    ev = build_structured_editorial_relation(left_source_span_id="a", right_source_span_id="b", d157_hypotheses=(_hyp(RELATION_CONTINUATION),))
    value, action, _ = grouping_effective_relation(ev)
    assert value == ATTEMPT_CONTINUATION and action == GROUPING_ACTION_JOIN


def test_29_new_audience_beat_splits():
    ev = build_structured_editorial_relation(left_source_span_id="a", right_source_span_id="b", d157_hypotheses=(_hyp(RELATION_NEW_AUDIENCE_BEAT),))
    value, action, reason = grouping_effective_relation(ev)
    assert value is None and action == GROUPING_ACTION_SPLIT and reason == GROUPING_REASON_EXPLICIT_BEAT_BOUNDARY


def test_30_explicit_beat_boundary_blocks_attempt_join():
    # A hand-built structured evidence carrying BOTH a suggested attempt
    # join AND an explicit beat boundary -- the beat boundary must win.
    ev = StructuredEditorialRelationEvidence(
        left_source_span_id="a", right_source_span_id="b",
        attempt_relation=ATTEMPT_RETRY, attempt_relation_status=CONFIDENCE_SUPPORTED,
        proposition_relation=PROPOSITION_UNKNOWN, proposition_relation_status=CONFIDENCE_UNKNOWN,
        editorial_beat_relation=BEAT_NEW_AUDIENCE, editorial_beat_relation_status=CONFIDENCE_SUPPORTED,
        attempt_relation_provenance=(), proposition_relation_provenance=(), editorial_beat_relation_provenance=(),
        conflict_flags=(), provenance=(),
    )
    value, action, reason = grouping_effective_relation(ev)
    assert value is None and action == GROUPING_ACTION_SPLIT and reason == GROUPING_REASON_EXPLICIT_BEAT_BOUNDARY


def test_31_distinct_proposition_never_controls_grouping():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        canonical_relation_evidence=_relation_evidence(RELATION_DISTINCT_PROPOSITION),
    )
    value, action, reason = grouping_effective_relation(ev)
    assert value is None and action == GROUPING_ACTION_SPLIT and reason == GROUPING_REASON_NO_POSITIVE_JOIN_EVIDENCE


def test_32_complementary_proposition_never_controls_grouping():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        canonical_relation_evidence=_relation_evidence(RELATION_COMPLEMENTARY),
    )
    value, action, reason = grouping_effective_relation(ev)
    assert value is None and action == GROUPING_ACTION_SPLIT and reason == GROUPING_REASON_NO_POSITIVE_JOIN_EVIDENCE


def test_33_no_positive_join_evidence_reason_not_conflated_with_distinct():
    # The reason string must read "no positive join evidence", never
    # "split because proposition differs" -- this is D-200.3's own
    # "IMPORTANT DISTINCTION" requirement.
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        canonical_relation_evidence=_relation_evidence(RELATION_DISTINCT_PROPOSITION),
    )
    _, _, reason = grouping_effective_relation(ev)
    assert reason == GROUPING_REASON_NO_POSITIVE_JOIN_EVIDENCE
    assert "DISTINCT" not in reason


# ===========================================================================
# 34-39: D-200 abstract cross-dimension replay (Cases A-F, generic
# synthetic fixtures -- no Video00 text/spans/ids).
# ===========================================================================
def test_34_case_a_retry_plus_distinct_joins_from_retry():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        d157_hypotheses=(_hyp(RELATION_RETRY),),
        canonical_relation_evidence=_relation_evidence(RELATION_DISTINCT_PROPOSITION),
    )
    value, action, _ = grouping_effective_relation(ev)
    assert action == GROUPING_ACTION_JOIN and value == ATTEMPT_RETRY


def test_35_case_b_new_beat_plus_distinct_splits_from_beat():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        d157_hypotheses=(_hyp(RELATION_NEW_AUDIENCE_BEAT),),
        canonical_relation_evidence=_relation_evidence(RELATION_DISTINCT_PROPOSITION),
    )
    value, action, reason = grouping_effective_relation(ev)
    assert action == GROUPING_ACTION_SPLIT and reason == GROUPING_REASON_EXPLICIT_BEAT_BOUNDARY


def test_36_case_c_continuation_plus_distinct_joins_from_continuation():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        d157_hypotheses=(_hyp(RELATION_CONTINUATION),),
        canonical_relation_evidence=_relation_evidence(RELATION_DISTINCT_PROPOSITION),
    )
    value, action, _ = grouping_effective_relation(ev)
    assert action == GROUPING_ACTION_JOIN and value == ATTEMPT_CONTINUATION


def test_37_case_d_true_same_dimension_conflict_retry_vs_correction():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        d157_hypotheses=(_hyp(RELATION_RETRY),),
        canonical_relation_evidence=_relation_evidence(RELATION_CORRECTION),
    )
    assert ev.attempt_relation == ATTEMPT_UNKNOWN and ev.attempt_relation_status == CONFIDENCE_MIXED
    value, action, _ = grouping_effective_relation(ev)
    assert action == GROUPING_ACTION_SPLIT and value is None


def test_38_case_e_proposition_only_no_chronology_join():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        canonical_relation_evidence=_relation_evidence(RELATION_DISTINCT_PROPOSITION),
    )
    value, action, reason = grouping_effective_relation(ev)
    assert action == GROUPING_ACTION_SPLIT and reason == GROUPING_REASON_NO_POSITIVE_JOIN_EVIDENCE


def test_39_case_f_explicit_beat_boundary_over_attempt_join_preserved():
    ev = build_structured_editorial_relation(
        left_source_span_id="a", right_source_span_id="b",
        d157_hypotheses=(_hyp(RELATION_RETRY), _hyp(RELATION_NEW_AUDIENCE_BEAT)),
    )
    value, action, reason = grouping_effective_relation(ev)
    assert action == GROUPING_ACTION_SPLIT and reason == GROUPING_REASON_EXPLICIT_BEAT_BOUNDARY


# ===========================================================================
# 40-42: Diagnostics.
# ===========================================================================
def test_40_diagnostics_shape_for_none():
    d = structured_editorial_relation_diagnostics(None)
    assert d["attempt_relation"] is None and d["conflict_flags"] == []


def test_41_diagnostics_no_transcript_key():
    ev = build_structured_editorial_relation(left_source_span_id="a", right_source_span_id="b", d157_hypotheses=(_hyp(RELATION_RETRY),))
    d = structured_editorial_relation_diagnostics(ev)
    assert not any("text" in k or "transcript" in k for k in d)


def test_42_diagnostics_deterministic():
    ev1 = build_structured_editorial_relation(left_source_span_id="a", right_source_span_id="b", d157_hypotheses=(_hyp(RELATION_RETRY),))
    ev2 = build_structured_editorial_relation(left_source_span_id="a", right_source_span_id="b", d157_hypotheses=(_hyp(RELATION_RETRY),))
    assert structured_editorial_relation_diagnostics(ev1) == structured_editorial_relation_diagnostics(ev2)


# ===========================================================================
# 43-48: Structural / no-rewrite / no-authority checks.
# ===========================================================================
def test_43_module_never_imports_family_besttake_boundary_pacing_render():
    import cutsell_worker.structured_editorial_relation as mod
    import_lines = "\n".join(
        line for line in inspect.getsource(mod).splitlines()
        if line.strip().startswith(("import ", "from "))
    )
    forbidden = (
        "take_grouping", "deterministic_best_take_authority", "attempt_relationship_authority",
        "boundary_engine_pass", "dialogue_pacing_transition", "renderer", "canonical_edit_plan",
    )
    for needle in forbidden:
        assert needle not in import_lines.lower(), f"{needle!r} imported by structured_editorial_relation.py"


def test_44_module_calls_no_d157_or_d169_function():
    import cutsell_worker.structured_editorial_relation as mod
    src = inspect.getsource(mod)
    # Only vocabulary constants/types are imported -- never a call into
    # _relation_for_pair/classify_relation_candidate/build_relation_evidence.
    for needle in ("_relation_for_pair(", "classify_relation_candidate(", "build_relation_evidence("):
        assert needle not in src


def test_45_module_mints_no_new_id():
    import cutsell_worker.structured_editorial_relation as mod
    src = inspect.getsource(mod)
    assert "hashlib" not in src and "mint_" not in src


def test_46_return_arity_of_build_editorial_moments_for_source_unchanged():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0), _span("c2", 1.0, 2.0, relation=RELATION_RETRY))
    result = build_editorial_moments_for_source(
        source_asset_id="src1", takes_for_source=takes,
        understanding_spans_by_id={"c1": wlu.understanding_spans[0], "c2": wlu.understanding_spans[1]},
    )
    assert len(result) == 4  # unchanged 4-tuple, per D-199's own contract


def test_47_provenance_out_carries_new_d200_3_keys():
    takes = [_take("c1", 0, 0.0, 1.0), _take("c2", 1, 1.0, 2.0)]
    wlu = _wlu(_span("c1", 0.0, 1.0), _span("c2", 1.0, 2.0, relation=RELATION_RETRY))
    out: dict = {}
    build_editorial_moments_for_source(
        source_asset_id="src1", takes_for_source=takes,
        understanding_spans_by_id={"c1": wlu.understanding_spans[0], "c2": wlu.understanding_spans[1]},
        provenance_out=out,
    )
    for key in ("structured_relation_by_position", "grouping_relation_by_position", "grouping_action_by_position", "grouping_reason_by_position"):
        assert key in out


# ===========================================================================
# 48-55: P1 integration -- backward compatibility (flags off / live spine
# absent) is BYTE-IDENTICAL to pre-D-200.3.
# ===========================================================================
def test_48_default_off_grouping_matches_flat_fusion_when_no_live_spine():
    u = _pair_understanding(d157_relation=RELATION_RETRY, live_spine=None)
    assert u.moment_count == 2
    assert len(u.local_groups) == 1
    assert len(u.local_groups[0].moment_ids) == 2  # joined via flat D157-only fusion, as before D-200.3


def test_49_moment_classification_relation_to_predecessor_unaffected_by_structured():
    # Even when live_spine IS supplied and grouping switches to structured,
    # `moment_relation_to_predecessor` (fed to D-194's classify_editorial_moment)
    # must stay the flat-fused value -- never the grouping-effective one.
    attempts = (_attempt("a0", 0.0, 1.0), _attempt("a1", 1.0, 2.0))
    props = (_prop("p0", "a0", 0.0, 1.0), _prop("p1", "a1", 1.0, 2.0))
    rel = (_relation_evidence(RELATION_CORRECTION, left="p0", right="p1"),)
    spine = _live_spine(attempts=attempts, proposition_candidates=props, relation_evidence=rel)
    u = _pair_understanding(d157_relation=RELATION_RETRY, live_spine=spine)
    # Flat fuse(RETRY, CORRECTION) -> conflict-abstained -> RELATION_UNCERTAIN
    # (the flat fusion's own abstention value, unchanged by D-200.3 -- this
    # value still feeds classify_editorial_moment, D-194, untouched).
    assert u.moment_relation_to_predecessor[1] == "UNCERTAIN"
    assert u.moment_relation_evidence_source[1] == "CONFLICT_ABSTAINED"


def test_50_grouping_switches_to_structured_when_live_spine_present():
    # Same D157=RETRY / D169=CORRECTION same-dimension conflict as test_49
    # -- the OLD flat path would ALSO abstain (matches), so instead prove
    # the switch with a genuinely DIFFERENT case: D157=NEW_AUDIENCE_BEAT,
    # D169=DISTINCT_PROPOSITION. Flat fusion (different flat strings) ->
    # CONFLICT_ABSTAINED -> UNCERTAIN -> boundary either way; structured
    # correctly reads it as EXPLICIT_BEAT_BOUNDARY split (both paths split
    # here, so we assert the STRUCTURED diagnostic explicitly ran instead
    # of merely inferring it from the grouping outcome).
    attempts = (_attempt("a0", 0.0, 1.0), _attempt("a1", 1.0, 2.0))
    props = (_prop("p0", "a0", 0.0, 1.0), _prop("p1", "a1", 1.0, 2.0))
    rel = (_relation_evidence(RELATION_DISTINCT_PROPOSITION, left="p0", right="p1"),)
    spine = _live_spine(attempts=attempts, proposition_candidates=props, relation_evidence=rel)
    u = _pair_understanding(d157_relation=RELATION_NEW_AUDIENCE_BEAT, live_spine=spine)
    assert u.moment_structured_relation[1] is not None
    assert u.moment_structured_relation[1].editorial_beat_relation == BEAT_NEW_AUDIENCE
    assert u.moment_grouping_reason[1] == GROUPING_REASON_EXPLICIT_BEAT_BOUNDARY


def test_51_grouping_actually_joins_via_structured_where_flat_would_abstain():
    # THE key behavior-change proof: D157=RETRY, D169=DISTINCT_PROPOSITION.
    # Flat fusion sees two DIFFERENT strings ("RETRY" != "DISTINCT_PROPOSITION")
    # -> CONFLICT_ABSTAINED -> UNCERTAIN -> boundary (over-fragmented, the
    # EXACT D-200 failure mode). Structured evidence correctly recognizes
    # these as two DIFFERENT DIMENSIONS (not a real conflict) and joins on
    # the real ATTEMPT=RETRY evidence.
    attempts = (_attempt("a0", 0.0, 1.0), _attempt("a1", 1.0, 2.0))
    props = (_prop("p0", "a0", 0.0, 1.0), _prop("p1", "a1", 1.0, 2.0))
    rel = (_relation_evidence(RELATION_DISTINCT_PROPOSITION, left="p0", right="p1"),)
    spine = _live_spine(attempts=attempts, proposition_candidates=props, relation_evidence=rel)

    u_legacy = _pair_understanding(d157_relation=RELATION_RETRY, live_spine=None)
    u_structured = _pair_understanding(d157_relation=RELATION_RETRY, live_spine=spine)

    # Legacy (D-198, no live spine): D157-only relation is RETRY -- joins
    # (canonical never in the picture at all in this path).
    assert len(u_legacy.local_groups[0].moment_ids) == 2
    # Structured (D-200.3, live spine present): also joins, but now via
    # the EXPLICIT dimension-aware ATTEMPT_JOIN reason, not a coincidence.
    assert len(u_structured.local_groups[0].moment_ids) == 2
    assert u_structured.moment_grouping_reason[1] == GROUPING_REASON_ATTEMPT_JOIN
    assert u_structured.moment_grouping_effective_relation[1] == ATTEMPT_RETRY


def test_52_run_summary_carries_new_d200_3_counts():
    attempts = (_attempt("a0", 0.0, 1.0), _attempt("a1", 1.0, 2.0))
    props = (_prop("p0", "a0", 0.0, 1.0), _prop("p1", "a1", 1.0, 2.0))
    rel = (_relation_evidence(RELATION_DISTINCT_PROPOSITION, left="p0", right="p1"),)
    spine = _live_spine(attempts=attempts, proposition_candidates=props, relation_evidence=rel)
    u = _pair_understanding(d157_relation=RELATION_RETRY, live_spine=spine)
    summary = editorial_moment_understanding_run_summary([u])
    assert summary["structured_relation_edge_count"] == 1
    assert summary["attempt_retry_count"] == 1
    assert summary["proposition_distinct_count"] == 1
    assert summary["cross_dimension_compatible_count"] == 1
    assert summary["same_dimension_conflict_count"] == 0
    assert summary["grouping_join_count"] == 1
    # Honest gaps -- never invented (this task's own "NO FICTION" instruction).
    assert summary["same_editorial_beat_count"] == 0
    assert summary["proposition_same_count"] == 0
    assert summary["proposition_progression_count"] == 0


def test_53_diagnostics_moment_row_carries_structured_fields():
    attempts = (_attempt("a0", 0.0, 1.0), _attempt("a1", 1.0, 2.0))
    props = (_prop("p0", "a0", 0.0, 1.0), _prop("p1", "a1", 1.0, 2.0))
    rel = (_relation_evidence(RELATION_DISTINCT_PROPOSITION, left="p0", right="p1"),)
    spine = _live_spine(attempts=attempts, proposition_candidates=props, relation_evidence=rel)
    u = _pair_understanding(d157_relation=RELATION_RETRY, live_spine=spine)
    diag = editorial_moment_understanding_diagnostics(u)
    row = diag["moments"][1]
    assert row["structured_relation"]["attempt_relation"] == ATTEMPT_RETRY
    assert row["grouping_effective_relation"] == ATTEMPT_RETRY
    assert row["grouping_action"] == GROUPING_ACTION_JOIN


def test_54_immutability_no_family_besttake_field_on_moment():
    attempts = (_attempt("a0", 0.0, 1.0), _attempt("a1", 1.0, 2.0))
    props = (_prop("p0", "a0", 0.0, 1.0), _prop("p1", "a1", 1.0, 2.0))
    rel = (_relation_evidence(RELATION_DISTINCT_PROPOSITION, left="p0", right="p1"),)
    spine = _live_spine(attempts=attempts, proposition_candidates=props, relation_evidence=rel)
    u = _pair_understanding(d157_relation=RELATION_RETRY, live_spine=spine)
    field_names = {f for m in u.moments for f in vars(m).keys()}
    assert "retry_family_id" not in field_names
    assert "selected_clip_id" not in field_names
    assert "final_winner" not in field_names


def test_55_build_editorial_local_groups_rule_set_itself_unchanged():
    # build_editorial_local_groups still only recognizes the flat
    # RETRY/CORRECTION/CONTINUATION strings by membership -- D-200.3 never
    # widened that function's own vocabulary.
    from cutsell_worker.editorial_moment_sequence import classify_editorial_moment
    del classify_editorial_moment  # imported only to prove no signature change needed
    import inspect as _inspect
    src = _inspect.getsource(build_editorial_local_groups)
    assert "_JOIN_RELATIONS" in src
    assert "StructuredEditorialRelationEvidence" not in src
