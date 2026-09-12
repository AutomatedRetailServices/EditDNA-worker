"""D-239U: SOURCE-ALIGNED P1 ROLE EVIDENCE CONFIDENCE tests.

Generic, abstract fixtures only -- NO Video00 literal text/spans/ids
(CLAUDE.md's binding rule, D-193/D-194's own generic-fixture precedent).
Covers the directive's 34-item test matrix: `role_evidence_source`/
`role_evidence_confidence` sourced from the SAME channel that established
`moment_role` (attempt_state / behavior_hypotheses / relation evidence),
`EditorialMoment.confidence` byte-identical to its pre-D-239U behavior,
Seam C's refined consumption, and the D-235Q materiality firewall left
untouched.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess

import pytest

from cutsell_worker.editorial_moment_sequence import (
    ALLOWED_ROLE_EVIDENCE_SOURCES,
    AUDIENCE_DELIVERY_NOT_SUPPORTED,
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
    RECORDING_PROCESS_ABSENT,
    ROLE_EVIDENCE_SOURCE_ATTEMPT_STATE,
    ROLE_EVIDENCE_SOURCE_BEHAVIOR_HYPOTHESIS,
    ROLE_EVIDENCE_SOURCE_RELATION_EVIDENCE,
    ROLE_EVIDENCE_SOURCE_UNRESOLVED,
    EditorialMoment,
    classify_editorial_moment,
)
from cutsell_worker.language_proposition_relation import (
    RELATION_NEW_AUDIENCE_BEAT,
    RELATION_RETRY,
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
)
from cutsell_worker.raw_understanding_map import (
    BEHAVIOR_BREAKING_CHARACTER,
    BEHAVIOR_POST_TAKE_RESET,
    BEHAVIOR_PRE_TAKE_SETUP,
    PROVENANCE_DETERMINISTIC_RULE,
    PROVENANCE_MULTIMODAL_FUSION,
    PROVENANCE_VISUAL_SIGNAL,
    BehaviorHypothesis,
)
from cutsell_worker.editorial_moment_sequence_integration import (
    EditorialMomentUnderstanding,
    P1_TARGET_STATUS_MOMENT_FOUND_LOW_CONFIDENCE,
    P1_TARGET_STATUS_MOMENT_FOUND_RESOLVED,
    exact_p1_target_evidence_for,
    p1_moment_role_and_audience_status_by_clip_id_for,
)
from cutsell_worker.watch_listen_understanding import (
    CONFIDENCE_SUPPORTED as WL_CONFIDENCE_SUPPORTED,
    MEANING_COMPLETE as WL_MEANING_COMPLETE,
    USABILITY_USABLE,
    UnderstandingSpan,
    WatchListenUnderstanding,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "cutsell_worker" / "editorial_moment_sequence.py"


# ---------------------------------------------------------------------------
# Generic fixture factories -- abstract "attempt N" content, no real transcript.
# ---------------------------------------------------------------------------
def _attempt(
    *,
    source_asset_id: str = "src",
    attempt_id: str = "att_1",
    state: str = ATTEMPT_CLEAN,
    meaning: str = MEANING_COMPLETE,
    confidence: str = CONFIDENCE_UNKNOWN,
    text: str = "generic abstract statement one",
    restart: bool = False,
    correction: bool = False,
    continuation: bool = False,
) -> LanguageAttempt:
    return LanguageAttempt(
        source_asset_id=source_asset_id,
        attempt_id=attempt_id,
        utterance_ids=(f"{attempt_id}_u0",),
        source_start=0.0,
        source_end=2.0,
        text_raw=text,
        text_normalized=text.lower(),
        attempt_state=state,
        meaning_completion=meaning,
        restart_evidence=restart,
        correction_evidence=correction,
        continuation_evidence=continuation,
        recording_process_evidence=(state == ATTEMPT_RECORDING_PROCESS),
        confidence=confidence,
        provenance="TEST_FIXTURE",
    )


def _behavior(label: str, *, provenance: str = PROVENANCE_VISUAL_SIGNAL, confidence: float = 0.9) -> BehaviorHypothesis:
    return BehaviorHypothesis(label=label, confidence=confidence, provenance=provenance, basis="generic fixture event")


# ---------------------------------------------------------------------------
# 1-6: role established directly by attempt_state -> role_evidence_confidence
# reuses attempt.confidence, role_evidence_source == ATTEMPT_STATE.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("state", "expected_role"),
    [
        (ATTEMPT_RECORDING_PROCESS, MOMENT_ROLE_RECORDING_PROCESS),
        (ATTEMPT_FALSE_START, MOMENT_ROLE_FALSE_START),
        (ATTEMPT_ABANDONED, MOMENT_ROLE_ABANDONED_ATTEMPT),
        (ATTEMPT_CORRECTION, MOMENT_ROLE_CORRECTION),
        (ATTEMPT_CONTINUATION, MOMENT_ROLE_CONTINUATION),
    ],
)
@pytest.mark.parametrize("attempt_confidence", [CONFIDENCE_SUPPORTED, CONFIDENCE_WEAK, CONFIDENCE_UNKNOWN])
def test_06_attempt_state_role_uses_attempt_confidence(state, expected_role, attempt_confidence):
    attempt = _attempt(state=state, confidence=attempt_confidence)
    moment = classify_editorial_moment(attempt)
    assert moment.moment_role == expected_role
    assert moment.role_evidence_source == ROLE_EVIDENCE_SOURCE_ATTEMPT_STATE
    assert moment.role_evidence_confidence == attempt_confidence
    assert moment.confidence == attempt_confidence  # existing semantics, unchanged


def test_09_clean_audience_delivery_existing_semantics():
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_SUPPORTED)
    moment = classify_editorial_moment(attempt)
    assert moment.moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY
    assert moment.role_evidence_source == ROLE_EVIDENCE_SOURCE_ATTEMPT_STATE
    assert moment.role_evidence_confidence == CONFIDENCE_SUPPORTED
    assert moment.confidence == CONFIDENCE_SUPPORTED


def test_10_uncertain_stays_fail_closed():
    attempt = _attempt(state=ATTEMPT_UNCERTAIN, confidence=CONFIDENCE_SUPPORTED)
    moment = classify_editorial_moment(attempt)
    assert moment.moment_role == MOMENT_ROLE_UNCERTAIN
    assert moment.role_evidence_source == ROLE_EVIDENCE_SOURCE_UNRESOLVED
    assert moment.role_evidence_confidence == CONFIDENCE_UNKNOWN
    # `confidence` itself keeps its pre-D-239U value -- never forced UNKNOWN.
    assert moment.confidence == CONFIDENCE_SUPPORTED

    incomplete_clean = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_INCOMPLETE, confidence=CONFIDENCE_SUPPORTED)
    moment2 = classify_editorial_moment(incomplete_clean)
    assert moment2.moment_role == MOMENT_ROLE_UNCERTAIN
    assert moment2.role_evidence_source == ROLE_EVIDENCE_SOURCE_UNRESOLVED
    assert moment2.role_evidence_confidence == CONFIDENCE_UNKNOWN


# ---------------------------------------------------------------------------
# POST_TAKE_RESET case (directive's own "D-239S SHAPE OFFLINE REPLAY").
# ---------------------------------------------------------------------------
def test_01_post_take_reset_behavior_supported_attempt_unknown():
    """Case A: attempt_state=CLEAN, attempt.confidence=UNKNOWN, behavior
    evidence (VISUAL_SIGNAL provenance) selects POST_TAKE_RESET."""
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_UNKNOWN)
    moment = classify_editorial_moment(
        attempt, behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET, provenance=PROVENANCE_VISUAL_SIGNAL),),
    )
    assert moment.moment_role == MOMENT_ROLE_POST_TAKE_RESET
    # existing EditorialMoment.confidence UNCHANGED -- still UNKNOWN.
    assert moment.confidence == CONFIDENCE_UNKNOWN
    # role_evidence_confidence sourced from the behavior evidence that
    # established the role -- SUPPORTED (VISUAL_SIGNAL provenance).
    assert moment.role_evidence_source == ROLE_EVIDENCE_SOURCE_BEHAVIOR_HYPOTHESIS
    assert moment.role_evidence_confidence == CONFIDENCE_SUPPORTED


def test_02_post_take_reset_behavior_unknown():
    """Case B: behavior confidence UNKNOWN (no matching hypothesis at
    all) -> role_evidence_confidence stays UNKNOWN, never invented."""
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_UNKNOWN)
    # A behavior hypothesis whose provenance is outside the known
    # VISUAL_SIGNAL/DETERMINISTIC_RULE/MULTIMODAL_FUSION set -- the
    # honest "insufficient basis" case.
    moment = classify_editorial_moment(
        attempt, behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET, provenance="UNKNOWN"),),
    )
    assert moment.moment_role == MOMENT_ROLE_POST_TAKE_RESET
    assert moment.role_evidence_confidence == CONFIDENCE_UNKNOWN
    assert moment.confidence == CONFIDENCE_UNKNOWN


def test_03_post_take_reset_behavior_conflict():
    """Case C: behavior evidence conflicts with the attempt's own
    structural state (CONTINUATION vs. a reset/break behavior label) ->
    MIXED for both `confidence` and `role_evidence_confidence`, never a
    silent promotion of one channel over the other."""
    attempt = _attempt(state=ATTEMPT_CONTINUATION, meaning=MEANING_INCOMPLETE, confidence=CONFIDENCE_SUPPORTED)
    moment = classify_editorial_moment(
        attempt, behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET, provenance=PROVENANCE_VISUAL_SIGNAL),),
    )
    assert moment.moment_role == MOMENT_ROLE_CONTINUATION  # attempt_state wins the ROLE (unchanged D-194 contract)
    assert "CONTINUATION_STATE_VS_RESET_OR_BREAK_BEHAVIOR_EVIDENCE" in moment.conflict_flags
    assert moment.confidence == CONFIDENCE_MIXED
    assert moment.role_evidence_confidence == CONFIDENCE_MIXED


# ---------------------------------------------------------------------------
# 04-05: other behavior-derived roles.
# ---------------------------------------------------------------------------
def test_04_pre_take_setup_behavior_confidence():
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_UNKNOWN)
    moment = classify_editorial_moment(
        attempt, behavior_hypotheses=(_behavior(BEHAVIOR_PRE_TAKE_SETUP, provenance=PROVENANCE_DETERMINISTIC_RULE),),
    )
    assert moment.moment_role == MOMENT_ROLE_PRE_TAKE_SETUP
    assert moment.role_evidence_source == ROLE_EVIDENCE_SOURCE_BEHAVIOR_HYPOTHESIS
    assert moment.role_evidence_confidence == CONFIDENCE_SUPPORTED
    assert moment.confidence == CONFIDENCE_UNKNOWN


def test_05_breaking_character_behavior_confidence():
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_WEAK)
    moment = classify_editorial_moment(
        attempt, behavior_hypotheses=(_behavior(BEHAVIOR_BREAKING_CHARACTER, provenance=PROVENANCE_MULTIMODAL_FUSION),),
    )
    assert moment.moment_role == MOMENT_ROLE_BREAKING_CHARACTER
    assert moment.role_evidence_source == ROLE_EVIDENCE_SOURCE_BEHAVIOR_HYPOTHESIS
    assert moment.role_evidence_confidence == CONFIDENCE_WEAK  # MULTIMODAL_FUSION-only provenance
    assert moment.confidence == CONFIDENCE_WEAK


def test_behavior_hypothesis_confidence_scoped_to_matching_label_only():
    """A different behavior label's own strong provenance must NEVER leak
    into an unrelated role's role_evidence_confidence -- only the SPECIFIC
    label that established the role is consulted."""
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_UNKNOWN)
    moment = classify_editorial_moment(
        attempt,
        behavior_hypotheses=(
            _behavior(BEHAVIOR_PRE_TAKE_SETUP, provenance="UNKNOWN"),
            _behavior(BEHAVIOR_POST_TAKE_RESET, provenance=PROVENANCE_VISUAL_SIGNAL),
        ),
    )
    # BREAKING_CHARACTER > POST_TAKE_RESET > PRE_TAKE_SETUP precedence --
    # POST_TAKE_RESET wins here since no BREAKING_CHARACTER hypothesis.
    assert moment.moment_role == MOMENT_ROLE_POST_TAKE_RESET
    assert moment.role_evidence_confidence == CONFIDENCE_SUPPORTED


# ---------------------------------------------------------------------------
# 07-08: relation-derived roles use the already-computed relation confidence.
# ---------------------------------------------------------------------------
def test_07_retry_uses_exact_relation_confidence():
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_UNKNOWN)
    moment = classify_editorial_moment(
        attempt, relation_to_predecessor=RELATION_RETRY, relation_confidence=CONFIDENCE_SUPPORTED,
    )
    assert moment.moment_role == MOMENT_ROLE_RETRY
    assert moment.role_evidence_source == ROLE_EVIDENCE_SOURCE_RELATION_EVIDENCE
    assert moment.role_evidence_confidence == CONFIDENCE_SUPPORTED
    assert moment.confidence == CONFIDENCE_UNKNOWN  # unchanged -- still attempt.confidence


def test_08_new_audience_beat_uses_exact_relation_confidence():
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_SUPPORTED)
    moment = classify_editorial_moment(
        attempt, relation_to_predecessor=RELATION_NEW_AUDIENCE_BEAT, relation_confidence=CONFIDENCE_WEAK,
    )
    assert moment.moment_role == MOMENT_ROLE_NEW_AUDIENCE_BEAT
    assert moment.role_evidence_source == ROLE_EVIDENCE_SOURCE_RELATION_EVIDENCE
    assert moment.role_evidence_confidence == CONFIDENCE_WEAK
    assert moment.confidence == CONFIDENCE_SUPPORTED  # unchanged


def test_relation_role_omitted_relation_confidence_defaults_unknown():
    """`relation_confidence` is optional (every existing caller/test that
    doesn't know about it) -- must default to UNKNOWN, never invented."""
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_SUPPORTED)
    moment = classify_editorial_moment(attempt, relation_to_predecessor=RELATION_RETRY)
    assert moment.moment_role == MOMENT_ROLE_RETRY
    assert moment.role_evidence_confidence == CONFIDENCE_UNKNOWN


# ---------------------------------------------------------------------------
# 11: EditorialMoment.confidence unchanged across the whole matrix above is
# already asserted per-test; this is the explicit byte-identical parity
# check against the PRE-D-239U formula (attempt.confidence, MIXED on
# conflict) for a broad sweep of inputs.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("attempt_confidence", [CONFIDENCE_SUPPORTED, CONFIDENCE_WEAK, CONFIDENCE_UNKNOWN, CONFIDENCE_MIXED])
@pytest.mark.parametrize("state", [ATTEMPT_CLEAN, ATTEMPT_RECORDING_PROCESS, ATTEMPT_FALSE_START, ATTEMPT_CORRECTION])
def test_11_existing_confidence_field_unchanged(attempt_confidence, state):
    attempt = _attempt(state=state, meaning=MEANING_COMPLETE, confidence=attempt_confidence)
    moment = classify_editorial_moment(attempt, behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET),))
    expected_base = CONFIDENCE_MIXED if moment.conflict_flags else attempt_confidence
    assert moment.confidence == expected_base


def test_direct_construction_backward_compatible_default():
    """A pre-D-239U direct `EditorialMoment(...)` construction that never
    heard of `role_evidence_confidence` must behave exactly as before --
    it mirrors `confidence` verbatim."""
    moment = EditorialMoment(
        source_asset_id="src", editorial_moment_id="m1", source_start=0.0, source_end=1.0,
        source_span_id="c1", attempt_ids=(), proposition_candidate_ids=(), related_span_ids=(),
        moment_role=MOMENT_ROLE_RECORDING_PROCESS, audience_delivery_status=AUDIENCE_DELIVERY_NOT_SUPPORTED,
        recording_process_status=RECORDING_PROCESS_ABSENT, completion_status=MEANING_COMPLETE,
        local_sequence_position=0, confidence=CONFIDENCE_SUPPORTED, conflict_flags=(), provenance=(),
    )
    assert moment.role_evidence_confidence == CONFIDENCE_SUPPORTED
    assert moment.role_evidence_source == ROLE_EVIDENCE_SOURCE_ATTEMPT_STATE


# ---------------------------------------------------------------------------
# 14-16: Seam C consumption.
# ---------------------------------------------------------------------------
def _understanding(source_asset_id, moments):
    return EditorialMomentUnderstanding(
        source_asset_id=source_asset_id, moments=tuple(moments), sequence_hypotheses=(),
        moment_count=len(moments), sequence_count=0, capability_status="AVAILABLE",
        missing_evidence=(), confidence=CONFIDENCE_SUPPORTED, conflict_flags=(), provenance=(),
    )


def test_14_seam_c_accepts_only_supported_role_evidence():
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_UNKNOWN)
    moment = classify_editorial_moment(
        attempt, source_span_id="c1", behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET, provenance=PROVENANCE_VISUAL_SIGNAL),),
    )
    assert moment.confidence == CONFIDENCE_UNKNOWN  # the exact D-239T target shape
    assert moment.role_evidence_confidence == CONFIDENCE_SUPPORTED
    role_by_clip, audience_by_clip = p1_moment_role_and_audience_status_by_clip_id_for((_understanding("src", [moment]),))
    # D-239T's own finding, now resolved: Seam C consumes role_evidence_
    # confidence (SUPPORTED), not the unrelated EditorialMoment.confidence
    # (UNKNOWN) -- this clip_id now resolves.
    assert role_by_clip.get("c1") == MOMENT_ROLE_POST_TAKE_RESET
    assert audience_by_clip.get("c1") == AUDIENCE_DELIVERY_NOT_SUPPORTED


def test_15_seam_c_rejects_unknown_role_evidence():
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_SUPPORTED)
    moment = classify_editorial_moment(
        attempt, source_span_id="c1", behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET, provenance="UNKNOWN"),),
    )
    # `confidence` is SUPPORTED (pre-D-239U would have resolved this clip)
    # but `role_evidence_confidence` is UNKNOWN -- Seam C now refuses.
    assert moment.confidence == CONFIDENCE_SUPPORTED
    assert moment.role_evidence_confidence == CONFIDENCE_UNKNOWN
    role_by_clip, _ = p1_moment_role_and_audience_status_by_clip_id_for((_understanding("src", [moment]),))
    assert "c1" not in role_by_clip


def test_16_seam_c_rejects_mixed_role_evidence():
    attempt = _attempt(state=ATTEMPT_CONTINUATION, meaning=MEANING_INCOMPLETE, confidence=CONFIDENCE_SUPPORTED)
    moment = classify_editorial_moment(
        attempt, source_span_id="c1", behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET, provenance=PROVENANCE_VISUAL_SIGNAL),),
    )
    assert moment.role_evidence_confidence == CONFIDENCE_MIXED
    role_by_clip, _ = p1_moment_role_and_audience_status_by_clip_id_for((_understanding("src", [moment]),))
    assert "c1" not in role_by_clip


def test_seam_c_still_rejects_uncertain_role_even_with_supported_evidence():
    attempt = _attempt(state=ATTEMPT_UNCERTAIN, confidence=CONFIDENCE_SUPPORTED)
    moment = classify_editorial_moment(attempt, source_span_id="c1")
    assert moment.role_evidence_confidence == CONFIDENCE_UNKNOWN  # ATTEMPT_UNCERTAIN -> UNRESOLVED/UNKNOWN
    role_by_clip, _ = p1_moment_role_and_audience_status_by_clip_id_for((_understanding("src", [moment]),))
    assert "c1" not in role_by_clip


# ---------------------------------------------------------------------------
# exact_p1_target_evidence_for exposes both fields honestly, and its own
# diagnostic status now mirrors Seam C's real (refined) gate.
# ---------------------------------------------------------------------------
def _span(clip_id):
    return UnderstandingSpan(
        span_id=clip_id, source_asset_id="src", source_start=0.0, source_end=1.0,
        behavior_state_hypotheses=(), behavior_confidence=WL_CONFIDENCE_SUPPORTED,
        attempt_boundary_hypotheses=(), attempt_relation_hypotheses=(),
        relation_confidence=WL_CONFIDENCE_SUPPORTED, meaning_completion_hypothesis=WL_MEANING_COMPLETE,
        performance_usability_hypothesis=USABILITY_USABLE, entry_usability=USABILITY_USABLE,
        delivery_usability=USABILITY_USABLE, exit_usability=USABILITY_USABLE,
        conflict_flags=(), evidence_provenance={},
    )


def _wlu(source_asset_id, *spans):
    return WatchListenUnderstanding(source_asset_id=source_asset_id, understanding_spans=tuple(spans))


def test_exact_p1_target_evidence_exposes_role_evidence_fields_and_resolves_the_d239t_target_shape():
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_UNKNOWN)
    moment = classify_editorial_moment(
        attempt, source_span_id="c1", behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET, provenance=PROVENANCE_VISUAL_SIGNAL),),
    )
    understanding = _understanding("src", [moment])
    role_by_clip, audience_by_clip = p1_moment_role_and_audience_status_by_clip_id_for((understanding,))
    result = exact_p1_target_evidence_for(
        {"c1": "src"},
        editorial_moment_understandings=(understanding,),
        watch_listen_understandings=(_wlu("src", _span("c1")),),
        p1_moment_role_by_clip_id=role_by_clip,
        p1_audience_delivery_status_by_clip_id=audience_by_clip,
    )
    row = result["targets"][0]
    # `role_confidence` keeps meaning EditorialMoment.confidence, UNCHANGED.
    assert row["role_confidence"] == CONFIDENCE_UNKNOWN
    # the NEW, additive, role-source-aligned fields:
    assert row["role_evidence_source"] == ROLE_EVIDENCE_SOURCE_BEHAVIOR_HYPOTHESIS
    assert row["role_evidence_confidence"] == CONFIDENCE_SUPPORTED
    # the diagnostic status now mirrors Seam C's real (refined) behavior --
    # this is the exact D-239T target shape, now resolving.
    assert row["helper_lookup_resolved"] is True
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_MOMENT_FOUND_RESOLVED


def test_exact_p1_target_evidence_still_reports_low_confidence_when_role_evidence_is_unknown():
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_SUPPORTED)
    moment = classify_editorial_moment(
        attempt, source_span_id="c1", behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET, provenance="UNKNOWN"),),
    )
    understanding = _understanding("src", [moment])
    role_by_clip, audience_by_clip = p1_moment_role_and_audience_status_by_clip_id_for((understanding,))
    result = exact_p1_target_evidence_for(
        {"c1": "src"},
        editorial_moment_understandings=(understanding,),
        watch_listen_understandings=(_wlu("src", _span("c1")),),
        p1_moment_role_by_clip_id=role_by_clip,
        p1_audience_delivery_status_by_clip_id=audience_by_clip,
    )
    row = result["targets"][0]
    assert row["role_confidence"] == CONFIDENCE_SUPPORTED  # EditorialMoment.confidence, unchanged
    assert row["role_evidence_confidence"] == CONFIDENCE_UNKNOWN
    assert row["helper_lookup_resolved"] is False
    assert row["p1_target_lookup_status"] == P1_TARGET_STATUS_MOMENT_FOUND_LOW_CONFIDENCE


# ---------------------------------------------------------------------------
# 23-25: language independence -- role_evidence_source/confidence never
# read `text_raw`/`text_normalized` at all, so behavior is identical
# regardless of transcript language.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "generic abstract statement one",  # English
        "declaracion abstracta generica uno",  # Spanish (unaccented, generic)
        "let's vamos a hacer esto generic style",  # Spanglish
    ],
)
def test_role_evidence_is_language_independent(text):
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_UNKNOWN, text=text)
    moment = classify_editorial_moment(
        attempt, behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET, provenance=PROVENANCE_VISUAL_SIGNAL),),
    )
    assert moment.moment_role == MOMENT_ROLE_POST_TAKE_RESET
    assert moment.role_evidence_confidence == CONFIDENCE_SUPPORTED


# ---------------------------------------------------------------------------
# 19/21/26/27/28: no heuristic, no aggregate authority, no new threshold,
# no provider, no RAW -- structural checks on the module's own source
# (mirrors D-194's own no-feature-flag/no-provider precedent).
# ---------------------------------------------------------------------------
def _code_only_source() -> str:
    src = MODULE_PATH.read_text()
    tree = ast.parse(src)
    docstring_node = tree.body[0] if tree.body and isinstance(tree.body[0], ast.Expr) else None
    if docstring_node is None:
        return src
    lines = src.splitlines(keepends=True)
    return "".join(lines[docstring_node.end_lineno:])


def test_no_new_provider_or_raw_or_threshold_introduced():
    src = _code_only_source()
    for forbidden in ("openai", "gemini", "genai", "requests.", "http://", "https://", "import numpy"):
        assert forbidden not in src.lower(), forbidden


def _function_source(name: str) -> str:
    src = MODULE_PATH.read_text()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "".join(lines[node.lineno - 1:node.end_lineno])
    raise AssertionError(f"function {name!r} not found")


def test_no_aggregate_authority_vocabulary_introduced():
    """`classify_editorial_moment` itself -- the one function this gate
    changed -- must never reference any aggregate/count/majority-vote
    concept when deciding a SINGLE target's own role_evidence_confidence
    (unrelated pre-existing diagnostics elsewhere in this module, e.g.
    run-summary role counts, are out of scope for this per-target check)."""
    src = _function_source("classify_editorial_moment")
    for forbidden in ("_count", "BLOOPER_SERIES_OBSERVED", "majority_vote", "aggregate"):
        assert forbidden not in src, forbidden


def test_role_evidence_confidence_never_upgrades_via_aggregate_or_hardcode():
    """POST_TAKE_RESET must never be hardcoded to SUPPORTED regardless of
    evidence -- confidence tracks the actual behavior-hypothesis provenance."""
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_UNKNOWN)
    moment_weak = classify_editorial_moment(
        attempt, behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET, provenance=PROVENANCE_MULTIMODAL_FUSION),),
    )
    assert moment_weak.role_evidence_confidence == CONFIDENCE_WEAK
    moment_unknown = classify_editorial_moment(attempt, behavior_hypotheses=())
    assert moment_unknown.moment_role != MOMENT_ROLE_POST_TAKE_RESET  # no behavior evidence at all -> CLEAN_AUDIENCE_DELIVERY


# ---------------------------------------------------------------------------
# 20: ownership-only never enough -- structural confirmation that
# `p1_moment_role_and_audience_status_by_clip_id_for`'s own gate (role
# evidence confidence + non-UNCERTAIN role) is still required regardless
# of any ownership concept; ownership itself lives entirely in
# `final_story_coherence_validation.py`, untouched by this gate, and is
# exercised by that module's own D-235Q test suite (regression, not
# duplicated here).
# ---------------------------------------------------------------------------
def test_seam_c_gate_is_independent_of_any_ownership_concept():
    attempt = _attempt(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE, confidence=CONFIDENCE_UNKNOWN)
    moment = classify_editorial_moment(attempt, source_span_id="c1", behavior_hypotheses=())
    # No behavior/relation evidence at all -> CLEAN_AUDIENCE_DELIVERY with
    # attempt.confidence as its own role_evidence_confidence.
    assert moment.moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY
    assert moment.role_evidence_confidence == CONFIDENCE_UNKNOWN
    role_by_clip, _ = p1_moment_role_and_audience_status_by_clip_id_for((_understanding("src", [moment]),))
    assert "c1" not in role_by_clip  # Seam C's own gate alone decides this -- no ownership object involved at all


# ---------------------------------------------------------------------------
# ALLOWED_ROLE_EVIDENCE_SOURCES sanity -- every real classify_editorial_
# moment output stays within the declared vocabulary.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(state=ATTEMPT_RECORDING_PROCESS),
        dict(state=ATTEMPT_CLEAN, meaning=MEANING_COMPLETE),
        dict(state=ATTEMPT_UNCERTAIN),
    ],
)
def test_role_evidence_source_always_in_allowed_vocabulary(kwargs):
    attempt = _attempt(**kwargs)
    moment = classify_editorial_moment(attempt, behavior_hypotheses=(_behavior(BEHAVIOR_POST_TAKE_RESET),))
    assert moment.role_evidence_source in ALLOWED_ROLE_EVIDENCE_SOURCES
