"""D-158 Phase C: Attempt Relationship structured authority -- the FIRST
editorial consumer of Watch+Listen Understanding V1 (D-157, unchanged).

Per docs/CUTSELL_DECISIONS.md D-148/D-154/D-155/D-156/D-157/D-158 and
`attempt_relationship_authority.py`'s own module docstring: this module
NEVER creates a merge from Watch+Listen evidence alone (False -> True is
never possible); it can only WITHHOLD a merge the existing Proposition
Identity evidence (arbiter `same_idea` / deterministic restart evidence)
already decided on, when real SUPPORTED-confidence Watch+Listen evidence
for that exact pair materially disagrees.

Generic fixtures throughout -- no Video00 wording, no real transcript
text beyond short, obviously-synthetic sentences.
"""
from __future__ import annotations

import os

import pytest

from cutsell_worker.attempt_relationship_authority import (
    FAMILY_ACTION_ABSTAIN_UNCERTAIN,
    FAMILY_ACTION_ELIGIBLE_RETRY_FAMILY,
    FAMILY_ACTION_SEPARATE_BEAT,
    FAMILY_ACTION_SEPARATE_CORRECTION,
    FAMILY_ACTION_SEPARATE_DISTINCT,
    FAMILY_ACTION_SEPARATE_NOT_COMPETING,
    RELATION_SOURCE_CONFLICT_RESOLUTION,
    RELATION_SOURCE_DETERMINISTIC_RESTART,
    RELATION_SOURCE_SEMANTIC_PROVIDER,
    RELATION_SOURCE_UNCERTAIN,
    RELATION_SOURCE_WATCH_LISTEN,
    attempt_relation_hypotheses_for_pair,
    attempt_relationship_diagnostics,
    build_understanding_span_index,
    resolve_final_attempt_relation,
    watch_listen_family_evidence_enabled,
)
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_grouping_provider import reconcile_semantic_idea_equivalence
from cutsell_worker.watch_listen_understanding import (
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
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

_ENV_FLAG = "CUTSELL_WATCH_LISTEN_FAMILY_EVIDENCE_ENABLED"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _relation(relation, *, confidence=CONFIDENCE_SUPPORTED, left_span_id="left", basis="fixture"):
    return AttemptRelationHypothesis(
        relation=relation, confidence=confidence, basis=basis,
        left_span_id=left_span_id, provenance=("MULTIMODAL_FUSION",),
    )


def _span(span_id, relations=()):
    """Minimal, structurally-valid `UnderstandingSpan` -- only the fields
    this authority actually reads (`span_id`, `attempt_relation_
    hypotheses`) carry meaningful fixture data; every other field is a
    harmless placeholder, exactly matching what `attempt_relation_
    hypotheses_for_pair`'s own docstring says this module reads."""
    return UnderstandingSpan(
        span_id=span_id, source_asset_id="src", source_start=0.0, source_end=1.0,
        behavior_state_hypotheses=(), behavior_confidence=CONFIDENCE_UNKNOWN,
        attempt_boundary_hypotheses=(), attempt_relation_hypotheses=relations,
        relation_confidence=CONFIDENCE_UNKNOWN,
        meaning_completion_hypothesis="UNCERTAIN",
        performance_usability_hypothesis="UNKNOWN",
        entry_usability="UNKNOWN", delivery_usability="UNKNOWN", exit_usability="UNKNOWN",
        conflict_flags=(), evidence_provenance={},
    )


def _take(clip_id, start, end, text, *, complete=True, source="src"):
    return CandidateTake(clip_id, source, 0, start, end, text, complete_idea=complete)


def _clear_flag(monkeypatch):
    monkeypatch.delenv(_ENV_FLAG, raising=False)


def _set_flag_on(monkeypatch):
    monkeypatch.setenv(_ENV_FLAG, "1")


# ===========================================================================
# 1-9. `resolve_final_attempt_relation` truth table -- the core structured-
# conflict-resolution function, covering every cell required by the
# directive's fixture-category list.
# ===========================================================================

def test_01_no_watch_listen_evidence_missing_evidence_fail_open_merge_true():
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source=RELATION_SOURCE_DETERMINISTIC_RESTART,
        watch_listen_relations=(),
    )
    assert final.would_merge is True
    assert final.conflict is False
    assert final.family_membership_action == FAMILY_ACTION_ELIGIBLE_RETRY_FAMILY
    assert final.watch_listen_relation_evaluated is False


def test_02_no_watch_listen_evidence_missing_evidence_fail_open_merge_false():
    final = resolve_final_attempt_relation(would_merge=False, would_merge_source="none")
    assert final.would_merge is False
    assert final.conflict is False
    assert final.family_membership_action == FAMILY_ACTION_ABSTAIN_UNCERTAIN


def test_03_same_proposition_plus_retry_agreement_merge_proceeds():
    """same proposition + retry: pre-D-158 wants a merge, Watch+Listen
    agrees RETRY -- merge proceeds, no conflict."""
    relations = (_relation(RELATION_RETRY),)
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source=RELATION_SOURCE_DETERMINISTIC_RESTART,
        watch_listen_relations=relations,
    )
    assert final.would_merge is True
    assert final.conflict is False
    assert final.relation == RELATION_RETRY
    assert final.family_membership_action == FAMILY_ACTION_ELIGIBLE_RETRY_FAMILY
    assert final.watch_listen_supported is True


def test_04_false_start_plus_retry_agreement_merge_proceeds():
    """false start + retry: identical shape to test_03 via a different
    semantic source (SEMANTIC_PROVIDER instead of DETERMINISTIC_RESTART)."""
    relations = (_relation(RELATION_RETRY),)
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source=RELATION_SOURCE_SEMANTIC_PROVIDER,
        watch_listen_relations=relations,
    )
    assert final.would_merge is True
    assert final.source == RELATION_SOURCE_SEMANTIC_PROVIDER


def test_05_abandoned_plus_retry_agreement_merge_proceeds():
    """abandoned + retry: same agreement shape, restated for the
    'abandoned attempt followed by a retry' fixture category."""
    relations = (_relation(RELATION_RETRY, basis="abandoned_then_retry"),)
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source=RELATION_SOURCE_DETERMINISTIC_RESTART,
        watch_listen_relations=relations,
    )
    assert final.would_merge is True and final.conflict is False


def test_06_incomplete_plus_continuation_conflict_withholds_merge():
    """incomplete + continuation: pre-D-158 proposed a merge (same
    lexical family), but Watch+Listen SUPPORTED evidence says the second
    span CONTINUES the first (not a competing retry) -- merge withheld,
    conflict recorded, never forced to CONTINUATION's own non-merge
    outcome silently."""
    relations = (_relation(RELATION_CONTINUATION),)
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source=RELATION_SOURCE_DETERMINISTIC_RESTART,
        watch_listen_relations=relations,
    )
    assert final.would_merge is False
    assert final.conflict is True
    assert final.relation == RELATION_CONTINUATION
    assert final.family_membership_action == FAMILY_ACTION_SEPARATE_NOT_COMPETING
    assert final.source == RELATION_SOURCE_CONFLICT_RESOLUTION


def test_07_correction_conflict_withholds_merge_separate_correction():
    relations = (_relation(RELATION_CORRECTION),)
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source=RELATION_SOURCE_DETERMINISTIC_RESTART,
        watch_listen_relations=relations,
    )
    assert final.would_merge is False
    assert final.conflict is True
    assert final.relation == RELATION_CORRECTION
    assert final.family_membership_action == FAMILY_ACTION_SEPARATE_CORRECTION


def test_08_complementary_conflict_withholds_merge_separate_not_competing():
    relations = (_relation(RELATION_COMPLEMENTARY),)
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source=RELATION_SOURCE_SEMANTIC_PROVIDER,
        watch_listen_relations=relations,
    )
    assert final.would_merge is False
    assert final.family_membership_action == FAMILY_ACTION_SEPARATE_NOT_COMPETING


def test_09_new_audience_beat_conflict_withholds_merge_separate_beat():
    relations = (_relation(RELATION_NEW_AUDIENCE_BEAT),)
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source=RELATION_SOURCE_DETERMINISTIC_RESTART,
        watch_listen_relations=relations,
    )
    assert final.would_merge is False
    assert final.family_membership_action == FAMILY_ACTION_SEPARATE_BEAT


# ===========================================================================
# 10-13. DISTINCT_PROPOSITION / same-topic / same-opener variants
# ===========================================================================

def test_10_distinct_proposition_conflict_withholds_merge_separate_distinct():
    """distinct proposition: even though D-157 never actually asserts
    DISTINCT_PROPOSITION above UNCERTAIN today, this module's own mapping
    table handles it defensively (forward-compatible, never crashes)."""
    relations = (_relation(RELATION_DISTINCT_PROPOSITION),)
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source=RELATION_SOURCE_DETERMINISTIC_RESTART,
        watch_listen_relations=relations,
    )
    assert final.would_merge is False
    assert final.family_membership_action == FAMILY_ACTION_SEPARATE_DISTINCT


def test_11_same_topic_distinct_proposition_never_merged_from_watch_listen_alone():
    """same-topic-distinct-proposition: would_merge starts False (the
    semantic layer never proposed a merge for these two), and Watch+Listen
    alone -- even a strong DISTINCT_PROPOSITION-shaped signal -- can never
    flip it to True."""
    relations = (_relation(RELATION_DISTINCT_PROPOSITION),)
    final = resolve_final_attempt_relation(
        would_merge=False, would_merge_source="none", watch_listen_relations=relations,
    )
    assert final.would_merge is False


def test_12_same_opener_new_beat_conflict_the_other_direction_never_forces_merge():
    """same-opener-new-beat: pre-D-158 said 'no shared idea' (would_merge
    False), but Watch+Listen sees NEW_AUDIENCE_BEAT-adjacent evidence
    shaped like a retry -- reported as a conflict, but NEVER forces a
    merge; family membership abstains (UNCERTAIN), staying separate."""
    relations = (_relation(RELATION_RETRY),)
    final = resolve_final_attempt_relation(would_merge=False, would_merge_source="none", watch_listen_relations=relations)
    assert final.would_merge is False
    assert final.conflict is True
    assert final.relation == RELATION_UNCERTAIN
    assert final.family_membership_action == FAMILY_ACTION_ABSTAIN_UNCERTAIN
    assert final.source == RELATION_SOURCE_CONFLICT_RESOLUTION


def test_13_would_merge_false_watch_listen_correction_also_never_forces_merge():
    relations = (_relation(RELATION_CORRECTION),)
    final = resolve_final_attempt_relation(would_merge=False, would_merge_source="none", watch_listen_relations=relations)
    assert final.would_merge is False
    assert final.conflict is True
    assert final.family_membership_action == FAMILY_ACTION_ABSTAIN_UNCERTAIN


# ===========================================================================
# 14-19. Agreement scenarios (would_merge False + non-competing relations)
# ===========================================================================

@pytest.mark.parametrize("relation", [RELATION_CONTINUATION, RELATION_COMPLEMENTARY, RELATION_NEW_AUDIENCE_BEAT])
def test_14_agreement_would_merge_false_and_watch_listen_agrees_not_competing(relation):
    relations = (_relation(relation),)
    final = resolve_final_attempt_relation(would_merge=False, would_merge_source="none", watch_listen_relations=relations)
    assert final.would_merge is False
    assert final.conflict is False, "agreement must never be reported as a conflict"
    assert final.source == RELATION_SOURCE_WATCH_LISTEN
    assert final.watch_listen_supported is True


def test_15_agreement_would_merge_true_retry_and_watch_listen_agrees_retry():
    relations = (_relation(RELATION_RETRY),)
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source=RELATION_SOURCE_SEMANTIC_PROVIDER,
        watch_listen_relations=relations,
    )
    assert final.would_merge is True
    assert final.conflict is False


# ===========================================================================
# 16-19. Missing/weak evidence fail-open, provider-unavailable fail-open
# ===========================================================================

def test_16_weak_confidence_evidence_treated_as_no_material_evidence():
    relations = (_relation(RELATION_CONTINUATION, confidence=CONFIDENCE_WEAK),)
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source=RELATION_SOURCE_DETERMINISTIC_RESTART,
        watch_listen_relations=relations,
    )
    assert final.would_merge is True
    assert final.conflict is False
    assert final.watch_listen_relation_evaluated is True  # relations were present...
    assert final.watch_listen_supported is False  # ...but none were material


def test_17_mixed_confidence_evidence_also_treated_as_no_material_evidence():
    relations = (_relation(RELATION_RETRY, confidence=CONFIDENCE_MIXED),)
    final = resolve_final_attempt_relation(would_merge=False, would_merge_source="none", watch_listen_relations=relations)
    assert final.would_merge is False
    assert final.conflict is False


def test_18_uncertain_relation_never_escalated_even_at_supported_confidence():
    relations = (_relation(RELATION_UNCERTAIN, confidence=CONFIDENCE_SUPPORTED),)
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source=RELATION_SOURCE_DETERMINISTIC_RESTART,
        watch_listen_relations=relations,
    )
    assert final.would_merge is True
    assert final.conflict is False


def test_19_provider_unavailable_fail_open_empty_index_no_evidence_for_pair():
    """provider-unavailable / missing-evidence: `attempt_relation_
    hypotheses_for_pair` returns `()` for an absent/None index -- callers
    never special-case this, it flows straight into the same fail-open
    branch as test_01."""
    assert attempt_relation_hypotheses_for_pair(None, "a", "b") == ()
    assert attempt_relation_hypotheses_for_pair({}, "a", "b") == ()
    index = build_understanding_span_index(())
    assert attempt_relation_hypotheses_for_pair(index, "a", "b") == ()


# ===========================================================================
# 20-22. `build_understanding_span_index` / `attempt_relation_hypotheses_
# for_pair` -- indexing correctness, order-independence, non-adjacent pairs
# ===========================================================================

def test_20_span_index_flattens_multiple_sources_by_span_id():
    span_a = _span("a")
    span_b = _span("b")
    understandings = (
        WatchListenUnderstanding(source_asset_id="src1", understanding_spans=(span_a,)),
        WatchListenUnderstanding(source_asset_id="src2", understanding_spans=(span_b,)),
    )
    index = build_understanding_span_index(understandings)
    assert index == {"a": span_a, "b": span_b}


def test_21_order_independence_index_built_in_either_order_same_result():
    span_a = _span("a")
    span_b = _span("b")
    u1 = WatchListenUnderstanding(source_asset_id="src1", understanding_spans=(span_a,))
    u2 = WatchListenUnderstanding(source_asset_id="src2", understanding_spans=(span_b,))
    forward = build_understanding_span_index((u1, u2))
    backward = build_understanding_span_index((u2, u1))
    assert forward == backward


def test_22_hypotheses_for_pair_filters_by_exact_left_span_id_only():
    relation_from_left = _relation(RELATION_RETRY, left_span_id="left")
    relation_from_other = _relation(RELATION_CONTINUATION, left_span_id="other")
    right_span = _span("right", relations=(relation_from_left, relation_from_other))
    index = {"right": right_span}
    result = attempt_relation_hypotheses_for_pair(index, "left", "right")
    assert result == (relation_from_left,)


def test_22b_non_adjacent_pair_with_no_computed_relation_is_empty():
    # D-157 only computes relations between immediate map-order neighbors;
    # a non-adjacent candidate pair simply has none -- never an error.
    right_span = _span("right", relations=(_relation(RELATION_RETRY, left_span_id="immediate_predecessor"),))
    index = {"right": right_span}
    assert attempt_relation_hypotheses_for_pair(index, "far_away_span", "right") == ()


# ===========================================================================
# 23. Deterministic topology / first-SUPPORTED-wins tie-break
# ===========================================================================

def test_23_deterministic_topology_first_supported_relation_wins_ties():
    relations = (
        _relation(RELATION_CONTINUATION, confidence=CONFIDENCE_WEAK),
        _relation(RELATION_RETRY, confidence=CONFIDENCE_SUPPORTED),
        _relation(RELATION_CORRECTION, confidence=CONFIDENCE_SUPPORTED),
    )
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source=RELATION_SOURCE_DETERMINISTIC_RESTART,
        watch_listen_relations=relations,
    )
    # First SUPPORTED, non-UNCERTAIN entry in append order (RETRY) wins --
    # deterministic, never a second run producing CORRECTION instead.
    assert final.relation == RELATION_RETRY
    assert final.would_merge is True


# ===========================================================================
# 24-25. `attempt_relationship_diagnostics` -- tail-safe counts-only summary
# ===========================================================================

def test_24_diagnostics_counts_every_relation_bucket_correctly():
    rows = [
        resolve_final_attempt_relation(would_merge=True, would_merge_source="x", watch_listen_relations=(_relation(RELATION_RETRY),)),
        resolve_final_attempt_relation(would_merge=True, would_merge_source="x", watch_listen_relations=(_relation(RELATION_CONTINUATION),)),
        resolve_final_attempt_relation(would_merge=True, would_merge_source="x", watch_listen_relations=(_relation(RELATION_CORRECTION),)),
        resolve_final_attempt_relation(would_merge=True, would_merge_source="x", watch_listen_relations=(_relation(RELATION_COMPLEMENTARY),)),
        resolve_final_attempt_relation(would_merge=True, would_merge_source="x", watch_listen_relations=(_relation(RELATION_NEW_AUDIENCE_BEAT),)),
        resolve_final_attempt_relation(would_merge=False, would_merge_source="x", watch_listen_relations=(_relation(RELATION_RETRY),)),
    ]
    diag = attempt_relationship_diagnostics(rows)
    assert diag["retry_count"] == 1
    assert diag["continuation_count"] == 1
    assert diag["correction_count"] == 1
    assert diag["complementary_count"] == 1
    assert diag["new_beat_count"] == 1
    assert diag["uncertain_relation_count"] == 1  # the last row: conflict the other way
    assert diag["pair_count"] == 6
    assert diag["watch_listen_conflict_count"] == 5  # every non-agreeing row above


def test_25_diagnostics_never_include_transcript_or_reason_text():
    rows = [resolve_final_attempt_relation(would_merge=True, would_merge_source="x", watch_listen_relations=(_relation(RELATION_CONTINUATION),))]
    diag = attempt_relationship_diagnostics(rows)
    for value in diag.values():
        assert not isinstance(value, str) or len(value) < 40  # counts-only shape; no long text field


# ===========================================================================
# 26. Fail-open capability flag itself
# ===========================================================================

def test_26_flag_defaults_off_when_env_unset(monkeypatch):
    _clear_flag(monkeypatch)
    assert watch_listen_family_evidence_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on"])
def test_26b_flag_recognizes_true_like_values(value):
    assert watch_listen_family_evidence_enabled({_ENV_FLAG: value}) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_26c_flag_recognizes_false_like_values(value):
    assert watch_listen_family_evidence_enabled({_ENV_FLAG: value}) is False


# ===========================================================================
# 27-32. Live wiring through `reconcile_semantic_idea_equivalence` --
# the REAL production call site (not just the isolated authority module).
# Reuses the SAME deterministic same_opening_restart fixture shape already
# proven in test_cutsell_d100_multimodal_retry_corroboration.py.
# ===========================================================================

_FAILED = "When my contract ended I spoke with my doctor about every test available today"
_CLEAN = "When my contract ended I switched to a different doctor about every test available today"


def test_27_flag_off_conflicting_watch_listen_evidence_never_consulted():
    """flag OFF (the true default): merges exactly as pre-D-158, even
    when `watch_listen_spans_by_id` carries evidence that WOULD have
    conflicted if the flag were on."""
    takes = (_take("failed", 0.0, 5.0, _FAILED), _take("clean", 6.0, 11.0, _CLEAN))
    right_span = _span("clean", relations=(_relation(RELATION_CONTINUATION, left_span_id="failed"),))
    spans_by_id = {"clean": right_span}
    merged, diag = reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",)), takes, None, watch_listen_spans_by_id=spans_by_id,
    )
    assert len(merged) == 1 and set(merged[0]) == {"failed", "clean"}
    assert diag["watch_listen_family_evidence"] == {"status": "disabled"}


def test_28_flag_on_no_evidence_supplied_merges_unchanged(monkeypatch):
    _set_flag_on(monkeypatch)
    takes = (_take("failed", 0.0, 5.0, _FAILED), _take("clean", 6.0, 11.0, _CLEAN))
    merged, diag = reconcile_semantic_idea_equivalence((("failed",), ("clean",)), takes, None)
    assert len(merged) == 1 and set(merged[0]) == {"failed", "clean"}
    assert diag["watch_listen_family_evidence"] == {"status": "no_pairs_evaluated"}


def test_29_flag_on_agreeing_evidence_merges_and_reports_evaluated(monkeypatch):
    _set_flag_on(monkeypatch)
    takes = (_take("failed", 0.0, 5.0, _FAILED), _take("clean", 6.0, 11.0, _CLEAN))
    right_span = _span("clean", relations=(_relation(RELATION_RETRY, left_span_id="failed"),))
    spans_by_id = {"clean": right_span}
    merged, diag = reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",)), takes, None, watch_listen_spans_by_id=spans_by_id,
    )
    assert len(merged) == 1 and set(merged[0]) == {"failed", "clean"}
    summary = diag["watch_listen_family_evidence"]
    assert summary["status"] == "evaluated"
    assert summary["retry_count"] == 1
    assert summary["conflict_blocked_count"] == 0


def test_30_flag_on_conflicting_evidence_blocks_the_merge(monkeypatch):
    """The one live behavior change this whole task authorizes: real
    SUPPORTED-confidence Watch+Listen CONTINUATION evidence for this
    EXACT pair withholds a merge the deterministic restart rule would
    otherwise have made."""
    _set_flag_on(monkeypatch)
    takes = (_take("failed", 0.0, 5.0, _FAILED), _take("clean", 6.0, 11.0, _CLEAN))
    right_span = _span("clean", relations=(_relation(RELATION_CONTINUATION, left_span_id="failed"),))
    spans_by_id = {"clean": right_span}
    merged, diag = reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",)), takes, None, watch_listen_spans_by_id=spans_by_id,
    )
    assert merged == (("failed",), ("clean",)), "merge must be withheld, never forced through"
    summary = diag["watch_listen_family_evidence"]
    assert summary["status"] == "evaluated"
    assert summary["conflict_blocked_count"] == 1
    assert summary["continuation_count"] == 1
    assert summary["watch_listen_conflict_count"] == 1


def test_31_flag_on_source_ids_and_timestamps_never_mutated_by_the_conflict_path(monkeypatch):
    _set_flag_on(monkeypatch)
    failed = _take("failed", 0.0, 5.0, _FAILED)
    clean = _take("clean", 6.0, 11.0, _CLEAN)
    right_span = _span("clean", relations=(_relation(RELATION_CONTINUATION, left_span_id="failed"),))
    spans_by_id = {"clean": right_span}
    reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",)), (failed, clean), None, watch_listen_spans_by_id=spans_by_id,
    )
    assert failed.clip_id == "failed" and failed.start == 0.0 and failed.end == 5.0
    assert clean.clip_id == "clean" and clean.start == 6.0 and clean.end == 11.0


def test_32_flag_on_no_watch_listen_index_supplied_is_a_total_noop(monkeypatch):
    """`watch_listen_spans_by_id=None` (the value `pipeline.py` passes
    whenever the flag is off, or no understandings were computed) must
    behave exactly like flag-off from this function's own perspective --
    belt-and-suspenders, not a second flag."""
    _set_flag_on(monkeypatch)
    takes = (_take("failed", 0.0, 5.0, _FAILED), _take("clean", 6.0, 11.0, _CLEAN))
    merged, diag = reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",)), takes, None, watch_listen_spans_by_id=None,
    )
    assert len(merged) == 1 and set(merged[0]) == {"failed", "clean"}
    assert diag["watch_listen_family_evidence"] == {"status": "no_pairs_evaluated"}


# ===========================================================================
# Module-leaf / no-import structural tests: D-150, D-123, D-128, Boundary,
# Pacing, BestTake non-interference; no provider/network call anywhere.
# ===========================================================================

def test_module_never_imports_a_downstream_authority_or_provider():
    import cutsell_worker.attempt_relationship_authority as mod
    source = open(mod.__file__, encoding="utf-8").read()
    forbidden = [
        "semantic_authority_observability",  # D-150 comparative-winner gate
        "deterministic_best_take_authority",  # BestTake
        "take_judge_provider",  # DeliveryScorer/BestTake ranking
        "boundary_engine",
        "temporal_editing",  # Boundary/Pacing physical timing
        "renderer",
        "composite_resolver",
        "google.generativeai",
        "requests",
        "httpx",
        "urllib",
    ]
    for name in forbidden:
        assert f"import {name}" not in source and f"from .{name}" not in source, name


def test_no_provider_or_network_symbol_referenced_anywhere_in_module():
    import cutsell_worker.attempt_relationship_authority as mod
    source = open(mod.__file__, encoding="utf-8").read()
    for token in ("generate_content", "genai.", "requests.post", "requests.get", "socket."):
        assert token not in source


def test_d150_family_complete_context_gate_unaffected_by_this_module():
    # D-150's own gate operates strictly AFTER family membership already
    # exists; this authority operates strictly BEFORE. Proven structurally:
    # this module never imports the D-150 gate module (see test above) and
    # exposes no symbol the D-150 module could call back into.
    import cutsell_worker.semantic_authority_observability as d150
    source = open(d150.__file__, encoding="utf-8").read()
    assert "attempt_relationship_authority" not in source
    assert "watch_listen_understanding" not in source


def test_take_grouping_provider_lazy_import_breaks_the_cycle_cleanly():
    # Confirms the D-158 lazy-import pattern actually works end-to-end
    # (regression guard for the circular-import bug fixed during this task).
    import cutsell_worker.take_grouping_provider as tgp
    import cutsell_worker  # noqa: F401 -- full-package import must also succeed
    module = tgp._attempt_relationship_authority()
    assert module.SCHEMA_VERSION.startswith("cutsell.attempt_relationship_authority")


def test_pipeline_wiring_present_and_flag_gated():
    # `cutsell_worker.pipeline.build_flow_b_draft` is monkey-patch-wrapped
    # by ~14 pre-existing `install_*` chains at package-import time (see
    # `cutsell_worker/__init__.py`), so `inspect.signature` on the imported
    # name reflects the OUTERMOST `(*args, **kwargs)` wrapper, not this
    # function's own real signature -- reading the module source directly
    # is the robust check (same pattern already used by this test file's
    # own module-leaf source-text assertions above).
    import cutsell_worker.pipeline as pipeline_mod
    source = open(pipeline_mod.__file__, encoding="utf-8").read()
    assert "watch_listen_understandings: Iterable[\"WatchListenUnderstanding\"] = ()" in source \
        or "watch_listen_understandings: Iterable[WatchListenUnderstanding] = ()" in source
    assert "watch_listen_spans_by_id=watch_listen_spans_by_id" in source
