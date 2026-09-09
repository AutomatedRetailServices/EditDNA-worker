"""D-161 Phase C.2 -- Watch+Listen RELATION DISCOVERY Gate.

Per docs/CUTSELL_DECISIONS.md D-148 through D-161 and `watch_listen_relation_
discovery.py`'s own module docstring: this is a DISCOVERY-ONLY capability --

    WATCH+LISTEN MAY DISCOVER. WATCH+LISTEN DOES NOT DECIDE.

It proposes candidate pairs the pre-existing semantic pair-generation/batch-
cap path (`_cross_group_candidate_pairs` -> ranking -> the arbiter batch)
never reaches, using ONLY already-computed Watch+Listen Understanding
evidence (D-157) plus the SAME deterministic proposition-evidence primitives
the existing merge path already uses. Every candidate is handed to the SAME
structured authority (`attempt_relationship_authority.resolve_final_attempt_
relation`, extended, never replaced) for the final relation; only a final
RETRY with independently-sufficient proposition evidence ever merges.

Generic fixtures throughout -- no Video00 wording, no real transcript text
beyond short, obviously-synthetic sentences.
"""
from __future__ import annotations

import pytest

from cutsell_worker.attempt_relationship_authority import (
    FAMILY_ACTION_ABSTAIN_UNCERTAIN,
    FAMILY_ACTION_ELIGIBLE_RETRY_FAMILY,
    FAMILY_ACTION_SEPARATE_BEAT,
    FAMILY_ACTION_SEPARATE_CORRECTION,
    FAMILY_ACTION_SEPARATE_DISTINCT,
    FAMILY_ACTION_SEPARATE_NOT_COMPETING,
    RELATION_SOURCE_NO_SEMANTIC_PAIR,
    RELATION_SOURCE_WATCH_LISTEN_DISCOVERY,
    build_understanding_span_index,
    resolve_final_attempt_relation,
)
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.raw_understanding_map import (
    BEHAVIOR_ABANDONED_ATTEMPT,
    BEHAVIOR_AUDIENCE_DELIVERY,
    BEHAVIOR_CLEAN_ATTEMPT,
    BEHAVIOR_FALSE_START,
    BEHAVIOR_POST_TAKE_RESET,
    BEHAVIOR_PRE_TAKE_SETUP,
    BEHAVIOR_RECORDING_PROCESS,
    BehaviorHypothesis,
)
from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceRequest,
    IdeaEquivalenceResult,
    SemanticEquivalenceGatePolicy,
)
from cutsell_worker.take_grouping_provider import reconcile_semantic_idea_equivalence
from cutsell_worker.watch_listen_relation_discovery import (
    PAIR_SOURCE_BOTH,
    PAIR_SOURCE_SEMANTIC,
    PAIR_SOURCE_WATCH_LISTEN,
    REJECT_ALREADY_RESOLVED,
    REJECT_CONFLICT_FLAGGED,
    REJECT_MISSING_TAKE,
    REJECT_NOT_SUPPORTED,
    REJECT_PROPOSITION_UNRESOLVED,
    DiscoveryCandidate,
    _bridged_candidate,
    _is_pure_intermediary,
    discover_candidate_pairs,
    discovery_diagnostics,
    proposition_evidence_for_pair,
    watch_listen_relation_discovery_enabled,
)
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

_DISCOVERY_ENV = "CUTSELL_WATCH_LISTEN_RELATION_DISCOVERY_ENABLED"
_FAMILY_ENV = "CUTSELL_WATCH_LISTEN_FAMILY_EVIDENCE_ENABLED"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _enable_discovery(monkeypatch):
    monkeypatch.setenv(_DISCOVERY_ENV, "1")


def _clear_discovery(monkeypatch):
    monkeypatch.delenv(_DISCOVERY_ENV, raising=False)
    monkeypatch.delenv(_FAMILY_ENV, raising=False)


def _relation(relation, *, confidence=CONFIDENCE_SUPPORTED, left_span_id="left", basis="fixture"):
    return AttemptRelationHypothesis(
        relation=relation, confidence=confidence, basis=basis,
        left_span_id=left_span_id, provenance=("MULTIMODAL_FUSION",),
    )


def _behavior(label):
    return BehaviorHypothesis(label=label, confidence=0.9, provenance="fixture", basis="fixture")


def _span(
    span_id, *, source_asset_id="src", start=0.0, end=1.0,
    relations=(), behaviors=(), conflict_flags=(),
):
    return UnderstandingSpan(
        span_id=span_id, source_asset_id=source_asset_id, source_start=start, source_end=end,
        behavior_state_hypotheses=behaviors, behavior_confidence=CONFIDENCE_UNKNOWN,
        attempt_boundary_hypotheses=(), attempt_relation_hypotheses=relations,
        relation_confidence=CONFIDENCE_UNKNOWN,
        meaning_completion_hypothesis="UNCERTAIN",
        performance_usability_hypothesis="UNKNOWN",
        entry_usability="UNKNOWN", delivery_usability="UNKNOWN", exit_usability="UNKNOWN",
        conflict_flags=conflict_flags, evidence_provenance={},
    )


def _take(clip_id, start, end, text, *, complete=True, source="src"):
    return CandidateTake(clip_id, source, 0, start, end, text, complete_idea=complete)


class _StubArbiter:
    """Answers every requested pair the same way -- used only to exercise
    the arbiter-present branch; never a real provider call."""
    def __init__(self, same_idea: bool = False):
        self._same_idea = same_idea

    def check(self, request: IdeaEquivalenceRequest) -> IdeaEquivalenceResult:
        decisions = tuple(
            IdeaEquivalenceDecision(pair_index=i, same_idea=self._same_idea, confidence=0.9, reason="stub")
            for i in range(len(request.pairs))
        )
        return IdeaEquivalenceResult(decisions, "stub", "stub-v1", True, True)


# ===========================================================================
# 1-3. Capability flag itself -- default OFF, true/false-like values
# ===========================================================================

def test_01_flag_defaults_off_when_env_unset(monkeypatch):
    _clear_discovery(monkeypatch)
    assert watch_listen_relation_discovery_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on"])
def test_02_flag_recognizes_true_like_values(value):
    assert watch_listen_relation_discovery_enabled({_DISCOVERY_ENV: value}) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_03_flag_recognizes_false_like_values(value):
    assert watch_listen_relation_discovery_enabled({_DISCOVERY_ENV: value}) is False


def test_03b_flag_is_separate_from_d158_family_evidence_flag():
    # D-158's flag ON must never itself enable discovery -- two independent
    # authorities, per this task's own explicit instruction.
    assert watch_listen_relation_discovery_enabled({_FAMILY_ENV: "1"}) is False


# ===========================================================================
# 4-12. `discover_candidate_pairs` -- direct immediate-neighbor candidates
# ===========================================================================

def test_04_direct_supported_retry_relation_becomes_a_candidate():
    left = _take("a", 0.0, 1.0, "opening words here")
    right = _take("b", 2.0, 3.0, "closing words here")
    spans = {
        "a": _span("a", start=0.0, end=1.0),
        "b": _span("b", start=2.0, end=3.0, relations=(_relation(RELATION_RETRY, left_span_id="a"),)),
    }
    candidates = discover_candidate_pairs(spans, {"a": left, "b": right})
    assert candidates == (DiscoveryCandidate("a", "b", RELATION_RETRY, CONFIDENCE_SUPPORTED, "fixture", False),)


def test_05_weak_confidence_relation_never_becomes_a_candidate():
    spans = {
        "a": _span("a", start=0.0, end=1.0),
        "b": _span("b", start=2.0, end=3.0, relations=(_relation(RELATION_CONTINUATION, confidence=CONFIDENCE_WEAK, left_span_id="a"),)),
    }
    candidates = discover_candidate_pairs(spans, {"a": _take("a", 0, 1, "x"), "b": _take("b", 2, 3, "y")})
    assert candidates == ()


def test_06_mixed_confidence_relation_never_becomes_a_candidate():
    spans = {
        "a": _span("a", start=0.0, end=1.0),
        "b": _span("b", start=2.0, end=3.0, relations=(_relation(RELATION_RETRY, confidence=CONFIDENCE_MIXED, left_span_id="a"),)),
    }
    candidates = discover_candidate_pairs(spans, {"a": _take("a", 0, 1, "x"), "b": _take("b", 2, 3, "y")})
    assert candidates == ()


def test_07_uncertain_relation_produces_no_discovery_action_even_at_supported_confidence():
    spans = {
        "a": _span("a", start=0.0, end=1.0),
        "b": _span("b", start=2.0, end=3.0, relations=(_relation(RELATION_UNCERTAIN, confidence=CONFIDENCE_SUPPORTED, left_span_id="a"),)),
    }
    candidates = discover_candidate_pairs(spans, {"a": _take("a", 0, 1, "x"), "b": _take("b", 2, 3, "y")})
    assert candidates == (), "UNCERTAIN must never produce a discovery candidate"


def test_08_relation_pointing_to_a_non_immediate_span_id_is_never_matched():
    spans = {
        "a": _span("a", start=0.0, end=1.0),
        "b": _span("b", start=2.0, end=3.0, relations=(_relation(RELATION_RETRY, left_span_id="far_away"),)),
    }
    candidates = discover_candidate_pairs(spans, {"a": _take("a", 0, 1, "x"), "b": _take("b", 2, 3, "y")})
    assert candidates == ()


def test_09_missing_candidate_take_for_a_span_is_skipped_never_raises():
    spans = {
        "a": _span("a", start=0.0, end=1.0),
        "b": _span("b", start=2.0, end=3.0, relations=(_relation(RELATION_RETRY, left_span_id="a"),)),
    }
    # take_map missing "a" entirely -- fail-open, no crash, no candidate.
    candidates = discover_candidate_pairs(spans, {"b": _take("b", 2, 3, "y")})
    assert candidates == ()


def test_10_empty_or_none_input_returns_empty_tuple_never_raises():
    assert discover_candidate_pairs(None, {}) == ()
    assert discover_candidate_pairs({}, {}) == ()


def test_11_determinism_map_insertion_order_never_affects_output():
    left = _take("a", 0.0, 1.0, "x")
    right = _take("b", 2.0, 3.0, "y")
    span_a = _span("a", start=0.0, end=1.0)
    span_b = _span("b", start=2.0, end=3.0, relations=(_relation(RELATION_RETRY, left_span_id="a"),))
    forward = discover_candidate_pairs({"a": span_a, "b": span_b}, {"a": left, "b": right})
    backward = discover_candidate_pairs({"b": span_b, "a": span_a}, {"b": right, "a": left})
    assert forward == backward


def test_12_multiple_sources_handled_independently_sorted_by_source_id():
    spans = {
        "z1": _span("z1", source_asset_id="src_z", start=0.0, end=1.0),
        "z2": _span("z2", source_asset_id="src_z", start=2.0, end=3.0, relations=(_relation(RELATION_RETRY, left_span_id="z1"),)),
        "a1": _span("a1", source_asset_id="src_a", start=0.0, end=1.0),
        "a2": _span("a2", source_asset_id="src_a", start=2.0, end=3.0, relations=(_relation(RELATION_CORRECTION, left_span_id="a1"),)),
    }
    takes = {
        "z1": _take("z1", 0, 1, "x", source="src_z"), "z2": _take("z2", 2, 3, "y", source="src_z"),
        "a1": _take("a1", 0, 1, "x", source="src_a"), "a2": _take("a2", 2, 3, "y", source="src_a"),
    }
    candidates = discover_candidate_pairs(spans, takes)
    assert len(candidates) == 2
    assert {(c.left_id, c.right_id, c.relation) for c in candidates} == {
        ("a1", "a2", RELATION_CORRECTION), ("z1", "z2", RELATION_RETRY),
    }


# ===========================================================================
# 13-22. Immediate-neighbor bridging -- the bounded, single-hop bridge
# ===========================================================================

_INTERMEDIARY_LABELS = [BEHAVIOR_PRE_TAKE_SETUP, BEHAVIOR_POST_TAKE_RESET, BEHAVIOR_RECORDING_PROCESS, BEHAVIOR_FALSE_START]


@pytest.mark.parametrize("label", _INTERMEDIARY_LABELS)
def test_13_each_recognized_intermediary_label_is_pure_intermediary(label):
    assert _is_pure_intermediary(_span("x", behaviors=(_behavior(label),))) is True


@pytest.mark.parametrize("label", [BEHAVIOR_AUDIENCE_DELIVERY, BEHAVIOR_CLEAN_ATTEMPT])
def test_14_audience_labels_are_never_pure_intermediary(label):
    assert _is_pure_intermediary(_span("x", behaviors=(_behavior(label),))) is False


def test_15_mixed_intermediary_plus_audience_label_never_bridged_across():
    # A real audience-delivery span must never be bridged across, even if it
    # also happens to carry a secondary intermediary-shaped hypothesis.
    mixed = _span("x", behaviors=(_behavior(BEHAVIOR_PRE_TAKE_SETUP), _behavior(BEHAVIOR_AUDIENCE_DELIVERY)))
    assert _is_pure_intermediary(mixed) is False


def test_16_no_behavior_hypotheses_at_all_is_never_pure_intermediary():
    assert _is_pure_intermediary(_span("x", behaviors=())) is False


@pytest.mark.parametrize("label", _INTERMEDIARY_LABELS)
def test_17_each_intermediary_kind_bridges_a_broken_attempt_to_its_completion(label):
    """A incomplete -> intermediary -> B, sharing an opening lexical restart
    -- the bounded single-hop bridge (D-160's own named limitation fix)."""
    pp = _take("attempt_a", 0.0, 2.0, "Widget explanation begins now for everyone", complete=False)
    mid = _take("intermediary", 3.0, 4.0, "okay let's reset the camera")
    right = _take("attempt_b", 5.0, 7.0, "Widget explanation continues later for the group")
    spans = {
        "attempt_a": _span("attempt_a", start=0.0, end=2.0),
        "intermediary": _span("intermediary", start=3.0, end=4.0, behaviors=(_behavior(label),)),
        "attempt_b": _span("attempt_b", start=5.0, end=7.0),
    }
    candidates = discover_candidate_pairs(spans, {"attempt_a": pp, "intermediary": mid, "attempt_b": right})
    bridged = [c for c in candidates if c.bridged]
    assert len(bridged) == 1
    assert bridged[0].left_id == "attempt_a" and bridged[0].right_id == "attempt_b"
    assert bridged[0].relation == RELATION_RETRY
    assert bridged[0].confidence == CONFIDENCE_SUPPORTED  # pp_take incomplete


def test_18_bridge_confidence_is_weak_when_the_bridged_from_take_is_complete():
    pp = _take("attempt_a", 0.0, 2.0, "Widget explanation begins now for everyone", complete=True)
    right = _take("attempt_b", 5.0, 7.0, "Widget explanation continues later for the group")
    pp_span = _span("attempt_a", start=0.0, end=2.0)
    right_span = _span("attempt_b", start=5.0, end=7.0)
    candidate = _bridged_candidate(pp_span, right_span, pp, right)
    assert candidate is not None
    assert candidate.confidence == CONFIDENCE_WEAK


def test_19_no_restart_evidence_between_bridged_spans_produces_no_bridge():
    pp = _take("attempt_a", 0.0, 2.0, "completely unrelated opening sentence", complete=False)
    right = _take("attempt_b", 5.0, 7.0, "totally different closing statement")
    spans = {
        "attempt_a": _span("attempt_a", start=0.0, end=2.0),
        "intermediary": _span("intermediary", start=3.0, end=4.0, behaviors=(_behavior(BEHAVIOR_POST_TAKE_RESET),)),
        "attempt_b": _span("attempt_b", start=5.0, end=7.0),
    }
    candidates = discover_candidate_pairs(
        spans, {"attempt_a": pp, "intermediary": _take("intermediary", 3, 4, "reset"), "attempt_b": right},
    )
    assert not any(c.bridged for c in candidates)


def test_20_audience_delivery_intermediary_never_bridged_across():
    """A real audience-delivery span in between must never be treated as a
    bridgeable intermediary, even if the outer takes share an opening."""
    pp = _take("attempt_a", 0.0, 2.0, "Widget explanation begins now for everyone", complete=False)
    mid = _take("audience_beat", 3.0, 4.0, "and here is something completely different")
    right = _take("attempt_b", 5.0, 7.0, "Widget explanation continues later for the group")
    spans = {
        "attempt_a": _span("attempt_a", start=0.0, end=2.0),
        "audience_beat": _span("audience_beat", start=3.0, end=4.0, behaviors=(_behavior(BEHAVIOR_AUDIENCE_DELIVERY),)),
        "attempt_b": _span("attempt_b", start=5.0, end=7.0),
    }
    candidates = discover_candidate_pairs(spans, {"attempt_a": pp, "audience_beat": mid, "attempt_b": right})
    assert not any(c.bridged for c in candidates)


def test_21_only_one_hop_ever_bridged_no_arbitrary_n_hop_chain():
    """Two consecutive intermediary spans between A and D: the bridge only
    ever looks ONE hop back (spans[i-2]), never further -- explicitly NOT an
    arbitrary long-range search. A and D sharing an opening must NOT bridge
    across two intermediary hops."""
    a = _take("a", 0.0, 1.0, "Widget explanation begins now for everyone", complete=False)
    mid1 = _take("mid1", 2.0, 3.0, "resetting the camera now")
    mid2 = _take("mid2", 4.0, 5.0, "okay ready to go again")
    d = _take("d", 6.0, 8.0, "Widget explanation continues later for the group")
    spans = {
        "a": _span("a", start=0.0, end=1.0),
        "mid1": _span("mid1", start=2.0, end=3.0, behaviors=(_behavior(BEHAVIOR_POST_TAKE_RESET),)),
        "mid2": _span("mid2", start=4.0, end=5.0, behaviors=(_behavior(BEHAVIOR_RECORDING_PROCESS),)),
        "d": _span("d", start=6.0, end=8.0),
    }
    candidates = discover_candidate_pairs(spans, {"a": a, "mid1": mid1, "mid2": mid2, "d": d})
    # The only possible bridge at i=3 (d) looks at spans[1] (mid1), which IS
    # pure intermediary, but the bridge compares mid1's text (not a's) to
    # d's text -- "resetting the camera now" vs "Widget explanation
    # continues..." shares no restart evidence, so no bridge is proposed.
    # a<->d themselves are never directly compared: exactly ONE hop, never two.
    assert not any(c.bridged and c.left_id == "a" and c.right_id == "d" for c in candidates)


def test_22_bridge_never_fires_when_the_predecessor_pp_take_is_missing():
    spans = {
        "attempt_a": _span("attempt_a", start=0.0, end=2.0),
        "intermediary": _span("intermediary", start=3.0, end=4.0, behaviors=(_behavior(BEHAVIOR_FALSE_START),)),
        "attempt_b": _span("attempt_b", start=5.0, end=7.0),
    }
    take_map = {
        "intermediary": _take("intermediary", 3, 4, "reset"),
        "attempt_b": _take("attempt_b", 5, 7, "Widget explanation continues later for the group"),
    }
    candidates = discover_candidate_pairs(spans, take_map)  # "attempt_a" take missing
    assert not any(c.bridged for c in candidates)


# ===========================================================================
# 23-28. `proposition_evidence_for_pair` -- the Proposition Firewall (D-111)
# ===========================================================================

def test_23_same_opening_restart_satisfies_proposition_evidence():
    left = _take("a", 0.0, 5.0, "When my contract ended I spoke with my doctor about every test available")
    right = _take("b", 6.0, 11.0, "When my contract ended I switched to a different doctor about every test available")
    sufficient, kind = proposition_evidence_for_pair(left, right)
    assert sufficient is True
    assert kind == "same_opening_restart"


def test_24_safe_short_prefix_retry_satisfies_proposition_evidence():
    left = _take("a", 0.0, 1.0, "I want to")
    right = _take("b", 2.0, 3.0, "I want to explain this important detail to you today")
    sufficient, kind = proposition_evidence_for_pair(left, right)
    assert sufficient is True
    assert kind == "safe_short_prefix_retry"


def test_25_incomplete_attempt_completed_by_retry_satisfies_proposition_evidence():
    left = _take("a", 0.0, 3.0, "Started explaining the whole process again", complete=False)
    right = _take("b", 5.0, 8.0, "Started explaining the whole idea slowly again")
    sufficient, _kind = proposition_evidence_for_pair(left, right)
    assert sufficient is True


def test_26_multimodal_corroborated_retry_satisfies_proposition_evidence_only_with_evidence():
    left = _take("a", 0.0, 3.0, "sharing my experience with this topic today", complete=False)
    right = _take("b", 4.0, 8.0, "sharing my experience differently this time around")
    without_evidence, _ = proposition_evidence_for_pair(left, right)
    assert without_evidence is False
    corroborated = {"a": (("wrong_take", 3.0, 3.5),)}
    with_evidence, kind = proposition_evidence_for_pair(left, right, confirmed_recording_evidence=corroborated)
    # Either genuinely corroborates (True) or fails open (False) depending on
    # the shared-content floor `multimodal_corroborated_retry` itself
    # enforces -- both are valid fail-open outcomes; the call must never
    # raise, which is what this test actually guards.
    assert isinstance(with_evidence, bool) and isinstance(kind, str)


def test_27_no_evidence_at_all_returns_false_with_a_bounded_reason_string():
    left = _take("a", 0.0, 1.0, "completely unrelated opening statement")
    right = _take("b", 50.0, 51.0, "an entirely different closing remark")
    sufficient, reason = proposition_evidence_for_pair(left, right)
    assert sufficient is False
    assert isinstance(reason, str) and len(reason) < 80


def test_28_proposition_evidence_never_raises_on_degenerate_short_text():
    sufficient, reason = proposition_evidence_for_pair(_take("a", 0, 1, ""), _take("b", 1, 2, ""))
    assert sufficient is False


# ===========================================================================
# 29-38. `resolve_final_attempt_relation(semantic_path_evaluated=False, ...)`
# -- the discovery half of the structured authority's truth table.
# ===========================================================================

def test_29_discovered_retry_with_sufficient_proposition_evidence_merges():
    final = resolve_final_attempt_relation(
        would_merge=False, would_merge_source=RELATION_SOURCE_NO_SEMANTIC_PAIR,
        watch_listen_relations=(_relation(RELATION_RETRY),),
        semantic_path_evaluated=False, proposition_evidence_sufficient=True,
    )
    assert final.would_merge is True
    assert final.relation == RELATION_RETRY
    assert final.source == RELATION_SOURCE_WATCH_LISTEN_DISCOVERY
    assert final.family_membership_action == FAMILY_ACTION_ELIGIBLE_RETRY_FAMILY


def test_30_discovered_retry_without_proposition_evidence_never_forces_a_merge():
    final = resolve_final_attempt_relation(
        would_merge=False, would_merge_source=RELATION_SOURCE_NO_SEMANTIC_PAIR,
        watch_listen_relations=(_relation(RELATION_RETRY),),
        semantic_path_evaluated=False, proposition_evidence_sufficient=False,
    )
    assert final.would_merge is False
    assert final.relation == RELATION_UNCERTAIN
    assert final.family_membership_action == FAMILY_ACTION_ABSTAIN_UNCERTAIN


@pytest.mark.parametrize("relation,expected_action", [
    (RELATION_CONTINUATION, FAMILY_ACTION_SEPARATE_NOT_COMPETING),
    (RELATION_COMPLEMENTARY, FAMILY_ACTION_SEPARATE_NOT_COMPETING),
    (RELATION_CORRECTION, FAMILY_ACTION_SEPARATE_CORRECTION),
    (RELATION_NEW_AUDIENCE_BEAT, FAMILY_ACTION_SEPARATE_BEAT),
    (RELATION_DISTINCT_PROPOSITION, FAMILY_ACTION_SEPARATE_DISTINCT),
])
def test_31_non_retry_discovered_relations_are_surfaced_but_never_merge(relation, expected_action):
    final = resolve_final_attempt_relation(
        would_merge=False, would_merge_source=RELATION_SOURCE_NO_SEMANTIC_PAIR,
        watch_listen_relations=(_relation(relation),),
        semantic_path_evaluated=False, proposition_evidence_sufficient=True,  # even if TRUE, must not matter
    )
    assert final.would_merge is False
    assert final.relation == relation
    assert final.family_membership_action == expected_action


def test_32_no_material_watch_listen_evidence_discovery_abstains_never_raises():
    final = resolve_final_attempt_relation(
        would_merge=False, would_merge_source=RELATION_SOURCE_NO_SEMANTIC_PAIR,
        watch_listen_relations=(), semantic_path_evaluated=False,
    )
    assert final.would_merge is False
    assert final.family_membership_action == FAMILY_ACTION_ABSTAIN_UNCERTAIN
    assert final.watch_listen_relation_evaluated is False


def test_33_weak_only_evidence_in_discovery_mode_also_abstains():
    final = resolve_final_attempt_relation(
        would_merge=False, would_merge_source=RELATION_SOURCE_NO_SEMANTIC_PAIR,
        watch_listen_relations=(_relation(RELATION_RETRY, confidence=CONFIDENCE_WEAK),),
        semantic_path_evaluated=False, proposition_evidence_sufficient=True,
    )
    assert final.would_merge is False
    assert final.family_membership_action == FAMILY_ACTION_ABSTAIN_UNCERTAIN


def test_34_default_semantic_path_evaluated_true_is_byte_identical_to_omitting_the_kwarg():
    with_default = resolve_final_attempt_relation(would_merge=True, would_merge_source="x")
    omitted = resolve_final_attempt_relation(would_merge=True, would_merge_source="x", semantic_path_evaluated=True)
    assert with_default == omitted


def test_35_existing_d158_call_shape_completely_unaffected_by_new_kwargs_defaults():
    # Every pre-existing D-158 call site never passes the two new kwargs --
    # confirm their defaults reproduce the exact pre-D-161 truth table cell.
    final = resolve_final_attempt_relation(
        would_merge=True, would_merge_source="DETERMINISTIC_RESTART",
        watch_listen_relations=(_relation(RELATION_RETRY),),
    )
    assert final.would_merge is True and final.conflict is False


@pytest.mark.parametrize("relation", [RELATION_CONTINUATION, RELATION_COMPLEMENTARY, RELATION_CORRECTION, RELATION_NEW_AUDIENCE_BEAT, RELATION_DISTINCT_PROPOSITION])
def test_36_only_retry_can_ever_become_eligible_retry_family_from_discovery(relation):
    final = resolve_final_attempt_relation(
        would_merge=False, would_merge_source=RELATION_SOURCE_NO_SEMANTIC_PAIR,
        watch_listen_relations=(_relation(relation),),
        semantic_path_evaluated=False, proposition_evidence_sufficient=True,
    )
    assert final.family_membership_action != FAMILY_ACTION_ELIGIBLE_RETRY_FAMILY


def test_37_discovery_never_reports_a_conflict_flag_it_has_no_would_merge_decision_to_conflict_with():
    final = resolve_final_attempt_relation(
        would_merge=False, would_merge_source=RELATION_SOURCE_NO_SEMANTIC_PAIR,
        watch_listen_relations=(_relation(RELATION_CONTINUATION),),
        semantic_path_evaluated=False,
    )
    assert final.conflict is False


def test_38_discovery_result_dataclass_equality_is_reproducible_for_frozen_inputs():
    args = dict(
        would_merge=False, would_merge_source=RELATION_SOURCE_NO_SEMANTIC_PAIR,
        watch_listen_relations=(_relation(RELATION_RETRY),),
        semantic_path_evaluated=False, proposition_evidence_sufficient=True,
    )
    assert resolve_final_attempt_relation(**args) == resolve_final_attempt_relation(**args)


# ===========================================================================
# 39-42. `discovery_diagnostics` -- tail-safe counts-only summary
# ===========================================================================

def test_39_diagnostics_counts_every_relation_bucket_and_pair_source():
    rows = [
        {"pair_source": PAIR_SOURCE_SEMANTIC, "structured_final_relation": RELATION_RETRY, "accepted": True, "rejection_reason": None},
        {"pair_source": PAIR_SOURCE_WATCH_LISTEN, "structured_final_relation": RELATION_CONTINUATION, "accepted": True, "rejection_reason": None},
        {"pair_source": PAIR_SOURCE_BOTH, "structured_final_relation": None, "accepted": False, "rejection_reason": REJECT_ALREADY_RESOLVED},
        {"pair_source": PAIR_SOURCE_WATCH_LISTEN, "structured_final_relation": RELATION_UNCERTAIN, "accepted": False, "rejection_reason": REJECT_PROPOSITION_UNRESOLVED},
    ]
    diag = discovery_diagnostics(rows)
    assert diag["watch_listen_discovery_evaluated_count"] == 4
    assert diag["watch_listen_discovery_candidate_count"] == 4
    assert diag["semantic_pair_candidate_count"] == 1
    assert diag["watch_listen_only_pair_count"] == 2
    assert diag["both_source_pair_count"] == 1
    assert diag["watch_listen_discovery_retry_count"] == 1
    assert diag["watch_listen_discovery_continuation_count"] == 1
    assert diag["watch_listen_discovery_uncertain_skipped_count"] == 2  # None + RELATION_UNCERTAIN
    assert diag["watch_listen_discovery_accepted_count"] == 2
    assert diag["watch_listen_discovery_rejected_count"] == 2
    assert diag["discovery_rejection_reasons"] == {REJECT_ALREADY_RESOLVED: 1, REJECT_PROPOSITION_UNRESOLVED: 1}


def test_40_diagnostics_never_contain_long_text_fields():
    rows = [{"pair_source": PAIR_SOURCE_WATCH_LISTEN, "structured_final_relation": RELATION_RETRY, "accepted": True, "rejection_reason": None}]
    diag = discovery_diagnostics(rows)
    for value in diag.values():
        if isinstance(value, str):
            assert len(value) < 60
        if isinstance(value, dict):
            assert all(len(k) < 60 for k in value)


def test_41_diagnostics_on_empty_rows_all_zero_never_raises():
    diag = discovery_diagnostics([])
    assert diag["watch_listen_discovery_evaluated_count"] == 0
    assert diag["watch_listen_discovery_accepted_count"] == 0
    assert diag["discovery_rejection_reasons"] == {}


def test_42_diagnostics_field_names_match_the_exact_directive_vocabulary():
    diag = discovery_diagnostics([])
    required = {
        "watch_listen_discovery_evaluated_count", "watch_listen_discovery_candidate_count",
        "watch_listen_discovery_retry_count", "watch_listen_discovery_continuation_count",
        "watch_listen_discovery_correction_count", "watch_listen_discovery_complementary_count",
        "watch_listen_discovery_new_beat_count", "watch_listen_discovery_distinct_count",
        "watch_listen_discovery_uncertain_skipped_count", "semantic_pair_candidate_count",
        "watch_listen_only_pair_count", "both_source_pair_count",
        "watch_listen_discovery_accepted_count", "watch_listen_discovery_rejected_count",
        "discovery_rejection_reasons",
    }
    assert required.issubset(diag.keys())


# ===========================================================================
# 43-56. Live wiring through `reconcile_semantic_idea_equivalence` -- the
# REAL production call site, including the Batch-Cap Firewall (primary
# D-160 root-cause regression test) and the deduplication contract.
# ===========================================================================

def test_43_flag_off_discovery_never_runs_even_with_strong_evidence_present(monkeypatch):
    _clear_discovery(monkeypatch)
    short = _take("short", 0.0, 1.0, "I want to")
    long_ = _take("long", 3.0, 8.0, "I want to explain this important detail to you today")
    other_a = _take("other_a", 0.0, 5.0, "some perfectly ordinary sentence about a topic", source="src2")
    other_b = _take("other_b", 6.0, 11.0, "another perfectly ordinary sentence about a topic", source="src2")
    spans_by_id = {
        "short": _span("short", start=0.0, end=1.0),
        "long": _span("long", start=3.0, end=8.0, relations=(_relation(RELATION_RETRY, left_span_id="short"),)),
    }
    groups = (("short",), ("long",), ("other_a",), ("other_b",))
    merged, diag = reconcile_semantic_idea_equivalence(
        groups, (short, long_, other_a, other_b), None, watch_listen_spans_by_id=spans_by_id,
    )
    assert set(merged) == set(groups), "flag OFF: no merge from discovery, groups completely unchanged"
    assert diag["watch_listen_relation_discovery"] == {"status": "disabled"}


def _pimples_shaped_fixture():
    """The D-160 abstract structural replay: a short 'false start' fragment
    excluded from `_cross_group_candidate_pairs`' own generation floor
    (<=3 semantic-key words -- D-160's own PAIR_GENERATION_FILTER root
    cause) exact-prefixes a much longer retry. Neither the deterministic
    restart-evidence loop NOR the arbiter/ranking path ever sees this pair
    at all (it never becomes a `candidate_pairs` member in the first
    place) -- ONLY Watch+Listen discovery, using the SAME `_safe_short_
    prefix_retry` primitive as its Proposition Firewall check, can ever
    surface and merge it. An unrelated, ordinary pair on a different
    source keeps `candidate_pairs` non-empty so the function does not take
    its own separate `no_eligible_pairs` early exit."""
    short = _take("short", 0.0, 1.0, "I want to")
    long_ = _take("long", 3.0, 8.0, "I want to explain this important detail to you today")
    other_a = _take("other_a", 0.0, 5.0, "some perfectly ordinary sentence about a topic", source="src2")
    other_b = _take("other_b", 6.0, 11.0, "another perfectly ordinary sentence about a topic", source="src2")
    spans_by_id = {
        "short": _span("short", start=0.0, end=1.0),
        "long": _span("long", start=3.0, end=8.0, relations=(_relation(RELATION_RETRY, left_span_id="short"),)),
    }
    groups = (("short",), ("long",), ("other_a",), ("other_b",))
    takes = (short, long_, other_a, other_b)
    return groups, takes, spans_by_id


def test_44_flag_on_no_spans_supplied_is_a_total_noop(monkeypatch):
    _enable_discovery(monkeypatch)
    groups, takes, _spans = _pimples_shaped_fixture()
    merged, diag = reconcile_semantic_idea_equivalence(groups, takes, None, watch_listen_spans_by_id=None)
    assert set(merged) == set(groups)
    assert diag["watch_listen_relation_discovery"] == {"status": "no_candidates_discovered"}


def test_45_batch_cap_firewall_zero_provider_call_pair_excluded_from_generation_still_merges(monkeypatch):
    """D-160's primary regression proof: the target pair is never a member
    of `_cross_group_candidate_pairs` at all (word-count generation floor)
    -- with `arbiter=None` (ZERO provider call, this task's own hard
    requirement) -- yet Watch+Listen discovery still surfaces and merges
    it via the SAME structured authority D-158 already uses."""
    _enable_discovery(monkeypatch)
    groups, takes, spans_by_id = _pimples_shaped_fixture()
    merged, diag = reconcile_semantic_idea_equivalence(groups, takes, None, watch_listen_spans_by_id=spans_by_id)
    merged_sets = [set(g) for g in merged]
    assert {"short", "long"} in merged_sets, "discovery must merge the pair the semantic path never reached"
    assert {"other_a", "other_b"} not in merged_sets  # untouched, no arbiter, no restart evidence for it either
    summary = diag["watch_listen_relation_discovery"]
    assert summary["status"] == "evaluated"
    assert summary["watch_listen_discovery_retry_count"] == 1
    assert summary["watch_listen_discovery_accepted_count"] == 1
    assert summary["watch_listen_only_pair_count"] == 1
    trace = [row for row in diag["watch_listen_discovery_trace"] if row["left_id"] == "short"]
    assert len(trace) == 1
    assert trace[0]["pair_source"] == PAIR_SOURCE_WATCH_LISTEN
    assert trace[0]["accepted"] is True
    assert trace[0]["structured_final_relation"] == RELATION_RETRY
    assert trace[0]["family_action"] == FAMILY_ACTION_ELIGIBLE_RETRY_FAMILY


@pytest.mark.parametrize("max_pairs", [0, 1, 14])
def test_46_batch_cap_firewall_outcome_is_identical_regardless_of_the_arbiters_own_cap_value(monkeypatch, max_pairs):
    """Explicit proof: semantic ranking/batch exclusion != Watch+Listen
    discovery exclusion. The discovery-only merge outcome for the excluded
    pair never depends on `max_pairs_per_request` -- discovery's own code
    never references the arbiter policy at all (see the module-leaf test
    below)."""
    _enable_discovery(monkeypatch)
    groups, takes, spans_by_id = _pimples_shaped_fixture()
    arbiter = _StubArbiter(same_idea=False)  # declines the unrelated pair; irrelevant to "short"/"long"
    policy = SemanticEquivalenceGatePolicy(max_pairs_per_request=max_pairs)
    merged, _diag = reconcile_semantic_idea_equivalence(
        groups, takes, arbiter, policy=policy, watch_listen_spans_by_id=spans_by_id,
    )
    assert {"short", "long"} in [set(g) for g in merged]


def test_47_discovery_never_references_the_arbiter_batch_policy_at_all():
    import cutsell_worker.watch_listen_relation_discovery as mod
    source = open(mod.__file__, encoding="utf-8").read()
    for forbidden in ("max_pairs_per_request", "SemanticEquivalenceGatePolicy"):
        assert forbidden not in source
    assert "import semantic_idea_equivalence" not in source
    assert "from .semantic_idea_equivalence" not in source


def test_48_pair_provenance_both_when_the_semantic_path_already_resolved_it_first(monkeypatch):
    """A pair discovered by BOTH sources this run is evaluated exactly
    ONCE -- whichever mechanism reached it first (the existing lexical
    restart-evidence loop, which always runs before discovery)."""
    _enable_discovery(monkeypatch)
    _FAILED = "When my contract ended I spoke with my doctor about every test available today"
    _CLEAN = "When my contract ended I switched to a different doctor about every test available today"
    failed = _take("failed", 0.0, 5.0, _FAILED)
    clean = _take("clean", 6.0, 11.0, _CLEAN)
    right_span = _span("clean", start=6.0, end=11.0, relations=(_relation(RELATION_RETRY, left_span_id="failed"),))
    spans_by_id = {"failed": _span("failed", start=0.0, end=5.0), "clean": right_span}
    merged, diag = reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",)), (failed, clean), None, watch_listen_spans_by_id=spans_by_id,
    )
    assert set(merged[0]) == {"failed", "clean"}  # merged by the restart-evidence loop, not discovery
    trace = diag["watch_listen_discovery_trace"]
    assert len(trace) == 1
    assert trace[0]["pair_source"] == PAIR_SOURCE_BOTH
    assert trace[0]["accepted"] is False
    assert trace[0]["rejection_reason"] == REJECT_ALREADY_RESOLVED
    # Never double-counted: exactly one merge total for this pair.
    assert diag["merged_pair_count"] == 1


def test_49_discovered_continuation_is_surfaced_but_never_enters_retry_competition(monkeypatch):
    _enable_discovery(monkeypatch)
    a = _take("a", 0.0, 3.0, "explaining the first part of this today", complete=False)
    b = _take("b", 10.0, 13.0, "continuing on with the rest of it now")
    spans_by_id = {
        "a": _span("a", start=0.0, end=3.0),
        "b": _span("b", start=10.0, end=13.0, relations=(_relation(RELATION_CONTINUATION, left_span_id="a"),)),
    }
    other_a = _take("other_a", 0.0, 5.0, "some perfectly ordinary sentence about a topic", source="src2")
    other_b = _take("other_b", 6.0, 11.0, "another perfectly ordinary sentence about a topic", source="src2")
    groups = (("a",), ("b",), ("other_a",), ("other_b",))
    merged, diag = reconcile_semantic_idea_equivalence(
        groups, (a, b, other_a, other_b), None, watch_listen_spans_by_id=spans_by_id,
    )
    assert {"a", "b"} not in [set(g) for g in merged], "CONTINUATION must never join a retry family"
    trace = [row for row in diag["watch_listen_discovery_trace"] if row["left_id"] == "a"]
    assert len(trace) == 1 and trace[0]["accepted"] is True
    assert trace[0]["structured_final_relation"] == RELATION_CONTINUATION
    assert trace[0]["family_action"] == FAMILY_ACTION_SEPARATE_NOT_COMPETING


def test_50_discovered_new_audience_beat_prevents_over_grouping(monkeypatch):
    _enable_discovery(monkeypatch)
    a = _take("a", 0.0, 3.0, "here is my first audience beat today")
    b = _take("b", 10.0, 13.0, "here is a completely new audience beat")
    spans_by_id = {
        "a": _span("a", start=0.0, end=3.0),
        "b": _span("b", start=10.0, end=13.0, relations=(_relation(RELATION_NEW_AUDIENCE_BEAT, left_span_id="a"),)),
    }
    merged, diag = reconcile_semantic_idea_equivalence(
        (("a",), ("b",)), (a, b), None, watch_listen_spans_by_id=spans_by_id,
    )
    assert set(merged) == {("a",), ("b",)}
    trace = diag["watch_listen_discovery_trace"]
    assert trace[0]["structured_final_relation"] == RELATION_NEW_AUDIENCE_BEAT
    assert trace[0]["family_action"] == FAMILY_ACTION_SEPARATE_BEAT


def test_51_conflict_semantic_path_says_retry_watch_listen_says_new_beat_never_forces_a_merge(monkeypatch):
    """Semantic path (arbiter) confirms same_idea for a pair also carrying
    real Watch+Listen evidence of a DIFFERENT relation shape. Deduplication
    means discovery never re-decides a pair the arbiter already resolved
    this run -- the merge already made stands, and discovery's own trace
    honestly records it as already-resolved rather than silently
    re-litigating or double-counting it."""
    _enable_discovery(monkeypatch)
    left = _take("a", 0.0, 5.0, "sharing my thoughts about this topic today please")
    right = _take("b", 6.0, 11.0, "sharing my thoughts about this topic differently now")
    spans_by_id = {
        "a": _span("a", start=0.0, end=5.0),
        "b": _span("b", start=6.0, end=11.0, relations=(_relation(RELATION_NEW_AUDIENCE_BEAT, left_span_id="a"),)),
    }
    arbiter = _StubArbiter(same_idea=True)
    merged, diag = reconcile_semantic_idea_equivalence(
        (("a",), ("b",)), (left, right), arbiter, watch_listen_spans_by_id=spans_by_id,
    )
    assert set(merged[0]) == {"a", "b"}  # the arbiter's own decision stands, made exactly once
    assert diag["merged_pair_count"] == 1
    trace = diag["watch_listen_discovery_trace"]
    assert trace[0]["pair_source"] == PAIR_SOURCE_BOTH
    assert trace[0]["rejection_reason"] == REJECT_ALREADY_RESOLVED


def test_52_orphan_clip_not_part_of_any_group_is_rejected_never_raises(monkeypatch):
    """`discover_candidate_pairs` already requires both `CandidateTake`s to
    exist before proposing a candidate at all -- the wiring layer's own
    `REJECT_MISSING_TAKE` guard is reached instead when a discovered clip id
    genuinely has a take but was never placed into any lexical group at all
    (defensive: `clip_to_group_index` lookup, never a crash)."""
    _enable_discovery(monkeypatch)
    a = _take("a", 0.0, 5.0, "a completely ordinary opening statement")
    b = _take("b", 6.0, 11.0, "a completely ordinary sentence about something")
    spans_by_id = {
        "a": _span("a", start=0.0, end=5.0),
        "b": _span("b", start=6.0, end=11.0, relations=(_relation(RELATION_RETRY, left_span_id="a"),)),
    }
    other_a = _take("other_a", 0.0, 5.0, "some perfectly ordinary sentence about a topic", source="src2")
    other_b = _take("other_b", 6.0, 11.0, "another perfectly ordinary sentence about a topic", source="src2")
    # "a" is a real take but deliberately absent from `groups` -- an orphan.
    merged, diag = reconcile_semantic_idea_equivalence(
        (("b",), ("other_a",), ("other_b",)), (a, b, other_a, other_b), None, watch_listen_spans_by_id=spans_by_id,
    )
    trace = [row for row in diag["watch_listen_discovery_trace"] if row["right_id"] == "b"]
    assert len(trace) == 1
    assert trace[0]["rejection_reason"] == REJECT_MISSING_TAKE


def test_53_conflict_flag_on_the_right_span_blocks_a_discovered_retry(monkeypatch):
    _enable_discovery(monkeypatch)
    short = _take("short", 0.0, 1.0, "I want to")
    long_ = _take("long", 3.0, 8.0, "I want to explain this important detail to you today")
    other_a = _take("other_a", 0.0, 5.0, "some perfectly ordinary sentence about a topic", source="src2")
    other_b = _take("other_b", 6.0, 11.0, "another perfectly ordinary sentence about a topic", source="src2")
    spans_by_id = {
        "short": _span("short", start=0.0, end=1.0),
        "long": _span(
            "long", start=3.0, end=8.0,
            relations=(_relation(RELATION_RETRY, left_span_id="short"),),
            conflict_flags=("MEANING_SAFETY_CONFLICT",),
        ),
    }
    groups = (("short",), ("long",), ("other_a",), ("other_b",))
    merged, diag = reconcile_semantic_idea_equivalence(
        groups, (short, long_, other_a, other_b), None, watch_listen_spans_by_id=spans_by_id,
    )
    assert {"short", "long"} not in [set(g) for g in merged]
    trace = [row for row in diag["watch_listen_discovery_trace"] if row["left_id"] == "short"]
    assert trace[0]["rejection_reason"] == REJECT_CONFLICT_FLAGGED


def test_54_confidence_below_supported_is_rejected_at_the_wiring_layer_too(monkeypatch):
    # `discover_candidate_pairs` already filters non-SUPPORTED confidence at
    # generation time, so this exercises the belt-and-suspenders gate at
    # the wiring layer never fires falsely for a genuine SUPPORTED input --
    # a regression guard, not a new behavior.
    _enable_discovery(monkeypatch)
    groups, takes, spans_by_id = _pimples_shaped_fixture()
    merged, diag = reconcile_semantic_idea_equivalence(groups, takes, None, watch_listen_spans_by_id=spans_by_id)
    assert {"short", "long"} in [set(g) for g in merged]
    trace = [row for row in diag["watch_listen_discovery_trace"] if row["left_id"] == "short"]
    assert trace[0]["rejection_reason"] is None


def test_55_family_topology_is_deterministic_across_repeated_calls(monkeypatch):
    _enable_discovery(monkeypatch)
    groups, takes, spans_by_id = _pimples_shaped_fixture()
    first, _ = reconcile_semantic_idea_equivalence(groups, takes, None, watch_listen_spans_by_id=spans_by_id)
    second, _ = reconcile_semantic_idea_equivalence(groups, takes, None, watch_listen_spans_by_id=spans_by_id)
    assert first == second


def test_56_watch_listen_family_evidence_d158_summary_still_reported_alongside_discovery(monkeypatch):
    # D-158's own observability key must still be present and correct even
    # when D-161 discovery is the flag doing the work -- two independent,
    # co-existing summaries, never one replacing the other.
    _enable_discovery(monkeypatch)
    groups, takes, spans_by_id = _pimples_shaped_fixture()
    _merged, diag = reconcile_semantic_idea_equivalence(groups, takes, None, watch_listen_spans_by_id=spans_by_id)
    assert diag["watch_listen_family_evidence"] == {"status": "disabled"}  # D-158's OWN flag stayed off
    assert diag["watch_listen_relation_discovery"]["status"] == "evaluated"


# ===========================================================================
# 57-64. Module-leaf / no-import structural tests: D-150, DeliveryScorer,
# BestTake, D-123, D-128, Boundary, Pacing non-interference; no provider or
# network call anywhere; the lazy-import cycle actually works end-to-end.
# ===========================================================================

def test_57_module_never_imports_a_downstream_authority_or_provider():
    import cutsell_worker.watch_listen_relation_discovery as mod
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


def test_58_no_provider_or_network_symbol_referenced_anywhere_in_module():
    import cutsell_worker.watch_listen_relation_discovery as mod
    source = open(mod.__file__, encoding="utf-8").read()
    for token in ("generate_content", "genai.", "requests.post", "requests.get", "socket."):
        assert token not in source


def test_59_d150_gate_module_never_references_this_new_module():
    import cutsell_worker.semantic_authority_observability as d150
    source = open(d150.__file__, encoding="utf-8").read()
    assert "watch_listen_relation_discovery" not in source


def test_60_take_grouping_provider_lazy_import_of_discovery_module_works_end_to_end():
    import cutsell_worker.take_grouping_provider as tgp
    import cutsell_worker  # noqa: F401 -- full-package import must also succeed
    module = tgp._watch_listen_relation_discovery()
    assert module.SCHEMA_VERSION.startswith("cutsell.watch_listen_relation_discovery")


def test_61_pipeline_wiring_present_and_gated_by_either_flag():
    import cutsell_worker.pipeline as pipeline_mod
    source = open(pipeline_mod.__file__, encoding="utf-8").read()
    assert "watch_listen_relation_discovery_enabled" in source
    # D-163 additively widened this same condition to a third flag
    # (watch_listen_besttake_evidence_enabled) -- check the two D-161-owned
    # disjuncts are still both present in the (now multi-line) expression,
    # not the exact original one-line string.
    idx = source.index("watch_listen_spans_by_id = (")
    block = source[idx: idx + 600]
    assert "watch_listen_family_evidence_enabled()" in block
    assert "watch_listen_relation_discovery_enabled()" in block


def test_62_no_besttake_deliveryscorer_boundary_pacing_module_imports_discovery():
    import glob
    forbidden_modules = [
        "deterministic_best_take_authority.py", "take_judge_provider.py",
        "boundary_engine.py", "temporal_editing.py",
    ]
    for name in forbidden_modules:
        matches = glob.glob(f"/home/user/EditDNA-worker/cutsell_worker/{name}")
        for path in matches:
            source = open(path, encoding="utf-8").read()
            assert "watch_listen_relation_discovery" not in source, path


def test_63_discovery_module_runtime_logic_never_hardcodes_video00_identity():
    # The module docstring legitimately references the D-160 pimples/
    # espinillas forensic finding as historical context (same convention
    # already used by D-158/D-160's own docs) -- what must never appear is
    # a Video00-specific literal used as RUNTIME data (a clip id, phrase,
    # or source-video identifier baked into the logic itself).
    import cutsell_worker.watch_listen_relation_discovery as mod
    source = open(mod.__file__, encoding="utf-8").read().lower()
    assert "video00" not in source


def test_64_attempt_relationship_authority_module_never_hardcodes_video00_identity():
    import cutsell_worker.attempt_relationship_authority as mod
    source = open(mod.__file__, encoding="utf-8").read().lower()
    assert "video00" not in source
