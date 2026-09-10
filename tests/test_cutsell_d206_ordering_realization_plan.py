"""D-206: Ordering Consolidation -- Phase A typed evidence layer offline
tests.

Covers: OrderingUnit/OrderingRelationEvidence/OrderedRealizationPlan
construction, the deterministic topological/fallback plan builder, the
reorder-only invariant, P1 local-sequence/continuation/correction/
composite-internal-order preservation, P2 supersession-survival and
meaning-conflict handling, cycle detection, multi-source behavior,
determinism/input-order-independence, and structural no-authority/no-
second-engine/no-provider audits (D-205 consolidation replay).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.editorial_moment_sequence import (
    CONFIDENCE_SUPPORTED,
    MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
    MOMENT_ROLE_CONTINUATION,
    MOMENT_ROLE_CORRECTION,
    AUDIENCE_DELIVERY_SUPPORTED,
    RECORDING_PROCESS_ABSENT,
    EditorialMoment,
)
from cutsell_worker.editorial_moment_sequence_integration import EditorialLocalGroup
from cutsell_worker.whole_video_editorial_reasoning import (
    SUPERSESSION_CONFLICTED,
    SUPERSESSION_PARTIAL,
    SUPERSESSION_SUPPORTED,
    WholeVideoSupersessionHypothesis,
)
from cutsell_worker import ordering_realization_plan as ordp

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_SOURCE = (REPO_ROOT / "cutsell_worker" / "ordering_realization_plan.py").read_text()


def _code_only(source: str) -> str:
    marker = '"""\nfrom __future__'
    idx = source.find(marker)
    return source[idx + len(marker):] if idx != -1 else source


CODE_ONLY = _code_only(MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Fixture helpers.
# ---------------------------------------------------------------------------
def _take(clip_id, source_asset_id, source_order, start, end, *, realization_id=None, source_span_id=None):
    return CandidateTake(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=source_order,
        start=start, end=end, text="x",
        signals=MediaSignals(source_asset_id=source_asset_id, start=start, end=end),
        realization_id=realization_id, source_span_id=source_span_id or clip_id,
    )


def _moment(source_asset_id, seed, start, end, role, *, proposition_ids=(), source_span_id=None):
    return EditorialMoment(
        source_asset_id=source_asset_id, editorial_moment_id=f"emom_{seed}", source_start=start, source_end=end,
        source_span_id=source_span_id or seed, attempt_ids=(f"att_{seed}",),
        proposition_candidate_ids=tuple(proposition_ids), related_span_ids=(), moment_role=role,
        audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED, recording_process_status=RECORDING_PROCESS_ABSENT,
        completion_status="COMPLETE", local_sequence_position=None, confidence=CONFIDENCE_SUPPORTED,
        conflict_flags=(), provenance=("TEST_FIXTURE",),
    )


def _local_group(source_asset_id, seed, moment_ids):
    return EditorialLocalGroup(
        source_asset_id=source_asset_id, group_id=f"elgrp_{seed}", moment_indices=tuple(range(len(moment_ids))),
        moment_ids=tuple(moment_ids), source_start=0.0, source_end=1.0,
        grouping_reason="RELATION_LINKED_CHAIN", relation_support=(), confidence=CONFIDENCE_SUPPORTED,
        conflict_flags=(), provenance=("TEST_FIXTURE",),
    )


def _hyp(seed, earlier_region_ids, later_region_ids, status, *, uncovered=(), meaning_conflict="MEANING_CONFLICT_NONE"):
    return WholeVideoSupersessionHypothesis(
        source_asset_id="src1", supersession_id=f"sup_{seed}", earlier_region_ids=tuple(earlier_region_ids),
        later_region_ids=tuple(later_region_ids), covered_proposition_candidate_ids=(),
        uncovered_earlier_proposition_candidate_ids=tuple(uncovered), coverage_status="FULL_COVERAGE",
        recording_process_support="SUPPORTED", audience_delivery_support="SUPPORTED",
        meaning_conflict_status=meaning_conflict, supersession_status=status, confidence=CONFIDENCE_SUPPORTED,
        conflict_flags=(), provenance=("TEST_FIXTURE",),
    )


# ---------------------------------------------------------------------------
# 1. OrderingUnit construction.
# ---------------------------------------------------------------------------
class TestOrderingUnitConstruction:
    def test_one_unit(self):
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0)])
        assert len(units) == 1
        assert units[0].realization_id == "a"

    def test_realization_id_preferred_over_clip_id(self):
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0, realization_id="real_a")])
        assert units[0].realization_id == "real_a"

    def test_two_units_source_order(self):
        units = ordp.build_ordering_units(realizations=[
            _take("b", "src1", 1, 5.0, 6.0), _take("a", "src1", 0, 0.0, 1.0),
        ])
        assert [u.realization_id for u in units] == ["a", "b"]

    def test_p1_evidence_reference_by_id(self):
        m = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, proposition_ids=("prop_1",))
        g = _local_group("src1", "g1", [m.editorial_moment_id])
        units = ordp.build_ordering_units(
            realizations=[_take("a", "src1", 0, 0.0, 1.0, source_span_id="s1")],
            moments_by_source={"src1": [m]}, local_groups_by_source={"src1": [g]},
        )
        assert units[0].proposition_candidate_ids == ("prop_1",)
        assert units[0].p1_local_group_id == "elgrp_g1"

    def test_composite_component_ids_default_empty(self):
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0)])
        assert units[0].composite_component_ids == ()

    def test_composite_component_ids_caller_supplied(self):
        units = ordp.build_ordering_units(
            realizations=[_take("a", "src1", 0, 0.0, 1.0)],
            composite_component_ids_by_realization={"a": ("b", "c")},
        )
        assert units[0].composite_component_ids == ("b", "c")


# ---------------------------------------------------------------------------
# 2. Deterministic plan: baseline / fallback / constraints.
# ---------------------------------------------------------------------------
class TestDeterministicPlan:
    def test_no_relations_source_fallback_unknown(self):
        units = ordp.build_ordering_units(realizations=[
            _take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 5.0, 6.0),
        ])
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=())
        assert plan.ordered_realization_ids == ("a", "b")
        assert plan.ordering_status == ordp.ORDERING_STATUS_UNKNOWN

    def test_explicit_must_precede(self):
        units = ordp.build_ordering_units(realizations=[
            _take("b", "src1", 1, 5.0, 6.0), _take("a", "src1", 0, 0.0, 1.0),
        ])
        rel = ordp.OrderingRelationEvidence(
            relation_id="r1", left_realization_id="b", right_realization_id="a",
            ordering_relation=ordp.RELATION_MUST_PRECEDE, ordering_reason=ordp.REASON_P2_PROPOSITION_PROGRESSION,
            confidence=CONFIDENCE_SUPPORTED, conflict_flags=(), provenance=(),
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=[rel])
        assert plan.ordered_realization_ids == ("b", "a")
        assert plan.ordering_status == ordp.ORDERING_STATUS_ORDERED

    def test_explicit_must_follow_equivalent_to_reverse_precede(self):
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 5.0, 6.0)])
        rel = ordp.OrderingRelationEvidence(
            relation_id="r1", left_realization_id="a", right_realization_id="b",
            ordering_relation=ordp.RELATION_MUST_FOLLOW, ordering_reason=ordp.REASON_P2_PROPOSITION_PROGRESSION,
            confidence=CONFIDENCE_SUPPORTED, conflict_flags=(), provenance=(),
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=[rel])
        assert plan.ordered_realization_ids == ("b", "a")

    def test_cycle_detected_and_conflicted(self):
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 5.0, 6.0)])
        r1 = ordp.OrderingRelationEvidence("r1", "a", "b", ordp.RELATION_MUST_PRECEDE, ordp.REASON_P2_PROPOSITION_PROGRESSION, CONFIDENCE_SUPPORTED, (), ())
        r2 = ordp.OrderingRelationEvidence("r2", "b", "a", ordp.RELATION_MUST_PRECEDE, ordp.REASON_P2_PROPOSITION_PROGRESSION, CONFIDENCE_SUPPORTED, (), ())
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=[r1, r2])
        assert plan.ordering_status == ordp.ORDERING_STATUS_CONFLICTED
        assert plan.unresolved_relation_ids
        assert len(plan.ordered_realization_ids) == 2  # still a total, deterministic list

    def test_input_order_independence(self):
        u1 = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 5.0, 6.0)])
        u2 = ordp.build_ordering_units(realizations=[_take("b", "src1", 1, 5.0, 6.0), _take("a", "src1", 0, 0.0, 1.0)])
        assert u1 == u2
        p1 = ordp.build_deterministic_ordering_plan(units=u1, relations=())
        p2 = ordp.build_deterministic_ordering_plan(units=u2, relations=())
        assert p1 == p2

    def test_deterministic_repeat(self):
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 5.0, 6.0)])
        p1 = ordp.build_deterministic_ordering_plan(units=units, relations=())
        p2 = ordp.build_deterministic_ordering_plan(units=units, relations=())
        assert p1 == p2

    def test_partial_order_some_unknown(self):
        units = ordp.build_ordering_units(realizations=[
            _take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 5.0, 6.0), _take("c", "src1", 2, 10.0, 11.0),
        ])
        rel = ordp.OrderingRelationEvidence("r1", "a", "b", ordp.RELATION_MUST_PRECEDE, ordp.REASON_P2_PROPOSITION_PROGRESSION, CONFIDENCE_SUPPORTED, (), ())
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=[rel])
        assert plan.ordering_status == ordp.ORDERING_STATUS_PARTIALLY_ORDERED
        assert plan.fallback_used is True

    def test_chronology_alone_never_creates_semantic_certainty(self):
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 5.0, 6.0)])
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=())
        assert plan.ordering_status != ordp.ORDERING_STATUS_ORDERED
        assert all(u.continuity_status != ordp.CONTINUITY_CONSTRAINT_SUPPORTED for u in plan.ordered_units)


# ---------------------------------------------------------------------------
# 3. Reorder-only invariant / no add-drop-duplicate.
# ---------------------------------------------------------------------------
class TestReorderOnlyInvariant:
    def test_valid_plan_passes(self):
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 5.0, 6.0)])
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=())
        assert ordp.validate_reorder_only_invariant(units, plan) is True

    def test_dropped_unit_fails(self):
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 5.0, 6.0)])
        bad_plan = ordp.OrderedRealizationPlan(
            source_asset_ids=("src1",), ordered_realization_ids=("a",), ordered_units=(),
            ordering_status=ordp.ORDERING_STATUS_UNKNOWN, fallback_used=True, fallback_reason="",
            unresolved_relation_ids=(), conflict_flags=(), confidence="UNKNOWN", provenance=(),
        )
        assert ordp.validate_reorder_only_invariant(units, bad_plan) is False

    def test_duplicated_unit_fails(self):
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0)])
        bad_plan = ordp.OrderedRealizationPlan(
            source_asset_ids=("src1",), ordered_realization_ids=("a", "a"), ordered_units=(),
            ordering_status=ordp.ORDERING_STATUS_UNKNOWN, fallback_used=True, fallback_reason="",
            unresolved_relation_ids=(), conflict_flags=(), confidence="UNKNOWN", provenance=(),
        )
        assert ordp.validate_reorder_only_invariant(units, bad_plan) is False

    def test_added_unit_fails(self):
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0)])
        bad_plan = ordp.OrderedRealizationPlan(
            source_asset_ids=("src1",), ordered_realization_ids=("a", "phantom"), ordered_units=(),
            ordering_status=ordp.ORDERING_STATUS_UNKNOWN, fallback_used=True, fallback_reason="",
            unresolved_relation_ids=(), conflict_flags=(), confidence="UNKNOWN", provenance=(),
        )
        assert ordp.validate_reorder_only_invariant(units, bad_plan) is False

    def test_real_builder_never_drops_across_many_fixtures(self):
        realizations = [_take(f"c{i}", "src1", i, float(i), float(i) + 1.0) for i in range(15)]
        units = ordp.build_ordering_units(realizations=realizations)
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=())
        assert ordp.validate_reorder_only_invariant(units, plan) is True


# ---------------------------------------------------------------------------
# 4. P1 local-sequence / continuation / correction preservation.
# ---------------------------------------------------------------------------
class TestP1Constraints:
    def test_local_sequence_order_preserved_no_relation_role(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 2.0, 3.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s2")
        g = _local_group("src1", "g1", [m1.editorial_moment_id, m2.editorial_moment_id])
        units = ordp.build_ordering_units(
            realizations=[_take("b", "src1", 1, 2.0, 3.0, source_span_id="s2"), _take("a", "src1", 0, 0.0, 1.0, source_span_id="s1")],
            moments_by_source={"src1": [m1, m2]}, local_groups_by_source={"src1": [g]},
        )
        relations = ordp.build_ordering_relation_evidence(
            units=units, moments_by_id={m1.editorial_moment_id: m1, m2.editorial_moment_id: m2},
            local_groups_by_id={g.group_id: g},
        )
        assert any(r.ordering_relation == ordp.RELATION_PRESERVE_INTERNAL_ORDER and r.left_realization_id == "a" and r.right_realization_id == "b" for r in relations)
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        assert plan.ordered_realization_ids == ("a", "b")
        assert plan.ordering_status == ordp.ORDERING_STATUS_ORDERED

    def test_continuation_creates_must_precede(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 2.0, 3.0, MOMENT_ROLE_CONTINUATION, source_span_id="s2")
        g = _local_group("src1", "g1", [m1.editorial_moment_id, m2.editorial_moment_id])
        units = ordp.build_ordering_units(
            realizations=[_take("a", "src1", 0, 0.0, 1.0, source_span_id="s1"), _take("b", "src1", 1, 2.0, 3.0, source_span_id="s2")],
            moments_by_source={"src1": [m1, m2]}, local_groups_by_source={"src1": [g]},
        )
        relations = ordp.build_ordering_relation_evidence(
            units=units, moments_by_id={m1.editorial_moment_id: m1, m2.editorial_moment_id: m2},
            local_groups_by_id={g.group_id: g},
        )
        rel = next(r for r in relations if r.ordering_reason == ordp.REASON_P1_CONTINUATION)
        assert rel.ordering_relation == ordp.RELATION_MUST_PRECEDE
        assert (rel.left_realization_id, rel.right_realization_id) == ("a", "b")

    def test_correction_creates_must_precede(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 2.0, 3.0, MOMENT_ROLE_CORRECTION, source_span_id="s2")
        g = _local_group("src1", "g1", [m1.editorial_moment_id, m2.editorial_moment_id])
        units = ordp.build_ordering_units(
            realizations=[_take("a", "src1", 0, 0.0, 1.0, source_span_id="s1"), _take("b", "src1", 1, 2.0, 3.0, source_span_id="s2")],
            moments_by_source={"src1": [m1, m2]}, local_groups_by_source={"src1": [g]},
        )
        relations = ordp.build_ordering_relation_evidence(
            units=units, moments_by_id={m1.editorial_moment_id: m1, m2.editorial_moment_id: m2},
            local_groups_by_id={g.group_id: g},
        )
        rel = next(r for r in relations if r.ordering_reason == ordp.REASON_P1_CORRECTION)
        assert rel.ordering_relation == ordp.RELATION_MUST_PRECEDE
        # Neither the earlier nor the correcting realization is dropped.
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        assert set(plan.ordered_realization_ids) == {"a", "b"}

    def test_no_relation_when_only_one_side_survived(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 2.0, 3.0, MOMENT_ROLE_CONTINUATION, source_span_id="s2")
        g = _local_group("src1", "g1", [m1.editorial_moment_id, m2.editorial_moment_id])
        # Only "a" (s1) survived to Freeze -- s2's realization was discarded upstream.
        units = ordp.build_ordering_units(
            realizations=[_take("a", "src1", 0, 0.0, 1.0, source_span_id="s1")],
            moments_by_source={"src1": [m1, m2]}, local_groups_by_source={"src1": [g]},
        )
        relations = ordp.build_ordering_relation_evidence(
            units=units, moments_by_id={m1.editorial_moment_id: m1, m2.editorial_moment_id: m2},
            local_groups_by_id={g.group_id: g},
        )
        assert relations == ()


# ---------------------------------------------------------------------------
# 5. Composite internal order.
# ---------------------------------------------------------------------------
class TestCompositeInternalOrder:
    def test_two_piece_composite_internal_order_preserved(self):
        units = ordp.build_ordering_units(realizations=[
            _take("b", "src1", 1, 5.0, 6.0), _take("a", "src1", 0, 0.0, 1.0),
        ])
        relations = ordp.build_ordering_relation_evidence(
            units=units, composite_group_by_realization={"a": "comp1", "b": "comp1"},
        )
        rel = next(r for r in relations if r.ordering_reason == ordp.REASON_COMPOSITE_INTERNAL_ORDER)
        assert (rel.left_realization_id, rel.right_realization_id) == ("a", "b")
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        assert plan.ordered_realization_ids == ("a", "b")

    def test_three_piece_composite_internal_order_preserved(self):
        units = ordp.build_ordering_units(realizations=[
            _take("c", "src1", 2, 10.0, 11.0), _take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 5.0, 6.0),
        ])
        relations = ordp.build_ordering_relation_evidence(
            units=units, composite_group_by_realization={"a": "comp1", "b": "comp1", "c": "comp1"},
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        assert plan.ordered_realization_ids == ("a", "b", "c")

    def test_two_composites_globally_reordered_internals_unchanged(self):
        # Composite 1: a,b (early); Composite 2: c,d (later). A P2-style
        # relation could reorder the composites relative to each other,
        # but internal order within each must never scramble.
        units = ordp.build_ordering_units(realizations=[
            _take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 2.0, 3.0),
            _take("c", "src1", 2, 4.0, 5.0), _take("d", "src1", 3, 6.0, 7.0),
        ])
        relations = ordp.build_ordering_relation_evidence(
            units=units, composite_group_by_realization={"a": "comp1", "b": "comp1", "c": "comp2", "d": "comp2"},
        )
        swap = ordp.OrderingRelationEvidence(
            "swap", "c", "a", ordp.RELATION_MUST_PRECEDE, ordp.REASON_P2_GLOBAL_CONTINUITY,
            CONFIDENCE_SUPPORTED, (), (),
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=list(relations) + [swap])
        pos = {rid: i for i, rid in enumerate(plan.ordered_realization_ids)}
        assert pos["c"] < pos["d"]
        assert pos["a"] < pos["b"]
        assert pos["c"] < pos["a"]


# ---------------------------------------------------------------------------
# 6. P2 supersession / meaning-conflict handling (unique-info + meaning
# firewalls, retry-survival conflict).
# ---------------------------------------------------------------------------
class TestP2SupersessionHandling:
    def test_supported_supersession_both_survive_is_conflict_not_deletion(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 10.0, 11.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s2")
        units = ordp.build_ordering_units(
            realizations=[_take("a", "src1", 0, 0.0, 1.0, source_span_id="s1"), _take("b", "src1", 1, 10.0, 11.0, source_span_id="s2")],
            moments_by_source={"src1": [m1, m2]},
            region_ids_by_moment_id={m1.editorial_moment_id: ("r_early",), m2.editorial_moment_id: ("r_late",)},
        )
        hyp = _hyp("1", ["r_early"], ["r_late"], SUPERSESSION_SUPPORTED)
        relations = ordp.build_ordering_relation_evidence(
            units=units, supersession_hypotheses=[hyp],
            region_ids_by_unit={"a": ("r_early",), "b": ("r_late",)},
        )
        conflict = next(r for r in relations if r.ordering_relation == ordp.RELATION_CONFLICTED)
        assert "SUPERSESSION_SURVIVAL_CONFLICT" in conflict.conflict_flags
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        # No unit is dropped -- Ordering has no delete authority.
        assert set(plan.ordered_realization_ids) == {"a", "b"}
        assert plan.ordering_status == ordp.ORDERING_STATUS_CONFLICTED

    def test_partial_supersession_preserves_unique_info(self):
        hyp = _hyp("2", ["r_early"], ["r_late"], SUPERSESSION_PARTIAL, uncovered=("prop_b",))
        relations = ordp.build_ordering_relation_evidence(
            units=ordp.build_ordering_units(realizations=[
                _take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 10.0, 11.0),
            ]),
            supersession_hypotheses=[hyp], region_ids_by_unit={"a": ("r_early",), "b": ("r_late",)},
        )
        assert any(r.ordering_relation == ordp.RELATION_CONFLICTED for r in relations)
        # The hypothesis's own uncovered proposition is preserved on the
        # hypothesis object itself -- Ordering never drops the unit that
        # carries it (proven by the reorder-only invariant elsewhere).
        assert hyp.uncovered_earlier_proposition_candidate_ids == ("prop_b",)

    def test_meaning_conflict_flagged(self):
        hyp = _hyp("3", ["r_early"], ["r_late"], SUPERSESSION_CONFLICTED, meaning_conflict="MEANING_CONFLICT_PRESENT")
        relations = ordp.build_ordering_relation_evidence(
            units=ordp.build_ordering_units(realizations=[
                _take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 10.0, 11.0),
            ]),
            supersession_hypotheses=[hyp], region_ids_by_unit={"a": ("r_early",), "b": ("r_late",)},
        )
        rel = next(r for r in relations if r.ordering_reason == ordp.REASON_MEANING_FIREWALL)
        assert rel.ordering_relation == ordp.RELATION_CONFLICTED

    def test_no_hypothesis_no_relation(self):
        relations = ordp.build_ordering_relation_evidence(
            units=ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0)]),
            supersession_hypotheses=[],
        )
        assert relations == ()

    def test_retry_duplicate_survival_marked_conflicted_never_resolved(self):
        hyp = _hyp("4", ["r1"], ["r2"], SUPERSESSION_SUPPORTED)
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)])
        relations = ordp.build_ordering_relation_evidence(
            units=units, supersession_hypotheses=[hyp], region_ids_by_unit={"a": ("r1",), "b": ("r2",)},
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        assert plan.ordering_status == ordp.ORDERING_STATUS_CONFLICTED
        assert ordp.validate_reorder_only_invariant(units, plan)


# ---------------------------------------------------------------------------
# 7. Multi-source / no fabricated global timeline.
# ---------------------------------------------------------------------------
class TestMultiSource:
    def test_source_identity_preserved(self):
        units = ordp.build_ordering_units(realizations=[
            _take("a", "srcA", 0, 0.0, 1.0), _take("b", "srcB", 0, 0.0, 1.0),
        ])
        assert {u.source_asset_id for u in units} == {"srcA", "srcB"}

    def test_same_source_fallback(self):
        units = ordp.build_ordering_units(realizations=[
            _take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0),
        ])
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=())
        assert plan.ordering_status == ordp.ORDERING_STATUS_UNKNOWN
        assert plan.ordered_realization_ids == ("a", "b")

    def test_cross_source_deterministic_fallback_marked_uncertain(self):
        # source_order is the explicit cross-source rank (never raw
        # per-file timestamps compared as if on one timeline).
        units = ordp.build_ordering_units(realizations=[
            _take("b", "srcB", 1, 0.0, 1.0), _take("a", "srcA", 0, 100.0, 101.0),
        ])
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=())
        assert plan.ordered_realization_ids == ("a", "b")  # by source_order, never raw start
        assert plan.ordering_status == ordp.ORDERING_STATUS_UNKNOWN
        assert plan.source_asset_ids == ("srcA", "srcB")

    def test_multi_source_plan_total_list(self):
        units = ordp.build_ordering_units(realizations=[
            _take("a", "srcA", 0, 0.0, 1.0), _take("b", "srcB", 1, 0.0, 1.0), _take("c", "srcA", 2, 5.0, 6.0),
        ])
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=())
        assert len(plan.ordered_realization_ids) == 3


# ---------------------------------------------------------------------------
# 8. Diagnostics / stable ids.
# ---------------------------------------------------------------------------
class TestDiagnosticsAndIds:
    def test_relation_id_deterministic_and_direction_sensitive(self):
        id1 = ordp._relation_id("a", "b", ordp.RELATION_MUST_PRECEDE, ordp.REASON_P1_CONTINUATION)
        id2 = ordp._relation_id("a", "b", ordp.RELATION_MUST_PRECEDE, ordp.REASON_P1_CONTINUATION)
        id3 = ordp._relation_id("b", "a", ordp.RELATION_MUST_PRECEDE, ordp.REASON_P1_CONTINUATION)
        assert id1 == id2
        assert id1 != id3

    def test_no_random_uuid_in_source(self):
        assert "uuid" not in CODE_ONLY.lower()

    def test_unit_diagnostics_bounded(self):
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0)])
        diag = ordp.ordering_unit_diagnostics(units[0])
        assert "transcript" not in diag and "text" not in diag

    def test_plan_diagnostics_and_run_summary(self):
        units = ordp.build_ordering_units(realizations=[_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)])
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=())
        diag = ordp.ordered_realization_plan_diagnostics(plan)
        summary = ordp.ordered_realization_plan_run_summary(plan)
        assert diag["ordering_status"] == plan.ordering_status
        assert summary["unit_count"] == 2
        assert all(not isinstance(v, float) for v in summary.values())


# ---------------------------------------------------------------------------
# 9. Structural no-authority / no-second-engine / no-provider audits
# (D-205 consolidation replay).
# ---------------------------------------------------------------------------
class TestStructuralAudits:
    def test_no_provider_client_or_llm_prompt(self):
        for banned in ("openai", "gemini", "responses.create", "OpenAI(", "hook", "cta_score", "narrative_quality"):
            assert banned not in CODE_ONLY, banned

    def test_no_composer_import(self):
        # The module docstring DISCUSSES composer_openai/composer_provider
        # (D-205's own consolidation finding) without ever importing or
        # calling either -- check real import/call statements, not the
        # documentation mentioning why they are deliberately untouched.
        for banned in (
            "from .composer_openai import", "from .composer_provider import", "from .composer import",
            "import composer_openai", "import composer_provider",
            "OpenAIComposerProvider(", "safe_compose_order(", "compose_selected(",
        ):
            assert banned not in MODULE_SOURCE

    def test_no_causal_validator_import(self):
        for banned in (
            "from .causal_order_validator import", "import causal_order_validator",
            "find_causal_order_breaks(",
        ):
            assert banned not in MODULE_SOURCE

    def test_causal_order_validator_reason_declared_never_emitted(self):
        assert "CAUSAL_ORDER_VALIDATOR" in MODULE_SOURCE
        # Never appears as an actually-assigned reason inside a builder body.
        assert "REASON_CAUSAL_ORDER_VALIDATOR)" not in CODE_ONLY.replace(
            'REASON_CAUSAL_ORDER_VALIDATOR = "CAUSAL_ORDER_VALIDATOR"', ""
        )

    def test_no_family_besttake_boundary_pacing_renderer_reference(self):
        forbidden = (
            "take_group_id", "_semantic_best_take", "bounded_finalist_authority",
            "bounded_finalist_arbiter", "boundary_engine_pass", "BoundaryEngine", "dialogue_pacing_transition",
            "render_plan", "RenderSegment", "take_grouping", "composite_resolver", "realization_resolver",
        )
        for name in forbidden:
            assert name not in CODE_ONLY
        # selected_clip_id is checked precisely (real field/assignment,
        # never a docstring mention of what this module deliberately
        # lacks) by test_no_delete_winner_or_selected_clip_field below.
        assert "selected_clip_id =" not in CODE_ONLY
        assert "selected_clip_id:" not in CODE_ONLY

    def test_no_delete_winner_or_selected_clip_field(self):
        from dataclasses import fields
        for cls in (ordp.OrderingUnit, ordp.OrderingRelationEvidence, ordp.OrderedRealizationPlan, ordp.OrderedUnitPlacement):
            for f in fields(cls):
                for banned in ("delete", "winner", "selected_clip", "final_winner", "action"):
                    assert banned not in f.name.lower()

    def test_no_master_score_field(self):
        from dataclasses import fields
        for cls in (ordp.OrderingUnit, ordp.OrderingRelationEvidence, ordp.OrderedRealizationPlan):
            for f in fields(cls):
                assert f.name not in ("score", "master_score", "quality_score")

    def test_no_commercial_or_funnel_fields(self):
        for banned in ("commercial", "sales_funnel", "funnel", "cta_score", "hook_strength"):
            assert banned not in CODE_ONLY.lower()

    def test_no_qa_reference(self):
        for banned in ("cut_ai", "cutai", "human_gold", "quality_ladder", "benchmark_label"):
            assert banned not in CODE_ONLY.lower()

    def test_no_pipeline_wiring_files_touched(self):
        for path in ("cutsell_worker/pipeline.py", "cutsell_worker/universal_clean_cut.py", "cutsell_worker/brain_runtime.py"):
            content = (REPO_ROOT / path).read_text()
            assert "ordering_realization_plan" not in content

    def test_reorder_only_invariant_matches_composer_repair_contract_shape(self):
        # Mirrors composer_provider._repair_order's own invariant (set
        # equality + no duplicates) without importing that module.
        units = ordp.build_ordering_units(realizations=[_take(f"c{i}", "src1", i, float(i), float(i) + 1.0) for i in range(5)])
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=())
        assert set(u.realization_id for u in units) == set(plan.ordered_realization_ids)


# ---------------------------------------------------------------------------
# 10. Compileall / import sanity.
# ---------------------------------------------------------------------------
def test_module_compiles_and_imports():
    import cutsell_worker.ordering_realization_plan  # noqa: F401
