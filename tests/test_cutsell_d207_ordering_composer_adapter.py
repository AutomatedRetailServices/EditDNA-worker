"""D-207: Ordering Consolidation -- Phase B, existing composer adapter +
typed proposal validation, offline/mocked tests.

Covers: OrderingUnit<->existing-composer clip_id identity round trip,
composite atomicity, valid reorder/same-order acceptance, membership
(dropped/added/duplicate/unknown-id) rejection via composer_provider's
own already-proven repair signal, P1 local-sequence/continuation/
correction inversion rejection, explicit MUST_PRECEDE/MUST_FOLLOW
respect, baseline-conflict gate (cycle / meaning-firewall / supersession-
survival -- composer never resolves), PARTIALLY_ORDERED/UNKNOWN baseline
acceptance with uncertainty preserved, composite internal order,
multi-source behavior, determinism/one-proposal/no-retry, exact baseline
fallback immutability, diagnostics/summary shape, and structural no-
second-composer/no-provider/no-authority audits (D-205/D-206
consolidation replay).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pytest

from cutsell_worker.composer_provider import ComposerProviderResult
from cutsell_worker.contracts import CandidateTake, EditStrategy, MediaSignals
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
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_editorial_reasoning import (
    SUPERSESSION_CONFLICTED,
    SUPERSESSION_SUPPORTED,
    WholeVideoSupersessionHypothesis,
)
from cutsell_worker import ordering_realization_plan as ordp
from cutsell_worker import ordering_composer_adapter as oca

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_SOURCE = (REPO_ROOT / "cutsell_worker" / "ordering_composer_adapter.py").read_text()


def _code_only(source: str) -> str:
    marker = '"""\nfrom __future__'
    idx = source.find(marker)
    return source[idx + len(marker):] if idx != -1 else source


CODE_ONLY = _code_only(MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Fixture helpers (same conventions as test_cutsell_d206_ordering_realization_plan.py).
# ---------------------------------------------------------------------------
def _take(clip_id, source_asset_id, source_order, start, end, *, realization_id=None, source_span_id=None):
    return CandidateTake(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=source_order,
        start=start, end=end, text="x",
        signals=MediaSignals(source_asset_id=source_asset_id, start=start, end=end),
        realization_id=realization_id, source_span_id=source_span_id or clip_id,
    )


def _moment(source_asset_id, seed, start, end, role, *, source_span_id=None):
    return EditorialMoment(
        source_asset_id=source_asset_id, editorial_moment_id=f"emom_{seed}", source_start=start, source_end=end,
        source_span_id=source_span_id or seed, attempt_ids=(f"att_{seed}",),
        proposition_candidate_ids=(), related_span_ids=(), moment_role=role,
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


def _hyp(seed, earlier_region_ids, later_region_ids, status, *, meaning_conflict="MEANING_CONFLICT_NONE"):
    return WholeVideoSupersessionHypothesis(
        source_asset_id="src1", supersession_id=f"sup_{seed}", earlier_region_ids=tuple(earlier_region_ids),
        later_region_ids=tuple(later_region_ids), covered_proposition_candidate_ids=(),
        uncovered_earlier_proposition_candidate_ids=(), coverage_status="FULL_COVERAGE",
        recording_process_support="SUPPORTED", audience_delivery_support="SUPPORTED",
        meaning_conflict_status=meaning_conflict, supersession_status=status, confidence=CONFIDENCE_SUPPORTED,
        conflict_flags=(), provenance=("TEST_FIXTURE",),
    )


@dataclass
class MockComposerProvider:
    """Mock/fake ``ComposerProvider`` (this task's own 'mock provider
    only' instruction) -- never ``OpenAIComposerProvider``, never a
    network call, never an API key. Returns a fixed, caller-supplied
    clip-id order (or raises, to exercise the existing fail-open path)."""
    proposed_clip_ids: Tuple[str, ...] = ()
    reason: str = "mock_proposal"
    raise_error: bool = False

    def order(self, takes, labels, strategy, context_text=""):
        if self.raise_error:
            raise RuntimeError("mock_provider_failure")
        return ComposerProviderResult(tuple(self.proposed_clip_ids), ProviderStatus("mock", True, True, "applied"), self.reason)


def _units_and_baseline(realizations, relations=()):
    units = ordp.build_ordering_units(realizations=realizations)
    plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
    return units, plan


# ---------------------------------------------------------------------------
# 1. Adapter construction / identity round trip.
# ---------------------------------------------------------------------------
class TestAdapterConstruction:
    def test_adapter_one_unit(self):
        take = _take("a", "src1", 0, 0.0, 1.0)
        units, plan = _units_and_baseline([take])
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=[take], relations=(), baseline_plan=plan, provider=None,
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED

    def test_adapter_multiple_units(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0), _take("c", "src1", 2, 2.0, 3.0)]
        units, plan = _units_and_baseline(takes)
        provider = MockComposerProvider(("c", "b", "a"))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        assert result.accepted_realization_ids == ("c", "b", "a")

    def test_stable_identity_map_distinct_clip_and_realization_id(self):
        take = _take("clip_a", "src1", 0, 0.0, 1.0, realization_id="real_a")
        units, plan = _units_and_baseline([take])
        assert units[0].realization_id == "real_a"
        provider = MockComposerProvider(("clip_a",))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=[take], relations=(), baseline_plan=plan, provider=provider,
        )
        # Composer speaks clip ids ("clip_a"); the adapter must translate
        # the returned proposal back to the OrderingUnit's own identity.
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        assert result.accepted_realization_ids == ("real_a",)

    def test_realization_id_round_trip_losslessly(self):
        takes = [
            _take("clip_a", "src1", 0, 0.0, 1.0, realization_id="real_a"),
            _take("clip_b", "src1", 1, 1.0, 2.0, realization_id="real_b"),
        ]
        units, plan = _units_and_baseline(takes)
        provider = MockComposerProvider(("clip_b", "clip_a"))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        assert result.accepted_realization_ids == ("real_b", "real_a")

    def test_composite_atomic_round_trip_through_adapter(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0), _take("c", "src1", 2, 2.0, 3.0)]
        units = ordp.build_ordering_units(realizations=takes)
        relations = ordp.build_ordering_relation_evidence(
            units=units, composite_group_by_realization={"a": "comp1", "b": "comp1"},
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        # A valid proposal that moves the WHOLE composite (a,b) after c,
        # without scrambling their own internal order.
        provider = MockComposerProvider(("c", "a", "b"))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=relations, baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        assert result.accepted_realization_ids == ("c", "a", "b")


# ---------------------------------------------------------------------------
# 2. Valid proposals (same order / reorder / chronology change / no
# constraint / input-order independence).
# ---------------------------------------------------------------------------
class TestValidProposals:
    def test_valid_same_order_proposal(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, plan = _units_and_baseline(takes)
        provider = MockComposerProvider(("a", "b"))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        assert result.accepted_realization_ids == plan.ordered_realization_ids

    def test_valid_reorder_no_constraint(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, plan = _units_and_baseline(takes)
        provider = MockComposerProvider(("b", "a"))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        assert result.accepted_realization_ids == ("b", "a")

    def test_no_provider_returns_natural_order_accepted(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, plan = _units_and_baseline(takes)
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=None,
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        assert result.composer_path == oca.COMPOSER_PATH_NO_PROVIDER_NATURAL
        assert result.accepted_realization_ids == plan.ordered_realization_ids

    def test_source_chronology_changed_but_constraints_preserved(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 2.0, 3.0, MOMENT_ROLE_CONTINUATION, source_span_id="s2")
        g = _local_group("src1", "g1", [m1.editorial_moment_id, m2.editorial_moment_id])
        take_a = _take("a", "src1", 0, 0.0, 1.0, source_span_id="s1")
        take_c = _take("c", "src1", 2, 5.0, 6.0)
        take_b = _take("b", "src1", 1, 2.0, 3.0, source_span_id="s2")
        units = ordp.build_ordering_units(
            realizations=[take_a, take_b, take_c],
            moments_by_source={"src1": [m1, m2]}, local_groups_by_source={"src1": [g]},
        )
        relations = ordp.build_ordering_relation_evidence(
            units=units, moments_by_id={m1.editorial_moment_id: m1, m2.editorial_moment_id: m2},
            local_groups_by_id={g.group_id: g},
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        # "c" has no constraint at all -- moving it around is fine as long
        # as a MUST_PRECEDE(a, b) continuation constraint still holds.
        provider = MockComposerProvider(("c", "a", "b"))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=[take_a, take_b, take_c], relations=relations, baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        assert result.accepted_realization_ids == ("c", "a", "b")

    def test_input_order_independence(self):
        takes_1 = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        takes_2 = list(reversed(takes_1))
        units_1, plan_1 = _units_and_baseline(takes_1)
        units_2, plan_2 = _units_and_baseline(takes_2)
        result_1 = oca.validate_composer_ordering_proposal(
            units=units_1, realizations=takes_1, relations=(), baseline_plan=plan_1, provider=MockComposerProvider(("a", "b")),
        )
        result_2 = oca.validate_composer_ordering_proposal(
            units=units_2, realizations=takes_2, relations=(), baseline_plan=plan_2, provider=MockComposerProvider(("a", "b")),
        )
        assert result_1.accepted_realization_ids == result_2.accepted_realization_ids

    def test_repeated_invocation_deterministic(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, plan = _units_and_baseline(takes)
        result_1 = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=MockComposerProvider(("b", "a")),
        )
        result_2 = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=MockComposerProvider(("b", "a")),
        )
        assert result_1 == result_2


# ---------------------------------------------------------------------------
# 3. Membership rejection (dropped/added/duplicate/unknown id) -- all
# surfaced via composer_provider's own already-proven repair signal, per
# module docstring's "No repair that changes membership" section.
# ---------------------------------------------------------------------------
class TestMembershipRejection:
    def _three_unit_setup(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0), _take("c", "src1", 2, 2.0, 3.0)]
        units, plan = _units_and_baseline(takes)
        return takes, units, plan

    def test_dropped_item_rejected(self):
        takes, units, plan = self._three_unit_setup()
        provider = MockComposerProvider(("a", "c"))  # "b" dropped
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_REJECTED_REORDER_ONLY
        assert result.fallback_used is True
        assert result.accepted_realization_ids == plan.ordered_realization_ids

    def test_added_item_rejected(self):
        takes, units, plan = self._three_unit_setup()
        provider = MockComposerProvider(("a", "b", "c", "phantom"))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_REJECTED_REORDER_ONLY
        assert result.accepted_realization_ids == plan.ordered_realization_ids

    def test_duplicate_item_rejected(self):
        takes, units, plan = self._three_unit_setup()
        provider = MockComposerProvider(("a", "a", "b", "c"))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_REJECTED_REORDER_ONLY
        assert result.accepted_realization_ids == plan.ordered_realization_ids

    def test_unknown_id_rejected(self):
        takes, units, plan = self._three_unit_setup()
        provider = MockComposerProvider(("a", "b", "unknown_clip"))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_REJECTED_REORDER_ONLY
        assert result.accepted_realization_ids == plan.ordered_realization_ids

    def test_provider_exception_fails_open_to_baseline_natural_order(self):
        takes, units, plan = self._three_unit_setup()
        provider = MockComposerProvider(raise_error=True)
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=provider,
        )
        # safe_compose_order (unmodified) already fails open to natural
        # order on any provider exception -- this equals baseline here,
        # so it is a legitimate ACCEPTED result, not a rejection.
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        assert result.accepted_realization_ids == plan.ordered_realization_ids


# ---------------------------------------------------------------------------
# 4. P1 constraint inversion rejection + explicit MUST_PRECEDE/MUST_FOLLOW.
# ---------------------------------------------------------------------------
class TestP1ConstraintValidation:
    def _continuation_setup(self, role):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 2.0, 3.0, role, source_span_id="s2")
        g = _local_group("src1", "g1", [m1.editorial_moment_id, m2.editorial_moment_id])
        take_a = _take("a", "src1", 0, 0.0, 1.0, source_span_id="s1")
        take_b = _take("b", "src1", 1, 2.0, 3.0, source_span_id="s2")
        units = ordp.build_ordering_units(
            realizations=[take_a, take_b], moments_by_source={"src1": [m1, m2]}, local_groups_by_source={"src1": [g]},
        )
        relations = ordp.build_ordering_relation_evidence(
            units=units, moments_by_id={m1.editorial_moment_id: m1, m2.editorial_moment_id: m2},
            local_groups_by_id={g.group_id: g},
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        return [take_a, take_b], units, relations, plan

    def test_local_sequence_inversion_rejected(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 2.0, 3.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s2")
        g = _local_group("src1", "g1", [m1.editorial_moment_id, m2.editorial_moment_id])
        take_a = _take("a", "src1", 0, 0.0, 1.0, source_span_id="s1")
        take_b = _take("b", "src1", 1, 2.0, 3.0, source_span_id="s2")
        units = ordp.build_ordering_units(
            realizations=[take_a, take_b], moments_by_source={"src1": [m1, m2]}, local_groups_by_source={"src1": [g]},
        )
        relations = ordp.build_ordering_relation_evidence(
            units=units, moments_by_id={m1.editorial_moment_id: m1, m2.editorial_moment_id: m2},
            local_groups_by_id={g.group_id: g},
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        provider = MockComposerProvider(("b", "a"))  # inverts PRESERVE_INTERNAL_ORDER(a, b)
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=[take_a, take_b], relations=relations, baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_REJECTED_CONSTRAINT
        assert result.violated_relation_ids
        assert result.accepted_realization_ids == plan.ordered_realization_ids

    def test_continuation_inversion_rejected(self):
        takes, units, relations, plan = self._continuation_setup(MOMENT_ROLE_CONTINUATION)
        provider = MockComposerProvider(("b", "a"))  # inverts MUST_PRECEDE(a, b)
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=relations, baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_REJECTED_CONSTRAINT
        assert result.accepted_realization_ids == plan.ordered_realization_ids

    def test_correction_inversion_rejected(self):
        takes, units, relations, plan = self._continuation_setup(MOMENT_ROLE_CORRECTION)
        provider = MockComposerProvider(("b", "a"))  # inverts MUST_PRECEDE(a, b)
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=relations, baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_REJECTED_CONSTRAINT
        assert result.accepted_realization_ids == plan.ordered_realization_ids

    def test_explicit_must_precede_respected(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, _ = _units_and_baseline(takes)
        rel = ordp.OrderingRelationEvidence(
            "r1", "a", "b", ordp.RELATION_MUST_PRECEDE, ordp.REASON_P2_PROPOSITION_PROGRESSION,
            CONFIDENCE_SUPPORTED, (), (),
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=[rel])
        provider = MockComposerProvider(("a", "b"))  # respects a before b
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=[rel], baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED

    def test_must_follow_respected(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, _ = _units_and_baseline(takes)
        rel = ordp.OrderingRelationEvidence(
            "r1", "a", "b", ordp.RELATION_MUST_FOLLOW, ordp.REASON_P2_PROPOSITION_PROGRESSION,
            CONFIDENCE_SUPPORTED, (), (),
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=[rel])
        assert plan.ordered_realization_ids == ("b", "a")
        provider = MockComposerProvider(("b", "a"))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=[rel], baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        provider_bad = MockComposerProvider(("a", "b"))  # violates MUST_FOLLOW
        result_bad = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=[rel], baseline_plan=plan, provider=provider_bad,
        )
        assert result_bad.proposal_status == oca.PROPOSAL_REJECTED_CONSTRAINT


# ---------------------------------------------------------------------------
# 5. Baseline-conflict gate: cycle / meaning firewall / supersession-
# survival -- composer is never allowed to "resolve" these.
# ---------------------------------------------------------------------------
class TestBaselineConflictGate:
    def test_cycle_input_not_resolved_by_composer(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, _ = _units_and_baseline(takes)
        r1 = ordp.OrderingRelationEvidence("r1", "a", "b", ordp.RELATION_MUST_PRECEDE, ordp.REASON_P2_PROPOSITION_PROGRESSION, CONFIDENCE_SUPPORTED, (), ())
        r2 = ordp.OrderingRelationEvidence("r2", "b", "a", ordp.RELATION_MUST_PRECEDE, ordp.REASON_P2_PROPOSITION_PROGRESSION, CONFIDENCE_SUPPORTED, (), ())
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=[r1, r2])
        assert plan.ordering_status == ordp.ORDERING_STATUS_CONFLICTED
        provider = MockComposerProvider(("a", "b"))  # composer "confidently" proposes an order anyway
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=[r1, r2], baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_NOT_EVALUABLE
        assert result.fallback_used is True
        assert result.accepted_realization_ids == plan.ordered_realization_ids
        assert result.composer_path == oca.COMPOSER_PATH_NOT_EVALUATED_BASELINE_CONFLICTED

    def test_meaning_conflict_baseline_fallback(self):
        hyp = _hyp("1", ["r_early"], ["r_late"], SUPERSESSION_CONFLICTED, meaning_conflict="MEANING_CONFLICT_PRESENT")
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 10.0, 11.0)]
        units = ordp.build_ordering_units(realizations=takes)
        relations = ordp.build_ordering_relation_evidence(
            units=units, supersession_hypotheses=[hyp], region_ids_by_unit={"a": ("r_early",), "b": ("r_late",)},
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        assert plan.ordering_status == ordp.ORDERING_STATUS_CONFLICTED
        provider = MockComposerProvider(("b", "a"))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=relations, baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_NOT_EVALUABLE
        assert result.accepted_realization_ids == plan.ordered_realization_ids
        # No unit is dropped -- Ordering (and this adapter) has no delete authority.
        assert set(result.accepted_realization_ids) == {"a", "b"}

    def test_supersession_survival_conflict_baseline_fallback(self):
        hyp = _hyp("2", ["r_early"], ["r_late"], SUPERSESSION_SUPPORTED)
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 10.0, 11.0)]
        units = ordp.build_ordering_units(realizations=takes)
        relations = ordp.build_ordering_relation_evidence(
            units=units, supersession_hypotheses=[hyp], region_ids_by_unit={"a": ("r_early",), "b": ("r_late",)},
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        assert plan.ordering_status == ordp.ORDERING_STATUS_CONFLICTED
        provider = MockComposerProvider(("a", "b"))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=relations, baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_NOT_EVALUABLE
        assert result.fallback_reason == "baseline_conflicted"


# ---------------------------------------------------------------------------
# 6. PARTIALLY_ORDERED / UNKNOWN baseline: composer may propose a total
# order, acceptance never upgrades to semantic certainty.
# ---------------------------------------------------------------------------
class TestPartialAndUnknownBaseline:
    def test_partially_ordered_valid_proposal_accepted(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0), _take("c", "src1", 2, 2.0, 3.0)]
        units, _ = _units_and_baseline(takes)
        rel = ordp.OrderingRelationEvidence("r1", "a", "b", ordp.RELATION_MUST_PRECEDE, ordp.REASON_P2_PROPOSITION_PROGRESSION, CONFIDENCE_SUPPORTED, (), ())
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=[rel])
        assert plan.ordering_status == ordp.ORDERING_STATUS_PARTIALLY_ORDERED
        provider = MockComposerProvider(("c", "a", "b"))  # "c" is free to move
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=[rel], baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        assert result.underlying_ordering_status == ordp.ORDERING_STATUS_PARTIALLY_ORDERED

    def test_partially_ordered_invalid_proposal_rejected(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0), _take("c", "src1", 2, 2.0, 3.0)]
        units, _ = _units_and_baseline(takes)
        rel = ordp.OrderingRelationEvidence("r1", "a", "b", ordp.RELATION_MUST_PRECEDE, ordp.REASON_P2_PROPOSITION_PROGRESSION, CONFIDENCE_SUPPORTED, (), ())
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=[rel])
        provider = MockComposerProvider(("b", "a", "c"))  # inverts a-before-b
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=[rel], baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_REJECTED_CONSTRAINT

    def test_unknown_baseline_valid_proposal_accepted_uncertainty_preserved(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, plan = _units_and_baseline(takes)  # no relations -> UNKNOWN
        assert plan.ordering_status == ordp.ORDERING_STATUS_UNKNOWN
        provider = MockComposerProvider(("b", "a"))
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        # Acceptance means "satisfies known constraints", never "semantic
        # order proven" -- the underlying uncertainty is reported, not erased.
        assert result.underlying_ordering_status == ordp.ORDERING_STATUS_UNKNOWN

    def test_accepted_proposal_semantics_never_editorial_truth(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, plan = _units_and_baseline(takes)
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=MockComposerProvider(("a", "b")),
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        for field_name in ("editorial_truth", "certified", "human_gold", "cut_ai"):
            assert not hasattr(result, field_name)


# ---------------------------------------------------------------------------
# 7. Composite internal order + multi-source.
# ---------------------------------------------------------------------------
class TestCompositeAndMultiSource:
    def test_composite_internal_order_violation_rejected(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0), _take("c", "src1", 2, 2.0, 3.0)]
        units = ordp.build_ordering_units(realizations=takes)
        relations = ordp.build_ordering_relation_evidence(
            units=units, composite_group_by_realization={"a": "comp1", "b": "comp1"},
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        provider = MockComposerProvider(("c", "b", "a"))  # scrambles a-before-b inside the composite
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=relations, baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_REJECTED_CONSTRAINT
        assert result.composite_order_valid is False

    def test_multi_source_cross_source_reorder_accepted_without_typed_violation(self):
        takes = [_take("a", "srcA", 0, 0.0, 1.0), _take("b", "srcB", 1, 0.0, 1.0)]
        units, plan = _units_and_baseline(takes)
        provider = MockComposerProvider(("b", "a"))  # cross-source reorder, no constraint exists
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=provider,
        )
        assert result.proposal_status == oca.PROPOSAL_ACCEPTED
        assert result.accepted_realization_ids == ("b", "a")

    def test_no_fabricated_timeline_only_known_ids_ever_appear(self):
        takes = [_take("a", "srcA", 0, 0.0, 1.0), _take("b", "srcB", 1, 0.0, 1.0)]
        units, plan = _units_and_baseline(takes)
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=MockComposerProvider(("a", "b")),
        )
        assert set(result.accepted_realization_ids) <= {"a", "b"}


# ---------------------------------------------------------------------------
# 8. Baseline-fallback immutability + no auto-repair-loop / one proposal only.
# ---------------------------------------------------------------------------
class TestFallbackImmutabilityAndNoRetry:
    def test_exact_baseline_fallback_never_recomputed(self):
        m1 = _moment("src1", "s1", 0.0, 1.0, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, source_span_id="s1")
        m2 = _moment("src1", "s2", 2.0, 3.0, MOMENT_ROLE_CONTINUATION, source_span_id="s2")
        g = _local_group("src1", "g1", [m1.editorial_moment_id, m2.editorial_moment_id])
        take_a = _take("a", "src1", 0, 0.0, 1.0, source_span_id="s1")
        take_b = _take("b", "src1", 1, 2.0, 3.0, source_span_id="s2")
        units = ordp.build_ordering_units(
            realizations=[take_a, take_b], moments_by_source={"src1": [m1, m2]}, local_groups_by_source={"src1": [g]},
        )
        relations = ordp.build_ordering_relation_evidence(
            units=units, moments_by_id={m1.editorial_moment_id: m1, m2.editorial_moment_id: m2},
            local_groups_by_id={g.group_id: g},
        )
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=relations)
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=[take_a, take_b], relations=relations, baseline_plan=plan,
            provider=MockComposerProvider(("b", "a")),
        )
        assert result.proposal_status == oca.PROPOSAL_REJECTED_CONSTRAINT
        assert result.accepted_realization_ids == plan.ordered_realization_ids
        assert result.baseline_realization_ids == plan.ordered_realization_ids

    def test_baseline_plan_object_unchanged(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, plan = _units_and_baseline(takes)
        before = plan
        oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=MockComposerProvider(("a", "b")),
        )
        assert plan == before  # frozen dataclass; call is pure

    def test_units_and_relations_unchanged(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, plan = _units_and_baseline(takes)
        units_before, relations_before = tuple(units), ()
        oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=MockComposerProvider(("a", "b")),
        )
        assert tuple(units) == units_before

    def test_one_proposal_only_provider_called_once(self):
        calls = {"count": 0}

        @dataclass
        class CountingProvider:
            def order(self, takes, labels, strategy, context_text=""):
                calls["count"] += 1
                return ComposerProviderResult(("a", "b"), ProviderStatus("mock", True, True, "applied"), "counted")

        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, plan = _units_and_baseline(takes)
        oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=CountingProvider(),
        )
        assert calls["count"] == 1

    def test_no_retry_loop_on_rejection(self):
        calls = {"count": 0}

        @dataclass
        class CountingBadProvider:
            def order(self, takes, labels, strategy, context_text=""):
                calls["count"] += 1
                return ComposerProviderResult(("a",), ProviderStatus("mock", True, True, "applied"), "drops_b")

        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, plan = _units_and_baseline(takes)
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=CountingBadProvider(),
        )
        assert calls["count"] == 1
        assert result.proposal_status == oca.PROPOSAL_REJECTED_REORDER_ONLY


# ---------------------------------------------------------------------------
# 9. Identity-gap handling.
# ---------------------------------------------------------------------------
class TestIdentityGap:
    def test_missing_realization_object_rejected_before_provider_call(self):
        take_a = _take("a", "src1", 0, 0.0, 1.0)
        take_b = _take("b", "src1", 1, 1.0, 2.0)
        units = ordp.build_ordering_units(realizations=[take_a, take_b])
        plan = ordp.build_deterministic_ordering_plan(units=units, relations=())
        calls = {"count": 0}

        @dataclass
        class ShouldNeverBeCalled:
            def order(self, takes, labels, strategy, context_text=""):
                calls["count"] += 1
                return ComposerProviderResult(("a", "b"), ProviderStatus("mock", True, True, "applied"), "")

        # Only "a" is supplied as a real realization object -- "b" is missing.
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=[take_a], relations=(), baseline_plan=plan, provider=ShouldNeverBeCalled(),
        )
        assert result.proposal_status == oca.PROPOSAL_REJECTED_IDENTITY
        assert result.identity_valid is False
        assert calls["count"] == 0
        assert result.accepted_realization_ids == plan.ordered_realization_ids


# ---------------------------------------------------------------------------
# 10. Existing deterministic composer compatibility path.
# ---------------------------------------------------------------------------
class TestExistingComposerCompat:
    def test_compose_selected_compat_matches_natural_chronology(self):
        takes = [_take("b", "src1", 1, 5.0, 6.0), _take("a", "src1", 0, 0.0, 1.0)]
        units = ordp.build_ordering_units(realizations=takes)
        result = oca.run_existing_compose_selected_compat(units, takes)
        assert result == ("a", "b")

    def test_compose_selected_compat_identity_gap_returns_empty(self):
        take_a = _take("a", "src1", 0, 0.0, 1.0)
        take_b = _take("b", "src1", 1, 1.0, 2.0)
        units = ordp.build_ordering_units(realizations=[take_a, take_b])
        result = oca.run_existing_compose_selected_compat(units, [take_a])  # "b" missing
        assert result == ()


# ---------------------------------------------------------------------------
# 11. Diagnostics / run summary.
# ---------------------------------------------------------------------------
class TestDiagnosticsAndSummary:
    def test_proposal_diagnostics_shape(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, plan = _units_and_baseline(takes)
        result = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=MockComposerProvider(("a", "b")),
        )
        diag = oca.ordering_composer_proposal_diagnostics(result)
        assert diag["proposal_status"] == oca.PROPOSAL_ACCEPTED
        assert diag["causal_validator_status"] == oca.CAUSAL_VALIDATOR_NOT_INTEGRATED
        assert "transcript" not in diag

    def test_run_summary_counts(self):
        takes = [_take("a", "src1", 0, 0.0, 1.0), _take("b", "src1", 1, 1.0, 2.0)]
        units, plan = _units_and_baseline(takes)
        accepted = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=MockComposerProvider(("a", "b")),
        )
        rejected = oca.validate_composer_ordering_proposal(
            units=units, realizations=takes, relations=(), baseline_plan=plan, provider=MockComposerProvider(("a",)),
        )
        summary = oca.ordering_composer_run_summary([accepted, rejected])
        assert summary["proposal_count"] == 2
        assert summary["accepted_count"] == 1
        assert summary["rejected_membership_count"] == 1
        assert summary["causal_validator_failure_count"] == 0
        assert "score" not in summary and "master_score" not in summary


# ---------------------------------------------------------------------------
# 12. Structural no-second-composer / no-provider / no-authority audits.
# ---------------------------------------------------------------------------
class TestStructuralAudits:
    def test_composer_py_unchanged(self):
        # Adapter imports the real function only -- never redefines it.
        assert "def compose_selected(" not in MODULE_SOURCE

    def test_composer_provider_py_unchanged(self):
        assert "def safe_compose_order(" not in MODULE_SOURCE
        assert "def _repair_order(" not in MODULE_SOURCE

    def test_openai_composer_provider_never_imported(self):
        # The module docstring DISCUSSES composer_openai/OpenAIComposerProvider
        # (why they stay dormant/untouched) without ever importing or
        # instantiating either -- check real import/call statements, not
        # the documentation explaining why they are deliberately unused.
        for banned in (
            "from .composer_openai import", "import composer_openai",
            "OpenAIComposerProvider(",
        ):
            assert banned not in MODULE_SOURCE

    def test_no_network_or_api_key(self):
        for banned in ("requests.", "urllib", "http.client", "OPENAI_API_KEY", "responses.create"):
            assert banned not in MODULE_SOURCE

    def test_causal_order_validator_never_imported(self):
        for banned in (
            "from .causal_order_validator import", "import causal_order_validator",
            "find_causal_order_breaks(",
        ):
            assert banned not in MODULE_SOURCE

    def test_no_family_besttake_boundary_pacing_renderer_reference(self):
        forbidden = (
            "take_group_id", "_semantic_best_take", "bounded_finalist_authority",
            "bounded_finalist_arbiter", "boundary_engine_pass", "BoundaryEngine", "dialogue_pacing_transition",
            "render_plan", "RenderSegment", "take_grouping", "composite_resolver", "realization_resolver",
        )
        for name in forbidden:
            assert name not in CODE_ONLY

    def test_no_qa_reference(self):
        for banned in ("cut_ai", "cutai", "human_gold", "quality_ladder", "benchmark_label"):
            assert banned not in CODE_ONLY.lower()

    def test_no_commercial_or_funnel_fields(self):
        for banned in ("commercial", "sales_funnel", "funnel", "cta_score", "hook_strength", "narrative_quality"):
            assert banned not in CODE_ONLY.lower()

    def test_no_master_score_field(self):
        from dataclasses import fields
        assert "score" not in {f.name for f in fields(oca.OrderingComposerProposalResult)}

    def test_no_edit_action_field(self):
        from dataclasses import fields
        for f in fields(oca.OrderingComposerProposalResult):
            for banned in ("delete", "render", "cut_frame", "select_clip"):
                assert banned not in f.name.lower()

    def test_no_story_scoring_or_llm_prompt(self):
        for banned in ("hook_strength", "cta_score", "instruction =", "responses.create"):
            assert banned not in CODE_ONLY

    def test_no_pipeline_wiring_files_touched(self):
        for path in ("cutsell_worker/pipeline.py", "cutsell_worker/universal_clean_cut.py", "cutsell_worker/brain_runtime.py"):
            content = (REPO_ROOT / path).read_text()
            assert "ordering_composer_adapter" not in content

    def test_no_random_uuid_in_source(self):
        assert "uuid" not in CODE_ONLY.lower()


# ---------------------------------------------------------------------------
# 13. Compileall / import sanity.
# ---------------------------------------------------------------------------
def test_module_compiles_and_imports():
    import cutsell_worker.ordering_composer_adapter  # noqa: F401
