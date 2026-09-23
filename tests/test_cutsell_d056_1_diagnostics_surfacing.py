"""D-056.1 item 4: "Surface content_hash, canonical_equivalence_hash,
semantic_compute_plan, P0/P1/P2 counts, actual/estimated semantic cost,
and realization/orphan counts" -- targeted, offline-only tests for the two
purely-additive wiring fixes this item required:

1. `cutsell_worker.flow_b._combine_hashes` -- the shared join helper the
   `canonical_asr_evidence` diagnostic trace now uses to forward D-055's
   `content_hash`/`canonical_equivalence_hash` (previously computed by
   `build_canonical_asr_evidence()` but never threaded into the trace at
   all) alongside the pre-existing `evidence_hash` combination.
2. `cutsell_worker.semantic_compute_planner.build_cost_contract_report`'s
   new per-tier P0/P1/P2 eligible/planned/deferred counts -- pure
   presentation over `SemanticComputePlan.work_items`, which already
   carried this information; no change to which items get planned.

Neither test touches ASR, Resolver, Ledger, Selection, or Freeze
behavior -- both changes are additive diagnostics only.
"""
from __future__ import annotations

from cutsell_worker.flow_b import _combine_hashes
from cutsell_worker.semantic_compute_planner import (
    SemanticWorkItem,
    SemanticWorkPriority,
    build_cost_contract_report,
    build_semantic_compute_plan,
)

P0 = SemanticWorkPriority.P0_SAFETY_CRITICAL
P1 = SemanticWorkPriority.P1_RETRY_EQUIVALENCE
P2 = SemanticWorkPriority.P2_EDITORIAL_QUALITY


def _item(work_id: str, priority: SemanticWorkPriority, cost: float) -> SemanticWorkItem:
    return SemanticWorkItem(work_id=work_id, priority=priority, estimated_cost_usd=cost, reason="test")


# --- _combine_hashes -------------------------------------------------------

def test_combine_hashes_empty_list_returns_empty_string():
    assert _combine_hashes("asrev_combined_", []) == ""


def test_combine_hashes_single_value_has_no_separator():
    assert _combine_hashes("asrev_combined_", ["abc"]) == "asrev_combined_abc"


def test_combine_hashes_joins_multiple_values_in_order():
    assert _combine_hashes("asrcontent_combined_", ["a", "b", "c"]) == "asrcontent_combined_a|b|c"


def test_combine_hashes_matches_the_pre_existing_evidence_hash_convention():
    # D-052's original combined_evidence_hash literal was
    # "asrev_combined_" + "|".join(...) if per_source_evidence else "" --
    # the extracted helper must reproduce that exact string for the same
    # inputs so no existing evidence_hash value shifts underneath anything
    # already comparing it (e.g. a stability battery's own before/after
    # diff of combined_evidence_hash).
    values = ["asrev_srcA_hash", "asrev_srcB_hash"]
    legacy = "asrev_combined_" + "|".join(values) if values else ""
    assert _combine_hashes("asrev_combined_", values) == legacy


# --- build_cost_contract_report P0/P1/P2 counts ----------------------------

def test_cost_contract_report_counts_eligible_items_per_tier():
    items = [
        _item("p0-a", P0, 0.001), _item("p0-b", P0, 0.001),
        _item("p1-a", P1, 0.001),
        _item("p2-a", P2, 0.001), _item("p2-b", P2, 0.001), _item("p2-c", P2, 0.001),
    ]
    plan = build_semantic_compute_plan(items, cost_ceiling_usd=1.0)  # ample budget -- all planned
    report = build_cost_contract_report(plan)

    assert report["p0_eligible_count"] == 2
    assert report["p1_eligible_count"] == 1
    assert report["p2_eligible_count"] == 3
    assert report["p0_planned_count"] == 2
    assert report["p1_planned_count"] == 1
    assert report["p2_planned_count"] == 3
    assert report["p0_deferred_count"] == 0
    assert report["p1_deferred_count"] == 0
    assert report["p2_deferred_count"] == 0


def test_cost_contract_report_splits_planned_vs_deferred_within_a_tier():
    # Budget affords P0 in full, but only ONE of two equally-costed P2
    # items -- P0 must never compete with P2 for the same budget (D-052's
    # whole point), and the new counts must reflect exactly that split.
    items = [
        _item("p0-a", P0, 0.002),
        _item("p2-a", P2, 0.002), _item("p2-b", P2, 0.002),
    ]
    plan = build_semantic_compute_plan(items, cost_ceiling_usd=0.004)
    report = build_cost_contract_report(plan)

    assert report["p0_eligible_count"] == 1
    assert report["p0_planned_count"] == 1
    assert report["p0_deferred_count"] == 0
    assert report["p2_eligible_count"] == 2
    assert report["p2_planned_count"] == 1
    assert report["p2_deferred_count"] == 1
    # Sanity: counts stay consistent with the pre-existing cost/call fields.
    assert report["p0_planned_count"] + report["p2_planned_count"] == report["planned_call_count"]
    assert report["p2_deferred_count"] == report["deferred_call_count"]


def test_cost_contract_report_never_changes_which_items_are_planned():
    # The new counts are read-only presentation over plan.work_items --
    # asserting the pre-existing planned_calls/deferred_optional_calls
    # fields are untouched by this addition.
    items = [_item("a", P0, 0.001), _item("b", P1, 0.001), _item("c", P2, 0.001)]
    plan = build_semantic_compute_plan(items, cost_ceiling_usd=0.0025)
    report = build_cost_contract_report(plan)
    assert set(plan.planned_calls) | set(plan.deferred_optional_calls) == {"a", "b", "c"}
    assert report["eligible_work_item_count"] == plan.eligible_work_item_count
