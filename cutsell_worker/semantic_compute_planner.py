"""D-052 Part B: provider-neutral semantic compute planning.

See ``docs/CUTSELL_DECISIONS.md`` D-051 (the audit that proved hybrid
semantic compute was being allocated purely by ``chunk_index`` call order
against one fixed, shared ``$0.0075`` dollar ledger -- so whichever chunks
happened to be requested fifth or sixth silently lost all semantic
evaluation, with zero regard for what they contained) and D-052 (this fix).

PROBLEM
=======
``hybrid_session_cleanup.apply_hybrid_session_cleanup`` iterates windows in
plain enumeration order and calls the paid transport for each one in turn;
``hybrid_google_transport.DollarBudgetLedger.reserve`` simply refuses the
Nth call once the ledger is exhausted. Nothing about that mechanism knows
or cares whether the window it is refusing contains a contradiction, a
critical-claim ambiguity, or an utterly ordinary BTS clip -- "safety
critical" and "cosmetic" compete for the same budget on a first-come basis.

FIX SHAPE (this module)
========================
A ``SemanticComputePlan`` is built BEFORE any paid call is made, from a
list of provider-neutral ``SemanticWorkItem`` values. Each item carries a
``priority`` (P0 safety-critical / P1 retry-equivalence-ambiguity / P2
ordinary editorial quality), an estimated cost, and a reason. The planner
sorts by priority (a *stable* sort -- ties keep their original relative
order, so changing the caller's iteration order never changes which items
of a given priority run first within that tier) and greedily reserves
budget P0 first, then P1, then P2, against one ``cost_ceiling_usd``.
Anything that does not fit is reported in ``deferred_optional_calls``
rather than being executed and silently failing.

This module knows nothing about Gemini, HTTP, or any specific transport --
see ``PRIORITY MODEL`` below and Section 11 of the D-052 directive. A
transport layer (today: ``hybrid_google_transport.GoogleGeminiTransport``)
is the thing that actually executes ``planned_calls`` in the order the
plan specifies; this module only decides WHICH items get a call and in
WHAT ORDER, never HOW the call itself is made.

This module is additive-only and not wired into the live pipeline by
default -- see ``hybrid_session_cleanup.py``'s
``CUTSELL_SEMANTIC_COMPUTE_PLANNER`` flag (default OFF, preserving today's
exact chunk_index-order behavior) for the opt-in integration point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Tuple


class SemanticWorkPriority(IntEnum):
    """Lower value = higher priority = reserved first. Ordering (not the
    exact numeric values) is the load-bearing part of this enum -- do not
    reorder these members."""
    P0_SAFETY_CRITICAL = 0
    P1_RETRY_EQUIVALENCE = 1
    P2_EDITORIAL_QUALITY = 2


# D-052 Section 10: what happens to a work item that priority-planning could
# not afford. Each name doubles as the diagnostic string surfaced in
# SemanticWorkOutcome.exhaustion_action so a report can say exactly why a
# given item never made a call, not just that it didn't.
P0_EXHAUSTION_ACTION = "REVIEW_REQUIRED_fail_closed"
P1_EXHAUSTION_ACTION = "REVIEW_REQUIRED_semantic_risk"
P2_EXHAUSTION_ACTION = "KEEP_deterministic_fallback"

_EXHAUSTION_ACTION_BY_PRIORITY = {
    SemanticWorkPriority.P0_SAFETY_CRITICAL: P0_EXHAUSTION_ACTION,
    SemanticWorkPriority.P1_RETRY_EQUIVALENCE: P1_EXHAUSTION_ACTION,
    SemanticWorkPriority.P2_EDITORIAL_QUALITY: P2_EXHAUSTION_ACTION,
}


@dataclass(frozen=True)
class SemanticWorkItem:
    """One unit of semantic work a caller wants a provider to evaluate.

    Provider-neutral by design (Section 11): no Gemini-specific field
    exists here. ``work_id`` is caller-defined and opaque to this module
    (a caller might use its own chunk/session id); it is only used to
    report which items were planned/deferred, never interpreted.
    """
    work_id: str
    priority: SemanticWorkPriority
    estimated_cost_usd: float
    reason: str
    safety_critical: bool = False


@dataclass(frozen=True)
class SemanticWorkOutcome:
    work_id: str
    priority: SemanticWorkPriority
    estimated_cost_usd: float
    reason: str
    planned: bool
    exhaustion_action: str | None = None


@dataclass(frozen=True)
class SemanticComputePlan:
    """The full plan, computed once before any paid call is made."""
    work_items: Tuple[SemanticWorkOutcome, ...]
    planned_calls: Tuple[str, ...]
    deferred_optional_calls: Tuple[str, ...]
    planned_cost_usd: float
    cost_ceiling_usd: float
    required_p0_cost_usd: float
    required_p1_cost_usd: float
    optional_p2_cost_usd: float

    @property
    def eligible_work_item_count(self) -> int:
        return len(self.work_items)

    def outcome_for(self, work_id: str) -> SemanticWorkOutcome | None:
        for outcome in self.work_items:
            if outcome.work_id == work_id:
                return outcome
        return None


def _priority_cost(items: Tuple[SemanticWorkItem, ...], priority: SemanticWorkPriority) -> float:
    return round(sum(item.estimated_cost_usd for item in items if item.priority == priority), 6)


def build_semantic_compute_plan(
    work_items: Tuple[SemanticWorkItem, ...] | list[SemanticWorkItem],
    *,
    cost_ceiling_usd: float,
) -> SemanticComputePlan:
    """D-052 Section 6/8: replace call-order budgeting with a plan computed
    up front from the full set of eligible work.

    Deterministic and provider-neutral: given the same ``work_items`` (as a
    *set*, not a specific input order) and the same ``cost_ceiling_usd``,
    this always plans the same calls in the same order, regardless of what
    order the caller happened to iterate its own candidates/chunks in --
    see ``tests/test_cutsell_d052_semantic_compute_planner.py``'s
    order-independence test.
    """
    items = tuple(work_items)
    ceiling = max(0.0, float(cost_ceiling_usd))

    # Stable sort: ties (same priority) keep their original relative order.
    # This is what makes "changing input iteration order does not change
    # which priority items execute" true -- reordering the CALLER's list
    # only reorders within a priority tier, and even that is deterministic
    # because Python's sort is stable and priority is the only sort key.
    ordered = sorted(items, key=lambda item: int(item.priority))

    reserved = 0.0
    outcomes: list[SemanticWorkOutcome] = []
    planned: list[str] = []
    deferred: list[str] = []
    for item in ordered:
        cost = max(0.0, float(item.estimated_cost_usd))
        if reserved + cost <= ceiling:
            reserved += cost
            planned.append(item.work_id)
            outcomes.append(SemanticWorkOutcome(
                work_id=item.work_id,
                priority=item.priority,
                estimated_cost_usd=cost,
                reason=item.reason,
                planned=True,
                exhaustion_action=None,
            ))
        else:
            deferred.append(item.work_id)
            outcomes.append(SemanticWorkOutcome(
                work_id=item.work_id,
                priority=item.priority,
                estimated_cost_usd=cost,
                reason=item.reason,
                planned=False,
                exhaustion_action=_EXHAUSTION_ACTION_BY_PRIORITY[item.priority],
            ))

    return SemanticComputePlan(
        work_items=tuple(outcomes),
        planned_calls=tuple(planned),
        deferred_optional_calls=tuple(deferred),
        planned_cost_usd=round(reserved, 6),
        cost_ceiling_usd=ceiling,
        required_p0_cost_usd=_priority_cost(items, SemanticWorkPriority.P0_SAFETY_CRITICAL),
        required_p1_cost_usd=_priority_cost(items, SemanticWorkPriority.P1_RETRY_EQUIVALENCE),
        optional_p2_cost_usd=_priority_cost(items, SemanticWorkPriority.P2_EDITORIAL_QUALITY),
    )


def build_cost_contract_report(plan: SemanticComputePlan, *, actual_cost_usd: float | None = None) -> dict:
    """D-052 Section 9: a predictable, reportable per-video semantic
    compute cost contract -- estimated before execution, actual after.
    Pure presentation of an already-built plan; never used to make a
    planning decision itself.
    """
    # D-056.1 item 4: per-tier P0/P1/P2 counts -- purely a presentation
    # addition over data build_semantic_compute_plan already computed
    # (each SemanticWorkOutcome already carries its own priority/planned
    # flag in plan.work_items); never changes which items are planned or
    # deferred. D-056's report could only compare P0/P1/P2 dollar buckets,
    # never how many work items actually made up each tier or how many of
    # them were planned vs deferred -- this closes that gap.
    def _count(priority: SemanticWorkPriority, *, planned: bool | None = None) -> int:
        return sum(
            1 for outcome in plan.work_items
            if outcome.priority == priority and (planned is None or outcome.planned == planned)
        )

    report = {
        "schema_version": "cutsell.semantic_compute_plan.v1",
        "eligible_work_item_count": plan.eligible_work_item_count,
        "planned_call_count": len(plan.planned_calls),
        "deferred_call_count": len(plan.deferred_optional_calls),
        "required_p0_cost_usd": plan.required_p0_cost_usd,
        "required_p1_cost_usd": plan.required_p1_cost_usd,
        "optional_p2_cost_usd": plan.optional_p2_cost_usd,
        "cost_ceiling_usd": plan.cost_ceiling_usd,
        "estimated_semantic_cost_usd": plan.planned_cost_usd,
        "p0_eligible_count": _count(SemanticWorkPriority.P0_SAFETY_CRITICAL),
        "p0_planned_count": _count(SemanticWorkPriority.P0_SAFETY_CRITICAL, planned=True),
        "p0_deferred_count": _count(SemanticWorkPriority.P0_SAFETY_CRITICAL, planned=False),
        "p1_eligible_count": _count(SemanticWorkPriority.P1_RETRY_EQUIVALENCE),
        "p1_planned_count": _count(SemanticWorkPriority.P1_RETRY_EQUIVALENCE, planned=True),
        "p1_deferred_count": _count(SemanticWorkPriority.P1_RETRY_EQUIVALENCE, planned=False),
        "p2_eligible_count": _count(SemanticWorkPriority.P2_EDITORIAL_QUALITY),
        "p2_planned_count": _count(SemanticWorkPriority.P2_EDITORIAL_QUALITY, planned=True),
        "p2_deferred_count": _count(SemanticWorkPriority.P2_EDITORIAL_QUALITY, planned=False),
    }
    if actual_cost_usd is not None:
        report["actual_semantic_cost_usd"] = round(max(0.0, float(actual_cost_usd)), 6)
    return report
