"""D-128 Phase 1 -- SHADOW-ONLY BestTake fallback trigger detection.

docs/CUTSELL_DECISIONS.md D-128, docs/CUTSELL_MULTIMODAL_FALLBACK_ARBITER_
FORENSIC_D127.md. This module detects exactly ONE general trigger class --
"Class B" (D-127 Section 4): a family where the semantic winner and
DeliveryScorer's own top pick AGREE, but a different meaning-sufficient
member's D-122 CASE B evidence is factually, unambiguously cleaner. It
NEVER decides a winner, NEVER calls a provider, and NEVER changes
`selected_clip_id`/`ranked`/`winner_path`/membership/Boundary -- every
field this module reads is an ALREADY-COMPUTED existing signal (D-081/
D-103 meaning sufficiency, D-101/D-103 safety-veto exclusion, D-122 CASE B
evidence); this module classifies them, never re-derives or re-scores
them.

D-123 already owns the disagreement case (semantic winner != DeliveryScorer
winner) -- this module explicitly excludes it (`SEMANTIC_DELIVERYSCORE_
DISAGREE`), matching D-127 Section 4's own Class-B scope boundary.

No threshold is invented anywhere in this module. `_factual_dominance` is a
pure partial-order comparison (D-127 Section 4/D-128's own "no threshold
invention" requirement): an alternative is considered factually cleaner
ONLY if it is no-worse on every available D-122 CASE B dimension (aggregate
event count, aggregate event duration, and EVERY individual event kind's
own count/duration -- never summed with a weight, per this task's explicit
"do not treat different event kinds as numerically equivalent" instruction)
and strictly better on at least one. A tie, or any dimension where the
alternative is worse, returns no dominance (`CASE_B_NOT_DOMINATED`) --
never a guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Tuple

from .case_b_performance_evidence import CaseBPerformanceEvidence

SCHEMA_VERSION = "cutsell.multimodal_besttake_fallback.v1"

CLASS_B = "CLASS_B"

# Non-eligible reasons (D-128's own required compact vocabulary).
NOT_MULTI_MEMBER = "NOT_MULTI_MEMBER"
NO_SEMANTIC_WINNER = "NO_SEMANTIC_WINNER"
NO_DELIVERYSCORE_WINNER = "NO_DELIVERYSCORE_WINNER"
SEMANTIC_DELIVERYSCORE_DISAGREE = "SEMANTIC_DELIVERYSCORE_DISAGREE"
NO_MEANING_SUFFICIENT_ALTERNATIVE = "NO_MEANING_SUFFICIENT_ALTERNATIVE"
NO_CASE_B_EVIDENCE = "NO_CASE_B_EVIDENCE"
CASE_B_NOT_DOMINATED = "CASE_B_NOT_DOMINATED"
SAFETY_EXCLUSION = "SAFETY_EXCLUSION"
# Eligible reason.
CLASS_B_ELIGIBLE = "CLASS_B_ELIGIBLE"


@dataclass(frozen=True)
class FallbackTriggerDecision:
    """The full, read-only result of evaluating one family for Class B
    eligibility. `eligible=False` on every non-`CLASS_B_ELIGIBLE` reason;
    `alternative_candidates`/`case_b_evidence_by_candidate` are empty
    whenever `eligible` is False -- callers must never build a fallback
    request from a non-eligible decision (see `multimodal_besttake_
    arbiter.build_multimodal_besttake_request`, which enforces this)."""
    trigger_class: str
    eligible: bool
    reason: str
    family_id: str | None
    structured_winner: str | None
    alternative_candidates: Tuple[str, ...]
    meaning_sufficient_candidates: Tuple[str, ...]
    semantic_winner: str | None
    deliveryscore_winner: str | None
    case_b_evidence_by_candidate: Mapping[str, CaseBPerformanceEvidence] = field(default_factory=dict)
    trigger_evidence: Mapping[str, object] = field(default_factory=dict)


def _factual_dominance(
    winner_evidence: CaseBPerformanceEvidence,
    alt_evidence: CaseBPerformanceEvidence,
) -> bool:
    """True iff `alt_evidence` is factually no-worse than `winner_evidence`
    on EVERY available D-122 CASE B dimension, and strictly better on at
    least one. A pure partial-order comparison -- no score, no weight, no
    threshold. Dimensions: aggregate `delivery_event_count`, aggregate
    `delivery_event_duration_total`, and every individual event kind's own
    count/duration (the union of both candidates' `count_by_kind`/
    `duration_by_kind` keys -- a kind present for only one candidate is
    compared against an implicit 0 for the other, never dropped)."""
    pairs: list[tuple[float, float]] = [
        (winner_evidence.delivery_event_count, alt_evidence.delivery_event_count),
        (winner_evidence.delivery_event_duration_total, alt_evidence.delivery_event_duration_total),
    ]
    for kind in set(winner_evidence.count_by_kind) | set(alt_evidence.count_by_kind):
        pairs.append((winner_evidence.count_by_kind.get(kind, 0), alt_evidence.count_by_kind.get(kind, 0)))
    for kind in set(winner_evidence.duration_by_kind) | set(alt_evidence.duration_by_kind):
        pairs.append((winner_evidence.duration_by_kind.get(kind, 0.0), alt_evidence.duration_by_kind.get(kind, 0.0)))

    strictly_better_somewhere = False
    for winner_value, alt_value in pairs:
        if alt_value > winner_value:
            return False  # worse on this dimension -- no safe dominance
        if alt_value < winner_value:
            strictly_better_somewhere = True
    return strictly_better_somewhere


def detect_class_b_trigger(
    *,
    family_id: str | None,
    member_ids: Tuple[str, ...],
    semantic_winner: str | None,
    deliveryscore_winner: str | None,
    meaning_sufficient_ids: Iterable[str],
    case_b_evidence_by_id: Mapping[str, CaseBPerformanceEvidence],
    safety_excluded_ids: Iterable[str],
) -> FallbackTriggerDecision:
    """Pure, read-only Class B trigger detector. All inputs are ALREADY-
    COMPUTED existing signals; this function classifies them, never
    re-derives or re-scores anything, and never calls a provider.

    Eligibility (ALL required, checked in the order that produces the
    most specific compact reason on failure):
      1. `member_ids` has >= 2 members (a legitimate multi-member family) --
         else `NOT_MULTI_MEMBER`.
      2. `semantic_winner` is not None (a decisive semantic label exists) --
         else `NO_SEMANTIC_WINNER`.
      3. `deliveryscore_winner` is truthy (DeliveryScorer's own top pick
         exists) -- else `NO_DELIVERYSCORE_WINNER`.
      4. `semantic_winner == deliveryscore_winner` (AGREEMENT -- D-123
         already owns the disagreement shape) -- else `SEMANTIC_
         DELIVERYSCORE_DISAGREE`.
      5. D-122 CASE B evidence exists for the structured (agreed) winner --
         else `NO_CASE_B_EVIDENCE`.
      6. At least one OTHER member is meaning-sufficient -- else `NO_
         MEANING_SUFFICIENT_ALTERNATIVE`.
      7. At least one meaning-sufficient alternative is not excluded by an
         existing deterministic safety rule (reused verbatim by the
         caller, e.g. `pipeline._single_winner_safety_veto` -- this module
         never re-implements that check) -- else `SAFETY_EXCLUSION`.
      8. At least one safety-eligible alternative has its own D-122 CASE B
         evidence -- else `NO_CASE_B_EVIDENCE`.
      9. At least one such alternative factually dominates the structured
         winner (`_factual_dominance`) -- else `CASE_B_NOT_DOMINATED`.
    """
    meaning_sufficient_ids = set(meaning_sufficient_ids)
    safety_excluded_ids = set(safety_excluded_ids)

    def _not_eligible(reason: str) -> FallbackTriggerDecision:
        return FallbackTriggerDecision(
            trigger_class=CLASS_B,
            eligible=False,
            reason=reason,
            family_id=family_id,
            structured_winner=deliveryscore_winner,
            alternative_candidates=(),
            meaning_sufficient_candidates=tuple(sorted(meaning_sufficient_ids)),
            semantic_winner=semantic_winner,
            deliveryscore_winner=deliveryscore_winner,
        )

    if len(member_ids) < 2:
        return _not_eligible(NOT_MULTI_MEMBER)
    if semantic_winner is None:
        return _not_eligible(NO_SEMANTIC_WINNER)
    if not deliveryscore_winner:
        return _not_eligible(NO_DELIVERYSCORE_WINNER)
    if semantic_winner != deliveryscore_winner:
        return _not_eligible(SEMANTIC_DELIVERYSCORE_DISAGREE)

    structured_winner = semantic_winner
    winner_evidence = case_b_evidence_by_id.get(structured_winner)
    if winner_evidence is None:
        return _not_eligible(NO_CASE_B_EVIDENCE)

    others = [cid for cid in member_ids if cid != structured_winner]
    meaning_sufficient_others = [cid for cid in others if cid in meaning_sufficient_ids]
    if not meaning_sufficient_others:
        return _not_eligible(NO_MEANING_SUFFICIENT_ALTERNATIVE)

    safety_ok_others = [cid for cid in meaning_sufficient_others if cid not in safety_excluded_ids]
    if not safety_ok_others:
        return _not_eligible(SAFETY_EXCLUSION)

    evidenced_others = [cid for cid in safety_ok_others if case_b_evidence_by_id.get(cid) is not None]
    if not evidenced_others:
        return _not_eligible(NO_CASE_B_EVIDENCE)

    dominating = sorted(
        cid for cid in evidenced_others
        if _factual_dominance(winner_evidence, case_b_evidence_by_id[cid])
    )
    if not dominating:
        return _not_eligible(CASE_B_NOT_DOMINATED)

    evidence_by_candidate: dict[str, CaseBPerformanceEvidence] = {structured_winner: winner_evidence}
    for cid in dominating:
        evidence_by_candidate[cid] = case_b_evidence_by_id[cid]

    return FallbackTriggerDecision(
        trigger_class=CLASS_B,
        eligible=True,
        reason=CLASS_B_ELIGIBLE,
        family_id=family_id,
        structured_winner=structured_winner,
        alternative_candidates=tuple(dominating),
        meaning_sufficient_candidates=tuple(sorted(meaning_sufficient_ids)),
        semantic_winner=semantic_winner,
        deliveryscore_winner=deliveryscore_winner,
        case_b_evidence_by_candidate=evidence_by_candidate,
        trigger_evidence={
            "winner_delivery_event_count": winner_evidence.delivery_event_count,
            "winner_delivery_event_duration_total": winner_evidence.delivery_event_duration_total,
            "dominating_alternative_delivery_event_counts": {
                cid: case_b_evidence_by_id[cid].delivery_event_count for cid in dominating
            },
        },
    )


def fallback_trigger_diagnostics(decision: FallbackTriggerDecision) -> dict:
    """Compact, JSON-safe, tail-window-friendly projection of a
    `FallbackTriggerDecision` for `take_judge_groups` diagnostics rows.
    Never includes raw `delivery_events` -- only the bounded aggregate
    fields already used elsewhere in D-122/D-123's own diagnostics."""
    return {
        "fallback_shadow_evaluated": True,
        "fallback_shadow_eligible": decision.eligible,
        "fallback_trigger_class": decision.trigger_class,
        "fallback_trigger_reason": decision.reason,
        "fallback_structured_winner": decision.structured_winner if decision.eligible else None,
        "fallback_candidate_ids": (
            [decision.structured_winner] + list(decision.alternative_candidates)
            if decision.eligible else []
        ),
        "fallback_case_b_dominance": {
            cid: {
                "delivery_event_count": evidence.delivery_event_count,
                "delivery_event_duration_total": evidence.delivery_event_duration_total,
            }
            for cid, evidence in decision.case_b_evidence_by_candidate.items()
        } if decision.eligible else {},
        "fallback_request_candidate_count": (1 + len(decision.alternative_candidates)) if decision.eligible else 0,
        # D-128 Phase 1 is shadow-only by construction -- these are always
        # this value in this task; a future Phase would flip them only
        # where an actual arbiter call/response occurred.
        "fallback_provider_invoked": False,
        "fallback_winner_changed": False,
    }
