"""D-128 -- bounded offline evaluation harness for the D-127 eval set.

Fixture/eval only. NO provider call, NO RAW, NO network. Every fixture is
a synthetic, hand-built `FallbackTriggerDecision` input constructed
directly (never through a live pipeline run) so this harness can run in
any offline test/CI context. It proves `detect_class_b_trigger` behaves
correctly across the D-127 Section 20 required case list -- it does not
call `MultimodalBestTakeArbiter` (there is nothing to call in Phase 1).

`run_offline_eval()` is the bounded "report": one row per case, naming the
case, the expected reason, the actual reason, and whether they match. A
future task may extend this harness with cases built from PERSISTED real
RAW evidence (e.g. D-126's own `tg_ef754f8f610ab360df`/`tg_dfa8f59296237ae030`
shapes) without changing its shape.
"""
from __future__ import annotations

from dataclasses import dataclass

from .case_b_performance_evidence import CaseBPerformanceEvidence
from .multimodal_besttake_fallback import (
    CASE_B_NOT_DOMINATED,
    CLASS_B_ELIGIBLE,
    NO_CASE_B_EVIDENCE,
    NO_MEANING_SUFFICIENT_ALTERNATIVE,
    NOT_MULTI_MEMBER,
    SAFETY_EXCLUSION,
    SEMANTIC_DELIVERYSCORE_DISAGREE,
    FallbackTriggerDecision,
    detect_class_b_trigger,
)


def _evidence(candidate_id: str, count_by_kind: dict[str, int], duration_by_kind: dict[str, float]) -> CaseBPerformanceEvidence:
    total_count = sum(count_by_kind.values())
    total_duration = round(sum(duration_by_kind.values()), 3)
    return CaseBPerformanceEvidence(
        candidate_id=candidate_id,
        source_asset_id="src",
        delivery_available=True,
        delivery_start=0.0,
        delivery_end=10.0,
        delivery_span_duration=10.0,
        delivery_events=(),
        delivery_event_count=total_count,
        delivery_event_duration_total=total_duration,
        count_by_kind=dict(count_by_kind),
        duration_by_kind=dict(duration_by_kind),
        event_density=(total_duration / 10.0) if total_duration else 0.0,
    )


@dataclass(frozen=True)
class EvalCase:
    name: str
    description: str
    kwargs: dict
    expected_eligible: bool
    expected_reason: str


def _case_pimples_shaped_positive() -> EvalCase:
    """D-127/D-126's own positive instance: agreed winner (A) carries MORE
    local-performance evidence than a meaning-sufficient alternative (B)."""
    winner_evidence = _evidence("A", {"hand_motion_reset_candidate": 8}, {"hand_motion_reset_candidate": 0.6})
    alt_evidence = _evidence("B", {"hand_motion_reset_candidate": 6}, {"hand_motion_reset_candidate": 0.47})
    return EvalCase(
        name="pimples_shaped_positive",
        description="D-126 shape: semantic==DeliveryScorer agree on A; B is meaning-sufficient and factually cleaner.",
        kwargs=dict(
            family_id="tg_eval_pimples", member_ids=("A", "B"),
            semantic_winner="A", deliveryscore_winner="A",
            meaning_sufficient_ids={"A", "B"},
            case_b_evidence_by_id={"A": winner_evidence, "B": alt_evidence},
            safety_excluded_ids=set(),
        ),
        expected_eligible=True, expected_reason=CLASS_B_ELIGIBLE,
    )


def _case_papillary_equivalent_realization_negative() -> EvalCase:
    """Agreement, and the two realizations are truly evidence-tied (no
    dominance) -- must NOT fire. Named for D-127 Section 20's papillary
    equivalent-realization negative control."""
    a = _evidence("A", {"facial_expression_shift_candidate": 1}, {"facial_expression_shift_candidate": 0.07})
    b = _evidence("B", {"facial_expression_shift_candidate": 1}, {"facial_expression_shift_candidate": 0.07})
    return EvalCase(
        name="papillary_equivalent_realization_negative",
        description="Agreement, tied CASE B evidence -- arbiter must not be needed.",
        kwargs=dict(
            family_id="tg_eval_papillary", member_ids=("A", "B"),
            semantic_winner="A", deliveryscore_winner="A",
            meaning_sufficient_ids={"A", "B"},
            case_b_evidence_by_id={"A": a, "B": b},
            safety_excluded_ids=set(),
        ),
        expected_eligible=False, expected_reason=CASE_B_NOT_DOMINATED,
    )


def _case_stomach_retry_negative() -> EvalCase:
    """D-097.11's own escalation-A instability is upstream semantic-arbiter
    variance, not a BestTake CASE B conflict -- modeled here as "no
    semantic winner" (the D-097.11 stomach family's own documented
    run-to-run flip in DECISIVENESS, not a clean agreement)."""
    a = _evidence("A", {"body_reset_candidate": 2}, {"body_reset_candidate": 0.13})
    b = _evidence("B", {"body_reset_candidate": 1}, {"body_reset_candidate": 0.07})
    return EvalCase(
        name="stomach_retry_negative",
        description="Upstream semantic-arbiter instability (D-097.11) -- not this arbiter's problem; no decisive semantic winner this run.",
        kwargs=dict(
            family_id="tg_eval_stomach", member_ids=("A", "B"),
            semantic_winner=None, deliveryscore_winner="A",
            meaning_sufficient_ids={"A", "B"},
            case_b_evidence_by_id={"A": a, "B": b},
            safety_excluded_ids=set(),
        ),
        expected_eligible=False, expected_reason="NO_SEMANTIC_WINNER",
    )


def _case_complementary_content_negative() -> EvalCase:
    """Two members carry genuinely distinct information -- D-019's KEEP/
    DISCARD-only doctrine means only ONE should ever be named; modeled as
    a disagreement (D-123's own territory), never a Class B trigger."""
    a = _evidence("A", {"hand_motion_reset_candidate": 3}, {"hand_motion_reset_candidate": 0.2})
    b = _evidence("B", {"hand_motion_reset_candidate": 1}, {"hand_motion_reset_candidate": 0.07})
    return EvalCase(
        name="complementary_content_negative",
        description="Semantic/DeliveryScorer disagree -- D-123's territory, never Class B, never KEEP_BOTH.",
        kwargs=dict(
            family_id="tg_eval_complementary", member_ids=("A", "B"),
            semantic_winner="A", deliveryscore_winner="B",
            meaning_sufficient_ids={"A", "B"},
            case_b_evidence_by_id={"A": a, "B": b},
            safety_excluded_ids=set(),
        ),
        expected_eligible=False, expected_reason=SEMANTIC_DELIVERYSCORE_DISAGREE,
    )


def _case_polarity_negation_safety_negative() -> EvalCase:
    """Alternative is factually cleaner on CASE B but excluded by an
    existing deterministic safety rule (D-101/D-103 contradiction/
    required-condition-realization veto) -- reused verbatim via
    `safety_excluded_ids`, never re-implemented here."""
    a = _evidence("A", {"hand_motion_reset_candidate": 5}, {"hand_motion_reset_candidate": 0.33})
    b = _evidence("B", {"hand_motion_reset_candidate": 1}, {"hand_motion_reset_candidate": 0.07})
    return EvalCase(
        name="polarity_negation_safety_negative",
        description="B is factually cleaner but excluded by an existing deterministic safety veto (e.g. contradicts A).",
        kwargs=dict(
            family_id="tg_eval_polarity", member_ids=("A", "B"),
            semantic_winner="A", deliveryscore_winner="A",
            meaning_sufficient_ids={"A", "B"},
            case_b_evidence_by_id={"A": a, "B": b},
            safety_excluded_ids={"B"},
        ),
        expected_eligible=False, expected_reason=SAFETY_EXCLUSION,
    )


def _case_legitimate_clean_retry_negative() -> EvalCase:
    """A trivial, single-member "family" (no real contest at all) --
    NOT_MULTI_MEMBER, structurally excluded before any evidence is even
    consulted."""
    a = _evidence("A", {}, {})
    return EvalCase(
        name="legitimate_clean_retry_negative",
        description="Single-member family -- no contest, no fallback consideration possible.",
        kwargs=dict(
            family_id="tg_eval_clean_retry", member_ids=("A",),
            semantic_winner="A", deliveryscore_winner="A",
            meaning_sufficient_ids={"A"},
            case_b_evidence_by_id={"A": a},
            safety_excluded_ids=set(),
        ),
        expected_eligible=False, expected_reason=NOT_MULTI_MEMBER,
    )


def _case_semantic_deliveryscore_disagreement_negative() -> EvalCase:
    """D-126's own confirmed real D-123 bypass shape
    (`tg_ef754f8f610ab360df`) -- a genuine disagreement D-123 already
    resolves; Class B must never fire here (D-123 owns it)."""
    a = _evidence("A", {"hand_motion_reset_candidate": 7}, {"hand_motion_reset_candidate": 0.469})
    b = _evidence("B", {"facial_expression_shift_candidate": 1, "hand_motion_reset_candidate": 1},
                  {"facial_expression_shift_candidate": 0.067, "hand_motion_reset_candidate": 0.067})
    return EvalCase(
        name="semantic_deliveryscore_disagreement_negative",
        description="D-126's own real D-123 bypass shape -- D-123's territory, Class B must defer.",
        kwargs=dict(
            family_id="tg_eval_d123_bypass_shape", member_ids=("A", "B"),
            semantic_winner="A", deliveryscore_winner="B",
            meaning_sufficient_ids={"A", "B"},
            case_b_evidence_by_id={"A": a, "B": b},
            safety_excluded_ids=set(),
        ),
        expected_eligible=False, expected_reason=SEMANTIC_DELIVERYSCORE_DISAGREE,
    )


def _case_ambiguous_tied_performance_negative() -> EvalCase:
    """Agreement, and the alternative is better on one dimension but worse
    on another -- no safe partial-order dominance -- must return
    CASE_B_NOT_DOMINATED, never a guess."""
    a = _evidence("A", {"hand_motion_reset_candidate": 3, "facial_expression_shift_candidate": 1},
                  {"hand_motion_reset_candidate": 0.2, "facial_expression_shift_candidate": 0.02})
    b = _evidence("B", {"hand_motion_reset_candidate": 1, "facial_expression_shift_candidate": 3},
                  {"hand_motion_reset_candidate": 0.07, "facial_expression_shift_candidate": 0.3})
    return EvalCase(
        name="ambiguous_tied_performance_negative",
        description="B better on hand_motion_reset_candidate, worse on facial_expression_shift_candidate -- no safe dominance.",
        kwargs=dict(
            family_id="tg_eval_ambiguous", member_ids=("A", "B"),
            semantic_winner="A", deliveryscore_winner="A",
            meaning_sufficient_ids={"A", "B"},
            case_b_evidence_by_id={"A": a, "B": b},
            safety_excluded_ids=set(),
        ),
        expected_eligible=False, expected_reason=CASE_B_NOT_DOMINATED,
    )


def _case_boundary_only_exit_negative() -> EvalCase:
    """The only observed difference is an ENTRY/EXIT-zone event -- which
    `case_b_performance_evidence.py` already excludes from CASE B evidence
    entirely (D-115's DELIVERY-zone-only filter). Modeled here directly as
    both candidates having IDENTICAL (zero) CASE B evidence, since that is
    exactly what an ENTRY/EXIT-only difference produces once it reaches
    this module -- CASE B never even sees it."""
    a = _evidence("A", {}, {})
    b = _evidence("B", {}, {})
    return EvalCase(
        name="boundary_only_exit_negative",
        description="ENTRY/EXIT-only defect never enters CASE B evidence at all (D-115); both candidates show zero DELIVERY-zone events.",
        kwargs=dict(
            family_id="tg_eval_boundary_exit", member_ids=("A", "B"),
            semantic_winner="A", deliveryscore_winner="A",
            meaning_sufficient_ids={"A", "B"},
            case_b_evidence_by_id={"A": a, "B": b},
            safety_excluded_ids=set(),
        ),
        expected_eligible=False, expected_reason=CASE_B_NOT_DOMINATED,
    )


_ALL_CASES = (
    _case_pimples_shaped_positive,
    _case_papillary_equivalent_realization_negative,
    _case_stomach_retry_negative,
    _case_complementary_content_negative,
    _case_polarity_negation_safety_negative,
    _case_legitimate_clean_retry_negative,
    _case_semantic_deliveryscore_disagreement_negative,
    _case_ambiguous_tied_performance_negative,
    _case_boundary_only_exit_negative,
)


def eval_cases() -> tuple[EvalCase, ...]:
    """All D-127 Section 20 required cases, built fresh (never cached
    mutable state)."""
    return tuple(builder() for builder in _ALL_CASES)


def run_offline_eval() -> list[dict]:
    """Runs every eval case through `detect_class_b_trigger` (never a
    provider) and returns one compact report row per case: `name`,
    `description`, `expected_eligible`, `actual_eligible`,
    `expected_reason`, `actual_reason`, `passed`."""
    rows: list[dict] = []
    for case in eval_cases():
        decision: FallbackTriggerDecision = detect_class_b_trigger(**case.kwargs)
        rows.append({
            "name": case.name,
            "description": case.description,
            "expected_eligible": case.expected_eligible,
            "actual_eligible": decision.eligible,
            "expected_reason": case.expected_reason,
            "actual_reason": decision.reason,
            "passed": (
                decision.eligible == case.expected_eligible
                and decision.reason == case.expected_reason
            ),
        })
    return rows
