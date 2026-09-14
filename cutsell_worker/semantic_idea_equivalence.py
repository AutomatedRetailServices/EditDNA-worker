"""Provider-neutral semantic idea-equivalence arbiter contract.

Phase 2 of the CutSell hybrid Selection rebalance. This module contains no
SDK imports and performs no network calls -- it defines the stable contract
between the deterministic lexical retry-family layer (take_grouping.py /
take_grouping_provider.py, unchanged by this phase) and an optional narrow
semantic arbiter that answers exactly one question per pair of candidate
texts:

    "Do these two deliveries represent recording attempts of the same
    intended idea/message?"

It must never perform full Selection, never sees clip_ids/timestamps/video
identity (text pairs only, so it cannot become a Video00-specific guard),
and fails open to "not the same idea" (preserve as separate story beats)
on any provider error or uncertainty -- merging two genuinely distinct
beats is the destructive direction; keeping them separate is not.

Mirrors hybrid_editorial.py's existing contract shape (Candidate/Session/
Decision/Result/Protocol/GatePolicy/should_request/validate/safe_call) so
this reads as the same kind of authority the codebase already has, not a
new pattern.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple


@dataclass(frozen=True)
class IdeaEquivalencePair:
    """Pure text evidence for one candidate pair. No clip_id, no timestamp,
    no source/video identity -- an arbiter given only this cannot encode a
    Video00-specific or clip-specific rule even by accident."""
    left_text: str
    right_text: str


@dataclass(frozen=True)
class IdeaEquivalenceRequest:
    pairs: Tuple[IdeaEquivalencePair, ...]


@dataclass(frozen=True)
class IdeaEquivalenceDecision:
    pair_index: int
    same_idea: bool
    confidence: float
    # Concise general evidence/reason (e.g. "shared topic and outcome,
    # different wording" or "different subject matter") -- never a
    # clip-specific or video-specific rule.
    reason: str = ""


@dataclass(frozen=True)
class IdeaEquivalenceResult:
    decisions: Tuple[IdeaEquivalenceDecision, ...]
    provider: str
    model: str
    requested: bool
    available: bool
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0


class SemanticEquivalenceArbiter(Protocol):
    def check(self, request: IdeaEquivalenceRequest) -> IdeaEquivalenceResult: ...


@dataclass(frozen=True)
class SemanticEquivalenceGatePolicy:
    """Batch/cost bounds. Values match hybrid_editorial.HybridGatePolicy's
    existing per-request bounds verbatim -- this arbiter's payload shape
    (short text pairs, tiny bool+float+short-string output) is smaller per
    item than a full editorial candidate, so reusing the same ceiling is
    conservative, not a new number chosen for this phase."""
    max_pairs_per_request: int = 14
    max_estimated_input_tokens: int = 12_000
    max_estimated_output_tokens: int = 1_000
    # D-094.2 (default OFF -- Product Owner decision, D-091 stop condition B):
    # let `split_incohesive_retry_groups` accept a SINGLETON-attaches-to-
    # component bridge on complete pairwise confirmation (every cross pair
    # already confirmed at >= the D-085 bridge floor, no cross-component
    # contradiction) WITHOUT the D-085 component-level probe. Component-to-
    # component bridges (>= 2 members on both sides) always keep the probe.
    accept_complete_pairwise_singleton_bridge: bool = False


def should_request_semantic_equivalence(
    request: IdeaEquivalenceRequest,
    policy: SemanticEquivalenceGatePolicy = SemanticEquivalenceGatePolicy(),
) -> bool:
    if not request.pairs:
        return False
    return len(request.pairs) <= policy.max_pairs_per_request


def validate_idea_equivalence_result(
    request: IdeaEquivalenceRequest,
    result: IdeaEquivalenceResult,
) -> IdeaEquivalenceResult:
    expected = set(range(len(request.pairs)))
    seen: set[int] = set()
    normalized: list[IdeaEquivalenceDecision] = []

    if result.estimated_input_tokens < 0 or result.estimated_output_tokens < 0:
        raise ValueError("semantic equivalence arbiter returned invalid token estimates")

    for decision in result.decisions:
        index = int(decision.pair_index)
        if index not in expected:
            raise ValueError("semantic equivalence arbiter returned unknown pair index")
        if index in seen:
            raise ValueError("semantic equivalence arbiter returned duplicate pair index")
        confidence = float(decision.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("semantic equivalence confidence outside 0..1")
        normalized.append(IdeaEquivalenceDecision(
            pair_index=index,
            same_idea=bool(decision.same_idea),
            confidence=confidence,
            reason=str(decision.reason or "")[:200],
        ))
        seen.add(index)

    if seen != expected:
        raise ValueError("semantic equivalence arbiter omitted a pair")

    return IdeaEquivalenceResult(
        decisions=tuple(sorted(normalized, key=lambda d: d.pair_index)),
        provider=str(result.provider or "unknown")[:80],
        model=str(result.model or "unknown")[:120],
        requested=bool(result.requested),
        available=bool(result.available),
        estimated_input_tokens=int(result.estimated_input_tokens),
        estimated_output_tokens=int(result.estimated_output_tokens),
    )


def safe_check_idea_equivalence(
    arbiter: SemanticEquivalenceArbiter | None,
    request: IdeaEquivalenceRequest,
    policy: SemanticEquivalenceGatePolicy = SemanticEquivalenceGatePolicy(),
) -> IdeaEquivalenceResult:
    """Call an injected arbiter only when the gate allows it; otherwise fail
    open. Any exception, validation failure, or over-budget response is
    treated identically to "not available" -- callers must then treat every
    pair as same_idea=False (preserve as separate), never as a merge."""
    if arbiter is None or not should_request_semantic_equivalence(request, policy):
        return IdeaEquivalenceResult((), "none", "none", False, False)

    try:
        result = validate_idea_equivalence_result(request, arbiter.check(request))
        if result.estimated_input_tokens > policy.max_estimated_input_tokens:
            raise ValueError("semantic equivalence input token budget exceeded")
        if result.estimated_output_tokens > policy.max_estimated_output_tokens:
            raise ValueError("semantic equivalence output token budget exceeded")
        return result
    except Exception as exc:
        detail = str(exc).strip().replace("\n", " ")[:120]
        reason = exc.__class__.__name__ + (f":{detail}" if detail else "")
        return IdeaEquivalenceResult((), reason[:160], "none", True, False)


def same_idea_by_pair_index(result: IdeaEquivalenceResult) -> dict[int, tuple[bool, float, str]]:
    """Fail-open lookup: an index absent from this mapping (arbiter
    unavailable, or gate declined to ask) must be treated as same_idea=False
    by every caller -- preserve as separate, never merge on absence of
    evidence."""
    if not result.available:
        return {}
    return {
        decision.pair_index: (decision.same_idea, decision.confidence, decision.reason)
        for decision in result.decisions
    }
