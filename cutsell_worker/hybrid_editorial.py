"""Provider-neutral hybrid editorial reasoning for CutSell Flow B.

This module deliberately contains no SDK imports and performs no network calls.
It defines the stable contract between the deterministic/local perception brain and
an optional paid or self-hosted editorial LLM judge.

The operating principle is:
- deterministic/local evidence remains the source of timestamps and boundaries;
- an editorial model may classify semantic intent for an already-bounded mini-session;
- model output can only reference supplied candidate IDs;
- invalid or low-confidence output fails open to the local decision;
- cost/rate policy is explicit and testable before any provider is connected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple


_ALLOWED_LABELS = {"winner", "alternate", "failed", "bts", "uncertain", "keep"}


@dataclass(frozen=True)
class EditorialCandidate:
    clip_id: str
    text: str
    start: float
    end: float
    local_label: str
    local_confidence: float
    evidence: tuple[tuple[str, float | str | bool], ...] = ()

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end) - float(self.start))


@dataclass(frozen=True)
class EditorialSession:
    session_id: str
    source_asset_id: str
    candidates: Tuple[EditorialCandidate, ...]
    local_confidence: float
    conflict_score: float = 0.0


@dataclass(frozen=True)
class EditorialDecision:
    clip_id: str
    label: str
    confidence: float
    reason_code: str


@dataclass(frozen=True)
class EditorialJudgeResult:
    decisions: Tuple[EditorialDecision, ...]
    provider: str
    model: str
    requested: bool
    available: bool
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0


class EditorialJudge(Protocol):
    def judge(self, session: EditorialSession) -> EditorialJudgeResult: ...


@dataclass(frozen=True)
class HybridGatePolicy:
    """Controls *when* semantic reasoning is useful, not vendor selection.

    Defaults intentionally allow the LLM to assist a meaningful share of ambiguous
    sessions during beta while keeping obvious local decisions local.
    """

    local_accept_confidence: float = 0.90
    semantic_assist_confidence: float = 0.60
    conflict_trigger: float = 0.30
    max_candidates_per_request: int = 14
    max_estimated_input_tokens: int = 12_000
    max_estimated_output_tokens: int = 1_000


def should_request_editorial_judge(
    session: EditorialSession,
    policy: HybridGatePolicy = HybridGatePolicy(),
) -> bool:
    """Return True when semantic reasoning is likely to add value.

    Obvious high-confidence, low-conflict sessions remain local. Ambiguous or
    internally-conflicting sessions are eligible for the judge. Very weak evidence is
    also eligible: the provider may still return ``uncertain`` and fail open.
    """
    if not session.candidates:
        return False
    if len(session.candidates) > policy.max_candidates_per_request:
        return True  # caller should batch within the same mini-session, never cross it
    if session.conflict_score >= policy.conflict_trigger:
        return True
    if session.local_confidence < policy.local_accept_confidence:
        return True
    return False


def validate_editorial_result(
    session: EditorialSession,
    result: EditorialJudgeResult,
) -> EditorialJudgeResult:
    """Strictly validate provider output before it can influence editing."""
    expected = {candidate.clip_id for candidate in session.candidates}
    seen: set[str] = set()
    normalized: list[EditorialDecision] = []

    if result.estimated_input_tokens < 0 or result.estimated_output_tokens < 0:
        raise ValueError("editorial judge returned invalid token estimates")

    for decision in result.decisions:
        if decision.clip_id not in expected:
            raise ValueError("editorial judge returned unknown clip id")
        if decision.clip_id in seen:
            raise ValueError("editorial judge returned duplicate clip id")
        label = str(decision.label).strip().lower()
        if label not in _ALLOWED_LABELS:
            raise ValueError("editorial judge returned invalid label")
        confidence = float(decision.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("editorial judge confidence outside 0..1")
        reason_code = str(decision.reason_code or "").strip()[:160]
        normalized.append(
            EditorialDecision(
                clip_id=decision.clip_id,
                label=label,
                confidence=confidence,
                reason_code=reason_code,
            )
        )
        seen.add(decision.clip_id)

    if seen != expected:
        raise ValueError("editorial judge omitted candidates")

    return EditorialJudgeResult(
        decisions=tuple(normalized),
        provider=str(result.provider or "unknown")[:80],
        model=str(result.model or "unknown")[:120],
        requested=bool(result.requested),
        available=bool(result.available),
        estimated_input_tokens=int(result.estimated_input_tokens),
        estimated_output_tokens=int(result.estimated_output_tokens),
    )


def safe_editorial_judge(
    provider: EditorialJudge | None,
    session: EditorialSession,
    policy: HybridGatePolicy = HybridGatePolicy(),
) -> EditorialJudgeResult:
    """Call an injected provider only when the gate requests it; otherwise fail open.

    This function itself has no network capability. A future provider adapter must be
    explicitly injected. One malformed provider response is retried once, matching the
    conservative contract-recovery pattern already used by Clean Cut.
    """
    if provider is None or not should_request_editorial_judge(session, policy):
        return EditorialJudgeResult((), "none", "none", False, False)

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            result = validate_editorial_result(session, provider.judge(session))
            if result.estimated_input_tokens > policy.max_estimated_input_tokens:
                raise ValueError("editorial judge input token budget exceeded")
            if result.estimated_output_tokens > policy.max_estimated_output_tokens:
                raise ValueError("editorial judge output token budget exceeded")
            return result
        except ValueError as exc:
            last_exc = exc
            if attempt == 0:
                continue
            break
        except Exception as exc:
            last_exc = exc
            break

    reason = last_exc.__class__.__name__ if last_exc is not None else "provider_error"
    return EditorialJudgeResult((), reason, "none", True, False)


def resolve_hybrid_labels(
    session: EditorialSession,
    result: EditorialJudgeResult,
    *,
    model_accept_confidence: float = 0.80,
) -> dict[str, str]:
    """Merge model semantics into local labels without surrendering safe fallback.

    Low-confidence/uncertain model classifications leave the local label untouched.
    The LLM does not create timestamps, clip IDs, or cross-session relationships.
    """
    resolved = {candidate.clip_id: candidate.local_label for candidate in session.candidates}
    if not result.available:
        return resolved

    for decision in result.decisions:
        if decision.confidence < model_accept_confidence or decision.label == "uncertain":
            continue
        resolved[decision.clip_id] = decision.label
    return resolved
