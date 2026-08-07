"""Optional high-confidence judge for obvious recording mistakes.

This provider is deliberately separate from semantic/commercial classification. It
may only judge whether an already-segmented candidate is valid speech, a whole-take
recording mistake, or mixed speech that needs later word-safe trimming.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

from .contracts import CandidateTake
from .providers import ProviderStatus


@dataclass(frozen=True)
class CleanCutJudgement:
    clip_id: str
    action: str  # keep | delete | mixed
    confidence: float
    reason: str


@dataclass(frozen=True)
class CleanCutProviderResult:
    judgements: Tuple[CleanCutJudgement, ...]
    status: ProviderStatus


class CleanCutProvider(Protocol):
    def judge(self, takes: Tuple[CandidateTake, ...]) -> CleanCutProviderResult: ...


def safe_clean_cut_judge(
    provider: CleanCutProvider | None,
    takes: Tuple[CandidateTake, ...],
) -> CleanCutProviderResult:
    """Fail open: provider failure or malformed output keeps all speech."""
    if provider is None or not takes:
        return CleanCutProviderResult(
            (),
            ProviderStatus("none", False, False, "not_requested"),
        )
    try:
        result = provider.judge(takes)
        expected = {take.clip_id for take in takes}
        seen = set()
        normalized = []
        for item in result.judgements:
            if item.clip_id not in expected or item.clip_id in seen:
                raise ValueError("clean cut judge returned invalid clip id")
            action = str(item.action).lower().strip()
            if action not in {"keep", "delete", "mixed"}:
                raise ValueError("clean cut judge returned invalid action")
            confidence = float(item.confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("clean cut judge confidence outside 0..1")
            normalized.append(CleanCutJudgement(
                clip_id=item.clip_id,
                action=action,
                confidence=confidence,
                reason=str(item.reason or "")[:240],
            ))
            seen.add(item.clip_id)
        if seen != expected:
            raise ValueError("clean cut judge omitted candidates")
        return CleanCutProviderResult(tuple(normalized), result.status)
    except Exception as exc:
        return CleanCutProviderResult(
            (),
            ProviderStatus(
                provider=provider.__class__.__name__,
                requested=True,
                available=False,
                status="provider_error",
                reason=exc.__class__.__name__,
            ),
        )


def apply_provider_judgements(
    takes: Tuple[CandidateTake, ...],
    result: CleanCutProviderResult,
    *,
    delete_threshold: float = 0.94,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], tuple[dict, ...]]:
    """Apply only high-confidence whole-candidate deletes; keep mixed/uncertain."""
    judgement_by_id = {item.clip_id: item for item in result.judgements}
    kept, deleted, diagnostics = [], [], []
    for take in takes:
        item = judgement_by_id.get(take.clip_id)
        applied_delete = bool(
            item is not None
            and item.action == "delete"
            and item.confidence >= delete_threshold
        )
        if applied_delete:
            deleted.append(take)
        else:
            kept.append(take)
        if item is not None:
            diagnostics.append({
                "clip_id": take.clip_id,
                "action": item.action,
                "confidence": item.confidence,
                "reason": item.reason,
                "applied_delete": applied_delete,
            })
    return tuple(kept), tuple(deleted), tuple(diagnostics)
