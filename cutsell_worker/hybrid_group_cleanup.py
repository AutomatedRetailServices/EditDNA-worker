"""Semantic cleanup inside already-bounded Flow B groups.

The LLM may label intent but never edits timestamps. High-confidence ``failed``/``bts``
labels can remove candidates; one high-confidence ``winner`` may guide Best Take.
Malformed, unavailable, uncertain, or low-confidence results fail open to the local
pipeline. This stage runs only after session-scoped retry grouping, so it never creates
cross-creator relationships itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .contracts import CandidateTake
from .hybrid_editorial import EditorialJudge, HybridGatePolicy, safe_editorial_judge
from .hybrid_take_judge import build_editorial_session_from_group
from .take_judge import rank_takes


@dataclass(frozen=True)
class HybridGroupCleanupResult:
    kept: Tuple[CandidateTake, ...]
    deleted: Tuple[CandidateTake, ...]
    preferred_winner_id: str | None
    requested: bool
    available: bool
    provider: str
    model: str
    diagnostics: tuple[dict, ...]


def apply_hybrid_group_cleanup(
    members: Tuple[CandidateTake, ...],
    editorial_judge: EditorialJudge | None,
    *,
    policy: HybridGatePolicy = HybridGatePolicy(),
    delete_confidence: float = 0.94,
    winner_confidence: float = 0.80,
) -> HybridGroupCleanupResult:
    if not members:
        return HybridGroupCleanupResult((), (), None, False, False, "none", "none", ())

    baseline = rank_takes(members)
    session = build_editorial_session_from_group(members, baseline)
    result = safe_editorial_judge(editorial_judge, session, policy)
    if not result.available:
        return HybridGroupCleanupResult(
            kept=members,
            deleted=(),
            preferred_winner_id=None,
            requested=bool(result.requested),
            available=False,
            provider=result.provider,
            model=result.model,
            diagnostics=(),
        )

    by_id = {member.clip_id: member for member in members}
    delete_ids = {
        decision.clip_id
        for decision in result.decisions
        if decision.label in {"failed", "bts"}
        and decision.confidence >= delete_confidence
    }
    kept = tuple(member for member in members if member.clip_id not in delete_ids)
    deleted = tuple(member for member in members if member.clip_id in delete_ids)

    winner_ids = [
        decision.clip_id
        for decision in result.decisions
        if decision.label == "winner"
        and decision.confidence >= winner_confidence
        and decision.clip_id not in delete_ids
    ]
    preferred_winner_id = winner_ids[0] if len(winner_ids) == 1 else None

    diagnostics = tuple({
        "clip_id": decision.clip_id,
        "label": decision.label,
        "confidence": decision.confidence,
        "reason_code": decision.reason_code,
        "applied_delete": decision.clip_id in delete_ids,
        "preferred_winner": decision.clip_id == preferred_winner_id,
    } for decision in result.decisions)

    # A valid all-BTS/all-failed group may legitimately disappear. Otherwise invalid
    # or low-confidence output already failed open above/through the thresholds.
    return HybridGroupCleanupResult(
        kept=kept,
        deleted=deleted,
        preferred_winner_id=preferred_winner_id,
        requested=bool(result.requested),
        available=True,
        provider=result.provider,
        model=result.model,
        diagnostics=diagnostics,
    )
