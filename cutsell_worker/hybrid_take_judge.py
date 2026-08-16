"""Hybrid Best Take adapter for real Flow B retry/mini-session groups.

The pipeline calls this only after grouping has been bounded by session walls. The
helper also exposes the exact bounded EditorialSession used by semantic cleanup so the
same evidence contract can drive BTS/failed removal and Best Take selection.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple

from .contracts import CandidateTake, RankedTake
from .hybrid_editorial import (
    EditorialCandidate,
    EditorialJudge,
    EditorialSession,
    HybridGatePolicy,
    safe_editorial_judge,
)
from .providers import ProviderStatus
from .take_judge import rank_takes
from .take_judge_provider import TakeJudgeProviderResult


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _signal_evidence(take: CandidateTake) -> tuple[tuple[str, float | str | bool], ...]:
    signals = take.signals
    if signals is None:
        return (("complete_idea", bool(take.complete_idea)),)
    return (
        ("complete_idea", bool(take.complete_idea)),
        ("audio_quality", round(float(signals.audio_quality), 4)),
        ("eye_contact", round(float(signals.eye_contact), 4)),
        ("framing_quality", round(float(signals.framing_quality), 4)),
        ("visual_fumble", round(float(signals.visual_fumble), 4)),
        ("expression_naturalness", round(float(signals.expression_naturalness), 4)),
        ("gesture_naturalness", round(float(signals.gesture_naturalness), 4)),
        ("delivery_energy", round(float(signals.delivery_energy), 4)),
        ("distraction_risk", round(float(signals.distraction_risk), 4)),
    )


def build_editorial_session_from_group(
    takes: Tuple[CandidateTake, ...],
    ranked: Tuple[RankedTake, ...] | None = None,
) -> EditorialSession:
    """Build one semantic request from one already-bounded retry/group partition."""
    if not takes:
        raise ValueError("hybrid take judge requires at least one candidate")
    take_by_id = {take.clip_id: take for take in takes}
    if len(take_by_id) != len(takes):
        raise ValueError("hybrid take judge received duplicate clip ids")

    ranking = tuple(ranked) if ranked is not None else rank_takes(takes)
    top_score = float(ranking[0].score) if ranking else 0.0
    second_score = float(ranking[1].score) if len(ranking) > 1 else 0.0
    gap = max(0.0, top_score - second_score)

    local_confidence = _clamp((0.55 * top_score) + (0.45 * min(1.0, gap * 4.0)))
    conflict_score = _clamp(max(0.0, 0.30 - gap) / 0.30)

    rank_by_id = {item.clip_id: item for item in ranking}
    candidates = []
    for take in takes:
        item = rank_by_id[take.clip_id]
        local_label = "winner" if take.clip_id == ranking[0].clip_id else "alternate"
        candidates.append(EditorialCandidate(
            clip_id=take.clip_id,
            text=take.text,
            start=take.start,
            end=take.end,
            local_label=local_label,
            local_confidence=_clamp(item.score),
            evidence=_signal_evidence(take),
        ))

    source_key = "|".join(sorted({take.source_asset_id for take in takes}))
    member_key = "|".join(sorted(take.clip_id for take in takes))
    session_id = "hs_" + hashlib.sha256(f"{source_key}|{member_key}".encode()).hexdigest()[:18]
    return EditorialSession(
        session_id=session_id,
        source_asset_id=takes[0].source_asset_id,
        candidates=tuple(candidates),
        local_confidence=local_confidence,
        conflict_score=conflict_score,
    )


# Backward-compatible internal name used by existing tests/callers.
def _session_from_group(takes: Tuple[CandidateTake, ...], ranked: Tuple[RankedTake, ...]) -> EditorialSession:
    return build_editorial_session_from_group(takes, ranked)


def _accepted_model_winner(session: EditorialSession, result, *, threshold: float) -> str | None:
    if not result.available:
        return None
    winners = [
        decision.clip_id
        for decision in result.decisions
        if decision.label == "winner" and decision.confidence >= threshold
    ]
    if len(winners) != 1:
        return None
    winner = winners[0]
    contradicted = any(
        decision.clip_id == winner
        and decision.label in {"failed", "bts"}
        and decision.confidence >= threshold
        for decision in result.decisions
    )
    return None if contradicted else winner


@dataclass(frozen=True)
class HybridTakeJudgeProvider:
    """TakeJudgeProvider that adds semantic Best Take help with local fallback."""

    editorial_judge: EditorialJudge | None = None
    policy: HybridGatePolicy = HybridGatePolicy()
    model_accept_confidence: float = 0.80

    def rank(self, takes: Tuple[CandidateTake, ...]) -> TakeJudgeProviderResult:
        baseline = rank_takes(takes)
        if len(takes) < 2:
            return TakeJudgeProviderResult(
                baseline,
                ProviderStatus("hybrid", False, True, "baseline_single_candidate"),
            )

        session = build_editorial_session_from_group(takes, baseline)
        result = safe_editorial_judge(self.editorial_judge, session, self.policy)
        winner = _accepted_model_winner(
            session,
            result,
            threshold=self.model_accept_confidence,
        )
        if winner is None:
            requested = bool(result.requested)
            status = "hybrid_fallback" if requested else "hybrid_local"
            reason = "model_unavailable_or_uncertain" if requested else "confidence_gate_local"
            return TakeJudgeProviderResult(
                baseline,
                ProviderStatus("hybrid", requested, True, status, reason),
            )

        by_id = {item.clip_id: item for item in baseline}
        chosen = by_id[winner]
        reranked = (
            RankedTake(chosen.clip_id, chosen.score, f"hybrid_editorial_winner:{chosen.reason}"),
            *(item for item in baseline if item.clip_id != winner),
        )
        return TakeJudgeProviderResult(
            reranked,
            ProviderStatus(
                provider=result.provider or "hybrid",
                requested=True,
                available=True,
                status="applied",
                reason=f"editorial_winner:{result.model or 'unknown'}",
            ),
        )
