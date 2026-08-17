"""Cost-efficient semantic cleanup across bounded creator mini-sessions.

Instead of paying once per retry group/singleton, this stage batches the candidates that
belong to one inferred creator mini-session. The model is asked only whether each item
is valid audience-facing speech (keep) or recording garbage (failed/BTS). Retry/Best
Take grouping still happens afterward in the deterministic brain.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Tuple

from .contracts import CandidateTake
from .hybrid_editorial import (
    EditorialCandidate,
    EditorialJudge,
    EditorialSession,
    HybridGatePolicy,
    safe_editorial_judge,
)
from .session_boundaries import partition_takes_by_sessions
from .whole_video_analysis import WholeVideoContext


@dataclass(frozen=True)
class HybridSessionCleanupResult:
    kept: Tuple[CandidateTake, ...]
    deleted: Tuple[CandidateTake, ...]
    requested_chunk_count: int
    available_chunk_count: int
    diagnostics: tuple[dict, ...]


def _evidence(take: CandidateTake) -> tuple[tuple[str, float | str | bool], ...]:
    signals = take.signals
    if signals is None:
        return (("complete_idea", bool(take.complete_idea)),)
    return (
        ("complete_idea", bool(take.complete_idea)),
        ("audio_quality", round(float(signals.audio_quality), 4)),
        ("eye_contact", round(float(signals.eye_contact), 4)),
        ("visual_fumble", round(float(signals.visual_fumble), 4)),
        ("expression_naturalness", round(float(signals.expression_naturalness), 4)),
        ("gesture_naturalness", round(float(signals.gesture_naturalness), 4)),
        ("delivery_energy", round(float(signals.delivery_energy), 4)),
        ("distraction_risk", round(float(signals.distraction_risk), 4)),
    )


def _chunks(items: Tuple[CandidateTake, ...], size: int) -> tuple[Tuple[CandidateTake, ...], ...]:
    if size <= 0:
        raise ValueError("hybrid session chunk size must be positive")
    return tuple(tuple(items[index : index + size]) for index in range(0, len(items), size))


def _editorial_session(
    members: Tuple[CandidateTake, ...],
    *,
    partition_index: int,
    chunk_index: int,
) -> EditorialSession:
    source_id = members[0].source_asset_id
    member_key = "|".join(member.clip_id for member in members)
    session_id = "hc_" + hashlib.sha256(
        f"{source_id}|{partition_index}|{chunk_index}|{member_key}".encode()
    ).hexdigest()[:18]
    return EditorialSession(
        session_id=session_id,
        source_asset_id=source_id,
        candidates=tuple(EditorialCandidate(
            clip_id=member.clip_id,
            text=member.text,
            start=member.start,
            end=member.end,
            local_label="keep",
            local_confidence=0.50,
            evidence=_evidence(member),
        ) for member in members),
        local_confidence=0.50,
        conflict_score=0.50,
        task="classify_recording_process_within_single_creator_session",
    )


def apply_hybrid_session_cleanup(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    editorial_judge: EditorialJudge | None,
    *,
    policy: HybridGatePolicy = HybridGatePolicy(),
    delete_confidence: float = 0.94,
    chunk_size: int = 12,
) -> HybridSessionCleanupResult:
    """Classify bounded session chunks with compact structured output.

    Run #27 proved that verbose 11-12 candidate responses could overrun the old
    structured-output budget, so Run #28 reduced chunks to six. Run #29 then proved the
    opposite bottleneck: six-candidate chunks created too many paid calls and exhausted
    the per-edit COGS guard after roughly five requests. The Gemini response contract is
    now compact (clip_id + label + confidence only), so twelve candidates restores
    cost-efficient coverage while staying below the 14-candidate hard request limit.
    """
    take_tuple = tuple(takes)
    if not take_tuple or editorial_judge is None:
        return HybridSessionCleanupResult(take_tuple, (), 0, 0, ())

    partitions = partition_takes_by_sessions(take_tuple, context)
    if not partitions:
        partitions = (take_tuple,)

    deleted_ids: set[str] = set()
    requested_chunks = 0
    available_chunks = 0
    diagnostics = []

    for partition_index, partition in enumerate(partitions):
        ordered = tuple(sorted(partition, key=lambda item: (item.start, item.end, item.clip_id)))
        for chunk_index, members in enumerate(_chunks(ordered, min(chunk_size, policy.max_candidates_per_request))):
            session = _editorial_session(
                members,
                partition_index=partition_index,
                chunk_index=chunk_index,
            )
            result = safe_editorial_judge(editorial_judge, session, policy)
            if result.requested:
                requested_chunks += 1
            if result.available:
                available_chunks += 1

            chunk_deleted = []
            decisions = []
            if result.available:
                for decision in result.decisions:
                    applied_delete = bool(
                        decision.label in {"failed", "bts"}
                        and decision.confidence >= delete_confidence
                    )
                    if applied_delete:
                        deleted_ids.add(decision.clip_id)
                        chunk_deleted.append(decision.clip_id)
                    decisions.append({
                        "clip_id": decision.clip_id,
                        "label": decision.label,
                        "confidence": decision.confidence,
                        "reason_code": decision.reason_code,
                        "applied_delete": applied_delete,
                    })

            diagnostics.append({
                "partition_index": partition_index,
                "chunk_index": chunk_index,
                "session_id": session.session_id,
                "member_ids": [member.clip_id for member in members],
                "requested": bool(result.requested),
                "available": bool(result.available),
                "provider": result.provider,
                "model": result.model,
                "deleted_ids": chunk_deleted,
                "decisions": decisions,
            })

    kept = tuple(take for take in take_tuple if take.clip_id not in deleted_ids)
    deleted = tuple(take for take in take_tuple if take.clip_id in deleted_ids)
    return HybridSessionCleanupResult(
        kept=kept,
        deleted=deleted,
        requested_chunk_count=requested_chunks,
        available_chunk_count=available_chunks,
        diagnostics=tuple(diagnostics),
    )
