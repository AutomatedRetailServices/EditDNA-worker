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
from .temporal_editing import harmful_events_for_take
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


def _source_context(
    context: WholeVideoContext | None,
    source_asset_id: str,
) -> tuple[tuple[str, str | float], ...]:
    """Return one compact whole-source narrative map for every bounded Hybrid chunk."""
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id != source_asset_id:
            continue
        summary = " ".join(str(source.summary or "").split())[:3600]
        return (
            ("summary", summary),
            ("creator_intent", str(source.creator_intent or "")[:500]),
            ("main_topic", str(source.main_topic or "")[:500]),
            ("product_or_subject", str(source.product_or_subject or "")[:500]),
            ("story_logic", str(source.story_logic or "")[:900]),
            ("edit_mode", str(source.edit_mode or "natural")),
            ("sales_intent", round(float(source.sales_intent), 4)),
        )
    return ()


def _failed_local_evidence(
    take: CandidateTake,
    context: WholeVideoContext | None,
) -> tuple[bool, tuple[str, ...]]:
    """Require an independent local signal before lowering the semantic delete floor.

    A medium-high Gemini ``failed`` label is useful but must not delete valid creator
    speech by itself. We only trust it below the historical 0.94 hard-delete floor when
    local temporal/visual perception independently observed a recording failure in the
    same take. This preserves the global fail-open rule while catching obvious retries
    such as a spoken false start followed by a visible reset.
    """
    reasons: list[str] = []
    for event in harmful_events_for_take(take, context, minimum_confidence=0.80):
        reasons.append(f"event:{event.kind}:{event.confidence:.2f}")

    signals = take.signals
    if signals is not None:
        if float(signals.visual_fumble) >= 0.72:
            reasons.append(f"visual_fumble:{float(signals.visual_fumble):.2f}")
        if float(signals.distraction_risk) >= 0.82:
            reasons.append(f"distraction_risk:{float(signals.distraction_risk):.2f}")
        if float(signals.expression_naturalness) <= 0.28:
            reasons.append(f"expression_naturalness:{float(signals.expression_naturalness):.2f}")
        if float(signals.gesture_naturalness) <= 0.28:
            reasons.append(f"gesture_naturalness:{float(signals.gesture_naturalness):.2f}")

    return bool(reasons), tuple(reasons)


def _chunks(items: Tuple[CandidateTake, ...], size: int) -> tuple[Tuple[CandidateTake, ...], ...]:
    if size <= 0:
        raise ValueError("hybrid session chunk size must be positive")
    return tuple(tuple(items[index : index + size]) for index in range(0, len(items), size))


def _editorial_session(
    members: Tuple[CandidateTake, ...],
    context: WholeVideoContext | None,
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
        source_context=_source_context(context, source_id),
    )


def apply_hybrid_session_cleanup(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    editorial_judge: EditorialJudge | None,
    *,
    policy: HybridGatePolicy = HybridGatePolicy(),
    delete_confidence: float = 0.94,
    corroborated_failed_confidence: float = 0.84,
    chunk_size: int = 6,
) -> HybridSessionCleanupResult:
    """Classify bounded session chunks with the stable six-candidate envelope.

    Direct semantic deletion remains conservative at ``delete_confidence``. A ``failed``
    decision may also delete at the lower corroborated threshold only when independent
    local performance evidence confirms the failure. BTS remains on the stricter floor.
    Every chunk receives read-only whole-source context so semantic decisions can respect
    the complete message/story rather than reasoning from six isolated fragments.
    """
    take_tuple = tuple(takes)
    if not take_tuple or editorial_judge is None:
        return HybridSessionCleanupResult(take_tuple, (), 0, 0, ())

    take_map = {take.clip_id: take for take in take_tuple}
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
                context,
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
                    take = take_map[decision.clip_id]
                    corroborated, local_reasons = _failed_local_evidence(take, context)
                    hard_semantic_delete = bool(
                        decision.label in {"failed", "bts"}
                        and decision.confidence >= delete_confidence
                    )
                    corroborated_failed_delete = bool(
                        decision.label == "failed"
                        and decision.confidence >= corroborated_failed_confidence
                        and corroborated
                    )
                    applied_delete = hard_semantic_delete or corroborated_failed_delete
                    if hard_semantic_delete:
                        delete_basis = "high_confidence_semantic"
                    elif corroborated_failed_delete:
                        delete_basis = "semantic_failed_plus_local_performance"
                    else:
                        delete_basis = "kept_fail_open"
                    if applied_delete:
                        deleted_ids.add(decision.clip_id)
                        chunk_deleted.append(decision.clip_id)
                    decisions.append({
                        "clip_id": decision.clip_id,
                        "label": decision.label,
                        "confidence": decision.confidence,
                        "reason_code": decision.reason_code,
                        "local_failure_corroborated": corroborated,
                        "local_failure_reasons": list(local_reasons),
                        "delete_basis": delete_basis,
                        "applied_delete": applied_delete,
                    })

            diagnostics.append({
                "partition_index": partition_index,
                "chunk_index": chunk_index,
                "session_id": session.session_id,
                "member_ids": [member.clip_id for member in members],
                "source_context_available": bool(session.source_context),
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
