"""Cost-efficient semantic cleanup across bounded creator mini-sessions.

Instead of paying once per retry group/singleton, this stage batches the candidates that
belong to one inferred creator mini-session. The model judges every candidate against a
compact whole-source transcript and overlapping local windows so a failed attempt and a
later clean retake are less likely to fall into unrelated semantic calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
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

_TOKEN_RE = re.compile(r"[\w'’-]+", re.UNICODE)
_RESET_CANDIDATES = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_CANDIDATES = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})


@dataclass(frozen=True)
class HybridSessionCleanupResult:
    kept: Tuple[CandidateTake, ...]
    deleted: Tuple[CandidateTake, ...]
    requested_chunk_count: int
    available_chunk_count: int
    diagnostics: tuple[dict, ...]
    semantic_decisions: tuple[tuple[str, str, float], ...] = ()


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(str(text or "")))


def _normalized_tokens(text: str) -> frozenset[str]:
    return frozenset(token.lower() for token in _TOKEN_RE.findall(str(text or "")) if len(token) >= 2)


def _semantic_overlap(left: CandidateTake, right: CandidateTake) -> float:
    left_tokens = _normalized_tokens(left.text)
    right_tokens = _normalized_tokens(right.text)
    if len(left_tokens) < 3 or len(right_tokens) < 3:
        return 0.0
    shared = len(left_tokens & right_tokens)
    return shared / max(1, min(len(left_tokens), len(right_tokens)))


def _later_semantic_retry_replacement(
    failed_take: CandidateTake,
    members: Tuple[CandidateTake, ...],
    decisions_by_id: dict[str, tuple[str, float]],
    *,
    minimum_label_confidence: float = 0.68,
    minimum_overlap: float = 0.50,
    maximum_delay_sec: float = 24.0,
) -> tuple[CandidateTake | None, float]:
    """Find a later complete retake of substantially the same spoken idea.

    This is not general semantic deletion. It only corroborates a Hybrid `failed`
    decision when the same bounded creator session contains a later complete delivery
    with strong lexical/semantic overlap. That closes the common case where the failed
    first take has no obvious visual fumble but the creator immediately records a clean
    replacement.
    """
    best: CandidateTake | None = None
    best_overlap = 0.0
    for candidate in members:
        if candidate.clip_id == failed_take.clip_id:
            continue
        if candidate.source_asset_id != failed_take.source_asset_id:
            continue
        if float(candidate.start) <= float(failed_take.end):
            continue
        if float(candidate.start) - float(failed_take.end) > maximum_delay_sec:
            continue
        if not bool(candidate.complete_idea):
            continue
        label, confidence = decisions_by_id.get(candidate.clip_id, ("", 0.0))
        if label not in {"winner", "alternate", "keep"} or confidence < minimum_label_confidence:
            continue
        overlap = _semantic_overlap(failed_take, candidate)
        if overlap >= minimum_overlap and overlap > best_overlap:
            best = candidate
            best_overlap = overlap
    return best, best_overlap


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _performance_event_summary(take: CandidateTake, context: WholeVideoContext | None) -> dict[str, int | float | bool]:
    events = tuple(
        event for event in _source_events(context, take.source_asset_id)
        if event.end >= take.start - 0.20 and event.start <= take.end + 0.20
    )
    resets = [event for event in events if str(event.kind) in _RESET_CANDIDATES and event.confidence >= 0.88]
    breaks = [event for event in events if str(event.kind) in _BREAK_CANDIDATES and event.confidence >= 0.76]
    return {
        "strong_reset_count": len(resets),
        "strong_break_count": len(breaks),
        "max_reset_confidence": round(max((float(event.confidence) for event in resets), default=0.0), 4),
        "max_break_confidence": round(max((float(event.confidence) for event in breaks), default=0.0), 4),
        "multimodal_reset": bool(resets and breaks),
    }


def _evidence(take: CandidateTake, context: WholeVideoContext | None) -> tuple[tuple[str, float | str | bool | int], ...]:
    performance = _performance_event_summary(take, context)
    signals = take.signals
    base: list[tuple[str, float | str | bool | int]] = [
        ("complete_idea", bool(take.complete_idea)),
        *(performance.items()),
    ]
    if signals is None:
        return tuple(base)
    base.extend((
        ("audio_quality", round(float(signals.audio_quality), 4)),
        ("eye_contact", round(float(signals.eye_contact), 4)),
        ("visual_fumble", round(float(signals.visual_fumble), 4)),
        ("expression_naturalness", round(float(signals.expression_naturalness), 4)),
        ("gesture_naturalness", round(float(signals.gesture_naturalness), 4)),
        ("delivery_energy", round(float(signals.delivery_energy), 4)),
        ("distraction_risk", round(float(signals.distraction_risk), 4)),
    ))
    return tuple(base)


def _source_context(
    context: WholeVideoContext | None,
    source_asset_id: str,
) -> tuple[tuple[str, str | float], ...]:
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
    """Independent Watch+Listen evidence that a take belongs to recording failure."""
    reasons: list[str] = []
    for event in harmful_events_for_take(take, context, minimum_confidence=0.80):
        reasons.append(f"event:{event.kind}:{event.confidence:.2f}")

    performance = _performance_event_summary(take, context)
    reset_count = int(performance["strong_reset_count"])
    break_count = int(performance["strong_break_count"])
    if reset_count >= 2 and break_count >= 1:
        reasons.append(f"multimodal_reset_cluster:{reset_count}:{break_count}")
    elif reset_count >= 4:
        reasons.append(f"dense_physical_reset:{reset_count}")

    signals = take.signals
    if signals is not None:
        if float(signals.visual_fumble) >= 0.68:
            reasons.append(f"visual_fumble:{float(signals.visual_fumble):.2f}")
        if float(signals.distraction_risk) >= 0.78:
            reasons.append(f"distraction_risk:{float(signals.distraction_risk):.2f}")
        if float(signals.expression_naturalness) <= 0.32:
            reasons.append(f"expression_naturalness:{float(signals.expression_naturalness):.2f}")
        if float(signals.gesture_naturalness) <= 0.32:
            reasons.append(f"gesture_naturalness:{float(signals.gesture_naturalness):.2f}")

    return bool(reasons), tuple(reasons)


def _overlapping_windows(
    items: Tuple[CandidateTake, ...],
    *,
    size: int,
    stride: int,
) -> tuple[Tuple[CandidateTake, ...], ...]:
    if size <= 0 or stride <= 0:
        raise ValueError("hybrid session window size/stride must be positive")
    if len(items) <= size:
        return (items,) if items else ()
    starts = list(range(0, max(1, len(items) - size + 1), stride))
    final_start = len(items) - size
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return tuple(tuple(items[start : start + size]) for start in starts)


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
            evidence=_evidence(member, context),
        ) for member in members),
        local_confidence=0.50,
        conflict_score=0.50,
        task="classify_recording_process_within_single_creator_session",
        source_context=_source_context(context, source_id),
    )


def _decision_priority(label: str, confidence: float) -> tuple[int, float]:
    order = {"failed": 5, "bts": 5, "winner": 4, "alternate": 3, "keep": 2, "uncertain": 1}
    return order.get(str(label), 0), float(confidence)


def apply_hybrid_session_cleanup(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    editorial_judge: EditorialJudge | None,
    *,
    policy: HybridGatePolicy = HybridGatePolicy(),
    delete_confidence: float = 0.94,
    corroborated_failed_confidence: float = 0.82,
    corroborated_bts_confidence: float = 0.84,
    micro_failed_confidence: float = 0.80,
    clustered_bts_confidence: float = 0.84,
    retry_replaced_failed_confidence: float = 0.84,
    chunk_size: int = 10,
    chunk_stride: int = 5,
) -> HybridSessionCleanupResult:
    """Classify overlapping creator-session windows while failing open on uncertainty."""
    take_tuple = tuple(takes)
    if not take_tuple or editorial_judge is None:
        return HybridSessionCleanupResult(take_tuple, (), 0, 0, (), ())

    take_map = {take.clip_id: take for take in take_tuple}
    partitions = partition_takes_by_sessions(take_tuple, context)
    if not partitions:
        partitions = (take_tuple,)

    deleted_ids: set[str] = set()
    requested_chunks = 0
    available_chunks = 0
    diagnostics = []
    best_semantic: dict[str, tuple[str, float]] = {}

    effective_size = min(chunk_size, policy.max_candidates_per_request)
    effective_stride = min(max(1, chunk_stride), effective_size)

    for partition_index, partition in enumerate(partitions):
        ordered = tuple(sorted(partition, key=lambda item: (item.start, item.end, item.clip_id)))
        windows = _overlapping_windows(ordered, size=effective_size, stride=effective_stride)
        for chunk_index, members in enumerate(windows):
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
            local_by_id: dict[str, tuple[bool, tuple[str, ...]]] = {}
            if result.available:
                decisions_by_id = {
                    decision.clip_id: (str(decision.label), float(decision.confidence))
                    for decision in result.decisions
                }
                for decision in result.decisions:
                    candidate = (decision.label, float(decision.confidence))
                    current = best_semantic.get(decision.clip_id)
                    if current is None or _decision_priority(*candidate) > _decision_priority(*current):
                        best_semantic[decision.clip_id] = candidate
                    local_by_id[decision.clip_id] = _failed_local_evidence(take_map[decision.clip_id], context)

                harmful = [
                    decision for decision in result.decisions
                    if decision.label in {"failed", "bts"} and decision.confidence >= 0.82
                ]
                corroborated_harmful = [
                    decision for decision in harmful
                    if local_by_id.get(decision.clip_id, (False, ()))[0]
                ]
                dense_semantic_failure_cluster = len(harmful) >= 3 and len(corroborated_harmful) >= 2

                for decision in result.decisions:
                    take = take_map[decision.clip_id]
                    corroborated, local_reasons = local_by_id[decision.clip_id]
                    replacement, replacement_overlap = _later_semantic_retry_replacement(
                        take, members, decisions_by_id
                    ) if decision.label == "failed" else (None, 0.0)
                    retry_replaced_failed_delete = bool(
                        decision.label == "failed"
                        and decision.confidence >= retry_replaced_failed_confidence
                        and replacement is not None
                    )
                    hard_semantic_delete = bool(
                        decision.label in {"failed", "bts"}
                        and decision.confidence >= delete_confidence
                    )
                    corroborated_failed_delete = bool(
                        decision.label == "failed"
                        and decision.confidence >= corroborated_failed_confidence
                        and corroborated
                    )
                    corroborated_bts_delete = bool(
                        decision.label == "bts"
                        and decision.confidence >= corroborated_bts_confidence
                        and corroborated
                    )
                    micro_failed_delete = bool(
                        decision.label == "failed"
                        and decision.confidence >= micro_failed_confidence
                        and corroborated
                        and take.duration_sec <= 1.25
                        and _token_count(take.text) <= 2
                    )
                    clustered_bts_delete = bool(
                        decision.label == "bts"
                        and decision.confidence >= clustered_bts_confidence
                        and dense_semantic_failure_cluster
                    )
                    applied_delete = (
                        hard_semantic_delete
                        or retry_replaced_failed_delete
                        or corroborated_failed_delete
                        or corroborated_bts_delete
                        or micro_failed_delete
                        or clustered_bts_delete
                    )
                    if hard_semantic_delete:
                        delete_basis = "high_confidence_semantic"
                    elif retry_replaced_failed_delete:
                        delete_basis = "semantic_failed_plus_later_overlapping_complete_retake"
                    elif micro_failed_delete:
                        delete_basis = "micro_failed_plus_local_performance"
                    elif corroborated_failed_delete:
                        delete_basis = "semantic_failed_plus_local_performance"
                    elif corroborated_bts_delete:
                        delete_basis = "semantic_bts_plus_local_performance"
                    elif clustered_bts_delete:
                        delete_basis = "semantic_bts_inside_corroborated_failure_cluster"
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
                        "later_retry_replacement_id": replacement.clip_id if replacement is not None else None,
                        "later_retry_semantic_overlap": round(float(replacement_overlap), 4),
                        "dense_semantic_failure_cluster": dense_semantic_failure_cluster,
                        "delete_basis": delete_basis,
                        "applied_delete": applied_delete,
                    })

            diagnostics.append({
                "partition_index": partition_index,
                "chunk_index": chunk_index,
                "session_id": session.session_id,
                "member_ids": [member.clip_id for member in members],
                "source_context_available": bool(session.source_context),
                "window_size": len(members),
                "window_stride": effective_stride,
                "requested": bool(result.requested),
                "available": bool(result.available),
                "provider": result.provider,
                "model": result.model,
                "deleted_ids": chunk_deleted,
                "decisions": decisions,
            })

    kept = tuple(take for take in take_tuple if take.clip_id not in deleted_ids)
    deleted = tuple(take for take in take_tuple if take.clip_id in deleted_ids)
    semantic_decisions = tuple(
        (clip_id, label, confidence)
        for clip_id, (label, confidence) in best_semantic.items()
    )
    return HybridSessionCleanupResult(
        kept=kept,
        deleted=deleted,
        requested_chunk_count=requested_chunks,
        available_chunk_count=available_chunks,
        diagnostics=tuple(diagnostics),
        semantic_decisions=semantic_decisions,
    )
