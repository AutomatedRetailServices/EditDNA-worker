"""Trim visible pre-roll/post-roll around spoken delivery boundaries.

Creators often finish a sentence, pause, visibly reset, lower a handheld microphone,
look away, or wait for the camera before beginning the next thought. Conversely a take
may begin with a short silent/setup beat before the first spoken word. This stage trims
only non-speech slack proven by word timing and recording-process visual evidence.
Spoken words are never removed here.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Tuple

from .contracts import CandidateTake
from .whole_video_analysis import WholeVideoContext

_AUTHORITATIVE_KINDS = frozenset({
    "unintentional_dead_air",
    "retry_setup",
    "searching_for_words",
    "false_start",
    "wrong_take",
    "breaking_character",
    "camera_adjustment",
})
_BODY_RESET_KINDS = frozenset({"body_reset_candidate"})
_HAND_RESET_KINDS = frozenset({"hand_motion_reset_candidate"})
_CAMERA_BREAK_KINDS = frozenset({"camera_disengagement_candidate"})
_FACE_BREAK_KINDS = frozenset({"facial_expression_shift_candidate"})
_TALKING_HEAD_STYLES = frozenset({"talking_head", "creator_raw", "head_talk", "head_talking", "yapping"})


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _source_context(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return None
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return source
    return None


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    source = _source_context(context, source_asset_id)
    return tuple(source.events) if source is not None else ()


def _talking_head_like(context: WholeVideoContext | None, source_asset_id: str) -> bool:
    source = _source_context(context, source_asset_id)
    if source is None:
        return False
    return _kind(source.dominant_style) in _TALKING_HEAD_STYLES


def _edge_cut_evidence(
    take: CandidateTake,
    start: float,
    end: float,
    context: WholeVideoContext | None,
) -> tuple[bool, tuple[str, ...], bool]:
    """Return ``(confirmed, evidence, safe_for_micro_slack)``.

    ``safe_for_micro_slack`` is intentionally stricter than ordinary edge evidence.
    It targets a proven non-speech delivery boundary where the creator has already
    completed the thought and visibly exits that delivery: hand/mic reset, strong body
    reset, or a combined camera/face break. A normal in-sentence gesture or glance must
    not create a frame-tight cut.
    """
    events = tuple(
        event for event in _source_events(context, take.source_asset_id)
        if event.end >= start - 0.20 and event.start <= end + 0.20
    )
    reasons = []
    authoritative = [
        event for event in events
        if _kind(event.kind) in _AUTHORITATIVE_KINDS and event.confidence >= 0.78
    ]
    if authoritative:
        strongest = max(authoritative, key=lambda item: item.confidence)
        reasons.append(f"event:{_kind(strongest.kind)}:{strongest.confidence:.2f}")
        return True, tuple(reasons), True

    body = [
        event for event in events
        if _kind(event.kind) in _BODY_RESET_KINDS and event.confidence >= 0.90
    ]
    hand = [
        event for event in events
        if _kind(event.kind) in _HAND_RESET_KINDS and event.confidence >= 0.90
    ]
    camera = [
        event for event in events
        if _kind(event.kind) in _CAMERA_BREAK_KINDS and event.confidence >= 0.80
    ]
    face = [
        event for event in events
        if _kind(event.kind) in _FACE_BREAK_KINDS and event.confidence >= 0.72
    ]

    hand_micro = len(hand) >= 2 or (len(hand) >= 1 and bool(camera or face))
    if hand_micro:
        reasons.append(f"hand_reset_cluster:{max(item.confidence for item in hand):.2f}")
        if camera:
            reasons.append(f"camera_disengagement:{max(item.confidence for item in camera):.2f}")
        if face:
            reasons.append(f"expression_break:{max(item.confidence for item in face):.2f}")
        return True, tuple(reasons), True

    if body:
        reasons.append(f"body_reset:{max(item.confidence for item in body):.2f}")
        return True, tuple(reasons), True

    if camera and face:
        reasons.append(f"camera_disengagement:{max(item.confidence for item in camera):.2f}")
        reasons.append(f"expression_break:{max(item.confidence for item in face):.2f}")
        return True, tuple(reasons), True
    return False, (), False


def trim_delivery_edge_slack(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    *,
    minimum_slack_sec: float = 0.32,
    micro_talking_head_slack_sec: float = 0.12,
    maximum_slack_sec: float = 5.0,
) -> tuple[Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    output = []
    diagnostics = []

    for take in takes:
        words = tuple(sorted(take.words, key=lambda word: (float(word.start), float(word.end))))
        if not words:
            output.append(take)
            continue

        new_start = float(take.start)
        new_end = float(take.end)
        reasons = []
        talking_head = _talking_head_like(context, take.source_asset_id)

        first_start = float(words[0].start)
        leading = first_start - new_start
        if micro_talking_head_slack_sec <= leading <= maximum_slack_sec:
            confirmed, evidence, micro_safe = _edge_cut_evidence(take, new_start, first_start, context)
            # Preserve normal pre-roll breathing/setup unless the slack reaches the ordinary
            # threshold. Human Gold here is about post-sentence visual debris; applying the
            # 120 ms micro rule to the beginning of a clip caused over-tight starts.
            threshold = minimum_slack_sec
            if confirmed and leading >= threshold:
                new_start = first_start
                reasons.append({
                    "action": "trim_leading_non_speech_setup",
                    "duration_sec": round(leading, 3),
                    "evidence": list(evidence),
                    "talking_head_micro_edge": False,
                })

        last_end = float(words[-1].end)
        trailing = new_end - last_end
        if micro_talking_head_slack_sec <= trailing <= maximum_slack_sec:
            confirmed, evidence, micro_safe = _edge_cut_evidence(take, last_end, new_end, context)
            threshold = micro_talking_head_slack_sec if talking_head and micro_safe else minimum_slack_sec
            if confirmed and trailing >= threshold:
                new_end = last_end
                reasons.append({
                    "action": "trim_trailing_non_speech_cut_signal",
                    "duration_sec": round(trailing, 3),
                    "evidence": list(evidence),
                    "talking_head_micro_edge": bool(talking_head and micro_safe),
                })

        if not reasons or new_end - new_start < 0.25:
            output.append(take)
            continue

        child = replace(
            take,
            start=new_start,
            end=new_end,
            signals=(replace(take.signals, start=new_start, end=new_end) if take.signals is not None else None),
        )
        output.append(child)
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "delivery_edge_non_speech_trim",
            "original_start": take.start,
            "original_end": take.end,
            "result_start": child.start,
            "result_end": child.end,
            "actions": reasons,
        })

    return tuple(output), tuple(diagnostics)


def install_delivery_edge_trim() -> None:
    """Run after whole-video temporal refinement and internal pause splitting."""
    from . import temporal_editing

    original = temporal_editing.refine_takes_with_temporal_context
    if getattr(original, "_cutsell_delivery_edge_trim", False):
        return

    def refine_with_delivery_edges(takes, context, **kwargs):
        refined, diagnostics = original(takes, context, **kwargs)
        refined, edge_diagnostics = trim_delivery_edge_slack(refined, context)
        return refined, tuple(diagnostics) + tuple(edge_diagnostics)

    refine_with_delivery_edges._cutsell_delivery_edge_trim = True
    temporal_editing.refine_takes_with_temporal_context = refine_with_delivery_edges
