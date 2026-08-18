"""Trim visible pre-roll/post-roll around spoken delivery boundaries.

Creators often finish a sentence, pause, visibly reset, or look away before the ASR
segment itself ends. Conversely a take may begin with a short silent/setup beat before
the first spoken word. This stage trims only non-speech slack proven by word timing and
recording-process visual evidence. Spoken words are never removed here.
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
_CAMERA_BREAK_KINDS = frozenset({"camera_disengagement_candidate"})
_FACE_BREAK_KINDS = frozenset({"facial_expression_shift_candidate"})


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _edge_has_cut_signal(
    take: CandidateTake,
    start: float,
    end: float,
    context: WholeVideoContext | None,
) -> tuple[bool, tuple[str, ...]]:
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
        return True, tuple(reasons)

    body = [
        event for event in events
        if _kind(event.kind) in _BODY_RESET_KINDS and event.confidence >= 0.90
    ]
    camera = [
        event for event in events
        if _kind(event.kind) in _CAMERA_BREAK_KINDS and event.confidence >= 0.80
    ]
    face = [
        event for event in events
        if _kind(event.kind) in _FACE_BREAK_KINDS and event.confidence >= 0.78
    ]
    # Body reset alone is strong enough at a proven non-speech edge. A face/camera
    # break needs both families so a normal glance or expression does not create a cut.
    if body:
        reasons.append(f"body_reset:{max(item.confidence for item in body):.2f}")
        return True, tuple(reasons)
    if camera and face:
        reasons.append(f"camera_disengagement:{max(item.confidence for item in camera):.2f}")
        reasons.append(f"expression_break:{max(item.confidence for item in face):.2f}")
        return True, tuple(reasons)
    return False, ()


def trim_delivery_edge_slack(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    *,
    minimum_slack_sec: float = 0.32,
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

        first_start = float(words[0].start)
        leading = first_start - new_start
        if minimum_slack_sec <= leading <= maximum_slack_sec:
            confirmed, evidence = _edge_has_cut_signal(take, new_start, first_start, context)
            if confirmed:
                new_start = first_start
                reasons.append({
                    "action": "trim_leading_non_speech_setup",
                    "duration_sec": round(leading, 3),
                    "evidence": list(evidence),
                })

        last_end = float(words[-1].end)
        trailing = new_end - last_end
        if minimum_slack_sec <= trailing <= maximum_slack_sec:
            confirmed, evidence = _edge_has_cut_signal(take, last_end, new_end, context)
            if confirmed:
                new_end = last_end
                reasons.append({
                    "action": "trim_trailing_non_speech_cut_signal",
                    "duration_sec": round(trailing, 3),
                    "evidence": list(evidence),
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
