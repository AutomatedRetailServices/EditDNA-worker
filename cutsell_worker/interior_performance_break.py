"""Remove strongly corroborated interior recording resets at safe speech gaps.

A talking-head creator can fumble physically inside an otherwise useful take: hands or
microphone reset, the face breaks delivery, and then the creator resumes. Whole-video
vision already sees these events, but edge-only trimming intentionally leaves them
inside the clip. This stage removes only the non-speech gap when local evidence is
multimodal and strong. It never cuts through a spoken word and fails open otherwise.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Iterable, Tuple

from .contracts import CandidateTake
from .whole_video_analysis import WholeVideoContext

_PHYSICAL_KINDS = frozenset({"hand_motion_reset_candidate", "body_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _child_id(take: CandidateTake, side: str, start: float, end: float) -> str:
    digest = hashlib.sha256(
        f"{take.clip_id}|interior-performance|{side}|{start:.3f}|{end:.3f}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{take.clip_id}__ip{side}{digest}"


def _text(words) -> str:
    return " ".join(str(word.text or "").strip() for word in words).strip()


def split_interior_performance_breaks(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    *,
    minimum_word_gap_sec: float = 0.18,
    evidence_radius_sec: float = 0.75,
    minimum_edge_margin_sec: float = 0.35,
) -> tuple[Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    output = []
    diagnostics = []

    for take in takes:
        words = tuple(sorted(take.words, key=lambda word: (float(word.start), float(word.end))))
        if len(words) < 4:
            output.append(take)
            continue

        events = tuple(_source_events(context, take.source_asset_id))
        best = None
        for index in range(len(words) - 1):
            left_word = words[index]
            right_word = words[index + 1]
            gap_start = float(left_word.end)
            gap_end = float(right_word.start)
            gap = gap_end - gap_start
            if gap < minimum_word_gap_sec:
                continue
            if gap_start <= take.start + minimum_edge_margin_sec:
                continue
            if gap_end >= take.end - minimum_edge_margin_sec:
                continue

            window_start = gap_start - evidence_radius_sec
            window_end = gap_end + evidence_radius_sec
            physical = [
                event for event in events
                if event.end >= window_start and event.start <= window_end
                and _kind(event.kind) in _PHYSICAL_KINDS and event.confidence >= 0.90
            ]
            breaks = [
                event for event in events
                if event.end >= window_start and event.start <= window_end
                and _kind(event.kind) in _BREAK_KINDS
                and event.confidence >= (0.72 if _kind(event.kind) == "facial_expression_shift_candidate" else 0.80)
            ]
            hand_count = sum(1 for event in physical if _kind(event.kind) == "hand_motion_reset_candidate")
            if len(physical) < 2 or not breaks or hand_count < 1:
                continue

            # Keep meaningful speech on both sides. This avoids turning a normal pause
            # near the beginning/end of a sentence into two microclips.
            if index + 1 < 2 or len(words) - (index + 1) < 2:
                continue

            score = (
                len(physical),
                len(breaks),
                max(event.confidence for event in physical),
                gap,
            )
            if best is None or score > best[0]:
                best = (score, index, gap_start, gap_end, physical, breaks)

        if best is None:
            output.append(take)
            continue

        _, index, gap_start, gap_end, physical, breaks = best
        left_words = words[: index + 1]
        right_words = words[index + 1 :]
        left_start = float(take.start)
        left_end = float(left_words[-1].end)
        right_start = float(right_words[0].start)
        right_end = float(take.end)
        if left_end <= left_start or right_end <= right_start:
            output.append(take)
            continue

        left = replace(
            take,
            clip_id=_child_id(take, "l", left_start, left_end),
            start=left_start,
            end=left_end,
            text=_text(left_words),
            words=left_words,
            signals=(replace(take.signals, start=left_start, end=left_end) if take.signals is not None else None),
        )
        right = replace(
            take,
            clip_id=_child_id(take, "r", right_start, right_end),
            start=right_start,
            end=right_end,
            text=_text(right_words),
            words=right_words,
            signals=(replace(take.signals, start=right_start, end=right_end) if take.signals is not None else None),
        )
        output.extend((left, right))
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "interior_multimodal_performance_break_split",
            "removed_gap_start": gap_start,
            "removed_gap_end": gap_end,
            "removed_gap_sec": round(gap_end - gap_start, 3),
            "physical_event_count": len(physical),
            "break_event_count": len(breaks),
            "left_clip_id": left.clip_id,
            "right_clip_id": right.clip_id,
        })

    return tuple(output), tuple(diagnostics)


def install_interior_performance_break_split() -> None:
    from . import temporal_editing

    original = temporal_editing.refine_takes_with_temporal_context
    if getattr(original, "_cutsell_interior_performance_break", False):
        return

    def refine_with_interior_breaks(takes, context, **kwargs):
        refined, diagnostics = original(takes, context, **kwargs)

        # After Best Take, preserve_clip_id=True means the logical winner identity is
        # already authoritative. Splitting that winner into fresh child IDs here used to
        # invalidate the group selection and could turn a valid draft into zero selected
        # clips (Benchmark #42). Post-Best-Take refinement may adjust physical edges, but
        # it must never replace the selected logical clip with new identities.
        if bool(kwargs.get("preserve_clip_id")):
            return refined, tuple(diagnostics)

        refined, interior_diagnostics = split_interior_performance_breaks(refined, context)
        return refined, tuple(diagnostics) + tuple(interior_diagnostics)

    refine_with_interior_breaks._cutsell_interior_performance_break = True
    temporal_editing.refine_takes_with_temporal_context = refine_with_interior_breaks
