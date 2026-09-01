"""Trim an obvious repeated restart from inside one otherwise useful take.

ASR can keep a failed restart inside the same candidate as valid speech. Best-Take
selection cannot solve that because there is only one candidate. This stage removes
only a trailing exact multi-word restart when word timing and either a nearby physical
reset or an immediate following take corroborate that the creator restarted the idea.
Intentional rhetorical repetition fails open unless that recording-process structure is
present.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_RESTART_LEADS = frozenset({"and", "but", "so", "if", "okay", "ok", "well", "then"})
_RESET_KINDS = frozenset({"body_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})


def _norm(text: str) -> str:
    found = _TOKEN_RE.findall(str(text or "").casefold())
    return found[0] if found else ""


def _tokens(take: CandidateTake) -> tuple[str, ...]:
    if take.words:
        return tuple(_norm(word.text) for word in take.words if _norm(word.text))
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(take.text or "")))


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _has_restart_break(
    take: CandidateTake,
    restart_time: float,
    context: WholeVideoContext | None,
) -> bool:
    events = tuple(
        event for event in _source_events(context, take.source_asset_id)
        if event.end >= restart_time - 0.45 and event.start <= take.end + 0.25
    )
    body_reset = any(event.kind in _RESET_KINDS and event.confidence >= 0.90 for event in events)
    camera_or_face = any(event.kind in _BREAK_KINDS and event.confidence >= 0.78 for event in events)
    return body_reset or camera_or_face


def _following_take(
    take: CandidateTake,
    original: tuple[CandidateTake, ...],
    *,
    maximum_gap_sec: float = 1.75,
) -> CandidateTake | None:
    later = [
        other for other in original
        if other.clip_id != take.clip_id
        and other.source_asset_id == take.source_asset_id
        and other.start >= take.end - 0.02
    ]
    if not later:
        return None
    following = min(later, key=lambda item: (item.start, item.end, item.clip_id))
    gap = following.start - take.end
    if not -0.02 <= gap <= maximum_gap_sec:
        return None
    if following.duration_sec < 1.0 or len(_tokens(following)) < 4:
        return None
    return following


def _trailing_repeat_start(take: CandidateTake) -> tuple[int, int] | None:
    """Return ``(trim_start_index, repeat_width)`` for a repeated trailing phrase."""
    tokens = _tokens(take)
    if len(tokens) < 10:
        return None

    for width in range(min(8, len(tokens) // 2), 3, -1):
        for first in range(0, len(tokens) - (width * 2) + 1):
            phrase = tokens[first : first + width]
            if len(set(phrase)) < 3:
                continue
            for second in range(first + width, len(tokens) - width + 1):
                if tokens[second : second + width] != phrase:
                    continue
                # The repeated occurrence must be the tail or almost the tail. This is
                # the structure of a restart leak, not ordinary repetition in a story.
                if len(tokens) - (second + width) > 1:
                    continue
                trim_start = second
                back = second - 1
                while back >= 0 and second - back <= 2 and tokens[back] in _RESTART_LEADS:
                    trim_start = back
                    back -= 1
                if trim_start < 4:
                    continue
                return trim_start, width
    return None


def trim_internal_repeated_restarts(
    kept: Iterable[CandidateTake],
    original_takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    original = tuple(sorted(original_takes, key=lambda item: (item.source_order, item.start, item.end)))
    output = []
    diagnostics = []

    for take in kept:
        found = _trailing_repeat_start(take)
        if found is None or not take.words:
            output.append(take)
            continue
        trim_index, repeat_width = found
        words = tuple(take.words)
        if trim_index >= len(words):
            output.append(take)
            continue
        restart_time = float(words[trim_index].start)
        following = _following_take(take, original)
        if not _has_restart_break(take, restart_time, context) and following is None:
            output.append(take)
            continue

        kept_words = words[:trim_index]
        if len(kept_words) < 4:
            output.append(take)
            continue
        new_end = float(kept_words[-1].end)
        if not take.start + 0.4 < new_end < take.end - 0.20:
            output.append(take)
            continue
        text = " ".join(str(word.text or "").strip() for word in kept_words).strip()
        if not text:
            output.append(take)
            continue

        child = replace(
            take,
            end=new_end,
            text=text,
            words=tuple(kept_words),
            signals=(replace(take.signals, end=new_end) if take.signals is not None else None),
        )
        output.append(child)
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "internal_trailing_repeated_restart_trim",
            "original_text": take.text,
            "kept_text": text,
            "repeat_width": repeat_width,
            "trim_start_word_index": trim_index,
            "following_clip_id": following.clip_id if following is not None else None,
        })

    return tuple(output), tuple(diagnostics)


def install_internal_repeat_trim() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_internal_repeat_trim", False):
        return

    def apply_with_internal_repeat_trim(takes, context=None):
        take_tuple = tuple(takes)
        kept, discarded, decisions = original(take_tuple, context)
        trimmed, diagnostics = trim_internal_repeated_restarts(kept, take_tuple, context)
        if not diagnostics:
            return kept, discarded, decisions
        extra = tuple(
            CleanCutDecision(
                clip_id=str(item["clip_id"]),
                keep=True,
                reason=str(item["reason"]),
                confidence=0.98,
            )
            for item in diagnostics
        )
        return trimmed, discarded, tuple(decisions) + extra

    apply_with_internal_repeat_trim._cutsell_internal_repeat_trim = True
    clean_cut.apply_clean_cut = apply_with_internal_repeat_trim
