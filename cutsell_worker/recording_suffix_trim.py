"""Trim recording self-critique suffixes only with multimodal physical corroboration."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, Word
from .whole_video_analysis import WholeVideoContext

_SUFFIX_PATTERNS = (
    re.compile(r"\bthat(?:'s| is) stupid[.!?]*\s*$", re.IGNORECASE),
    re.compile(r"\bthat was stupid[.!?]*\s*$", re.IGNORECASE),
    re.compile(r"\bthat(?:'s| is) dumb[.!?]*\s*$", re.IGNORECASE),
    re.compile(r"\bthat was dumb[.!?]*\s*$", re.IGNORECASE),
    re.compile(r"\bthat sounded bad[.!?]*\s*$", re.IGNORECASE),
    re.compile(r"\bthat was awkward[.!?]*\s*$", re.IGNORECASE),
)
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})
_TOKEN_RE = re.compile(r"[\w'’-]+", re.UNICODE)


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _suffix_word_start(take: CandidateTake) -> int | None:
    text = str(take.text or "")
    match = None
    for pattern in _SUFFIX_PATTERNS:
        found = pattern.search(text)
        if found is not None and (match is None or found.start() < match.start()):
            match = found
    if match is None:
        return None

    prefix_tokens = _TOKEN_RE.findall(text[: match.start()])
    all_tokens = _TOKEN_RE.findall(text)
    if len(prefix_tokens) < 4 or len(all_tokens) <= len(prefix_tokens):
        return None
    if len(take.words) != len(all_tokens):
        return None
    return len(prefix_tokens)


def _has_multimodal_suffix_break(
    take: CandidateTake,
    suffix_start_sec: float,
    context: WholeVideoContext | None,
) -> bool:
    start = max(take.start, suffix_start_sec - 0.65)
    end = take.end + 0.20
    events = tuple(
        event for event in _source_events(context, take.source_asset_id)
        if event.end >= start and event.start <= end
    )
    has_reset = any(event.kind in _RESET_KINDS and event.confidence >= 0.72 for event in events)
    has_break = any(event.kind in _BREAK_KINDS and event.confidence >= 0.72 for event in events)
    return has_reset and has_break


def _child_id(take: CandidateTake, end: float) -> str:
    digest = hashlib.sha256(
        f"{take.clip_id}|self-critique-suffix|{take.start:.3f}|{end:.3f}".encode("utf-8")
    ).hexdigest()[:14]
    return f"{take.clip_id}__sc{digest}"


def trim_visual_self_critique_suffixes(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    """Trim only recording-meta suffixes backed by reset + expression/camera break."""
    output = []
    diagnostics = []

    for take in takes:
        suffix_index = _suffix_word_start(take)
        if suffix_index is None:
            output.append(take)
            continue

        words = tuple(take.words)
        suffix_start_sec = words[suffix_index].start
        if not _has_multimodal_suffix_break(take, suffix_start_sec, context):
            output.append(take)
            continue

        kept_words = words[:suffix_index]
        if not kept_words:
            output.append(take)
            continue
        new_end = kept_words[-1].end
        if new_end - take.start < 0.30:
            output.append(take)
            continue

        new_text = " ".join(word.text for word in kept_words).strip()
        child = replace(
            take,
            clip_id=_child_id(take, new_end),
            end=new_end,
            text=new_text,
            words=kept_words,
            signals=(replace(take.signals, end=new_end) if take.signals is not None else None),
        )
        output.append(child)
        diagnostics.append({
            "original_clip_id": take.clip_id,
            "result_clip_id": child.clip_id,
            "action": "trim_self_critique_suffix",
            "original_text": take.text,
            "result_text": child.text,
            "result_end": child.end,
        })

    return tuple(output), tuple(diagnostics)


def install_visual_self_critique_suffix_trim() -> None:
    """Install suffix trimming before runtime modules import temporal refinement."""
    from . import temporal_editing

    original = temporal_editing.refine_takes_with_temporal_context
    if getattr(original, "_cutsell_visual_self_critique_suffix", False):
        return

    def refine_with_suffix_trim(takes, context, **kwargs):
        refined, diagnostics = original(takes, context, **kwargs)
        refined, suffix_diagnostics = trim_visual_self_critique_suffixes(refined, context)
        return refined, tuple(diagnostics) + tuple(suffix_diagnostics)

    refine_with_suffix_trim._cutsell_visual_self_critique_suffix = True
    temporal_editing.refine_takes_with_temporal_context = refine_with_suffix_trim
