"""Conservative trimming of broken internal self-correction tails.

This module does not treat negation as a generic editing signal.  It only trims a
very small tail when word timing shows a lexical self-correction pattern and the
next take begins essentially at the same source boundary, making the retained
prefix usable as a natural continuation.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_FILLERS = frozenset({"this", "that", "it", "uh", "um"})


def _norm(text: str) -> str:
    parts = _TOKEN_RE.findall(str(text or "").casefold())
    return parts[0] if parts else ""


def _word_tokens(take: CandidateTake) -> tuple[str, ...]:
    return tuple(_norm(word.text) for word in take.words if _norm(word.text))


def _correction_tail_start(take: CandidateTake) -> int | None:
    """Return first word index of a narrow broken correction tail, if present.

    Accepted structure near the end is roughly ``short-form [filler] not long-form``
    where short-form is a lexical prefix of long-form (for example ``med ... not
    medicine``).  This is intentionally narrower than generic ``not`` handling.
    """
    words = tuple(take.words)
    if len(words) < 8:
        return None
    tokens = tuple(_norm(word.text) for word in words)
    if not all(tokens):
        return None

    tail_floor = max(1, len(tokens) - 5)
    for not_index in range(tail_floor, len(tokens) - 1):
        if tokens[not_index] != "not":
            continue
        corrected = tokens[not_index + 1]
        if len(corrected) < 5:
            continue

        source_index = not_index - 1
        while source_index >= tail_floor and tokens[source_index] in _FILLERS:
            source_index -= 1
        if source_index < tail_floor:
            continue
        source = tokens[source_index]
        if len(source) < 3 or source == corrected:
            continue
        if not corrected.startswith(source):
            continue

        # Preserve a meaningful spoken prefix; do not reduce a short reaction to
        # an even smaller fragment.
        if source_index < 6:
            continue
        return source_index
    return None


def _contiguous_following(
    take: CandidateTake,
    ordered: tuple[CandidateTake, ...],
    *,
    maximum_gap_sec: float = 0.20,
) -> CandidateTake | None:
    later = [
        other for other in ordered
        if other.source_asset_id == take.source_asset_id and other.start >= take.end - 0.02
    ]
    if not later:
        return None
    following = min(later, key=lambda item: (item.start, item.end))
    gap = following.start - take.end
    if not -0.02 <= gap <= maximum_gap_sec:
        return None
    if not str(following.text or "").strip():
        return None
    return following


def trim_internal_self_corrections(
    kept: Iterable[CandidateTake],
    original_takes: Iterable[CandidateTake],
) -> tuple[Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    ordered = tuple(sorted(original_takes, key=lambda item: (item.source_order, item.start, item.end)))
    output = []
    diagnostics = []

    for take in kept:
        start_index = _correction_tail_start(take)
        if start_index is None or not take.words:
            output.append(take)
            continue
        following = _contiguous_following(take, ordered)
        if following is None:
            output.append(take)
            continue

        kept_words = tuple(take.words[:start_index])
        if len(kept_words) < 6:
            output.append(take)
            continue
        new_end = float(kept_words[-1].end)
        if new_end <= take.start + 0.3:
            output.append(take)
            continue
        text = " ".join(str(word.text or "").strip() for word in kept_words).strip()
        if not text:
            output.append(take)
            continue

        output.append(replace(
            take,
            end=new_end,
            text=text,
            words=kept_words,
            signals=(replace(take.signals, end=new_end) if take.signals is not None else None),
        ))
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "internal_self_correction_suffix_trim",
            "original_text": take.text,
            "kept_text": text,
            "following_clip_id": following.clip_id,
            "following_text": following.text,
            "trim_start_word_index": start_index,
        })

    return tuple(output), tuple(diagnostics)


def install_internal_self_correction_trim() -> None:
    """Install after other deterministic cleanup wrappers."""
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_internal_self_correction", False):
        return

    def apply_with_internal_self_correction(takes, context=None):
        take_tuple = tuple(takes)
        kept, discarded, decisions = original(take_tuple, context)
        trimmed, diagnostics = trim_internal_self_corrections(kept, take_tuple)
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

    apply_with_internal_self_correction._cutsell_internal_self_correction = True
    clean_cut.apply_clean_cut = apply_with_internal_self_correction
