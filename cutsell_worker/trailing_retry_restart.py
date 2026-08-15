"""Trim a repeated retry restart from the tail of an otherwise useful boundary clip.

A short ASR segment can straddle two editorial events: the final word(s) of a valid
sentence followed immediately by the beginning of an earlier retry. Deleting the
whole segment loses the valid boundary; keeping it leaks the restart. This module
uses word timing and strong sequence evidence from an earlier take to keep only the
small boundary prefix. It fails open when word timing or retry evidence is missing.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_LEADING_RESTART_FILLERS = frozenset({"if", "you", "so", "and", "but", "okay", "ok", "now"})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _comparison_tokens(text: str) -> tuple[str, ...]:
    """Normalize ASR punctuation/contractions only for retry-sequence comparison.

    Whisper can render the same speech as ``we're`` in one take and ``were`` in
    another. Removing apostrophes makes those orthographic variants comparable while
    the surrounding 3-4 exact-token requirement still protects ordinary transitions.
    """
    return tuple(token.replace("'", "") for token in _tokens(text))


def _word_tokens(take: CandidateTake) -> tuple[str, ...]:
    return tuple(_tokens(word.text)[0] for word in take.words if _tokens(word.text))


def _immediate_previous(
    take: CandidateTake,
    kept: tuple[CandidateTake, ...],
    *,
    maximum_gap_sec: float = 0.15,
) -> CandidateTake | None:
    prior = [
        other for other in kept
        if other.clip_id != take.clip_id
        and other.source_asset_id == take.source_asset_id
        and other.end <= take.start + 0.02
    ]
    if not prior:
        return None
    previous = max(prior, key=lambda item: (item.end, item.start))
    gap = take.start - previous.end
    if not -0.02 <= gap <= maximum_gap_sec:
        return None
    if previous.duration_sec < 1.0 or len(_tokens(previous.text)) < 4:
        return None
    return previous


def _retry_suffix_start(
    take: CandidateTake,
    previous: CandidateTake,
    original: tuple[CandidateTake, ...],
) -> int | None:
    tokens = _comparison_tokens(take.text)
    if not (4 <= len(tokens) <= 8 and take.duration_sec <= 2.0):
        return None

    earlier = [
        other for other in original
        if other.clip_id not in {take.clip_id, previous.clip_id}
        and other.source_asset_id == take.source_asset_id
        and other.end <= previous.start + 0.02
        and len(_comparison_tokens(other.text)) >= 6
    ]
    if not earlier:
        return None

    for keep_count in (1, 2):
        if len(tokens) - keep_count < 3:
            continue
        tail = tokens[keep_count:]
        for filler_count in range(0, min(2, len(tail) - 3) + 1):
            fillers = tail[:filler_count]
            if fillers and not all(token in _LEADING_RESTART_FILLERS for token in fillers):
                continue
            retry = tail[filler_count:]
            if len(retry) < 3:
                continue
            for other in earlier:
                other_tokens = _comparison_tokens(other.text)
                match_len = min(len(retry), len(other_tokens))
                if match_len < 3:
                    continue
                required = 4 if len(retry) >= 4 else 3
                if match_len >= required and retry[:required] == other_tokens[:required]:
                    return keep_count
    return None


def trim_trailing_retry_restarts(
    kept: Iterable[CandidateTake],
    original_takes: Iterable[CandidateTake],
) -> tuple[Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    kept_tuple = tuple(sorted(kept, key=lambda item: (item.source_order, item.start, item.end)))
    original = tuple(sorted(original_takes, key=lambda item: (item.source_order, item.start, item.end)))
    output = []
    diagnostics = []

    for take in kept_tuple:
        previous = _immediate_previous(take, kept_tuple)
        if previous is None or not take.words:
            output.append(take)
            continue
        keep_count = _retry_suffix_start(take, previous, original)
        if keep_count is None:
            output.append(take)
            continue

        word_tokens = _word_tokens(take)
        if len(word_tokens) < keep_count or word_tokens[:keep_count] != _tokens(take.text)[:keep_count]:
            output.append(take)
            continue
        kept_words = tuple(take.words[:keep_count])
        new_end = float(kept_words[-1].end)
        if not take.start < new_end < take.end - 0.15:
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
            "reason": "trailing_retry_restart_suffix_trim",
            "original_text": take.text,
            "kept_text": text,
            "previous_clip_id": previous.clip_id,
            "kept_word_count": keep_count,
        })

    return tuple(output), tuple(diagnostics)


def install_trailing_retry_restart_trim() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_trailing_retry_restart", False):
        return

    def apply_with_trailing_retry_restart(takes, context=None):
        take_tuple = tuple(takes)
        kept, discarded, decisions = original(take_tuple, context)
        trimmed, diagnostics = trim_trailing_retry_restarts(kept, take_tuple)
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

    apply_with_trailing_retry_restart._cutsell_trailing_retry_restart = True
    clean_cut.apply_clean_cut = apply_with_trailing_retry_restart
