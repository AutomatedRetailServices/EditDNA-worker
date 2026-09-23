"""Conservative cleanup for open-ended retry fragments before a fuller attempt."""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .take_grouping import retry_similarity
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_OPEN_AUX = frozenset({
    "am", "are", "is", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
})
_OPEN_FUNCTION = frozenset({"a", "an", "the", "to", "for", "with", "because", "and", "but", "or", "of", "in", "on", "at", "from", "into", "about", "that", "which", "who"})
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _strong_reset_near_end(take: CandidateTake, context: WholeVideoContext | None) -> bool:
    start = max(take.start, take.end - 0.9)
    return any(
        event.kind in _RESET_KINDS
        and event.confidence >= 0.90
        and event.end >= start
        and event.start <= take.end + 0.2
        for event in _source_events(context, take.source_asset_id)
    )


def _content_overlap(left: CandidateTake, right: CandidateTake) -> float:
    a = {token for token in _tokens(left.text) if len(token) >= 3}
    b = {token for token in _tokens(right.text) if len(token) >= 3}
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _same_idea(left: CandidateTake, right: CandidateTake) -> bool:
    return retry_similarity(left.text, right.text) >= 0.62 or _content_overlap(left, right) >= 0.72


def _is_open_ended(take: CandidateTake) -> bool:
    tokens = _tokens(take.text)
    if len(tokens) < 4:
        return False
    return tokens[-1] in _OPEN_AUX or tokens[-1] in _OPEN_FUNCTION


def _has_fuller_nearby_retry(
    take: CandidateTake,
    candidates: tuple[CandidateTake, ...],
    *,
    horizon_sec: float = 18.0,
) -> bool:
    words = _tokens(take.text)
    for other in candidates:
        if other.clip_id == take.clip_id or other.source_asset_id != take.source_asset_id:
            continue
        if other.start < take.end:
            continue
        if other.start - take.end > horizon_sec:
            continue
        if len(_tokens(other.text)) < len(words) + 3:
            continue
        if _same_idea(take, other):
            return True
    return False


def apply_incomplete_retry_suffix_cleanup(
    kept: Iterable[CandidateTake],
    context: WholeVideoContext | None = None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[CleanCutDecision, ...]]:
    kept_tuple = tuple(kept)
    survivors = []
    removed = []
    decisions = []
    for take in kept_tuple:
        should_remove = (
            take.duration_sec <= 7.0
            and _is_open_ended(take)
            and _strong_reset_near_end(take, context)
            and _has_fuller_nearby_retry(take, kept_tuple)
        )
        if not should_remove:
            survivors.append(take)
            continue
        removed.append(take)
        decisions.append(CleanCutDecision(
            clip_id=take.clip_id,
            keep=False,
            reason="open_ended_retry_before_fuller_attempt_with_reset",
            confidence=0.96,
        ))
    return tuple(survivors), tuple(removed), tuple(decisions)


def install_incomplete_retry_suffix_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_incomplete_retry_suffix", False):
        return

    def apply_with_incomplete_retry_suffix(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, removed, extra = apply_incomplete_retry_suffix_cleanup(kept, context)
        if not removed:
            return kept, discarded, decisions
        return kept, tuple(discarded) + tuple(removed), tuple(decisions) + tuple(extra)

    apply_with_incomplete_retry_suffix._cutsell_incomplete_retry_suffix = True
    clean_cut.apply_clean_cut = apply_with_incomplete_retry_suffix
