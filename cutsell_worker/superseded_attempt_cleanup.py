"""Remove partial creator attempts only when a fuller same-idea retake exists later.

This targets the common talking-head pattern where a creator starts a sentence, stops,
then delivers the same idea cleanly. It is intentionally conservative: unique audience
content is preserved, and a fragment is removed only when a later take clearly covers
its meaningful content with a substantially fuller delivery.
"""
from __future__ import annotations

import re
from typing import Iterable

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "so", "that", "the", "this", "to", "was",
    "we", "with", "you", "your", "al", "como", "con", "cuando", "de", "del", "el", "en",
    "es", "la", "las", "le", "les", "lo", "los", "me", "mi", "mis", "o", "para", "pero",
    "por", "porque", "que", "se", "si", "sin", "su", "sus", "un", "una", "unos", "unas", "y", "yo",
})
_OPEN_TAIL = frozenset({
    "and", "but", "because", "for", "if", "of", "so", "that", "the", "to", "when", "with",
    "aunque", "como", "con", "cuando", "de", "del", "el", "la", "para", "pero", "por", "porque", "que", "si", "y",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> tuple[str, ...]:
    return tuple(token for token in _tokens(text) if len(token) >= 3 and token not in _STOP)


def _coverage(earlier: CandidateTake, later: CandidateTake) -> tuple[int, float]:
    a = set(_content(earlier.text))
    b = set(_content(later.text))
    if not a or not b:
        return 0, 0.0
    shared = len(a & b)
    return shared, shared / max(1, len(a))


def _repetition_pathology(text: str) -> bool:
    tokens = _content(text)
    if len(tokens) < 5:
        return False
    for index in range(len(tokens) - 2):
        token = tokens[index]
        if token == tokens[index + 1] == tokens[index + 2]:
            return True
    return False


def _open_micro_fragment(take: CandidateTake) -> bool:
    tokens = _tokens(take.text)
    if not tokens or take.duration_sec > 2.2 or len(tokens) > 4:
        return False
    if str(take.text or "").strip().endswith((".", "!", "?")):
        return False
    return tokens[-1] in _OPEN_TAIL


def _superseding_later_take(
    take: CandidateTake,
    ordered: tuple[CandidateTake, ...],
    *,
    maximum_lookahead_sec: float = 90.0,
) -> CandidateTake | None:
    candidates = []
    for other in ordered:
        if other.clip_id == take.clip_id or other.source_asset_id != take.source_asset_id:
            continue
        if other.start <= take.start:
            continue
        if other.start - take.end > maximum_lookahead_sec:
            continue
        shared, coverage = _coverage(take, other)
        if shared < 2 or coverage < 0.55:
            continue
        if other.duration_sec < max(5.0, take.duration_sec + 1.5):
            continue
        if len(_content(other.text)) < len(_content(take.text)) + 3:
            continue
        candidates.append((coverage, shared, other.duration_sec, other))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][-1]


def remove_superseded_attempts(
    kept: Iterable[CandidateTake],
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    ordered = tuple(sorted(kept, key=lambda item: (item.source_order, item.start, item.end, item.clip_id)))
    removed_ids: set[str] = set()
    diagnostics = []

    for take in ordered:
        later = _superseding_later_take(take, ordered)
        if later is None:
            # One especially safe micro case: an obviously grammatically open fragment
            # followed by a materially longer delivery very soon after it.
            if _open_micro_fragment(take):
                nearby = [
                    other for other in ordered
                    if other.source_asset_id == take.source_asset_id
                    and other.start > take.end
                    and 0.0 <= other.start - take.end <= 12.0
                    and other.duration_sec >= 5.0
                    and len(_content(other.text)) >= 5
                ]
                if nearby:
                    later = min(nearby, key=lambda item: item.start)
                else:
                    continue
            else:
                continue

        earlier_content = _content(take.text)
        short_partial = take.duration_sec <= 6.0 and len(earlier_content) <= 10
        repeated_broken_delivery = _repetition_pathology(take.text)
        incomplete = not bool(take.complete_idea)
        if not (short_partial or repeated_broken_delivery or incomplete or _open_micro_fragment(take)):
            continue

        removed_ids.add(take.clip_id)
        shared, coverage = _coverage(take, later)
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "superseded_partial_attempt",
            "superseding_clip_id": later.clip_id,
            "shared_content_tokens": shared,
            "earlier_content_coverage": round(coverage, 3),
            "earlier_text": take.text,
            "later_text": later.text,
        })

    survivors = tuple(take for take in ordered if take.clip_id not in removed_ids)
    removed = tuple(take for take in ordered if take.clip_id in removed_ids)
    return survivors, removed, tuple(diagnostics)


def install_superseded_attempt_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_superseded_attempt_cleanup", False):
        return

    def apply_with_superseded_attempts(takes, context: WholeVideoContext | None = None):
        kept, discarded, decisions = original(tuple(takes), context)
        kept, extra_removed, diagnostics = remove_superseded_attempts(kept)
        if not diagnostics:
            return kept, discarded, decisions
        extra = tuple(
            CleanCutDecision(
                clip_id=take.clip_id,
                keep=False,
                reason="superseded_partial_attempt",
                confidence=0.98,
            )
            for take in extra_removed
        )
        return kept, tuple(discarded) + extra_removed, tuple(decisions) + extra

    apply_with_superseded_attempts._cutsell_superseded_attempt_cleanup = True
    clean_cut.apply_clean_cut = apply_with_superseded_attempts
