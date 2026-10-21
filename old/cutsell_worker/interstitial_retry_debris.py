"""Remove tiny content-bearing debris trapped between same-idea retry attempts."""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_STOP = frozenset({"a","an","and","are","as","at","be","but","by","for","from","i","in","is","it","of","oh","on","or","so","that","the","this","to","we","with","you","your","okay","ok"})
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if len(t) >= 4 and t not in _STOP}


def _shares_content(micro: CandidateTake, other: CandidateTake) -> bool:
    content = _content_tokens(micro.text)
    if not content:
        return False
    return bool(content.intersection(_content_tokens(other.text)))


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _has_reset(take: CandidateTake, context: WholeVideoContext | None) -> bool:
    return any(
        e.kind in _RESET_KINDS
        and e.confidence >= 0.90
        and e.end >= take.start - 0.25
        and e.start <= take.end + 0.25
        for e in _source_events(context, take.source_asset_id)
    )


def apply_interstitial_retry_debris_cleanup(
    kept: Iterable[CandidateTake],
    discarded: Iterable[CandidateTake],
    context: WholeVideoContext | None = None,
    *,
    maximum_gap_sec: float = 4.0,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[CleanCutDecision, ...]]:
    kept_tuple = tuple(kept)
    discarded_tuple = tuple(discarded)
    all_candidates = tuple(sorted(kept_tuple + discarded_tuple, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    discarded_ids = {t.clip_id for t in discarded_tuple}
    survivors = []
    removed = []
    decisions = []

    for take in kept_tuple:
        tokens = _tokens(take.text)
        if not (1 <= len(tokens) <= 3 and take.duration_sec <= 2.5 and _content_tokens(take.text)):
            survivors.append(take)
            continue
        before = [
            c for c in all_candidates
            if c.clip_id != take.clip_id
            and c.source_asset_id == take.source_asset_id
            and c.end <= take.start
            and 0.0 <= take.start - c.end <= maximum_gap_sec
            and _shares_content(take, c)
        ]
        after = [
            c for c in all_candidates
            if c.clip_id != take.clip_id
            and c.source_asset_id == take.source_asset_id
            and c.start >= take.end
            and 0.0 <= c.start - take.end <= maximum_gap_sec
            and _shares_content(take, c)
        ]
        if not before or not after:
            survivors.append(take)
            continue
        if not any(c.clip_id in discarded_ids for c in before + after):
            survivors.append(take)
            continue
        if not _has_reset(take, context):
            survivors.append(take)
            continue

        removed.append(take)
        decisions.append(CleanCutDecision(
            clip_id=take.clip_id,
            keep=False,
            reason="content_microtake_inside_retry_envelope_with_reset",
            confidence=0.96,
        ))

    return tuple(survivors), tuple(removed), tuple(decisions)


def install_interstitial_retry_debris_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_interstitial_retry_debris", False):
        return

    def apply_with_interstitial_retry_debris(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, removed, extra = apply_interstitial_retry_debris_cleanup(kept, discarded, context)
        if not removed:
            return kept, discarded, decisions
        return kept, tuple(discarded) + tuple(removed), tuple(decisions) + tuple(extra)

    apply_with_interstitial_retry_debris._cutsell_interstitial_retry_debris = True
    clean_cut.apply_clean_cut = apply_with_interstitial_retry_debris
