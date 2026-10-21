"""Remove kept fragments trapped inside a proven failed-retry envelope.

Clean Cut can correctly discard several failed attempts while one short middle fragment
remains fail-open. This module uses those already-discarded attempts as structural
context: a kept take is removable only when same-idea discarded retries occur on both
sides within a tight time window, and the kept take is materially shorter or visibly
reset. It never promotes discarded material and never groups broad-topic speech.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .take_grouping import retry_similarity
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content_overlap(left: CandidateTake, right: CandidateTake) -> float:
    a = {token for token in _tokens(left.text) if len(token) >= 3}
    b = {token for token in _tokens(right.text) if len(token) >= 3}
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _same_retry_idea(left: CandidateTake, right: CandidateTake) -> bool:
    return retry_similarity(left.text, right.text) >= 0.62 or _content_overlap(left, right) >= 0.70


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _has_reset(take: CandidateTake, context: WholeVideoContext | None) -> bool:
    return any(
        event.kind in _RESET_KINDS
        and event.confidence >= 0.90
        and event.end >= take.start - 0.20
        and event.start <= take.end + 0.20
        for event in _source_events(context, take.source_asset_id)
    )


def apply_orphan_retry_cleanup(
    kept: Iterable[CandidateTake],
    discarded: Iterable[CandidateTake],
    context: WholeVideoContext | None = None,
    *,
    maximum_side_gap_sec: float = 12.0,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[CleanCutDecision, ...]]:
    kept_tuple = tuple(kept)
    discarded_tuple = tuple(discarded)
    survivors = []
    removed = []
    decisions = []

    for take in kept_tuple:
        words = _tokens(take.text)
        if not (4 <= len(words) <= 12 and take.duration_sec <= 6.0):
            survivors.append(take)
            continue

        before = []
        after = []
        for failed in discarded_tuple:
            if failed.source_asset_id != take.source_asset_id:
                continue
            if not _same_retry_idea(take, failed):
                continue
            if failed.end <= take.start:
                gap = take.start - failed.end
                if gap <= maximum_side_gap_sec:
                    before.append(failed)
            elif failed.start >= take.end:
                gap = failed.start - take.end
                if gap <= maximum_side_gap_sec:
                    after.append(failed)

        if not before or not after:
            survivors.append(take)
            continue

        related = tuple(before + after)
        longest = max((len(_tokens(item.text)) for item in related), default=0)
        materially_shorter = longest >= len(words) + 3
        if not materially_shorter and not _has_reset(take, context):
            survivors.append(take)
            continue

        removed.append(take)
        decisions.append(CleanCutDecision(
            clip_id=take.clip_id,
            keep=False,
            reason="orphan_fragment_inside_failed_retry_envelope",
            confidence=0.96,
        ))

    return tuple(survivors), tuple(removed), tuple(decisions)


def install_orphan_retry_cleanup() -> None:
    """Wrap clean_cut so deterministic discarded retries can invalidate one orphan fragment."""
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_orphan_retry_cleanup", False):
        return

    def apply_with_orphan_retry_cleanup(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, orphaned, orphan_decisions = apply_orphan_retry_cleanup(kept, discarded, context)
        if not orphaned:
            return kept, discarded, decisions
        return (
            kept,
            tuple(discarded) + tuple(orphaned),
            tuple(decisions) + tuple(orphan_decisions),
        )

    apply_with_orphan_retry_cleanup._cutsell_orphan_retry_cleanup = True
    clean_cut.apply_clean_cut = apply_with_orphan_retry_cleanup
