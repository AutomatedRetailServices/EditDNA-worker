"""Conservative cleanup for visibly abandoned open-ended delivery."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[\w'’-]+", re.UNICODE)
_OPEN_ENDINGS = frozenset({
    "a", "an", "the", "to", "for", "with", "without", "because", "and", "but", "or",
    "of", "in", "on", "at", "from", "into", "about", "than", "that", "which", "who",
})
_AMBIGUOUS_AUX_ENDINGS = frozenset({
    "am", "are", "is", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
})
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


def _has_strong_reset_near_end(
    take: CandidateTake,
    context: WholeVideoContext | None,
    *,
    lookback_sec: float = 0.85,
) -> bool:
    start = max(take.start, take.end - lookback_sec)
    return any(
        event.kind in _RESET_KINDS
        and event.confidence >= 0.88
        and event.end >= start
        and event.start <= take.end + 0.15
        for event in _source_events(context, take.source_asset_id)
    )


def _lexical_churn(tokens: tuple[str, ...]) -> bool:
    """Detect strong within-take restart churn without treating ordinary emphasis as enough."""
    if len(tokens) < 7:
        return False
    stems = [token[:7] for token in tokens if len(token) >= 5]
    counts = Counter(stems)
    if any(count >= 3 for count in counts.values()):
        return True

    # Also catch near-adjacent repeated content words such as "perfectly perfect ... perfect"
    # while ignoring short function words and intentional two-word emphasis.
    for index, token in enumerate(tokens):
        if len(token) < 5:
            continue
        stem = token[:6]
        matches = sum(1 for other in tokens[max(0, index - 6): index + 7] if len(other) >= 5 and other[:6] == stem)
        if matches >= 3:
            return True
    return False


def _dangling_reason(take: CandidateTake, context: WholeVideoContext | None) -> str | None:
    tokens = _tokens(take.text)
    if len(tokens) < 4 or not _has_strong_reset_near_end(take, context):
        return None

    last = tokens[-1]
    if last in _OPEN_ENDINGS:
        return "dangling_open_ending_with_physical_reset"

    if last in _AMBIGUOUS_AUX_ENDINGS and _lexical_churn(tokens[:-1]):
        return "dangling_auxiliary_with_internal_restart_and_reset"

    return None


def apply_dangling_delivery_cleanup(
    kept: Iterable[CandidateTake],
    context: WholeVideoContext | None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    survivors = []
    removed = []
    diagnostics = []
    for take in tuple(kept):
        reason = _dangling_reason(take, context)
        if reason is None:
            survivors.append(take)
            continue
        removed.append(take)
        diagnostics.append({"clip_id": take.clip_id, "reason": reason, "text": take.text})
    return tuple(survivors), tuple(removed), tuple(diagnostics)


def install_dangling_delivery_cleanup() -> None:
    """Install contextual dangling-delivery cleanup after earlier clean-cut wrappers."""
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_dangling_delivery", False):
        return

    def apply_with_dangling_delivery(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, contextual_discarded, diagnostics = apply_dangling_delivery_cleanup(kept, context)
        if not contextual_discarded:
            return kept, discarded, decisions

        reason_by_id = {item["clip_id"]: item["reason"] for item in diagnostics}
        extra_decisions = tuple(
            CleanCutDecision(
                clip_id=take.clip_id,
                keep=False,
                reason=reason_by_id[take.clip_id],
                confidence=0.96,
            )
            for take in contextual_discarded
        )
        return (
            kept,
            tuple(discarded) + tuple(contextual_discarded),
            tuple(decisions) + extra_decisions,
        )

    apply_with_dangling_delivery._cutsell_dangling_delivery = True
    clean_cut.apply_clean_cut = apply_with_dangling_delivery
