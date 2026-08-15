"""Conservative cleanup for repeated restarts ending in soft frustration.

Phrases such as ``oh my god`` or ``oh my goodness`` are valid creator reactions and
are never destructive on their own.  They become cleanup evidence only when the same
take also contains a repeated multi-word phrase and local visual evidence shows both
physical reset and expression/engagement break families.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_SOFT_FRUSTRATION_RE = re.compile(
    r"\boh\s+my\s+god\b|\boh\s+my\s+goodness(?:\s+gracious)?\b",
    re.IGNORECASE,
)
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _has_repeated_multiword_phrase(text: str) -> bool:
    tokens = _tokens(text)
    if len(tokens) < 6:
        return False
    for width in range(min(6, len(tokens) // 2), 1, -1):
        seen: set[tuple[str, ...]] = set()
        for index in range(0, len(tokens) - width + 1):
            gram = tokens[index:index + width]
            if len(set(gram)) < 2:
                continue
            if gram in seen:
                return True
            seen.add(gram)
    return False


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _has_multimodal_break(take: CandidateTake, context: WholeVideoContext | None) -> bool:
    events = tuple(
        event for event in _source_events(context, take.source_asset_id)
        if event.end >= take.start - 0.35 and event.start <= take.end + 0.35
    )
    has_reset = any(event.kind in _RESET_KINDS and event.confidence >= 0.90 for event in events)
    has_break = any(event.kind in _BREAK_KINDS and event.confidence >= 0.72 for event in events)
    return has_reset and has_break


def apply_soft_frustration_restart_cleanup(
    kept: Iterable[CandidateTake],
    context: WholeVideoContext | None = None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    survivors = []
    removed = []
    diagnostics = []
    for take in tuple(kept):
        text = str(take.text or "")
        should_remove = (
            bool(_SOFT_FRUSTRATION_RE.search(text))
            and _has_repeated_multiword_phrase(text)
            and _has_multimodal_break(take, context)
        )
        if not should_remove:
            survivors.append(take)
            continue
        removed.append(take)
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "repeated_restart_with_soft_frustration_and_visual_break",
            "text": take.text,
        })
    return tuple(survivors), tuple(removed), tuple(diagnostics)


def install_soft_frustration_restart_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_soft_frustration_restart", False):
        return

    def apply_with_soft_frustration_restart(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, contextual_discarded, diagnostics = apply_soft_frustration_restart_cleanup(kept, context)
        if not contextual_discarded:
            return kept, discarded, decisions
        reason_by_id = {item["clip_id"]: item["reason"] for item in diagnostics}
        extra = tuple(
            CleanCutDecision(
                clip_id=take.clip_id,
                keep=False,
                reason=reason_by_id[take.clip_id],
                confidence=0.96,
            )
            for take in contextual_discarded
        )
        return kept, tuple(discarded) + tuple(contextual_discarded), tuple(decisions) + extra

    apply_with_soft_frustration_restart._cutsell_soft_frustration_restart = True
    clean_cut.apply_clean_cut = apply_with_soft_frustration_restart
