"""Remove a merged recording self-review/confusion take with strong physical evidence.

ASR can merge ``what did I just say?`` and the following confused reaction into one
candidate, bypassing the two-take recording-break rule.  This cleanup only handles
that merged shape: explicit self-review at the start, a short confusion question at
the tail, and a dense reset/break window.  Ordinary rhetorical questions fail open.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_SELF_REVIEW_START_RE = re.compile(r"^\s*what\s+did\s+i\s+(?:just\s+)?say\b", re.IGNORECASE)
_CONFUSION_TAIL_RE = re.compile(r"\bwhat[?!.\s]*$", re.IGNORECASE)
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _has_dense_break(take: CandidateTake, context: WholeVideoContext | None) -> bool:
    events = tuple(
        event for event in _source_events(context, take.source_asset_id)
        if event.end >= take.start - 0.35 and event.start <= take.end + 0.35
    )
    resets = sum(1 for event in events if event.kind in _RESET_KINDS and event.confidence >= 0.90)
    breaks = sum(1 for event in events if event.kind in _BREAK_KINDS and event.confidence >= 0.72)
    return resets >= 4 and breaks >= 1


def _is_merged_self_review(take: CandidateTake) -> bool:
    text = str(take.text or "").strip()
    tokens = _TOKEN_RE.findall(text)
    return (
        take.duration_sec <= 4.5
        and 7 <= len(tokens) <= 18
        and bool(_SELF_REVIEW_START_RE.search(text))
        and bool(_CONFUSION_TAIL_RE.search(text))
    )


def apply_merged_self_review_cleanup(
    kept: Iterable[CandidateTake],
    context: WholeVideoContext | None = None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    survivors, removed, diagnostics = [], [], []
    for take in tuple(kept):
        if _is_merged_self_review(take) and _has_dense_break(take, context):
            removed.append(take)
            diagnostics.append({
                "clip_id": take.clip_id,
                "reason": "merged_speech_self_review_confusion_with_physical_reset",
                "text": take.text,
            })
        else:
            survivors.append(take)
    return tuple(survivors), tuple(removed), tuple(diagnostics)


def install_merged_self_review_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_merged_self_review", False):
        return

    def apply_with_merged_self_review(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, extra_discarded, diagnostics = apply_merged_self_review_cleanup(kept, context)
        if not extra_discarded:
            return kept, discarded, decisions
        reason_by_id = {item["clip_id"]: item["reason"] for item in diagnostics}
        extra = tuple(
            CleanCutDecision(take.clip_id, False, reason_by_id[take.clip_id], 0.97)
            for take in extra_discarded
        )
        return kept, tuple(discarded) + tuple(extra_discarded), tuple(decisions) + extra

    apply_with_merged_self_review._cutsell_merged_self_review = True
    clean_cut.apply_clean_cut = apply_with_merged_self_review
