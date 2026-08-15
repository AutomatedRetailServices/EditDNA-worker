"""Conservative cleanup for short restart questions around a proven retake.

A phrase such as ``What again?`` can be valid dialogue.  It is removed only when a
same-source take immediately restarts with ``Again...`` and dense local performance
evidence shows both physical reset and expression/engagement break around the short
question.  No text-only deletion is allowed.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _has_reset_and_break(
    take: CandidateTake,
    context: WholeVideoContext | None,
    *,
    padding_sec: float = 0.75,
) -> bool:
    events = tuple(
        event for event in _source_events(context, take.source_asset_id)
        if event.end >= take.start - padding_sec and event.start <= take.end + padding_sec
    )
    reset_count = sum(
        1 for event in events
        if event.kind in _RESET_KINDS and event.confidence >= 0.90
    )
    break_count = sum(
        1 for event in events
        if event.kind in _BREAK_KINDS and event.confidence >= 0.72
    )
    return reset_count >= 2 and break_count >= 1


def _following_restart(
    take: CandidateTake,
    ordered: tuple[CandidateTake, ...],
    *,
    maximum_gap_sec: float = 3.0,
) -> CandidateTake | None:
    later = [
        other for other in ordered
        if other.source_asset_id == take.source_asset_id
        and other.clip_id != take.clip_id
        and other.start >= take.end - 0.05
    ]
    if not later:
        return None
    following = min(later, key=lambda item: (item.start, item.end))
    gap = following.start - take.end
    if not -0.05 <= gap <= maximum_gap_sec:
        return None
    tokens = _tokens(following.text)
    if not tokens or tokens[0] != "again":
        return None
    return following


def apply_short_restart_question_cleanup(
    kept: Iterable[CandidateTake],
    context: WholeVideoContext | None = None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    ordered = tuple(sorted(kept_tuple, key=lambda item: (item.source_order, item.start, item.end)))
    survivors = []
    removed = []
    diagnostics = []

    for take in kept_tuple:
        tokens = _tokens(take.text)
        reason = None
        if (
            tokens in {("what", "again"), ("again", "what")}
            and take.duration_sec <= 1.6
            and _following_restart(take, ordered) is not None
            and _has_reset_and_break(take, context)
        ):
            reason = "short_restart_question_with_multimodal_reset"

        if reason is None:
            survivors.append(take)
            continue
        removed.append(take)
        diagnostics.append({"clip_id": take.clip_id, "reason": reason, "text": take.text})

    return tuple(survivors), tuple(removed), tuple(diagnostics)


def install_short_restart_question_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_short_restart_question_cleanup", False):
        return

    def apply_with_short_restart_questions(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, contextual_discarded, diagnostics = apply_short_restart_question_cleanup(kept, context)
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

    apply_with_short_restart_questions._cutsell_short_restart_question_cleanup = True
    clean_cut.apply_clean_cut = apply_with_short_restart_questions
