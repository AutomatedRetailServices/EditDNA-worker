"""Confirm obvious local product-handling failures from multimodal evidence.

A hand gesture is not a failure.  This cleanup requires a dense hand-motion event,
a facial reaction, an additional break/disengagement cue, and a nearby same-idea
retry.  The combination targets drops/fumbles while preserving normal demos.
"""
from __future__ import annotations

from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .take_grouping import retry_similarity
from .whole_video_analysis import WholeVideoContext

_HAND_KIND = "hand_motion_reset_candidate"
_FACE_KIND = "facial_expression_shift_candidate"
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "body_reset_candidate"})


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _events_for_take(take: CandidateTake, context: WholeVideoContext | None):
    return tuple(
        event for event in _source_events(context, take.source_asset_id)
        if event.end >= take.start - 0.20 and event.start <= take.end + 0.35
    )


def _has_handling_break(take: CandidateTake, context: WholeVideoContext | None) -> bool:
    events = _events_for_take(take, context)
    hand = sum(1 for e in events if e.kind == _HAND_KIND and e.confidence >= 0.88)
    face = sum(1 for e in events if e.kind == _FACE_KIND and e.confidence >= 0.76)
    extra_break = sum(1 for e in events if e.kind in _BREAK_KINDS and e.confidence >= 0.76)
    return hand >= 2 and face >= 1 and extra_break >= 1


def _has_nearby_retry(
    take: CandidateTake,
    candidates: tuple[CandidateTake, ...],
    *,
    maximum_gap_sec: float = 10.0,
    minimum_similarity: float = 0.62,
) -> bool:
    for other in candidates:
        if other.clip_id == take.clip_id or other.source_asset_id != take.source_asset_id:
            continue
        if other.start < take.end:
            continue
        gap = other.start - take.end
        if gap > maximum_gap_sec:
            continue
        if retry_similarity(take.text, other.text) >= minimum_similarity:
            return True
    return False


def apply_product_handling_failure_cleanup(
    kept: Iterable[CandidateTake],
    context: WholeVideoContext | None = None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    survivors = []
    removed = []
    diagnostics = []
    for take in kept_tuple:
        should_remove = (
            take.duration_sec <= 8.0
            and _has_handling_break(take, context)
            and _has_nearby_retry(take, kept_tuple)
        )
        if not should_remove:
            survivors.append(take)
            continue
        removed.append(take)
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "product_handling_fumble_with_face_reaction_before_retry",
            "text": take.text,
        })
    return tuple(survivors), tuple(removed), tuple(diagnostics)


def install_product_handling_failure_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_product_handling_failure", False):
        return

    def apply_with_product_handling_failure(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, contextual_discarded, diagnostics = apply_product_handling_failure_cleanup(kept, context)
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

    apply_with_product_handling_failure._cutsell_product_handling_failure = True
    clean_cut.apply_clean_cut = apply_with_product_handling_failure
