"""Conservative sequence cleanup around explicit recording-process anchors."""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_STOP_RE = re.compile(
    r"\b(?:okay\s+)?(?:stop|stopped)|\bstart\s+over\b|\blet\s+me\s+(?:start\s+over|redo)\b|\bredo\s+that\b",
    re.IGNORECASE,
)
_STRONG_STOP_RE = re.compile(
    r"(?:^|[.!?]\s*|\b(?:damn(?:\s+it)?|ugh|oops)\b[,.!]?\s*)"
    r"(?:okay\s+)?(?:stop|stopped)\b|"
    r"\bstart\s+over\b|\blet\s+me\s+(?:start\s+over|redo)\b|\bredo\s+that\b",
    re.IGNORECASE,
)
_POST_STOP_META_RE = re.compile(
    r"\b(?:that\s+better\s+have\s+been\s+good|was\s+that\s+good|did\s+that\s+sound\s+right|i\s+hope\s+that\s+was\s+good)\b",
    re.IGNORECASE,
)
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})


def _is_stop_anchor(take: CandidateTake) -> bool:
    return bool(_STOP_RE.search(str(take.text or "")))


def _is_strong_stop_anchor(take: CandidateTake) -> bool:
    """Recognize explicit recording-control speech without relying on a provider.

    Plain semantic uses of the word ``stop`` are intentionally not enough.  The
    deterministic path only accepts strong recording-process forms such as
    ``okay stop``, a frustration/interjection followed by stop, ``start over``,
    or ``redo that``.
    """
    text = str(take.text or "").strip()
    if not text or take.duration_sec > 5.0:
        return False
    lowered = text.lower()
    if re.search(r"\b(?:start\s+over|let\s+me\s+(?:start\s+over|redo)|redo\s+that)\b", lowered):
        return True
    if re.search(r"\bokay\s+(?:stop|stopped)\b", lowered):
        return True
    if re.search(r"\b(?:damn(?:\s+it)?|ugh|oops)\b.{0,20}\b(?:stop|stopped)\b", lowered):
        return True
    return bool(_STRONG_STOP_RE.search(text))


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _has_multimodal_reset_between(
    take: CandidateTake,
    anchor: CandidateTake,
    context: WholeVideoContext | None,
) -> bool:
    start = max(take.start, take.end - 0.60)
    end = anchor.end
    events = tuple(
        event for event in _source_events(context, take.source_asset_id)
        if event.end >= start and event.start <= end
    )
    has_reset = any(event.kind in _RESET_KINDS and event.confidence >= 0.72 for event in events)
    has_break = any(event.kind in _BREAK_KINDS and event.confidence >= 0.72 for event in events)
    return has_reset and has_break


def apply_recording_process_neighbors(
    kept: Iterable[CandidateTake],
    discarded: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    *,
    pre_anchor_gap_sec: float = 3.5,
    post_anchor_gap_sec: float = 5.0,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    """Remove only takes strongly tied to an explicit stop/restart recording event.

    Strong recording-control anchors are detected deterministically even when the
    baseline/local brain initially kept them.  A pre-anchor take is removed only
    when an explicit stop/restart follows shortly and dense visual evidence contains
    both a reset family and an engagement/expression break family.  Post-anchor
    speech is removed only for explicit recording-meta phrases.  No profanity or
    ordinary wording is treated as evidence by itself.
    """
    kept_tuple = tuple(kept)
    discarded_tuple = tuple(discarded)

    anchors_by_id = {
        take.clip_id: take
        for take in discarded_tuple
        if _is_stop_anchor(take)
    }
    for take in kept_tuple:
        if _is_strong_stop_anchor(take):
            anchors_by_id[take.clip_id] = take
    anchors = tuple(sorted(anchors_by_id.values(), key=lambda take: (take.source_order, take.start, take.end)))
    if not anchors:
        return kept_tuple, (), ()

    removed = []
    diagnostics = []
    survivors = []

    for take in kept_tuple:
        reason = None
        anchor_id = None

        if _is_strong_stop_anchor(take):
            reason = "explicit_recording_stop_anchor"
            anchor_id = take.clip_id
        else:
            for anchor in anchors:
                if anchor.source_asset_id != take.source_asset_id:
                    continue

                pre_gap = anchor.start - take.end
                if (
                    0.0 <= pre_gap <= pre_anchor_gap_sec
                    and take.duration_sec <= 4.0
                    and _has_multimodal_reset_between(take, anchor, context)
                ):
                    reason = "failed_take_before_explicit_stop_with_visual_reset"
                    anchor_id = anchor.clip_id
                    break

                post_gap = take.start - anchor.end
                if (
                    0.0 <= post_gap <= post_anchor_gap_sec
                    and _POST_STOP_META_RE.search(str(take.text or ""))
                ):
                    reason = "recording_meta_after_explicit_stop"
                    anchor_id = anchor.clip_id
                    break

        if reason is None:
            survivors.append(take)
            continue

        removed.append(take)
        diagnostics.append({
            "clip_id": take.clip_id,
            "anchor_clip_id": anchor_id,
            "reason": reason,
            "text": take.text,
        })

    return tuple(survivors), tuple(removed), tuple(diagnostics)


def install_recording_process_context_cleanup() -> None:
    """Install sequence-aware cleanup before runtime modules import apply_clean_cut."""
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_recording_process_context", False):
        return

    def apply_with_recording_context(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, contextual_discarded, diagnostics = apply_recording_process_neighbors(
            kept, discarded, context
        )
        if not contextual_discarded:
            return kept, discarded, decisions

        reason_by_id = {item["clip_id"]: item["reason"] for item in diagnostics}
        confidence_by_reason = {
            "explicit_recording_stop_anchor": 0.98,
            "failed_take_before_explicit_stop_with_visual_reset": 0.96,
            "recording_meta_after_explicit_stop": 0.96,
        }
        extra_decisions = tuple(
            CleanCutDecision(
                clip_id=take.clip_id,
                keep=False,
                reason=reason_by_id[take.clip_id],
                confidence=confidence_by_reason.get(reason_by_id[take.clip_id], 0.96),
            )
            for take in contextual_discarded
        )
        return (
            kept,
            tuple(discarded) + tuple(contextual_discarded),
            tuple(decisions) + extra_decisions,
        )

    apply_with_recording_context._cutsell_recording_process_context = True
    clean_cut.apply_clean_cut = apply_with_recording_context
