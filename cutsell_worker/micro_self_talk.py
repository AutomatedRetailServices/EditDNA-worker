"""Remove short behind-the-scenes self-talk only inside a proven retry window.

Short reactions, profanity, and criticism are common valid creator content.  This
cleanup therefore requires three independent signals: recording-style wording,
nearby takes on both sides, and local reset/expression evidence.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_SELF_TALK_RE = re.compile(
    r"^\s*(?:oh\s+)?(?:shit|fuck|damn|crap)[.!?\s]*$|"
    r"^\s*(?:this|that|it)\s+(?:is|'s)\s+(?:crap|shit|stupid|terrible|awful)[.!?\s]*$|"
    r"^\s*(?:i\s*(?:am|'m)\s+done|i\s+hate\s+(?:this|that))[.!?\s]*$",
    re.IGNORECASE,
)
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _has_visual_break(take: CandidateTake, context: WholeVideoContext | None) -> bool:
    events = tuple(
        e for e in _source_events(context, take.source_asset_id)
        if e.end >= take.start - 0.25 and e.start <= take.end + 0.25
    )
    has_reset = any(e.kind in _RESET_KINDS and e.confidence >= 0.86 for e in events)
    has_break = any(e.kind in _BREAK_KINDS and e.confidence >= 0.74 for e in events)
    return has_reset and has_break


def _neighbor_window_ids(
    takes: tuple[CandidateTake, ...],
    *,
    maximum_gap_sec: float = 3.0,
) -> set[str]:
    ordered = tuple(sorted(takes, key=lambda x: (x.source_order, x.start, x.end)))
    eligible: set[str] = set()
    for index, take in enumerate(ordered):
        if index == 0 or index == len(ordered) - 1:
            continue
        before = ordered[index - 1]
        after = ordered[index + 1]
        if before.source_asset_id != take.source_asset_id or after.source_asset_id != take.source_asset_id:
            continue
        if not (0.0 <= take.start - before.end <= maximum_gap_sec):
            continue
        if not (0.0 <= after.start - take.end <= maximum_gap_sec):
            continue
        # Neighboring speech must have enough substance to look like actual attempts.
        if before.duration_sec < 1.0 or after.duration_sec < 1.0:
            continue
        eligible.add(take.clip_id)
    return eligible


def apply_micro_self_talk_cleanup(
    kept: Iterable[CandidateTake],
    context: WholeVideoContext | None = None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    window_ids = _neighbor_window_ids(kept_tuple)
    survivors = []
    removed = []
    diagnostics = []
    for take in kept_tuple:
        text = str(take.text or "").strip()
        should_remove = (
            take.duration_sec <= 3.5
            and take.clip_id in window_ids
            and bool(_SELF_TALK_RE.search(text))
            and _has_visual_break(take, context)
        )
        if not should_remove:
            survivors.append(take)
            continue
        removed.append(take)
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "micro_self_talk_inside_retry_window_with_visual_break",
            "text": take.text,
        })
    return tuple(survivors), tuple(removed), tuple(diagnostics)


def install_micro_self_talk_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_micro_self_talk", False):
        return

    def apply_with_micro_self_talk(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, contextual_discarded, diagnostics = apply_micro_self_talk_cleanup(kept, context)
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

    apply_with_micro_self_talk._cutsell_micro_self_talk = True
    clean_cut.apply_clean_cut = apply_with_micro_self_talk
