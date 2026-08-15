"""Remove behind-the-scenes self-talk only inside a proven recording-break window.

Short reactions, profanity, criticism, and words such as ``done`` or ``cool`` can be
valid creator content. They are never destructive on their own. Cleanup requires
neighboring real takes plus either a visual break or a dense physical reset window.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_SELF_TALK_RE = re.compile(
    r"^\s*(?:oh\s+)?(?:shit|fuck|damn|crap|ugh)[.!?\s]*$|"
    r"^\s*(?:this|that|it)\s+(?:is|'s)\s+(?:crap|shit|stupid|terrible|awful)[.!?\s]*$|"
    r"^\s*(?:i\s*(?:am|'m)\s+done|i\s+hate\s+(?:this|that))[.!?\s]*$",
    re.IGNORECASE,
)
_PROCESS_RE = re.compile(
    r"\bwhy\s+do\s+i\s+(?:keep\s+)?say(?:ing)?\b|"
    r"\bi\s+hate\s+(?:saying|being)\b|"
    r"\bcall\s+to\s+action\b|"
    r"\b(?:how|what)\s+(?:do|should)\s+i\s+(?:end|say)\b|"
    r"\bi\s+(?:do\s+not|don't|dont)\s+know\s+how\s+to\s+end\b|"
    r"\b(?:you(?:'re|\s+are)|i\s*(?:am|'m))\s+stupid\b",
    re.IGNORECASE,
)
_SHORT_RECOVERY_RE = re.compile(r"^\s*(?:no|done|cool|okay|ok|ugh|fuck|shit|damn)[.!?\s]*$", re.IGNORECASE)
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


def _events_for_take(take: CandidateTake, context: WholeVideoContext | None):
    return tuple(
        e for e in _source_events(context, take.source_asset_id)
        if e.end >= take.start - 0.25 and e.start <= take.end + 0.35
    )


def _has_visual_break(take: CandidateTake, context: WholeVideoContext | None) -> bool:
    events = _events_for_take(take, context)
    has_reset = any(e.kind in _RESET_KINDS and e.confidence >= 0.86 for e in events)
    has_break = any(e.kind in _BREAK_KINDS and e.confidence >= 0.74 for e in events)
    return has_reset and has_break


def _has_dense_physical_reset(take: CandidateTake, context: WholeVideoContext | None) -> bool:
    events = _events_for_take(take, context)
    resets = sum(1 for e in events if e.kind in _RESET_KINDS and e.confidence >= 0.90)
    return resets >= 4


def _window_has_dense_breaks(source_asset_id: str, start: float, end: float, context: WholeVideoContext | None) -> bool:
    events = tuple(
        e for e in _source_events(context, source_asset_id)
        if e.end >= start - 0.35 and e.start <= end + 0.35
    )
    resets = sum(1 for e in events if e.kind in _RESET_KINDS and e.confidence >= 0.86)
    breaks = sum(1 for e in events if e.kind in _BREAK_KINDS and e.confidence >= 0.74)
    return resets >= 4 and breaks >= 2


def _is_anchor(take: CandidateTake) -> bool:
    text = str(take.text or "").strip()
    return bool(_SELF_TALK_RE.search(text) or _PROCESS_RE.search(text))


def _bts_window_ids(takes: tuple[CandidateTake, ...], context: WholeVideoContext | None, *, maximum_anchor_gap_sec: float = 6.0, maximum_span_sec: float = 35.0) -> set[str]:
    ordered = tuple(sorted(takes, key=lambda x: (x.source_order, x.start, x.end)))
    remove_ids: set[str] = set()
    index = 0
    while index < len(ordered):
        if not _is_anchor(ordered[index]):
            index += 1
            continue
        anchors = [ordered[index]]
        cursor = index + 1
        last_anchor = ordered[index]
        while cursor < len(ordered):
            take = ordered[cursor]
            if take.source_asset_id != anchors[0].source_asset_id or take.end - anchors[0].start > maximum_span_sec:
                break
            if _is_anchor(take):
                if take.start - last_anchor.end > maximum_anchor_gap_sec:
                    break
                anchors.append(take)
                last_anchor = take
            cursor += 1
        if len(anchors) >= 3 and _window_has_dense_breaks(anchors[0].source_asset_id, anchors[0].start, anchors[-1].end, context):
            window_start = anchors[0].start - 1.0
            window_end = anchors[-1].end + 1.0
            for take in ordered:
                if take.source_asset_id != anchors[0].source_asset_id or take.end < window_start or take.start > window_end:
                    continue
                text = str(take.text or "").strip()
                tiny_debris = take.duration_sec <= 1.25 and len(_tokens(text)) <= 3
                if _is_anchor(take) or _SHORT_RECOVERY_RE.search(text) or tiny_debris:
                    remove_ids.add(take.clip_id)
        index += 1
    return remove_ids


def _neighbor_window_ids(takes: tuple[CandidateTake, ...], *, maximum_gap_sec: float = 3.0) -> set[str]:
    ordered = tuple(sorted(takes, key=lambda x: (x.source_order, x.start, x.end)))
    eligible: set[str] = set()
    for index, take in enumerate(ordered):
        if index == 0 or index == len(ordered) - 1:
            continue
        before = ordered[index - 1]
        after = ordered[index + 1]
        if before.source_asset_id != take.source_asset_id or after.source_asset_id != take.source_asset_id:
            continue
        if not (0.0 <= take.start - before.end <= maximum_gap_sec and 0.0 <= after.start - take.end <= maximum_gap_sec):
            continue
        if before.duration_sec < 1.0 or after.duration_sec < 1.0:
            continue
        eligible.add(take.clip_id)
    return eligible


def apply_micro_self_talk_cleanup(kept: Iterable[CandidateTake], context: WholeVideoContext | None = None) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    # Face/camera break evidence stays on the tight historical window. Dense physical
    # reset evidence is much stronger and can bridge a slightly wider post-bloopers
    # pause after earlier cleanup removed silence/debris. This catches a short exact
    # self-talk reaction such as ``Oh shit`` without making isolated profanity a rule.
    local_window_ids = _neighbor_window_ids(kept_tuple, maximum_gap_sec=3.0)
    physical_window_ids = _neighbor_window_ids(kept_tuple, maximum_gap_sec=7.0)
    bts_window_ids = _bts_window_ids(kept_tuple, context)
    survivors, removed, diagnostics = [], [], []
    for take in kept_tuple:
        text = str(take.text or "").strip()
        reason = None
        if take.clip_id in bts_window_ids:
            reason = "corroborated_behind_the_scenes_self_talk_window"
        elif take.duration_sec <= 3.5 and bool(_SELF_TALK_RE.search(text)):
            if take.clip_id in local_window_ids and _has_visual_break(take, context):
                reason = "micro_self_talk_inside_retry_window_with_visual_break"
            elif take.clip_id in physical_window_ids and _has_dense_physical_reset(take, context):
                reason = "micro_self_talk_inside_retry_window_with_dense_physical_reset"
        if reason is None:
            survivors.append(take)
        else:
            removed.append(take)
            diagnostics.append({"clip_id": take.clip_id, "reason": reason, "text": take.text})
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
        extra = tuple(CleanCutDecision(clip_id=take.clip_id, keep=False, reason=reason_by_id[take.clip_id], confidence=0.96) for take in contextual_discarded)
        return kept, tuple(discarded) + tuple(contextual_discarded), tuple(decisions) + extra
    apply_with_micro_self_talk._cutsell_micro_self_talk = True
    clean_cut.apply_clean_cut = apply_with_micro_self_talk
