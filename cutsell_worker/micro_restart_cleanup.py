"""Conservative cleanup for small failed delivery fragments.

This module targets short restart debris that is easy for transcript-only rules to
miss: repeated words/phrases, short partial-prefix attempts before a fuller retry,
and explicit comments about the recording delivery.  Profanity and creator reactions
are never destructive by themselves.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_DELIVERY_META_RE = re.compile(
    r"\b(?:okay|ok)\s+now\b.{0,40}\b(?:whole|full)\s+sentence\b|"
    r"\b(?:whole|full)\s+sentence\b.{0,30}\b(?:okay|ok|again|redo|restart)\b|"
    r"\b(?:say|do|start|take)\s+(?:the\s+)?(?:whole|full)\s+sentence(?:\s+again)?\b",
    re.IGNORECASE,
)
_SELF_EVAL_RE = re.compile(
    r"\b(?:this|that|it)\s+(?:is|'s)\s+(?:crap|shit|stupid|terrible|awful)\b",
    re.IGNORECASE,
)
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})
_PROTECTED_REPEAT_WORDS = frozenset({
    "so", "very", "really", "super", "much", "more", "yes", "no", "okay", "ok",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


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
        if event.end >= take.start - 0.30 and event.start <= take.end + 0.30
    )
    reset_count = sum(
        1 for event in events
        if event.kind in _RESET_KINDS and event.confidence >= 0.90
    )
    break_count = sum(
        1 for event in events
        if event.kind in _BREAK_KINDS and event.confidence >= 0.75
    )
    return reset_count >= 2 and break_count >= 1


def _has_adjacent_duplicate_content_word(text: str) -> bool:
    tokens = _tokens(text)
    for left, right in zip(tokens, tokens[1:]):
        if left == right and len(left) >= 4 and left not in _PROTECTED_REPEAT_WORDS:
            return True
    return False


def _has_repeated_multiword_phrase(text: str) -> bool:
    tokens = _tokens(text)
    if len(tokens) < 4:
        return False
    for width in range(min(5, len(tokens) // 2), 1, -1):
        seen: set[tuple[str, ...]] = set()
        for index in range(0, len(tokens) - width + 1):
            gram = tokens[index:index + width]
            if len(set(gram)) < 2:
                continue
            if gram in seen:
                return True
            seen.add(gram)
    return False


def _token_overlap(left: CandidateTake, right: CandidateTake) -> float:
    a = set(_tokens(left.text))
    b = set(_tokens(right.text))
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _short_prefix_retry_ids(
    takes: tuple[CandidateTake, ...],
    context: WholeVideoContext | None,
    *,
    horizon_sec: float = 45.0,
) -> set[str]:
    """Find short broken attempts followed by multiple same-idea retries.

    A short phrase is not deleted just because a longer related sentence appears.
    It must also show a local physical/expression break and have at least two later
    same-source attempts, one of which is materially fuller.
    """
    ordered = tuple(sorted(takes, key=lambda item: (item.source_order, item.start, item.end)))
    remove_ids: set[str] = set()
    for index, take in enumerate(ordered):
        words = _tokens(take.text)
        if not (2 <= len(words) <= 5 and take.duration_sec <= 3.5):
            continue
        if not _has_multimodal_break(take, context):
            continue
        related = []
        for later in ordered[index + 1:]:
            if later.source_asset_id != take.source_asset_id:
                continue
            if later.start - take.end > horizon_sec:
                break
            if _token_overlap(take, later) >= 0.50:
                related.append(later)
        fuller = [item for item in related if len(_tokens(item.text)) >= len(words) + 4]
        if len(related) >= 2 and fuller:
            remove_ids.add(take.clip_id)
    return remove_ids


def apply_micro_restart_cleanup(
    kept: Iterable[CandidateTake],
    context: WholeVideoContext | None = None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    short_prefix_ids = _short_prefix_retry_ids(kept_tuple, context)
    survivors = []
    removed = []
    diagnostics = []
    for take in kept_tuple:
        text = str(take.text or "").strip()
        reason = None
        confidence = 0.96
        if _DELIVERY_META_RE.search(text):
            reason = "explicit_delivery_process_meta"
            confidence = 0.97
        elif take.duration_sec <= 4.0 and _SELF_EVAL_RE.search(text) and _has_multimodal_break(take, context):
            reason = "negative_self_evaluation_with_visual_break"
        elif _has_adjacent_duplicate_content_word(text) and _has_multimodal_break(take, context):
            reason = "adjacent_content_word_restart_with_visual_break"
        elif _has_repeated_multiword_phrase(text) and _has_multimodal_break(take, context):
            reason = "repeated_phrase_restart_with_visual_break"
        elif take.clip_id in short_prefix_ids:
            reason = "short_prefix_before_fuller_retry_with_visual_break"
        if reason is None:
            survivors.append(take)
            continue
        removed.append(take)
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": reason,
            "confidence": confidence,
            "text": take.text,
        })
    return tuple(survivors), tuple(removed), tuple(diagnostics)


def install_micro_restart_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_micro_restart_cleanup", False):
        return

    def apply_with_micro_restart_cleanup(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, contextual_discarded, diagnostics = apply_micro_restart_cleanup(kept, context)
        if not contextual_discarded:
            return kept, discarded, decisions
        diagnostic_by_id = {item["clip_id"]: item for item in diagnostics}
        extra = tuple(
            CleanCutDecision(
                clip_id=take.clip_id,
                keep=False,
                reason=diagnostic_by_id[take.clip_id]["reason"],
                confidence=float(diagnostic_by_id[take.clip_id]["confidence"]),
            )
            for take in contextual_discarded
        )
        return kept, tuple(discarded) + tuple(contextual_discarded), tuple(decisions) + extra

    apply_with_micro_restart_cleanup._cutsell_micro_restart_cleanup = True
    clean_cut.apply_clean_cut = apply_with_micro_restart_cleanup
