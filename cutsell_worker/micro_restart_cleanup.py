"""Conservative cleanup for small failed delivery fragments.

This module targets short restart debris that is easy for transcript-only rules to
miss: repeated words/phrases, short partial-prefix attempts before a fuller retry,
and explicit comments about the recording delivery. Profanity and creator reactions
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
        if event.kind in _BREAK_KINDS and event.confidence >= 0.72
    )
    return reset_count >= 2 and break_count >= 1


def _window_has_multimodal_break(
    takes: tuple[CandidateTake, ...],
    context: WholeVideoContext | None,
) -> bool:
    if not takes or context is None:
        return False
    start = min(take.start for take in takes) - 0.30
    end = max(take.end for take in takes) + 0.30
    events = tuple(
        event for event in _source_events(context, takes[0].source_asset_id)
        if event.end >= start and event.start <= end
    )
    resets = sum(1 for event in events if event.kind in _RESET_KINDS and event.confidence >= 0.90)
    breaks = sum(1 for event in events if event.kind in _BREAK_KINDS and event.confidence >= 0.72)
    return resets >= 4 and breaks >= 1


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
    """Find short broken attempts followed by multiple same-idea retries."""
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


def _sandwiched_retry_debris_ids(
    takes: tuple[CandidateTake, ...],
    context: WholeVideoContext | None,
    *,
    maximum_gap_sec: float = 4.0,
) -> set[str]:
    """Remove only tiny fragments trapped between two related attempts.

    The fragment must be short, physically broken, and overlap both neighboring
    attempts. This catches debris such as a single content word plus ``okay`` between
    two longer retries without turning ordinary short reactions into delete rules.
    """
    ordered = tuple(sorted(takes, key=lambda item: (item.source_order, item.start, item.end)))
    remove_ids: set[str] = set()
    for index in range(1, len(ordered) - 1):
        take = ordered[index]
        before = ordered[index - 1]
        after = ordered[index + 1]
        words = _tokens(take.text)
        if not (1 <= len(words) <= 4 and take.duration_sec <= 2.5):
            continue
        if before.source_asset_id != take.source_asset_id or after.source_asset_id != take.source_asset_id:
            continue
        if not (0.0 <= take.start - before.end <= maximum_gap_sec):
            continue
        if not (0.0 <= after.start - take.end <= maximum_gap_sec):
            continue
        if before.duration_sec <= take.duration_sec or after.duration_sec <= take.duration_sec:
            continue
        if _token_overlap(take, before) < 0.50 or _token_overlap(take, after) < 0.50:
            continue
        if not _has_multimodal_break(take, context):
            continue
        remove_ids.add(take.clip_id)
    return remove_ids


def _shares_word_search_stem(left: str, right: str) -> bool:
    """Require a strong same-stem stumble, not an intentional list of short words."""
    if left == right or len(left) < 6 or len(right) < 6:
        return False
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    return prefix >= 5


def _word_search_cluster_ids(
    takes: tuple[CandidateTake, ...],
    context: WholeVideoContext | None,
    *,
    maximum_gap_sec: float = 1.8,
    following_gap_sec: float = 2.0,
) -> set[str]:
    """Remove microtakes produced while searching for a word before a fuller attempt.

    The cluster needs at least two tiny takes, a strong lexical-stem stumble between
    them, two distinct cluster words echoed in the following substantive take, and a
    dense reset/break window. Intentional one-word lists therefore remain protected.
    """
    ordered = tuple(sorted(takes, key=lambda item: (item.source_order, item.start, item.end)))
    remove_ids: set[str] = set()
    index = 0
    while index < len(ordered) - 2:
        first = ordered[index]
        first_tokens = _tokens(first.text)
        if not (first.duration_sec <= 1.2 and 1 <= len(first_tokens) <= 2):
            index += 1
            continue
        cluster = [first]
        cursor = index + 1
        while cursor < len(ordered):
            previous = cluster[-1]
            current = ordered[cursor]
            current_tokens = _tokens(current.text)
            if current.source_asset_id != first.source_asset_id:
                break
            if current.start - previous.end > maximum_gap_sec:
                break
            if not (current.duration_sec <= 1.2 and 1 <= len(current_tokens) <= 2):
                break
            cluster.append(current)
            cursor += 1
        if len(cluster) < 2 or cursor >= len(ordered):
            index += 1
            continue
        following = ordered[cursor]
        if following.source_asset_id != first.source_asset_id:
            index += 1
            continue
        if not (0.0 <= following.start - cluster[-1].end <= following_gap_sec):
            index += 1
            continue
        following_tokens = set(_tokens(following.text))
        if following.duration_sec < 2.0 or len(following_tokens) < 4:
            index += 1
            continue
        cluster_tokens = [token for item in cluster for token in _tokens(item.text) if len(token) >= 4]
        stem_stumble = any(
            _shares_word_search_stem(left, right)
            for i, left in enumerate(cluster_tokens)
            for right in cluster_tokens[i + 1:]
        )
        echoed = len(set(cluster_tokens) & following_tokens)
        if stem_stumble and echoed >= 2 and _window_has_multimodal_break(tuple(cluster) + (following,), context):
            remove_ids.update(item.clip_id for item in cluster)
            index = cursor
            continue
        index += 1
    return remove_ids


def apply_micro_restart_cleanup(
    kept: Iterable[CandidateTake],
    context: WholeVideoContext | None = None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    short_prefix_ids = _short_prefix_retry_ids(kept_tuple, context)
    sandwiched_ids = _sandwiched_retry_debris_ids(kept_tuple, context)
    word_search_ids = _word_search_cluster_ids(kept_tuple, context)
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
        elif take.clip_id in word_search_ids:
            reason = "word_search_microtake_cluster_with_visual_break"
        elif take.clip_id in sandwiched_ids:
            reason = "sandwiched_retry_debris_with_visual_break"
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
