"""Remove short multi-segment word-search attempts under dense recording resets.

Some ASR boundaries merge one-word stumbles into neighboring script words, so the
microtake-only word-search cleanup can miss them.  This sequence rule requires a very
specific structure across adjacent short takes: a long token repeated at least three
times, a different long token sharing a five-character stem with it, at least two
other target words shared by two attempts, and a dense multimodal reset window.
Intentional lists, slogans, and ordinary lexical repetition therefore fail open.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})
_STOP = frozenset({"about", "after", "again", "also", "and", "because", "before", "being", "have", "into", "just", "that", "their", "there", "these", "they", "this", "those", "with", "your"})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold().replace("'", "") for token in _TOKEN_RE.findall(str(text or "")))


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in _tokens(text) if len(token) >= 4 and token not in _STOP)


def _shares_stem(left: str, right: str) -> bool:
    if left == right or len(left) < 6 or len(right) < 6:
        return False
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    return prefix >= 5


def _lexically_linked(left: CandidateTake, right: CandidateTake) -> bool:
    a = set(_content_tokens(left.text))
    b = set(_content_tokens(right.text))
    if a & b:
        return True
    return any(_shares_stem(x, y) for x in a for y in b)


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _dense_multimodal_window(cluster: tuple[CandidateTake, ...], context: WholeVideoContext | None) -> bool:
    if not cluster or context is None:
        return False
    start = cluster[0].start - 0.35
    end = cluster[-1].end + 0.35
    events = tuple(
        event for event in _source_events(context, cluster[0].source_asset_id)
        if event.end >= start and event.start <= end
    )
    resets = sum(1 for event in events if event.kind in _RESET_KINDS and event.confidence >= 0.90)
    breaks = sum(1 for event in events if event.kind in _BREAK_KINDS and event.confidence >= 0.72)
    return resets >= 6 and breaks >= 1


def _is_word_search_cluster(cluster: tuple[CandidateTake, ...], context: WholeVideoContext | None) -> bool:
    if len(cluster) < 2 or not _dense_multimodal_window(cluster, context):
        return False

    per_take = [tuple(_content_tokens(take.text)) for take in cluster]
    all_tokens = [token for tokens in per_take for token in tokens if len(token) >= 6]
    counts = Counter(all_tokens)
    repeated = {token for token, count in counts.items() if count >= 3}
    if not repeated:
        return False

    variants = {
        token for token in set(all_tokens)
        if any(_shares_stem(token, repeat) for repeat in repeated)
    }
    if not variants:
        return False

    contaminated = repeated | variants
    take_presence: dict[str, set[int]] = defaultdict(set)
    for index, tokens in enumerate(per_take):
        for token in set(tokens):
            if token not in contaminated:
                take_presence[token].add(index)
    shared_target = [token for token, indexes in take_presence.items() if len(indexes) >= 2]
    return len(shared_target) >= 2


def _word_search_cluster_ids(
    takes: tuple[CandidateTake, ...],
    context: WholeVideoContext | None,
    *,
    maximum_gap_sec: float = 2.5,
    maximum_take_sec: float = 5.5,
    maximum_span_sec: float = 16.0,
) -> set[str]:
    ordered = tuple(sorted(takes, key=lambda item: (item.source_order, item.start, item.end)))
    remove_ids: set[str] = set()
    index = 0
    while index < len(ordered) - 1:
        first = ordered[index]
        if first.duration_sec > maximum_take_sec:
            index += 1
            continue
        cluster = [first]
        cursor = index + 1
        while cursor < len(ordered):
            current = ordered[cursor]
            previous = cluster[-1]
            if current.source_asset_id != first.source_asset_id:
                break
            if current.duration_sec > maximum_take_sec:
                break
            if current.start - previous.end > maximum_gap_sec:
                break
            if current.end - first.start > maximum_span_sec:
                break
            if not any(_lexically_linked(existing, current) for existing in cluster):
                break
            cluster.append(current)
            cursor += 1
        cluster_tuple = tuple(cluster)
        if _is_word_search_cluster(cluster_tuple, context):
            remove_ids.update(take.clip_id for take in cluster_tuple)
            index = cursor
            continue
        index += 1
    return remove_ids


def apply_word_search_attempt_cleanup(
    kept: Iterable[CandidateTake],
    context: WholeVideoContext | None = None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    remove_ids = _word_search_cluster_ids(kept_tuple, context)
    survivors, removed, diagnostics = [], [], []
    for take in kept_tuple:
        if take.clip_id in remove_ids:
            removed.append(take)
            diagnostics.append({
                "clip_id": take.clip_id,
                "reason": "multi_segment_word_search_cluster_with_dense_reset",
                "text": take.text,
            })
        else:
            survivors.append(take)
    return tuple(survivors), tuple(removed), tuple(diagnostics)


def install_word_search_attempt_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_word_search_attempts", False):
        return

    def apply_with_word_search_attempts(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, extra_discarded, diagnostics = apply_word_search_attempt_cleanup(kept, context)
        if not extra_discarded:
            return kept, discarded, decisions
        reason_by_id = {item["clip_id"]: item["reason"] for item in diagnostics}
        extra = tuple(
            CleanCutDecision(take.clip_id, False, reason_by_id[take.clip_id], 0.97)
            for take in extra_discarded
        )
        return kept, tuple(discarded) + tuple(extra_discarded), tuple(decisions) + extra

    apply_with_word_search_attempts._cutsell_word_search_attempts = True
    clean_cut.apply_clean_cut = apply_with_word_search_attempts
