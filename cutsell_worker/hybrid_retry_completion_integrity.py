"""Conservative post-Hybrid integrity for cross-group retries and broken final clauses.

Hybrid judges bounded creator sessions, so it can recognize retries that deterministic
TakeGroup boundaries missed.  This pass uses that semantic evidence only when it is
corroborated by direct lexical coverage and/or recording-process evidence.  It never
invents a retry from topic similarity alone.

A second narrow repair prevents a retained winner from ending on a visibly broken
parallel clause when the immediately following short suffix is already proven failed.
Instead of restoring the failed suffix, it rolls the winner back to the preceding clean
parallel clause at a real word boundary.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable

from .contracts import CandidateTake

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "so", "that", "the", "this", "to",
    "was", "we", "were", "what", "with", "you", "your", "al", "como", "con", "cuando",
    "de", "del", "el", "en", "ella", "ellas", "ellos", "es", "esta", "este", "la", "las",
    "le", "les", "lo", "los", "me", "mi", "mis", "o", "para", "pero", "por", "porque",
    "que", "se", "si", "sin", "su", "sus", "un", "una", "unos", "unas", "y", "yo",
})
_CRITICAL = frozenset({"no", "not", "never", "nunca", "sin", "without"})
_BRIDGE_END = frozenset({
    "a", "al", "and", "because", "but", "by", "con", "de", "del", "el", "for", "in",
    "la", "las", "los", "of", "para", "pero", "por", "porque", "que", "the", "to", "with", "y",
})
_PARALLEL_MARKERS = frozenset({
    "i", "you", "we", "they", "he", "she", "it", "me", "my", "your",
    "yo", "tu", "tú", "te", "me", "se", "le", "les", "nos",
})
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*\s*$")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) >= 4 and token not in _STOP}


def _critical(text: str) -> set[str]:
    tokens = _tokens(text)
    return {token for token in tokens if token in _CRITICAL or any(ch.isdigit() for ch in token)}


def _shared(left: CandidateTake, right: CandidateTake) -> int:
    return len(_content(left.text) & _content(right.text))


def _coverage(take: CandidateTake, others: Iterable[CandidateTake]) -> float:
    own = _content(take.text)
    if not own:
        return 0.0
    covered: set[str] = set()
    for other in others:
        covered.update(_content(other.text))
    return len(own & covered) / max(1, len(own))


def _gap(left: CandidateTake, right: CandidateTake) -> float:
    if left.source_asset_id != right.source_asset_id:
        return float("inf")
    if left.end <= right.start:
        return max(0.0, right.start - left.end)
    if right.end <= left.start:
        return max(0.0, left.start - right.end)
    return 0.0


def _same_opening(left: CandidateTake, right: CandidateTake) -> bool:
    a = tuple(token for token in _tokens(left.text) if len(token) >= 4 and token not in _STOP)
    b = tuple(token for token in _tokens(right.text) if len(token) >= 4 and token not in _STOP)
    return len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2]


def _local_failure(take: CandidateTake, context) -> bool:
    # Reuse the exact Watch+Listen corroboration already used by Hybrid deletion.
    from .hybrid_session_cleanup import _failed_local_evidence

    failed, _ = _failed_local_evidence(take, context)
    return bool(failed)


def _safe_failed_retry(
    take: CandidateTake,
    peers: tuple[CandidateTake, ...],
    semantic: dict[str, tuple[str, float]],
    context,
) -> CandidateTake | None:
    label, confidence = semantic.get(take.clip_id, ("", 0.0))
    if label not in {"failed", "bts"} or confidence < 0.78 or not _local_failure(take, context):
        return None
    candidates = []
    for peer in peers:
        if peer.clip_id == take.clip_id or _gap(take, peer) > 30.0:
            continue
        peer_label, peer_conf = semantic.get(peer.clip_id, ("", 0.0))
        if peer_label not in {"winner", "keep"} or peer_conf < 0.85:
            continue
        shared = _shared(take, peer)
        coverage = _coverage(take, (peer,))
        if shared < 4:
            continue
        if not (_same_opening(take, peer) and coverage >= 0.55) and coverage < 0.72:
            continue
        candidates.append((coverage, shared, peer_conf, peer))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][:3] == best[:3]:
        return None
    return best[-1]


def _safe_short_alternate_debris(
    take: CandidateTake,
    previous: CandidateTake | None,
    following: CandidateTake | None,
    semantic: dict[str, tuple[str, float]],
) -> bool:
    label, confidence = semantic.get(take.clip_id, ("", 0.0))
    content = _content(take.text)
    if label != "alternate" or confidence < 0.74:
        return False
    if not (3 <= len(content) <= 8 and take.duration_sec <= 6.0):
        return False
    if previous is None or following is None:
        return False
    if _gap(previous, take) > 15.0 or _gap(take, following) > 15.0:
        return False
    if previous.source_asset_id != take.source_asset_id or following.source_asset_id != take.source_asset_id:
        return False
    if _coverage(take, (previous, following)) < 0.75:
        return False
    if _shared(take, previous) < 1 or _shared(take, following) < 1:
        return False
    neighbor_critical = _critical(previous.text) | _critical(following.text)
    if not _critical(take.text).issubset(neighbor_critical):
        return False
    return True


def _safe_full_alternate_retry(
    take: CandidateTake,
    ordered: tuple[CandidateTake, ...],
    index: int,
    semantic: dict[str, tuple[str, float]],
    context,
) -> CandidateTake | None:
    label, confidence = semantic.get(take.clip_id, ("", 0.0))
    if label != "alternate" or confidence < 0.70 or take.duration_sec <= 6.0:
        return None
    if not _local_failure(take, context):
        return None

    winners = []
    for candidate in ordered:
        if candidate.clip_id == take.clip_id or _gap(take, candidate) > 30.0:
            continue
        peer_label, peer_conf = semantic.get(candidate.clip_id, ("", 0.0))
        if peer_label == "winner" and peer_conf >= 0.90 and _shared(take, candidate) >= 5:
            winners.append(candidate)
    if len(winners) != 1:
        return None
    winner = winners[0]
    if _coverage(take, (winner,)) >= 0.45:
        return winner

    # A winner may itself be split at a long-form pause while ending on a grammatical
    # bridge (e.g. "... de los" + "cánceres ...").  Count only the immediate continuation,
    # never arbitrary later story paragraphs, when proving alternate coverage.
    try:
        winner_index = ordered.index(winner)
    except ValueError:
        return None
    if winner_index + 1 >= len(ordered):
        return None
    continuation = ordered[winner_index + 1]
    winner_tokens = _tokens(winner.text)
    if not winner_tokens or winner_tokens[-1] not in _BRIDGE_END:
        return None
    if continuation.source_asset_id != winner.source_asset_id or _gap(winner, continuation) > 3.0:
        return None
    combined_shared = len(_content(take.text) & (_content(winner.text) | _content(continuation.text)))
    combined_coverage = _coverage(take, (winner, continuation))
    if combined_shared >= 7 and combined_coverage >= 0.35:
        return winner
    return None


def _trim_parallel_clause_before_failed_tail(
    take: CandidateTake,
    failed_tail: CandidateTake,
) -> CandidateTake | None:
    """Roll back only a broken final parallel clause to the prior clean clause.

    This is deliberately narrower than generic punctuation repair.  It activates only
    when Hybrid has already proven the immediate suffix failed, and only when the end of
    the retained take contains a repeated short pronoun/subject marker introducing two
    parallel clauses ("... te protegen te reparan ...").
    """
    if _SENTENCE_END_RE.search(str(take.text or "").strip()):
        return None
    words = tuple(take.words)
    if len(words) < 8:
        return None
    word_tokens = [(_tokens(word.text)[0] if _tokens(word.text) else "") for word in words]
    window_start = max(1, len(word_tokens) - 12)
    positions: dict[str, list[int]] = {}
    for position in range(window_start, len(word_tokens)):
        token = word_tokens[position]
        if token in _PARALLEL_MARKERS:
            positions.setdefault(token, []).append(position)
    candidates = []
    for token, indexes in positions.items():
        if len(indexes) < 2:
            continue
        first, second = indexes[-2], indexes[-1]
        if second - first < 2 or len(word_tokens) - second > 6:
            continue
        candidates.append((second, first, token))
    if not candidates:
        return None
    second, _, marker = max(candidates)
    kept_words = words[:second]
    if len(kept_words) < 6 or kept_words[-1].end <= take.start:
        return None
    new_text = " ".join(word.text.strip() for word in kept_words if word.text.strip()).strip()
    if not new_text:
        return None
    new_end = float(kept_words[-1].end)
    return replace(
        take,
        end=new_end,
        text=new_text,
        words=tuple(kept_words),
        signals=(replace(take.signals, end=new_end) if take.signals is not None else None),
    )


def apply_hybrid_retry_completion_integrity(result, source_takes, context=None):
    """Return a HybridSessionCleanupResult with only strongly proven extra repairs."""
    if not result.kept or not result.semantic_decisions:
        return result

    source_tuple = tuple(sorted(source_takes, key=lambda item: (item.source_order, item.start, item.end, item.clip_id)))
    semantic = {str(cid): (str(label), float(conf)) for cid, label, conf in result.semantic_decisions}
    kept = list(sorted(result.kept, key=lambda item: (item.source_order, item.start, item.end, item.clip_id)))
    removed_ids: set[str] = set()
    diagnostics = []

    for index, take in enumerate(tuple(kept)):
        winner = _safe_failed_retry(take, tuple(kept), semantic, context)
        if winner is not None:
            removed_ids.add(take.clip_id)
            diagnostics.append({
                "clip_id": take.clip_id,
                "reason": "semantic_failed_cross_group_retry_covered",
                "winner_clip_id": winner.clip_id,
            })
            continue
        previous = kept[index - 1] if index > 0 else None
        following = kept[index + 1] if index + 1 < len(kept) else None
        if _safe_short_alternate_debris(take, previous, following, semantic):
            removed_ids.add(take.clip_id)
            diagnostics.append({
                "clip_id": take.clip_id,
                "reason": "semantic_short_alternate_covered_by_neighbors",
            })
            continue
        winner = _safe_full_alternate_retry(take, tuple(kept), index, semantic, context)
        if winner is not None:
            removed_ids.add(take.clip_id)
            diagnostics.append({
                "clip_id": take.clip_id,
                "reason": "semantic_reset_backed_full_alternate_retry",
                "winner_clip_id": winner.clip_id,
            })

    survivors = [take for take in kept if take.clip_id not in removed_ids]

    # Completion repair is intentionally after retry cleanup and uses the original source
    # adjacency, so a failed tail cannot be mistaken for ordinary removed retry material.
    deleted_semantic = {
        take.clip_id: take
        for take in source_tuple
        if semantic.get(take.clip_id, ("", 0.0))[0] in {"failed", "bts"}
        and semantic.get(take.clip_id, ("", 0.0))[1] >= 0.85
    }
    repaired = []
    for take in survivors:
        candidate = take
        immediate = [
            tail for tail in deleted_semantic.values()
            if tail.source_asset_id == take.source_asset_id
            and 0.0 <= tail.start - take.end <= 0.12
            and tail.duration_sec <= 3.5
            and not tail.complete_idea
        ]
        if len(immediate) == 1:
            trimmed = _trim_parallel_clause_before_failed_tail(take, immediate[0])
            if trimmed is not None:
                diagnostics.append({
                    "clip_id": take.clip_id,
                    "reason": "completion_preserving_rollback_before_failed_tail",
                    "failed_tail_clip_id": immediate[0].clip_id,
                    "original_end": take.end,
                    "result_end": trimmed.end,
                    "original_text": take.text,
                    "result_text": trimmed.text,
                })
                candidate = trimmed
        repaired.append(candidate)

    if not removed_ids and not diagnostics:
        return result

    deleted_ids = {take.clip_id for take in result.deleted} | removed_ids
    deleted = tuple(take for take in source_tuple if take.clip_id in deleted_ids)
    return type(result)(
        kept=tuple(repaired),
        deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=tuple(result.diagnostics) + ({"hybrid_retry_completion_integrity": diagnostics},),
        semantic_decisions=result.semantic_decisions,
    )


def install_hybrid_retry_completion_integrity() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_hybrid_retry_completion_integrity", False):
        return

    def apply_with_retry_completion(*args, **kwargs):
        if args:
            source_takes = tuple(args[0])
            context = args[1] if len(args) > 1 else kwargs.get("context")
            result = original(*args, **kwargs)
        else:
            source_takes = tuple(kwargs.get("takes") or ())
            context = kwargs.get("context")
            result = original(**kwargs)
        return apply_hybrid_retry_completion_integrity(result, source_takes, context)

    apply_with_retry_completion._cutsell_hybrid_retry_completion_integrity = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_retry_completion
