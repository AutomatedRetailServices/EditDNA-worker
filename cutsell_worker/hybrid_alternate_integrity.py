"""Remove structurally stranded Hybrid alternates beside a clear final winner.

Hybrid intentionally treats ``alternate`` as fail-open because a different delivery can
contain unique story coverage. Benchmark 47 exposed a narrower case: an explicit lexical
self-correction may be split into a long punctuation-open prefix plus a tiny corrected
suffix, and both children can remain as independent alternates immediately before a
complete winner of the same idea.

This pass is deliberately conservative. It never removes winners/keeps and it requires a
nearby high-confidence winner. A longer alternate must be grammatically open *and* share
several meaningful content tokens with that winner. A tiny closed alternate can be
removed only when its meaningful token(s) are repeated in the winner and it immediately
follows one of those already-proven open alternates. This preserves valid short hooks and
unique story paragraphs while preventing malformed retry debris from becoming final
speech.
"""
from __future__ import annotations

import re
from typing import Iterable

from .contracts import CandidateTake

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*\s*$")
_CONTENT_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "for", "from",
    "has", "have", "i", "in", "is", "it", "of", "on", "or", "so", "that", "the", "this",
    "to", "was", "we", "were", "with", "you", "your", "de", "del", "el", "en", "es",
    "la", "las", "los", "para", "por", "que", "un", "una", "y",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) >= 3 and token not in _CONTENT_STOP}


def _sentence_closed(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(str(text or "").strip()))


def _punctuation_open(text: str) -> bool:
    return str(text or "").rstrip().endswith((",", ":", ";", "-", "–", "—"))


def _forward_gap(left: CandidateTake, right: CandidateTake) -> float:
    if left.source_asset_id != right.source_asset_id or right.start < left.end:
        return float("inf")
    return max(0.0, right.start - left.end)


def _nearby_winners(
    take: CandidateTake,
    semantic: dict[str, tuple[str, float]],
    take_map: dict[str, CandidateTake],
    *,
    maximum_gap_sec: float,
) -> tuple[CandidateTake, ...]:
    output = []
    for other_id, other in take_map.items():
        if other_id == take.clip_id:
            continue
        label, confidence = semantic.get(other_id, ("", 0.0))
        if label != "winner" or float(confidence) < 0.90:
            continue
        if _forward_gap(take, other) <= maximum_gap_sec:
            output.append(other)
    return tuple(sorted(output, key=lambda item: (item.start, item.end, item.clip_id)))


def suppress_stranded_hybrid_alternates(
    kept: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    semantic = {
        str(clip_id): (str(label), float(confidence))
        for clip_id, label, confidence in semantic_decisions
    }
    take_map = {take.clip_id: take for take in kept_tuple}
    removed_ids: set[str] = set()
    diagnostics: list[dict] = []

    # Pass 1: a long malformed alternate can be suppressed only when it is explicitly
    # open-ended and substantially overlaps one unique nearby semantic winner.
    for take in kept_tuple:
        label, confidence = semantic.get(take.clip_id, ("", 0.0))
        if label != "alternate" or confidence < 0.78:
            continue
        if take.duration_sec > 14.0 or _sentence_closed(take.text) or not _punctuation_open(take.text):
            continue
        winners = _nearby_winners(take, semantic, take_map, maximum_gap_sec=12.0)
        if len(winners) != 1:
            continue
        winner = winners[0]
        shared = _content_tokens(take.text) & _content_tokens(winner.text)
        if len(shared) < 3:
            continue
        removed_ids.add(take.clip_id)
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "semantic_alternate_open_retry_before_winner",
            "semantic_confidence": round(confidence, 4),
            "winner_clip_id": winner.clip_id,
            "shared_content": sorted(shared),
            "text": take.text,
        })

    # Pass 2: remove a tiny corrected suffix only when it follows one of the malformed
    # alternates above, precedes one unique winner, and its content is repeated there.
    # This catches a stranded ``priceless.`` after cutting ``worthless. Oh,`` without
    # deleting an isolated short punchline.
    for take in kept_tuple:
        if take.clip_id in removed_ids:
            continue
        label, confidence = semantic.get(take.clip_id, ("", 0.0))
        tokens = _content_tokens(take.text)
        if label != "alternate" or confidence < 0.70:
            continue
        if take.duration_sec > 1.5 or not tokens:
            continue
        prior_removed = [
            other for other in kept_tuple
            if other.clip_id in removed_ids
            and other.source_asset_id == take.source_asset_id
            and 0.0 <= take.start - other.end <= 3.0
        ]
        if not prior_removed:
            continue
        winners = _nearby_winners(take, semantic, take_map, maximum_gap_sec=8.0)
        if len(winners) != 1:
            continue
        winner = winners[0]
        if not tokens.issubset(_content_tokens(winner.text)):
            continue
        removed_ids.add(take.clip_id)
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "semantic_alternate_corrected_suffix_repeated_by_winner",
            "semantic_confidence": round(confidence, 4),
            "winner_clip_id": winner.clip_id,
            "text": take.text,
        })

    survivors = tuple(take for take in kept_tuple if take.clip_id not in removed_ids)
    removed = tuple(take for take in kept_tuple if take.clip_id in removed_ids)
    return survivors, removed, tuple(diagnostics)


def install_hybrid_alternate_integrity() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_hybrid_alternate_integrity", False):
        return

    def apply_with_hybrid_alternate_integrity(*args, **kwargs):
        if args:
            source_takes = tuple(args[0])
            result = original(source_takes, *args[1:], **kwargs)
        else:
            source_takes = tuple(kwargs.get("takes") or ())
            call_kwargs = dict(kwargs)
            call_kwargs["takes"] = source_takes
            result = original(**call_kwargs)

        if not result.kept or not result.semantic_decisions:
            return result
        kept, extra_deleted, guard_diagnostics = suppress_stranded_hybrid_alternates(
            result.kept,
            result.semantic_decisions,
        )
        if not guard_diagnostics:
            return result

        deleted_ids = {take.clip_id for take in result.deleted}
        deleted_ids.update(take.clip_id for take in extra_deleted)
        deleted = tuple(take for take in source_takes if take.clip_id in deleted_ids)
        diagnostics = tuple(result.diagnostics) + ({
            "hybrid_alternate_integrity": list(guard_diagnostics),
            "deleted_ids": [item["clip_id"] for item in guard_diagnostics],
        },)
        return type(result)(
            kept=kept,
            deleted=deleted,
            requested_chunk_count=result.requested_chunk_count,
            available_chunk_count=result.available_chunk_count,
            diagnostics=diagnostics,
            semantic_decisions=result.semantic_decisions,
        )

    apply_with_hybrid_alternate_integrity._cutsell_hybrid_alternate_integrity = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_hybrid_alternate_integrity
