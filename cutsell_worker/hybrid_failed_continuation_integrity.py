"""Remove continuations that belong to a Hybrid-confirmed failed split retry.

Attempt reconstruction can split one bad delivery into a failed prefix plus an immediate
continuation. If Hybrid sees the prefix as failed but the continuation falls into an
unavailable semantic chunk, the continuation can survive and reconstruct the very retry
we intended to remove.

This pass is conservative: the failed prefix must be incomplete, the continuation must be
immediate and not have an authoritative winner/keep label, and the combined failed chain
must be substantially covered by one nearby high-confidence winner/keep delivery with the
same critical negation/number facts.
"""
from __future__ import annotations

import re
from typing import Iterable

from .contracts import CandidateTake

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "for", "from",
    "has", "have", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "so",
    "that", "the", "this", "to", "was", "we", "were", "what", "with", "you", "your",
    "al", "como", "con", "cuando", "de", "del", "el", "en", "ella", "ellas", "ellos",
    "es", "esta", "este", "la", "las", "le", "les", "lo", "los", "mi", "mis", "o",
    "para", "pero", "por", "porque", "que", "se", "si", "su", "sus", "un", "una",
    "unos", "unas", "y", "yo",
})
_NEGATION = frozenset({
    "no", "not", "never", "nunca", "sin", "without", "nadie", "ningun", "ningún",
    "ninguna", "ninguno", "nobody", "none", "neither",
})
_NEGATION_CANONICAL = "__negation__"


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) >= 3 and token not in _STOP}


def _critical(text: str) -> set[str]:
    raw = str(text or "")
    out: set[str] = {f"num:{number}" for number in _NUMBER_RE.findall(raw)}
    if any(token in _NEGATION for token in _tokens(raw)):
        out.add(_NEGATION_CANONICAL)
    return out


def _combined_coverage(left: CandidateTake, right: CandidateTake, winner: CandidateTake) -> tuple[float, int, bool]:
    failed_content = _content(left.text) | _content(right.text)
    winner_content = _content(winner.text)
    shared = len(failed_content & winner_content)
    coverage = shared / max(1, len(failed_content))
    failed_critical = _critical(left.text) | _critical(right.text)
    critical_preserved = failed_critical.issubset(_critical(winner.text))
    return coverage, shared, critical_preserved


def collapse_failed_split_retry_continuations(
    kept: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
    *,
    maximum_continuation_gap_sec: float = 3.0,
    maximum_winner_gap_sec: float = 45.0,
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    kept_tuple = tuple(sorted(kept, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    semantic = {
        str(clip_id): (str(label), float(confidence))
        for clip_id, label, confidence in semantic_decisions
    }
    removed_ids: set[str] = set()
    diagnostics: list[dict] = []

    for index, failed in enumerate(kept_tuple):
        label, confidence = semantic.get(failed.clip_id, ("", 0.0))
        if label != "failed" or confidence < 0.80 or failed.complete_idea:
            continue

        continuation = None
        for other in kept_tuple[index + 1 :]:
            if other.source_asset_id != failed.source_asset_id:
                continue
            gap = other.start - failed.end
            if gap < 0:
                continue
            if gap > maximum_continuation_gap_sec:
                break
            other_label, other_confidence = semantic.get(other.clip_id, ("", 0.0))
            if other_label in {"winner", "keep"} and other_confidence >= 0.85:
                continue
            continuation = other
            break
        if continuation is None:
            continue

        winners = []
        for other in kept_tuple:
            if other.clip_id in {failed.clip_id, continuation.clip_id}:
                continue
            if other.source_asset_id != failed.source_asset_id:
                continue
            winner_label, winner_confidence = semantic.get(other.clip_id, ("", 0.0))
            if winner_label not in {"winner", "keep"} or winner_confidence < 0.90:
                continue
            if other.end <= failed.start:
                gap = failed.start - other.end
            elif continuation.end <= other.start:
                gap = other.start - continuation.end
            else:
                gap = 0.0
            if gap <= maximum_winner_gap_sec:
                winners.append(other)

        best = None
        for winner in winners:
            coverage, shared, critical_preserved = _combined_coverage(failed, continuation, winner)
            candidate = (coverage, shared, critical_preserved, winner)
            if best is None or (coverage, shared) > (best[0], best[1]):
                best = candidate
        if best is None:
            continue

        coverage, shared, critical_preserved, winner = best
        if shared < 6 or coverage < 0.50 or not critical_preserved:
            continue

        removed_ids.update({failed.clip_id, continuation.clip_id})
        diagnostics.append({
            "failed_clip_id": failed.clip_id,
            "continuation_clip_id": continuation.clip_id,
            "winner_clip_id": winner.clip_id,
            "reason": "failed_split_retry_covered_by_authoritative_winner",
            "failed_confidence": round(confidence, 4),
            "combined_coverage": round(coverage, 4),
            "shared_content_tokens": shared,
            "critical_preserved": True,
            "failed_text": failed.text,
            "continuation_text": continuation.text,
            "winner_text": winner.text,
        })

    survivors = tuple(take for take in kept_tuple if take.clip_id not in removed_ids)
    removed = tuple(take for take in kept_tuple if take.clip_id in removed_ids)
    return survivors, removed, tuple(diagnostics)


def install_hybrid_failed_continuation_integrity() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_hybrid_failed_continuation_integrity", False):
        return

    def apply_with_failed_continuation_integrity(*args, **kwargs):
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
        kept, extra_deleted, integrity_diagnostics = collapse_failed_split_retry_continuations(
            result.kept,
            result.semantic_decisions,
        )
        if not integrity_diagnostics:
            return result

        deleted_ids = {take.clip_id for take in result.deleted}
        deleted_ids.update(take.clip_id for take in extra_deleted)
        deleted = tuple(take for take in source_takes if take.clip_id in deleted_ids)
        diagnostics = tuple(result.diagnostics) + ({
            "hybrid_failed_continuation_integrity": list(integrity_diagnostics),
            "deleted_ids": sorted(take.clip_id for take in extra_deleted),
        },)
        return type(result)(
            kept=kept,
            deleted=deleted,
            requested_chunk_count=result.requested_chunk_count,
            available_chunk_count=result.available_chunk_count,
            diagnostics=diagnostics,
            semantic_decisions=result.semantic_decisions,
        )

    apply_with_failed_continuation_integrity._cutsell_hybrid_failed_continuation_integrity = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_failed_continuation_integrity
