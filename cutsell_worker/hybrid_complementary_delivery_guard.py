"""Final Hybrid authority for complementary human-style deliveries.

Two conservative cases are handled after every Hybrid wrapper has finished:

1. A complete delivery deleted specifically by cross-group retry collapse may be
   restored when it contains a meaningful audience-facing tail that the authoritative
   peer does not cover. This prevents lexical overlap from erasing a complementary
   sub-delivery that a human editor would keep and combine.
2. If one or more Hybrid windows were unavailable, an undecided incomplete take may
   yield to a nearby *earlier* complete delivery when it clearly restarts the same idea,
   preserves critical numeric/explicit-negation facts, and begins with strong lexical
   overlap. This complements the existing later-retry fallback without making ordinary
   incomplete continuations destructive.

No benchmark ids, timestamps, phrases, or Gold data are embedded here.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from .contracts import CandidateTake

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "de", "del",
    "el", "en", "es", "esta", "este", "for", "from", "in", "is", "it", "la", "las", "lo",
    "los", "me", "mi", "mis", "of", "on", "or", "para", "pero", "por", "porque", "que",
    "se", "so", "su", "sus", "that", "the", "this", "to", "un", "una", "was", "we",
    "with", "y", "yo",
})
_EXPLICIT_NEGATION = frozenset({"no", "not", "never", "nunca", "sin", "without"})


def _canon(token: str) -> str:
    raw = unicodedata.normalize("NFKD", str(token or "").casefold())
    return "".join(char for char in raw if not unicodedata.combining(char))


def _lexeme(token: str) -> str:
    token = _canon(token)
    if len(token) >= 5 and token.isalpha() and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _ordered_content(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (_lexeme(item) for item in _TOKEN_RE.findall(str(text or "")))
        if len(token) >= 3 and token not in _STOP
    )


def _content(text: str) -> set[str]:
    return set(_ordered_content(text))


def _critical(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(str(text or "")):
        token = _canon(raw)
        if token in _EXPLICIT_NEGATION:
            out.add("__negation__")
        if any(ch.isdigit() for ch in token):
            out.add(token)
    return out


def _gap(left: CandidateTake, right: CandidateTake) -> float:
    if left.source_asset_id != right.source_asset_id:
        return float("inf")
    if left.end <= right.start:
        return max(0.0, float(right.start) - float(left.end))
    if right.end <= left.start:
        return max(0.0, float(left.start) - float(right.end))
    return 0.0


def _semantic_map(rows: Iterable[tuple[str, str, float]]) -> dict[str, tuple[str, float]]:
    return {
        str(clip_id): (str(label), float(confidence))
        for clip_id, label, confidence in rows
    }


def _cross_group_deleted_ids(diagnostics: Iterable[dict]) -> set[str]:
    ids: set[str] = set()
    for row in diagnostics:
        if not isinstance(row, dict):
            continue
        entries = row.get("hybrid_cross_group_retry_integrity")
        if not isinstance(entries, list):
            continue
        for item in entries:
            if isinstance(item, dict) and item.get("clip_id"):
                ids.add(str(item["clip_id"]))
    return ids


def _restore_complementary_cross_group_deletions(
    kept: tuple[CandidateTake, ...],
    deleted: tuple[CandidateTake, ...],
    semantic: dict[str, tuple[str, float]],
    cross_group_deleted: set[str],
    *,
    maximum_gap_sec: float = 45.0,
) -> tuple[set[str], list[dict]]:
    """Restore only cross-group deletions with a material unique tail.

    A candidate must be a complete delivery of at least three seconds and have an
    authoritative winner/keep peer. The peer must overlap enough to explain why the
    cross-group guard considered them retries, but at least 15% of the candidate's
    content lexemes must remain unique. One unique lexeme is sufficient for short
    complete deliveries because descriptive qualifiers can be editorially meaningful.
    """
    restore_ids: set[str] = set()
    rows: list[dict] = []
    for candidate in deleted:
        if candidate.clip_id not in cross_group_deleted:
            continue
        if not bool(candidate.complete_idea) or candidate.duration_sec < 3.0:
            continue
        label, confidence = semantic.get(candidate.clip_id, ("", 0.0))
        if label not in {"alternate", "failed"} or confidence < 0.65:
            continue
        own = _content(candidate.text)
        if len(own) < 4:
            continue

        best = None
        best_shared = 0
        best_coverage = 0.0
        best_unique: set[str] = set()
        for peer in kept:
            if peer.source_asset_id != candidate.source_asset_id:
                continue
            if _gap(candidate, peer) > maximum_gap_sec:
                continue
            peer_label, peer_conf = semantic.get(peer.clip_id, ("", 0.0))
            if peer_label not in {"winner", "keep"} or peer_conf < 0.75:
                continue
            peer_content = _content(peer.text)
            shared = len(own & peer_content)
            coverage = shared / max(1, len(own))
            unique = own - peer_content
            if shared > best_shared or (shared == best_shared and coverage > best_coverage):
                best = peer
                best_shared = shared
                best_coverage = coverage
                best_unique = unique

        if best is None or best_shared < 2 or best_coverage < 0.50:
            continue
        unique_fraction = len(best_unique) / max(1, len(own))
        if not best_unique or unique_fraction < 0.15:
            continue

        restore_ids.add(candidate.clip_id)
        rows.append({
            "clip_id": candidate.clip_id,
            "peer_clip_id": best.clip_id,
            "reason": "restore_complete_complementary_delivery_with_unique_tail",
            "semantic_label": label,
            "semantic_confidence": round(confidence, 4),
            "shared_content_tokens": best_shared,
            "coverage": round(best_coverage, 4),
            "unique_content_tokens": sorted(best_unique),
            "unique_fraction": round(unique_fraction, 4),
        })
    return restore_ids, rows


def _delete_unavailable_prior_restarts(
    kept: tuple[CandidateTake, ...],
    semantic: dict[str, tuple[str, float]],
    *,
    maximum_prior_gap_sec: float = 8.0,
) -> tuple[set[str], list[dict]]:
    """Delete an undecided incomplete restart after a nearby complete delivery.

    This is intentionally stricter than ordinary semantic dedupe. The candidate must be
    incomplete, undecided by Hybrid, start shortly after the complete peer, share at
    least four content lexemes with it, have >=45% candidate coverage, and restart with
    strong overlap in its first eight content lexemes. Critical numbers and explicit
    negations must already be present in the complete peer.
    """
    decided_ids = set(semantic)
    delete_ids: set[str] = set()
    rows: list[dict] = []
    ordered = tuple(sorted(kept, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))

    for candidate in ordered:
        if candidate.clip_id in decided_ids or bool(candidate.complete_idea):
            continue
        own_ordered = _ordered_content(candidate.text)
        own = set(own_ordered)
        if len(own) < 6:
            continue
        best = None
        best_shared = 0
        best_coverage = 0.0
        best_prefix_shared = 0

        for peer in ordered:
            if peer.clip_id == candidate.clip_id:
                continue
            if peer.source_asset_id != candidate.source_asset_id:
                continue
            if not bool(peer.complete_idea):
                continue
            if peer.end > candidate.start:
                continue
            gap = float(candidate.start) - float(peer.end)
            if gap < 0.0 or gap > maximum_prior_gap_sec:
                continue
            if not _critical(candidate.text).issubset(_critical(peer.text)):
                continue
            peer_content = _content(peer.text)
            shared = len(own & peer_content)
            coverage = shared / max(1, len(own))
            prefix = tuple(own_ordered[:8])
            prefix_shared = sum(1 for token in prefix if token in peer_content)
            prefix_ratio = prefix_shared / max(1, len(prefix))
            if shared < 4 or coverage < 0.45:
                continue
            if prefix_shared < 3 or prefix_ratio < 0.40:
                continue
            if coverage > best_coverage or (coverage == best_coverage and shared > best_shared):
                best = peer
                best_shared = shared
                best_coverage = coverage
                best_prefix_shared = prefix_shared

        if best is None:
            continue
        delete_ids.add(candidate.clip_id)
        rows.append({
            "clip_id": candidate.clip_id,
            "prior_complete_clip_id": best.clip_id,
            "reason": "hybrid_unavailable_incomplete_restart_yields_to_prior_complete_delivery",
            "shared_content_tokens": best_shared,
            "coverage": round(best_coverage, 4),
            "prefix_shared_tokens": best_prefix_shared,
            "gap_sec": round(float(candidate.start) - float(best.end), 3),
        })
    return delete_ids, rows


def install_hybrid_complementary_delivery_guard() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_hybrid_complementary_delivery_guard", False):
        return

    def apply_with_complementary_delivery_guard(*args, **kwargs):
        if args:
            source_takes = tuple(args[0])
            result = original(source_takes, *args[1:], **kwargs)
        else:
            source_takes = tuple(kwargs.get("takes") or ())
            call_kwargs = dict(kwargs)
            call_kwargs["takes"] = source_takes
            result = original(**call_kwargs)

        semantic = _semantic_map(result.semantic_decisions)
        cross_group_deleted = _cross_group_deleted_ids(result.diagnostics)
        restore_ids, restore_rows = _restore_complementary_cross_group_deletions(
            tuple(result.kept),
            tuple(result.deleted),
            semantic,
            cross_group_deleted,
        )

        kept_ids = {take.clip_id for take in result.kept} | restore_ids
        kept = tuple(take for take in source_takes if take.clip_id in kept_ids)

        delete_ids: set[str] = set()
        delete_rows: list[dict] = []
        if result.requested_chunk_count > result.available_chunk_count:
            delete_ids, delete_rows = _delete_unavailable_prior_restarts(kept, semantic)
            if delete_ids:
                kept = tuple(take for take in kept if take.clip_id not in delete_ids)

        if not restore_rows and not delete_rows:
            return result

        final_kept_ids = {take.clip_id for take in kept}
        deleted = tuple(take for take in source_takes if take.clip_id not in final_kept_ids)
        diagnostics = tuple(result.diagnostics) + ({
            "hybrid_complementary_delivery_guard": {
                "restored": restore_rows,
                "deleted_unavailable_prior_restarts": delete_rows,
            },
            "restored_ids": sorted(restore_ids),
            "deleted_ids": sorted(delete_ids),
        },)
        return type(result)(
            kept=kept,
            deleted=deleted,
            requested_chunk_count=result.requested_chunk_count,
            available_chunk_count=result.available_chunk_count,
            diagnostics=diagnostics,
            semantic_decisions=result.semantic_decisions,
        )

    apply_with_complementary_delivery_guard._cutsell_hybrid_complementary_delivery_guard = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_complementary_delivery_guard
