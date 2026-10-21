"""Preserve a full semantic alternate when it is complementary, not redundant.

Hybrid retry-completion integrity may correctly recognize a complete alternate as a retry
of one strong winner, but lexical overlap alone is not proof that the alternate carries
no additional audience-facing information. This authority runs after Hybrid cleanup and
before final session grouping. It restores only alternates that were explicitly removed
as ``semantic_reset_backed_full_alternate_retry`` and that contribute material unique
content relative to the named winner.

The winner and complementary alternate are then split into singleton retry groups so the
ordinary one-winner Best Take reducer cannot collapse the intended composite delivery.
No benchmark ids, timestamps, phrases, or Human Gold data are embedded here.
"""
from __future__ import annotations

from contextvars import ContextVar
import re
import sys
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

_SPLIT_IDS: ContextVar[frozenset[str]] = ContextVar(
    "cutsell_semantic_complementary_split_ids", default=frozenset()
)


def _canon(token: str) -> str:
    raw = unicodedata.normalize("NFKD", str(token or "").casefold())
    return "".join(ch for ch in raw if not unicodedata.combining(ch))


def _content(text: str) -> set[str]:
    return {
        token
        for token in (_canon(item) for item in _TOKEN_RE.findall(str(text or "")))
        if len(token) >= 3 and token not in _STOP
    }


def _semantic_map(rows: Iterable[tuple[str, str, float]]) -> dict[str, tuple[str, float]]:
    best: dict[str, tuple[str, float]] = {}
    for clip_id, label, confidence in rows:
        clip_id = str(clip_id)
        confidence = float(confidence)
        current = best.get(clip_id)
        if current is None or confidence > current[1]:
            best[clip_id] = (str(label), confidence)
    return best


def _completion_removed_pairs(diagnostics: Iterable[dict]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for row in diagnostics:
        if not isinstance(row, dict):
            continue
        entries = row.get("hybrid_retry_completion_integrity")
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            if item.get("reason") != "semantic_reset_backed_full_alternate_retry":
                continue
            clip_id = str(item.get("clip_id") or "")
            winner_id = str(item.get("winner_clip_id") or "")
            if clip_id and winner_id:
                pairs.append((clip_id, winner_id))
    return tuple(dict.fromkeys(pairs))


def _gap(left: CandidateTake, right: CandidateTake) -> float:
    if left.source_asset_id != right.source_asset_id:
        return float("inf")
    if left.end <= right.start:
        return max(0.0, float(right.start) - float(left.end))
    if right.end <= left.start:
        return max(0.0, float(left.start) - float(right.end))
    return 0.0


def complementary_relation(
    alternate: CandidateTake,
    winner: CandidateTake,
    semantic: dict[str, tuple[str, float]],
    *,
    minimum_confidence: float = 0.75,
    minimum_shared_tokens: int = 4,
    minimum_overlap: float = 0.42,
    maximum_overlap: float = 0.88,
    minimum_unique_tokens: int = 2,
    minimum_unique_fraction: float = 0.16,
    maximum_gap_sec: float = 30.0,
) -> dict | None:
    label, confidence = semantic.get(alternate.clip_id, ("", 0.0))
    winner_label, winner_confidence = semantic.get(winner.clip_id, ("", 0.0))
    if label != "alternate" or confidence < minimum_confidence:
        return None
    if winner_label not in {"winner", "keep"} or winner_confidence < 0.85:
        return None
    if not bool(alternate.complete_idea) or alternate.duration_sec <= 6.0:
        return None
    if not bool(winner.complete_idea) or _gap(alternate, winner) > maximum_gap_sec:
        return None

    own = _content(alternate.text)
    peer = _content(winner.text)
    if len(own) < 5 or len(peer) < 5:
        return None
    shared = len(own & peer)
    coverage = shared / max(1, len(own))
    if shared < minimum_shared_tokens or coverage < minimum_overlap or coverage > maximum_overlap:
        return None
    unique = own - peer
    unique_fraction = len(unique) / max(1, len(own))
    if len(unique) < minimum_unique_tokens or unique_fraction < minimum_unique_fraction:
        return None
    return {
        "clip_id": alternate.clip_id,
        "winner_clip_id": winner.clip_id,
        "reason": "semantic_full_alternate_has_material_unique_information",
        "semantic_confidence": round(confidence, 4),
        "winner_confidence": round(winner_confidence, 4),
        "shared_content_tokens": shared,
        "alternate_coverage_by_winner": round(coverage, 4),
        "unique_content_tokens": sorted(unique),
        "unique_fraction": round(unique_fraction, 4),
    }


def _split_groups(groups, split_ids: set[str], natural_ids: tuple[str, ...]):
    order = {clip_id: i for i, clip_id in enumerate(natural_ids)}
    output: list[tuple[str, ...]] = []
    for raw_group in groups:
        group = tuple(str(item) for item in raw_group)
        hits = tuple(cid for cid in group if cid in split_ids)
        remainder = tuple(cid for cid in group if cid not in split_ids)
        if remainder:
            output.append(remainder)
        output.extend((cid,) for cid in hits)
    output.sort(key=lambda group: min(order.get(cid, 10**9) for cid in group))
    return tuple(output)


def install_hybrid_semantic_complementary_rescue() -> None:
    from . import hybrid_session_cleanup, session_boundaries

    original_hybrid = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if not getattr(original_hybrid, "_cutsell_semantic_complementary_rescue", False):
        def apply_with_semantic_complementary_rescue(*args, **kwargs):
            _SPLIT_IDS.set(frozenset())
            if args:
                source_takes = tuple(args[0])
                result = original_hybrid(source_takes, *args[1:], **kwargs)
            else:
                source_takes = tuple(kwargs.get("takes") or ())
                call_kwargs = dict(kwargs)
                call_kwargs["takes"] = source_takes
                result = original_hybrid(**call_kwargs)

            pairs = _completion_removed_pairs(result.diagnostics)
            if not pairs:
                return result
            by_id = {take.clip_id: take for take in source_takes}
            semantic = _semantic_map(result.semantic_decisions)
            restore_ids: set[str] = set()
            split_ids: set[str] = set()
            audit: list[dict] = []
            for alternate_id, winner_id in pairs:
                alternate = by_id.get(alternate_id)
                winner = by_id.get(winner_id)
                if alternate is None or winner is None:
                    continue
                relation = complementary_relation(alternate, winner, semantic)
                if relation is None:
                    continue
                restore_ids.add(alternate_id)
                split_ids.update((alternate_id, winner_id))
                audit.append(relation)

            if not restore_ids:
                return result
            kept_ids = {take.clip_id for take in result.kept} | restore_ids
            kept = tuple(take for take in source_takes if take.clip_id in kept_ids)
            deleted = tuple(take for take in source_takes if take.clip_id not in kept_ids)
            _SPLIT_IDS.set(frozenset(split_ids))
            diagnostics = tuple(result.diagnostics) + ({
                "hybrid_semantic_complementary_rescue": audit,
                "restored_ids": sorted(restore_ids),
                "split_group_clip_ids": sorted(split_ids),
            },)
            return type(result)(
                kept=kept,
                deleted=deleted,
                requested_chunk_count=result.requested_chunk_count,
                available_chunk_count=result.available_chunk_count,
                diagnostics=diagnostics,
                semantic_decisions=result.semantic_decisions,
            )

        apply_with_semantic_complementary_rescue._cutsell_semantic_complementary_rescue = True
        hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_semantic_complementary_rescue
        pipeline_module = sys.modules.get(f"{__package__}.pipeline")
        if pipeline_module is not None:
            pipeline_module.apply_hybrid_session_cleanup = apply_with_semantic_complementary_rescue

    original_grouping = session_boundaries.safe_group_takes_by_sessions
    if not getattr(original_grouping, "_cutsell_semantic_complementary_group_split", False):
        def group_with_semantic_complementary_split(*args, **kwargs):
            result = original_grouping(*args, **kwargs)
            takes = tuple(args[1]) if len(args) >= 2 else tuple(kwargs.get("takes") or ())
            split_ids = set(_SPLIT_IDS.get())
            _SPLIT_IDS.set(frozenset())
            if not takes or not split_ids:
                return result
            natural_ids = tuple(take.clip_id for take in takes)
            relevant = split_ids & set(natural_ids)
            if not relevant:
                return result
            groups = _split_groups(result.groups, relevant, natural_ids)
            if groups == tuple(result.groups):
                return result
            return type(result)(
                groups=groups,
                status=result.status,
                reason="; ".join(part for part in (
                    result.reason,
                    f"semantic_complementary_group_split:{len(relevant)}",
                ) if part),
            )

        group_with_semantic_complementary_split._cutsell_semantic_complementary_group_split = True
        session_boundaries.safe_group_takes_by_sessions = group_with_semantic_complementary_split
        pipeline_module = sys.modules.get(f"{__package__}.pipeline")
        if pipeline_module is not None:
            pipeline_module.safe_group_takes_by_sessions = group_with_semantic_complementary_split
