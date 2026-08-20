"""Final post-Hybrid integrity for retries stranded across deterministic groups.

Real creator retries can survive every earlier pass when ASR/attempt reconstruction splits
semantically equivalent deliveries into different groups. Hybrid may still correctly label
one delivery as ``failed``/``alternate`` and another as ``winner``/``keep``. This final
pass makes that semantic relationship actionable after story-coverage restoration and
other fail-open guards have finished.

The pass is deliberately conservative:
- it only considers Hybrid ``failed`` or ``alternate`` candidates;
- it requires nearby high-confidence ``winner``/``keep`` peers in the same source;
- the peers must collectively cover a substantial share of the candidate's content;
- critical meaning such as negation and numbers must also be preserved by the peers;
- genuinely unique audience-facing material remains fail-open.
"""
from __future__ import annotations

import re
from typing import Iterable

from .contracts import CandidateTake

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "for", "from",
    "has", "have", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "so",
    "that", "the", "this", "to", "was", "we", "were", "what", "with", "you", "your",
    "al", "como", "con", "cuando", "de", "del", "el", "en", "ella", "ellas", "ellos",
    "es", "esta", "este", "la", "las", "le", "les", "lo", "los", "mi", "mis", "o",
    "para", "pero", "por", "porque", "que", "se", "si", "su", "sus", "un", "una",
    "unos", "unas", "y", "yo",
})
_CRITICAL = frozenset({"no", "not", "never", "nunca", "sin", "without"})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> set[str]:
    return {
        token for token in _tokens(text)
        if len(token) >= 3 and token not in _STOP
    }


def _critical(text: str) -> set[str]:
    return {
        token for token in _tokens(text)
        if token in _CRITICAL or any(ch.isdigit() for ch in token)
    }


def _gap(left: CandidateTake, right: CandidateTake) -> float:
    if left.source_asset_id != right.source_asset_id:
        return float("inf")
    if left.end <= right.start:
        return max(0.0, right.start - left.end)
    if right.end <= left.start:
        return max(0.0, left.start - right.end)
    return 0.0


def _authoritative_peers(
    candidate: CandidateTake,
    kept: tuple[CandidateTake, ...],
    semantic: dict[str, tuple[str, float]],
    *,
    maximum_gap_sec: float = 45.0,
) -> tuple[CandidateTake, ...]:
    peers = []
    for other in kept:
        if other.clip_id == candidate.clip_id:
            continue
        if other.source_asset_id != candidate.source_asset_id:
            continue
        if _gap(candidate, other) > maximum_gap_sec:
            continue
        label, confidence = semantic.get(other.clip_id, ("", 0.0))
        if label not in {"winner", "keep"} or float(confidence) < 0.85:
            continue
        peers.append(other)
    return tuple(sorted(peers, key=lambda item: (_gap(candidate, item), item.start, item.clip_id)))


def _covered_by_authoritative_peers(
    candidate: CandidateTake,
    peers: tuple[CandidateTake, ...],
) -> tuple[bool, dict]:
    own = _content(candidate.text)
    if len(own) < 2 or not peers:
        return False, {}

    # Use at most the four closest authoritative deliveries. This allows a complete idea
    # to be represented by a winner plus its immediate continuation without letting an
    # entire later story accidentally erase unique material.
    local = peers[:4]
    union: set[str] = set()
    strongest_shared = 0
    strongest_peer = None
    for peer in local:
        peer_content = _content(peer.text)
        shared = len(own & peer_content)
        if shared > strongest_shared:
            strongest_shared = shared
            strongest_peer = peer.clip_id
        union.update(peer_content)

    shared_union = len(own & union)
    coverage = shared_union / max(1, len(own))
    critical = _critical(candidate.text)
    peer_critical: set[str] = set()
    for peer in local:
        peer_critical.update(_critical(peer.text))
    critical_preserved = critical.issubset(peer_critical)

    if candidate.duration_sec <= 6.0:
        enough = shared_union >= 2 and coverage >= 0.50
    elif candidate.duration_sec <= 14.0:
        enough = shared_union >= 4 and coverage >= 0.45
    else:
        enough = shared_union >= 5 and coverage >= 0.45

    # Require at least one direct relationship, not just diffuse topic overlap across
    # several unrelated paragraphs.
    direct = strongest_shared >= (2 if candidate.duration_sec <= 6.0 else 3)
    return bool(enough and direct and critical_preserved), {
        "coverage": round(coverage, 4),
        "shared_union": shared_union,
        "content_token_count": len(own),
        "strongest_shared": strongest_shared,
        "strongest_peer_clip_id": strongest_peer,
        "critical_tokens": sorted(critical),
        "critical_preserved": critical_preserved,
        "peer_clip_ids": [peer.clip_id for peer in local],
    }


def collapse_cross_group_semantic_retries(
    kept: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    """Remove only semantically-proven retries already covered by authoritative peers."""
    kept_tuple = tuple(sorted(kept, key=lambda item: (item.source_order, item.start, item.end, item.clip_id)))
    semantic = {
        str(clip_id): (str(label), float(confidence))
        for clip_id, label, confidence in semantic_decisions
    }
    removed_ids: set[str] = set()
    diagnostics: list[dict] = []

    for candidate in kept_tuple:
        label, confidence = semantic.get(candidate.clip_id, ("", 0.0))
        if label not in {"failed", "alternate"} or confidence < 0.75:
            continue
        peers = _authoritative_peers(candidate, kept_tuple, semantic)
        covered, evidence = _covered_by_authoritative_peers(candidate, peers)
        if not covered:
            continue
        removed_ids.add(candidate.clip_id)
        diagnostics.append({
            "clip_id": candidate.clip_id,
            "reason": "cross_group_semantic_retry_covered_by_authoritative_delivery",
            "semantic_label": label,
            "semantic_confidence": round(confidence, 4),
            "text": candidate.text,
            **evidence,
        })

    survivors = tuple(take for take in kept_tuple if take.clip_id not in removed_ids)
    removed = tuple(take for take in kept_tuple if take.clip_id in removed_ids)
    return survivors, removed, tuple(diagnostics)


def install_hybrid_cross_group_retry_integrity() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_hybrid_cross_group_retry_integrity", False):
        return

    def apply_with_cross_group_retry_integrity(*args, **kwargs):
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
        kept, extra_deleted, guard_diagnostics = collapse_cross_group_semantic_retries(
            result.kept,
            result.semantic_decisions,
        )
        if not guard_diagnostics:
            return result

        deleted_ids = {take.clip_id for take in result.deleted}
        deleted_ids.update(take.clip_id for take in extra_deleted)
        deleted = tuple(take for take in source_takes if take.clip_id in deleted_ids)
        diagnostics = tuple(result.diagnostics) + ({
            "hybrid_cross_group_retry_integrity": list(guard_diagnostics),
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

    apply_with_cross_group_retry_integrity._cutsell_hybrid_cross_group_retry_integrity = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_cross_group_retry_integrity
