"""Cross-group authority for an earlier truncated delivery superseded by a later complete retry.

Attempt reconstruction can occasionally mark a take as ``complete_idea=True`` even though
its transcript visibly ends mid-clause (for example on a bridge word or ellipsis). If a
later nearby high-confidence delivery strongly covers that earlier content, preserves all
numbers/negations, and materially completes the information, the earlier truncated take
is recording-process debris rather than a second audience-facing paragraph.

This module is deliberately not benchmark-specific. It exposes pure selection logic and
is installed separately only after regression validation.
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
_NEGATIONS = frozenset({"no", "not", "never", "nunca", "sin", "without"})
_BRIDGE_END = frozenset({
    "a", "al", "and", "because", "but", "by", "con", "de", "del", "for", "of", "para",
    "pero", "por", "porque", "que", "the", "to", "with", "y",
})


def _canon(token: str) -> str:
    raw = unicodedata.normalize("NFKD", str(token or "").casefold())
    return "".join(ch for ch in raw if not unicodedata.combining(ch))


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_canon(item) for item in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) >= 3 and token not in _STOP}


def _critical(text: str) -> set[str]:
    out = set()
    for token in _tokens(text):
        if token in _NEGATIONS:
            out.add("__negation__")
        if any(ch.isdigit() for ch in token):
            out.add(token)
    return out


def _visibly_truncated(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if raw.endswith("...") or raw.endswith("…"):
        return True
    tokens = _tokens(raw)
    if not tokens:
        return False
    return tokens[-1] in _BRIDGE_END and not re.search(r"[.!?][\"'”’)]*$", raw)


def _gap(left: CandidateTake, right: CandidateTake) -> float:
    if left.source_asset_id != right.source_asset_id:
        return float("inf")
    if left.end <= right.start:
        return max(0.0, float(right.start) - float(left.end))
    if right.end <= left.start:
        return max(0.0, float(left.start) - float(right.end))
    return 0.0


def truncated_retry_relation(
    earlier: CandidateTake,
    later: CandidateTake,
    *,
    maximum_gap_sec: float = 35.0,
    minimum_shared_tokens: int = 4,
    minimum_earlier_coverage: float = 0.45,
    minimum_later_unique_tokens: int = 3,
) -> dict | None:
    if later.start <= earlier.start or _gap(earlier, later) > maximum_gap_sec:
        return None
    if not _visibly_truncated(earlier.text):
        return None
    earlier_content = _content(earlier.text)
    later_content = _content(later.text)
    if len(earlier_content) < 5 or len(later_content) < 5:
        return None
    shared = len(earlier_content & later_content)
    coverage = shared / max(1, len(earlier_content))
    if shared < minimum_shared_tokens or coverage < minimum_earlier_coverage:
        return None
    if not _critical(earlier.text).issubset(_critical(later.text)):
        return None
    unique = later_content - earlier_content
    if len(unique) < minimum_later_unique_tokens:
        return None
    return {
        "clip_id": earlier.clip_id,
        "later_retry_clip_id": later.clip_id,
        "reason": "earlier_visibly_truncated_delivery_completed_by_later_retry",
        "shared_content_tokens": shared,
        "earlier_coverage": round(coverage, 4),
        "later_unique_content_tokens": sorted(unique),
    }


def collapse_truncated_cross_group_winners(
    kept: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    ordered = tuple(sorted(kept, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    semantic: dict[str, tuple[str, float]] = {}
    for clip_id, label, confidence in semantic_decisions:
        clip_id = str(clip_id)
        confidence = float(confidence)
        current = semantic.get(clip_id)
        if current is None or confidence > current[1]:
            semantic[clip_id] = (str(label), confidence)

    removed: set[str] = set()
    audit: list[dict] = []
    for earlier in ordered:
        label, confidence = semantic.get(earlier.clip_id, ("", 0.0))
        if label not in {"winner", "keep", "alternate"} or confidence < 0.75:
            continue
        matches = []
        for later in ordered:
            if later.clip_id == earlier.clip_id or later.source_asset_id != earlier.source_asset_id:
                continue
            later_label, later_confidence = semantic.get(later.clip_id, ("", 0.0))
            if later_label not in {"winner", "keep"} or later_confidence < 0.88:
                continue
            relation = truncated_retry_relation(earlier, later)
            if relation is not None:
                matches.append((relation["earlier_coverage"], later_confidence, later, relation))
        if not matches:
            continue
        matches.sort(key=lambda row: (row[0], row[1], row[2].start), reverse=True)
        best = matches[0]
        if len(matches) > 1 and matches[1][:2] == best[:2]:
            continue
        removed.add(earlier.clip_id)
        audit.append({
            **best[3],
            "semantic_label": label,
            "semantic_confidence": round(confidence, 4),
            "later_semantic_confidence": round(best[1], 4),
        })

    return (
        tuple(take for take in ordered if take.clip_id not in removed),
        tuple(take for take in ordered if take.clip_id in removed),
        tuple(audit),
    )


def install_cross_group_truncated_winner_authority() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_cross_group_truncated_winner_authority", False):
        return

    def apply_with_truncated_authority(*args, **kwargs):
        if args:
            source_takes = tuple(args[0])
            result = original(source_takes, *args[1:], **kwargs)
        else:
            source_takes = tuple(kwargs.get("takes") or ())
            call_kwargs = dict(kwargs)
            call_kwargs["takes"] = source_takes
            result = original(**call_kwargs)
        kept, removed, audit = collapse_truncated_cross_group_winners(
            result.kept, result.semantic_decisions
        )
        if not audit:
            return result
        deleted_ids = {take.clip_id for take in result.deleted} | {take.clip_id for take in removed}
        deleted = tuple(take for take in source_takes if take.clip_id in deleted_ids)
        return type(result)(
            kept=kept,
            deleted=deleted,
            requested_chunk_count=result.requested_chunk_count,
            available_chunk_count=result.available_chunk_count,
            diagnostics=tuple(result.diagnostics) + ({
                "cross_group_truncated_winner_authority": list(audit),
                "deleted_ids": [row["clip_id"] for row in audit],
            },),
            semantic_decisions=result.semantic_decisions,
        )

    apply_with_truncated_authority._cutsell_cross_group_truncated_winner_authority = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_truncated_authority
