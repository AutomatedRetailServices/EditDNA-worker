"""Bridge semantic complementary rescues into Composite Best Take.

The semantic rescue authority can restore a complete alternate that retry-completion
previously removed. Two follow-up responsibilities belong here:

1. Revoke rescues that are actually same-opening retries. A take that restarts with the
   same strong lexical opening as its named winner is not complementary merely because
   it contains a few unmatched tail tokens.
2. Normalize valid semantic rescues into the same ``peer_clip_id`` audit shape consumed
   by Composite Best Take, so a left rescue and a right rescue can jointly replace one
   monolithic winner.

This module is benchmark-agnostic. It uses only retry relations already established by
upstream authorities, lexical opening identity, and critical information preservation.
"""
from __future__ import annotations

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


def _canon(token: str) -> str:
    raw = unicodedata.normalize("NFKD", str(token or "").casefold())
    return "".join(ch for ch in raw if not unicodedata.combining(ch))


def _ordered_content(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (_canon(item) for item in _TOKEN_RE.findall(str(text or "")))
        if len(token) >= 3 and token not in _STOP
    )


def strong_same_opening(
    left_text: str,
    right_text: str,
    *,
    width: int = 4,
    minimum_shared_prefix: int = 3,
) -> bool:
    left = _ordered_content(left_text)
    right = _ordered_content(right_text)
    if len(left) < minimum_shared_prefix or len(right) < minimum_shared_prefix:
        return False
    width = min(width, len(left), len(right))
    if width < minimum_shared_prefix:
        return False
    shared = sum(1 for a, b in zip(left[:width], right[:width]) if a == b)
    return shared >= minimum_shared_prefix


def _semantic_rescue_rows(diagnostics: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in diagnostics:
        if not isinstance(row, dict):
            continue
        entries = row.get("hybrid_semantic_complementary_rescue")
        if isinstance(entries, list):
            rows.extend(item for item in entries if isinstance(item, dict))
    return rows


def reconcile_semantic_rescues(
    source_takes: tuple[CandidateTake, ...],
    kept: tuple[CandidateTake, ...],
    diagnostics: Iterable[dict],
) -> tuple[tuple[CandidateTake, ...], tuple[dict, ...], tuple[dict, ...]]:
    by_id = {take.clip_id: take for take in source_takes}
    kept_ids = {take.clip_id for take in kept}
    revoked: list[dict] = []
    normalized: list[dict] = []

    for row in _semantic_rescue_rows(diagnostics):
        clip_id = str(row.get("clip_id") or "")
        winner_id = str(row.get("winner_clip_id") or "")
        alternate = by_id.get(clip_id)
        winner = by_id.get(winner_id)
        if alternate is None or winner is None or clip_id not in kept_ids:
            continue

        if strong_same_opening(alternate.text, winner.text):
            kept_ids.discard(clip_id)
            revoked.append({
                "clip_id": clip_id,
                "winner_clip_id": winner_id,
                "reason": "semantic_rescue_revoked_strong_same_opening_retry",
            })
            continue

        normalized.append({
            "clip_id": clip_id,
            "peer_clip_id": winner_id,
            "reason": "semantic_complementary_rescue_normalized_for_composite_best_take",
            "semantic_label": "alternate",
            "semantic_confidence": float(row.get("semantic_confidence") or 0.0),
            "coverage": float(row.get("alternate_coverage_by_winner") or 0.0),
            "unique_content_tokens": list(row.get("unique_content_tokens") or ()),
            "unique_fraction": float(row.get("unique_fraction") or 0.0),
        })

    repaired = tuple(take for take in source_takes if take.clip_id in kept_ids)
    return repaired, tuple(revoked), tuple(normalized)


def install_hybrid_semantic_composite_bridge() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_semantic_composite_bridge", False):
        return

    def apply_with_semantic_composite_bridge(*args, **kwargs):
        if args:
            source_takes = tuple(args[0])
            result = original(source_takes, *args[1:], **kwargs)
        else:
            source_takes = tuple(kwargs.get("takes") or ())
            call_kwargs = dict(kwargs)
            call_kwargs["takes"] = source_takes
            result = original(**call_kwargs)

        kept, revoked, normalized = reconcile_semantic_rescues(
            source_takes,
            tuple(result.kept),
            result.diagnostics,
        )
        if not revoked and not normalized:
            return result

        # Semantic rescue has a private split-id context. If a rescue is revoked, remove
        # that clip from the pending split set so grouping does not manufacture a singleton.
        if revoked:
            try:
                from . import hybrid_semantic_complementary_rescue as rescue_module
                pending = set(rescue_module._SPLIT_IDS.get())
                pending.difference_update(str(row["clip_id"]) for row in revoked)
                rescue_module._SPLIT_IDS.set(frozenset(pending))
            except Exception:
                pass

        kept_ids = {take.clip_id for take in kept}
        deleted = tuple(take for take in source_takes if take.clip_id not in kept_ids)
        extra = {
            "hybrid_semantic_composite_bridge": {
                "revoked_same_opening_rescues": list(revoked),
                "normalized_composite_rescues": list(normalized),
            },
            # Composite Best Take already consumes this canonical guard shape.
            "hybrid_complementary_delivery_guard": {
                "restored": list(normalized),
                "deleted_unavailable_prior_restarts": [],
            },
            "deleted_ids": [str(row["clip_id"]) for row in revoked],
            "restored_ids": [str(row["clip_id"]) for row in normalized],
        }
        return type(result)(
            kept=kept,
            deleted=deleted,
            requested_chunk_count=result.requested_chunk_count,
            available_chunk_count=result.available_chunk_count,
            diagnostics=tuple(result.diagnostics) + (extra,),
            semantic_decisions=result.semantic_decisions,
        )

    apply_with_semantic_composite_bridge._cutsell_semantic_composite_bridge = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_semantic_composite_bridge

    pipeline_module = sys.modules.get(f"{__package__}.pipeline")
    if pipeline_module is not None:
        pipeline_module.apply_hybrid_session_cleanup = apply_with_semantic_composite_bridge
