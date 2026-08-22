"""Round 9 final authority for orphan failed prefixes with proven distant continuations.

Round 9 exposed a segmentation drift that the Round 8 guard could not see: the losing
open prefix still ended at 334.24, but its discarded continuation was reconstructed at
340.38 instead of 335.88.  The old 3.5-second adjacency guard therefore failed even
though upstream ``hybrid_cross_group_retry_integrity`` had already proven that the
continuation was fully covered by the earlier authoritative delivery.

This module does not simply widen the time window.  Beyond the original short-gap rule,
it removes an orphan prefix only when all of the following are true:
- the selected prefix is semantically failed with high confidence and ends open;
- a later discarded continuation is explicitly tied by upstream cross-group retry
  diagnostics to an earlier selected authoritative peer;
- the continuation remains within a conservative same-session window;
- prefix + continuation and the authoritative peer substantially cover one another;
- critical numeric and negation facts remain compatible.

No words, timestamps, retry groups, or semantic relationships are invented here.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .contracts import DraftClip
from . import final_draft_retry_integrity as base
from . import round8_retry_reconciliation as round8

_ROUND8_ORIGINAL = round8.suppress_orphan_failed_open_prefix


def _proven_cross_group_authorities(diagnostics: dict) -> dict[str, str]:
    """Map discarded retry clip -> already-proven authoritative peer clip."""
    out: dict[str, tuple[float, float, str]] = {}
    for chunk in diagnostics.get("hybrid_editorial_chunks") or ():
        if not isinstance(chunk, dict):
            continue
        for item in chunk.get("hybrid_cross_group_retry_integrity") or ():
            if not isinstance(item, dict):
                continue
            if item.get("reason") != "cross_group_semantic_retry_covered_by_authoritative_delivery":
                continue
            clip_id = str(item.get("clip_id") or "")
            peer_id = str(item.get("strongest_peer_clip_id") or "")
            if not clip_id or not peer_id:
                continue
            coverage = float(item.get("coverage") or 0.0)
            peer_coverage = float(item.get("strongest_peer_coverage") or 0.0)
            semantic_confidence = float(item.get("semantic_confidence") or 0.0)
            if coverage < 0.80 or peer_coverage < 0.80 or semantic_confidence < 0.75:
                continue
            score = (coverage, peer_coverage, peer_id)
            current = out.get(clip_id)
            if current is None or score[:2] > current[:2]:
                out[clip_id] = score
    return {clip_id: value[2] for clip_id, value in out.items()}


def suppress_orphan_failed_open_prefix_v2(
    selected: Iterable[DraftClip],
    discarded: Iterable[DraftClip],
    diagnostics: dict,
    *,
    maximum_proven_continuation_gap_sec: float = 12.0,
    maximum_prior_gap_sec: float = 45.0,
) -> tuple[tuple[DraftClip, ...], tuple[DraftClip, ...], tuple[dict, ...]]:
    """Extend the Round 8 orphan-prefix guard using explicit upstream retry authority."""
    selected_tuple = tuple(selected)
    discarded_tuple = tuple(discarded)

    # Preserve the original short-gap behavior exactly.  Use the frozen original
    # reference so installation of this module cannot recurse back into itself.
    short_selected, short_discarded, short_audit = _ROUND8_ORIGINAL(
        selected_tuple,
        discarded_tuple,
        diagnostics,
    )
    if short_audit:
        return short_selected, short_discarded, short_audit

    selected_list = list(sorted(selected_tuple, key=lambda c: (c.source_order, c.start, c.end, c.clip_id)))
    discarded_list = list(sorted(discarded_tuple, key=lambda c: (c.source_order, c.start, c.end, c.clip_id)))
    failures = base._semantic_failures(diagnostics)
    authorities = _proven_cross_group_authorities(diagnostics)
    selected_by_id = {clip.clip_id: clip for clip in selected_list}
    removed_ids: set[str] = set()
    audit: list[dict] = []

    for prefix in selected_list:
        failure_conf = failures.get(prefix.clip_id, 0.0)
        if failure_conf < 0.80 or not base._is_open_text(prefix.text):
            continue

        options = []
        for continuation in discarded_list:
            if continuation.source_asset_id != prefix.source_asset_id:
                continue
            if float(continuation.start) < float(prefix.end):
                continue
            continuation_gap = float(continuation.start) - float(prefix.end)
            if continuation_gap > maximum_proven_continuation_gap_sec:
                continue

            peer_id = authorities.get(continuation.clip_id)
            if not peer_id:
                continue
            prior = selected_by_id.get(peer_id)
            if prior is None or prior.clip_id == prefix.clip_id or prior.clip_id in removed_ids:
                continue
            if prior.source_asset_id != prefix.source_asset_id or float(prior.end) > float(prefix.start):
                continue
            prior_gap = float(prefix.start) - float(prior.end)
            if prior_gap > maximum_prior_gap_sec:
                continue

            combined_text = f"{prefix.text} {continuation.text}".strip()
            shared, combined_cov, prior_cov = base._coverage(combined_text, prior.text)
            if shared < 7 or combined_cov < 0.35 or prior_cov < 0.35:
                continue
            if not round8._critical_compatible(combined_text, prior.text):
                continue

            options.append((
                combined_cov,
                prior_cov,
                shared,
                -continuation_gap,
                -prior_gap,
                continuation,
                prior,
            ))

        if not options:
            continue

        combined_cov, prior_cov, shared, neg_cont_gap, _, continuation, prior = max(
            options, key=lambda item: item[:5]
        )
        removed_ids.add(prefix.clip_id)
        audit.append({
            "reason": "orphan_failed_open_prefix_yields_via_proven_cross_group_continuation",
            "removed_clip_id": prefix.clip_id,
            "discarded_continuation_clip_id": continuation.clip_id,
            "prior_winner_clip_id": prior.clip_id,
            "failed_prefix_confidence": round(failure_conf, 4),
            "continuation_gap_sec": round(-neg_cont_gap, 4),
            "combined_shared_content_tokens": shared,
            "combined_coverage": round(combined_cov, 4),
            "prior_coverage": round(prior_cov, 4),
            "removed_text": prefix.text,
            "continuation_text": continuation.text,
            "prior_text": prior.text,
        })

    if not removed_ids:
        return tuple(selected_list), tuple(discarded_list), ()

    removed = [clip for clip in selected_list if clip.clip_id in removed_ids]
    selected_out = tuple(clip for clip in selected_list if clip.clip_id not in removed_ids)
    existing = {clip.clip_id for clip in discarded_list}
    discarded_out = tuple(discarded_list + [
        replace(clip, selected=False)
        for clip in removed
        if clip.clip_id not in existing
    ])
    return selected_out, discarded_out, tuple(audit)


def install_round9_orphan_prefix_integrity() -> None:
    current = round8.suppress_orphan_failed_open_prefix
    if getattr(current, "_cutsell_round9_orphan_prefix_integrity", False):
        return

    def suppress_with_round9_authority(selected, discarded, diagnostics, **kwargs):
        short_selected, short_discarded, short_audit = _ROUND8_ORIGINAL(
            selected,
            discarded,
            diagnostics,
            **kwargs,
        )
        if short_audit:
            return short_selected, short_discarded, short_audit
        return suppress_orphan_failed_open_prefix_v2(
            selected,
            discarded,
            diagnostics,
        )

    suppress_with_round9_authority._cutsell_round9_orphan_prefix_integrity = True
    round8.suppress_orphan_failed_open_prefix = suppress_with_round9_authority
