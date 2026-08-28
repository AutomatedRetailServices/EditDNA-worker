"""Explicit final Selection authority for Universal Clean Cut.

Critical Selection ownership must not depend on import-time wrapper order. This module
runs the final semantic membership passes in a fixed order immediately before speech
recovery and the Selection freeze:

1. stabilize complementary information-preserving retry families;
2. trim a proven internal spoken retake only when the later delivery fully recovers it;
3. arbitrate selected retry losers using final Hybrid + physical evidence;
4. keep semantic alternates available for manual SWAP rather than discarding them.

No Boundary operation is installed or invoked here. Ambiguity fails open.
"""
from __future__ import annotations

from dataclasses import replace

from .final_selection_retry_arbiter import apply_final_selection_retry_arbiter
from .post_selection_complementary_family_stabilizer import (
    apply_post_selection_complementary_family_stabilizer,
)
from .post_selection_internal_retake_trim import trim_selected_internal_retakes


def _apply_internal_retake_trim(draft):
    diagnostics = dict(draft.diagnostics or {})
    selected, audit = trim_selected_internal_retakes(draft.selected, diagnostics)
    if not audit:
        return draft
    diagnostics["post_selection_internal_retake_trim"] = list(audit)
    return replace(draft, selected=selected, diagnostics=diagnostics)


def _restore_semantic_alternates_to_swap(before_arbiter, after_arbiter):
    """Move physical/semantic alternates out of Selected but keep them swappable."""
    diagnostics = dict(after_arbiter.diagnostics or {})
    audit = diagnostics.get("final_selection_retry_arbiter") or ()
    swap_ids = {
        str(row.get("clip_id"))
        for row in audit
        if isinstance(row, dict)
        and float(row.get("alternate_confidence") or 0.0) >= 0.60
        and float(row.get("failed_confidence") or 0.0) < 0.80
        and row.get("clip_id")
    }
    if not swap_ids:
        return after_arbiter

    removed_by_id = {
        clip.clip_id: clip
        for clip in before_arbiter.selected
        if clip.clip_id in swap_ids
    }
    if not removed_by_id:
        return after_arbiter

    existing_alt_ids = {clip.clip_id for clip in after_arbiter.alternates}
    alternates = list(after_arbiter.alternates)
    for clip_id, clip in removed_by_id.items():
        if clip_id not in existing_alt_ids:
            alternates.append(replace(clip, selected=False))
    alternates.sort(key=lambda c: (c.source_order, float(c.start), float(c.end), c.clip_id))

    discarded = tuple(
        clip for clip in after_arbiter.discarded
        if clip.clip_id not in swap_ids
    )
    diagnostics["selection_swap_preserved_alternates"] = [
        {
            "clip_id": clip_id,
            "reason": "semantic_alternate_removed_from_auto_edit_but_preserved_for_swap",
        }
        for clip_id in sorted(removed_by_id)
    ]
    return replace(
        after_arbiter,
        alternates=tuple(alternates),
        discarded=discarded,
        diagnostics=diagnostics,
    )


def apply_selection_phase_authority(draft):
    """Execute final semantic Selection deterministically in one explicit location."""
    diagnostics = dict(draft.diagnostics or {})
    input_selected_count = len(draft.selected)

    draft = apply_post_selection_complementary_family_stabilizer(draft)
    draft = _apply_internal_retake_trim(draft)

    before_arbiter = draft
    draft = apply_final_selection_retry_arbiter(draft)
    draft = _restore_semantic_alternates_to_swap(before_arbiter, draft)

    diagnostics = dict(draft.diagnostics or {})
    diagnostics["selection_phase_authority"] = {
        "status": "executed",
        "input_selected_count": input_selected_count,
        "output_selected_count": len(draft.selected),
        "alternate_count": len(draft.alternates),
        "discarded_count": len(draft.discarded),
        "ordered_passes": [
            "post_selection_complementary_family_stabilizer",
            "post_selection_internal_retake_trim",
            "final_selection_retry_arbiter",
            "swap_alternate_preservation",
        ],
    }
    return replace(draft, diagnostics=diagnostics)
