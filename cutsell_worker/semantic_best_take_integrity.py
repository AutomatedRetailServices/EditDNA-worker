"""Conservative semantic tie-break for proven retry groups.

Hybrid can occasionally identify the locally highest-scoring take as a clear failed/BTS
attempt while a single peer in the *same deterministic retry group* is a high-confidence
keep.  The baseline Best-Take ranker must remain authoritative for ordinary ambiguity,
but it should not knowingly select the failed delivery when the semantic evidence has
one unmistakable non-failed alternative.

This module never creates retry groups and never deletes unique story material.  It only
changes the winner inside a group that deterministic grouping already proved contains
competing deliveries of the same idea.
"""
from __future__ import annotations


def _prefer_clear_nonfailed_peer(
    members,
    semantic_decisions,
    local_selected_clip_id: str,
    *,
    failed_confidence: float = 0.80,
    peer_confidence: float = 0.85,
):
    local_label, local_confidence = semantic_decisions.get(
        local_selected_clip_id, ("", 0.0)
    )
    if local_label not in {"failed", "bts"} or float(local_confidence) < failed_confidence:
        return None

    eligible = []
    for member in members:
        if member.clip_id == local_selected_clip_id:
            continue
        label, confidence = semantic_decisions.get(member.clip_id, ("", 0.0))
        if label in {"winner", "keep"} and float(confidence) >= peer_confidence:
            eligible.append((member.clip_id, float(confidence)))

    # Ambiguity remains fail-open.  We only override when there is exactly one clear
    # audience-facing peer inside the already-proven retry family.
    if len(eligible) != 1:
        return None
    return eligible[0][0]


def install_semantic_best_take_integrity() -> None:
    from . import pipeline

    original = pipeline._semantic_best_take
    if getattr(original, "_cutsell_semantic_best_take_integrity", False):
        return

    def semantic_best_take_with_integrity(
        members,
        semantic_decisions,
        local_selected_clip_id,
        *,
        winner_confidence=0.85,
    ):
        selected, preferred = original(
            members,
            semantic_decisions,
            local_selected_clip_id,
            winner_confidence=winner_confidence,
        )
        if preferred is not None or selected != local_selected_clip_id:
            return selected, preferred

        safer = _prefer_clear_nonfailed_peer(
            members,
            semantic_decisions,
            local_selected_clip_id,
        )
        if safer is None:
            return selected, preferred
        return safer, safer

    semantic_best_take_with_integrity._cutsell_semantic_best_take_integrity = True
    pipeline._semantic_best_take = semantic_best_take_with_integrity
