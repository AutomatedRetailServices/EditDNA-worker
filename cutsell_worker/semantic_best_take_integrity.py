"""Conservative semantic tie-break for proven retry groups.

Hybrid can occasionally identify the locally highest-scoring take as a clear failed/BTS
attempt while a single peer in the *same deterministic retry group* is a high-confidence
keep. The baseline Best-Take ranker must remain authoritative for ordinary ambiguity,
but it should not knowingly select the failed delivery when the semantic evidence has
one unmistakable non-failed alternative.

A second integrity rule protects critical facts. When deterministic grouping has already
proved two complete deliveries are competing takes of the same idea, a shorter semantic
winner may not replace a complete peer if doing so drops numeric facts carried by that
peer. Numeric facts are audience-facing information, not stylistic slack.

This module never creates retry groups and never deletes unique story material. It only
changes the winner inside a group that deterministic grouping already proved contains
competing deliveries of the same idea.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)


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

    # Ambiguity remains fail-open. We only override when there is exactly one clear
    # audience-facing peer inside the already-proven retry family.
    if len(eligible) != 1:
        return None
    return eligible[0][0]


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(str(text or ""))}


def _critical_numeric_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if any(ch.isdigit() for ch in token)}


def _semantic_overlap(left_text: str, right_text: str) -> float:
    left = {token for token in _tokens(left_text) if len(token) >= 3}
    right = {token for token in _tokens(right_text) if len(token) >= 3}
    if len(left) < 3 or len(right) < 3:
        return 0.0
    shared = len(left & right)
    return shared / max(1, min(len(left), len(right)))


def _prefer_complete_peer_with_preserved_critical_facts(
    members,
    semantic_decisions,
    selected_clip_id: str,
    *,
    selected_confidence_floor: float = 0.85,
    peer_confidence_floor: float = 0.75,
    minimum_overlap: float = 0.45,
):
    selected = next((member for member in members if member.clip_id == selected_clip_id), None)
    if selected is None or not bool(getattr(selected, "complete_idea", True)):
        return None

    selected_label, selected_confidence = semantic_decisions.get(selected_clip_id, ("", 0.0))
    if selected_label not in {"winner", "keep"} or float(selected_confidence) < selected_confidence_floor:
        return None

    selected_critical = _critical_numeric_tokens(selected.text)
    candidates = []
    for peer in members:
        if peer.clip_id == selected_clip_id:
            continue
        if not bool(getattr(peer, "complete_idea", True)):
            continue
        label, confidence = semantic_decisions.get(peer.clip_id, ("", 0.0))
        if label not in {"alternate", "winner", "keep"} or float(confidence) < peer_confidence_floor:
            continue
        peer_critical = _critical_numeric_tokens(peer.text)
        missing_from_selected = peer_critical - selected_critical
        if not missing_from_selected:
            continue
        overlap = _semantic_overlap(peer.text, selected.text)
        if overlap < minimum_overlap:
            continue
        candidates.append((peer, float(confidence), overlap, missing_from_selected))

    # Stay conservative: only one complete peer may claim unique critical facts.
    if len(candidates) != 1:
        return None
    peer, _confidence, _overlap, _missing = candidates[0]
    return peer.clip_id


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

        critical_peer = _prefer_complete_peer_with_preserved_critical_facts(
            members,
            semantic_decisions,
            selected,
        )
        if critical_peer is not None:
            return critical_peer, critical_peer

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
