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

A third rule resolves the otherwise ambiguous case where Hybrid marks multiple members
of one already-proven retry group as equally strong winners. In that narrow case, prefer
the complete delivery with materially greater audience-facing information coverage when
it still strongly overlaps the local winner and preserves every critical numeric fact.

This module never creates retry groups and never deletes unique story material. It only
changes the winner inside a group that deterministic grouping already proved contains
competing deliveries of the same idea.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "de", "del",
    "el", "en", "es", "esta", "este", "for", "from", "in", "is", "it", "la", "las", "lo",
    "los", "me", "mi", "mis", "of", "on", "or", "para", "pero", "por", "porque", "que",
    "se", "so", "su", "sus", "that", "the", "this", "to", "un", "una", "was", "we",
    "with", "y", "yo",
})


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

    if len(eligible) != 1:
        return None
    return eligible[0][0]


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(str(text or ""))}


def _content_tokens(text: str) -> set[str]:
    return {
        token for token in _tokens(text)
        if len(token) >= 3 and token not in _STOP
    }


def _critical_numeric_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if any(ch.isdigit() for ch in token)}


def _semantic_overlap(left_text: str, right_text: str) -> float:
    left = _content_tokens(left_text)
    right = _content_tokens(right_text)
    if len(left) < 3 or len(right) < 3:
        return 0.0
    shared = len(left & right)
    return shared / max(1, min(len(left), len(right)))


def _prefer_information_rich_tied_winner(
    members,
    semantic_decisions,
    selected_clip_id: str,
    *,
    winner_confidence_floor: float = 0.85,
    confidence_tolerance: float = 0.06,
    minimum_overlap: float = 0.50,
    minimum_unique_tokens: int = 2,
    minimum_information_growth: float = 0.18,
):
    """Resolve multiple high-confidence winners by information coverage.

    This is intentionally narrow. It only runs inside an already-proven retry group,
    only compares complete high-confidence semantic winners/keeps, and refuses to move
    away from the local winner if the richer peer would drop a critical numeric fact.
    """
    selected = next((member for member in members if member.clip_id == selected_clip_id), None)
    if selected is None or not bool(getattr(selected, "complete_idea", True)):
        return None

    selected_label, selected_conf = semantic_decisions.get(selected_clip_id, ("", 0.0))
    selected_conf = float(selected_conf)
    if selected_label not in {"winner", "keep"} or selected_conf < winner_confidence_floor:
        return None

    selected_content = _content_tokens(selected.text)
    if len(selected_content) < 4:
        return None
    selected_critical = _critical_numeric_tokens(selected.text)

    candidates = []
    for peer in members:
        if peer.clip_id == selected_clip_id or not bool(getattr(peer, "complete_idea", True)):
            continue
        label, confidence = semantic_decisions.get(peer.clip_id, ("", 0.0))
        confidence = float(confidence)
        if label not in {"winner", "keep"} or confidence < winner_confidence_floor:
            continue
        if confidence + confidence_tolerance < selected_conf:
            continue
        peer_content = _content_tokens(peer.text)
        if len(peer_content) < 4:
            continue
        shared = len(selected_content & peer_content)
        overlap = shared / max(1, min(len(selected_content), len(peer_content)))
        if overlap < minimum_overlap:
            continue
        if not selected_critical.issubset(_critical_numeric_tokens(peer.text)):
            continue
        unique = peer_content - selected_content
        growth = (len(peer_content) - len(selected_content)) / max(1, len(selected_content))
        if len(unique) < minimum_unique_tokens or growth < minimum_information_growth:
            continue
        candidates.append((peer, confidence, len(peer_content), len(unique), overlap))

    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (item[1], item[2], item[3], item[4], getattr(item[0], "duration_sec", 0.0)),
        reverse=True,
    )
    best = candidates[0]
    if len(candidates) > 1:
        second = candidates[1]
        if best[1:4] == second[1:4]:
            return None
    return best[0].clip_id


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
        ranked=(),
        *,
        winner_confidence=0.85,
        semantic_delete_recommended=None,
    ):
        # D-082: `original` (pipeline._semantic_best_take) now takes `ranked`
        # and returns a 3-tuple (selected, preferred, reason) -- passed
        # through unchanged here; this wrapper's own three extra checks
        # below are unmodified and still layer on top, in the same order,
        # as an additional, independently-tested safety net.
        selected, preferred, reason = original(
            members,
            semantic_decisions,
            local_selected_clip_id,
            ranked,
            winner_confidence=winner_confidence,
            semantic_delete_recommended=semantic_delete_recommended,
        )

        critical_peer = _prefer_complete_peer_with_preserved_critical_facts(
            members,
            semantic_decisions,
            selected,
        )
        if critical_peer is not None:
            return critical_peer, critical_peer, "critical_peer_preserved_facts"

        if preferred is None and selected == local_selected_clip_id:
            richer = _prefer_information_rich_tied_winner(
                members,
                semantic_decisions,
                selected,
                winner_confidence_floor=winner_confidence,
            )
            if richer is not None:
                return richer, richer, "information_rich_tied_winner"

        if preferred is not None or selected != local_selected_clip_id:
            return selected, preferred, reason

        safer = _prefer_clear_nonfailed_peer(
            members,
            semantic_decisions,
            local_selected_clip_id,
        )
        if safer is None:
            return selected, preferred, reason
        return safer, safer, "clear_nonfailed_peer_preferred"

    semantic_best_take_with_integrity._cutsell_semantic_best_take_integrity = True
    pipeline._semantic_best_take = semantic_best_take_with_integrity
