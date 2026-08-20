"""Rescue a clean completion prefix from a high-confidence failed tail.

A creator may finish an idea with a short clean phrase and then immediately fumble. Hybrid
is correct to label the *whole* fragment failed, but deleting the entire fragment can
amputate the semantic completion of the previous delivery. This pass keeps only the clean
prefix before an independently visible repetition/collision pattern and leaves the failed
remainder discarded.

Example Gold case (Video 03):
    previous: "...te protegen te reparan"
    failed:   "la barrera cutánea te la te hace como"
    rescue:   "la barrera cutánea"

The pass is intentionally narrow and runs after all Hybrid cleanup guards.
"""
from __future__ import annotations

import re
from dataclasses import replace

from .contracts import CandidateTake

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_CONTENT_STOP = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with",
    "al", "de", "del", "el", "en", "la", "las", "lo", "los", "o", "para", "por", "un", "una", "y",
})
_SHORT_FUNCTION = frozenset({
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "yo", "tu", "tú", "el", "él", "ella", "me", "te", "se", "lo", "la", "le", "nos", "los", "las", "les",
})


def _norm(value: str) -> str:
    match = _TOKEN_RE.search(str(value or ""))
    return match.group(0).casefold() if match else ""


def _content_count(words) -> int:
    return sum(1 for word in words if len(_norm(word.text)) >= 3 and _norm(word.text) not in _CONTENT_STOP)


def _collision_index(words) -> int | None:
    """Locate the start of a compact repeated-function-word fumble.

    The repeated token must reappear with one short function word between it (e.g.
    ``te la te``). This is much narrower than generic word repetition and therefore does
    not treat ordinary rhetorical repetition as a fumble boundary.
    """
    normalized = tuple(_norm(word.text) for word in words)
    for index in range(1, len(normalized) - 2):
        first, middle, third = normalized[index : index + 3]
        if not first or first != third:
            continue
        if first not in _SHORT_FUNCTION or middle not in _SHORT_FUNCTION:
            continue
        return index
    return None


def _nearest_previous_kept(candidate: CandidateTake, kept: tuple[CandidateTake, ...], *, max_gap_sec: float = 3.0):
    prior = [
        take for take in kept
        if take.source_asset_id == candidate.source_asset_id
        and take.end <= candidate.start
        and candidate.start - take.end <= max_gap_sec
    ]
    if not prior:
        return None
    return max(prior, key=lambda take: (take.end, take.start))


def rescue_failed_completion_prefixes(
    kept: tuple[CandidateTake, ...],
    deleted: tuple[CandidateTake, ...],
    semantic_decisions: tuple[tuple[str, str, float], ...],
) -> tuple[tuple[CandidateTake, ...], tuple[dict, ...]]:
    semantic = {
        str(clip_id): (str(label), float(confidence))
        for clip_id, label, confidence in semantic_decisions
    }
    rescued = []
    diagnostics = []

    for candidate in deleted:
        label, confidence = semantic.get(candidate.clip_id, ("", 0.0))
        if label != "failed" or confidence < 0.90:
            continue
        if candidate.duration_sec > 5.0 or len(candidate.words) < 6:
            continue
        previous = _nearest_previous_kept(candidate, kept)
        if previous is None:
            continue

        collision = _collision_index(candidate.words)
        if collision is None or collision < 2:
            continue
        prefix_words = tuple(candidate.words[:collision])
        if len(prefix_words) < 2 or _content_count(prefix_words) < 2:
            continue

        prefix_end = float(prefix_words[-1].end)
        if prefix_end <= candidate.start or prefix_end >= candidate.end:
            continue
        prefix_text = " ".join(str(word.text).strip() for word in prefix_words).strip()
        child = replace(
            candidate,
            clip_id=f"{candidate.clip_id}_completion_prefix",
            end=prefix_end,
            text=prefix_text,
            words=prefix_words,
            signals=(
                replace(candidate.signals, end=prefix_end)
                if candidate.signals is not None else None
            ),
            complete_idea=True,
        )
        rescued.append(child)
        diagnostics.append({
            "source_clip_id": candidate.clip_id,
            "rescued_clip_id": child.clip_id,
            "reason": "failed_tail_clean_completion_prefix_before_collision",
            "semantic_confidence": round(confidence, 4),
            "previous_clip_id": previous.clip_id,
            "original_text": candidate.text,
            "rescued_text": child.text,
            "rescued_start": round(child.start, 3),
            "rescued_end": round(child.end, 3),
        })

    if not rescued:
        return kept, ()
    merged = tuple(sorted(
        (*kept, *rescued),
        key=lambda take: (take.source_order, take.start, take.end, take.clip_id),
    ))
    return merged, tuple(diagnostics)


def install_failed_prefix_completion_rescue() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_failed_prefix_completion_rescue", False):
        return

    def apply_with_completion_prefix_rescue(*args, **kwargs):
        result = original(*args, **kwargs)
        if not result.kept or not result.deleted or not result.semantic_decisions:
            return result
        kept, rescue_diagnostics = rescue_failed_completion_prefixes(
            tuple(result.kept),
            tuple(result.deleted),
            tuple(result.semantic_decisions),
        )
        if not rescue_diagnostics:
            return result
        diagnostics = tuple(result.diagnostics) + ({
            "failed_prefix_completion_rescue": list(rescue_diagnostics),
        },)
        return type(result)(
            kept=kept,
            deleted=result.deleted,
            requested_chunk_count=result.requested_chunk_count,
            available_chunk_count=result.available_chunk_count,
            diagnostics=diagnostics,
            semantic_decisions=result.semantic_decisions,
        )

    apply_with_completion_prefix_rescue._cutsell_failed_prefix_completion_rescue = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_completion_prefix_rescue
