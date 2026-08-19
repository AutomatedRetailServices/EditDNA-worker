"""Conservative post-Hybrid guard for structurally obvious failed speech debris.

Benchmark #40 proved that the semantic judge can correctly label tiny false starts and
broken repeated deliveries as ``failed`` while the cleanup stage still keeps them because
there is no visual corroboration.  That fail-open policy is desirable for normal speech,
but not for transcript structures that are independently incompatible with a finished
creator delivery.

This guard therefore adds *textual structure* as the second piece of evidence.  It only
acts after Hybrid has already labelled a candidate failed/BTS with medium-high confidence,
and only for tiny/open fragments, filler BTS, or severe repetition pathology.  Unique
short hooks labelled keep/winner/alternate are untouched.
"""
from __future__ import annotations

import re
from typing import Iterable

from .contracts import CandidateTake

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*\s*$")
_OPEN_TAIL = frozenset({
    # English connectors / auxiliaries that cannot normally finish a take.
    "a", "an", "and", "are", "as", "at", "because", "been", "being", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "if", "in", "into",
    "is", "of", "on", "or", "so", "than", "that", "the", "to", "was", "were", "when",
    "which", "while", "who", "with", "without",
    # Spanish connectors / auxiliaries.
    "a", "al", "como", "con", "cuando", "de", "del", "el", "en", "era", "es", "fue",
    "ha", "han", "la", "las", "los", "para", "pero", "por", "porque", "que", "si",
    "sin", "un", "una", "y",
})
_FILLER_PHRASES = frozenset({
    "you know", "i mean", "um", "uh", "erm", "eh", "okay so", "ok so",
    "o sea", "este", "pues",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _has_sentence_end(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(str(text or "").strip()))


def _micro_fragment(take: CandidateTake) -> bool:
    tokens = _tokens(take.text)
    if not tokens or _has_sentence_end(take.text):
        return False
    return take.duration_sec <= 1.60 and len(tokens) <= 3


def _open_fragment(take: CandidateTake) -> bool:
    tokens = _tokens(take.text)
    if not tokens or _has_sentence_end(take.text):
        return False
    stripped = str(take.text or "").strip()
    punctuation_open = stripped.endswith((",", ":", ";", "-", "–", "—"))
    return (
        take.duration_sec <= 3.0
        and len(tokens) <= 7
        and (tokens[-1] in _OPEN_TAIL or punctuation_open)
    )


def _repetition_pathology(take: CandidateTake) -> bool:
    tokens = _tokens(take.text)
    if len(tokens) < 5:
        return False
    # Triple token repetition catches e.g. ``non-gmo non-gmo non-gmo`` after ASR
    # normalization.  Also catch an immediately repeated 2-token phrase.
    for index in range(len(tokens) - 2):
        if tokens[index] == tokens[index + 1] == tokens[index + 2]:
            return True
    for index in range(len(tokens) - 3):
        if tokens[index : index + 2] == tokens[index + 2 : index + 4]:
            return True
    return False


def _filler_bts(take: CandidateTake) -> bool:
    normalized = " ".join(_tokens(take.text))
    return take.duration_sec <= 1.8 and normalized in _FILLER_PHRASES


def _structural_reason(take: CandidateTake, label: str, confidence: float) -> str | None:
    label = str(label)
    confidence = float(confidence)
    if label == "failed" and confidence >= 0.80:
        if _micro_fragment(take):
            return "semantic_failed_micro_fragment"
        if _open_fragment(take):
            return "semantic_failed_open_fragment"
        if confidence >= 0.82 and _repetition_pathology(take):
            return "semantic_failed_repetition_pathology"
    if label == "bts" and confidence >= 0.88:
        if _filler_bts(take) or _micro_fragment(take):
            return "semantic_bts_micro_debris"
    return None


def remove_semantic_fragment_debris(
    kept: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    semantic = {
        str(clip_id): (str(label), float(confidence))
        for clip_id, label, confidence in semantic_decisions
    }
    removed_ids: set[str] = set()
    diagnostics: list[dict] = []

    for take in kept_tuple:
        label, confidence = semantic.get(take.clip_id, ("", 0.0))
        reason = _structural_reason(take, label, confidence)
        if reason is None:
            continue
        removed_ids.add(take.clip_id)
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": reason,
            "semantic_label": label,
            "semantic_confidence": round(confidence, 4),
            "duration_sec": round(take.duration_sec, 3),
            "text": take.text,
        })

    survivors = tuple(take for take in kept_tuple if take.clip_id not in removed_ids)
    removed = tuple(take for take in kept_tuple if take.clip_id in removed_ids)
    return survivors, removed, tuple(diagnostics)


def install_semantic_fragment_guard() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_semantic_fragment_guard", False):
        return

    def apply_with_semantic_fragment_guard(*args, **kwargs):
        result = original(*args, **kwargs)
        if not result.kept or not result.semantic_decisions:
            return result

        kept, extra_deleted, guard_diagnostics = remove_semantic_fragment_debris(
            result.kept,
            result.semantic_decisions,
        )
        if not guard_diagnostics:
            return result

        deleted_ids = {take.clip_id for take in result.deleted}
        deleted_ids.update(take.clip_id for take in extra_deleted)
        # Preserve source order from the original input whenever available.
        source_takes = tuple(args[0]) if args else tuple(kwargs.get("takes") or ())
        if source_takes:
            deleted = tuple(take for take in source_takes if take.clip_id in deleted_ids)
        else:
            deleted = tuple(result.deleted) + tuple(extra_deleted)

        diagnostics = tuple(result.diagnostics) + ({
            "semantic_fragment_guard": list(guard_diagnostics),
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

    apply_with_semantic_fragment_guard._cutsell_semantic_fragment_guard = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_semantic_fragment_guard
