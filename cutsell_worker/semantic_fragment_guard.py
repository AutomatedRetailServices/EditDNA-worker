"""Conservative post-Hybrid guard for structurally obvious failed speech debris.

Benchmark #40 proved that the semantic judge can correctly label tiny false starts and
broken repeated deliveries as ``failed`` while the cleanup stage still keeps them because
there is no visual corroboration. That fail-open policy is desirable for normal speech,
but not for transcript structures that are independently incompatible with a finished
creator delivery.

This guard therefore adds *textual structure* as the second piece of evidence. It only
acts after Hybrid has already labelled a candidate failed/BTS with medium-high confidence,
and only for tiny/open fragments, filler BTS, recording-process self-talk, or severe
repetition pathology. A micro ``alternate`` is also removable when it sits inside a
provider-confirmed failure cluster beside a much fuller winner; isolated short hooks
remain fail-open.
"""
from __future__ import annotations

import re
from typing import Iterable

from .contracts import CandidateTake

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*\s*$")
_BTS_SELF_TALK_RE = re.compile(
    r"\btrying\s+to\s+(?:say|remember|stay|keep)\b|"
    r"\b(?:stay|staying|keep|keeping)\s+in\s+character\b|"
    r"\bwhat\s+am\s+i\s+(?:trying\s+to\s+)?say\b|"
    r"\bi\s+(?:can'?t|cannot)\s+(?:talk|speak)\b",
    re.IGNORECASE,
)
_OPEN_TAIL = frozenset({
    "a", "an", "and", "are", "as", "at", "because", "been", "being", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "if", "in", "into",
    "is", "of", "on", "or", "so", "than", "that", "the", "to", "was", "were", "when",
    "which", "while", "who", "with", "without",
    "a", "al", "como", "con", "cuando", "de", "del", "el", "en", "era", "es", "fue",
    "ha", "han", "la", "las", "los", "para", "pero", "por", "porque", "que", "si",
    "sin", "un", "una", "y",
})
_FILLER_PHRASES = frozenset({
    "you know", "i mean", "um", "uh", "erm", "eh", "okay so", "ok so",
    "o sea", "este", "pues",
})
_SUBJECT_PRONOUNS = frozenset({
    "i", "you", "he", "she", "it", "we", "they",
    "yo", "tu", "tú", "el", "él", "ella", "nosotros", "nosotras", "ellos", "ellas",
})
_COMMON_VERB_LIKE = frozenset({
    "am", "are", "is", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "should", "may", "might",
    "think", "know", "feel", "look", "like", "want", "need", "love", "make", "made",
    "say", "said", "see", "saw", "get", "got", "go", "went",
    "soy", "eres", "es", "somos", "son", "era", "estoy", "esta", "está", "estamos",
    "tengo", "tiene", "tenemos", "hago", "hace", "puedo", "puede", "quiero", "quiere",
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


def _short_unfinished_fragment(take: CandidateTake) -> bool:
    tokens = _tokens(take.text)
    if not tokens or _has_sentence_end(take.text):
        return False
    return take.duration_sec <= 3.50 and len(tokens) <= 7


def _pronoun_collision_fragment(take: CandidateTake) -> bool:
    """Detect a compact ASR-visible subject collision from a broken restart.

    Example from Benchmark 49: ``I people It was very funny``. We deliberately require
    pronoun + non-verb token + pronoun at the opening so ordinary speech such as
    ``I think it works`` remains fail-open.
    """
    tokens = _tokens(take.text)
    if _has_sentence_end(take.text) or take.duration_sec > 3.50 or not (4 <= len(tokens) <= 7):
        return False
    first, middle, third = tokens[:3]
    return (
        first in _SUBJECT_PRONOUNS
        and third in _SUBJECT_PRONOUNS
        and middle not in _SUBJECT_PRONOUNS
        and middle not in _COMMON_VERB_LIKE
    )


def _open_fragment(take: CandidateTake) -> bool:
    tokens = _tokens(take.text)
    if not tokens or _has_sentence_end(take.text):
        return False
    stripped = str(take.text or "").strip()
    punctuation_open = stripped.endswith((",", ":", ";", "-", "–", "—"))
    return (
        take.duration_sec <= 4.0
        and len(tokens) <= 9
        and (tokens[-1] in _OPEN_TAIL or punctuation_open)
    )


def _repetition_pathology(take: CandidateTake) -> bool:
    tokens = _tokens(take.text)
    if len(tokens) < 5:
        return False
    for width in range(1, min(5, len(tokens) // 2) + 1):
        span = width * 2
        for index in range(len(tokens) - span + 1):
            if tokens[index : index + width] == tokens[index + width : index + span]:
                if width > 1:
                    return True
                if index + 2 < len(tokens) and tokens[index] == tokens[index + 2]:
                    return True
    return False


def _filler_bts(take: CandidateTake) -> bool:
    normalized = " ".join(_tokens(take.text))
    return take.duration_sec <= 1.8 and normalized in _FILLER_PHRASES


def _recording_process_bts(take: CandidateTake) -> bool:
    return take.duration_sec <= 4.0 and bool(_BTS_SELF_TALK_RE.search(str(take.text or "")))


def _temporal_gap(left: CandidateTake, right: CandidateTake) -> float:
    if left.source_asset_id != right.source_asset_id:
        return float("inf")
    if left.end <= right.start:
        return max(0.0, right.start - left.end)
    if right.end <= left.start:
        return max(0.0, left.start - right.end)
    return 0.0


def _alternate_micro_in_failure_cluster(
    take: CandidateTake,
    label: str,
    confidence: float,
    semantic: dict[str, tuple[str, float]],
    take_map: dict[str, CandidateTake],
    *,
    failed_neighbor_gap_sec: float = 8.0,
    winner_gap_sec: float = 45.0,
) -> bool:
    if str(label) != "alternate" or float(confidence) < 0.72:
        return False
    if take.complete_idea or not _micro_fragment(take):
        return False

    failed_neighbor = any(
        other_id != take.clip_id
        and other.source_asset_id == take.source_asset_id
        and other_label == "failed"
        and other_confidence >= 0.74
        and _temporal_gap(take, other) <= failed_neighbor_gap_sec
        for other_id, other in take_map.items()
        for other_label, other_confidence in (semantic.get(other_id, ("", 0.0)),)
    )
    if not failed_neighbor:
        return False

    return any(
        other_id != take.clip_id
        and other.source_asset_id == take.source_asset_id
        and other_label == "winner"
        and other_confidence >= 0.90
        and other.duration_sec >= max(8.0, take.duration_sec * 5.0)
        and _temporal_gap(take, other) <= winner_gap_sec
        for other_id, other in take_map.items()
        for other_label, other_confidence in (semantic.get(other_id, ("", 0.0)),)
    )


def _structural_reason(take: CandidateTake, label: str, confidence: float) -> str | None:
    label = str(label)
    confidence = float(confidence)

    if _repetition_pathology(take):
        if label == "failed" and confidence >= 0.65:
            return "semantic_failed_repetition_pathology"
        if (
            label == "alternate"
            and confidence >= 0.60
            and take.duration_sec <= 7.0
            and not _has_sentence_end(take.text)
        ):
            return "semantic_nonwinner_repetition_pathology"

    if label == "failed":
        if confidence >= 0.74 and _micro_fragment(take):
            return "semantic_failed_micro_fragment"
        if confidence >= 0.84 and _pronoun_collision_fragment(take):
            return "semantic_failed_pronoun_collision_fragment"
        # Preserve the established fail-open contract for generic complete-looking short
        # speech at 0.85; only stronger 0.86+ semantic failure may use brevity alone.
        if confidence >= 0.86 and _short_unfinished_fragment(take):
            return "semantic_failed_short_fragment"
        if confidence >= 0.80 and _open_fragment(take):
            return "semantic_failed_open_fragment"

    if label == "bts" and confidence >= 0.84:
        if _filler_bts(take) or _micro_fragment(take) or _recording_process_bts(take):
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
    take_map = {take.clip_id: take for take in kept_tuple}
    removed_ids: set[str] = set()
    diagnostics: list[dict] = []

    for take in kept_tuple:
        label, confidence = semantic.get(take.clip_id, ("", 0.0))
        reason = _structural_reason(take, label, confidence)
        if reason is None and _alternate_micro_in_failure_cluster(
            take,
            label,
            confidence,
            semantic,
            take_map,
        ):
            reason = "semantic_nonwinner_micro_failure_cluster"
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
        if args:
            source_takes = tuple(args[0])
            call_args = (source_takes, *args[1:])
            result = original(*call_args, **kwargs)
        else:
            source_takes = tuple(kwargs.get("takes") or ())
            call_kwargs = dict(kwargs)
            call_kwargs["takes"] = source_takes
            result = original(**call_kwargs)

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
