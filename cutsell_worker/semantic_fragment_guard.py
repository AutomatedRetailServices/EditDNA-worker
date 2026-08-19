"""Conservative post-Hybrid guard for structurally obvious failed speech debris.

Benchmark #40 proved that the semantic judge can correctly label tiny false starts and
broken repeated deliveries as ``failed`` while the cleanup stage still keeps them because
there is no visual corroboration. That fail-open policy is desirable for normal speech,
but not for transcript structures that are independently incompatible with a finished
creator delivery.

This guard therefore adds *textual structure* as the second piece of evidence. It only
acts after Hybrid has already labelled a candidate failed/BTS with medium-high confidence,
and only for tiny/open fragments, filler BTS, or severe repetition pathology. Unique
short hooks labelled keep/winner are untouched.
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


def _short_unfinished_fragment(take: CandidateTake) -> bool:
    """A short provider-confirmed failure that is too small to be a finished delivery."""
    tokens = _tokens(take.text)
    if not tokens or _has_sentence_end(take.text):
        return False
    return take.duration_sec <= 3.50 and len(tokens) <= 7


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

    # A semantic non-winner plus an immediately repeated phrase is independent evidence
    # of a broken delivery even when the creator keeps eye contact. Check adjacent
    # n-grams up to five tokens so ASR-normalized compounds such as ``non-gmo non-gmo
    # non-gmo`` (``non gmo`` repeated) and longer restarts such as ``if they're not
    # eating if they're not eating`` are both recognized.
    for width in range(1, min(5, len(tokens) // 2) + 1):
        span = width * 2
        for index in range(len(tokens) - span + 1):
            if tokens[index : index + width] == tokens[index + width : index + span]:
                # One-word duplication is only structural when repeated at least
                # three times; two identical ordinary words can be emphatic speech.
                if width > 1:
                    return True
                if index + 2 < len(tokens) and tokens[index] == tokens[index + 2]:
                    return True
    return False


def _filler_bts(take: CandidateTake) -> bool:
    normalized = " ".join(_tokens(take.text))
    return take.duration_sec <= 1.8 and normalized in _FILLER_PHRASES


def _structural_reason(take: CandidateTake, label: str, confidence: float) -> str | None:
    label = str(label)
    confidence = float(confidence)

    # Exact phrase repetition is stronger textual corroboration than brevity. Benchmark
    # 44 showed the same broken delivery can fluctuate between ``failed`` and a low-
    # confidence ``alternate`` across overlapping Hybrid windows. Never apply this to a
    # winner/keep; a short low-confidence non-winner also has to lack sentence closure.
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
        # A two/three-word orphan plus a semantic-failure decision is enough independent
        # evidence at medium confidence. This catches fragments such as ``you're tired``
        # without touching valid short hooks labelled winner/alternate.
        if confidence >= 0.74 and _micro_fragment(take):
            return "semantic_failed_micro_fragment"
        # Slightly longer malformed fragments require substantially stronger semantic
        # confidence. This catches stranded self-talk such as ``trying to say in
        # character`` while keeping the guard fail-open for ordinary short speech.
        if confidence >= 0.86 and _short_unfinished_fragment(take):
            return "semantic_failed_short_fragment"
        if confidence >= 0.80 and _open_fragment(take):
            return "semantic_failed_open_fragment"

    if label == "bts" and confidence >= 0.84:
        if _filler_bts(take) or _micro_fragment(take) or _short_unfinished_fragment(take):
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
        # Snapshot the input before the wrapped function consumes a possible generator.
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
