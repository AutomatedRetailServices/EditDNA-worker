"""Remove an explicit lexical slip inside an otherwise unique delivery.

Some creator mistakes are not separate retakes. A single useful paragraph can contain a
one-word slip immediately followed by an audible correction, e.g. ``worthless. Oh,
priceless.`` Best-Take grouping cannot fix that without throwing away the unique story
around it.

This pass is intentionally narrow. It requires word timestamps, a correction marker
(``oh``/``sorry``), and strong morphology between the mistaken and corrected word (a
shared prefix or suffix of at least four characters). It then splits the original take
around only the mistaken word + correction marker. The two surviving source ranges stay
in natural order, so rendering produces one hard cut over the slip without rewriting or
synthesizing speech.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .source_identity import stable_clip_id

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_CORRECTION_MARKERS = frozenset({"oh", "sorry"})


def _norm(text: str) -> str:
    found = _TOKEN_RE.findall(str(text or "").casefold())
    return found[0] if found else ""


def _shared_affix_length(left: str, right: str) -> int:
    if not left or not right or left == right:
        return 0
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    suffix = 0
    for a, b in zip(reversed(left), reversed(right)):
        if a != b:
            break
        suffix += 1
    return max(prefix, suffix)


def _correction_indices(take: CandidateTake) -> tuple[int, int, int] | None:
    """Return ``(wrong, marker, corrected)`` for a high-confidence lexical correction."""
    words = tuple(take.words)
    if len(words) < 8:
        return None
    tokens = tuple(_norm(word.text) for word in words)
    if not all(tokens):
        return None

    # Only inspect the trailing portion of a substantial delivery. This avoids treating
    # an ordinary early interjection as an edit instruction.
    floor = max(2, len(tokens) - 7)
    for marker_index in range(floor, len(tokens) - 1):
        if tokens[marker_index] not in _CORRECTION_MARKERS:
            continue
        wrong_index = marker_index - 1
        corrected_index = marker_index + 1
        wrong = tokens[wrong_index]
        corrected = tokens[corrected_index]
        if len(wrong) < 6 or len(corrected) < 6:
            continue
        if _shared_affix_length(wrong, corrected) < 4:
            continue
        # Leave enough real message before the slip and keep the correction itself.
        if wrong_index < 5:
            continue
        removed_span = float(words[marker_index].end) - float(words[wrong_index].start)
        if not 0.0 < removed_span <= 2.5:
            continue
        return wrong_index, marker_index, corrected_index
    return None


def _child_take(
    parent: CandidateTake,
    words,
    *,
    start: float,
    end: float,
) -> CandidateTake:
    word_tuple = tuple(words)
    text = " ".join(str(word.text or "").strip() for word in word_tuple).strip()
    return CandidateTake(
        clip_id=stable_clip_id(parent.source_asset_id, start, end, text),
        source_asset_id=parent.source_asset_id,
        source_order=parent.source_order,
        start=start,
        end=end,
        text=text,
        words=word_tuple,
        signals=(
            replace(parent.signals, start=start, end=end)
            if parent.signals is not None
            else None
        ),
        complete_idea=parent.complete_idea,
    )


def split_explicit_lexical_self_corrections(
    kept: Iterable[CandidateTake],
) -> tuple[Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    output: list[CandidateTake] = []
    diagnostics: list[dict] = []

    for take in kept:
        found = _correction_indices(take)
        if found is None:
            output.append(take)
            continue
        wrong_index, marker_index, corrected_index = found
        words = tuple(take.words)
        prefix_words = words[:wrong_index]
        suffix_words = words[corrected_index:]
        if len(prefix_words) < 5 or not suffix_words:
            output.append(take)
            continue

        prefix_end = float(prefix_words[-1].end)
        suffix_start = float(suffix_words[0].start)
        if prefix_end <= take.start + 0.4 or suffix_start >= take.end:
            output.append(take)
            continue
        if suffix_start - prefix_end > 2.8:
            output.append(take)
            continue

        prefix = _child_take(
            take,
            prefix_words,
            start=float(take.start),
            end=prefix_end,
        )
        suffix = _child_take(
            take,
            suffix_words,
            start=suffix_start,
            end=float(take.end),
        )
        output.extend((prefix, suffix))
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "explicit_lexical_self_correction_cut",
            "original_text": take.text,
            "wrong_word": str(words[wrong_index].text or "").strip(),
            "marker": str(words[marker_index].text or "").strip(),
            "corrected_word": str(words[corrected_index].text or "").strip(),
            "kept_clip_ids": [prefix.clip_id, suffix.clip_id],
            "kept_text": [prefix.text, suffix.text],
            "cut_start": round(float(words[wrong_index].start), 3),
            "cut_end": round(suffix_start, 3),
        })

    return tuple(output), tuple(diagnostics)


def install_explicit_lexical_self_correction_cut() -> None:
    """Install as the final deterministic Clean Cut wrapper."""
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_explicit_lexical_self_correction", False):
        return

    def apply_with_explicit_lexical_self_correction(takes, context=None):
        take_tuple = tuple(takes)
        kept, discarded, decisions = original(take_tuple, context)
        repaired, diagnostics = split_explicit_lexical_self_corrections(kept)
        if not diagnostics:
            return kept, discarded, decisions
        extra = tuple(
            CleanCutDecision(
                clip_id=str(item["clip_id"]),
                keep=True,
                reason=str(item["reason"]),
                confidence=0.99,
            )
            for item in diagnostics
        )
        return repaired, discarded, tuple(decisions) + extra

    apply_with_explicit_lexical_self_correction._cutsell_explicit_lexical_self_correction = True
    clean_cut.apply_clean_cut = apply_with_explicit_lexical_self_correction
