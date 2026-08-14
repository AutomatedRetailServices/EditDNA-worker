"""Conservative local cleanup for explicit recording-break speech.

This module targets creator/process failures that are semantically about the recording
itself (for example ``I can't talk`` or ``let's do that again``).  It deliberately
does not treat profanity alone, ordinary negation, or intentional single-word emphasis
as destructive evidence.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_EXPLICIT_FAILURE_RE = re.compile(
    r"\bi\s+(?:can(?:not|'t)|cant)\s+talk\b|"
    r"\bi\s+(?:do\s+not|don't|dont)\s+know\s+how\s+to\s+end\b|"
    r"\b(?:let's|lets|let\s+us)\s+do\s+that\s+again\b|"
    r"\b(?:let's|lets|let\s+us)\s+(?:try|start)\s+(?:that\s+)?again\b",
    re.IGNORECASE,
)
_HAND_SELF_DIRECTION_RE = re.compile(
    r"\bwhat\s+are\s+you\s+doing\s+with\s+your\s+hands\b",
    re.IGNORECASE,
)
_FRUSTRATION_TOKENS = frozenset({
    "fuck", "fucking", "frig", "frick", "damn", "ugh", "oops", "stupid", "crap",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _has_frustration(text: str) -> bool:
    return bool(set(_tokens(text)).intersection(_FRUSTRATION_TOKENS))


def _has_repeated_multiword_phrase(text: str) -> bool:
    """Find repeated 2-6 word phrases while ignoring single-word emphasis.

    Two occurrences are enough only when another independent signal (frustration)
    is present.  This catches abandoned restart loops such as a phrase repeated after
    an expletive without turning normal rhetorical repetition into a deletion rule.
    """
    tokens = _tokens(text)
    if len(tokens) < 6:
        return False
    for width in range(min(6, len(tokens) // 2), 1, -1):
        seen: set[tuple[str, ...]] = set()
        for index in range(0, len(tokens) - width + 1):
            gram = tokens[index:index + width]
            if len(set(gram)) < 2:
                continue
            if gram in seen:
                return True
            seen.add(gram)
    return False


def _recording_break_reason(take: CandidateTake) -> str | None:
    text = str(take.text or "").strip()
    if not text:
        return None
    if _EXPLICIT_FAILURE_RE.search(text):
        return "explicit_recording_failure"
    if _HAND_SELF_DIRECTION_RE.search(text) and _has_frustration(text):
        return "frustrated_self_direction"
    if _has_frustration(text) and _has_repeated_multiword_phrase(text):
        return "frustrated_internal_restart_repetition"
    return None


def apply_recording_break_cleanup(
    kept: Iterable[CandidateTake],
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    survivors = []
    removed = []
    diagnostics = []
    for take in tuple(kept):
        reason = _recording_break_reason(take)
        if reason is None:
            survivors.append(take)
            continue
        removed.append(take)
        diagnostics.append({"clip_id": take.clip_id, "reason": reason, "text": take.text})
    return tuple(survivors), tuple(removed), tuple(diagnostics)


def install_recording_break_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_recording_break_cleanup", False):
        return

    def apply_with_recording_breaks(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, contextual_discarded, diagnostics = apply_recording_break_cleanup(kept)
        if not contextual_discarded:
            return kept, discarded, decisions

        reason_by_id = {item["clip_id"]: item["reason"] for item in diagnostics}
        extra = tuple(
            CleanCutDecision(
                clip_id=take.clip_id,
                keep=False,
                reason=reason_by_id[take.clip_id],
                confidence=0.96,
            )
            for take in contextual_discarded
        )
        return kept, tuple(discarded) + tuple(contextual_discarded), tuple(decisions) + extra

    apply_with_recording_breaks._cutsell_recording_break_cleanup = True
    clean_cut.apply_clean_cut = apply_with_recording_breaks
