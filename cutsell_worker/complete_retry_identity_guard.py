"""Conservative identity guard for complete-take retry supersession.

A complete spoken idea may be auto-deleted as a superseded retry only when the later
candidate is strongly identifiable as the same delivery, not merely a continuation on
the same topic. This keeps the engine aligned with NEVER REMOVE SPOKEN INFORMATION and
WHEN UNCERTAIN, KEEP without hardcoding benchmark phrases or timestamps.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_COMPLETE_RETRY_MIN_OVERLAP = 0.64
_COMPLETE_RETRY_MIN_SEQUENCE = 0.52


def _numbers(text: str) -> frozenset[str]:
    return frozenset(match.group(0).replace(",", ".") for match in _NUMBER_RE.finditer(str(text or "")))


def _normalized_text(text: str) -> str:
    raw = unicodedata.normalize("NFKD", str(text or "").lower())
    raw = "".join(char for char in raw if not unicodedata.combining(char))
    raw = re.sub(r"[^\w]+", " ", raw, flags=re.UNICODE)
    return " ".join(raw.split())


def _sequence_identity(left_text: str, right_text: str) -> float:
    left = _normalized_text(left_text)
    right = _normalized_text(right_text)
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right).ratio())


def install_complete_retry_identity_guard() -> None:
    from . import hybrid_session_cleanup as cleanup

    original = cleanup._later_semantic_retry_replacement
    if getattr(original, "_cutsell_complete_retry_identity_guard", False):
        return

    def protected(
        failed_take,
        members,
        decisions_by_id,
        *,
        minimum_label_confidence: float = 0.68,
        minimum_overlap: float = 0.50,
        maximum_delay_sec: float = 24.0,
    ):
        # Incomplete retries can still use the original looser matching: by definition
        # they often contain only a fragment of the later successful delivery.
        if not bool(getattr(failed_take, "complete_idea", False)):
            return original(
                failed_take,
                members,
                decisions_by_id,
                minimum_label_confidence=minimum_label_confidence,
                minimum_overlap=minimum_overlap,
                maximum_delay_sec=maximum_delay_sec,
            )

        # Complete deliveries carry information authority. A later take must preserve
        # every numeric fact already spoken before it can supersede the earlier take.
        failed_numbers = _numbers(getattr(failed_take, "text", ""))
        compatible_members = tuple(
            candidate
            for candidate in members
            if candidate.clip_id == failed_take.clip_id
            or not failed_numbers
            or failed_numbers.issubset(_numbers(getattr(candidate, "text", "")))
        )

        replacement, overlap = original(
            failed_take,
            compatible_members,
            decisions_by_id,
            minimum_label_confidence=minimum_label_confidence,
            minimum_overlap=max(float(minimum_overlap), _COMPLETE_RETRY_MIN_OVERLAP),
            maximum_delay_sec=maximum_delay_sec,
        )
        if replacement is None:
            return None, overlap

        # Topic overlap alone is not enough. Require recognizable delivery-level
        # sequence identity as a second independent signal. This rejects narrative
        # continuations that reuse nouns while introducing a new fact or event.
        sequence_identity = _sequence_identity(
            getattr(failed_take, "text", ""),
            getattr(replacement, "text", ""),
        )
        if sequence_identity < _COMPLETE_RETRY_MIN_SEQUENCE:
            return None, overlap
        return replacement, overlap

    protected._cutsell_complete_retry_identity_guard = True
    cleanup._later_semantic_retry_replacement = protected
