"""Conservative production cleanup for CutSell Flow B."""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision

_PRODUCTION_PHRASES = (
    "start over",
    "let me start over",
    "let me redo",
    "redo that",
    "one more time",
    "take two",
    "take three",
    "okay stop",
    "ok stop",
    "what am i saying",
    "what was i saying",
    "why is it so wobbly",
    "why is this so wobbly",
    "otra vez",
    "déjame empezar",
    "dejame empezar",
    "empiezo de nuevo",
    "déjame hacerlo de nuevo",
    "dejame hacerlo de nuevo",
)

_SHORT_RESTART_MARKERS = {
    "again",
    "again.",
    "otra vez",
    "de nuevo",
}

_ONE_MORE_RE = re.compile(
    r"\bone more\b\s+(?:time|take|because|cuz|cause|since|you|we|i)\b",
    re.IGNORECASE,
)


def _normalized(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _looks_like_explicit_recording_direction(text: str) -> bool:
    if any(phrase in text for phrase in _PRODUCTION_PHRASES):
        return True
    if _ONE_MORE_RE.search(text):
        return True
    return False


def evaluate_take(take: CandidateTake) -> CleanCutDecision:
    """Delete only obvious production mistakes; uncertainty stays keep=True."""
    text = _normalized(take.text)
    words = text.split()
    if take.duration_sec <= 0.12:
        return CleanCutDecision(take.clip_id, False, "impossible_microfragment", 0.99)
    if not text and take.duration_sec >= 0.5:
        return CleanCutDecision(take.clip_id, False, "dead_air", 0.95)
    if _looks_like_explicit_recording_direction(text):
        return CleanCutDecision(take.clip_id, False, "explicit_restart_direction", 0.97)
    if text in _SHORT_RESTART_MARKERS and take.duration_sec <= 1.6 and len(words) <= 2:
        return CleanCutDecision(take.clip_id, False, "isolated_restart_marker", 0.94)
    if take.signals and take.signals.silence_ratio >= 0.96 and len(words) <= 1:
        return CleanCutDecision(take.clip_id, False, "unusable_silence", 0.92)
    if take.signals and take.signals.visual_fumble >= 0.97 and not take.complete_idea:
        return CleanCutDecision(take.clip_id, False, "obvious_visual_fumble", 0.90)
    return CleanCutDecision(take.clip_id, True, "valid_or_uncertain_speech", 0.50)


def apply_clean_cut(takes: Iterable[CandidateTake]) -> Tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[CleanCutDecision, ...]]:
    kept, discarded, decisions = [], [], []
    for take in takes:
        decision = evaluate_take(take)
        decisions.append(decision)
        (kept if decision.keep else discarded).append(take)
    return tuple(kept), tuple(discarded), tuple(decisions)
