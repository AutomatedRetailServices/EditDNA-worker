"""Conservative production cleanup for CutSell Flow B."""
from __future__ import annotations

from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision

_PRODUCTION_PHRASES = (
    "start over",
    "let me start over",
    "one more time",
    "take two",
    "take three",
    "otra vez",
    "déjame empezar",
    "dejame empezar",
    "empiezo de nuevo",
)


def evaluate_take(take: CandidateTake) -> CleanCutDecision:
    """Delete only obvious production mistakes; uncertainty stays keep=True."""
    text = " ".join(take.text.lower().split())
    if take.duration_sec <= 0.12:
        return CleanCutDecision(take.clip_id, False, "impossible_microfragment", 0.99)
    if not text and take.duration_sec >= 0.5:
        return CleanCutDecision(take.clip_id, False, "dead_air", 0.95)
    if any(phrase in text for phrase in _PRODUCTION_PHRASES):
        return CleanCutDecision(take.clip_id, False, "explicit_restart_direction", 0.95)
    if take.signals and take.signals.silence_ratio >= 0.96 and len(text.split()) <= 1:
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
