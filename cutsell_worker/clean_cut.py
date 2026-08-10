"""Context-aware production cleanup for CutSell Flow B."""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .temporal_editing import harmful_coverage_ratio, harmful_events_for_take
from .whole_video_analysis import WholeVideoContext

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

_SHORT_RESTART_MARKERS = {"again", "again.", "otra vez", "de nuevo"}
_ONE_MORE_RE = re.compile(
    r"\bone more\b\s+(?:time|take|because|cuz|cause|since|you|we|i)\b",
    re.IGNORECASE,
)


def _normalized(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _looks_like_explicit_recording_direction(text: str) -> bool:
    return any(phrase in text for phrase in _PRODUCTION_PHRASES) or bool(_ONE_MORE_RE.search(text))


def evaluate_take(
    take: CandidateTake,
    whole_video_context: WholeVideoContext | None = None,
) -> CleanCutDecision:
    """Remove obvious recording-process material while protecting uncertainty.

    Whole-video events are now first-class evidence. A take can therefore be a
    wrong take even when its transcript is grammatically complete. We only delete
    from temporal evidence when a high-confidence bad-performance event dominates
    the take; precise edge reactions/resets are handled earlier by temporal trim.
    """
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

    harmful = harmful_events_for_take(take, whole_video_context, minimum_confidence=0.76)
    coverage = harmful_coverage_ratio(take, harmful)
    if harmful and coverage >= 0.62:
        strongest = max(harmful, key=lambda item: item.confidence)
        return CleanCutDecision(
            take.clip_id,
            False,
            f"whole_video_bad_take:{strongest.kind}",
            min(0.99, max(0.82, strongest.confidence)),
        )

    # Take-level visual evidence remains a fallback. The lower threshold is safe
    # only when the spoken idea is incomplete; complete speech needs temporal/global
    # corroboration above so a normal gesture cannot cause deletion by itself.
    if take.signals and take.signals.visual_fumble >= 0.90 and not take.complete_idea:
        return CleanCutDecision(take.clip_id, False, "obvious_visual_fumble", 0.90)
    return CleanCutDecision(take.clip_id, True, "valid_or_uncertain_speech", 0.50)


def apply_clean_cut(
    takes: Iterable[CandidateTake],
    whole_video_context: WholeVideoContext | None = None,
) -> Tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[CleanCutDecision, ...]]:
    kept, discarded, decisions = [], [], []
    for take in takes:
        decision = evaluate_take(take, whole_video_context)
        decisions.append(decision)
        (kept if decision.keep else discarded).append(take)
    return tuple(kept), tuple(discarded), tuple(decisions)
