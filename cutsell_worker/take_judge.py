"""Deterministic fail-open baseline ranking for valid takes.

Provider-backed multimodal judging can replace/augment this scorer without changing
contracts. This module never deletes content.
"""
from __future__ import annotations

from typing import Iterable, Tuple

from .contracts import CandidateTake, RankedTake


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _handling_failure_penalty(signal) -> float:
    """Penalize a visually broken product take without turning it into a delete rule.

    A low product-visibility frame is common in legitimate talking-head footage, so it
    is never penalized on its own.  The interaction only applies when local vision also
    sees a strong fumble/distraction plus a visibly broken expression or gesture.  This
    makes a dropped/mis-handled product lose a retry-group ranking contest while leaving
    ordinary product-away-from-frame speech unaffected.
    """
    broken_delivery = (
        signal.visual_fumble >= 0.55
        and signal.distraction_risk >= 0.45
        and (
            signal.expression_naturalness <= 0.45
            or signal.gesture_naturalness <= 0.40
        )
    )
    if not broken_delivery:
        return 0.0

    penalty = 0.10
    if signal.product_visibility <= 0.45:
        penalty += 0.08
    if signal.motion_stability <= 0.45:
        penalty += 0.05
    return penalty


def score_take(take: CandidateTake) -> RankedTake:
    signal = take.signals
    completeness = 1.0 if take.complete_idea else 0.45
    duration_fit = 1.0 if 0.7 <= take.duration_sec <= 20.0 else 0.65
    if signal is None:
        score = 0.70 * completeness + 0.30 * duration_fit
        return RankedTake(take.clip_id, round(_bounded(score), 4), "text_timing_baseline")

    # Clean Cut Best Take must judge the *delivery*, not only whether the words are
    # complete. Dense local MediaPipe/OpenCV signals expose expression, gesture,
    # energy and distraction evidence; include them directly so the RunPod-local path
    # can prefer a natural successful delivery without an external multimodal provider.
    score = (
        0.16 * completeness
        + 0.06 * duration_fit
        + 0.12 * signal.audio_quality
        + 0.08 * signal.face_visibility
        + 0.09 * signal.eye_contact
        + 0.06 * signal.framing_quality
        + 0.05 * signal.product_visibility
        + 0.07 * signal.motion_stability
        + 0.07 * signal.continuity
        + 0.10 * signal.expression_naturalness
        + 0.07 * signal.gesture_naturalness
        + 0.07 * signal.delivery_energy
        - 0.12 * signal.visual_fumble
        - 0.08 * signal.distraction_risk
        - _handling_failure_penalty(signal)
    )
    return RankedTake(take.clip_id, round(_bounded(score), 4), "watch_listen_baseline")


def rank_takes(takes: Iterable[CandidateTake]) -> Tuple[RankedTake, ...]:
    return tuple(sorted((score_take(take) for take in takes), key=lambda item: (-item.score, item.clip_id)))
