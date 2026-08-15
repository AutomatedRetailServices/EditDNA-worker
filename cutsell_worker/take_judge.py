"""Deterministic fail-open baseline ranking for valid takes.

Provider-backed multimodal judging can replace/augment this scorer without changing
contracts. This module never deletes content.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, RankedTake

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _handling_failure_penalty(signal) -> float:
    """Penalize a visually broken product take without turning it into a delete rule.

    A low product-visibility frame is common in legitimate talking-head footage, so it
    is never penalized on its own. The interaction only applies when local vision also
    sees a strong fumble/distraction plus a visibly broken expression or gesture. This
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


def _material_prefix_fragment(candidate: CandidateTake, reference: CandidateTake) -> bool:
    """Return True when candidate is clearly an abandoned prefix of a fuller retry.

    This is group-relative ranking evidence only. It does not delete the fragment and
    it does not use semantic/sales roles. Exact lexical prefix plus a meaningful length
    and duration advantage is required so ordinary concise alternatives are protected.
    """
    left = _tokens(candidate.text)
    right = _tokens(reference.text)
    if not 2 <= len(left) <= 8:
        return False
    if len(right) - len(left) < 3:
        return False
    if candidate.duration_sec + 0.70 > reference.duration_sec:
        return False
    return right[: len(left)] == left


def _repetitive_restart_fragment(candidate: CandidateTake, reference: CandidateTake) -> bool:
    """Detect a retry fragment dominated by repeated words when a fuller take exists.

    This is deliberately group-relative. A repeated slogan or intentional phrase is
    untouched unless another take in the same retry group is materially longer and
    shares nearly all of the fragment's vocabulary.
    """
    left = _tokens(candidate.text)
    right = _tokens(reference.text)
    if not 4 <= len(left) <= 10:
        return False
    if len(right) - len(left) < 3:
        return False
    if candidate.duration_sec + 0.70 > reference.duration_sec:
        return False
    counts = Counter(left)
    repetitive = max(counts.values(), default=0) >= 3 or (len(set(left)) / max(1, len(left))) <= 0.55
    if not repetitive:
        return False
    unique_left = set(left)
    overlap = unique_left & set(right)
    if len(overlap) < 2:
        return False
    return len(overlap) / max(1, len(unique_left)) >= 0.66


def rank_takes(takes: Iterable[CandidateTake]) -> Tuple[RankedTake, ...]:
    take_tuple = tuple(takes)
    base = {take.clip_id: score_take(take) for take in take_tuple}
    adjusted: list[RankedTake] = []

    for take in take_tuple:
        item = base[take.clip_id]
        is_prefix_fragment = any(
            other.clip_id != take.clip_id and _material_prefix_fragment(take, other)
            for other in take_tuple
        )
        is_repetitive_restart = any(
            other.clip_id != take.clip_id and _repetitive_restart_fragment(take, other)
            for other in take_tuple
        )
        score = item.score
        reasons = [item.reason]
        if is_prefix_fragment:
            score -= 0.22
            reasons.append("material_prefix_fragment_penalty")
        if is_repetitive_restart:
            score -= 0.28
            reasons.append("repetitive_restart_fragment_penalty")
        adjusted.append(RankedTake(
            take.clip_id,
            round(_bounded(score), 4),
            "+".join(reasons),
        ))

    return tuple(sorted(adjusted, key=lambda item: (-item.score, item.clip_id)))
