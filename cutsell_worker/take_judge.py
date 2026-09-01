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
_RESTART_TAIL_RE = re.compile(
    r"\b(?:okay|ok)\s+now(?:\s+(?:we|i|let(?:'s|\s+us)))?\b",
    re.IGNORECASE,
)
_CONTENT_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "that", "the", "this",
    "to", "was", "we", "what", "with", "you", "your", "okay", "ok", "now", "let",
    "lets", "us",
})


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in _tokens(text) if len(token) >= 3 and token not in _CONTENT_STOP)


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    """Return True only for a single-character ASR drift between content words."""
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    short, long = (left, right) if len(left) < len(right) else (right, left)
    i = j = mismatches = 0
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
            continue
        mismatches += 1
        if mismatches > 1:
            return False
        j += 1
    return True


def _near_prefix_tokens(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    """Allow one narrow content-word ASR typo inside an otherwise exact prefix."""
    if len(right) < len(left):
        return False
    mismatches = []
    for candidate_token, reference_token in zip(left, right[: len(left)]):
        if candidate_token == reference_token:
            continue
        mismatches.append((candidate_token, reference_token))
    if len(mismatches) != 1:
        return False
    candidate_token, reference_token = mismatches[0]
    if (
        len(candidate_token) < 4
        or len(reference_token) < 4
        or candidate_token in _CONTENT_STOP
        or reference_token in _CONTENT_STOP
    ):
        return False
    return _edit_distance_at_most_one(candidate_token, reference_token)


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

    Exact lexical prefix is preferred. One single-character ASR drift in a >=4-letter
    content word is also accepted when every other aligned token is exact. The latter
    covers transcription pairs such as ``crop``/``croc`` without turning semantic
    similarity into a broad prefix rule.
    """
    left = _tokens(candidate.text)
    right = _tokens(reference.text)
    if not 2 <= len(left) <= 8:
        return False
    if len(right) - len(left) < 3:
        return False
    if candidate.duration_sec + 0.70 > reference.duration_sec:
        return False
    exact_prefix = right[: len(left)] == left
    return exact_prefix or _near_prefix_tokens(left, right)


def _repetitive_restart_fragment(candidate: CandidateTake, reference: CandidateTake) -> bool:
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


def _restart_tail_fragment(candidate: CandidateTake, reference: CandidateTake) -> bool:
    text = str(candidate.text or "")
    match = _RESTART_TAIL_RE.search(text)
    if match is None:
        return False

    left = _tokens(text)
    right = _tokens(reference.text)
    if len(left) < 6 or len(right) < len(left) + 3:
        return False
    if candidate.duration_sec + 0.70 > reference.duration_sec:
        return False

    prefix_text = text[: match.start()].strip()
    prefix = _content_tokens(prefix_text)
    reference_content = set(_content_tokens(reference.text))
    if len(prefix) < 3 or not reference_content:
        return False
    overlap = sum(1 for token in set(prefix) if token in reference_content)
    if overlap < 3 or overlap / max(1, len(set(prefix))) < 0.60:
        return False

    marker_token_index = len(_tokens(text[: match.start()]))
    return marker_token_index >= max(3, len(left) // 2)


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
        is_restart_tail = any(
            other.clip_id != take.clip_id and _restart_tail_fragment(take, other)
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
        if is_restart_tail:
            score -= 0.18
            reasons.append("restart_tail_fragment_penalty")
        adjusted.append(RankedTake(
            take.clip_id,
            round(_bounded(score), 4),
            "+".join(reasons),
        ))

    return tuple(sorted(adjusted, key=lambda item: (-item.score, item.clip_id)))
