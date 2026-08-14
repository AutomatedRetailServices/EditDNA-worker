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
_EXPLICIT_RECORDING_META_RE = re.compile(
    r"\b(?:i\s+)?(?:have|need)\s+to\s+(?:look|check)\s+(?:at\s+)?(?:the\s+)?(?:word|script|line|notes?)\b|"
    r"\bi\s+(?:forgot|forget)\s+(?:the\s+)?(?:word|line|script)\b|"
    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?(?:word|line)\b|"
    r"\bhow\s+do\s+(?:i|you)\s+(?:say|pronounce)\b",
    re.IGNORECASE,
)
_RESET_CANDIDATE_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_CANDIDATE_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})
_DISCOURSE_PREFIXES = frozenset({"and", "but", "so", "okay", "ok", "well"})
_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def _normalized(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _tokens(text: str) -> tuple[str, ...]:
    tokens = tuple(token.lower() for token in _TOKEN_RE.findall(str(text or "")))
    while tokens and tokens[0] in _DISCOURSE_PREFIXES:
        tokens = tokens[1:]
    return tokens


def _looks_like_explicit_recording_direction(text: str) -> bool:
    return any(phrase in text for phrase in _PRODUCTION_PHRASES) or bool(_ONE_MORE_RE.search(text))


def _source_events(whole_video_context: WholeVideoContext | None, source_asset_id: str):
    if whole_video_context is None:
        return ()
    for source in whole_video_context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _has_reset_and_break_near_take(
    take: CandidateTake,
    whole_video_context: WholeVideoContext | None,
    *,
    padding_sec: float = 0.45,
    minimum_confidence: float = 0.72,
) -> bool:
    events = tuple(
        event
        for event in _source_events(whole_video_context, take.source_asset_id)
        if event.end >= take.start - padding_sec and event.start <= take.end + padding_sec
    )
    has_reset = any(
        str(event.kind or "").strip().lower().replace("-", "_").replace(" ", "_") in _RESET_CANDIDATE_KINDS
        and event.confidence >= minimum_confidence
        for event in events
    )
    has_break = any(
        str(event.kind or "").strip().lower().replace("-", "_").replace(" ", "_") in _BREAK_CANDIDATE_KINDS
        and event.confidence >= minimum_confidence
        for event in events
    )
    return has_reset and has_break


def _looks_like_recording_meta(
    take: CandidateTake,
    whole_video_context: WholeVideoContext | None,
) -> bool:
    text = str(take.text or "").strip()
    match = _EXPLICIT_RECORDING_META_RE.search(text)
    if match is None or take.duration_sec > 4.5:
        return False
    normalized = _normalized(text)
    # First-person forms are explicit enough on their own.  ASR sometimes drops
    # the leading "I"; in that case require local visual corroboration so ordinary
    # content about a word/script is never removed from text alone.
    if normalized.startswith("i ") or re.search(r"\bi\s+(?:forgot|forget)\b", normalized):
        return True
    return _has_reset_and_break_near_take(take, whole_video_context)


def _dense_reset_evidence(
    take: CandidateTake,
    whole_video_context: WholeVideoContext | None,
    *,
    minimum_confidence: float = 0.90,
    minimum_count: int = 3,
    minimum_span_sec: float = 0.70,
) -> bool:
    """Confirm repeated physical reset evidence inside one short spoken take."""
    if whole_video_context is None:
        return False
    matches = []
    for event in _source_events(whole_video_context, take.source_asset_id):
        kind = str(event.kind or "").strip().lower().replace("-", "_").replace(" ", "_")
        if kind not in _RESET_CANDIDATE_KINDS or event.confidence < minimum_confidence:
            continue
        if event.end <= take.start or event.start >= take.end:
            continue
        matches.append(event)
    if len(matches) < minimum_count:
        return False
    matches.sort(key=lambda item: (item.start, item.end))
    return matches[-1].end - matches[0].start >= minimum_span_sec


def _strong_retry_match(first: CandidateTake, second: CandidateTake) -> bool:
    """Return True only for strong lexical retry/prefix evidence."""
    left = _tokens(first.text)
    right = _tokens(second.text)
    if not left or not right:
        return False
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if tuple(longer[: len(shorter)]) == tuple(shorter):
        return True
    shorter_set = set(shorter)
    if not shorter_set:
        return False
    overlap = len(shorter_set.intersection(longer)) / len(shorter_set)
    return overlap >= 0.80


def _nearby_retry_corroboration(
    take: CandidateTake,
    takes: tuple[CandidateTake, ...],
    *,
    maximum_gap_sec: float = 6.0,
) -> bool:
    """Require close same-source lexical retry evidence before dense-reset deletion."""
    for other in takes:
        if other.clip_id == take.clip_id or other.source_asset_id != take.source_asset_id:
            continue
        if other.end <= take.start:
            gap = take.start - other.end
        elif other.start >= take.end:
            gap = other.start - take.end
        else:
            gap = 0.0
        if gap > maximum_gap_sec:
            continue
        if _strong_retry_match(take, other):
            return True
    return False


def evaluate_take(
    take: CandidateTake,
    whole_video_context: WholeVideoContext | None = None,
) -> CleanCutDecision:
    """Remove obvious recording-process material while protecting uncertainty."""
    text = _normalized(take.text)
    words = text.split()
    if take.duration_sec <= 0.12:
        return CleanCutDecision(take.clip_id, False, "impossible_microfragment", 0.99)
    if not text and take.duration_sec >= 0.5:
        return CleanCutDecision(take.clip_id, False, "dead_air", 0.95)
    if _looks_like_explicit_recording_direction(text):
        return CleanCutDecision(take.clip_id, False, "explicit_restart_direction", 0.97)
    if _looks_like_recording_meta(take, whole_video_context):
        return CleanCutDecision(take.clip_id, False, "explicit_recording_meta", 0.96)
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

    if (
        not take.complete_idea
        and take.duration_sec <= 2.6
        and len(words) <= 5
        and _dense_reset_evidence(take, whole_video_context)
    ):
        return CleanCutDecision(take.clip_id, False, "incomplete_microtake_dense_reset", 0.93)

    if take.signals and take.signals.visual_fumble >= 0.90 and not take.complete_idea:
        return CleanCutDecision(take.clip_id, False, "obvious_visual_fumble", 0.90)
    return CleanCutDecision(take.clip_id, True, "valid_or_uncertain_speech", 0.50)


def apply_clean_cut(
    takes: Iterable[CandidateTake],
    whole_video_context: WholeVideoContext | None = None,
) -> Tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[CleanCutDecision, ...]]:
    take_tuple = tuple(takes)
    kept, discarded, decisions = [], [], []
    for take in take_tuple:
        decision = evaluate_take(take, whole_video_context)
        if (
            not decision.keep
            and decision.reason == "incomplete_microtake_dense_reset"
            and not _nearby_retry_corroboration(take, take_tuple)
        ):
            decision = CleanCutDecision(
                take.clip_id,
                True,
                "dense_reset_without_retry_corroboration",
                0.50,
            )
        decisions.append(decision)
        (kept if decision.keep else discarded).append(take)
    return tuple(kept), tuple(discarded), tuple(decisions)
