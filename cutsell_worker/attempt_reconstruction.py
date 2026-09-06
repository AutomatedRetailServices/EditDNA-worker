"""Reconstruct creator delivery attempts before Clean Cut deletion.

Whisper/ASR boundaries are transcription boundaries, not editorial takes.  This module
combines nearby speech fragments into the larger delivery attempt a human editor would
judge, while keeping retry/reset boundaries separate so Best Take can compare complete
attempts rather than isolated fragments.

The policy is deliberately conservative: uncertain neighboring speech stays together.
A boundary is created only from a real pause, a hard creator/session discontinuity,
strong multi-family recording-process evidence, or clear lexical restart evidence.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, MediaSignals
from .session_boundaries import infer_session_boundaries
from .source_identity import stable_clip_id
from .whole_video_analysis import TemporalEvent, WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_TERMINAL_PUNCT_RE = re.compile(r"[.!?…][\"')\]]*$")
_STOP = frozenset({
    # English
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "so", "that", "the",
    "this", "to", "was", "we", "with", "you", "your",
    # Spanish
    "a", "al", "como", "con", "cuando", "de", "del", "el", "en", "es", "esta",
    "este", "la", "las", "le", "les", "lo", "los", "me", "mi", "mis", "o", "para",
    "pero", "por", "porque", "que", "se", "si", "sin", "su", "sus", "un", "una",
    "unos", "unas", "y", "yo",
})

_RESET_KINDS = frozenset({
    "body_reset", "body_reset_candidate", "hand_reset", "hand_motion_reset_candidate",
})
_CAMERA_KINDS = frozenset({
    "camera_disengagement", "camera_disengagement_candidate", "camera_adjustment",
})
_FACE_KINDS = frozenset({
    "facial_expression_shift", "facial_expression_shift_candidate", "breaking_character",
})
_EXPLICIT_ATTEMPT_BREAK_KINDS = frozenset({
    "retry_setup", "wrong_take", "false_start", "frustration", "recording_joke",
    "accidental_laughter", "searching_for_words", "verbal_fumble", "visual_fumble",
    "product_handling_mistake",
})


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in _tokens(text) if len(token) >= 3 and token not in _STOP)


def _source_events(context: WholeVideoContext | None, source_asset_id: str) -> tuple[TemporalEvent, ...]:
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _events_near_transition(
    context: WholeVideoContext | None,
    left: CandidateTake,
    right: CandidateTake,
    *,
    padding_sec: float = 0.36,
) -> tuple[TemporalEvent, ...]:
    start = min(left.end, right.start) - padding_sec
    end = max(left.end, right.start) + padding_sec
    return tuple(
        event
        for event in _source_events(context, left.source_asset_id)
        if event.end >= start and event.start <= end
    )


def _hard_session_boundary_between(
    context: WholeVideoContext | None,
    left: CandidateTake,
    right: CandidateTake,
    *,
    tolerance_sec: float = 0.14,
) -> bool:
    if context is None:
        return False
    for boundary in infer_session_boundaries(context, left.source_asset_id):
        if left.end - tolerance_sec <= boundary.at <= right.start + tolerance_sec:
            return True
    return False


def _restart_evidence(left_text: str, right_text: str) -> bool:
    """Detect an immediate verbal restart without classifying broad topic similarity."""
    left = _content_tokens(left_text)
    right = _content_tokens(right_text)
    if len(left) < 2 or len(right) < 2:
        return False

    # The same meaningful opening is the strongest common creator retry pattern.
    if left[:2] == right[:2]:
        return True

    # A creator can restart the last phrase of a longer attempt rather than the whole
    # paragraph. Require a three-token ordered phrase to avoid topic-only collisions.
    if len(right) >= 3:
        needle = right[:3]
        tail = left[-10:]
        for index in range(max(0, len(tail) - 2)):
            if tail[index : index + 3] == needle:
                return True
    return False


def _short_incomplete_suffix(left: CandidateTake, right: CandidateTake) -> bool:
    """Keep a tiny unfinished tail separate from an already usable delivery.

    ASR can fluctuate between runs and attach the last broken three-second self-correction
    directly to the preceding good paragraph. Once those pieces are merged, Hybrid sees
    one long winner and can no longer remove only the bad tail. Preserve the boundary
    when the preceding candidate is already complete and the new candidate is a short,
    explicitly incomplete suffix. If more speech follows, that suffix can still merge
    forward into its continuation on the next reconstruction step.
    """
    if not left.complete_idea or right.complete_idea:
        return False
    tokens = _tokens(right.text)
    if not tokens or len(tokens) > 10 or right.duration_sec > 4.5:
        return False
    gap = max(0.0, right.start - left.end)
    return gap <= 0.35


def _tiny_nonterminal_continuation(
    left: CandidateTake,
    right: CandidateTake,
    *,
    max_continuation_gap_sec: float,
) -> bool:
    """Protect a tiny lexical tail from a false visual-reset boundary.

    Dense body/hand reset detectors often fire during a normal breath at the end of a
    sentence. If Whisper split the final one or two words into a new candidate while the
    preceding text is visibly non-terminal, Best Take must judge the completed delivery,
    not the truncated prefix and suffix as competing takes. Hard session walls, lexical
    restarts and explicit recording-process breaks are still evaluated separately.
    """
    if left.source_asset_id != right.source_asset_id:
        return False
    gap = max(0.0, right.start - left.end)
    if gap > min(max_continuation_gap_sec, 1.20):
        return False
    right_tokens = _tokens(right.text)
    if not 1 <= len(right_tokens) <= 2 or right.duration_sec > 1.80:
        return False
    if len(_tokens(left.text)) < 4:
        return False
    left_text = str(left.text or "").rstrip()
    if not left_text or _TERMINAL_PUNCT_RE.search(left_text):
        return False
    return True


def _attempt_boundary_reason(
    context: WholeVideoContext | None,
    left: CandidateTake,
    right: CandidateTake,
    *,
    max_continuation_gap_sec: float,
) -> str | None:
    if left.source_asset_id != right.source_asset_id:
        return "source_change"

    gap = max(0.0, right.start - left.end)
    if _hard_session_boundary_between(context, left, right):
        return "hard_session_boundary"

    # Lexical restarts must remain separate candidates so retry grouping/Best Take can
    # compare them. Require a non-zero word boundary so overlapping ASR duplicates are
    # not joined into one self-repeating sentence either.
    if right.start >= left.end - 0.03 and _restart_evidence(left.text, right.text):
        return "lexical_restart"

    # Do not let a short unfinished final self-correction poison a preceding coherent
    # paragraph merely because Whisper made the two chunks nearly contiguous.  This is
    # deliberately directional: complete -> short incomplete.  An incomplete fragment
    # can still merge forward into later continuation speech.
    if _short_incomplete_suffix(left, right):
        return "short_incomplete_suffix"

    # A one/two-word tail after non-terminal text is more reliable speech-continuation
    # evidence than an isolated body/hand reset candidate. Keep it attached before local
    # performance evidence is allowed to split the attempt.
    if _tiny_nonterminal_continuation(
        left,
        right,
        max_continuation_gap_sec=max_continuation_gap_sec,
    ):
        return None

    nearby = _events_near_transition(context, left, right)
    strong = tuple(event for event in nearby if float(event.confidence) >= 0.80)
    kinds = {_kind(event.kind) for event in strong}
    has_reset = bool(kinds & _RESET_KINDS)
    has_camera = bool(kinds & _CAMERA_KINDS)
    has_face = bool(kinds & _FACE_KINDS)

    if any(
        _kind(event.kind) in _EXPLICIT_ATTEMPT_BREAK_KINDS and float(event.confidence) >= 0.90
        for event in nearby
    ):
        return "explicit_recording_process_break"

    # A reset plus an independent disengagement/face change is the talking-head pattern
    # for "finished this try, preparing to say it again". One reset alone is not enough.
    if has_reset and (has_camera or has_face):
        return "multi_family_delivery_reset"

    # With an actual pause, one exceptionally strong reset is sufficient corroboration.
    if gap >= 0.65 and any(
        _kind(event.kind) in _RESET_KINDS and float(event.confidence) >= 0.95
        for event in nearby
    ):
        return "pause_plus_strong_reset"

    if gap > max_continuation_gap_sec:
        return "real_speech_pause"
    return None


def _merge_signals(members: tuple[CandidateTake, ...]) -> MediaSignals | None:
    signaled = [member for member in members if member.signals is not None]
    if not signaled:
        return None
    total = sum(max(0.001, member.duration_sec) for member in members)

    def weighted(name: str, default: float) -> float:
        numerator = 0.0
        for member in members:
            duration = max(0.001, member.duration_sec)
            value = getattr(member.signals, name) if member.signals is not None else default
            numerator += float(value) * duration
        return numerator / total

    return MediaSignals(
        source_asset_id=members[0].source_asset_id,
        start=members[0].start,
        end=members[-1].end,
        silence_ratio=weighted("silence_ratio", 0.0),
        audio_quality=weighted("audio_quality", 0.5),
        face_visibility=weighted("face_visibility", 0.5),
        eye_contact=weighted("eye_contact", 0.5),
        framing_quality=weighted("framing_quality", 0.5),
        product_visibility=weighted("product_visibility", 0.0),
        motion_stability=weighted("motion_stability", 0.5),
        continuity=weighted("continuity", 0.5),
        visual_fumble=max(
            (member.signals.visual_fumble if member.signals is not None else 0.0)
            for member in members
        ),
        expression_naturalness=weighted("expression_naturalness", 0.5),
        gesture_naturalness=weighted("gesture_naturalness", 0.5),
        delivery_energy=weighted("delivery_energy", 0.5),
        distraction_risk=max(
            (member.signals.distraction_risk if member.signals is not None else 0.0)
            for member in members
        ),
    )


def _merge_attempt(members: tuple[CandidateTake, ...]) -> CandidateTake:
    if len(members) == 1:
        return members[0]
    text = " ".join(member.text.strip() for member in members if member.text.strip()).strip()
    start = members[0].start
    end = members[-1].end
    return CandidateTake(
        clip_id=stable_clip_id(members[0].source_asset_id, start, end, text),
        source_asset_id=members[0].source_asset_id,
        source_order=members[0].source_order,
        start=start,
        end=end,
        text=text,
        words=tuple(word for member in members for word in member.words),
        signals=_merge_signals(members),
        # Completeness belongs to the tail of the delivery. A complete earlier sentence
        # must not hide an unfinished final fragment.
        complete_idea=members[-1].complete_idea,
    )


def reconstruct_delivery_attempts(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    *,
    max_continuation_gap_sec: float = 1.20,
) -> tuple[Tuple[CandidateTake, ...], dict]:
    """Build paragraph/delivery-level candidates from ASR-level candidates.

    The function never deletes speech. It only changes the unit that later Clean Cut and
    Best Take judge. This is intentionally fail-open: absent a clear boundary, nearby
    fragments remain one complete creator delivery attempt.
    """
    ordered = tuple(sorted(
        takes,
        key=lambda take: (take.source_order, take.start, take.end, take.clip_id),
    ))
    if not ordered:
        return (), {
            "input_take_count": 0,
            "attempt_count": 0,
            "merged_fragment_count": 0,
            "boundaries": [],
            "attempts": [],
        }

    buckets: list[list[CandidateTake]] = []
    boundaries: list[dict] = []
    for take in ordered:
        if not buckets:
            buckets.append([take])
            continue

        left = buckets[-1][-1]
        reason = _attempt_boundary_reason(
            context,
            left,
            take,
            max_continuation_gap_sec=max_continuation_gap_sec,
        )
        if reason is not None:
            boundaries.append({
                "after_clip_id": left.clip_id,
                "before_clip_id": take.clip_id,
                "at": round((float(left.end) + float(take.start)) / 2.0, 3),
                "gap_sec": round(max(0.0, float(take.start) - float(left.end)), 3),
                "reason": reason,
            })
            buckets.append([take])
            continue
        buckets[-1].append(take)

    attempts = tuple(_merge_attempt(tuple(bucket)) for bucket in buckets)
    diagnostics = {
        "input_take_count": len(ordered),
        "attempt_count": len(attempts),
        "merged_fragment_count": max(0, len(ordered) - len(attempts)),
        "boundaries": boundaries[:300],
        "attempts": [
            {
                "clip_id": attempt.clip_id,
                "source_asset_id": attempt.source_asset_id,
                "start": round(attempt.start, 3),
                "end": round(attempt.end, 3),
                "duration_sec": round(attempt.duration_sec, 3),
                "member_clip_ids": [member.clip_id for member in bucket],
                "member_count": len(bucket),
                "complete_idea": attempt.complete_idea,
            }
            for attempt, bucket in zip(attempts, buckets)
        ][:300],
    }
    return attempts, diagnostics
