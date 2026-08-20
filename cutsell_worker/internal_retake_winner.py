"""Prefer a clean internal retake over an earlier broken delivery inside one candidate.

Best-Take can only compare separate candidates. In raw talking-head recordings ASR may
merge a failed delivery, a visible retry/reset, and the clean retake into one CandidateTake.
This pass detects that structure conservatively and keeps the later retake.

The pass requires all of the following:
- a strong recording-process retry/failure event inside the take;
- meaningful speech on both sides of the event;
- substantial lexical/content overlap showing that the later speech is a retry of the
  earlier communication attempt rather than a new story beat;
- the later side is long enough to stand as a useful delivery.

It never cuts through a spoken word and fails open when evidence is ambiguous.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "de", "del",
    "el", "en", "es", "esta", "este", "for", "from", "in", "is", "it", "la", "las", "lo",
    "los", "me", "mi", "mis", "of", "on", "or", "para", "pero", "por", "que", "se", "so",
    "su", "sus", "that", "the", "this", "to", "un", "una", "was", "we", "with", "y", "yo",
})
_RETRY_EVENT_KINDS = frozenset({
    "retry_setup",
    "false_start",
    "wrong_take",
    "searching_for_words",
    "breaking_character",
    "unintentional_dead_air",
})
_RESET_EVENT_KINDS = frozenset({
    "body_reset_candidate",
    "camera_disengagement_candidate",
    "facial_expression_shift_candidate",
})


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _norm(value: str) -> str:
    found = _TOKEN_RE.findall(str(value or "").casefold())
    return found[0] if found else ""


def _content_words(words) -> set[str]:
    out = set()
    for word in words:
        token = _norm(getattr(word, "text", ""))
        if len(token) >= 3 and token not in _STOP:
            out.add(token)
    return out


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _retry_boundaries(take: CandidateTake, context: WholeVideoContext | None):
    events = []
    for event in _source_events(context, take.source_asset_id):
        if event.end < take.start or event.start > take.end:
            continue
        kind = _kind(event.kind)
        confidence = float(event.confidence)
        if kind in _RETRY_EVENT_KINDS and confidence >= 0.78:
            events.append((event, "authoritative"))
        elif kind in _RESET_EVENT_KINDS and confidence >= (
            0.90 if kind == "body_reset_candidate" else 0.80
        ):
            events.append((event, "reset"))
    return tuple(events)


def _split_index_for_event(words, event) -> int | None:
    """Find the first word that begins after the retry/reset event midpoint."""
    if not words:
        return None
    pivot = (float(event.start) + float(event.end)) / 2.0
    for index, word in enumerate(words):
        if float(word.start) >= pivot:
            return index
    return None


def _overlap(left_words, right_words) -> tuple[int, float, float]:
    left = _content_words(left_words)
    right = _content_words(right_words)
    shared = len(left & right)
    left_cov = shared / max(1, len(left))
    right_cov = shared / max(1, len(right))
    return shared, left_cov, right_cov


def prefer_internal_clean_retakes(
    kept: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    *,
    minimum_right_words: int = 5,
    minimum_shared_content: int = 3,
    minimum_directional_coverage: float = 0.45,
) -> tuple[Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    output = []
    diagnostics = []

    for take in kept:
        words = tuple(sorted(take.words, key=lambda word: (float(word.start), float(word.end))))
        if len(words) < 10:
            output.append(take)
            continue

        candidates = []
        for event, evidence_type in _retry_boundaries(take, context):
            split = _split_index_for_event(words, event)
            if split is None or split < 4 or len(words) - split < minimum_right_words:
                continue

            # Compare the nearby communication attempts rather than the entire take. A
            # long story may contain unrelated speech before/after the retry.
            left_start = max(0, split - 12)
            right_end = min(len(words), split + 14)
            left_window = words[left_start:split]
            right_window = words[split:right_end]
            shared, left_cov, right_cov = _overlap(left_window, right_window)
            if shared < minimum_shared_content:
                continue
            if max(left_cov, right_cov) < minimum_directional_coverage:
                continue

            # Stronger explicit retry/failure events outrank generic physical resets.
            priority = 2 if evidence_type == "authoritative" else 1
            score = (priority, shared, max(left_cov, right_cov), float(event.confidence))
            candidates.append((score, split, event, shared, left_cov, right_cov, evidence_type))

        if not candidates:
            output.append(take)
            continue

        _, split, event, shared, left_cov, right_cov, evidence_type = max(candidates, key=lambda item: item[0])
        right_words = words[split:]
        new_start = float(right_words[0].start)
        if new_start <= take.start + 0.25 or take.end - new_start < 1.0:
            output.append(take)
            continue

        text = " ".join(str(word.text or "").strip() for word in right_words).strip()
        if not text:
            output.append(take)
            continue

        child = replace(
            take,
            start=new_start,
            text=text,
            words=right_words,
            signals=(replace(take.signals, start=new_start) if take.signals is not None else None),
        )
        output.append(child)
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "internal_broken_attempt_yields_to_clean_retake",
            "event_kind": _kind(event.kind),
            "event_confidence": round(float(event.confidence), 4),
            "evidence_type": evidence_type,
            "original_start": round(float(take.start), 3),
            "retake_start": round(new_start, 3),
            "shared_content_tokens": shared,
            "left_coverage": round(left_cov, 4),
            "right_coverage": round(right_cov, 4),
            "kept_text": text,
        })

    return tuple(output), tuple(diagnostics)


def install_internal_retake_winner() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_internal_retake_winner", False):
        return

    def apply_with_internal_retake_winner(takes, context=None):
        take_tuple = tuple(takes)
        kept, discarded, decisions = original(take_tuple, context)
        resolved, diagnostics = prefer_internal_clean_retakes(kept, context)
        if not diagnostics:
            return kept, discarded, decisions
        extra = tuple(
            CleanCutDecision(
                clip_id=str(item["clip_id"]),
                keep=True,
                reason=str(item["reason"]),
                confidence=0.99,
            )
            for item in diagnostics
        )
        return resolved, discarded, tuple(decisions) + extra

    apply_with_internal_retake_winner._cutsell_internal_retake_winner = True
    clean_cut.apply_clean_cut = apply_with_internal_retake_winner
