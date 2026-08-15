"""Conservative local cleanup for explicit recording-break speech.

This module targets creator/process failures that are semantically about the recording
itself (for example ``I can't talk`` or ``let's do that again``).  It deliberately
does not treat profanity alone, ordinary negation, intentional single-word emphasis,
or an isolated creator reaction as destructive evidence.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_EXPLICIT_FAILURE_RE = re.compile(
    r"\bi\s+(?:can(?:not|'t)|cant)\s+talk\b|"
    r"\bwhy\s+(?:can(?:not|'t)|cant)\s+i\s+talk\b|"
    r"\bi\s+(?:do\s+not|don't|dont)\s+know\s+how\s+to\s+end\b|"
    r"\b(?:let's|lets|let\s+us)\s+do\s+that\s+again\b|"
    r"\b(?:let's|lets|let\s+us)\s+(?:try|start)\s+(?:that\s+)?again\b",
    re.IGNORECASE,
)
_HAND_SELF_DIRECTION_RE = re.compile(
    r"\bwhat\s+are\s+you\s+doing\s+with\s+your\s+hands\b",
    re.IGNORECASE,
)
_REACTION_RE = re.compile(
    r"\bwhat\s+(?:just\s+)?(?:happened|is\s+happening|the\s+(?:frig|frick|fuck))\b|"
    r"\b(?:what|the)\s+(?:frig|frick|fuck)\s+(?:is\s+)?happening\b|"
    r"\b(?:fuck|frig|frick)\s+is\s+happening\b",
    re.IGNORECASE,
)
_SPEECH_SELF_REVIEW_RE = re.compile(
    r"\bwhat\s+did\s+i\s+(?:just\s+)?say\b",
    re.IGNORECASE,
)
_SHORT_CONFUSION_RE = re.compile(r"^\s*what[?!.\s]*$", re.IGNORECASE)
_RECOVERY_TRANSITION_RE = re.compile(
    r"^\s*(?:okay|ok)[,.!?\s]*(?:anyways?|anyway)\b",
    re.IGNORECASE,
)
_SELF_CRITIQUE_RE = re.compile(
    r"\bwhy\s+do\s+i\s+(?:keep\s+)?say(?:ing)?\b",
    re.IGNORECASE,
)
_FRUSTRATION_TOKENS = frozenset({
    "fuck", "fucking", "frig", "frick", "damn", "ugh", "oops", "stupid", "crap",
})
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _has_frustration(text: str) -> bool:
    return bool(set(_tokens(text)).intersection(_FRUSTRATION_TOKENS))


def _has_repeated_multiword_phrase(text: str) -> bool:
    """Find repeated 2-6 word phrases while ignoring single-word emphasis.

    Two occurrences are enough only when another independent signal (frustration)
    is present.  This catches abandoned restart loops such as a phrase repeated after
    an expletive without turning normal rhetorical repetition into a deletion rule.
    """
    tokens = _tokens(text)
    if len(tokens) < 6:
        return False
    for width in range(min(6, len(tokens) // 2), 1, -1):
        seen: set[tuple[str, ...]] = set()
        for index in range(0, len(tokens) - width + 1):
            gram = tokens[index:index + width]
            if len(set(gram)) < 2:
                continue
            if gram in seen:
                return True
            seen.add(gram)
    return False


def _recording_break_reason(take: CandidateTake) -> str | None:
    text = str(take.text or "").strip()
    if not text:
        return None
    if _EXPLICIT_FAILURE_RE.search(text):
        return "explicit_recording_failure"
    if _HAND_SELF_DIRECTION_RE.search(text) and _has_frustration(text):
        return "frustrated_self_direction"
    if _has_frustration(text) and _has_repeated_multiword_phrase(text):
        return "frustrated_internal_restart_repetition"
    return None


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _multimodal_break_counts(
    takes: tuple[CandidateTake, ...],
    context: WholeVideoContext | None,
) -> tuple[int, int]:
    if not takes or context is None:
        return 0, 0
    source_asset_id = takes[0].source_asset_id
    start = min(take.start for take in takes) - 0.45
    end = max(take.end for take in takes) + 0.45
    events = tuple(
        event for event in _source_events(context, source_asset_id)
        if event.end >= start and event.start <= end
    )
    reset_count = sum(
        1 for event in events
        if event.kind in _RESET_KINDS and event.confidence >= 0.90
    )
    break_count = sum(
        1 for event in events
        if event.kind in _BREAK_KINDS and event.confidence >= 0.72
    )
    return reset_count, break_count


def _multimodal_break_window(
    takes: tuple[CandidateTake, ...],
    context: WholeVideoContext | None,
) -> bool:
    """Require dense physical reset plus expression/engagement break evidence."""
    reset_count, break_count = _multimodal_break_counts(takes, context)
    return reset_count >= 4 and break_count >= 2


def _reaction_cluster_ids(
    kept: tuple[CandidateTake, ...],
    context: WholeVideoContext | None,
    *,
    maximum_gap_sec: float = 2.25,
    maximum_span_sec: float = 14.0,
) -> set[str]:
    """Return IDs in a corroborated multi-take recording-break reaction cluster."""
    ordered = tuple(sorted(kept, key=lambda item: (item.source_order, item.start, item.end)))
    eligible = lambda take: (
        take.duration_sec <= 4.0
        and bool(_REACTION_RE.search(str(take.text or "")) or _RECOVERY_TRANSITION_RE.search(str(take.text or "")))
    )
    remove_ids: set[str] = set()
    index = 0
    while index < len(ordered):
        if not eligible(ordered[index]):
            index += 1
            continue
        cluster = [ordered[index]]
        cursor = index + 1
        while cursor < len(ordered):
            previous = cluster[-1]
            current = ordered[cursor]
            if current.source_asset_id != previous.source_asset_id:
                break
            if not eligible(current):
                break
            if current.start - previous.end > maximum_gap_sec:
                break
            if current.end - cluster[0].start > maximum_span_sec:
                break
            cluster.append(current)
            cursor += 1
        reaction_count = sum(1 for take in cluster if _REACTION_RE.search(str(take.text or "")))
        if len(cluster) >= 3 and reaction_count >= 2 and _multimodal_break_window(tuple(cluster), context):
            remove_ids.update(take.clip_id for take in cluster)
        index = max(index + 1, cursor)
    return remove_ids


def _speech_self_review_pair_ids(
    kept: tuple[CandidateTake, ...],
    context: WholeVideoContext | None,
    *,
    maximum_gap_sec: float = 2.0,
) -> set[str]:
    """Remove a self-review question plus immediate confused echo during a reset.

    ``What?`` by itself is protected.  The pair becomes destructive only when the
    creator first explicitly reviews their own just-spoken line (``what did I just
    say?``), immediately follows with a tiny confusion reaction, and local vision sees
    dense physical reset plus at least one expression/engagement break across the pair.
    """
    ordered = tuple(sorted(kept, key=lambda item: (item.source_order, item.start, item.end)))
    remove_ids: set[str] = set()
    for index, take in enumerate(ordered[:-1]):
        if take.duration_sec > 2.5 or not _SPEECH_SELF_REVIEW_RE.search(str(take.text or "")):
            continue
        following = ordered[index + 1]
        if following.source_asset_id != take.source_asset_id:
            continue
        if following.duration_sec > 1.5 or not _SHORT_CONFUSION_RE.search(str(following.text or "")):
            continue
        gap = following.start - take.end
        if not 0.0 <= gap <= maximum_gap_sec:
            continue
        resets, breaks = _multimodal_break_counts((take, following), context)
        if resets >= 4 and breaks >= 1:
            remove_ids.update((take.clip_id, following.clip_id))
    return remove_ids


def _contextual_self_critique_ids(
    kept: tuple[CandidateTake, ...],
    context: WholeVideoContext | None,
    *,
    maximum_gap_sec: float = 1.5,
) -> set[str]:
    """Remove self-critique only when it leads directly into a proven failure.

    Rhetorical speech such as ``why do I say this every morning?`` can be valid
    creator content.  The local path therefore requires a following same-source take
    that explicitly admits a recording failure plus dense multimodal reset/break
    evidence across the two-take window.
    """
    ordered = tuple(sorted(kept, key=lambda item: (item.source_order, item.start, item.end)))
    remove_ids: set[str] = set()
    for index, take in enumerate(ordered[:-1]):
        if not _SELF_CRITIQUE_RE.search(str(take.text or "")):
            continue
        following = ordered[index + 1]
        if following.source_asset_id != take.source_asset_id:
            continue
        gap = following.start - take.end
        if not -0.05 <= gap <= maximum_gap_sec:
            continue
        if not _EXPLICIT_FAILURE_RE.search(str(following.text or "")):
            continue
        if _multimodal_break_window((take, following), context):
            remove_ids.add(take.clip_id)
    return remove_ids


def apply_recording_break_cleanup(
    kept: Iterable[CandidateTake],
    context: WholeVideoContext | None = None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    cluster_ids = _reaction_cluster_ids(kept_tuple, context)
    speech_self_review_ids = _speech_self_review_pair_ids(kept_tuple, context)
    self_critique_ids = _contextual_self_critique_ids(kept_tuple, context)
    survivors = []
    removed = []
    diagnostics = []
    for take in kept_tuple:
        reason = _recording_break_reason(take)
        if reason is None and take.clip_id in cluster_ids:
            reason = "multimodal_recording_break_reaction_cluster"
        if reason is None and take.clip_id in speech_self_review_ids:
            reason = "speech_self_review_confusion_pair_with_physical_reset"
        if reason is None and take.clip_id in self_critique_ids:
            reason = "self_critique_before_explicit_recording_failure"
        if reason is None:
            survivors.append(take)
            continue
        removed.append(take)
        diagnostics.append({"clip_id": take.clip_id, "reason": reason, "text": take.text})
    return tuple(survivors), tuple(removed), tuple(diagnostics)


def install_recording_break_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_recording_break_cleanup", False):
        return

    def apply_with_recording_breaks(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, contextual_discarded, diagnostics = apply_recording_break_cleanup(kept, context)
        if not contextual_discarded:
            return kept, discarded, decisions

        reason_by_id = {item["clip_id"]: item["reason"] for item in diagnostics}
        extra = tuple(
            CleanCutDecision(
                clip_id=take.clip_id,
                keep=False,
                reason=reason_by_id[take.clip_id],
                confidence=0.96,
            )
            for take in contextual_discarded
        )
        return kept, tuple(discarded) + tuple(contextual_discarded), tuple(decisions) + extra

    apply_with_recording_breaks._cutsell_recording_break_cleanup = True
    clean_cut.apply_clean_cut = apply_with_recording_breaks
