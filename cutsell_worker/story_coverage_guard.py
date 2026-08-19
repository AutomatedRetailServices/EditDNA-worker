"""Fail-open protection for unique long-form audience-facing story coverage.

Clean Cut must not destroy a creator's story simply because one local/semantic stage
mislabels a long unique paragraph. This wrapper restores only long, information-dense
material when there is no strong recording-failure evidence and no competing retry that
covers the same idea. It intentionally does not rescue short fragments, explicit BTS,
or repeated attempts that have a better peer elsewhere in the same source.
"""
from __future__ import annotations

import re
from typing import Iterable

from .contracts import CandidateTake, CleanCutDecision
from .temporal_editing import harmful_events_for_take
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "so", "that", "the", "this", "to", "was",
    "we", "with", "you", "your", "al", "como", "con", "cuando", "de", "del", "el", "en",
    "es", "la", "las", "le", "les", "lo", "los", "me", "mi", "mis", "o", "para", "pero",
    "por", "porque", "que", "se", "si", "sin", "su", "sus", "un", "una", "unos", "unas", "y", "yo",
})
_META_RE = re.compile(
    r"\b(?:let'?s|lets)\s+(?:do|try|start)\s+(?:that\s+)?again\b|"
    r"\bi\s+(?:can(?:not|'t)|cant)\s+talk\b|"
    r"\bwhat\s+did\s+i\s+(?:just\s+)?say\b|"
    r"\b(?:start|take)\s+(?:that\s+)?again\b|"
    r"\b(?:cut|stop)\s+(?:the\s+)?(?:camera|recording)\b",
    re.IGNORECASE,
)
_STRONG_KINDS = frozenset({
    "false_start", "wrong_take", "verbal_fumble", "visual_fumble", "retry_setup",
    "breaking_character", "recording_joke", "camera_adjustment", "searching_for_words",
    "product_handling_mistake",
})
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token.casefold() for token in _TOKEN_RE.findall(str(text or ""))
        if len(token) >= 3 and token.casefold() not in _STOP
    )


def _retry_peer(take: CandidateTake, all_takes: tuple[CandidateTake, ...]) -> bool:
    a = set(_content_tokens(take.text))
    if len(a) < 5:
        return False
    for other in all_takes:
        if other.clip_id == take.clip_id or other.source_asset_id != take.source_asset_id:
            continue
        b = set(_content_tokens(other.text))
        if len(b) < 5:
            continue
        shared = len(a & b)
        containment = shared / max(1, min(len(a), len(b)))
        if shared >= 4 and containment >= 0.55:
            return True
    return False


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _strong_failure(take: CandidateTake, context: WholeVideoContext | None) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    for event in harmful_events_for_take(take, context, minimum_confidence=0.90):
        if str(event.kind) in _STRONG_KINDS:
            reasons.append(f"event:{event.kind}:{event.confidence:.2f}")

    events = tuple(
        event for event in _source_events(context, take.source_asset_id)
        if event.end >= take.start - 0.15 and event.start <= take.end + 0.15
    )
    resets = [event for event in events if event.kind in _RESET_KINDS and event.confidence >= 0.92]
    breaks = [event for event in events if event.kind in _BREAK_KINDS and event.confidence >= 0.84]
    if len(resets) >= 3 and len(breaks) >= 2:
        reasons.append(f"dense_multimodal_break:{len(resets)}:{len(breaks)}")

    signals = take.signals
    if signals is not None:
        if float(signals.visual_fumble) >= 0.90:
            reasons.append(f"visual_fumble:{float(signals.visual_fumble):.2f}")
        if float(signals.distraction_risk) >= 0.93:
            reasons.append(f"distraction_risk:{float(signals.distraction_risk):.2f}")
    return bool(reasons), tuple(reasons)


def restore_unique_story_coverage(
    kept: Iterable[CandidateTake],
    discarded: Iterable[CandidateTake],
    original_takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    *,
    minimum_duration_sec: float = 7.0,
    minimum_content_tokens: int = 8,
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    discarded_tuple = tuple(discarded)
    originals = tuple(original_takes)
    restored_ids: set[str] = set()
    diagnostics = []

    for take in discarded_tuple:
        tokens = _content_tokens(take.text)
        if take.duration_sec < minimum_duration_sec or len(tokens) < minimum_content_tokens:
            continue
        if _META_RE.search(str(take.text or "")):
            continue
        strong_failure, reasons = _strong_failure(take, context)
        retry_peer = _retry_peer(take, originals)
        if strong_failure or retry_peer:
            continue
        restored_ids.add(take.clip_id)
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "restore_unique_story_coverage",
            "duration_sec": round(take.duration_sec, 3),
            "content_token_count": len(tokens),
            "retry_peer": False,
            "strong_failure": False,
            "strong_failure_reasons": list(reasons),
            "text": take.text,
        })

    if not restored_ids:
        return kept_tuple, discarded_tuple, ()

    restored = tuple(take for take in discarded_tuple if take.clip_id in restored_ids)
    survivors = tuple(take for take in discarded_tuple if take.clip_id not in restored_ids)
    merged_kept = tuple(sorted(
        (*kept_tuple, *restored),
        key=lambda item: (item.source_order, item.start, item.end, item.clip_id),
    ))
    return merged_kept, survivors, tuple(diagnostics)


def install_story_coverage_guard() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_story_coverage_guard", False):
        return

    def apply_with_story_guard(takes, context=None):
        take_tuple = tuple(takes)
        kept, discarded, decisions = original(take_tuple, context)
        kept, discarded, diagnostics = restore_unique_story_coverage(
            kept, discarded, take_tuple, context
        )
        if not diagnostics:
            return kept, discarded, decisions
        restored_ids = {item["clip_id"] for item in diagnostics}
        filtered_decisions = tuple(
            decision for decision in decisions
            if not (decision.clip_id in restored_ids and not decision.keep)
        )
        extra = tuple(
            CleanCutDecision(
                clip_id=str(item["clip_id"]),
                keep=True,
                reason="restore_unique_story_coverage",
                confidence=0.99,
            )
            for item in diagnostics
        )
        return kept, discarded, filtered_decisions + extra

    apply_with_story_guard._cutsell_story_coverage_guard = True
    clean_cut.apply_clean_cut = apply_with_story_guard
