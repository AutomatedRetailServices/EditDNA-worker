"""Final transcript authority for complete spoken ideas.

This module is intentionally the last source-timeline boundary pass for Clean Cut.
Selection authority may choose *which* delivery survives, but it may not leave a
render boundary after the real beginning of that delivery's valid sentence/idea or
before its last valid word.  The full-source transcript is used so a clipped draft
cannot pretend that its first remaining token was the actual beginning.

The authority is conservative around recording-process evidence: it will not expand
across a high-confidence retry/fumble/reset event or into another selected clip.
Ambiguous expansions fail open, but every accepted envelope is snapped outside the
first/last word with small acoustic guards.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable

from .contracts import DraftClip, ProcessingResult, TranscriptSegment, Word
from .whole_video_analysis import WholeVideoContext

_TERMINAL_RE = re.compile(r"[.!?…][\"'”’)]*$")
_HARMFUL_KINDS = {
    "false_start", "wrong_take", "verbal_fumble", "visual_fumble", "body_reset",
    "retry_setup", "frustration", "breaking_character", "recording_joke",
    "accidental_laughter", "camera_adjustment", "product_handling_mistake",
    "searching_for_words", "unintentional_dead_air",
}


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _terminal(word: Word) -> bool:
    return bool(_TERMINAL_RE.search(str(word.text or "").strip()))


def _words_by_source(transcripts: Iterable[TranscriptSegment]) -> dict[str, tuple[Word, ...]]:
    output: dict[str, list[Word]] = {}
    for segment in transcripts:
        bucket = output.setdefault(segment.source_asset_id, [])
        if segment.words:
            bucket.extend(segment.words)
        elif str(segment.text or "").strip():
            # Segment-only ASR has no safe word envelope; do not synthesize timestamps.
            continue
    return {
        source_id: tuple(sorted(words, key=lambda w: (float(w.start), float(w.end))))
        for source_id, words in output.items()
    }


def _boundary_after(left: Word, right: Word, *, pause_sec: float = 0.68) -> bool:
    return _terminal(left) or float(right.start) - float(left.end) >= pause_sec


def _nearest_word_indices(words: tuple[Word, ...], clip: DraftClip) -> tuple[int, int] | None:
    overlapping = [
        i for i, word in enumerate(words)
        if float(word.end) > float(clip.start) - 0.08 and float(word.start) < float(clip.end) + 0.08
    ]
    if not overlapping:
        return None
    return min(overlapping), max(overlapping)


def _idea_indices(words: tuple[Word, ...], first: int, last: int, *, max_expand_sec: float = 4.0) -> tuple[int, int]:
    idea_first = first
    while idea_first > 0:
        prev = words[idea_first - 1]
        current = words[idea_first]
        if _boundary_after(prev, current):
            break
        if float(words[first].start) - float(prev.start) > max_expand_sec:
            break
        idea_first -= 1

    idea_last = last
    while idea_last + 1 < len(words):
        current = words[idea_last]
        nxt = words[idea_last + 1]
        if _boundary_after(current, nxt):
            break
        if float(nxt.end) - float(words[last].end) > max_expand_sec:
            break
        idea_last += 1
    return idea_first, idea_last


def _harmful_crossing(context: WholeVideoContext | None, source_id: str, start: float, end: float) -> bool:
    if context is None or end <= start:
        return False
    for source in context.sources:
        if source.source_asset_id != source_id:
            continue
        for event in source.events:
            if float(event.confidence) < 0.72 or _kind(event.kind) not in _HARMFUL_KINDS:
                continue
            if float(event.end) > start and float(event.start) < end:
                return True
    return False


def _overlaps_other_selected(candidate_start: float, candidate_end: float, clip: DraftClip, selected: tuple[DraftClip, ...]) -> bool:
    for other in selected:
        if other.clip_id == clip.clip_id and other.start == clip.start and other.end == clip.end:
            continue
        if other.source_asset_id != clip.source_asset_id:
            continue
        if float(other.end) > candidate_start + 0.01 and float(other.start) < candidate_end - 0.01:
            return True
    return False


def enforce_complete_idea_boundaries(
    result: ProcessingResult,
    transcripts: Iterable[TranscriptSegment],
    whole_video_context: WholeVideoContext | None,
    *,
    leading_guard_sec: float = 0.045,
    trailing_guard_sec: float = 0.055,
) -> ProcessingResult:
    """Expand clipped selected spans to the complete source idea and speech envelope.

    The function never chooses a different take. It only repairs physical boundaries
    of the already-selected take using the full source transcript. Expansion is refused
    when it would cross proven recording garbage or another selected span.
    """
    if not hasattr(result.draft, "selected"):
        return result

    source_words = _words_by_source(transcripts)
    selected = tuple(result.draft.selected)
    repaired: list[DraftClip] = []
    diagnostics: list[dict] = []

    for clip in selected:
        words = source_words.get(clip.source_asset_id, ())
        indices = _nearest_word_indices(words, clip)
        if indices is None:
            repaired.append(clip)
            diagnostics.append({"clip_id": clip.clip_id, "action": "keep", "reason": "no_full_source_word_evidence"})
            continue

        first, last = indices
        idea_first, idea_last = _idea_indices(words, first, last)
        idea_words = tuple(words[idea_first:idea_last + 1])
        candidate_start = max(0.0, float(idea_words[0].start) - leading_guard_sec)
        candidate_end = float(idea_words[-1].end) + trailing_guard_sec

        expand_left = candidate_start < float(clip.start) - 0.01
        expand_right = candidate_end > float(clip.end) + 0.01

        if expand_left and _harmful_crossing(whole_video_context, clip.source_asset_id, candidate_start, float(clip.start)):
            candidate_start = float(clip.start)
            idea_words = tuple(word for word in idea_words if float(word.end) > candidate_start - 0.01)
            expand_left = False
        if expand_right and _harmful_crossing(whole_video_context, clip.source_asset_id, float(clip.end), candidate_end):
            candidate_end = float(clip.end)
            idea_words = tuple(word for word in idea_words if float(word.start) < candidate_end + 0.01)
            expand_right = False

        if _overlaps_other_selected(candidate_start, candidate_end, clip, selected):
            candidate_start, candidate_end = float(clip.start), float(clip.end)
            idea_words = tuple(clip.words)
            expand_left = expand_right = False
            reason = "expansion_would_overlap_other_selected_clip"
        else:
            reason = "complete_idea_envelope_locked"

        if not idea_words:
            repaired.append(clip)
            diagnostics.append({"clip_id": clip.clip_id, "action": "keep", "reason": "empty_safe_idea_words"})
            continue

        # Absolute word-envelope invariant: accepted boundaries must sit outside speech.
        first_word_start = float(idea_words[0].start)
        last_word_end = float(idea_words[-1].end)
        candidate_start = min(candidate_start, first_word_start - min(leading_guard_sec, 0.02))
        candidate_start = max(0.0, candidate_start)
        candidate_end = max(candidate_end, last_word_end + min(trailing_guard_sec, 0.025))

        text = " ".join(str(word.text).strip() for word in idea_words).strip()
        new_clip = replace(
            clip,
            start=candidate_start,
            end=candidate_end,
            words=idea_words,
            text=text or clip.text,
            caption_text=text or clip.caption_text,
        )
        repaired.append(new_clip)
        diagnostics.append({
            "clip_id": clip.clip_id,
            "action": "repair" if (abs(candidate_start - float(clip.start)) > 0.01 or abs(candidate_end - float(clip.end)) > 0.01) else "validate",
            "reason": reason,
            "original_start": round(float(clip.start), 3),
            "original_end": round(float(clip.end), 3),
            "result_start": round(candidate_start, 3),
            "result_end": round(candidate_end, 3),
            "idea_first_word": str(idea_words[0].text),
            "idea_last_word": str(idea_words[-1].text),
            "first_word_start": round(first_word_start, 3),
            "last_word_end": round(last_word_end, 3),
            "leading_speech_lock_ok": candidate_start <= first_word_start + 1e-6,
            "trailing_speech_lock_ok": candidate_end >= last_word_end - 1e-6,
            "expanded_left": expand_left,
            "expanded_right": expand_right,
        })

    diag = dict(result.draft.diagnostics or {})
    diag["idea_boundary_authority"] = diagnostics[:600]
    diag["idea_boundary_authority_ok"] = all(
        row.get("leading_speech_lock_ok", True) and row.get("trailing_speech_lock_ok", True)
        for row in diagnostics
    )
    draft = replace(result.draft, selected=tuple(repaired), diagnostics=diag)
    return replace(result, draft=draft)
