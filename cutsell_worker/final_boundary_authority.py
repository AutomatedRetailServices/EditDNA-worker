"""Final source-aware boundary authority for Universal Clean Cut.

Selection authority chooses WHICH delivery survives. This module owns only WHERE the
surviving delivery may begin and end. A boundary is invalid when it starts after the
true beginning of the same spoken idea or ends before that idea's last valid word.

Rules:
- inspect the full source transcript, not only the already-trimmed clip transcript;
- never keep a boundary that lands inside a spoken word;
- conservatively expand through tightly-connected speech until a real idea wall is
  reached (terminal punctuation or a substantial inter-word pause);
- never cross an obvious recording/session pause;
- preserve source order and logical clip identity;
- never allow complete-idea recovery to create overlapping/duplicated source speech
  between adjacent selected clips;
- preserve intentional speech-safe gaps created by boundary polish inside one logical
  clip; final idea recovery must not glue proven dead-air/reset slack back in;
- fail open: when transcript evidence is ambiguous, retain more speech, never less.

This is intentionally not a semantic composer and never changes take selection.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .asr import ASRProvider
from .contracts import DraftClip, ProcessingResult, Word

_TERMINAL = (".", "?", "!", "…")


def _terminal(word: Word) -> bool:
    return str(word.text or "").strip().endswith(_TERMINAL)


def _source_words(
    local_paths: Mapping[str, str],
    asr_provider: ASRProvider,
) -> dict[str, tuple[Word, ...]]:
    out: dict[str, tuple[Word, ...]] = {}
    for source_id, path in local_paths.items():
        segments = asr_provider.transcribe(path, source_asset_id=source_id, language_hint=None)
        words = tuple(
            sorted(
                (word for segment in segments for word in tuple(segment.words)),
                key=lambda item: (float(item.start), float(item.end)),
            )
        )
        out[source_id] = words
    return out


def _overlapping_indices(words: tuple[Word, ...], start: float, end: float) -> tuple[int, int] | None:
    hits = [
        index for index, word in enumerate(words)
        if float(word.end) > start + 1e-6 and float(word.start) < end - 1e-6
    ]
    if not hits:
        return None
    return min(hits), max(hits)


def _expand_left(
    words: tuple[Word, ...],
    first_index: int,
    *,
    max_lookback_sec: float = 3.0,
    idea_pause_sec: float = 0.62,
) -> int:
    first = first_index
    anchor = float(words[first_index].start)
    while first > 0:
        previous = words[first - 1]
        current = words[first]
        gap = float(current.start) - float(previous.end)
        if gap >= idea_pause_sec:
            break
        if _terminal(previous):
            break
        if anchor - float(previous.start) > max_lookback_sec:
            break
        first -= 1
    return first


def _expand_right(
    words: tuple[Word, ...],
    last_index: int,
    *,
    max_lookahead_sec: float = 3.0,
    idea_pause_sec: float = 0.62,
) -> int:
    last = last_index
    anchor = float(words[last_index].end)
    while last + 1 < len(words):
        current = words[last]
        following = words[last + 1]
        if _terminal(current):
            break
        gap = float(following.start) - float(current.end)
        if gap >= idea_pause_sec:
            break
        if float(following.end) - anchor > max_lookahead_sec:
            break
        last += 1
    return last


def _clip_from_envelope(
    clip: DraftClip,
    source_words: tuple[Word, ...],
) -> tuple[DraftClip, dict]:
    overlap = _overlapping_indices(source_words, float(clip.start), float(clip.end))
    if overlap is None:
        return clip, {
            "clip_id": clip.clip_id,
            "action": "keep_no_source_word_alignment",
            "original_start": round(float(clip.start), 3),
            "original_end": round(float(clip.end), 3),
        }

    original_first, original_last = overlap
    first = _expand_left(source_words, original_first)
    last = _expand_right(source_words, original_last)

    while first > 0 and float(source_words[first - 1].start) < float(clip.start) < float(source_words[first - 1].end):
        first -= 1
    while last + 1 < len(source_words) and float(source_words[last + 1].start) < float(clip.end) < float(source_words[last + 1].end):
        last += 1

    envelope_words = tuple(source_words[first:last + 1])
    new_start = min(float(clip.start), float(envelope_words[0].start))
    new_end = max(float(clip.end), float(envelope_words[-1].end))
    text = " ".join(str(word.text).strip() for word in envelope_words).strip()

    timing_changed = abs(new_start - float(clip.start)) > 1e-4 or abs(new_end - float(clip.end)) > 1e-4
    semantic_changed = tuple(clip.words) != envelope_words or str(clip.text or "").strip() != text
    changed = timing_changed or semantic_changed
    updated = replace(
        clip,
        start=new_start,
        end=new_end,
        words=envelope_words,
        text=text or clip.text,
        caption_text=text or clip.caption_text,
    ) if changed else clip

    if timing_changed:
        action = "expand_to_complete_idea_envelope"
    elif semantic_changed:
        action = "refresh_complete_idea_envelope_text"
    else:
        action = "keep_complete_idea_envelope"

    return updated, {
        "clip_id": clip.clip_id,
        "action": action,
        "original_start": round(float(clip.start), 3),
        "original_end": round(float(clip.end), 3),
        "result_start": round(float(updated.start), 3),
        "result_end": round(float(updated.end), 3),
        "added_leading_sec": round(max(0.0, float(clip.start) - float(updated.start)), 3),
        "added_trailing_sec": round(max(0.0, float(updated.end) - float(clip.end)), 3),
        "first_word": str(envelope_words[0].text),
        "last_word": str(envelope_words[-1].text),
        "word_count": len(envelope_words),
    }


def _words_inside(words: tuple[Word, ...], start: float, end: float) -> tuple[Word, ...]:
    return tuple(
        word for word in words
        if float(word.start) >= start - 1e-6 and float(word.end) <= end + 1e-6
    )


def _rebuild_clip(clip: DraftClip, words: tuple[Word, ...], start: float, end: float) -> DraftClip:
    kept_words = _words_inside(words, start, end)
    if not kept_words:
        return replace(clip, start=start, end=end)
    safe_start = min(start, float(kept_words[0].start))
    safe_end = max(end, float(kept_words[-1].end))
    text = " ".join(str(word.text).strip() for word in kept_words).strip()
    return replace(
        clip,
        start=safe_start,
        end=safe_end,
        words=kept_words,
        text=text or clip.text,
        caption_text=text or clip.caption_text,
    )


def _reconcile_same_source_overlaps(
    originals: tuple[DraftClip, ...],
    expanded: list[DraftClip],
    source_map: dict[str, tuple[Word, ...]],
) -> tuple[list[DraftClip], list[dict]]:
    """Remove expansion overlap and preserve already-proven interior microtrim gaps."""
    output = list(expanded)
    rows: list[dict] = []
    for index in range(len(output) - 1):
        left = output[index]
        right = output[index + 1]
        left_orig = originals[index]
        right_orig = originals[index + 1]
        if left.source_asset_id != right.source_asset_id:
            continue

        words = source_map.get(left.source_asset_id, ())
        original_gap = float(right_orig.start) - float(left_orig.end)

        if left_orig.clip_id == right_orig.clip_id and original_gap > 0.02:
            fixed_left = _rebuild_clip(
                left, words, float(left.start), min(float(left.end), float(left_orig.end))
            )
            fixed_right = _rebuild_clip(
                right, words, max(float(right.start), float(right_orig.start)), float(right.end)
            )
            output[index] = fixed_left
            output[index + 1] = fixed_right
            rows.append({
                "action": "preserve_polished_interior_gap",
                "clip_id": left_orig.clip_id,
                "gap_start": round(float(left_orig.end), 3),
                "gap_end": round(float(right_orig.start), 3),
                "gap_sec": round(original_gap, 3),
            })
            continue

        if float(left.end) <= float(right.start) + 1e-6:
            continue

        if float(left_orig.end) > float(right_orig.start) + 1e-6:
            rows.append({
                "action": "keep_original_selection_overlap",
                "left_clip_id": left.clip_id,
                "right_clip_id": right.clip_id,
                "overlap_sec": round(float(left.end) - float(right.start), 3),
            })
            continue

        left_limit = float(right_orig.start)
        right_limit = float(left_orig.end)
        new_left_end = min(float(left.end), left_limit)
        new_right_start = max(float(right.start), right_limit)

        if new_left_end > new_right_start:
            seam = (float(left_orig.end) + float(right_orig.start)) / 2.0
            new_left_end = min(new_left_end, seam)
            new_right_start = max(new_right_start, seam)

        new_left_end = max(new_left_end, float(left_orig.end))
        new_right_start = min(new_right_start, float(right_orig.start))

        fixed_left = _rebuild_clip(left, words, float(left.start), new_left_end)
        fixed_right = _rebuild_clip(right, words, new_right_start, float(right.end))

        if float(fixed_left.end) > float(fixed_right.start) + 1e-6:
            seam = (float(left_orig.end) + float(right_orig.start)) / 2.0
            fixed_left = _rebuild_clip(left, words, float(left.start), max(float(left_orig.end), seam))
            fixed_right = _rebuild_clip(right, words, min(float(right_orig.start), seam), float(right.end))

        output[index] = fixed_left
        output[index + 1] = fixed_right
        rows.append({
            "action": "reconcile_expansion_overlap",
            "left_clip_id": left.clip_id,
            "right_clip_id": right.clip_id,
            "original_left_end": round(float(left_orig.end), 3),
            "original_right_start": round(float(right_orig.start), 3),
            "expanded_left_end": round(float(left.end), 3),
            "expanded_right_start": round(float(right.start), 3),
            "result_left_end": round(float(fixed_left.end), 3),
            "result_right_start": round(float(fixed_right.start), 3),
        })
    return output, rows


def enforce_complete_idea_boundaries(
    result: ProcessingResult,
    local_paths: Mapping[str, str],
    *,
    asr_provider: ASRProvider,
) -> ProcessingResult:
    """Make complete-idea + complete-word protection the final source boundary authority."""
    if not hasattr(result.draft, "selected") or not result.draft.selected:
        return result

    source_map = _source_words(local_paths, asr_provider)
    originals = tuple(result.draft.selected)
    selected: list[DraftClip] = []
    diagnostics: list[dict] = []
    for clip in originals:
        words = source_map.get(clip.source_asset_id, ())
        if not words:
            selected.append(clip)
            diagnostics.append({
                "clip_id": clip.clip_id,
                "action": "keep_source_transcript_unavailable",
                "original_start": round(float(clip.start), 3),
                "original_end": round(float(clip.end), 3),
            })
            continue
        updated, row = _clip_from_envelope(clip, words)
        selected.append(updated)
        diagnostics.append(row)

    selected, overlap_rows = _reconcile_same_source_overlaps(originals, selected, source_map)
    diagnostics.extend(overlap_rows)

    preserved_gap_rows = [row for row in overlap_rows if row.get("action") == "preserve_polished_interior_gap"]
    diag = dict(result.draft.diagnostics or {})
    diag["final_boundary_authority"] = diagnostics[:600]
    diag["final_boundary_authority_rule"] = (
        "full_source_transcript -> complete idea envelope -> complete word lock -> "
        "preserve proven polished interior gaps -> neighbor-original-span overlap guard -> visual slack only"
    )
    diag["final_boundary_overlap_reconciliation_count"] = len(overlap_rows)
    diag["final_boundary_preserved_polish_gap_count"] = len(preserved_gap_rows)
    draft = replace(result.draft, selected=tuple(selected), diagnostics=diag)
    return replace(result, draft=draft)
