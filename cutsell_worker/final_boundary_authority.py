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

    # Hard word lock. If the existing boundary is inside a word, the full word wins.
    while first > 0 and float(source_words[first - 1].start) < float(clip.start) < float(source_words[first - 1].end):
        first -= 1
    while last + 1 < len(source_words) and float(source_words[last + 1].start) < float(clip.end) < float(source_words[last + 1].end):
        last += 1

    envelope_words = tuple(source_words[first:last + 1])
    new_start = min(float(clip.start), float(envelope_words[0].start))
    new_end = max(float(clip.end), float(envelope_words[-1].end))
    text = " ".join(str(word.text).strip() for word in envelope_words).strip()

    changed = abs(new_start - float(clip.start)) > 1e-4 or abs(new_end - float(clip.end)) > 1e-4
    updated = replace(
        clip,
        start=new_start,
        end=new_end,
        words=envelope_words,
        text=text or clip.text,
        caption_text=text or clip.caption_text,
    ) if changed else clip

    return updated, {
        "clip_id": clip.clip_id,
        "action": "expand_to_complete_idea_envelope" if changed else "keep_complete_idea_envelope",
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
    selected: list[DraftClip] = []
    diagnostics: list[dict] = []
    for clip in result.draft.selected:
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

    diag = dict(result.draft.diagnostics or {})
    diag["final_boundary_authority"] = diagnostics[:600]
    diag["final_boundary_authority_rule"] = (
        "full_source_transcript -> complete idea envelope -> complete word lock -> visual slack only"
    )
    draft = replace(result.draft, selected=tuple(selected), diagnostics=diag)
    return replace(result, draft=draft)
