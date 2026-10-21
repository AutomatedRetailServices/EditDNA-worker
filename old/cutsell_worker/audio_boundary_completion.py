"""Audio-confirmed completion repair for ASR boundary misses.

A first-pass ASR segmentation can occasionally omit the final semantic word of an
otherwise clean delivery while still placing the physical edit boundary after that word.
If the following short fragment is then correctly classified as a failed fumble, a
text-only rescue cannot recover the missing word without fabricating language.

This module solves that narrow failure by listening to a short source-audio window again.
It never guesses a word from Gold text or topic context. A repair is allowed only when:
- a selected clip is immediately followed by a short discarded fragment;
- Hybrid marked that fragment failed with >= 0.90 confidence;
- the discarded fragment is low-content/fumble-like;
- the second ASR decode re-establishes the selected clip's trailing lexical anchor;
- the second decode contains a compact repeated-function-word collision after the anchor;
- one to three new content words occur between the anchor and that collision;
- those words land at the selected boundary (or just beyond it) in the actual source.

The selected clip's text/word metadata is repaired from the second decode. Its physical
end is extended only when the confirmed completion word actually ends after the current
boundary. Uncertain cases fail open unchanged.
"""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata
from typing import Iterable, Mapping

from .asr import ASRProvider
from .contracts import DraftClip, DraftTimeline, ProcessingResult, TranscriptSegment, Word
from . import final_draft_retry_integrity as retry_base
from .failed_prefix_completion_rescue import _collision_index

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "as", "at", "by", "como", "con", "de", "del", "el", "en",
    "for", "from", "in", "la", "las", "lo", "los", "of", "on", "or", "para", "por",
    "que", "the", "to", "un", "una", "with", "y",
})


def _norm(value: str) -> str:
    match = _TOKEN_RE.search(str(value or ""))
    if not match:
        return ""
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", match.group(0).casefold())
        if not unicodedata.combining(ch)
    )
    return text


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in (_norm(x) for x in _TOKEN_RE.findall(str(text or ""))) if len(token) >= 3 and token not in _STOP)


def _failed_semantics(diagnostics: dict) -> dict[str, float]:
    return retry_base._semantic_failures(diagnostics)


def _candidate_pairs(
    draft: DraftTimeline,
    *,
    maximum_gap_sec: float = 0.45,
    maximum_failed_duration_sec: float = 4.5,
) -> tuple[tuple[DraftClip, DraftClip], ...]:
    """Find only selected -> short failed-fragment boundaries worth re-listening."""
    failures = _failed_semantics(dict(draft.diagnostics or {}))
    selected = tuple(sorted(draft.selected, key=lambda c: (c.source_order, c.start, c.end, c.clip_id)))
    discarded = tuple(sorted(draft.discarded, key=lambda c: (c.source_order, c.start, c.end, c.clip_id)))
    out = []
    for kept in selected:
        options = []
        for failed in discarded:
            if failed.source_asset_id != kept.source_asset_id:
                continue
            if failures.get(failed.clip_id, 0.0) < 0.90:
                continue
            duration = max(0.0, float(failed.end) - float(failed.start))
            if duration <= 0.0 or duration > maximum_failed_duration_sec:
                continue
            gap = float(failed.start) - float(kept.end)
            if gap < -0.08 or gap > maximum_gap_sec:
                continue
            tokens = _TOKEN_RE.findall(str(failed.text or ""))
            content = _content_tokens(failed.text)
            # The re-listen path is for tiny garbled tails, not ordinary discarded ideas.
            if len(tokens) > 10 or len(content) > 2:
                continue
            options.append((abs(gap), failed.start, failed.end, failed))
        if options:
            out.append((kept, min(options, key=lambda item: item[:3])[3]))
    return tuple(out)


def _decoded_words(segments: Iterable[TranscriptSegment]) -> tuple[Word, ...]:
    return tuple(
        word
        for segment in segments
        for word in tuple(segment.words or ())
        if _norm(word.text)
    )


def _find_anchor_raw_index(selected_text: str, decoded_words: tuple[Word, ...], collision: int) -> int | None:
    selected_content = _content_tokens(selected_text)
    if len(selected_content) < 2:
        return None
    decoded_content = [
        (_norm(word.text), index)
        for index, word in enumerate(decoded_words[:collision])
        if len(_norm(word.text)) >= 3 and _norm(word.text) not in _STOP
    ]
    decoded_only = [token for token, _ in decoded_content]
    for width in range(min(3, len(selected_content)), 1, -1):
        needle = list(selected_content[-width:])
        for start in range(len(decoded_only) - width, -1, -1):
            if decoded_only[start : start + width] == needle:
                return decoded_content[start + width - 1][1]
    return None


def _shift_word(word: Word, offset: float) -> Word:
    return Word(
        text=str(word.text).strip(),
        start=float(word.start) + offset,
        end=float(word.end) + offset,
        confidence=word.confidence,
    )


def reconcile_audio_confirmed_completion(
    selected: DraftClip,
    failed: DraftClip,
    decoded_segments: Iterable[TranscriptSegment],
    *,
    window_start: float,
) -> tuple[DraftClip, dict | None]:
    """Pure reconciliation: recover only words proven by the second audio decode."""
    decoded = _decoded_words(decoded_segments)
    if len(decoded) < 6:
        return selected, None
    collision = _collision_index(decoded)
    if collision is None or collision < 3:
        return selected, None

    anchor_raw_index = _find_anchor_raw_index(selected.text, decoded, collision)
    if anchor_raw_index is None or anchor_raw_index >= collision - 1:
        return selected, None

    missing_raw = tuple(decoded[anchor_raw_index + 1 : collision])
    missing_content = tuple(
        word for word in missing_raw
        if len(_norm(word.text)) >= 3 and _norm(word.text) not in _STOP
    )
    if not (1 <= len(missing_content) <= 3) or len(missing_raw) > 4:
        return selected, None

    # The completion must be genuinely new relative to the current selected suffix.
    selected_content = set(_content_tokens(selected.text)[-5:])
    new_content = tuple(word for word in missing_content if _norm(word.text) not in selected_content)
    if not new_content:
        return selected, None

    shifted = tuple(_shift_word(word, window_start) for word in missing_raw)
    completion_start = min(word.start for word in shifted)
    completion_end = max(word.end for word in shifted)
    collision_start = float(decoded[collision].start) + window_start

    # Re-decoded anchor/completion must physically sit at the existing edit boundary.
    if completion_end < float(selected.end) - 1.0:
        return selected, None
    if completion_start > float(selected.end) + 0.75:
        return selected, None
    if completion_end > float(selected.end) + 1.20:
        return selected, None
    if collision_start < completion_end - 0.08:
        return selected, None
    if collision_start > float(failed.end) + 0.50:
        return selected, None

    completion_text = " ".join(str(word.text).strip() for word in missing_raw).strip()
    if not completion_text:
        return selected, None

    existing_words = tuple(selected.words or ())
    merged_words = tuple(sorted((*existing_words, *shifted), key=lambda w: (w.start, w.end, w.text)))
    repaired = replace(
        selected,
        end=max(float(selected.end), completion_end),
        text=f"{str(selected.text).rstrip()} {completion_text}".strip(),
        caption_text=f"{str(selected.caption_text).rstrip()} {completion_text}".strip(),
        words=merged_words,
    )
    return repaired, {
        "reason": "audio_confirmed_missing_completion_before_failed_collision",
        "selected_clip_id": selected.clip_id,
        "failed_clip_id": failed.clip_id,
        "completion_text": completion_text,
        "original_end": round(float(selected.end), 3),
        "repaired_end": round(float(repaired.end), 3),
        "collision_start": round(collision_start, 3),
        "window_start": round(window_start, 3),
    }


def _extract_audio_window(source_path: str, destination: str, *, start: float, end: float) -> None:
    duration = max(0.2, float(end) - float(start))
    ffmpeg = str(os.environ.get("FFMPEG_BIN") or "ffmpeg")
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{max(0.0, float(start)):.3f}", "-t", f"{duration:.3f}",
            "-i", source_path, "-vn", "-ac", "1", "-ar", "16000", destination,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def repair_audio_confirmed_boundary_completions(
    result: ProcessingResult,
    local_paths: Mapping[str, str],
    asr_provider: ASRProvider,
    *,
    language_hint: str | None = None,
) -> ProcessingResult:
    """Re-listen only suspicious failed boundaries and repair source-confirmed words."""
    draft = result.draft
    pairs = _candidate_pairs(draft)
    if not pairs:
        return result

    selected_map = {clip.clip_id: clip for clip in draft.selected}
    diagnostics = []
    checked_sources: set[tuple[str, str]] = set()

    with tempfile.TemporaryDirectory(prefix="cutsell-boundary-relisten-") as directory:
        for selected, failed in pairs:
            source_path = local_paths.get(selected.source_asset_id)
            if not source_path or not Path(source_path).exists():
                continue
            key = (selected.clip_id, failed.clip_id)
            if key in checked_sources:
                continue
            checked_sources.add(key)

            window_start = max(0.0, float(selected.end) - 2.6)
            window_end = min(float(failed.end) + 0.45, float(failed.end) + 0.45)
            if window_end - window_start < 1.0:
                continue
            audio_path = str(Path(directory) / f"{selected.clip_id}-{failed.clip_id}.wav")
            try:
                _extract_audio_window(source_path, audio_path, start=window_start, end=window_end)
                decoded = asr_provider.transcribe(
                    audio_path,
                    source_asset_id=selected.source_asset_id,
                    language_hint=language_hint,
                )
            except Exception:
                # Confirmation is optional evidence. Any decode/extraction failure fails open.
                continue

            current = selected_map.get(selected.clip_id, selected)
            repaired, audit = reconcile_audio_confirmed_completion(
                current,
                failed,
                decoded,
                window_start=window_start,
            )
            if audit is None:
                continue
            selected_map[selected.clip_id] = repaired
            diagnostics.append(audit)

    if not diagnostics:
        return result

    selected_out = tuple(selected_map.get(clip.clip_id, clip) for clip in draft.selected)
    draft_diagnostics = dict(draft.diagnostics or {})
    draft_diagnostics["audio_boundary_completion"] = diagnostics
    repaired_draft = replace(draft, selected=selected_out, diagnostics=draft_diagnostics)
    return replace(result, draft=repaired_draft)
