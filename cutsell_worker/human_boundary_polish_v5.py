"""Human Watch+Listen micro-polish v5.

This pass handles the final sub-second recording-process gaps that can remain after v4.
It is intentionally conservative: only gaps between source-aligned words are eligible,
and short gaps require strong stored face/body reset evidence. No benchmark timestamps,
phrases, or source names are hardcoded. Ambiguous cases fail open unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .contracts import DraftClip, ProcessingResult, Word
from .human_boundary_polish_v2 import _timeline_proxies, _reset_score
from .human_boundary_polish_v4 import polish_human_boundaries_v4


def _words_before(words: tuple[Word, ...], end: float) -> tuple[Word, ...]:
    return tuple(word for word in words if float(word.end) <= end + 1e-6)


def _words_after(words: tuple[Word, ...], start: float) -> tuple[Word, ...]:
    return tuple(word for word in words if float(word.start) >= start - 1e-6)


def _clip_from_words(clip: DraftClip, start: float, end: float, words: tuple[Word, ...]) -> DraftClip | None:
    if end - start < 0.18 or not words:
        return None
    text = " ".join(str(word.text).strip() for word in words).strip()
    if not text:
        return None
    return replace(clip, start=start, end=end, words=words, text=text, caption_text=text)


def _micro_reset_evidence(timeline, start: float, end: float) -> tuple[float, int]:
    """Return aggregate reset score and number of strong visual reset events."""
    if timeline is None:
        return 0.0, 0
    score = _reset_score(timeline, start, end, pad=0.28)
    strong = 0
    for event in getattr(timeline, "events", ()):
        if event.end < start - 0.28 or event.start > end + 0.28:
            continue
        if event.kind not in {
            "body_reset_candidate",
            "hand_motion_reset_candidate",
            "camera_disengagement_candidate",
            "facial_expression_shift_candidate",
        }:
            continue
        if float(event.confidence) >= 0.82:
            strong += 1
    return score, strong


def _remove_micro_visual_reset_word_gaps(
    clip: DraftClip,
    timeline,
) -> tuple[tuple[DraftClip, ...], list[dict]]:
    words = tuple(sorted(tuple(clip.words), key=lambda word: (word.start, word.end)))
    if len(words) < 2 or timeline is None:
        return (clip,), []

    candidates: list[tuple[float, float, float, int]] = []
    for left, right in zip(words, words[1:]):
        gap_start = float(left.end)
        gap_end = float(right.start)
        gap = gap_end - gap_start

        # v4 already owns >=0.55s word gaps. v5 is only the final micro-polish zone.
        if gap < 0.22 or gap >= 0.55:
            continue

        score, strong = _micro_reset_evidence(timeline, gap_start, gap_end)
        # Very short gaps require multiple corroborating visual events or a very high
        # aggregate reset score. Slightly longer micro-gaps may pass with one strong
        # face/body reset plus high aggregate evidence.
        if gap < 0.34:
            eligible = score >= 1.55 or (score >= 1.30 and strong >= 2)
        else:
            eligible = score >= 1.20 and strong >= 1
        if not eligible:
            continue

        candidates.append((gap_start, gap_end, score, strong))

    if not candidates:
        return (clip,), []

    pieces: list[DraftClip] = [clip]
    diagnostics: list[dict] = []
    for gap_start, gap_end, score, strong in candidates:
        rebuilt: list[DraftClip] = []
        changed = False
        for piece in pieces:
            if not (float(piece.start) < gap_start and float(piece.end) > gap_end):
                rebuilt.append(piece)
                continue
            left_words = _words_before(tuple(piece.words), gap_start)
            right_words = _words_after(tuple(piece.words), gap_end)
            left_piece = _clip_from_words(piece, float(piece.start), gap_start, left_words)
            right_piece = _clip_from_words(piece, gap_end, float(piece.end), right_words)
            if left_piece is None or right_piece is None:
                rebuilt.append(piece)
                continue
            rebuilt.extend((left_piece, right_piece))
            changed = True
        if changed:
            pieces = rebuilt
            diagnostics.append({
                "action": "remove_micro_visual_reset_word_gap",
                "start": round(gap_start, 3),
                "end": round(gap_end, 3),
                "duration_sec": round(gap_end - gap_start, 3),
                "reset_score": round(score, 3),
                "strong_reset_events": strong,
            })

    return tuple(pieces), diagnostics


def polish_human_boundaries_v5(result: ProcessingResult, local_paths: Mapping[str, str]) -> ProcessingResult:
    result = polish_human_boundaries_v4(result, local_paths)
    if not hasattr(result.draft, "selected"):
        return result

    timelines = _timeline_proxies(result)
    polished: list[DraftClip] = []
    diagnostics: list[dict] = []

    for clip in result.draft.selected:
        pieces, rows = _remove_micro_visual_reset_word_gaps(clip, timelines.get(clip.source_asset_id))
        polished.extend(pieces)
        diagnostics.extend({"clip_id": clip.clip_id, **row} for row in rows)

    if not diagnostics:
        return result

    existing = list((result.draft.diagnostics or {}).get("human_boundary_polish") or ())
    diag = dict(result.draft.diagnostics or {})
    diag["human_boundary_polish"] = [*existing, *diagnostics][:600]
    draft = replace(result.draft, selected=tuple(polished), diagnostics=diag)
    return replace(result, draft=draft)
