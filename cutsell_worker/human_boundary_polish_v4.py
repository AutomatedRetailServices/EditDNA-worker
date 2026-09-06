"""Human Watch+Listen authority v4.

This pass fixes two remaining human-review failures without hardcoded benchmark times:
1) a selected delivery can be an inferior/partial take while a nearby later alternate
   contains the same idea more completely and with equal-or-better performance evidence;
2) a visible face/body reset can sit inside a selected take between spoken words even
   when amplitude-based silence detection is too conservative to remove it.

Selection and timing remain source-evidenced. No words are fabricated. Ambiguous cases
fail open unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .contracts import DraftClip, ProcessingResult, Word
from .human_boundary_polish_v2 import _timeline_proxies, _reset_score
from .human_boundary_polish_v3 import polish_human_boundaries_v3

_STOP = frozenset({
    "a", "al", "and", "as", "at", "by", "con", "de", "del", "el", "en", "for",
    "from", "in", "la", "las", "lo", "los", "of", "on", "or", "para", "por", "que",
    "the", "to", "un", "una", "with", "y",
})


def _tokens(text: str) -> tuple[str, ...]:
    import re
    return tuple(token.casefold() for token in re.findall(r"[a-z0-9áéíóúñü]+", str(text or ""), re.IGNORECASE))


def _content(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) >= 3 and token not in _STOP}


def _reset_density(timeline, clip: DraftClip) -> float:
    if timeline is None:
        return 0.0
    duration = max(0.40, float(clip.end) - float(clip.start))
    score = 0.0
    for event in getattr(timeline, "events", ()):
        if event.end <= clip.start or event.start >= clip.end:
            continue
        if event.confidence < 0.72:
            continue
        if event.kind not in {
            "body_reset_candidate",
            "hand_motion_reset_candidate",
            "camera_disengagement_candidate",
            "facial_expression_shift_candidate",
        }:
            continue
        score += float(event.confidence)
    return score / duration


def _promote_more_complete_later_retry(
    result: ProcessingResult,
    timelines: Mapping[str, object],
) -> tuple[ProcessingResult, list[dict]]:
    """Promote a nearby later alternate when it clearly subsumes the selected take.

    Guardrails:
    - same source and nearby in recording order;
    - most selected content appears in the alternate;
    - alternate adds meaningful content / duration;
    - alternate is not visually worse by reset density;
    - no overlapping selected clip already occupies the alternate span.
    """
    selected = list(result.draft.selected)
    alternates = list(result.draft.alternates)
    if not selected or not alternates:
        return result, []

    promoted_alt_ids: set[str] = set()
    demoted: list[DraftClip] = []
    diagnostics: list[dict] = []

    for i, current in enumerate(list(selected)):
        cur_content = _content(current.text)
        if len(cur_content) < 3:
            continue
        timeline = timelines.get(current.source_asset_id)
        current_density = _reset_density(timeline, current)
        candidates: list[tuple[float, float, float, DraftClip]] = []

        for alt in alternates:
            if alt.clip_id in promoted_alt_ids or alt.source_asset_id != current.source_asset_id:
                continue
            # Human retakes usually follow shortly after the failed/partial delivery.
            source_gap = float(alt.start) - float(current.end)
            if source_gap < -0.30 or source_gap > 18.0:
                continue
            alt_content = _content(alt.text)
            if len(alt_content) < len(cur_content) + 2:
                continue
            shared = len(cur_content & alt_content)
            selected_coverage = shared / max(1, len(cur_content))
            if shared < 3 or selected_coverage < 0.72:
                continue
            duration_gain = (float(alt.end) - float(alt.start)) - (float(current.end) - float(current.start))
            content_gain = len(alt_content - cur_content)
            if duration_gain < 0.60 and content_gain < 3:
                continue
            # Do not create overlapping selected timeline material.
            if any(
                j != i
                and peer.source_asset_id == alt.source_asset_id
                and float(peer.end) > float(alt.start) + 0.05
                and float(peer.start) < float(alt.end) - 0.05
                for j, peer in enumerate(selected)
            ):
                continue
            alt_density = _reset_density(timeline, alt)
            if alt_density > current_density + 0.08:
                continue
            candidates.append((
                -selected_coverage,
                -(content_gain + max(0.0, duration_gain)),
                alt_density,
                alt,
            ))

        if not candidates:
            continue
        _, _, alt_density, winner = min(candidates, key=lambda row: row[:3])
        selected[i] = replace(winner, selected=True)
        demoted.append(replace(current, selected=False))
        promoted_alt_ids.add(winner.clip_id)
        diagnostics.append({
            "action": "promote_more_complete_later_retry",
            "removed_clip_id": current.clip_id,
            "winner_clip_id": winner.clip_id,
            "selected_content_tokens": len(cur_content),
            "winner_content_tokens": len(_content(winner.text)),
            "removed_reset_density": round(current_density, 3),
            "winner_reset_density": round(alt_density, 3),
        })

    if not diagnostics:
        return result, diagnostics

    selected.sort(key=lambda clip: (clip.source_order, float(clip.start), float(clip.end)))
    new_alternates = [alt for alt in alternates if alt.clip_id not in promoted_alt_ids]
    new_alternates.extend(demoted)
    draft = replace(
        result.draft,
        selected=tuple(selected),
        alternates=tuple(new_alternates),
    )
    return replace(result, draft=draft), diagnostics


def _words_before(words: tuple[Word, ...], end: float) -> tuple[Word, ...]:
    return tuple(word for word in words if float(word.end) <= end + 1e-6)


def _words_after(words: tuple[Word, ...], start: float) -> tuple[Word, ...]:
    return tuple(word for word in words if float(word.start) >= start - 1e-6)


def _clip_from_words(clip: DraftClip, start: float, end: float, words: tuple[Word, ...]) -> DraftClip | None:
    if end - start < 0.22 or not words:
        return None
    text = " ".join(str(word.text).strip() for word in words).strip()
    if not text:
        return None
    return replace(clip, start=start, end=end, words=words, text=text, caption_text=text)


def _remove_visual_reset_word_gaps(
    clip: DraftClip,
    timeline,
) -> tuple[tuple[DraftClip, ...], list[dict]]:
    """Remove source gaps between words when local visual-reset evidence corroborates them."""
    words = tuple(sorted(tuple(clip.words), key=lambda word: (word.start, word.end)))
    if len(words) < 2 or timeline is None:
        return (clip,), []

    candidates: list[tuple[float, float, float]] = []
    for left, right in zip(words, words[1:]):
        gap_start = float(left.end)
        gap_end = float(right.start)
        gap = gap_end - gap_start
        if gap < 0.55:
            continue
        reset = _reset_score(timeline, gap_start, gap_end, pad=0.35)
        # A 0.55-1.2s pause needs strong reset evidence. Longer gaps can use a slightly
        # lower threshold but still require visible recording-process evidence.
        threshold = 0.92 if gap < 1.20 else 0.70
        if reset < threshold:
            continue
        candidates.append((gap_start, gap_end, reset))

    if not candidates:
        return (clip,), []

    pieces: list[DraftClip] = [clip]
    diagnostics: list[dict] = []
    for gap_start, gap_end, reset in candidates:
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
                "action": "remove_visual_reset_word_gap",
                "start": round(gap_start, 3),
                "end": round(gap_end, 3),
                "duration_sec": round(gap_end - gap_start, 3),
                "reset_score": round(reset, 3),
            })
    return tuple(pieces), diagnostics


def polish_human_boundaries_v4(result: ProcessingResult, local_paths: Mapping[str, str]) -> ProcessingResult:
    result = polish_human_boundaries_v3(result, local_paths)
    if not hasattr(result.draft, "selected"):
        return result

    diagnostics: list[dict] = []
    timelines = _timeline_proxies(result)

    result, rows = _promote_more_complete_later_retry(result, timelines)
    diagnostics.extend(rows)

    # Rebuild timeline proxies in case v3/v4 changed draft diagnostics or selection.
    timelines = _timeline_proxies(result)
    polished: list[DraftClip] = []
    for clip in result.draft.selected:
        pieces, rows = _remove_visual_reset_word_gaps(clip, timelines.get(clip.source_asset_id))
        polished.extend(pieces)
        diagnostics.extend({"clip_id": clip.clip_id, **row} for row in rows)

    if not diagnostics:
        return result

    existing = list((result.draft.diagnostics or {}).get("human_boundary_polish") or ())
    diag = dict(result.draft.diagnostics or {})
    diag["human_boundary_polish"] = [*existing, *diagnostics][:500]
    draft = replace(result.draft, selected=tuple(polished), diagnostics=diag)
    return replace(result, draft=draft)
