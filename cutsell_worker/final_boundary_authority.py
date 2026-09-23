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
- fail open: when transcript evidence is ambiguous, retain more speech, never less;
- D-289.11: when a later selected delivery RE-OPENS with the closing words of a
  preceding complete selected delivery (e.g. ``... Así que cuídate.`` then
  ``Por eso cuídate, aliméntate bien ...``), trim only that re-opened closing from
  the later clip at the start of its first remaining word -- bounded by recency,
  by ASR punctuation/pause structure at the removed phrase, by a remaining-content
  floor, and refused outright when the removed words carry a number, a negation or
  a distinct-addition marker. The earlier complete delivery is never touched. This
  runs BEFORE Selection Freeze because it changes the token stream; nothing after
  Freeze may do that (``enforce_selection_contract`` verifies).

This is intentionally not a semantic composer and never changes take selection.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Mapping

from .asr import ASRProvider
from .contracts import DraftClip, ProcessingResult, Word
from .take_grouping import _DANGLING_FUNCTION_WORDS, _DISTINCT_ADDITION_MARKERS

_TERMINAL = (".", "?", "!", "…")

# --- D-289.11: re-opened closing restatement (general, no Video00 content) ---
_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
# Leading DISCOURSE connectives a speaker uses to re-open a sentence with the
# words just closed on ("por eso", "así que", "entonces", "so", "and then").
# Deliberately NOT articles/determiners: "... se mandó a biopsia." followed by
# "La biopsia confirmó ..." is a noun re-mention carrying the story forward,
# never a re-opened closing -- a determiner before the repeated word is a
# refusal, not a skip.
_REOPEN_DISCOURSE_CONNECTIVES = frozenset({
    "por", "eso", "así", "asi", "que", "entonces", "pues", "bueno", "y", "e", "o",
    "sea", "ahora", "también", "tambien", "además", "ademas",
    "so", "and", "then", "therefore", "hence", "well", "okay", "ok", "now", "also",
})
# Tokens that never count as content (connectives + the grouping authority's
# dangling function words) -- for the content floor and the lone-word check.
_REOPEN_CONNECTIVE_TOKENS = _REOPEN_DISCOURSE_CONNECTIVES | _DANGLING_FUNCTION_WORDS | frozenset({"bien"})
# Mirrors contradiction_signal._NEGATION_MARKERS (not imported: that module
# pulls the grouping provider graph; the boundary authority stays leaf-level).
_REOPEN_NEGATION_TOKENS = frozenset({"no", "not", "never", "nunca", "sin", "without", "nadie", "ni"})
_REOPEN_MAX_LEADING_CONNECTIVES = 2
_REOPEN_MAX_PHRASE_TOKENS = 6
_REOPEN_MIN_REMAINING_CONTENT_TOKENS = 2
_REOPEN_MAX_INTERVENING_CLIPS = 2
_REOPEN_MAX_INTERVENING_SEC = 10.0
_REOPEN_PHRASE_BREAK_PUNCT = (",", ";", ":", ".", "!", "?", "…")
_REOPEN_PHRASE_BREAK_PAUSE_SEC = 0.25
_REOPEN_MIN_REMAINING_SEC = 0.5


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


def _same_semantic_parent(left: DraftClip, right: DraftClip) -> bool:
    left_key = getattr(left, "parent_semantic_clip_id", None) or left.clip_id
    right_key = getattr(right, "parent_semantic_clip_id", None) or right.clip_id
    return left_key == right_key


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

        # D-036/D-046 provenance: physical siblings of one frozen delivery
        # may carry fragment ids (post_selection_interior_gap_trim's
        # `__psig*` pieces) instead of the parent's clip_id -- the shared
        # key is parent_semantic_clip_id. D-095.2: without this, an
        # audio-silence split could be re-filled by the envelope expansion.
        if _same_semantic_parent(left_orig, right_orig) and original_gap > 0.02:
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


def _word_token(word: Word) -> str:
    found = _TOKEN_RE.findall(str(word.text or "").casefold())
    return found[0] if found else ""


def _tokenized_words(words: tuple[Word, ...]) -> list[tuple[str, Word]]:
    return [(token, word) for token, word in ((_word_token(word), word) for word in words) if token]


def _reopened_closing_match(
    left_words: tuple[Word, ...],
    right_words: tuple[Word, ...],
) -> tuple[int, int] | None:
    """Return ``(skip, width)`` when the right clip re-opens -- after at most
    ``skip`` leading connective tokens -- with the last ``width`` (1..3)
    tokens of the left clip's terminal-punctuated closing; else None. Longer
    matches are tried first: more repeated words is more evidence, never less."""
    left = _tokenized_words(left_words)
    right = _tokenized_words(right_words)
    if not left or not right or not _terminal(left[-1][1]):
        return None
    left_tokens = [token for token, _ in left]
    right_tokens = [token for token, _ in right]
    for skip in range(0, _REOPEN_MAX_LEADING_CONNECTIVES + 1):
        if skip and any(token not in _REOPEN_DISCOURSE_CONNECTIVES for token in right_tokens[:skip]):
            break
        for width in range(_REOPEN_MAX_PHRASE_TOKENS, 0, -1):
            if len(left_tokens) < width or len(right_tokens) < skip + width:
                continue
            if left_tokens[-width:] != right_tokens[skip:skip + width]:
                continue
            if width == 1 and left_tokens[-1] in _REOPEN_CONNECTIVE_TOKENS:
                continue  # a lone function word is not a closing phrase
            return skip, width
    return None


def _reopened_closing_refusal(right_words: tuple[Word, ...], skip: int, width: int) -> str | None:
    """Why a matched re-opened closing must NOT be trimmed (fail open)."""
    right = _tokenized_words(right_words)
    removed = right[:skip + width]
    remaining = right[skip + width:]
    removed_tokens = [token for token, _ in removed]
    removed_text = " ".join(str(word.text) for _, word in removed)
    if any(any(ch.isdigit() for ch in token) for token in removed_tokens):
        return "removed_prefix_carries_number"
    if any(token in _REOPEN_NEGATION_TOKENS for token in removed_tokens):
        return "removed_prefix_carries_negation"
    lowered = f" {removed_text.casefold()} "
    if any(marker in lowered for marker in _DISTINCT_ADDITION_MARKERS):
        return "removed_prefix_carries_distinct_addition_marker"
    if not remaining:
        return "nothing_remains_after_repeated_closing"
    last_removed = removed[-1][1]
    first_remaining = remaining[0][1]
    punct_break = str(last_removed.text or "").rstrip().endswith(_REOPEN_PHRASE_BREAK_PUNCT)
    pause_break = float(first_remaining.start) - float(last_removed.end) >= _REOPEN_PHRASE_BREAK_PAUSE_SEC
    if not (punct_break or pause_break):
        return "repeated_closing_not_a_separate_phrase"
    if remaining[0][0] in _DANGLING_FUNCTION_WORDS:
        return "remaining_delivery_would_open_on_dangling_word"
    content_left = sum(1 for token, _ in remaining if token not in _REOPEN_CONNECTIVE_TOKENS)
    if content_left < _REOPEN_MIN_REMAINING_CONTENT_TOKENS:
        return "remaining_delivery_below_content_floor"
    if float(remaining[-1][1].end) - float(first_remaining.start) < _REOPEN_MIN_REMAINING_SEC:
        return "remaining_delivery_too_short"
    return None


def _trim_reopened_closings(
    selected: list[DraftClip],
    source_map: dict[str, tuple[Word, ...]],
) -> tuple[list[DraftClip], list[dict]]:
    """D-289.11: trim a later selected delivery's re-opened closing phrase.

    For each selected clip R, look back over at most
    ``_REOPEN_MAX_INTERVENING_CLIPS`` earlier selected clips spanning at most
    ``_REOPEN_MAX_INTERVENING_SEC`` of output time for a same-source, earlier,
    non-overlapping complete delivery L whose closing 1..3 tokens R re-opens
    with (1..6 tokens, after <= 2 leading connectives). The trim starts R at the start of
    its first remaining source word (never inside a word); L is never edited;
    every refusal is recorded. One trim per R at most."""
    output = list(selected)
    rows: list[dict] = []
    for index in range(1, len(output)):
        right = output[index]
        right_words = tuple(right.words)
        if not right_words:
            continue
        intervening_sec = 0.0
        for back in range(1, _REOPEN_MAX_INTERVENING_CLIPS + 2):
            left_index = index - back
            if left_index < 0:
                break
            if back > 1:
                mid = output[index - back + 1]
                intervening_sec += max(0.0, float(mid.end) - float(mid.start))
                if intervening_sec > _REOPEN_MAX_INTERVENING_SEC:
                    break
            left = output[left_index]
            if left.source_asset_id != right.source_asset_id or not left.words:
                continue
            if float(left.end) > float(right.start) + 1e-6:
                continue
            match = _reopened_closing_match(tuple(left.words), right_words)
            if match is None:
                continue
            skip, width = match
            tokenized = _tokenized_words(right_words)
            base_row = {
                "left_clip_id": left.clip_id,
                "right_clip_id": right.clip_id,
                "intervening_clip_count": back - 1,
                "intervening_sec": round(intervening_sec, 3),
                "repeated_tokens": [token for token, _ in tokenized[skip:skip + width]],
                "removed_leading_tokens": [token for token, _ in tokenized[:skip + width]],
            }
            refusal = _reopened_closing_refusal(right_words, skip, width)
            if refusal is not None:
                rows.append({"action": "keep_reopened_closing", "reason": refusal, **base_row})
                break
            first_remaining = tokenized[skip + width][1]
            new_start = float(first_remaining.start)
            if new_start <= float(right.start) + 1e-6 or new_start >= float(right.end) - 1e-6:
                rows.append({"action": "keep_reopened_closing", "reason": "cut_point_outside_clip", **base_row})
                break
            source_words = source_map.get(right.source_asset_id) or right_words
            rebuilt = _rebuild_clip(right, source_words, new_start, float(right.end))
            expected = [token for token, _ in tokenized[skip + width:]]
            if [token for token, _ in _tokenized_words(tuple(rebuilt.words))] != expected:
                rows.append({"action": "keep_reopened_closing", "reason": "rebuilt_words_diverge_from_expected", **base_row})
                break
            output[index] = rebuilt
            rows.append({
                "action": "trim_reopened_closing_restatement",
                "original_start": round(float(right.start), 3),
                "result_start": round(float(rebuilt.start), 3),
                "removed_sec": round(float(rebuilt.start) - float(right.start), 3),
                "first_remaining_word": str(first_remaining.text),
                **base_row,
            })
            break
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

    # D-289.11: after the envelopes are complete and overlap-free, trim a
    # later delivery's re-opened closing phrase (pre-Freeze token change).
    selected, reopen_rows = _trim_reopened_closings(selected, source_map)
    diagnostics.extend(reopen_rows)
    reopen_trims = [row for row in reopen_rows if row.get("action") == "trim_reopened_closing_restatement"]

    preserved_gap_rows = [row for row in overlap_rows if row.get("action") == "preserve_polished_interior_gap"]
    diag = dict(result.draft.diagnostics or {})
    diag["final_boundary_authority"] = diagnostics[:600]
    diag["final_boundary_authority_rule"] = (
        "full_source_transcript -> complete idea envelope -> complete word lock -> "
        "preserve proven polished interior gaps -> neighbor-original-span overlap guard -> "
        "re-opened closing restatement trim (D-289.11, word-start cut, fail open) -> visual slack only"
    )
    diag["final_boundary_overlap_reconciliation_count"] = len(overlap_rows)
    diag["final_boundary_preserved_polish_gap_count"] = len(preserved_gap_rows)
    diag["final_boundary_reopened_closing_trim_count"] = len(reopen_trims)
    diag["final_boundary_reopened_closing_refusal_count"] = len(reopen_rows) - len(reopen_trims)
    draft = replace(result.draft, selected=tuple(selected), diagnostics=diag)
    return replace(result, draft=draft)
