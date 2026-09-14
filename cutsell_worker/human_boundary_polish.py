"""Human-style post-selection boundary polish for Clean Cut.

This pass is intentionally late: semantic selection and Best Take happen first.  It only
changes physical edit boundaries when source evidence is strong enough to reproduce what
a careful human editor would do:

* do not hard-cut a sentence immediately before a tiny suffix that completes it;
* remove edge dead air when the source audio itself proves silence;
* remove interior dead air only when silence is long OR local face/body reset evidence
  corroborates that it is recording-process material;
* remove a repeated trailing word/phrase when the next short selected delivery clearly
  re-opens with the same words (e.g. ``cuídate`` -> ``por eso cuídate``).

No words are fabricated.  Uncertain cases fail open unchanged.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import re
import subprocess
from typing import Iterable, Mapping

from .contracts import DraftClip, DraftTimeline, ProcessingResult, Word
from .local_performance import LocalPerformanceTimeline

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_TERMINAL_RE = re.compile(r"[.!?…][\"')\]]*\s*$")
_RESET_KINDS = frozenset({
    "body_reset_candidate",
    "hand_motion_reset_candidate",
    "camera_disengagement_candidate",
    "facial_expression_shift_candidate",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _clip_id(parent: str, start: float, end: float, label: str) -> str:
    digest = hashlib.sha256(f"{parent}|{label}|{start:.3f}|{end:.3f}".encode()).hexdigest()[:12]
    return f"{parent}__hb{digest}"


def _words_in(words: Iterable[Word], start: float, end: float) -> tuple[Word, ...]:
    return tuple(word for word in words if float(word.end) > start and float(word.start) < end)


def _text_from_words(words: tuple[Word, ...], fallback: str) -> str:
    return " ".join(str(word.text).strip() for word in words).strip() or fallback


def _reset_near(timeline: LocalPerformanceTimeline | None, start: float, end: float, *, pad: float = 0.30) -> bool:
    if timeline is None:
        return False
    hits = [
        event for event in timeline.events
        if event.kind in _RESET_KINDS
        and event.confidence >= 0.78
        and event.end >= start - pad
        and event.start <= end + pad
    ]
    return len(hits) >= 2 or any(event.confidence >= 0.95 for event in hits)


def _silences(path: str, start: float, end: float, *, minimum_sec: float = 0.55) -> tuple[tuple[float, float], ...]:
    """Return source-absolute silence intervals using ffmpeg's audio detector."""
    duration = max(0.0, end - start)
    if duration <= minimum_sec:
        return ()
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", path, "-vn", "-af", f"silencedetect=noise=-43dB:d={minimum_sec:.3f}",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False, timeout=90)
    except Exception:
        return ()
    starts: list[float] = []
    intervals: list[tuple[float, float]] = []
    for line in str(proc.stderr or "").splitlines():
        match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if match:
            starts.append(start + float(match.group(1)))
            continue
        match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if match and starts:
            s = starts.pop(0)
            e = start + float(match.group(1))
            if e > s:
                intervals.append((s, e))
    if starts:
        for s in starts:
            intervals.append((s, end))
    return tuple(intervals)


def _trim_edge_silence(clip: DraftClip, silences: tuple[tuple[float, float], ...]) -> tuple[DraftClip, list[dict]]:
    start, end = float(clip.start), float(clip.end)
    diagnostics: list[dict] = []
    for s, e in silences:
        if s <= start + 0.18 and e - start >= 0.55 and e < end - 0.25:
            start = e
            diagnostics.append({"action": "trim_leading_dead_air", "start": round(s, 3), "end": round(e, 3)})
        if e >= end - 0.18 and end - s >= 0.55 and s > start + 0.25:
            end = s
            diagnostics.append({"action": "trim_trailing_dead_air", "start": round(s, 3), "end": round(e, 3)})
    if end - start < 0.25 or (start == clip.start and end == clip.end):
        return clip, []
    words = _words_in(clip.words, start, end)
    return replace(
        clip,
        start=start,
        end=end,
        words=words,
        text=_text_from_words(words, clip.text),
        caption_text=_text_from_words(words, clip.caption_text),
    ), diagnostics


def _split_interior_dead_air(
    clip: DraftClip,
    silences: tuple[tuple[float, float], ...],
    timeline: LocalPerformanceTimeline | None,
) -> tuple[tuple[DraftClip, ...], list[dict]]:
    pieces = [clip]
    diagnostics: list[dict] = []
    for s, e in silences:
        silence_len = e - s
        if silence_len < 0.85:
            continue
        if s <= clip.start + 0.25 or e >= clip.end - 0.25:
            continue
        corroborated = silence_len >= 1.35 or _reset_near(timeline, s, e)
        if not corroborated:
            continue
        # Never split through a word.  If ASR words overlap the silent interval, fail open.
        if any(float(word.end) > s and float(word.start) < e for word in clip.words):
            continue
        rebuilt: list[DraftClip] = []
        applied = False
        for piece in pieces:
            if not (piece.start < s and piece.end > e):
                rebuilt.append(piece)
                continue
            if s - piece.start < 0.30 or piece.end - e < 0.30:
                rebuilt.append(piece)
                continue
            left_words = _words_in(piece.words, piece.start, s)
            right_words = _words_in(piece.words, e, piece.end)
            left = replace(
                piece,
                clip_id=_clip_id(piece.clip_id, piece.start, s, "left"),
                end=s,
                words=left_words,
                text=_text_from_words(left_words, piece.text),
                caption_text=_text_from_words(left_words, piece.caption_text),
            )
            right = replace(
                piece,
                clip_id=_clip_id(piece.clip_id, e, piece.end, "right"),
                start=e,
                words=right_words,
                text=_text_from_words(right_words, piece.text),
                caption_text=_text_from_words(right_words, piece.caption_text),
            )
            rebuilt.extend((left, right))
            applied = True
        if applied:
            pieces = rebuilt
            diagnostics.append({
                "action": "remove_interior_dead_air",
                "start": round(s, 3), "end": round(e, 3),
                "duration_sec": round(silence_len, 3),
                "visual_reset_corroborated": _reset_near(timeline, s, e),
            })
    return tuple(pieces), diagnostics


def _bridge_short_completion(clips: list[DraftClip]) -> tuple[list[DraftClip], list[dict]]:
    """Avoid a hard visual cut immediately before a tiny sentence-completion suffix."""
    if len(clips) < 2:
        return clips, []
    out: list[DraftClip] = []
    diagnostics: list[dict] = []
    index = 0
    while index < len(clips):
        current = clips[index]
        if index + 1 >= len(clips):
            out.append(current); break
        nxt = clips[index + 1]
        gap = float(nxt.start) - float(current.end)
        nxt_tokens = _tokens(nxt.text)
        same_source = current.source_asset_id == nxt.source_asset_id
        current_incomplete = not _TERMINAL_RE.search(str(current.text or "").strip())
        tiny_completion = 1 <= len(nxt_tokens) <= 3 and nxt.end - nxt.start <= 1.20
        if same_source and current_incomplete and tiny_completion and 0.0 <= gap <= 1.15:
            # Preserve the source span continuously.  This intentionally prefers a short
            # natural pause over cutting the creator's sentence before its final word.
            words = tuple(sorted((*current.words, *nxt.words), key=lambda w: (w.start, w.end)))
            merged_text = f"{str(current.text).rstrip()} {str(nxt.text).lstrip()}".strip()
            out.append(replace(
                current,
                end=nxt.end,
                text=merged_text,
                caption_text=merged_text,
                words=words,
            ))
            diagnostics.append({
                "action": "bridge_sentence_completion",
                "left_clip_id": current.clip_id,
                "right_clip_id": nxt.clip_id,
                "source_gap_sec": round(max(0.0, gap), 3),
            })
            index += 2
            continue
        out.append(current)
        index += 1
    return out, diagnostics


def _dedupe_repeated_tail(clips: list[DraftClip]) -> tuple[list[DraftClip], list[dict]]:
    diagnostics: list[dict] = []
    if len(clips) < 2:
        return clips, diagnostics
    out = list(clips)
    for index in range(len(out) - 1):
        left, right = out[index], out[index + 1]
        if left.source_asset_id != right.source_asset_id or right.end - right.start > 8.0:
            continue
        lt = _tokens(left.text)
        rt = _tokens(right.text)
        if not lt or not rt:
            continue
        width = 0
        for candidate in (3, 2, 1):
            if len(lt) >= candidate and len(rt) >= candidate and tuple(lt[-candidate:]) == tuple(rt[:candidate]):
                width = candidate; break
            if len(lt) >= candidate and len(rt) >= candidate + 2 and tuple(lt[-candidate:]) == tuple(rt[2:2+candidate]):
                width = candidate; break
        if width == 0 or not left.words:
            continue
        normalized_words = [(_tokens(word.text)[0] if _tokens(word.text) else "", word) for word in left.words]
        tail = [token for token, _ in normalized_words if token]
        if len(tail) < width or tuple(tail[-width:]) != tuple(lt[-width:]):
            continue
        cut_word = [word for token, word in normalized_words if token][-width]
        new_end = float(cut_word.start)
        if new_end - left.start < 0.35:
            continue
        words = tuple(word for word in left.words if float(word.end) <= new_end + 1e-6)
        if not words:
            continue
        new_text = " ".join(str(word.text).strip() for word in words).strip()
        out[index] = replace(left, end=new_end, words=words, text=new_text, caption_text=new_text)
        diagnostics.append({
            "action": "remove_repeated_trailing_phrase",
            "left_clip_id": left.clip_id,
            "right_clip_id": right.clip_id,
            "token_count": width,
            "new_end": round(new_end, 3),
        })
    return out, diagnostics


def polish_human_boundaries(
    result: ProcessingResult,
    local_paths: Mapping[str, str],
    timelines: Iterable[LocalPerformanceTimeline],
) -> ProcessingResult:
    """Apply conservative source-evidenced boundary polish to selected draft clips."""
    timeline_by_source = {timeline.source_asset_id: timeline for timeline in timelines}
    selected = list(result.draft.selected)
    diagnostics: list[dict] = []

    # First preserve syntactic completion across tiny suffix fragments.  This runs before
    # dead-air removal intentionally: a sentence-final word is more important than a
    # sub-second cosmetic pause.
    selected, bridge_diag = _bridge_short_completion(selected)
    diagnostics.extend(bridge_diag)

    polished: list[DraftClip] = []
    for clip in selected:
        path = local_paths.get(clip.source_asset_id)
        if not path:
            polished.append(clip); continue
        silences = _silences(path, float(clip.start), float(clip.end))
        edge_clip, edge_diag = _trim_edge_silence(clip, silences)
        diagnostics.extend({"clip_id": clip.clip_id, **row} for row in edge_diag)
        pieces, interior_diag = _split_interior_dead_air(
            edge_clip, silences, timeline_by_source.get(clip.source_asset_id)
        )
        diagnostics.extend({"clip_id": clip.clip_id, **row} for row in interior_diag)
        polished.extend(pieces)

    polished, dedupe_diag = _dedupe_repeated_tail(polished)
    diagnostics.extend(dedupe_diag)
    if not diagnostics:
        return result

    draft_diag = dict(result.draft.diagnostics or {})
    draft_diag["human_boundary_polish"] = diagnostics[:300]
    draft = replace(result.draft, selected=tuple(polished), diagnostics=draft_diag)
    return replace(result, draft=draft)
