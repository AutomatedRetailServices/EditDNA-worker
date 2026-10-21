"""Human boundary polish v2 using already-stored multimodal evidence.

This wrapper fixes three practical gaps found during human Watch+Listen:
- a tiny completion fragment can live in alternates, so selected-only bridging misses it;
- the late universal wrapper no longer has the in-memory local timeline, but the merged
  local performance events are already serialized into draft diagnostics;
- short real pauses need visual reset corroboration instead of requiring >=0.85s silence.

No timestamps, source names, or benchmark phrases are hardcoded. Uncertain cases fail open.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import re
import subprocess
from typing import Iterable, Mapping

from .contracts import DraftClip, ProcessingResult, Word
from .human_boundary_polish import polish_human_boundaries

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_TERMINAL_RE = re.compile(r"[.!?…][\"')\]]*\s*$")
_RESET_KINDS = frozenset({
    "body_reset_candidate",
    "hand_motion_reset_candidate",
    "camera_disengagement_candidate",
    "facial_expression_shift_candidate",
})
_STOP = frozenset({
    "a", "al", "and", "as", "at", "by", "con", "de", "del", "el", "en", "for",
    "from", "in", "la", "las", "lo", "los", "of", "on", "or", "para", "por", "que",
    "the", "to", "un", "una", "with", "y",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _stem(token: str) -> str:
    token = token.casefold()
    if len(token) > 5 and token.endswith("es"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _content(text: str) -> set[str]:
    return {_stem(t) for t in _tokens(text) if len(t) >= 3 and t not in _STOP}


def _timeline_proxies(result: ProcessingResult) -> dict[str, SimpleNamespace]:
    out: dict[str, SimpleNamespace] = {}
    whole = dict(result.draft.diagnostics or {}).get("whole_video_context") or {}
    for source in whole.get("sources") or ():
        source_id = str(source.get("source_asset_id") or "")
        if not source_id:
            continue
        events = []
        for row in source.get("events") or ():
            try:
                events.append(SimpleNamespace(
                    source_asset_id=source_id,
                    start=float(row.get("start", 0.0)),
                    end=float(row.get("end", 0.0)),
                    kind=str(row.get("kind") or ""),
                    confidence=float(row.get("confidence", 0.0)),
                    description=str(row.get("description") or ""),
                ))
            except Exception:
                continue
        out[source_id] = SimpleNamespace(source_asset_id=source_id, events=tuple(events))
    return out


def _reset_score(timeline, start: float, end: float, *, pad: float = 0.45) -> float:
    if timeline is None:
        return 0.0
    score = 0.0
    for event in timeline.events:
        if event.kind not in _RESET_KINDS:
            continue
        if event.end < start - pad or event.start > end + pad:
            continue
        if event.confidence < 0.72:
            continue
        score += max(0.0, min(1.0, float(event.confidence)))
    return score


def _restore_tiny_completion_alternates(result: ProcessingResult) -> tuple[ProcessingResult, list[dict]]:
    selected = list(result.draft.selected)
    alternates = list(result.draft.alternates)
    if not selected or not alternates:
        return result, []
    used: set[str] = set()
    diagnostics: list[dict] = []
    for index, left in enumerate(selected):
        if _TERMINAL_RE.search(str(left.text or "").strip()):
            continue
        candidates = []
        for alt in alternates:
            if alt.clip_id in used or alt.source_asset_id != left.source_asset_id:
                continue
            gap = float(alt.start) - float(left.end)
            tokens = _tokens(alt.text)
            duration = float(alt.end) - float(alt.start)
            if 0.0 <= gap <= 1.35 and 1 <= len(tokens) <= 3 and duration <= 1.35:
                candidates.append((gap, duration, alt))
        if not candidates:
            continue
        _, _, alt = min(candidates, key=lambda item: (item[0], item[1]))
        words = tuple(sorted((*left.words, *alt.words), key=lambda w: (w.start, w.end)))
        merged_text = f"{str(left.text).rstrip()} {str(alt.text).lstrip()}".strip()
        selected[index] = replace(left, end=alt.end, text=merged_text, caption_text=merged_text, words=words)
        used.add(alt.clip_id)
        diagnostics.append({
            "action": "restore_tiny_sentence_completion_from_alternate",
            "left_clip_id": left.clip_id,
            "alternate_clip_id": alt.clip_id,
            "source_gap_sec": round(max(0.0, float(alt.start) - float(left.end)), 3),
        })
    if not used:
        return result, diagnostics
    draft_diag = dict(result.draft.diagnostics or {})
    draft = replace(
        result.draft,
        selected=tuple(selected),
        alternates=tuple(alt for alt in alternates if alt.clip_id not in used),
        diagnostics=draft_diag,
    )
    return replace(result, draft=draft), diagnostics


def _silences(path: str, start: float, end: float, *, minimum_sec: float = 0.32) -> tuple[tuple[float, float], ...]:
    duration = max(0.0, end - start)
    if duration <= minimum_sec:
        return ()
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", path, "-vn", "-af", f"silencedetect=noise=-40dB:d={minimum_sec:.3f}",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False, timeout=90)
    except Exception:
        return ()
    starts: list[float] = []
    intervals: list[tuple[float, float]] = []
    for line in str(proc.stderr or "").splitlines():
        m = re.search(r"silence_start:\s*([0-9.]+)", line)
        if m:
            starts.append(start + float(m.group(1)))
            continue
        m = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m and starts:
            s = starts.pop(0)
            e = start + float(m.group(1))
            if e > s:
                intervals.append((s, e))
    for s in starts:
        intervals.append((s, end))
    return tuple(intervals)


def _words_in(words: Iterable[Word], start: float, end: float) -> tuple[Word, ...]:
    return tuple(word for word in words if float(word.end) > start and float(word.start) < end)


def _split_short_reset_dead_air(clip: DraftClip, path: str, timeline) -> tuple[tuple[DraftClip, ...], list[dict]]:
    silences = _silences(path, float(clip.start), float(clip.end))
    pieces: list[DraftClip] = [clip]
    diagnostics: list[dict] = []
    for s, e in silences:
        span = e - s
        if span < 0.32 or s <= clip.start + 0.20 or e >= clip.end - 0.20:
            continue
        reset_score = _reset_score(timeline, s, e)
        if span < 1.30 and reset_score < 0.95:
            continue
        # If word timing explicitly claims speech inside the measured silence, fail open.
        if any(float(w.end) > s and float(w.start) < e for w in clip.words):
            continue
        rebuilt: list[DraftClip] = []
        applied = False
        for piece in pieces:
            if not (piece.start < s and piece.end > e):
                rebuilt.append(piece)
                continue
            if s - piece.start < 0.25 or piece.end - e < 0.25:
                rebuilt.append(piece)
                continue
            left_words = _words_in(piece.words, piece.start, s)
            right_words = _words_in(piece.words, e, piece.end)
            left_text = " ".join(w.text.strip() for w in left_words).strip() or piece.text
            right_text = " ".join(w.text.strip() for w in right_words).strip() or piece.text
            left = replace(piece, end=s, words=left_words, text=left_text, caption_text=left_text)
            right = replace(piece, start=e, words=right_words, text=right_text, caption_text=right_text)
            rebuilt.extend((left, right))
            applied = True
        if applied:
            pieces = rebuilt
            diagnostics.append({
                "action": "remove_short_dead_air_with_visual_reset",
                "start": round(s, 3),
                "end": round(e, 3),
                "duration_sec": round(span, 3),
                "reset_score": round(reset_score, 3),
            })
    return tuple(pieces), diagnostics


def _event_density(timeline, clip: DraftClip) -> float:
    if timeline is None:
        return 0.0
    duration = max(0.4, float(clip.end) - float(clip.start))
    score = 0.0
    for event in timeline.events:
        if event.kind not in _RESET_KINDS or event.confidence < 0.72:
            continue
        if event.end <= clip.start or event.start >= clip.end:
            continue
        score += float(event.confidence)
    return score / duration


def _collapse_redundant_retry_pair(result: ProcessingResult, timelines: Mapping[str, object]) -> tuple[ProcessingResult, list[dict]]:
    """Drop one nearby duplicate delivery only when visual-performance quality clearly wins.

    The shorter idea must be substantially covered by its peer and the local reset-event
    density must differ enough to justify a human-style Best Take decision.
    """
    clips = list(result.draft.selected)
    if len(clips) < 2:
        return result, []
    removed: set[int] = set()
    diagnostics: list[dict] = []
    for i in range(len(clips) - 1):
        if i in removed:
            continue
        a = clips[i]
        for j in range(i + 1, min(len(clips), i + 3)):
            if j in removed:
                continue
            b = clips[j]
            if a.source_asset_id != b.source_asset_id:
                continue
            gap = float(b.start) - float(a.end)
            if gap < 0.0 or gap > 12.0:
                continue
            ac, bc = _content(a.text), _content(b.text)
            if min(len(ac), len(bc)) < 2:
                continue
            shared = len(ac & bc)
            coverage = shared / max(1, min(len(ac), len(bc)))
            if shared < 2 or coverage < 0.62:
                continue
            timeline = timelines.get(a.source_asset_id)
            da, db = _event_density(timeline, a), _event_density(timeline, b)
            if abs(da - db) < 0.12:
                continue
            loser = i if da > db else j
            winner = j if loser == i else i
            removed.add(loser)
            diagnostics.append({
                "action": "collapse_redundant_retry_by_visual_quality",
                "removed_clip_id": clips[loser].clip_id,
                "winner_clip_id": clips[winner].clip_id,
                "content_coverage": round(coverage, 3),
                "removed_reset_density": round(max(da, db), 3),
                "winner_reset_density": round(min(da, db), 3),
            })
            break
    if not removed:
        return result, diagnostics
    draft = replace(result.draft, selected=tuple(c for idx, c in enumerate(clips) if idx not in removed))
    return replace(result, draft=draft), diagnostics


def polish_human_boundaries_v2(result: ProcessingResult, local_paths: Mapping[str, str]) -> ProcessingResult:
    if not hasattr(result.draft, "selected") or not hasattr(result.draft, "alternates"):
        return result
    all_diag: list[dict] = []
    result, rows = _restore_tiny_completion_alternates(result)
    all_diag.extend(rows)
    timelines = _timeline_proxies(result)
    result, rows = _collapse_redundant_retry_pair(result, timelines)
    all_diag.extend(rows)

    # Keep the proven v1 edge-silence and repeated-tail cleanup, now with reconstructed
    # timeline proxies so its reset corroboration can operate after serialization.
    result = polish_human_boundaries(result, local_paths, tuple(timelines.values()))

    polished: list[DraftClip] = []
    for clip in result.draft.selected:
        path = local_paths.get(clip.source_asset_id)
        if not path:
            polished.append(clip)
            continue
        pieces, rows = _split_short_reset_dead_air(clip, path, timelines.get(clip.source_asset_id))
        polished.extend(pieces)
        all_diag.extend({"clip_id": clip.clip_id, **row} for row in rows)

    existing = list((result.draft.diagnostics or {}).get("human_boundary_polish") or ())
    if not all_diag:
        return result
    diagnostics = dict(result.draft.diagnostics or {})
    diagnostics["human_boundary_polish"] = [*existing, *all_diag][:300]
    draft = replace(result.draft, selected=tuple(polished), diagnostics=diagnostics)
    return replace(result, draft=draft)
