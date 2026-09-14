"""Human Watch+Listen boundary authority v3.

Adds conservative source-evidenced cleanup for failure modes that remained after v2:
- clusters of silence/reset inside a selected take are recording-process debris;
- a later nearby retry can own the repeated tail while unique earlier speech is preserved;
- a dangling incomplete tail before a complete short follow-up can be trimmed at a safe word gap.

No benchmark timestamps, phrases, or source names are hardcoded. Source audio and word
boundaries remain the physical authority; uncertain cases fail open.
"""
from __future__ import annotations

from dataclasses import replace
import re
import subprocess
from typing import Mapping

from .contracts import DraftClip, ProcessingResult, Word
from .human_boundary_polish_v2 import polish_human_boundaries_v2, _timeline_proxies, _reset_score

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "as", "at", "by", "con", "de", "del", "el", "en", "for",
    "from", "in", "la", "las", "lo", "los", "of", "on", "or", "para", "por", "que",
    "the", "to", "un", "una", "with", "y",
})
_DANGLING = (
    ("así", "que"), ("asi", "que"), ("so",), ("and",), ("but",),
    ("therefore",), ("entonces",), ("por", "eso"),
)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> set[str]:
    return {t for t in _tokens(text) if len(t) >= 3 and t not in _STOP}


def _silences(path: str, start: float, end: float, *, minimum_sec: float = 0.24) -> tuple[tuple[float, float], ...]:
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
    out: list[tuple[float, float]] = []
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
                out.append((s, e))
    for s in starts:
        out.append((s, end))
    return tuple(out)


def _words_in(words: tuple[Word, ...], start: float, end: float) -> tuple[Word, ...]:
    return tuple(w for w in words if float(w.end) > start and float(w.start) < end)


def _piece(clip: DraftClip, start: float, end: float) -> DraftClip | None:
    if end - start < 0.22:
        return None
    words = _words_in(tuple(clip.words), start, end)
    text = " ".join(str(w.text).strip() for w in words).strip()
    if not text and clip.words:
        return None
    return replace(clip, start=start, end=end, words=words, text=text or clip.text, caption_text=text or clip.caption_text)


def _remove_interval(clip: DraftClip, cut_start: float, cut_end: float) -> tuple[DraftClip, ...]:
    if cut_start <= clip.start + 0.12 and cut_end >= clip.end - 0.12:
        return ()
    out: list[DraftClip] = []
    if cut_start > clip.start + 0.12:
        left = _piece(clip, float(clip.start), cut_start)
        if left is not None:
            out.append(left)
    if cut_end < clip.end - 0.12:
        right = _piece(clip, cut_end, float(clip.end))
        if right is not None:
            out.append(right)
    return tuple(out) or (clip,)


def _silence_clusters(silences: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float, float, int], ...]:
    """Return compact clusters: start, end, total_silence, count."""
    if not silences:
        return ()
    clusters: list[list[tuple[float, float]]] = []
    for interval in silences:
        if not clusters or interval[0] - clusters[-1][-1][1] > 1.15:
            clusters.append([interval])
        else:
            clusters[-1].append(interval)
    out = []
    for rows in clusters:
        total = sum(e - s for s, e in rows)
        out.append((rows[0][0], rows[-1][1], total, len(rows)))
    return tuple(out)


def _clean_recording_process_clusters(clip: DraftClip, path: str, timeline) -> tuple[tuple[DraftClip, ...], list[dict]]:
    silences = _silences(path, float(clip.start), float(clip.end))
    pieces: tuple[DraftClip, ...] = (clip,)
    diagnostics: list[dict] = []
    candidates: list[tuple[float, float, str, float]] = []

    for s, e in silences:
        span = e - s
        reset = _reset_score(timeline, s, e)
        # Source audio is stronger than ASR chunk timing for a real long pause.
        if span >= 1.05 or (span >= 0.55 and reset >= 0.90):
            candidates.append((s, e, "long_dead_air", reset))

    for s, e, total, count in _silence_clusters(silences):
        width = e - s
        reset = _reset_score(timeline, s, e)
        # Multiple pauses separated by tiny speech islands are characteristic of a
        # fumble/reset/retake setup. Remove the whole cluster, not only the silence.
        if count >= 2 and width <= 6.5 and total >= 1.25 and (reset >= 0.70 or total >= 1.70):
            candidates.append((s, e, "fumble_dead_air_cluster", reset))

    # Apply widest/highest-value intervals first and avoid nested double edits.
    candidates.sort(key=lambda row: (-(row[1] - row[0]), row[0]))
    applied: list[tuple[float, float]] = []
    for s, e, kind, reset in candidates:
        if any(not (e <= a or s >= b) for a, b in applied):
            continue
        rebuilt: list[DraftClip] = []
        changed = False
        for piece in pieces:
            if e <= piece.start or s >= piece.end:
                rebuilt.append(piece)
                continue
            cut_s = max(float(piece.start), s)
            cut_e = min(float(piece.end), e)
            new = _remove_interval(piece, cut_s, cut_e)
            if new != (piece,):
                changed = True
            rebuilt.extend(new)
        if changed:
            pieces = tuple(rebuilt)
            applied.append((s, e))
            diagnostics.append({
                "action": "remove_recording_process_cluster",
                "kind": kind,
                "start": round(s, 3), "end": round(e, 3),
                "duration_sec": round(e - s, 3),
                "reset_score": round(reset, 3),
            })
    return pieces, diagnostics


def _trim_repeated_tail_before_later_retry(result: ProcessingResult) -> tuple[ProcessingResult, list[dict]]:
    clips = list(result.draft.selected)
    diagnostics: list[dict] = []
    for i in range(len(clips) - 1):
        left, right = clips[i], clips[i + 1]
        if left.source_asset_id != right.source_asset_id or not left.words:
            continue
        gap = float(right.start) - float(left.end)
        if gap < 0 or gap > 15.0:
            continue
        right_content = _content(right.text)
        if len(right_content) < 2:
            continue
        words = list(left.words)
        # Examine only the final delivery clause; unique earlier material must survive.
        for width in range(min(12, len(words)), 3, -1):
            tail_words = words[-width:]
            tail_text = " ".join(str(w.text) for w in tail_words)
            tail_content = _content(tail_text)
            shared = len(tail_content & right_content)
            coverage = shared / max(1, min(len(tail_content), len(right_content)))
            if shared < 2 or coverage < 0.55:
                continue
            cut = float(tail_words[0].start)
            if cut - float(left.start) < 1.0:
                continue
            kept = tuple(w for w in words if float(w.end) <= cut + 1e-6)
            if not kept:
                continue
            text = " ".join(str(w.text).strip() for w in kept).strip()
            clips[i] = replace(left, end=cut, words=kept, text=text, caption_text=text)
            diagnostics.append({
                "action": "trim_inferior_retry_tail_before_later_take",
                "left_clip_id": left.clip_id,
                "right_clip_id": right.clip_id,
                "shared_content_tokens": shared,
                "coverage": round(coverage, 3),
                "new_end": round(cut, 3),
            })
            break
    if not diagnostics:
        return result, diagnostics
    draft = replace(result.draft, selected=tuple(clips))
    return replace(result, draft=draft), diagnostics


def _trim_dangling_tail_before_complete_followup(result: ProcessingResult) -> tuple[ProcessingResult, list[dict]]:
    clips = list(result.draft.selected)
    diagnostics: list[dict] = []
    for i in range(len(clips) - 1):
        left, right = clips[i], clips[i + 1]
        if left.source_asset_id != right.source_asset_id or not left.words:
            continue
        if float(right.end) - float(right.start) > 8.0:
            continue
        lt = _tokens(left.text)
        dangling = next((d for d in _DANGLING if len(lt) >= len(d) and tuple(lt[-len(d):]) == d), None)
        if dangling is None:
            continue
        words = list(left.words)
        # Find the last natural word gap before the dangling tail and trim there.
        tail_start_idx = max(0, len(words) - max(8, len(dangling) + 2))
        cut = None
        for j in range(len(words) - len(dangling) - 1, tail_start_idx - 1, -1):
            gap = float(words[j + 1].start) - float(words[j].end)
            if gap >= 0.38:
                cut = float(words[j].end)
                break
        if cut is None:
            # Fallback: trim exactly before the dangling connector if word alignment is clear.
            token_words = [w for w in words if _tokens(w.text)]
            if len(token_words) <= len(dangling):
                continue
            cut = float(token_words[-len(dangling)].start)
        if cut - float(left.start) < 1.0:
            continue
        kept = tuple(w for w in words if float(w.end) <= cut + 1e-6)
        if not kept:
            continue
        text = " ".join(str(w.text).strip() for w in kept).strip()
        clips[i] = replace(left, end=cut, words=kept, text=text, caption_text=text)
        diagnostics.append({
            "action": "trim_dangling_incomplete_tail_before_complete_followup",
            "left_clip_id": left.clip_id,
            "right_clip_id": right.clip_id,
            "new_end": round(cut, 3),
        })
    if not diagnostics:
        return result, diagnostics
    draft = replace(result.draft, selected=tuple(clips))
    return replace(result, draft=draft), diagnostics


def polish_human_boundaries_v3(result: ProcessingResult, local_paths: Mapping[str, str]) -> ProcessingResult:
    result = polish_human_boundaries_v2(result, local_paths)
    if not hasattr(result.draft, "selected"):
        return result
    diagnostics: list[dict] = []

    result, rows = _trim_repeated_tail_before_later_retry(result)
    diagnostics.extend(rows)
    result, rows = _trim_dangling_tail_before_complete_followup(result)
    diagnostics.extend(rows)

    timelines = _timeline_proxies(result)
    polished: list[DraftClip] = []
    for clip in result.draft.selected:
        path = local_paths.get(clip.source_asset_id)
        if not path:
            polished.append(clip)
            continue
        pieces, rows = _clean_recording_process_clusters(clip, path, timelines.get(clip.source_asset_id))
        polished.extend(pieces)
        diagnostics.extend({"clip_id": clip.clip_id, **row} for row in rows)

    if not diagnostics:
        return result
    existing = list((result.draft.diagnostics or {}).get("human_boundary_polish") or ())
    diag = dict(result.draft.diagnostics or {})
    diag["human_boundary_polish"] = [*existing, *diagnostics][:400]
    draft = replace(result.draft, selected=tuple(polished), diagnostics=diag)
    return replace(result, draft=draft)
