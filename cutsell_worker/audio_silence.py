"""Objective audio dead-air intervals from the source media (D-095.2).

Why this exists: every silence signal the engine had BEFORE Selection Freeze
was derived from ASR word timing (`silence_analysis.word_silence_gaps`,
`take_segmentation._speech_units`, `post_selection_interior_gap_trim`'s
word-gap scan). Whisper-style word timestamps are routinely stretched over
real silence inside a hesitant delivery, so a kept take could carry seconds
of dead air that no word gap ever revealed -- and the first objective audio
measurement happened only AFTER render, in `post_render_media_qc`
(LINGERING_ACCIDENTAL_SILENCE, -35 dB for >= 1.2 s, unrepairable when it
sits mid-segment). Run 33995806350: a 12.5 s kept candidate carrying a
2.36 s interior silence invalidated the whole render.

This module measures the same thing the QC measures, on the SOURCE, once
per source, with ffmpeg's `silencedetect` (the exact filter
`render.tighten_trailing_silence` already trusts for trailing post-roll) and
publishes the intervals as `TemporalEvent`s of kind
``audio_silence_interval`` on the whole-video context -- the same channel
local-performance reset events already travel on -- so the existing
interior-gap trimmer can use them as evidence. Observability + evidence
only: nothing here changes semantic membership.
"""
from __future__ import annotations

from dataclasses import replace
import re
import subprocess
from typing import Iterable, Mapping

from .whole_video_analysis import TemporalEvent, WholeVideoContext

AUDIO_SILENCE_EVENT_KIND = "audio_silence_interval"
DEFAULT_NOISE_DB = -35.0
DEFAULT_MINIMUM_SILENCE_SEC = 0.60
_SUBPROCESS_TIMEOUT_SEC = 600

# D-097 Priority C (C-12 reconciliation). Reproduced offline with synthetic
# audio: when a pause carries room tone right at the -35 dB floor, the SAME
# silencedetect filter reports a shorter, later silence on the source (noise
# peaks above the floor fragment the run and a 0.6 s probe drops the pieces)
# than on the AAC-re-encoded render (whose noise floor sits lower). Run
# 34008386434: the source pass published no qualifying interval where the
# render QC then measured 2.32 s. Two measures close the gap without touching
# the QC's own threshold: (1) probe at a finer minimum and MERGE runs that
# are separated by less than ``merge_gap_sec`` of near-floor noise, so a
# fragmented pause is reported as the one pause it is; (2) a second,
# RELAXED floor (-30 dB) whose intervals are published at lower confidence
# -- speech never sits below -30 dBFS for over a second, so a relaxed-floor
# interval is still objective dead air, only measured more tolerantly.
DEFAULT_PROBE_MINIMUM_SEC = 0.30
DEFAULT_MERGE_GAP_SEC = 0.30
RELAXED_NOISE_DB = -30.0
RELAXED_MINIMUM_SILENCE_SEC = 1.20
RELAXED_CONFIDENCE = 0.90

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[0-9.]+)")


def merge_silence_runs(
    intervals: Iterable[tuple[float, float]], *, merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
) -> tuple[tuple[float, float], ...]:
    """Join silence runs separated by less than ``merge_gap_sec``: a burst of
    near-floor noise inside one pause is still the same pause."""
    merged: list[list[float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in intervals):
        if merged and start - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((s, e) for s, e in merged)


def detect_audio_silence_intervals(
    path: str,
    *,
    noise_db: float = DEFAULT_NOISE_DB,
    minimum_silence_sec: float = DEFAULT_MINIMUM_SILENCE_SEC,
    ffmpeg_bin: str = "ffmpeg",
    probe_minimum_sec: float = DEFAULT_PROBE_MINIMUM_SEC,
    merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
) -> tuple[tuple[float, float], ...]:
    """Closed silence intervals (source seconds) at or below ``noise_db`` lasting
    at least ``minimum_silence_sec`` after merging runs closer than
    ``merge_gap_sec`` (probed at ``probe_minimum_sec``). Never raises: an
    unreadable file, a missing ffmpeg or a timeout yields an empty tuple (the
    caller records the count, so an empty result is visible, never silent)."""
    probe = min(float(probe_minimum_sec), float(minimum_silence_sec))
    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "info", "-nostats",
        "-i", str(path), "-vn",
        "-af", f"silencedetect=noise={noise_db:.1f}dB:d={probe:.3f}",
        "-f", "null", "-",
    ]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=_SUBPROCESS_TIMEOUT_SEC,
        )
    except Exception:  # noqa: BLE001 -- evidence source unavailable; reported via count, never fatal
        return ()
    if completed.returncode != 0:
        return ()
    intervals: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in completed.stderr.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            pending_start = max(0.0, float(start_match.group(1)))
        end_match = _SILENCE_END_RE.search(line)
        if end_match and pending_start is not None:
            end = float(end_match.group(1))
            if end - pending_start >= probe - 1e-3:
                intervals.append((pending_start, end))
            pending_start = None
    # A silence still open at end-of-file is trailing post-roll, not an
    # interior interval; tighten_trailing_silence owns that case.
    merged = merge_silence_runs(intervals, merge_gap_sec=merge_gap_sec)
    return tuple((s, e) for s, e in merged if e - s >= minimum_silence_sec - 1e-3)


def _covered(interval: tuple[float, float], others: Iterable[tuple[float, float]], *, tolerance_sec: float = 0.15) -> bool:
    start, end = interval
    return any(o_start - tolerance_sec <= start and end <= o_end + tolerance_sec for o_start, o_end in others)


def audio_silence_events(
    local_paths: Mapping[str, str],
    *,
    noise_db: float = DEFAULT_NOISE_DB,
    minimum_silence_sec: float = DEFAULT_MINIMUM_SILENCE_SEC,
    relaxed_noise_db: float | None = RELAXED_NOISE_DB,
    relaxed_minimum_silence_sec: float = RELAXED_MINIMUM_SILENCE_SEC,
) -> dict[str, tuple[TemporalEvent, ...]]:
    """Primary-floor intervals at confidence 1.0 plus (D-097 C) relaxed-floor
    intervals >= ``relaxed_minimum_silence_sec`` that no primary interval
    already covers, at ``RELAXED_CONFIDENCE``; both share the one event kind
    every consumer already reads. ``relaxed_noise_db=None`` disables the
    second pass."""
    out: dict[str, tuple[TemporalEvent, ...]] = {}
    for source_asset_id, path in sorted(local_paths.items()):
        intervals = detect_audio_silence_intervals(path, noise_db=noise_db, minimum_silence_sec=minimum_silence_sec)
        events = [
            TemporalEvent(
                source_asset_id=str(source_asset_id),
                start=float(start),
                end=float(end),
                kind=AUDIO_SILENCE_EVENT_KIND,
                confidence=1.0,
                description=f"ffmpeg silencedetect <= {noise_db:.0f} dB for {end - start:.2f}s",
            )
            for start, end in intervals
        ]
        if relaxed_noise_db is not None:
            relaxed = detect_audio_silence_intervals(
                path, noise_db=relaxed_noise_db, minimum_silence_sec=relaxed_minimum_silence_sec,
            )
            events.extend(
                TemporalEvent(
                    source_asset_id=str(source_asset_id),
                    start=float(start),
                    end=float(end),
                    kind=AUDIO_SILENCE_EVENT_KIND,
                    confidence=RELAXED_CONFIDENCE,
                    description=f"ffmpeg silencedetect relaxed floor <= {relaxed_noise_db:.0f} dB for {end - start:.2f}s",
                )
                for start, end in relaxed
                if not _covered((start, end), intervals)
            )
        out[str(source_asset_id)] = tuple(sorted(events, key=lambda e: (e.start, e.end)))
    return out


def merge_audio_silence_into_context(
    context: WholeVideoContext,
    events_by_source: Mapping[str, Iterable[TemporalEvent]],
) -> WholeVideoContext:
    """Add the audio silence events to each matching source of the whole-video
    context (deduplicated on kind/start/end), mirroring
    ``local_performance.merge_local_events_into_context``."""
    merged = []
    for source in context.sources:
        additions_source = events_by_source.get(source.source_asset_id)
        if not additions_source:
            merged.append(source)
            continue
        known = {(e.kind, round(float(e.start), 3), round(float(e.end), 3)) for e in source.events}
        additions = tuple(
            e for e in additions_source
            if (e.kind, round(float(e.start), 3), round(float(e.end), 3)) not in known
        )
        merged.append(replace(source, events=tuple(sorted(
            tuple(source.events) + additions, key=lambda e: (float(e.start), float(e.end), e.kind),
        ))))
    return replace(context, sources=tuple(merged))


# --- D-291.9: ASR word timings reconciled against the measured silence ------
#
# RAW #126 (project video00-modal-35931561397-1), verified on the source
# audio and on the rendered MP4: Whisper placed a sentence-final word and,
# elsewhere, a clause-initial negation particle ENTIRELY inside a silence
# the same run had measured with ffmpeg silencedetect (one under the
# primary -35 dB floor, the other under the relaxed -30 dB floor). The real
# words sit right next to the pause: the sentence-final word ends where the
# silence starts (speech energy up to ~0.3 s before it) and the particle
# starts where the last silence before its clause ends. Every downstream
# authority trusted the ASR spans: the take ended on the previous word,
# inside the real one; a 0.6 s clip of pure silence was rendered for it;
# the AttemptReconstructor split the particle from its clause across the
# measured dead air (D-097.5, correct on its own evidence) and the rendered
# sentence lost its negation. The same padding stretches sentence-final
# words over the following pause (one ran 1 s past a measured 0.64 s
# silence that started inside it).
#
# This is the measured-silence authority (D-095.2 / D-097 Priority C), so
# the reconciliation lives here: a word whose whole ASR span lies inside a
# measured silence was not spoken there; it is re-anchored to the adjacent
# non-silent room next to its neighbouring words (the larger room wins), and
# a word a primary-floor measured silence starts inside ends where that
# silence starts (symmetrically for a padded start). No word is invented,
# dropped or re-ordered; a word with no room on either side is left as it
# was and reported. A re-anchored word that becomes adjacent to the
# neighbouring ASR segment joins it (the same 0.75 s speech-unit gap
# `take_segmentation`/`canonical_asr_evidence` already use), so the
# sentence-final word is back in its sentence and the negated clause stays
# one unit.
WORD_RECONCILIATION_EDGE_TOLERANCE_SEC = 0.05
WORD_RECONCILIATION_MINIMUM_ROOM_SEC = 0.05
WORD_RECONCILIATION_RULE_BEFORE = "reanchored_before_measured_silence"
WORD_RECONCILIATION_RULE_AFTER = "reanchored_after_measured_silence"
WORD_RECONCILIATION_RULE_NO_ROOM = "inside_measured_silence_no_adjacent_room"
WORD_RECONCILIATION_RULE_END = "padded_end_clamped_to_silence_start"
WORD_RECONCILIATION_RULE_START = "padded_start_clamped_to_silence_end"


_TERMINAL_MARKS = (".", "!", "?", "…")


def _is_sentence_final(text: str) -> bool:
    return str(text or "").strip().rstrip("\"'”’)").endswith(_TERMINAL_MARKS)


def _measured_silences(events: Iterable, *, minimum_confidence: float) -> list[tuple[float, float, float]]:
    """(start, end, confidence) triples of the measured silence events at or
    above `minimum_confidence`, sorted by start."""
    out = []
    for event in events or ():
        kind = getattr(event, "kind", None) if not isinstance(event, dict) else event.get("kind")
        if str(kind or "") != AUDIO_SILENCE_EVENT_KIND:
            continue
        conf = getattr(event, "confidence", 0.0) if not isinstance(event, dict) else event.get("confidence", 0.0)
        if float(conf or 0.0) < minimum_confidence:
            continue
        start = float(getattr(event, "start", 0.0) if not isinstance(event, dict) else event.get("start", 0.0))
        end = float(getattr(event, "end", 0.0) if not isinstance(event, dict) else event.get("end", 0.0))
        if end > start:
            out.append((start, end, float(conf or 0.0)))
    return sorted(out)


def reconcile_transcript_words_with_measured_silence(
    segments: Iterable,
    events_by_source: Mapping[str, Iterable],
    *,
    minimum_confidence: float = RELAXED_CONFIDENCE,
    clamp_minimum_confidence: float = 1.0,
    adjacent_gap_sec: float | None = None,
    edge_tolerance_sec: float = WORD_RECONCILIATION_EDGE_TOLERANCE_SEC,
    minimum_room_sec: float = WORD_RECONCILIATION_MINIMUM_ROOM_SEC,
) -> tuple[tuple, tuple[dict, ...]]:
    """Return (segments, rows): the transcript segments with every word span
    reconciled against the measured silence of its source (see the module
    comment above) and one observability row per changed or unplaceable
    word. Segments of a source with no measured silence are returned as the
    same objects; nothing is invented, dropped or re-ordered.

    Evidence rules (RAW #126 dry run over the whole source, checked against
    the audio peaks): the relaxed -30 dB floor swallows the quiet tail of a
    trailing word (peaks of 700-1100 under the 1036 relaxed peak), so a word
    is CLAMPED only by a primary-floor silence (`clamp_minimum_confidence`);
    a fully covered word prefers the primary floor too and falls back to the
    relaxed floor (the negation-particle case sits only under the relaxed
    floor). A sentence-final word never re-anchors to the right (it belongs
    to the sentence before the pause) and a right re-anchor never leaves the
    word's own ASR segment span (the ASR grouped the particle with its
    clause; it did not group a sentence-final word with the next sentence)."""
    from dataclasses import replace as _replace
    from .canonical_asr_evidence import DEFAULT_SPLIT_GAP_SEC

    gap_sec = float(DEFAULT_SPLIT_GAP_SEC if adjacent_gap_sec is None else adjacent_gap_sec)
    segment_list = list(segments)
    rows: list[dict] = []
    # per-segment mutable word lists: seg_idx -> list of Word (in ASR order)
    seg_words: dict[int, list] = {idx: list(seg.words) for idx, seg in enumerate(segment_list)}
    changed_segments: set[int] = set()

    by_source: dict[str, list[int]] = {}
    for idx, seg in enumerate(segment_list):
        by_source.setdefault(str(seg.source_asset_id), []).append(idx)

    for source_asset_id, seg_indices in by_source.items():
        silences = _measured_silences(events_by_source.get(source_asset_id, ()), minimum_confidence=minimum_confidence)
        if not silences:
            continue
        clamp_silences = [s for s in silences if s[2] >= clamp_minimum_confidence]
        # time-ordered word entries: (start, end, seg_idx, word_idx)
        entries = []
        for seg_idx in seg_indices:
            for word_idx, word in enumerate(seg_words[seg_idx]):
                entries.append([float(word.start), float(word.end), seg_idx, word_idx])
        entries.sort(key=lambda e: (e[0], e[1], e[2], e[3]))
        moves: list[tuple[int, int, str, dict]] = []  # (seg_idx, word_idx, rule, row)

        for pos, entry in enumerate(entries):
            ws, we, seg_idx, word_idx = entry
            word = seg_words[seg_idx][word_idx]
            if we <= ws:
                continue  # zero-length ASR word: no span to reconcile
            duration = we - ws
            prev_end = entries[pos - 1][1] if pos > 0 else None
            next_start = entries[pos + 1][0] if pos + 1 < len(entries) else None
            covering = [s for s in silences if s[0] <= ws + edge_tolerance_sec and we <= s[1] + edge_tolerance_sec]
            new_start, new_end, rule, silence = ws, we, None, None
            segment_end = float(segment_list[seg_idx].end)
            if covering:
                # primary floor first, then the largest overlap
                silence = max(covering, key=lambda s: (s[2], min(we, s[1]) - max(ws, s[0])))
                left_room = (silence[0] - prev_end) if prev_end is not None else 0.0
                anchor_end = silence[1]
                if next_start is not None:
                    later = [s[1] for s in silences if s[1] >= silence[0] and s[1] <= next_start + edge_tolerance_sec]
                    if later:
                        anchor_end = max(later)
                right_room = (next_start - anchor_end) if next_start is not None else 0.0
                right_allowed = (
                    not _is_sentence_final(word.text)
                    and next_start is not None
                    and anchor_end <= segment_end + edge_tolerance_sec
                )
                if left_room >= right_room and left_room >= minimum_room_sec:
                    new_end = silence[0]
                    new_start = max(prev_end, silence[0] - duration)
                    rule = WORD_RECONCILIATION_RULE_BEFORE
                elif right_allowed and right_room >= minimum_room_sec:
                    new_start = anchor_end
                    new_end = min(next_start, anchor_end + duration)
                    rule = WORD_RECONCILIATION_RULE_AFTER
                elif left_room >= minimum_room_sec:
                    new_end = silence[0]
                    new_start = max(prev_end, silence[0] - duration)
                    rule = WORD_RECONCILIATION_RULE_BEFORE
                else:
                    rule = WORD_RECONCILIATION_RULE_NO_ROOM
            else:
                # padded end: a primary-floor silence that starts inside the word --
                # one word never contains a measured >= 0.6 s silence, so the word
                # ended where the earliest such silence starts (the ASR padded the
                # word to its segment end, past the pause)
                ends = [s for s in clamp_silences if ws + edge_tolerance_sec < s[0] < we - edge_tolerance_sec]
                if ends:
                    silence = min(ends)
                    new_end = silence[0]
                    rule = WORD_RECONCILIATION_RULE_END
                # padded start: a primary-floor silence that covers the word's start and
                # ends inside it -- the word started where the silence ends
                starts = [s for s in clamp_silences if s[0] <= ws + edge_tolerance_sec and ws + edge_tolerance_sec < s[1] < new_end - edge_tolerance_sec]
                if starts:
                    silence_start = max(starts)
                    new_start = silence_start[1]
                    rule = WORD_RECONCILIATION_RULE_START if rule is None else rule + "+" + WORD_RECONCILIATION_RULE_START
                    silence = silence if silence is not None else silence_start
            if rule is None:
                continue
            row = {
                "source_asset_id": source_asset_id,
                "word": str(word.text),
                "rule": rule,
                "from_start": round(ws, 3), "from_end": round(we, 3),
                "to_start": round(new_start, 3), "to_end": round(new_end, 3),
                "silence_start": round(silence[0], 3) if silence else None,
                "silence_end": round(silence[1], 3) if silence else None,
                "moved_to_adjacent_segment": False,
            }
            rows.append(row)
            if rule != WORD_RECONCILIATION_RULE_NO_ROOM and new_end > new_start:
                seg_words[seg_idx][word_idx] = _replace(word, start=round(new_start, 3), end=round(new_end, 3))
                entry[0], entry[1] = new_start, new_end
                changed_segments.add(seg_idx)
                if rule in (WORD_RECONCILIATION_RULE_BEFORE, WORD_RECONCILIATION_RULE_AFTER):
                    moves.append((seg_idx, word_idx, rule, row))

        # a re-anchored FIRST/LAST word that now sits next to the neighbouring
        # ASR segment belongs to that segment (same speech-unit gap rule)
        ordered = sorted(seg_indices, key=lambda i: (float(segment_list[i].start), i))
        for seg_idx, word_idx, rule, row in moves:
            words_here = seg_words[seg_idx]
            if not words_here or words_here[word_idx] is None:
                continue
            word = words_here[word_idx]
            position = ordered.index(seg_idx)
            if rule == WORD_RECONCILIATION_RULE_BEFORE and word_idx == 0 and position > 0:
                target = ordered[position - 1]
                target_words = [w for w in seg_words[target] if w is not None]
                if target_words and float(word.start) - float(target_words[-1].end) <= gap_sec:
                    seg_words[target].append(word)
                    words_here[word_idx] = None
                    changed_segments.update((seg_idx, target))
                    row["moved_to_adjacent_segment"] = True
            elif rule == WORD_RECONCILIATION_RULE_AFTER and word_idx == len(words_here) - 1 and position + 1 < len(ordered):
                target = ordered[position + 1]
                target_words = [w for w in seg_words[target] if w is not None]
                if target_words and float(target_words[0].start) - float(word.end) <= gap_sec:
                    seg_words[target].insert(0, word)
                    words_here[word_idx] = None
                    changed_segments.update((seg_idx, target))
                    row["moved_to_adjacent_segment"] = True

    if not changed_segments:
        return tuple(segment_list), tuple(rows)
    out = []
    for idx, seg in enumerate(segment_list):
        if idx not in changed_segments:
            out.append(seg)
            continue
        words = tuple(sorted((w for w in seg_words[idx] if w is not None), key=lambda w: (float(w.start), float(w.end))))
        if not words:
            continue
        out.append(_replace(
            seg,
            start=float(words[0].start),
            end=float(words[-1].end),
            text=" ".join(str(w.text) for w in words).strip(),
            words=words,
        ))
    return tuple(out), tuple(rows)
