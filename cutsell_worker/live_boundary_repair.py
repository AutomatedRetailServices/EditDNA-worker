"""Targeted, Boundary-only, single-segment physical repair -- D-030.

`repair_loop.py` (D-026) repairs a SEMANTIC finding (STORY_ORDER_BREAK) by
reordering already-selected clips; it never touches physical timing. This
module is the physical mirror the canonical directive's live-wiring order
requires: given ONE physical `PostRenderFinding` located in the actual
rendered OUTPUT timeline (from `post_render_media_qc.py`, run against the
real local file), trim the ONE `RenderSegment` whose edge the defect sits
against -- nothing else in the timeline moves, no other segment's start/end
changes, and no semantic membership is touched. This is BoundaryEngine's own
kind of authority (physical timing only), never Selection's.

## Why only an edge-adjacent defect is repairable here

A defect (dead frame, freeze, accidental silence, a hard splice
discontinuity) sitting at a segment's own leading or trailing edge is
exactly the shape a boundary TRIM can fix without guessing: shrink that one
edge by the defect's own duration. A defect in the MIDDLE of a segment's own
source footage is not a boundary problem at all -- trimming an edge cannot
reach it, and inventing some other physical mutation to "fix" it would be
guessing, not a targeted repair. `repair_segment_for_finding` returns `None`
for that case (and for a trim that would eat too much of the segment's real
content), and the caller (`live_render_qc.py`) must treat `None` as "no safe
physical repair exists" and stop, per the same "WHEN UNCERTAIN, KEEP" /
never-guess posture this whole codebase already applies elsewhere
(`repair_loop.py`'s own honest scope, `deterministic_best_take_authority`'s
thin-score-gap refusal).
## Evidence a trim must have (D-291.6)

`_EDGE_TOLERANCE_SEC` is a ROUTING width: it decides which segment's edge a
finding belongs to. It was also, until D-291.6, the amount of real content
a repair could silently remove: a measured silence that STRADDLES a join
(it begins up to the tolerance before this segment's window, or ends up to
the tolerance after it) was trimmed by its WHOLE duration from this
segment's edge, so up to the tolerance of speech at the head or tail of
the segment went with it, and no word boundary was ever consulted on
either edge (the trailing branch only trusted the renderer's silence
tightener). Two evidence rules now bound every trim:

1. **Only the defect measured INSIDE this segment's own rendered window is
   trimmed.** The part of a finding that lies in a neighbouring segment's
   window is that segment's defect, never this one's content.
2. **A trim never enters a word.** When the caller supplies the selected
   clips' own word timings (`protected_speech_by_clip_id`, source seconds,
   built from the frozen draft's `DraftClip.words`), the trim is clamped to
   the non-speech room at that edge. A finding with NO measured extent
   inside the window (a join-instant probe) is repairable only when word
   evidence proves there is room; without it the repair is refused, fail
   closed, and the loop records PHYSICAL_FAIL_UNREPAIRABLE instead of
   cutting a phoneme to make a probe pass.

`None` still means "no safe repair"; callers never retry with another guess.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .post_render_watch_listen_qc import PostRenderFinding
from .render_plan import RenderSegment

_EDGE_TOLERANCE_SEC = 0.6
_MAX_TRIM_FRACTION = 0.4
_MIN_TRIM_SEC = 0.05
_MIN_REMAINING_SEGMENT_SEC = 0.5

SpeechIntervals = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class SegmentRepairAttempt:
    segment_index: int
    clip_id: str
    finding_kind: str
    edge: str  # "trailing" | "leading"
    original_start: float
    original_end: float
    repaired_start: float
    repaired_end: float
    trim_sec: float
    reason: str
    # D-097.4: the renderer-tightened trailing edge the trim was taken from
    # (None for a leading-edge repair) -- the repair record shows the edge
    # the output really carried, not only the plan's own end.
    tightened_end: float | None = None
    # D-291.6: the defect extent measured inside this segment's own rendered
    # window (the only part a trim may remove) and the non-speech room at
    # the trimmed edge per the clip's word timings (None when the caller
    # supplied no word evidence for this clip).
    measured_inside_window_sec: float | None = None
    speech_room_sec: float | None = None


def segment_output_windows(segments: tuple[RenderSegment, ...]) -> list[tuple[float, float]]:
    """Cumulative [output_start, output_end) window for each segment in the
    concatenated output timeline `render_preview` actually produces.

    Reuses `render.tighten_trailing_silence` (the SAME per-segment trim
    `render_preview` itself applies before concatenating) so this mapping
    from output-timeline offsets back to segments never silently drifts from
    what the real renderer does -- one implementation, not a second guess.
    """
    from .render import RENDER_FPS_DEFAULT, rendered_segment_duration_sec, tighten_trailing_silence

    windows: list[tuple[float, float]] = []
    cursor = 0.0
    for seg in segments:
        tightened = tighten_trailing_silence(seg)
        # D-097.2: the renderer places every segment on a frame-aligned
        # output timeline (`rendered_segment_duration_sec`); mapping with the
        # raw duration drifted from the real joins by ~20-60 ms per part
        # under the old part+concat-demuxer renderer and would still drift by
        # one frame-rounding per part here.
        duration = rendered_segment_duration_sec(tightened.duration_sec, fps=RENDER_FPS_DEFAULT)
        windows.append((cursor, cursor + duration))
        cursor += duration
    return windows


def _speech_for_segment(
    seg: RenderSegment, protected_speech_by_clip_id: Mapping[str, SpeechIntervals] | None,
) -> SpeechIntervals | None:
    """The caller-supplied word intervals (source seconds) for this segment's
    clip, looked up by the segment's own clip id and then by its parent
    semantic clip id (a physical fragment inherits its parent's words).
    `None` means no word evidence was supplied for this clip at all."""
    if protected_speech_by_clip_id is None:
        return None
    for key in (seg.clip_id, getattr(seg, "parent_semantic_clip_id", None)):
        if key and key in protected_speech_by_clip_id:
            return tuple(
                (float(start), float(end)) for start, end in protected_speech_by_clip_id[key]
            )
    return None


def speech_room_at_edge(
    seg: RenderSegment, edge: str, *, edge_time: float, speech: SpeechIntervals,
) -> float:
    """Seconds of non-speech material between a segment's edge and the
    nearest word inside the segment: for the leading edge, from `seg.start`
    to the first word start; for the trailing edge, from the last word end
    to `edge_time` (the edge the output really carries). Words that do not
    overlap the segment are ignored; a segment with no overlapping word
    has the whole segment as room (there is no speech to protect)."""
    seg_start, seg_end = float(seg.start), float(edge_time)
    inside = [(s, e) for s, e in speech if s < seg_end and e > seg_start]
    if not inside:
        return max(0.0, seg_end - seg_start)
    if edge == "leading":
        return max(0.0, min(s for s, _ in inside) - seg_start)
    return max(0.0, seg_end - max(e for _, e in inside))


def repair_segment_for_finding(
    segments: tuple[RenderSegment, ...], finding: PostRenderFinding,
    *,
    protected_speech_by_clip_id: Mapping[str, SpeechIntervals] | None = None,
) -> tuple[tuple[RenderSegment, ...], SegmentRepairAttempt] | None:
    """Attempt ONE targeted, Boundary-only physical repair for a single
    physical `PostRenderFinding` located in the OUTPUT timeline. Returns
    `None` if this finding is not safely repairable this way -- callers must
    treat `None` as "no safe repair", never retry with a different guess.

    `protected_speech_by_clip_id` (D-291.6): word intervals in SOURCE
    seconds per selected clip id. When supplied, a trim never enters a
    word, and a finding with no measured extent inside the segment's window
    needs word evidence for that clip before any trim is taken.
    """
    from .render import tighten_trailing_silence

    windows = segment_output_windows(segments)
    finding_start, finding_end = float(finding.start), float(finding.end)

    for index, (seg, (win_start, win_end)) in enumerate(zip(segments, windows)):
        if finding_start < win_start - _EDGE_TOLERANCE_SEC or finding_start > win_end + _EDGE_TOLERANCE_SEC:
            continue

        near_trailing_edge = finding_start >= win_start - 1e-6 and abs(finding_end - win_end) <= _EDGE_TOLERANCE_SEC
        near_leading_edge = finding_end <= win_end + 1e-6 and abs(finding_start - win_start) <= _EDGE_TOLERANCE_SEC
        if not (near_trailing_edge or near_leading_edge):
            continue  # a mid-segment defect -- a boundary trim cannot reach it safely

        # D-097.4: the edge the OUTPUT actually carries is the renderer's
        # trailing-silence-tightened edge (`segment_output_windows` maps
        # with it), so a trailing repair must trim from THAT edge. Run
        # 34034507983 trimmed 50 ms from the un-tightened plan end instead:
        # the remaining silent tail (0.236 s) fell under the tightener's
        # 0.28 s minimum, the tightening vanished, and the "repair" made the
        # rendered segment 0.236 s LONGER, re-exposing dead air the renderer
        # had already removed and moving every later join by +0.233 s.
        tightened_end = tighten_trailing_silence(seg).end if near_trailing_edge else None
        effective_duration = (tightened_end - seg.start) if tightened_end is not None else seg.duration_sec
        edge = "trailing" if near_trailing_edge else "leading"
        # D-291.6 rule 1: only the defect measured INSIDE this segment's own
        # rendered window is this segment's to trim. A silence that
        # straddles the join belongs to this segment only for the part
        # that lies inside its window; the rest is the neighbour's.
        measured_inside = max(0.0, min(finding_end, win_end) - max(finding_start, win_start))
        defect_duration = max(_MIN_TRIM_SEC, measured_inside)
        max_trim = effective_duration * _MAX_TRIM_FRACTION
        trim = min(defect_duration, max_trim)
        # D-291.6 rule 2: never enter a word. With word evidence for this
        # clip the trim is clamped to the non-speech room at the edge; a
        # finding with no measured extent inside the window (a join-instant
        # probe) is repairable only on that evidence.
        speech = _speech_for_segment(seg, protected_speech_by_clip_id)
        speech_room = None
        if speech is not None:
            edge_time = float(tightened_end) if tightened_end is not None else float(seg.end)
            speech_room = speech_room_at_edge(seg, edge, edge_time=edge_time, speech=speech)
            trim = min(trim, speech_room)
        elif protected_speech_by_clip_id is not None and measured_inside < _MIN_TRIM_SEC:
            continue  # no measured defect inside the window and no word evidence: refuse
        if trim < _MIN_TRIM_SEC - 1e-9:
            continue  # the safe trim is below the minimum: a word sits at the edge -- refuse
        if effective_duration - trim < _MIN_REMAINING_SEGMENT_SEC:
            continue  # would eat too much of the real segment -- refuse, do not guess

        if edge == "trailing":
            repaired_seg = replace(seg, end=float(tightened_end) - trim)
        else:
            repaired_seg = replace(seg, start=seg.start + trim)

        new_segments = list(segments)
        new_segments[index] = repaired_seg
        attempt = SegmentRepairAttempt(
            segment_index=index,
            clip_id=seg.clip_id,
            finding_kind=finding.kind,
            edge=edge,
            original_start=seg.start,
            original_end=seg.end,
            repaired_start=repaired_seg.start,
            repaired_end=repaired_seg.end,
            trim_sec=trim,
            reason=f"trimmed_{edge}_edge_by_{trim:.3f}s_for_{finding.kind}",
            tightened_end=tightened_end,
            measured_inside_window_sec=measured_inside,
            speech_room_sec=speech_room,
        )
        return tuple(new_segments), attempt

    return None
