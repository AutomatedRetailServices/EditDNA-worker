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
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .post_render_watch_listen_qc import PostRenderFinding
from .render_plan import RenderSegment

_EDGE_TOLERANCE_SEC = 0.6
_MAX_TRIM_FRACTION = 0.4
_MIN_TRIM_SEC = 0.05
_MIN_REMAINING_SEGMENT_SEC = 0.5


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


def segment_output_windows(segments: tuple[RenderSegment, ...]) -> list[tuple[float, float]]:
    """Cumulative [output_start, output_end) window for each segment in the
    concatenated output timeline `render_preview` actually produces.

    Reuses `render.tighten_trailing_silence` (the SAME per-segment trim
    `render_preview` itself applies before concatenating) so this mapping
    from output-timeline offsets back to segments never silently drifts from
    what the real renderer does -- one implementation, not a second guess.
    """
    from .render import tighten_trailing_silence

    windows: list[tuple[float, float]] = []
    cursor = 0.0
    for seg in segments:
        tightened = tighten_trailing_silence(seg)
        duration = tightened.duration_sec
        windows.append((cursor, cursor + duration))
        cursor += duration
    return windows


def repair_segment_for_finding(
    segments: tuple[RenderSegment, ...], finding: PostRenderFinding,
) -> tuple[tuple[RenderSegment, ...], SegmentRepairAttempt] | None:
    """Attempt ONE targeted, Boundary-only physical repair for a single
    physical `PostRenderFinding` located in the OUTPUT timeline. Returns
    `None` if this finding is not safely repairable this way -- callers must
    treat `None` as "no safe repair", never retry with a different guess.
    """
    windows = segment_output_windows(segments)
    finding_start, finding_end = float(finding.start), float(finding.end)

    for index, (seg, (win_start, win_end)) in enumerate(zip(segments, windows)):
        if finding_start < win_start - _EDGE_TOLERANCE_SEC or finding_start > win_end + _EDGE_TOLERANCE_SEC:
            continue

        near_trailing_edge = finding_start >= win_start - 1e-6 and abs(finding_end - win_end) <= _EDGE_TOLERANCE_SEC
        near_leading_edge = finding_end <= win_end + 1e-6 and abs(finding_start - win_start) <= _EDGE_TOLERANCE_SEC
        if not (near_trailing_edge or near_leading_edge):
            continue  # a mid-segment defect -- a boundary trim cannot reach it safely

        defect_duration = max(_MIN_TRIM_SEC, finding_end - finding_start)
        max_trim = seg.duration_sec * _MAX_TRIM_FRACTION
        trim = min(defect_duration, max_trim)
        if seg.duration_sec - trim < _MIN_REMAINING_SEGMENT_SEC:
            continue  # would eat too much of the real segment -- refuse, do not guess

        edge = "trailing" if near_trailing_edge else "leading"
        if edge == "trailing":
            repaired_seg = replace(seg, end=seg.end - trim)
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
        )
        return tuple(new_segments), attempt

    return None
