"""Live PostRenderWatchListenQC + bounded physical repair wiring -- D-030.

This is the actual execution-path glue the canonical directive's live order
requires:

    Validated CanonicalEditPlan -> Selection Freeze -> BoundaryEngine
    -> Render actual MP4 -> PostRenderWatchListenQC on that actual file
    -> PASS?
        YES -> final output
        NO physical issue -> targeted Boundary repair -> re-render -> QC again
        NO semantic mismatch -> invalidate candidate / route upstream
                              (BoundaryEngine never touches semantics)

`render_with_post_render_qc` is a drop-in replacement for a bare
`render.render_preview(...)` call: it renders to a LOCAL file, runs every
real check (structural render-plan-vs-edit-plan checks, which need no
media decode, plus D-028's ffmpeg/ffprobe media checks against the actual
local file) BEFORE the caller uploads/packages anything, and only ever
returns a path when the file has genuinely passed. The RunPod worker
already has this file on local disk during the export job -- this never
downloads an artifact back from storage to check it.

## Authority rule, in code

A finding this module treats as "physical" is exactly
`post_render_watch_listen_qc.is_physical_finding_kind` -- everything else
(every `STRUCTURAL_*` kind: a render segment that does not cover, duplicates,
or reorders a `keep_sequence` clip) is treated as a semantic/structural
mismatch: the candidate is invalidated immediately, `BoundaryEngine` is never
asked to "fix" it by trimming, and the caller must route it upstream rather
than deliver it. Only a physical finding is ever routed to
`live_boundary_repair.repair_segment_for_finding`, and only for
`max_attempts` bounded render/QC cycles; exhausting them without a clean
pass reports `NEEDS_HUMAN_REVIEW`, never `PASS`. Global semantic re-
computation (re-running `repair_loop`/`CanonicalEditPlan`) never happens
here -- this module only ever mutates the local `RenderSegment` list it was
given, never `draft.selected` or anything upstream of Boundary.

The structural checks (segment coverage/order/duplication vs. the frozen
CanonicalEditPlan) run ONCE, against attempt 0's own untouched segments,
never again after a physical repair. A physical repair deliberately trims
one segment's edge -- Boundary's own authorized territory, identical in
kind to `render.tighten_trailing_silence`'s existing silence trim -- and
re-running a full-coverage check against an intentionally-shortened segment
would misreport a real Boundary repair as a semantic mismatch, defeating
the physical repair loop entirely. (Found and fixed via this module's own
integration tests, not by inspection -- see
`tests/test_cutsell_live_render_qc.py`.)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from typing import Sequence

from .canonical_edit_plan import CanonicalEditPlan, build_canonical_edit_plan
from .live_boundary_repair import repair_segment_for_finding, segment_output_windows
from .post_render_media_qc import run_post_render_media_qc
from .post_render_watch_listen_qc import (
    PostRenderFinding,
    PostRenderQCResult,
    check_no_duplicate_render_segments,
    check_render_plan_covers_edit_plan,
    check_render_sequence_matches_edit_plan,
    is_physical_finding_kind,
)
from .render import render_preview
from .render_plan import RenderSegment

DEFAULT_MAX_RENDER_ATTEMPTS = 3


class PostRenderQCFailure(RuntimeError):
    """The rendered candidate did not pass PostRenderWatchListenQC and must
    NOT be delivered/uploaded. Carries the full `LiveRenderQCResult` for
    observability -- callers should surface `result` rather than re-deriving
    the failure reason from the exception message alone."""

    def __init__(self, result: "LiveRenderQCResult"):
        super().__init__(f"post_render_watch_listen_qc_failed:{result.status}")
        self.result = result


@dataclass(frozen=True)
class RenderAttemptRecord:
    render_attempt: int
    plan_id: str
    plan_version: int
    semantic_hash: str
    input_boundary_state: tuple[dict, ...]
    findings: tuple[dict, ...]
    finding_types: tuple[str, ...]  # "physical" | "semantic_structural", one per finding
    repair_requested: bool
    repair_applied: dict | None
    status: str  # "PASS" | "PHYSICAL_FAIL_REPAIRED" | "PHYSICAL_FAIL_UNREPAIRABLE" | "SEMANTIC_MISMATCH"


@dataclass(frozen=True)
class LiveRenderQCResult:
    status: str  # "PASS" | "NEEDS_HUMAN_REVIEW" | "SEMANTIC_MISMATCH_INVALIDATED"
    output_path: str | None
    plan_id: str
    plan_version: int
    semantic_hash: str
    attempts: tuple[RenderAttemptRecord, ...]


def _segment_state(segments: Sequence[RenderSegment]) -> tuple[dict, ...]:
    return tuple({"clip_id": s.clip_id, "start": s.start, "end": s.end} for s in segments)


def _finding_dict(finding: PostRenderFinding) -> dict:
    return dataclasses.asdict(finding)


def _merge_qc_results(*results: PostRenderQCResult) -> PostRenderQCResult:
    findings = tuple(f for result in results for f in result.findings)
    status = "FAIL" if findings else "PASS"
    return PostRenderQCResult(status=status, findings=findings)


def _resolve_edit_plan(draft) -> CanonicalEditPlan:
    """A fresh, structurally-correct CanonicalEditPlan built from the exact
    same draft the render segments came from -- but its plan_version is
    overridden from `draft.diagnostics["canonical_edit_plan"]` when present,
    so this preserves the TRUE repair-loop-derived version history for
    observability rather than silently resetting it to 1. `plan_id`/
    `semantic_hash` are derived identically either way (both are pure
    functions of `draft.selected`'s text content), so this never disagrees
    with the plan FinalEditReviewer actually reviewed and Freeze recorded."""
    edit_plan = build_canonical_edit_plan(draft)
    stored = (draft.diagnostics or {}).get("canonical_edit_plan") or {}
    plan_id = str(stored.get("plan_id") or edit_plan.plan_id)
    plan_version = int(stored.get("plan_version") or edit_plan.plan_version)
    return replace(edit_plan, plan_id=plan_id, plan_version=plan_version)


def render_with_post_render_qc(
    draft,
    segments: Sequence[RenderSegment],
    output_path: str,
    *,
    text_overlays=(),
    media_overlays=(),
    max_attempts: int = DEFAULT_MAX_RENDER_ATTEMPTS,
    protected_pause_windows: Sequence[tuple[float, float]] = (),
) -> LiveRenderQCResult:
    """Render `segments` to `output_path` and run the full live
    PostRenderWatchListenQC + bounded physical repair loop against that
    ACTUAL local file before any caller uploads/packages it. See module
    docstring for the exact required order and authority rule."""
    edit_plan = _resolve_edit_plan(draft)
    current_segments = tuple(segments)
    attempts: list[RenderAttemptRecord] = []

    for attempt_index in range(max_attempts):
        render_preview(current_segments, output_path, text_overlays=text_overlays, media_overlays=media_overlays)

        # Structural checks validate that THIS segment SET correctly
        # represents the frozen CanonicalEditPlan's clip membership and
        # order -- a question settled once, against the plan's own frozen
        # segments, before any physical repair. A physical repair
        # deliberately trims one segment's edge (Boundary's own authorized
        # territory, identical in kind to `tighten_trailing_silence`'s
        # existing silence trim) -- re-running a full-coverage check against
        # an intentionally-trimmed segment would misreport a real Boundary
        # repair as a semantic mismatch and defeat the physical repair loop
        # entirely. So this only runs on the FIRST attempt, before any
        # repair has touched `current_segments`.
        if attempt_index == 0:
            structural = _merge_qc_results(
                check_render_plan_covers_edit_plan(current_segments, edit_plan),
                check_render_sequence_matches_edit_plan(current_segments, edit_plan),
                check_no_duplicate_render_segments(current_segments),
            )
            if structural.status == "FAIL":
                attempts.append(RenderAttemptRecord(
                    render_attempt=attempt_index + 1,
                    plan_id=edit_plan.plan_id, plan_version=edit_plan.plan_version, semantic_hash=edit_plan.semantic_hash,
                    input_boundary_state=_segment_state(current_segments),
                    findings=tuple(_finding_dict(f) for f in structural.findings),
                    finding_types=("semantic_structural",) * len(structural.findings),
                    repair_requested=False, repair_applied=None, status="SEMANTIC_MISMATCH",
                ))
                return LiveRenderQCResult(
                    status="SEMANTIC_MISMATCH_INVALIDATED", output_path=None,
                    plan_id=edit_plan.plan_id, plan_version=edit_plan.plan_version, semantic_hash=edit_plan.semantic_hash,
                    attempts=tuple(attempts),
                )

        boundary_timestamps = [w[1] for w in segment_output_windows(current_segments)[:-1]]
        media = run_post_render_media_qc(
            output_path,
            boundary_timestamps=boundary_timestamps,
            protected_pause_windows=protected_pause_windows,
        )

        if media.status == "PASS":
            attempts.append(RenderAttemptRecord(
                render_attempt=attempt_index + 1,
                plan_id=edit_plan.plan_id, plan_version=edit_plan.plan_version, semantic_hash=edit_plan.semantic_hash,
                input_boundary_state=_segment_state(current_segments),
                findings=(), finding_types=(), repair_requested=False, repair_applied=None, status="PASS",
            ))
            return LiveRenderQCResult(
                status="PASS", output_path=output_path,
                plan_id=edit_plan.plan_id, plan_version=edit_plan.plan_version, semantic_hash=edit_plan.semantic_hash,
                attempts=tuple(attempts),
            )

        non_physical = [f for f in media.findings if not is_physical_finding_kind(f.kind)]
        if non_physical:
            attempts.append(RenderAttemptRecord(
                render_attempt=attempt_index + 1,
                plan_id=edit_plan.plan_id, plan_version=edit_plan.plan_version, semantic_hash=edit_plan.semantic_hash,
                input_boundary_state=_segment_state(current_segments),
                findings=tuple(_finding_dict(f) for f in media.findings),
                finding_types=tuple(
                    "physical" if is_physical_finding_kind(f.kind) else "semantic_structural" for f in media.findings
                ),
                repair_requested=False, repair_applied=None, status="SEMANTIC_MISMATCH",
            ))
            return LiveRenderQCResult(
                status="SEMANTIC_MISMATCH_INVALIDATED", output_path=None,
                plan_id=edit_plan.plan_id, plan_version=edit_plan.plan_version, semantic_hash=edit_plan.semantic_hash,
                attempts=tuple(attempts),
            )

        # Every remaining finding is physical -- attempt ONE targeted,
        # Boundary-only repair for the first, re-render, and re-check.
        finding = media.findings[0]
        repair = repair_segment_for_finding(current_segments, finding)
        if repair is None:
            attempts.append(RenderAttemptRecord(
                render_attempt=attempt_index + 1,
                plan_id=edit_plan.plan_id, plan_version=edit_plan.plan_version, semantic_hash=edit_plan.semantic_hash,
                input_boundary_state=_segment_state(current_segments),
                findings=tuple(_finding_dict(f) for f in media.findings),
                finding_types=("physical",) * len(media.findings),
                repair_requested=True, repair_applied=None, status="PHYSICAL_FAIL_UNREPAIRABLE",
            ))
            break

        new_segments, repair_attempt = repair
        attempts.append(RenderAttemptRecord(
            render_attempt=attempt_index + 1,
            plan_id=edit_plan.plan_id, plan_version=edit_plan.plan_version, semantic_hash=edit_plan.semantic_hash,
            input_boundary_state=_segment_state(current_segments),
            findings=tuple(_finding_dict(f) for f in media.findings),
            finding_types=("physical",) * len(media.findings),
            repair_requested=True, repair_applied=dataclasses.asdict(repair_attempt), status="PHYSICAL_FAIL_REPAIRED",
        ))
        current_segments = new_segments

    return LiveRenderQCResult(
        status="NEEDS_HUMAN_REVIEW", output_path=None,
        plan_id=edit_plan.plan_id, plan_version=edit_plan.plan_version, semantic_hash=edit_plan.semantic_hash,
        attempts=tuple(attempts),
    )
