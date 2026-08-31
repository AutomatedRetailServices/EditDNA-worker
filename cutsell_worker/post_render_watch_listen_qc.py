"""PostRenderWatchListenQC -- durable interface contract only (D-024).

Clean Cut Core V1 does not implement or activate this stage. This module
defines the SHAPE a future physical/output QA pass must have so it can be
added later without another architecture reset -- see the canonical
directive's "POST-RENDER WATCH + LISTEN QC -- INTERFACE NOW" section.

Scope, once implemented: physical/output QA AFTER render --
clipped words, unsafe word boundaries, awkward physical cuts, lingering
accidental silence, body/mic/camera reset debris, awkward post-line
expression, A/V synchronization, framing integrity, decode/export
integrity. It must NEVER change semantic membership. A finding whose root
cause is semantic (e.g. a clipped word reveals the wrong take was selected)
routes back upstream to the owning canonical authority; a finding that is
purely physical/timing (e.g. a cut lands 40ms into a word) routes to
BoundaryEngine. Nothing in this module performs that upstream/Boundary
routing itself -- it only defines the result shape a caller would act on.

No provider is implemented. `PostRenderWatchListenQCProvider` is a Protocol
so a future implementation can be type-checked against this contract
without this module depending on it.

## What IS implemented (D-025): the structural cross-check

`check_render_plan_covers_edit_plan` is a real, deterministic check --
not a stub -- but it is deliberately scoped to the RENDER PLAN artifact
(`render_plan.build_render_plan`'s `RenderSegment` list), not the decoded
MP4 bytes. This sandbox cannot reach the actual rendered file (the
human-review artifact ZIP is unreachable -- Azure Blob egress is blocked by
org policy, confirmed repeatedly this session), so a true decode-level
check (does the exported file's audio/video actually contain what the plan
says, frame-accurate) is not something to fabricate without real signal
processing. What IS checkable without that: does the deterministic render
plan (computed from the SAME frozen draft) actually cover every clip
CanonicalEditPlan says survived to KEEP? This catches a distinct failure
mode from `selection_boundary_contract`'s existing token-stream hash check
(text-based -- would not notice a segment with correct text but wrong/
truncated time boundaries): a `build_render_plan`/coalescing bug that
silently drops or shrinks a segment's actual time range would still pass
the token-stream check (the TEXT is unaffected) but would ship a rendered
video missing real audio for that clip.

All the genuinely perceptual checks (clipped phonemes, fumble frames,
framing, A/V drift, decode/export integrity against the real file) remain
unimplemented, per the module-level docstring above -- this section does
not change that.

## What IS now implemented (D-028): real ffmpeg/ffprobe media checks

`post_render_media_qc.py` (a separate module, so this one stays free of a
subprocess/ffmpeg dependency at import time) implements DECODE_EXPORT_
INTEGRITY, LINGERING_ACCIDENTAL_SILENCE, FROZEN_OR_REPEATED_FRAME,
DEAD_BLACK_FRAME, and ABRUPT_AUDIO_DISCONTINUITY for real against actual
decoded/probed media (ffmpeg/ffprobe confirmed installed this cycle) --
never transcript-only. See that module's own docstring for exactly what
remains an honest, unbuilt gap (phoneme-level truncation, breath-cut
detection, body/mic/camera reset debris, facial expression, face/body jump,
fine A/V drift) and why.

## What IS now active in the live pipeline (D-030)

`check_render_sequence_matches_edit_plan` (added this cycle) plus
`check_render_plan_covers_edit_plan` and `check_no_duplicate_render_segments`
above are all now actually invoked -- not just tested in isolation -- by
`live_render_qc.render_with_post_render_qc`, which `export_job.py`'s real
export job calls in place of a bare `render.render_preview(...)`. See
`live_render_qc.py`'s own docstring for the full live execution order,
the bounded physical repair loop, and the authority rule that keeps
`BoundaryEngine` from ever touching a semantic/structural mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .canonical_edit_plan import CanonicalEditPlan

STRUCTURAL_SEGMENT_MISSING = "STRUCTURAL_SEGMENT_MISSING"
STRUCTURAL_SEGMENT_TRUNCATED = "STRUCTURAL_SEGMENT_TRUNCATED"
STRUCTURAL_DUPLICATE_SEGMENT = "STRUCTURAL_DUPLICATE_SEGMENT"
STRUCTURAL_SEQUENCE_MISMATCH = "STRUCTURAL_SEQUENCE_MISMATCH"

CLIPPED_WORD = "CLIPPED_WORD"
UNSAFE_WORD_BOUNDARY = "UNSAFE_WORD_BOUNDARY"
AWKWARD_PHYSICAL_CUT = "AWKWARD_PHYSICAL_CUT"
LINGERING_ACCIDENTAL_SILENCE = "LINGERING_ACCIDENTAL_SILENCE"
RESET_DEBRIS = "RESET_DEBRIS"
AWKWARD_POST_LINE_EXPRESSION = "AWKWARD_POST_LINE_EXPRESSION"
AV_SYNC_DRIFT = "AV_SYNC_DRIFT"
FRAMING_INTEGRITY = "FRAMING_INTEGRITY"
DECODE_EXPORT_INTEGRITY = "DECODE_EXPORT_INTEGRITY"
# D-028: real ffmpeg/ffprobe-backed additions -- see post_render_media_qc.py.
FROZEN_OR_REPEATED_FRAME = "FROZEN_OR_REPEATED_FRAME"
DEAD_BLACK_FRAME = "DEAD_BLACK_FRAME"
ABRUPT_AUDIO_DISCONTINUITY = "ABRUPT_AUDIO_DISCONTINUITY"

_PHYSICAL_KINDS = frozenset({
    CLIPPED_WORD, UNSAFE_WORD_BOUNDARY, AWKWARD_PHYSICAL_CUT,
    LINGERING_ACCIDENTAL_SILENCE, RESET_DEBRIS, AWKWARD_POST_LINE_EXPRESSION,
    AV_SYNC_DRIFT, FRAMING_INTEGRITY, DECODE_EXPORT_INTEGRITY,
    FROZEN_OR_REPEATED_FRAME, DEAD_BLACK_FRAME, ABRUPT_AUDIO_DISCONTINUITY,
})


def is_physical_finding_kind(kind: str) -> bool:
    """True for a purely physical/timing finding kind -- one BoundaryEngine
    (plus a re-render) could plausibly fix, never a semantic-membership
    finding. Exposed so a repair/routing loop outside this module (see
    `post_render_media_qc.run_bounded_physical_repair_loop`) can assert it
    is never asked to "fix" a semantic finding."""
    return kind in _PHYSICAL_KINDS


@dataclass(frozen=True)
class PostRenderFinding:
    kind: str
    start: float
    end: float
    detail: dict
    # "BoundaryEngine" for a purely physical/timing finding; the name of the
    # owning canonical authority (e.g. "StoryValidator") when the physical
    # symptom's real cause is upstream and semantic.
    routes_to: str


@dataclass(frozen=True)
class PostRenderQCResult:
    status: str  # "PASS" | "FAIL"
    findings: tuple[PostRenderFinding, ...]


class PostRenderWatchListenQCProvider(Protocol):
    """Contract a future concrete provider must satisfy. Not implemented in
    Clean Cut Core V1 -- no provider is constructed or invoked anywhere in
    the active pipeline."""

    def review(self, rendered_video_path: str, edit_plan: CanonicalEditPlan) -> PostRenderQCResult:
        ...


def check_render_plan_covers_edit_plan(
    render_segments: Iterable, edit_plan: CanonicalEditPlan,
) -> PostRenderQCResult:
    """Verify every clip in ``edit_plan.keep_sequence`` is fully covered by
    some segment in the render plan, in the same source. See module
    docstring for exactly what this does and does not check.

    ``render_segments`` is ``render_plan.build_render_plan(...)``'s return
    value (or anything duck-typed the same way: ``clip_id``,
    ``source_asset_id``, ``start``, ``end``) -- not imported by name here to
    avoid this module depending on the render pipeline at import time.
    """
    segments = list(render_segments)
    findings: list[PostRenderFinding] = []
    for clip in edit_plan.keep_sequence:
        covering = [
            seg for seg in segments
            if seg.source_asset_id == clip.source_asset_id
            and float(seg.start) <= float(clip.start)
            and float(clip.end) <= float(seg.end)
        ]
        if not covering:
            # Distinguish "vanished entirely" from "shrunk but partially
            # present" -- the latter still has SOME overlapping segment,
            # just not one that fully contains the clip's range.
            overlapping = [
                seg for seg in segments
                if seg.source_asset_id == clip.source_asset_id
                and float(seg.start) < float(clip.end)
                and float(clip.start) < float(seg.end)
            ]
            kind = STRUCTURAL_SEGMENT_TRUNCATED if overlapping else STRUCTURAL_SEGMENT_MISSING
            findings.append(PostRenderFinding(
                kind=kind,
                start=clip.start,
                end=clip.end,
                detail={
                    "clip_id": clip.clip_id,
                    "idea_id": clip.idea_id,
                    "is_composite_piece": clip.is_composite_piece,
                },
                routes_to="SelectionFreeze",
            ))
    status = "FAIL" if findings else "PASS"
    return PostRenderQCResult(status=status, findings=tuple(findings))


def check_render_sequence_matches_edit_plan(
    render_segments: Iterable, edit_plan: CanonicalEditPlan,
) -> PostRenderQCResult:
    """Verify the render segments' clip_id order is consistent with
    `edit_plan.keep_sequence`'s order -- "rendered spoken sequence matches
    the frozen CanonicalEditPlan" / "semantic order unchanged" (D-030).

    `render_plan._coalesce_contiguous_segments` merges an adjacent same-
    source run into one segment and keeps only its FIRST clip_id, so a
    legitimate render's segment list is not clip_id-for-clip_id identical to
    `keep_sequence` -- it is `keep_sequence`'s clip_id order with any
    coalesced-away ids dropped. This check reduces the expected order to
    exactly the ids that DO appear in the render (preserving relative order)
    and compares that against the render's own order; anything else -- a
    clip rendered out of its frozen order, or one that appears in the render
    but was never in `keep_sequence` at all -- is a real mismatch.
    """
    rendered_order = [seg.clip_id for seg in render_segments]
    expected_ids = {clip.clip_id for clip in edit_plan.keep_sequence}
    expected_order_filtered = [
        clip.clip_id for clip in edit_plan.keep_sequence if clip.clip_id in set(rendered_order)
    ]
    unexpected = [cid for cid in rendered_order if cid not in expected_ids]
    if not unexpected and rendered_order == expected_order_filtered:
        return PostRenderQCResult(status="PASS", findings=())
    finding = PostRenderFinding(
        kind=STRUCTURAL_SEQUENCE_MISMATCH,
        start=0.0, end=0.0,
        detail={
            "rendered_order": rendered_order,
            "expected_order": expected_order_filtered,
            "unexpected_clip_ids": unexpected,
        },
        routes_to="SelectionFreeze",
    )
    return PostRenderQCResult(status="FAIL", findings=(finding,))


def check_no_duplicate_render_segments(render_segments: Iterable) -> PostRenderQCResult:
    """Detect a clip_id appearing in more than one render segment -- the
    same clip rendered twice into the final output.

    `render_plan._coalesce_contiguous_segments` merges two ADJACENT
    same-source, same-settings segments into one, keeping only the FIRST's
    clip_id -- a legitimate render never produces two segments sharing one
    clip_id. Seeing that is a real render-plan bug (the "no duplicate
    rendered segment" requirement), not a coalescing artifact.
    """
    segments = list(render_segments)
    occurrences: dict[str, int] = {}
    for seg in segments:
        occurrences[seg.clip_id] = occurrences.get(seg.clip_id, 0) + 1

    findings: list[PostRenderFinding] = []
    reported: set[str] = set()
    for seg in segments:
        count = occurrences.get(seg.clip_id, 0)
        if count > 1 and seg.clip_id not in reported:
            reported.add(seg.clip_id)
            findings.append(PostRenderFinding(
                kind=STRUCTURAL_DUPLICATE_SEGMENT,
                start=float(seg.start),
                end=float(seg.end),
                detail={"clip_id": seg.clip_id, "occurrence_count": count},
                routes_to="SelectionFreeze",
            ))
    status = "FAIL" if findings else "PASS"
    return PostRenderQCResult(status=status, findings=tuple(findings))
