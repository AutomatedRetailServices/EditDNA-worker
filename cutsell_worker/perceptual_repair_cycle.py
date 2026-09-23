"""D-288: bounded, safe, POST-RENDER PERCEPTUAL repair cycle.

Audit finding this closes (`docs/CUTSELL_DECISIONS.md` D-288): a perceptual
finding's `routes_to` label (`perceptual_watch_listen.PerceptualFinding.
routes_to`) was, before this module, read in exactly ONE place in the whole
codebase -- a summary counter inside `PerceptualReview.as_dict()`'s
`routing` dict. Nothing consumed it to actually route a finding to a real
repair authority. This module is that real, bounded connection:

    finding -> owning authority -> safe repair -> render -> technical QC
    -> Watch+Listen (re-review)

THREE DISTINCT REPAIR LOOPS -- never conflate them in diagnostics
------------------------------------------------------------------
1. EDITORIAL repair loop, PRE-Freeze (`repair_loop.py`, D-026): fixes a
   FinalEditReviewer finding (STORY_ORDER_BREAK only) before Selection
   Freeze ever runs, on `draft.selected` / CanonicalEditPlan v1/v2/...
   Diagnostics key: `repair_loop`.
2. TECHNICAL repair loop, POST-render (`live_render_qc.py` /
   `live_boundary_repair.py`, D-030): fixes a PHYSICAL PostRenderFinding
   (dead frame, freeze, accidental silence, splice discontinuity) via a
   bounded Boundary-only edge trim, then re-renders and re-QCs. Runs
   BEFORE this module -- by the time this module's caller has a rendered
   candidate to perceptually review at all, that loop already reached
   `LiveRenderQCResult.status == "PASS"`. Diagnostics key: `live_render_qc`.
3. PERCEPTUAL repair cycle, POST-render (THIS module, D-288): consumes
   `perceptual_watch_listen.PerceptualFinding`s -- ADVISORY findings
   computed AFTER loop 2 above already reached PASS. Diagnostics key:
   `perceptual_repair_cycle` (`repair_loop_kind` field on every result,
   always `"perceptual_post_render"`, so a caller reading raw JSON can
   never mistake this for loop 1 or loop 2 above).

Two finding shapes, two different responses -- never guess
------------------------------------------------------------
- `routes_to=BoundaryEngine` (physical: interior dead air, cut-adjacent
  speech energy, reset/break debris at an edge): the SAME kind of physical
  defect the technical loop already knows how to fix. This module reuses
  the EXISTING `live_boundary_repair.repair_segment_for_finding` authority
  UNMODIFIED -- `PerceptualFinding` already carries the `.start`/`.end`/
  `.kind` fields that function reads, so no adapter is needed and no
  second physical-repair implementation is written. Every existing safety
  property of that function is therefore inherited automatically: the
  words-are-the-hard-floor invariant (an edge trim can only ever shrink
  toward a measured silence/defect boundary, never cross into real
  content), the "no safe repair for a mid-segment defect" refusal, and the
  "would eat too much of the real segment" refusal. A repair is applied,
  the candidate is RE-RENDERED and RE-QC'd through the caller-supplied
  technical pipeline (never skipped), and only THEN is the perceptual
  review re-run. Bounded by `max_attempts`; a finding with no safe repair,
  or a repair that itself breaks technical QC, stops the cycle at BLOCKED
  with an explicit, never-silent reason.
- `routes_to=BestTakeResolver` (semantic: REPEATED_AUDIENCE_CONTENT --
  two rendered pieces from DIFFERENT semantic clips saying the same
  thing): NO automatic repair exists here, by deliberate design, exactly
  mirroring `repair_loop.py`'s own honest-scope doctrine for
  DUPLICATE_IDEA/UNRESOLVED_RETRY ("an automatic fix... means the system
  guessing which content is correct... guessing here is a regression in
  editorial judgment, not a repair"). This module NEVER removes or trims
  selected content to resolve one of these -- it reports `NEEDS_SEMANTIC_
  REVIEW` and stops. Should a human/PO-authorized replacement draft later
  be supplied, `build_semantic_repair_plan` below is the CONTRACT for
  returning it to Selection: a new CanonicalEditPlan version (the SAME
  `build_canonical_edit_plan(...)` + `replace(plan_version=plan.plan_
  version + 1)` pattern `repair_loop.py` already established for the
  pre-Freeze loop) that must then pass a fresh Selection Freeze review --
  never a silent post-Freeze content removal.

An `ERROR` capability status (the measurement itself did not run -- e.g. a
decode failure) has no findings to route or repair; it is reported as
BLOCKED with an explicit `error_capability` reason, never silently
skipped and never treated as a repairable physical finding.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from .canonical_edit_plan import AuthoritativePlanSource, CanonicalEditPlan, build_canonical_edit_plan
from .live_boundary_repair import repair_segment_for_finding
from .perceptual_watch_listen import (
    ROUTE_BOUNDARY,
    WATCH_LISTEN_BLOCKED,
    WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
    WATCH_LISTEN_SYSTEM_PASS,
    PerceptualReview,
)

REPAIR_LOOP_KIND_PERCEPTUAL = "perceptual_post_render"

DEFAULT_MAX_PERCEPTUAL_REPAIR_ATTEMPTS = 2

STATUS_PASS = "PASS"
STATUS_BLOCKED = "BLOCKED"
STATUS_NEEDS_SEMANTIC_REVIEW = "NEEDS_SEMANTIC_REVIEW"
STATUS_NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"

REPAIR_KIND_PHYSICAL_BOUNDARY = "physical_boundary_repair"
REPAIR_KIND_NO_AUTOMATIC_SEMANTIC = "no_automatic_semantic_repair"
REPAIR_KIND_ERROR_CAPABILITY = "error_capability_no_repair_target"


@dataclass(frozen=True)
class PerceptualRepairAttempt:
    attempt: int
    finding_kind: str | None
    routes_to: str | None
    repair_kind: str
    repaired: bool
    reason: str
    repair_detail: dict | None = None


@dataclass(frozen=True)
class PerceptualRepairCycleResult:
    status: str  # PASS | BLOCKED | NEEDS_SEMANTIC_REVIEW | NEEDS_HUMAN_REVIEW
    output_path: str | None
    final_segments: tuple
    final_review: dict | None
    attempts: tuple[PerceptualRepairAttempt, ...]
    # D-288: always this exact value -- the one field a caller reading raw
    # diagnostics JSON needs to tell this loop apart from `repair_loop`
    # (pre-Freeze editorial) and `live_render_qc` (technical, post-render).
    repair_loop_kind: str = REPAIR_LOOP_KIND_PERCEPTUAL


def run_perceptual_repair_cycle(
    draft,
    segments: Sequence,
    output_path: str,
    *,
    render_and_technical_qc: Callable[[object, Sequence, str], object],
    perceptual_review: Callable[[str, object, Sequence, Sequence[tuple[float, float]]], PerceptualReview],
    output_windows: Callable[[Sequence], list[tuple[float, float]]],
    max_attempts: int = DEFAULT_MAX_PERCEPTUAL_REPAIR_ATTEMPTS,
) -> PerceptualRepairCycleResult:
    """Run the bounded perceptual repair cycle against an ALREADY
    technically-PASSing rendered candidate. See module docstring for the
    exact authority/safety contract.

    Dependency-injected (never imports `render`/`live_render_qc`/
    `perceptual_watch_listen.review_rendered_candidate` directly): the
    caller supplies the exact same render+technical-QC callable and
    perceptual-review callable it already uses elsewhere, so this module
    can never silently diverge into a second render/review implementation
    -- and so its own tests never need a real ffmpeg/decode pipeline.
    `render_and_technical_qc(draft, segments, output_path)` must return an
    object with `.status` ("PASS" or not) and `.output_path`, matching
    `live_render_qc.LiveRenderQCResult`'s own shape."""
    current_segments = tuple(segments)
    current_output_path = output_path
    attempts: list[PerceptualRepairAttempt] = []

    for attempt_index in range(max_attempts + 1):
        windows = output_windows(current_segments)
        review = perceptual_review(current_output_path, draft, current_segments, windows)
        watch_listen_status = review.watch_listen_status

        if watch_listen_status == WATCH_LISTEN_SYSTEM_PASS:
            return PerceptualRepairCycleResult(
                status=STATUS_PASS, output_path=current_output_path,
                final_segments=current_segments, final_review=review.as_dict(),
                attempts=tuple(attempts),
            )
        if watch_listen_status == WATCH_LISTEN_HUMAN_REVIEW_REQUIRED:
            # UNCERTAIN/NOT_IMPLEMENTED only -- nothing CONFIRMED broken.
            # "No conviertas automáticamente cada FAIL en un recorte": an
            # uncertain finding is never auto-repaired, it goes to human
            # review exactly as perceptual_watch_listen.py's own contract
            # already intends (D-154/D-155).
            return PerceptualRepairCycleResult(
                status=STATUS_NEEDS_HUMAN_REVIEW, output_path=current_output_path,
                final_segments=current_segments, final_review=review.as_dict(),
                attempts=tuple(attempts),
            )

        # WATCH_LISTEN_BLOCKED: at least one EVALUATED_FAIL or ERROR
        # capability. Only a PHYSICAL, BoundaryEngine-routed, CONFIRMED
        # FAIL finding has a safe automatic repair path here.
        fail_findings = [f for f in review.findings if f.severity == "FAIL"]
        physical_fail = next((f for f in fail_findings if f.routes_to == ROUTE_BOUNDARY), None)
        semantic_fail = next((f for f in fail_findings if f.routes_to != ROUTE_BOUNDARY), None)

        if physical_fail is None:
            if fail_findings:
                target = semantic_fail or fail_findings[0]
                attempts.append(PerceptualRepairAttempt(
                    attempt=attempt_index, finding_kind=target.kind, routes_to=target.routes_to,
                    repair_kind=REPAIR_KIND_NO_AUTOMATIC_SEMANTIC, repaired=False,
                    reason="no_automatic_repair_for_semantic_finding",
                ))
                status = STATUS_NEEDS_SEMANTIC_REVIEW
            else:
                # BLOCKED with zero FAIL findings -- an ERROR capability
                # (the measurement itself never ran). No finding exists to
                # route or repair; never fabricate one.
                attempts.append(PerceptualRepairAttempt(
                    attempt=attempt_index, finding_kind=None, routes_to=None,
                    repair_kind=REPAIR_KIND_ERROR_CAPABILITY, repaired=False,
                    reason="error_capability_no_repair_target",
                ))
                status = STATUS_BLOCKED
            return PerceptualRepairCycleResult(
                status=status, output_path=current_output_path,
                final_segments=current_segments, final_review=review.as_dict(),
                attempts=tuple(attempts),
            )

        if attempt_index >= max_attempts:
            attempts.append(PerceptualRepairAttempt(
                attempt=attempt_index, finding_kind=physical_fail.kind, routes_to=physical_fail.routes_to,
                repair_kind=REPAIR_KIND_PHYSICAL_BOUNDARY, repaired=False,
                reason="max_attempts_exhausted",
            ))
            return PerceptualRepairCycleResult(
                status=STATUS_BLOCKED, output_path=current_output_path,
                final_segments=current_segments, final_review=review.as_dict(),
                attempts=tuple(attempts),
            )

        repair = repair_segment_for_finding(current_segments, physical_fail)
        if repair is None:
            attempts.append(PerceptualRepairAttempt(
                attempt=attempt_index, finding_kind=physical_fail.kind, routes_to=physical_fail.routes_to,
                repair_kind=REPAIR_KIND_PHYSICAL_BOUNDARY, repaired=False,
                reason="no_safe_repair_within_hard_floor",
            ))
            return PerceptualRepairCycleResult(
                status=STATUS_BLOCKED, output_path=current_output_path,
                final_segments=current_segments, final_review=review.as_dict(),
                attempts=tuple(attempts),
            )

        new_segments, repair_attempt = repair
        qc_result = render_and_technical_qc(draft, new_segments, current_output_path)
        if getattr(qc_result, "status", None) != "PASS":
            attempts.append(PerceptualRepairAttempt(
                attempt=attempt_index, finding_kind=physical_fail.kind, routes_to=physical_fail.routes_to,
                repair_kind=REPAIR_KIND_PHYSICAL_BOUNDARY, repaired=False,
                reason="repair_broke_technical_qc",
                repair_detail=dataclasses.asdict(repair_attempt),
            ))
            return PerceptualRepairCycleResult(
                status=STATUS_BLOCKED, output_path=current_output_path,
                final_segments=current_segments, final_review=review.as_dict(),
                attempts=tuple(attempts),
            )

        attempts.append(PerceptualRepairAttempt(
            attempt=attempt_index, finding_kind=physical_fail.kind, routes_to=physical_fail.routes_to,
            repair_kind=REPAIR_KIND_PHYSICAL_BOUNDARY, repaired=True,
            reason="boundary_edge_trim_applied_and_reverified",
            repair_detail=dataclasses.asdict(repair_attempt),
        ))
        current_segments = new_segments
        current_output_path = getattr(qc_result, "output_path", None) or current_output_path

    # Unreachable (the loop's own max_attempts+1 range always returns from
    # inside the body above) -- fails closed if it is ever reached anyway.
    return PerceptualRepairCycleResult(
        status=STATUS_BLOCKED, output_path=current_output_path, final_segments=current_segments,
        final_review=None, attempts=tuple(attempts),
    )


def build_semantic_repair_plan(
    repaired_draft,
    previous_plan: CanonicalEditPlan,
    *,
    authoritative_source: AuthoritativePlanSource | None = None,
) -> CanonicalEditPlan:
    """D-288: the CONTRACT for returning a PERCEPTUAL semantic finding
    (`routes_to=BestTakeResolver`) to Selection. This function never
    decides WHICH clip to remove/keep -- `repaired_draft` must already be
    the caller-supplied (human/PO-authorized) corrected draft; this
    function only builds the new, correctly-VERSIONED CanonicalEditPlan
    that corrected draft must pass a fresh Selection Freeze review with,
    reusing the SAME `build_canonical_edit_plan(...)` + `plan_version + 1`
    pattern `repair_loop.py` already established for the pre-Freeze
    editorial loop (never a second plan-versioning scheme). The caller is
    responsible for re-running Selection Freeze / FinalEditReviewer on the
    result before it is ever rendered again -- this function does not do
    that itself, so a caller can never mistake "a new plan object exists"
    for "it was re-frozen"."""
    new_plan = build_canonical_edit_plan(repaired_draft, authoritative_source=authoritative_source)
    return replace(new_plan, plan_version=previous_plan.plan_version + 1)
