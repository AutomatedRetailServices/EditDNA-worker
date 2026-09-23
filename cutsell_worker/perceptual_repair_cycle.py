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

## STILL DISCONNECTED (D-288, second-pass correction) -- extra
## preconditions above and beyond `repair_segment_for_finding` itself

A real-code review of `live_boundary_repair.repair_segment_for_finding`
found it does NOT itself check word boundaries, and at a SHARED CUT
(two adjacent segments whose windows touch, e.g. segment i's trailing
edge at the same instant as segment i+1's leading edge) its "first match
wins" loop can attribute a finding to the WRONG segment -- one whose own
edge merely happens to fall within `_EDGE_TOLERANCE_SEC` of the finding,
not the segment the defect actually belongs to. Its leading-edge repair
branch in particular has NO measured-silence (let alone word-level)
floor at all -- only `tighten_trailing_silence`'s trailing-edge branch
does. This module is therefore kept disconnected from every live caller
until these three additional, INDEPENDENT preconditions all clear, none
of them delegated to `repair_segment_for_finding`'s own internal checks:

1. **Exact, unambiguous segment/edge identity** (`_disambiguate_target`):
   re-derives every segment/edge whose window falls within the SAME
   tolerance `repair_segment_for_finding` itself uses. Exactly one match
   is required; a shared-cut ambiguity (more than one plausible match)
   or no match at all refuses the repair outright -- this module never
   lets `repair_segment_for_finding`'s own "first in list wins" behavior
   silently pick a neighboring clip.
2. **A confirmed, corroborated defect** (`_is_confirmed_repairable_
   defect`): an explicit allowlist of finding kinds (never "any FAIL"),
   each independently re-verified against its own `detail` -- a reset/
   break "candidate" motion event without a real measured pause nearby,
   or a dead-air interval shorter than the capability's own FAIL floor,
   is refused even if `severity` claims FAIL. A generic 0.35s edge-
   routing window is a ROUTING width, never repair authorization on its
   own.
3. **Protected speech boundaries, independently verified**
   (`_word_floor_respects_repair`): the caller MUST supply real word-
   timing evidence (`word_floor_by_clip_id`) for the exact segment/clip a
   repair would touch; this module then checks the CANDIDATE repair's
   own computed boundary against it. No evidence supplied for that
   clip_id -> refuse, fail closed -- this module never trusts `repair_
   segment_for_finding`'s internal (trailing-edge-only, silence-based,
   not word-based) protection as sufficient on its own.

Live-wiring this into `export_job.py`/`universal_clean_cut_validation.py`
remains a SEPARATE, not-yet-authorized gate even after these preconditions
-- see the D-288 decision log entry.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

from .canonical_edit_plan import AuthoritativePlanSource, CanonicalEditPlan, build_canonical_edit_plan
from .live_boundary_repair import _EDGE_TOLERANCE_SEC, repair_segment_for_finding
from .perceptual_watch_listen import (
    DEAD_AIR_FAIL_SEC,
    INTERIOR_DEAD_AIR,
    RESET_DEBRIS_AT_EDGE,
    ROUTE_BOUNDARY,
    WATCH_LISTEN_BLOCKED,
    WATCH_LISTEN_HUMAN_REVIEW_REQUIRED,
    WATCH_LISTEN_SYSTEM_PASS,
    PerceptualFinding,
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

# D-288 (finding 3): the explicit allowlist of physical finding kinds this
# module will ever consider repairable, each with its OWN independent
# corroboration check against `finding.detail` -- `severity == "FAIL"`
# alone is never sufficient. `SPEECH_ENERGY_AT_CUT` is deliberately
# absent: `_speech_energy_at_cuts` never emits it above UNCERTAIN, so it
# can never legitimately reach this allowlist at all; if it ever did
# (a defect elsewhere), it is correctly refused here as not allowlisted.
def _dead_air_is_corroborated(finding: PerceptualFinding) -> bool:
    return float(finding.detail.get("duration_sec") or 0.0) >= DEAD_AIR_FAIL_SEC


def _reset_debris_is_corroborated(finding: PerceptualFinding) -> bool:
    # D-149: a reset/break "candidate" motion event is a kinematic
    # magnitude, not a genuine-reset probability -- only a candidate
    # co-occurring with a REAL measured pause is trustworthy; a bare
    # generic-window match within EDGE_DEBRIS_WINDOW_SEC is never itself
    # authorization (see this module's own docstring).
    return finding.detail.get("measured_pause_nearby") is True


_CONFIRMED_REPAIRABLE_KINDS: dict[str, Callable[[PerceptualFinding], bool]] = {
    INTERIOR_DEAD_AIR: _dead_air_is_corroborated,
    RESET_DEBRIS_AT_EDGE: _reset_debris_is_corroborated,
}


def _is_confirmed_repairable_defect(finding: PerceptualFinding) -> bool:
    if finding.severity != "FAIL":
        return False
    if not (finding.start < finding.end):
        return False  # malformed coordinates -- never trust them
    check = _CONFIRMED_REPAIRABLE_KINDS.get(finding.kind)
    return bool(check) and check(finding)


def _disambiguate_target(
    segments: Sequence, windows: Sequence[tuple[float, float]], finding: PerceptualFinding,
) -> tuple[int, str] | None:
    """D-288 (finding 3): every (segment_index, edge) pair whose window
    edge falls within the SAME tolerance `repair_segment_for_finding`
    itself uses, re-derived independently here (never imported from that
    function's own internals, so this check cannot silently drift out of
    sync only by accident -- see the module's own test suite for a
    permutation proving the two must agree). Returns the SINGLE match, or
    `None` when there is no match or more than one (a shared-cut
    boundary ambiguity) -- never guesses which one `repair_segment_for_
    finding` would pick first."""
    finding_start, finding_end = float(finding.start), float(finding.end)
    matches: list[tuple[int, str]] = []
    for index, (win_start, win_end) in enumerate(windows):
        if finding_start < win_start - _EDGE_TOLERANCE_SEC or finding_start > win_end + _EDGE_TOLERANCE_SEC:
            continue
        near_trailing_edge = finding_start >= win_start - 1e-6 and abs(finding_end - win_end) <= _EDGE_TOLERANCE_SEC
        near_leading_edge = finding_end <= win_end + 1e-6 and abs(finding_start - win_start) <= _EDGE_TOLERANCE_SEC
        if near_trailing_edge:
            matches.append((index, "trailing"))
        if near_leading_edge:
            matches.append((index, "leading"))
    if len(matches) != 1:
        return None
    return matches[0]


def _word_floor_respects_repair(repair_attempt, word_floor: tuple[float, float] | None) -> bool:
    """D-288 (finding 3): independently re-verifies the CANDIDATE repair
    `repair_segment_for_finding` already computed against real word-
    timing evidence the caller supplied for that exact clip -- never
    trusts that function's own (trailing-edge-only, silence-based, not
    word-based) protection as sufficient by itself. `word_floor is None`
    (no evidence supplied for this clip_id) fails closed: refused, not
    silently allowed through."""
    if word_floor is None:
        return False
    first_word_start, last_word_end = word_floor
    if repair_attempt.edge == "trailing":
        return repair_attempt.repaired_end >= last_word_end - 1e-6
    return repair_attempt.repaired_start <= first_word_start + 1e-6


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
    word_floor_by_clip_id: Mapping[str, tuple[float, float]],
    segments_as_rendered: Callable[[Sequence, Sequence[dict]], tuple[Sequence, int]],
    max_attempts: int = DEFAULT_MAX_PERCEPTUAL_REPAIR_ATTEMPTS,
) -> PerceptualRepairCycleResult:
    """Run the bounded perceptual repair cycle against an ALREADY
    technically-PASSing rendered candidate. See module docstring for the
    exact authority/safety contract -- including the D-288 second-pass
    preconditions (`word_floor_by_clip_id` required, no default) this
    function now enforces before any repair is ever applied.

    Dependency-injected (never imports `render`/`live_render_qc`/
    `perceptual_watch_listen.review_rendered_candidate` directly): the
    caller supplies the exact same render+technical-QC callable and
    perceptual-review callable it already uses elsewhere, so this module
    can never silently diverge into a second render/review implementation
    -- and so its own tests never need a real ffmpeg/decode pipeline.
    `render_and_technical_qc(draft, segments, output_path)` must return an
    object with `.status` ("PASS" or not), `.output_path`, and `.attempts`
    (each carrying `.renderer_trailing_trims`), matching `live_render_qc.
    LiveRenderQCResult`'s own shape -- `segments_as_rendered` (the SAME
    function `universal_clean_cut_validation.py` already uses, injected
    here rather than imported, to keep this module's own import surface
    light) is applied to the repaired segments after EVERY internal
    render, using that render's own renderer-side trims, before the next
    re-review -- finding 4: never review a new file with stale, pre-
    render timings, and never treat a freshly overwritten file as if a
    previous review still describes it (each loop iteration re-reviews
    the CURRENT file against the CURRENT segments, nothing cached).

    `word_floor_by_clip_id` maps a segment's `clip_id` to that clip's own
    `(first_aligned_word_start, last_aligned_word_end)` in SOURCE time
    (the same coordinate system as `RenderSegment.start`/`.end`). A
    segment whose `clip_id` is absent from this mapping can never be
    repaired -- this module refuses rather than trusting `repair_segment_
    for_finding`'s own internal protection (trailing-edge-only, silence-
    based) as sufficient by itself (finding 3)."""
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
        # (allowlisted kind + corroborated, D-288 finding 3) FAIL finding
        # has a safe automatic repair path here -- `severity == "FAIL"`
        # alone is never sufficient.
        fail_findings = [f for f in review.findings if f.severity == "FAIL"]
        physical_fail = next(
            (f for f in fail_findings if f.routes_to == ROUTE_BOUNDARY and _is_confirmed_repairable_defect(f)),
            None,
        )
        unconfirmed_boundary_fail = next(
            (f for f in fail_findings if f.routes_to == ROUTE_BOUNDARY and not _is_confirmed_repairable_defect(f)),
            None,
        )
        semantic_fail = next((f for f in fail_findings if f.routes_to != ROUTE_BOUNDARY), None)

        if physical_fail is None:
            if unconfirmed_boundary_fail is not None:
                # A BoundaryEngine-routed FAIL exists but did not clear
                # the allowlist/corroboration check -- e.g. a "candidate"
                # motion event with no measured pause nearby. Never
                # authorizes a repair on a generic routing window alone.
                attempts.append(PerceptualRepairAttempt(
                    attempt=attempt_index, finding_kind=unconfirmed_boundary_fail.kind,
                    routes_to=unconfirmed_boundary_fail.routes_to,
                    repair_kind=REPAIR_KIND_PHYSICAL_BOUNDARY, repaired=False,
                    reason="finding_not_confirmed_repairable_defect",
                ))
                status = STATUS_BLOCKED
            elif semantic_fail is not None:
                attempts.append(PerceptualRepairAttempt(
                    attempt=attempt_index, finding_kind=semantic_fail.kind, routes_to=semantic_fail.routes_to,
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

        # D-288 finding 3, precondition 1: exact, unambiguous segment/edge
        # identity -- re-derived independently, never trusting `repair_
        # segment_for_finding`'s own "first match wins" loop at a shared
        # cut boundary.
        target = _disambiguate_target(current_segments, windows, physical_fail)
        if target is None:
            attempts.append(PerceptualRepairAttempt(
                attempt=attempt_index, finding_kind=physical_fail.kind, routes_to=physical_fail.routes_to,
                repair_kind=REPAIR_KIND_PHYSICAL_BOUNDARY, repaired=False,
                reason="ambiguous_or_no_matching_segment_edge",
            ))
            return PerceptualRepairCycleResult(
                status=STATUS_BLOCKED, output_path=current_output_path,
                final_segments=current_segments, final_review=review.as_dict(),
                attempts=tuple(attempts),
            )
        target_index, _target_edge = target
        target_clip_id = current_segments[target_index].clip_id

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
        if repair_attempt.clip_id != target_clip_id:
            # `repair_segment_for_finding` disagreed with our own
            # independently-derived target -- never trust either guess
            # over the other; refuse rather than silently pick one.
            attempts.append(PerceptualRepairAttempt(
                attempt=attempt_index, finding_kind=physical_fail.kind, routes_to=physical_fail.routes_to,
                repair_kind=REPAIR_KIND_PHYSICAL_BOUNDARY, repaired=False,
                reason="repair_target_disagrees_with_disambiguated_segment",
                repair_detail=dataclasses.asdict(repair_attempt),
            ))
            return PerceptualRepairCycleResult(
                status=STATUS_BLOCKED, output_path=current_output_path,
                final_segments=current_segments, final_review=review.as_dict(),
                attempts=tuple(attempts),
            )

        # D-288 finding 3, precondition 3: protected speech boundaries,
        # independently re-verified against caller-supplied real word
        # timing -- never trusted to `repair_segment_for_finding`'s own
        # (trailing-edge-only, silence-based) internal protection alone.
        word_floor = word_floor_by_clip_id.get(target_clip_id)
        if not _word_floor_respects_repair(repair_attempt, word_floor):
            attempts.append(PerceptualRepairAttempt(
                attempt=attempt_index, finding_kind=physical_fail.kind, routes_to=physical_fail.routes_to,
                repair_kind=REPAIR_KIND_PHYSICAL_BOUNDARY, repaired=False,
                reason=(
                    "no_word_floor_evidence_supplied" if word_floor is None
                    else "repair_would_cross_protected_speech_boundary"
                ),
                repair_detail=dataclasses.asdict(repair_attempt),
            ))
            return PerceptualRepairCycleResult(
                status=STATUS_BLOCKED, output_path=current_output_path,
                final_segments=current_segments, final_review=review.as_dict(),
                attempts=tuple(attempts),
            )

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
        # D-288 finding 4: use the segments AS ACTUALLY RENDERED -- the
        # renderer's own additional trims (e.g. trailing-silence
        # tightening beyond what this repair explicitly requested),
        # recorded on the fresh qc_result's own last attempt, are folded
        # in BEFORE the next loop iteration re-reviews. Never re-review
        # the new file against the pre-render repair-only timings.
        last_attempt = qc_result.attempts[-1] if getattr(qc_result, "attempts", None) else None
        renderer_trims = getattr(last_attempt, "renderer_trailing_trims", ()) if last_attempt is not None else ()
        rendered_segments, _applied = segments_as_rendered(new_segments, renderer_trims)
        current_segments = tuple(rendered_segments)
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
