"""FinalEditReviewer -- bounded review of the CanonicalEditPlan before
Selection Freeze (D-024).

This is NOT another authority that edits clips directly, and it invents no
new detection heuristic: every finding below is read directly from evidence
`final_story_coherence_validation.py` (StoryValidator) already computed on
the same draft the CanonicalEditPlan was built from. What this module adds
is the standardized PASS/FAIL finding vocabulary the canonical directive
specifies, and an explicit `owning_authority` on every finding so a FAIL
routes back to whichever canonical component is responsible for resolving
it, rather than this reviewer attempting to rewrite membership itself.

Blocking vs. non-blocking: DUPLICATE_IDEA, UNRESOLVED_RETRY,
IDEA_COVERAGE_LOST, CONTRADICTION, and (D-025) STORY_ORDER_BREAK for a
disordered composite always block. UNIQUE_FACT_LOST (D-031, changed from
always-blocking) blocks only when `final_story_coherence_validation`'s own
`lost_semantic_atoms` row says `blocking: true` -- a genuinely critical or
uncertain semantic atom lost, or the broader content-loss signal; a
CONTEXTUAL-only atom loss (e.g. an incidental year -- see
`semantic_atom_importance.py`) is instead a non-blocking warning, exactly
like REQUIRED_CONTINUATION_LOST below (mirrors `possible_missing_story_
ending`'s existing "flag only" contract).

STORY_ORDER_BREAK (D-025, narrow, real detector): when an Idea's
realization is a Composite (2+ component clips), its components must
appear in the final KEEP sequence in the same relative temporal order they
were recorded in -- that is what "natural continuation" (hybrid_composite_
best_take.py's own requirement for accepting a composite in the first
place) means operationally. This is deliberately narrow: it does NOT
attempt general narrative/causal-order checking across INDEPENDENT ideas,
because the Composer stage is explicitly allowed to reorder independent
ideas for pacing/sales logic, and a blanket "KEEP order must match source
order" check would false-positive on that legitimate behavior. Checking
only within one accepted composite's own components avoids that false-
positive risk entirely while still closing a real gap: RAW 33366538992's
own regression harness flagged `pimples_micro_order`/`sonography_good_
before_diagnosis`, which is exactly this failure shape.

CAUSAL_ORDER_BREAK (D-027, general, real detector): the general, cross-idea
complement to STORY_ORDER_BREAK -- see `causal_order_validator.py` for the
full mechanism (source chronology + general connector-language evidence,
with a bounded-arbiter escalation path for ambiguous hits). Unlike
STORY_ORDER_BREAK, this has NO automatic repair in `repair_loop.py`: a
cross-idea reorder risks undoing an intentional Composer pacing choice, so
this blocks Freeze and routes to human review rather than being
auto-corrected.

Honest gap, not fabricated coverage: INCOMPLETE_DELIVERY, ORPHAN_FRAGMENT,
and INCOMPATIBLE_COMPOSITE are recognized in the finding vocabulary (so a
future detector has a defined shape to populate) but have no existing
deterministic detector to draw from, and none is invented here. They are
never emitted. A finding type appearing in `_UNIMPLEMENTED_KINDS` below is
the honest record of that gap -- do not remove it by inventing a heuristic
under time pressure; add a real, tested detector first.
"""
from __future__ import annotations

from dataclasses import dataclass

from .canonical_edit_plan import CanonicalEditPlan
from .causal_order_validator import CausalOrderArbiter, find_causal_order_breaks

# Finding kinds the canonical directive specifies. Every one is a valid
# value for Finding.kind; only the ones NOT in _UNIMPLEMENTED_KINDS are ever
# actually emitted by review() today.
DUPLICATE_IDEA = "DUPLICATE_IDEA"
UNRESOLVED_RETRY = "UNRESOLVED_RETRY"
INCOMPLETE_DELIVERY = "INCOMPLETE_DELIVERY"
ORPHAN_FRAGMENT = "ORPHAN_FRAGMENT"
UNIQUE_FACT_LOST = "UNIQUE_FACT_LOST"
IDEA_COVERAGE_LOST = "IDEA_COVERAGE_LOST"
CONTRADICTION = "CONTRADICTION"
INCOMPATIBLE_COMPOSITE = "INCOMPATIBLE_COMPOSITE"
REQUIRED_CONTINUATION_LOST = "REQUIRED_CONTINUATION_LOST"
STORY_ORDER_BREAK = "STORY_ORDER_BREAK"
CAUSAL_ORDER_BREAK = "CAUSAL_ORDER_BREAK"

_UNIMPLEMENTED_KINDS = frozenset({
    INCOMPLETE_DELIVERY, ORPHAN_FRAGMENT, INCOMPATIBLE_COMPOSITE,
})


@dataclass(frozen=True)
class Finding:
    kind: str
    plan_id: str
    plan_version: int
    idea_id: str | None
    clip_ids: tuple[str, ...]
    detail: dict
    owning_authority: str
    blocking: bool


@dataclass(frozen=True)
class FinalEditReviewResult:
    status: str  # "PASS" | "FAIL"
    findings: tuple[Finding, ...]   # blocking findings
    warnings: tuple[Finding, ...]   # non-blocking findings


def _composite_order_findings(edit_plan: CanonicalEditPlan, plan_id: str, plan_version: int) -> list[Finding]:
    """STORY_ORDER_BREAK: an accepted composite's own components must
    appear in the final KEEP sequence in the same relative temporal order
    they were recorded in. See module docstring for why this stays scoped
    to one composite's own components rather than general cross-idea order."""
    position_by_clip = {clip.clip_id: index for index, clip in enumerate(edit_plan.keep_sequence)}
    start_by_clip = {clip.clip_id: clip.start for clip in edit_plan.keep_sequence}
    findings: list[Finding] = []
    for idea in edit_plan.ideas:
        if not idea.is_composite or len(idea.winning_clip_ids) < 2:
            continue
        present = [cid for cid in idea.winning_clip_ids if cid in position_by_clip]
        if len(present) < 2:
            continue
        by_recording_time = sorted(present, key=lambda cid: start_by_clip[cid])
        by_keep_position = sorted(present, key=lambda cid: position_by_clip[cid])
        if by_recording_time != by_keep_position:
            findings.append(Finding(
                kind=STORY_ORDER_BREAK,
                plan_id=plan_id,
                plan_version=plan_version,
                idea_id=idea.idea_id,
                clip_ids=tuple(present),
                detail={
                    "reason": "composite_components_reordered_relative_to_recording",
                    "recording_order": by_recording_time,
                    "keep_sequence_order": by_keep_position,
                },
                owning_authority="CompositeResolver",
                blocking=True,
            ))
    return findings


def _causal_order_findings(
    edit_plan: CanonicalEditPlan, plan_id: str, plan_version: int, arbiter: CausalOrderArbiter | None,
) -> list[Finding]:
    """CAUSAL_ORDER_BREAK (D-027): general, cross-idea order/dependency
    check -- see causal_order_validator.py for the full mechanism."""
    findings: list[Finding] = []
    for dep in find_causal_order_breaks(edit_plan, arbiter=arbiter):
        findings.append(Finding(
            kind=CAUSAL_ORDER_BREAK,
            plan_id=plan_id,
            plan_version=plan_version,
            idea_id=dep.dependent_idea_id,
            clip_ids=(dep.required_clip_id, dep.dependent_clip_id),
            detail={
                "reason": "dependent_clip_precedes_or_is_missing_its_required_context",
                "required_clip_id": dep.required_clip_id,
                "dependent_clip_id": dep.dependent_clip_id,
                "required_idea_id": dep.required_idea_id,
                "dependent_idea_id": dep.dependent_idea_id,
                "evidence": dep.evidence,
                "confidence": dep.confidence,
                "resolved_by": dep.resolved_by,
            },
            owning_authority="StoryValidator",
            blocking=True,
        ))
    return findings


def review(
    edit_plan: CanonicalEditPlan, *, causal_order_arbiter: CausalOrderArbiter | None = None,
) -> FinalEditReviewResult:
    """Review the CanonicalEditPlan. Never mutates it or the draft it came
    from -- returns PASS or structured findings for the caller to act on
    (universal_clean_cut.py already skips Freeze/Boundary whenever
    freeze_blocked is set, which is exactly the "route back" mechanism for
    every blocking finding here)."""
    plan_id = edit_plan.plan_id
    plan_version = edit_plan.plan_version
    findings: list[Finding] = []
    warnings: list[Finding] = []

    for idea in edit_plan.ideas:
        if idea.coverage_status == "unresolved_ambiguous":
            findings.append(Finding(
                kind=DUPLICATE_IDEA,
                plan_id=plan_id, plan_version=plan_version,
                idea_id=idea.idea_id,
                clip_ids=idea.winning_clip_ids,
                detail={"reason": "two_or_more_members_of_one_idea_still_in_final_keep"},
                owning_authority="BestTakeResolver+SemanticArbiter",
                blocking=True,
            ))
            findings.append(Finding(
                kind=UNRESOLVED_RETRY,
                plan_id=plan_id, plan_version=plan_version,
                idea_id=idea.idea_id,
                clip_ids=idea.winning_clip_ids,
                detail={"reason": "retry_family_score_gap_too_thin_and_arbiter_did_not_resolve"},
                owning_authority="StoryValidator",
                blocking=True,
            ))
        elif idea.coverage_status == "missing":
            findings.append(Finding(
                kind=IDEA_COVERAGE_LOST,
                plan_id=plan_id, plan_version=plan_version,
                idea_id=idea.idea_id,
                clip_ids=idea.discarded_clip_ids,
                detail={"reason": "every_member_of_this_idea_was_discarded"},
                owning_authority="StoryValidator",
                blocking=True,
            ))

    for row in edit_plan.contradiction_findings:
        findings.append(Finding(
            kind=CONTRADICTION,
            plan_id=plan_id, plan_version=plan_version,
            idea_id=str(row.get("group_id") or "") or None,
            clip_ids=(str(row.get("left_clip_id") or ""), str(row.get("right_clip_id") or "")),
            detail=dict(row),
            owning_authority="StoryValidator",
            blocking=True,
        ))

    for row in edit_plan.lost_semantic_atoms:
        # D-031: a CONTEXTUAL-only atom loss (e.g. an incidental year) is
        # recorded as a non-blocking warning -- observability, not a
        # Freeze-blocking finding. Anything the coverage ledger itself
        # marked `blocking` (a critical/uncertain atom, or the broader
        # content-loss signal) still blocks exactly as before. A row from
        # before this field existed defaults to blocking=True (safe).
        target = findings if row.get("blocking", True) else warnings
        target.append(Finding(
            kind=UNIQUE_FACT_LOST,
            plan_id=plan_id, plan_version=plan_version,
            idea_id=None,
            clip_ids=(str(row.get("clip_id") or ""),),
            detail=dict(row),
            owning_authority="StoryValidator",
            blocking=bool(row.get("blocking", True)),
        ))

    findings.extend(_composite_order_findings(edit_plan, plan_id, plan_version))
    findings.extend(_causal_order_findings(edit_plan, plan_id, plan_version, causal_order_arbiter))

    if edit_plan.possible_missing_story_ending:
        warnings.append(Finding(
            kind=REQUIRED_CONTINUATION_LOST,
            plan_id=plan_id, plan_version=plan_version,
            idea_id=None,
            clip_ids=(),
            detail={"reason": "chronologically_last_kept_take_in_a_source_was_discarded"},
            owning_authority="StoryValidator",
            blocking=False,
        ))

    status = "FAIL" if findings else "PASS"
    return FinalEditReviewResult(status=status, findings=tuple(findings), warnings=tuple(warnings))
