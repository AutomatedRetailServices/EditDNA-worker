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
IDEA_COVERAGE_LOST, CONTRADICTION, and UNIQUE_FACT_LOST are exactly the
findings that already set `final_story_coherence_validation`'s own
`freeze_blocked` -- surfacing them here through this vocabulary does not
change what blocks Freeze today. REQUIRED_CONTINUATION_LOST is included as
a non-blocking warning (mirrors `possible_missing_story_ending`'s existing
"flag only" contract -- promoting it to blocking would be a new behavior
change unvalidated by CleanCutBench, not something to slip in here).

Honest gap, not fabricated coverage: INCOMPLETE_DELIVERY, ORPHAN_FRAGMENT,
INCOMPATIBLE_COMPOSITE, and STORY_ORDER_BREAK are recognized in the finding
vocabulary (so a future detector has a defined shape to populate) but have
no existing deterministic detector to draw from, and none is invented here.
They are never emitted. A finding type appearing in `_UNIMPLEMENTED_KINDS`
below is the honest record of that gap -- do not remove it by inventing a
heuristic under time pressure; add a real, tested detector first.
"""
from __future__ import annotations

from dataclasses import dataclass

from .canonical_edit_plan import CanonicalEditPlan

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

_UNIMPLEMENTED_KINDS = frozenset({
    INCOMPLETE_DELIVERY, ORPHAN_FRAGMENT, INCOMPATIBLE_COMPOSITE, STORY_ORDER_BREAK,
})


@dataclass(frozen=True)
class Finding:
    kind: str
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


def review(edit_plan: CanonicalEditPlan) -> FinalEditReviewResult:
    """Review the CanonicalEditPlan. Never mutates it or the draft it came
    from -- returns PASS or structured findings for the caller to act on
    (universal_clean_cut.py already skips Freeze/Boundary whenever
    freeze_blocked is set, which is exactly the "route back" mechanism for
    every blocking finding here)."""
    findings: list[Finding] = []
    warnings: list[Finding] = []

    for idea in edit_plan.ideas:
        if idea.coverage_status == "unresolved_ambiguous":
            findings.append(Finding(
                kind=DUPLICATE_IDEA,
                idea_id=idea.idea_id,
                clip_ids=idea.winning_clip_ids,
                detail={"reason": "two_or_more_members_of_one_idea_still_in_final_keep"},
                owning_authority="BestTakeResolver+SemanticArbiter",
                blocking=True,
            ))
            findings.append(Finding(
                kind=UNRESOLVED_RETRY,
                idea_id=idea.idea_id,
                clip_ids=idea.winning_clip_ids,
                detail={"reason": "retry_family_score_gap_too_thin_and_arbiter_did_not_resolve"},
                owning_authority="StoryValidator",
                blocking=True,
            ))
        elif idea.coverage_status == "missing":
            findings.append(Finding(
                kind=IDEA_COVERAGE_LOST,
                idea_id=idea.idea_id,
                clip_ids=idea.discarded_clip_ids,
                detail={"reason": "every_member_of_this_idea_was_discarded"},
                owning_authority="StoryValidator",
                blocking=True,
            ))

    for row in edit_plan.contradiction_findings:
        findings.append(Finding(
            kind=CONTRADICTION,
            idea_id=str(row.get("group_id") or "") or None,
            clip_ids=(str(row.get("left_clip_id") or ""), str(row.get("right_clip_id") or "")),
            detail=dict(row),
            owning_authority="StoryValidator",
            blocking=True,
        ))

    for row in edit_plan.lost_semantic_atoms:
        findings.append(Finding(
            kind=UNIQUE_FACT_LOST,
            idea_id=None,
            clip_ids=(str(row.get("clip_id") or ""),),
            detail=dict(row),
            owning_authority="StoryValidator",
            blocking=True,
        ))

    if edit_plan.possible_missing_story_ending:
        warnings.append(Finding(
            kind=REQUIRED_CONTINUATION_LOST,
            idea_id=None,
            clip_ids=(),
            detail={"reason": "chronologically_last_kept_take_in_a_source_was_discarded"},
            owning_authority="StoryValidator",
            blocking=False,
        ))

    status = "FAIL" if findings else "PASS"
    return FinalEditReviewResult(status=status, findings=tuple(findings), warnings=tuple(warnings))
