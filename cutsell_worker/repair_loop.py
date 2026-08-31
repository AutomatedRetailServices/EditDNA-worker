"""Bounded, targeted semantic repair loop -- D-026.

CanonicalEditPlan v1 -> FinalEditReviewer FAIL -> route ONLY the affected
Idea/family to a repair strategy owned by the finding's `owning_authority`
-> CanonicalEditPlan v2 -> re-review -> PASS -> Freeze (or exhaust bounded
attempts -> NEEDS_HUMAN_REVIEW, never Freeze).

## Honest scope: which findings actually have a safe automatic repair

Only `STORY_ORDER_BREAK` (an accepted composite's own components out of
recording order) has a repair strategy here, because it is the one finding
type this architecture can fix WITHOUT guessing at content: the fix is a
pure reorder of two already-selected clips back into their recorded
sequence, touching no other clip's membership, text, or position.

`DUPLICATE_IDEA`, `UNRESOLVED_RETRY`, `IDEA_COVERAGE_LOST`, `CONTRADICTION`,
and `UNIQUE_FACT_LOST` have NO automatic repair here, by design, not by
omission: an automatic "fix" for any of them means the system guessing
which content is correct (which take wins a still-ambiguous contest, which
discarded clip to blindly restore, which side of a contradiction is true).
CLAUDE.md's own "WHEN UNCERTAIN, KEEP" rule and this whole session's
established conservative philosophy (deterministic_best_take_authority
already declines to force a decision on a thin score gap; CompositeResolver's
restore functions already require strong, specific evidence rather than a
blanket "restore anything the coverage ledger flagged") both say the same
thing: guessing here is a regression in editorial judgment, not a repair.
The loop still runs for these findings (recording an attempt with
`repaired=False`) so the audit trail and bounded-termination behavior are
uniform, but they always route straight to `NEEDS_HUMAN_REVIEW`.

## What "targeted" means here

A repair mutates only the specific clips a finding names, at their existing
positions in `draft.selected` -- nothing else in the timeline moves,
nothing else is discarded or restored. Because Final Story Coherence
Validation's own checks (`lost_semantic_atoms`, `contradiction_findings`,
`missing_idea_coverage`) are order-independent (they read `draft.selected`/
`discarded` as sets of clips, not sequences), a pure reorder repair never
needs to re-run that whole validation pass -- only CanonicalEditPlan
(order-sensitive) and FinalEditReviewer are rebuilt for a fresh review.
That is what keeps this "targeted" rather than "globally re-run everything".
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .canonical_edit_plan import CanonicalEditPlan, build_canonical_edit_plan
from .causal_order_validator import CausalOrderArbiter
from .final_edit_reviewer import STORY_ORDER_BREAK, FinalEditReviewResult, review

DEFAULT_MAX_REPAIR_ATTEMPTS = 3


@dataclass(frozen=True)
class RepairAttempt:
    plan_id: str
    previous_plan_version: int
    new_plan_version: int
    finding_kind: str
    idea_id: str | None
    owning_authority: str
    previous_realization: tuple[str, ...]
    replacement_realization: tuple[str, ...]
    coverage_before: str
    coverage_after: str
    reason: str
    unaffected_ideas_changed: bool
    repaired: bool


@dataclass(frozen=True)
class RepairLoopResult:
    status: str  # "PASS" | "NEEDS_HUMAN_REVIEW"
    final_draft: object
    final_plan: CanonicalEditPlan
    final_review: FinalEditReviewResult
    attempts: tuple[RepairAttempt, ...]


def _repair_story_order_break(draft, finding):
    """Reorder a composite's components back into recording order, at the
    exact positions they already occupy in draft.selected. Every other
    clip's position is untouched."""
    target_order = list(finding.detail.get("recording_order") or ())
    if len(target_order) < 2:
        return None
    clip_ids = set(target_order)
    by_id = {clip.clip_id: clip for clip in draft.selected if clip.clip_id in clip_ids}
    if set(by_id) != clip_ids:
        return None  # a named clip is missing from selected -- do not guess
    positions = sorted(i for i, clip in enumerate(draft.selected) if clip.clip_id in clip_ids)
    if len(positions) != len(target_order):
        return None
    new_selected = list(draft.selected)
    for pos, cid in zip(positions, target_order):
        new_selected[pos] = by_id[cid]
    return replace(draft, selected=tuple(new_selected))


_REPAIR_STRATEGIES = {
    STORY_ORDER_BREAK: _repair_story_order_break,
}


def _idea_coverage_label(plan: CanonicalEditPlan, idea_id: str | None) -> str:
    for idea in plan.ideas:
        if idea.idea_id == idea_id:
            return idea.coverage_status
    return "unknown"


def run_repair_loop(
    draft,
    *,
    max_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS,
    causal_order_arbiter: CausalOrderArbiter | None = None,
) -> RepairLoopResult:
    """Build CanonicalEditPlan v1, review it, and -- only for finding types
    with a safe repair strategy -- apply bounded, targeted repairs and
    re-review, up to ``max_attempts``. Never mutates unrelated ideas. Never
    guesses content. See module docstring for the honest scope.

    ``causal_order_arbiter`` is forwarded to every `review()` call (D-027):
    CAUSAL_ORDER_BREAK has no repair strategy in `_REPAIR_STRATEGIES` below
    (a cross-idea reorder risks undoing an intentional Composer pacing
    choice), so it always routes straight to NEEDS_HUMAN_REVIEW here -- the
    arbiter only affects whether review() emits the finding at all, not
    whether this loop can fix it."""
    current_draft = draft
    plan = build_canonical_edit_plan(current_draft)
    result = review(plan, causal_order_arbiter=causal_order_arbiter)
    attempts: list[RepairAttempt] = []

    for _ in range(max_attempts):
        if result.status == "PASS":
            break
        repairable = [f for f in result.findings if f.kind in _REPAIR_STRATEGIES]
        if not repairable:
            # Record why the loop is stopping (for the audit trail) rather
            # than stopping silently -- no strategy exists for any current
            # blocking finding, so guessing one is not an option.
            unrepairable = result.findings[0]
            attempts.append(RepairAttempt(
                plan_id=plan.plan_id,
                previous_plan_version=plan.plan_version,
                new_plan_version=plan.plan_version,
                finding_kind=unrepairable.kind,
                idea_id=unrepairable.idea_id,
                owning_authority=unrepairable.owning_authority,
                previous_realization=unrepairable.clip_ids,
                replacement_realization=unrepairable.clip_ids,
                coverage_before=_idea_coverage_label(plan, unrepairable.idea_id),
                coverage_after=_idea_coverage_label(plan, unrepairable.idea_id),
                reason="no_repair_strategy_exists_for_this_finding_kind",
                unaffected_ideas_changed=False,
                repaired=False,
            ))
            break  # nothing this loop knows how to fix -- stop, do not guess

        finding = repairable[0]
        strategy = _REPAIR_STRATEGIES[finding.kind]
        repaired_draft = strategy(current_draft, finding)
        coverage_before = _idea_coverage_label(plan, finding.idea_id)

        if repaired_draft is None:
            attempts.append(RepairAttempt(
                plan_id=plan.plan_id,
                previous_plan_version=plan.plan_version,
                new_plan_version=plan.plan_version,
                finding_kind=finding.kind,
                idea_id=finding.idea_id,
                owning_authority=finding.owning_authority,
                previous_realization=finding.clip_ids,
                replacement_realization=finding.clip_ids,
                coverage_before=coverage_before,
                coverage_after=coverage_before,
                reason="no_repair_strategy_could_apply_safely",
                unaffected_ideas_changed=False,
                repaired=False,
            ))
            break

        new_plan = build_canonical_edit_plan(repaired_draft)
        new_plan = replace(new_plan, plan_version=plan.plan_version + 1)
        new_result = review(new_plan, causal_order_arbiter=causal_order_arbiter)

        other_ideas_before = {i.idea_id: i.winning_clip_ids for i in plan.ideas if i.idea_id != finding.idea_id}
        other_ideas_after = {i.idea_id: i.winning_clip_ids for i in new_plan.ideas if i.idea_id != finding.idea_id}
        unaffected_changed = other_ideas_before != other_ideas_after

        attempts.append(RepairAttempt(
            plan_id=plan.plan_id,
            previous_plan_version=plan.plan_version,
            new_plan_version=new_plan.plan_version,
            finding_kind=finding.kind,
            idea_id=finding.idea_id,
            owning_authority=finding.owning_authority,
            previous_realization=finding.clip_ids,
            replacement_realization=tuple(finding.detail.get("recording_order") or finding.clip_ids),
            coverage_before=coverage_before,
            coverage_after=_idea_coverage_label(new_plan, finding.idea_id),
            reason="reordered_composite_components_to_recording_order",
            unaffected_ideas_changed=unaffected_changed,
            repaired=True,
        ))

        current_draft = repaired_draft
        plan = new_plan
        result = new_result

    status = "PASS" if result.status == "PASS" else "NEEDS_HUMAN_REVIEW"
    return RepairLoopResult(
        status=status,
        final_draft=current_draft,
        final_plan=plan,
        final_review=result,
        attempts=tuple(attempts),
    )
