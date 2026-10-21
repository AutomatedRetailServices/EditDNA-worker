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

from .canonical_edit_plan import AuthoritativePlanSource, CanonicalEditPlan, build_canonical_edit_plan
from .causal_order_validator import CausalOrderArbiter
from .final_edit_reviewer import STORY_ORDER_BREAK, FinalEditReviewResult, review
from .lost_atom_repair_suppression import (
    REASON_NO_REPAIR_STRATEGY,
    REASON_SUPPRESSED_SAME_ATOM,
    LostAtomRepairSuppressionDecision,
    all_blocking_findings_safely_suppressed,
)

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
    # D-235S: purely additive provenance passthrough -- carries the SAME
    # `lost_atom_provenance_id` a `_lost_semantic_atoms()` row already
    # minted (see final_story_coherence_validation.py) all the way through
    # to whichever attempt row records `finding` here, WHEN that finding's
    # own `detail` carries one (only ever true for a UNIQUE_FACT_LOST
    # finding; every other finding kind's own `detail` never has this key,
    # so `.get()` below is `None` for them, exactly as before this task).
    # No behavior change: this field is never read by this loop's own
    # repair/termination decisions, only recorded for a future,
    # separately-authorized consumer (D-235T).
    source_lost_atom_provenance_id: str | None = None


@dataclass(frozen=True)
class RepairLoopResult:
    status: str  # "PASS" | "NEEDS_HUMAN_REVIEW"
    final_draft: object
    final_plan: CanonicalEditPlan
    final_review: FinalEditReviewResult
    attempts: tuple[RepairAttempt, ...]
    # D-235T: purely additive. `final_review` above is NEVER mutated by
    # this task -- it is always the real, honest `FinalEditReviewResult`
    # `review()` returned, so `final_review.status` can still legitimately
    # read "FAIL" with its own real findings even when THIS loop's own
    # `status` above reads "PASS" because every one of those findings
    # independently, deterministically re-verified (same D-235Q/D-235R
    # pure functions, same row data) as an already-safely-suppressed
    # non-material/retry/redundant lost atom. Defaults to `False` for
    # every pre-existing code path and every flag-off run -- see
    # `lost_atom_repair_suppression.py`'s own module docstring.
    blocking_findings_suppressed: bool = False
    # D-239F: the SAME per-finding `LostAtomRepairSuppressionDecision`
    # tuple `all_blocking_findings_safely_suppressed` below already
    # computes -- previously discarded once its own `all_suppressed`
    # boolean was consumed. Captured here verbatim (never recomputed) so
    # a caller can observe D-235S/D-235T's own actual verdict per atom,
    # not just this loop's own aggregate outcome. `()` whenever that
    # function was never invoked this pass (e.g. a repairable finding was
    # handled first, or `result.status` was already "PASS").
    suppression_decisions: tuple[LostAtomRepairSuppressionDecision, ...] = ()


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
    authoritative_source: AuthoritativePlanSource | None = None,
    # D-235X: an optional, caller-supplied `lost_atom_provenance_id ->
    # CompleteLostSemanticAtomMateriality` map -- the SAME already-computed
    # D-235Q result `final_story_coherence_validation.py`'s own Freeze-
    # composition seam produced (see `DraftTimeline.lost_atom_materiality_
    # by_provenance_id`, the natural source a caller passes here). Forwarded
    # unchanged to `all_blocking_findings_safely_suppressed` below so D-235T
    # reads the SAME result D-235R already used, instead of recomputing an
    # incomplete row-only materiality. `None` (the default, every pre-
    # D-235X caller) preserves the exact prior behavior -- no global
    # mutable state, no module singleton cache, purely a per-call parameter.
    lost_atom_materiality_by_provenance_id: dict | None = None,
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
    whether this loop can fix it.

    ``authoritative_source`` (D-087, AUTHORITATIVE resolver mode only) is
    forwarded unchanged to every `build_canonical_edit_plan` call so a
    repaired plan v2/v3 is represented from the same authoritative verdict
    as v1 -- see canonical_edit_plan.py's SINGLE-TRUTH CONTRACT note."""
    current_draft = draft
    plan = build_canonical_edit_plan(current_draft, authoritative_source=authoritative_source)
    result = review(plan, causal_order_arbiter=causal_order_arbiter)
    attempts: list[RepairAttempt] = []
    blocking_findings_suppressed = False
    # D-239F: captured from whichever `all_blocking_findings_safely_
    # suppressed` call below actually runs (at most once, since the loop
    # `break`s the same iteration it's computed in) -- see this field's
    # own docstring on `RepairLoopResult`.
    suppression_decisions: tuple[LostAtomRepairSuppressionDecision, ...] = ()

    for _ in range(max_attempts):
        if result.status == "PASS":
            break
        repairable = [f for f in result.findings if f.kind in _REPAIR_STRATEGIES]
        if not repairable:
            # D-235T: before unconditionally escalating to NEEDS_HUMAN_
            # REVIEW over the first blocking finding, check whether EVERY
            # current blocking finding independently, deterministically
            # re-verifies as an already-safely-suppressed non-material/
            # retry/redundant lost atom (same pure D-235Q/D-235R functions,
            # same row data -- see lost_atom_repair_suppression.py's own
            # module docstring). Default-OFF and byte-identical to the
            # pre-D-235T behavior below whenever the flag is off or even
            # one finding does not unanimously qualify.
            all_suppressed, this_pass_decisions = all_blocking_findings_safely_suppressed(
                result.findings, materiality_by_provenance_id=lost_atom_materiality_by_provenance_id,
            )
            # D-239F: capture regardless of `all_suppressed` -- every branch
            # below (suppressed and preserved) gets an honest, real D-235S/
            # D-235T verdict in the returned result, never just the
            # aggregate boolean.
            suppression_decisions = this_pass_decisions
            if all_suppressed:
                for finding, decision in zip(result.findings, suppression_decisions):
                    attempts.append(RepairAttempt(
                        plan_id=plan.plan_id,
                        previous_plan_version=plan.plan_version,
                        new_plan_version=plan.plan_version,
                        finding_kind=finding.kind,
                        idea_id=finding.idea_id,
                        owning_authority=finding.owning_authority,
                        previous_realization=finding.clip_ids,
                        replacement_realization=finding.clip_ids,
                        coverage_before=_idea_coverage_label(plan, finding.idea_id),
                        coverage_after=_idea_coverage_label(plan, finding.idea_id),
                        reason=REASON_SUPPRESSED_SAME_ATOM,
                        unaffected_ideas_changed=False,
                        repaired=False,
                        source_lost_atom_provenance_id=finding.detail.get("lost_atom_provenance_id"),
                    ))
                blocking_findings_suppressed = True
                break  # every blocking finding safely suppressed -- no guess made, nothing repaired

            # Unchanged pre-D-235T behavior: record why the loop is
            # stopping (for the audit trail) rather than stopping silently
            # -- no strategy exists for any current blocking finding, and
            # this batch of findings did not unanimously qualify for
            # same-atom suppression, so guessing one is not an option.
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
                reason=REASON_NO_REPAIR_STRATEGY,
                unaffected_ideas_changed=False,
                repaired=False,
                source_lost_atom_provenance_id=unrepairable.detail.get("lost_atom_provenance_id"),
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
                source_lost_atom_provenance_id=finding.detail.get("lost_atom_provenance_id"),
            ))
            break

        new_plan = build_canonical_edit_plan(repaired_draft, authoritative_source=authoritative_source)
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
            source_lost_atom_provenance_id=finding.detail.get("lost_atom_provenance_id"),
        ))

        current_draft = repaired_draft
        plan = new_plan
        result = new_result

    # D-235T: `blocking_findings_suppressed` reuses the loop's own existing
    # "PASS" value (the narrowest existing non-human-review terminal
    # status -- no third value invented) whenever every blocking finding
    # was safely suppressed above; `result` (-> `final_review`) itself is
    # NEVER mutated, so the real reviewer verdict stays honestly readable.
    status = "PASS" if (result.status == "PASS" or blocking_findings_suppressed) else "NEEDS_HUMAN_REVIEW"
    return RepairLoopResult(
        status=status,
        final_draft=current_draft,
        final_plan=plan,
        final_review=result,
        attempts=tuple(attempts),
        blocking_findings_suppressed=blocking_findings_suppressed,
        suppression_decisions=suppression_decisions,
    )
