"""Claim-coverage-aware Best-Take override -- D-038.

Runs immediately after `deterministic_best_take_authority.apply_
deterministic_best_take_authority`, before Final Story Coherence
Validation. Adds no new similarity/grouping heuristic of its own -- like
that module, it only reads `diagnostics["take_judge_groups"]` (the RankedTake
scores pipeline.py's take_judge.rank_takes already computed per retry-family
group) and the group's own current bucket assignment.

The question this module asks, for every genuine retry-family contest (2+
ranked members) with exactly one current winner: does that winner cover
every CRITICAL audience-facing claim (`semantic_claims.py`) found across
the GROUP'S OWN members? A visually/performance-clean take must never beat
a semantically complete one (RAW 33423953391: DeliveryScorer picked a take
missing the diagnosis-confirmation claim over the one that had it).

Resolution, bounded:
  1. If exactly one OTHER member covers every critical claim in the group
     (a strictly more complete realization), it becomes the new winner --
     the current winner is discarded, same KEEP/DISCARD-only move pattern
     `deterministic_best_take_authority.py` already uses (D-019: no SWAP).
  2. If no single member covers everything, but a PAIR of members --
     source-compatible, non-overlapping in time, not near-duplicates of
     each other -- together cover the full critical-claim set, both are
     kept as a narrow composite (ordered by recording time, matching
     `final_edit_reviewer._composite_order_findings`'s own requirement).
     This is a narrow, claim-coverage-triggered fallback, not a
     replacement for the general CompositeResolver upstream. Guarded
     against a real failure mode found while building this module's own
     CleanCutBench fixtures: two members whose UNIQUE contributions share a
     claim_type (e.g. both NEGATION) are more likely one idea's coarse
     paraphrase split across two attempts than genuinely complementary
     facts, so that pairing is never composited -- see the guard's own
     comment at the composite-pair loop for why, and why disjoint
     claim_types stay safe to composite.
  3. Anything broader than that (3+ members each carrying a different
     required claim, or no compatible pair) is left exactly as upstream
     decided it -- flagged in this module's own diagnostics for
     observability. The real backstop is `final_story_coherence_
     validation._lost_critical_claims`, which independently blocks Freeze
     on any critical claim still missing from the actual winning
     realization, regardless of whether this module could safely fix it.

Ambiguity fails open throughout, same posture as `deterministic_best_take_
authority.py`: any contest this module is not confident about is left
exactly as it was.
"""
from __future__ import annotations

from dataclasses import replace

from .semantic_claims import (
    ClaimEquivalenceArbiter,
    ClauseRoleArbiter,
    CRITICAL,
    claim_coverage,
    dedupe_claims,
    extract_claims,
    resolve_ambiguous_coverage,
)

_COMPOSITE_OVERLAP_TOLERANCE_SEC = 0.05


def _group_critical_claims(members: list[tuple[str, object]], *, clause_role_arbiter: ClauseRoleArbiter | None = None):
    """Every CRITICAL claim found across the group's own members, deduped
    across near-identical restatements between sibling attempts. D-040:
    `extract_claims` already splits a multi-clause sentence into its own
    CORE/SUPPORTING/CONTEXTUAL clauses, so a critical fact bundled with a
    merely-supporting reason surfaces as two separate claims here, not one
    -- only the core one is ever checked as CRITICAL."""
    all_claims = []
    for clip_id, clip in members:
        all_claims.extend(extract_claims(clip_id, str(clip.text or ""), clause_role_arbiter=clause_role_arbiter))
    deduped = dedupe_claims(tuple(all_claims))
    return tuple(c for c in deduped if c.importance == CRITICAL)


def _covered_claim_ids(claims, text: str, *, arbiter: ClaimEquivalenceArbiter | None) -> frozenset:
    covered = set()
    for claim in claims:
        coverage = claim_coverage(claim, text)
        if resolve_ambiguous_coverage(claim, text, coverage=coverage, arbiter=arbiter):
            covered.add(claim.claim_id)
    return frozenset(covered)


def _time_compatible(left, right) -> bool:
    if left.source_asset_id != right.source_asset_id:
        return False
    a, b = (left, right) if left.start <= right.start else (right, left)
    return float(b.start) >= float(a.end) - _COMPOSITE_OVERLAP_TOLERANCE_SEC


def apply_claim_coverage_best_take(
    draft, *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
    clause_role_arbiter: ClauseRoleArbiter | None = None,
):
    """See module docstring. Never invents a new grouping/similarity
    heuristic; only reads `take_judge_groups` and moves clips between the
    existing selected/discarded buckets (KEEP/DISCARD only, D-019)."""
    groups = list((draft.diagnostics or {}).get("take_judge_groups") or ())
    if not groups:
        return draft

    selected_by_id = {clip.clip_id: clip for clip in draft.selected}
    alternates_by_id = {clip.clip_id: clip for clip in draft.alternates}
    discarded_by_id = {clip.clip_id: clip for clip in draft.discarded}
    all_clips = {**selected_by_id, **alternates_by_id, **discarded_by_id}

    def bucket_of(clip_id: str) -> str:
        if clip_id in selected_by_id:
            return "select"
        if clip_id in alternates_by_id:
            return "swap"
        return "discard"

    new_selected = dict(selected_by_id)
    new_alternates = dict(alternates_by_id)
    new_discarded = dict(discarded_by_id)
    overrides: list[dict] = []
    composites: list[dict] = []
    unresolved_gaps: list[dict] = []

    def move(clip_id: str, target: str) -> None:
        clip = all_clips[clip_id]
        new_selected.pop(clip_id, None)
        new_alternates.pop(clip_id, None)
        new_discarded.pop(clip_id, None)
        {"select": new_selected, "swap": new_alternates, "discard": new_discarded}[target][clip_id] = \
            replace(clip, selected=(target == "select"))

    for group in groups:
        group_id = group.get("group_id")
        ranked = list(group.get("ranked") or ())
        member_ids = [str(row.get("clip_id") or "") for row in ranked]
        members = [(cid, all_clips[cid]) for cid in member_ids if cid in all_clips]
        if len(members) < 2:
            continue

        current_winners = [cid for cid, _clip in members if bucket_of(cid) == "select"]
        if len(current_winners) != 1:
            # Not this module's job: an unresolved (2+ still selected) or
            # already-fully-lost (0 selected) family is StoryValidator's
            # existing territory.
            continue
        winner_id = current_winners[0]
        winner_clip = all_clips[winner_id]

        critical_claims = _group_critical_claims(members, clause_role_arbiter=clause_role_arbiter)
        if not critical_claims:
            continue

        winner_covered = _covered_claim_ids(critical_claims, str(winner_clip.text or ""), arbiter=claim_equivalence_arbiter)
        missing = [c for c in critical_claims if c.claim_id not in winner_covered]
        if not missing:
            continue

        # 1. Does a single OTHER member cover every critical claim?
        full_coverage_candidate = None
        for clip_id, clip in members:
            if clip_id == winner_id:
                continue
            covered = _covered_claim_ids(critical_claims, str(clip.text or ""), arbiter=claim_equivalence_arbiter)
            if covered == frozenset(c.claim_id for c in critical_claims):
                full_coverage_candidate = clip_id
                break

        if full_coverage_candidate is not None:
            move(winner_id, "discard")
            move(full_coverage_candidate, "select")
            overrides.append({
                "group_id": group_id,
                "previous_winner_clip_id": winner_id,
                "new_winner_clip_id": full_coverage_candidate,
                "reason": "single_candidate_covers_all_critical_claims_previous_winner_did_not",
                "missing_claim_ids": [c.claim_id for c in missing],
                "missing_claim_texts": [c.text for c in missing],
            })
            continue

        # 2. Bounded 2-piece composite: do exactly two members, together,
        # cover everything, and are they safe to place side by side?
        claim_by_id = {c.claim_id: c for c in critical_claims}
        composite_found = False
        for i, (id_a, clip_a) in enumerate(members):
            for id_b, clip_b in members[i + 1:]:
                if id_a == winner_id and id_b == winner_id:
                    continue
                covered_a = _covered_claim_ids(critical_claims, str(clip_a.text or ""), arbiter=claim_equivalence_arbiter)
                covered_b = _covered_claim_ids(critical_claims, str(clip_b.text or ""), arbiter=claim_equivalence_arbiter)
                union = covered_a | covered_b
                if union != frozenset(c.claim_id for c in critical_claims):
                    continue
                # Guard against forcing two members into a false composite
                # when their UNIQUE contributions share a claim_type: two
                # same-typed claims (e.g. both NEGATION) are more likely a
                # coarse-classifier miss on one restated idea than genuinely
                # complementary distinct facts -- exactly how a paraphrased
                # retry family (each side worded its own negation slightly
                # differently) could otherwise get frozen as a fake
                # "composite" instead of correctly collapsing to one winner
                # via the existing semantic-equivalence-arbiter tie-break
                # this module never overrides. Disjoint claim_types (e.g. a
                # STATE_RESULT fact and a DIAGNOSIS_IDENTIFICATION fact) are
                # structurally unlikely to be the same proposition, so those
                # stay compositable. No arbiter exists yet for "are these
                # two claims equivalent" specifically (a different, pairwise
                # question from ClaimEquivalenceArbiter's own coverage
                # check) -- honest gap, fails closed to unresolved_gaps.
                unique_to_a = covered_a - covered_b
                unique_to_b = covered_b - covered_a
                types_a = {claim_by_id[cid].claim_type for cid in unique_to_a}
                types_b = {claim_by_id[cid].claim_type for cid in unique_to_b}
                if types_a & types_b:
                    continue
                if not _time_compatible(clip_a, clip_b):
                    continue
                first, second = (clip_a, clip_b) if clip_a.start <= clip_b.start else (clip_b, clip_a)
                move(first.clip_id, "select")
                move(second.clip_id, "select")
                for clip_id, _clip in members:
                    if clip_id not in (first.clip_id, second.clip_id) and bucket_of(clip_id) != "discard":
                        move(clip_id, "discard")
                composites.append({
                    "group_id": group_id,
                    "clip_ids": [first.clip_id, second.clip_id],
                    "reason": "claim_coverage_complementary",
                    "covered_claim_ids": sorted(union),
                })
                composite_found = True
                break
            if composite_found:
                break

        if not composite_found:
            unresolved_gaps.append({
                "group_id": group_id,
                "winner_clip_id": winner_id,
                "missing_claim_ids": [c.claim_id for c in missing],
                "missing_claim_texts": [c.text for c in missing],
                "reason": "no_single_or_paired_candidate_safely_covers_every_critical_claim",
            })

    if not (overrides or composites or unresolved_gaps):
        return draft

    def _order(clip):
        return (clip.source_order, float(clip.start), float(clip.end), clip.clip_id)

    selected = tuple(sorted(new_selected.values(), key=_order))
    alternates = tuple(sorted(new_alternates.values(), key=_order))
    discarded = tuple(sorted(new_discarded.values(), key=_order))

    diagnostics = dict(draft.diagnostics or {})
    diagnostics["claim_coverage_best_take"] = {
        "status": "applied",
        "overrides": overrides,
        "composites": composites,
        "unresolved_gaps": unresolved_gaps,
    }
    return replace(draft, selected=selected, alternates=alternates, discarded=discarded, diagnostics=diagnostics)
