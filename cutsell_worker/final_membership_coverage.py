"""Shared interpretation of explicit final-membership coverage proofs.

The final membership guard is allowed to remove a realization only when it
records which still-selected realization carries the same content.  Both the
StoryValidator and CanonicalEditPlan must consume that same proof; otherwise
one can correctly accept the final KEEP set while the other falsely reports
that the removed retry family vanished.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FinalMembershipCoverageCredit:
    reason: str
    witness_clip_ids: tuple[str, ...]


_EXPLICIT_WINNER_REASONS = frozenset({
    "direct_equivalence_confirmed_final_winner",
    "deterministic_retry_final_membership_resolution",
    "failed_unusable_retry_component_yields_to_complete_winner",
    "deterministic_retry_component_failed_debris",
    "provider_rejected_restatement_already_fully_delivered",
    "dependent_opening_yields_to_complete_family_peer",
    "contained_fragment_of_confirmed_duplicate",
    "terminal_negation_abandoned_restart",
    "orphaned_anaphoric_fragment_of_confirmed_retry",
})


def final_membership_coverage_credit(
    diagnostics: Mapping | None,
    discarded_id: str,
    selected_ids: set[str],
) -> FinalMembershipCoverageCredit | None:
    """Return the guard's explicit surviving witness for ``discarded_id``.

    This performs no similarity inference.  A credit exists only when the
    guard recorded a supported replacement/coverage relation and every
    witness named by that relation still belongs to final Selection.
    """
    for row in (diagnostics or {}).get("selection_conflicted_bridge_guard") or ():
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "")
        if reason == "later_continuation_chain_repeats_nearby_critical_claim":
            removed_ids = {str(value) for value in row.get("clip_ids") or ()}
            witnesses = tuple(str(value) for value in row.get("prior_clip_ids") or ())
            if (
                discarded_id in removed_ids
                and witnesses
                and set(witnesses).issubset(selected_ids)
            ):
                return FinalMembershipCoverageCredit(
                    "final_membership_nearby_chain_coverage", witnesses,
                )
        if reason == "later_selected_realization_fully_contained_in_nearby_delivery":
            winner_id = str(row.get("winner_clip_id") or "")
            if discarded_id == str(row.get("clip_id") or "") and winner_id in selected_ids:
                return FinalMembershipCoverageCredit(
                    "final_membership_contained_realization_coverage", (winner_id,),
                )
        if reason in _EXPLICIT_WINNER_REASONS:
            winner_id = str(row.get("winner_clip_id") or "")
            if discarded_id == str(row.get("clip_id") or "") and winner_id in selected_ids:
                return FinalMembershipCoverageCredit(
                    "final_membership_explicit_winner_coverage", (winner_id,),
                )
    return None
