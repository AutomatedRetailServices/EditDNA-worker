"""CompositeResolver -- Clean Cut Core V1's single, directly-callable authority
for delivery-attempt restoration, semantic rescue, and composite construction.

## Why this module exists (D-023)

Before this module, the same responsibility -- deciding which candidate
deliveries survive Hybrid's initial failed/BTS classification, which
deleted deliveries get restored as complementary, and which pairs of
deliveries get marked for a composite -- was implemented as 14 separate
``install_*()`` functions (see D-021's CompositeResolver row and D-022),
each called once at ``cutsell_worker`` import time, each monkeypatching the
SAME mutable module attribute (``hybrid_session_cleanup.
apply_hybrid_session_cleanup``, and in two cases also ``session_boundaries.
safe_group_takes_by_sessions``) on top of whichever wrapper the previous
one had already installed. The net behavior was one long, implicit,
import-order-dependent chain with no single owner, no explicit decision
record, and no way to answer "what does CompositeResolver do" without
reading all 14 files and the exact order ``cutsell_worker/__init__.py``
happened to call them in.

This module IS that chain, made explicit: the exact same algorithms
(unchanged), called directly, in one documented order, from one function.
Nothing here reimplements any matching/threshold logic -- every step below
delegates to the existing, already-tested pure function or glue that used
to live only inside an install-time closure (two of them, `hybrid_failed_
soft_restore` and `hybrid_unavailable_retry_fallback`, had their inline
closure logic extracted to a named module-level function for this purpose;
their own `install_*()` now delegates to that same function, so their
existing monkeypatch-based tests are unchanged).

``cutsell_worker/__init__.py`` no longer calls any of the 14 modules'
``install_*()`` functions. ``hybrid_session_cleanup.apply_hybrid_session_
cleanup`` and ``session_boundaries.safe_group_takes_by_sessions`` are
therefore guaranteed to stay their pure, unwrapped selves for the lifetime
of the process; ``pipeline.py`` calls ``apply_composite_resolution`` (this
module) and ``apply_composite_group_split`` (this module) directly instead.

## Canonical order (unchanged from the old chain -- see D-023's classification
table for why each step is still needed and what it owns)

1.  hybrid_session_cleanup.apply_hybrid_session_cleanup   (base: LLM classify + corroborated delete)
2.  hybrid_retry_completion_integrity                     (cross-group retry completion + parallel-clause rollback)
3.  hybrid_story_guard                                    (restore unique story paragraphs from non-authoritative deletes)
4.  hybrid_alternate_integrity                            (suppress stranded short alternates beside a clear winner)
5.  hybrid_cross_group_retry_integrity                    (collapse retries stranded across deterministic groups)
6.  hybrid_failed_continuation_integrity                  (repair split-fragment failed retries, both directions)
7.  hybrid_retry_winner_authority                         (drop a proven failed attempt superseded by a later clean winner)
8.  hybrid_gold_reconciliation                             (two narrow Human-Gold-exposed repairs)
9.  hybrid_failed_soft_restore                             (undo weak cross-group "failed" deletes lacking real authority)
10. hybrid_unavailable_retry_fallback                      (delete undecided incomplete retries when Hybrid was unavailable)
11. hybrid_complementary_delivery_guard                    (restore a complementary tail; delete unavailable prior restarts)
12. hybrid_semantic_complementary_rescue                   (restore a complete alternate with material unique content)
13. hybrid_semantic_composite_bridge                       (revoke same-opening rescues; normalize for composite matching)
14. hybrid_composite_best_take                             (restore performance-only deletes; build two-piece composites)
15. hybrid_semantic_conflict_arbitration                   (resolve label conflicts across overlapping windows)

Step 16, ``apply_post_selection_complementary_family_stabilizer`` (from
``post_selection_complementary_family_stabilizer.py``), is CompositeResolver's
one downstream extension: it operates on the already-built ``DraftTimeline``
(after grouping/ranking), not on raw takes, so it cannot be folded into the
take-level chain above. It is still owned and called explicitly by this
module (``apply_composite_family_stabilization``) rather than by monkeypatching
``pipeline.build_flow_b_draft`` -- see D-023.

## Composite group-splitting

Two of the 15 take-level steps (12 and 14) can mark a pair of deliveries as
a composite: pieces that must survive Best-Take's one-winner competition
together rather than be collapsed back into a single retry contest. The old
chain tracked this via two SEPARATE ``ContextVar`` instances (one per
module) and two separate monkeypatches of ``safe_group_takes_by_sessions``.
This module tracks it as one explicit return value (``split_ids``) from
``apply_composite_resolution``, and exposes ONE ``apply_composite_group_split``
function ``pipeline.py`` calls directly after grouping -- no ContextVar,
no monkeypatch.
"""
from __future__ import annotations

from typing import Iterable

from .contracts import CandidateTake
from .hybrid_alternate_integrity import suppress_stranded_hybrid_alternates
from .hybrid_complementary_delivery_guard import (
    _cross_group_deleted_ids,
    _delete_unavailable_prior_restarts,
    _restore_complementary_cross_group_deletions,
    _semantic_map as _guard_semantic_map,
)
from .hybrid_composite_best_take import (
    _choose_composite_replacements,
    _decision_map,
    _delete_strong_prefix_prior_restarts,
    _existing_restored_rows,
    _restore_performance_only_unique_deliveries,
    _semantic_map as _composite_semantic_map,
)
from .hybrid_cross_group_retry_integrity import collapse_cross_group_semantic_retries
from .hybrid_failed_continuation_integrity import (
    collapse_failed_split_retry_continuations,
    suppress_selected_prefixes_with_failed_suffixes,
)
from .hybrid_failed_soft_restore import restore_weak_failed_cross_group_deletions
from .hybrid_gold_reconciliation import reconcile_human_gold_hybrid
from .hybrid_retry_completion_integrity import apply_hybrid_retry_completion_integrity
from .hybrid_retry_winner_authority import enforce_proven_retry_winners
from .hybrid_semantic_complementary_rescue import (
    _completion_removed_pairs,
    _semantic_map as _rescue_semantic_map,
    complementary_relation,
)
from .hybrid_semantic_composite_bridge import reconcile_semantic_rescues
from .hybrid_semantic_conflict_arbitration import reconcile as reconcile_semantic_conflicts
from .hybrid_session_cleanup import HybridSessionCleanupResult, apply_hybrid_session_cleanup
from .hybrid_story_guard import restore_hybrid_story_coverage
from .hybrid_unavailable_retry_fallback import apply_unavailable_retry_fallback
from .post_selection_complementary_family_stabilizer import (
    apply_post_selection_complementary_family_stabilizer,
)


# --- Step 4: hybrid_alternate_integrity (glue copied verbatim from its own
# install_*() closure -- see D-023) ------------------------------------------

def _step_alternate_integrity(result, source_takes):
    if not result.kept or not result.semantic_decisions:
        return result
    kept, extra_deleted, guard_diagnostics = suppress_stranded_hybrid_alternates(
        result.kept, result.semantic_decisions,
    )
    if not guard_diagnostics:
        return result
    deleted_ids = {take.clip_id for take in result.deleted}
    deleted_ids.update(take.clip_id for take in extra_deleted)
    deleted = tuple(take for take in source_takes if take.clip_id in deleted_ids)
    diagnostics = tuple(result.diagnostics) + ({
        "hybrid_alternate_integrity": list(guard_diagnostics),
        "deleted_ids": [item["clip_id"] for item in guard_diagnostics],
    },)
    return type(result)(
        kept=kept, deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=diagnostics, semantic_decisions=result.semantic_decisions,
    )


# --- Step 5: hybrid_cross_group_retry_integrity ------------------------------

def _step_cross_group_retry_integrity(result, source_takes):
    if not result.kept or not result.semantic_decisions:
        return result
    kept, extra_deleted, guard_diagnostics = collapse_cross_group_semantic_retries(
        result.kept, result.semantic_decisions,
    )
    if not guard_diagnostics:
        return result
    deleted_ids = {take.clip_id for take in result.deleted}
    deleted_ids.update(take.clip_id for take in extra_deleted)
    deleted = tuple(take for take in source_takes if take.clip_id in deleted_ids)
    diagnostics = tuple(result.diagnostics) + ({
        "hybrid_cross_group_retry_integrity": list(guard_diagnostics),
        "deleted_ids": [item["clip_id"] for item in guard_diagnostics],
    },)
    return type(result)(
        kept=kept, deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=diagnostics, semantic_decisions=result.semantic_decisions,
    )


# --- Step 6: hybrid_failed_continuation_integrity (two-part repair) ---------

def _step_failed_continuation_integrity(result, source_takes):
    if not result.kept or not result.semantic_decisions:
        return result
    kept, first_removed, first_diagnostics = collapse_failed_split_retry_continuations(
        result.kept, result.semantic_decisions,
    )
    deleted_pool_ids = {take.clip_id for take in result.deleted}
    deleted_pool_ids.update(take.clip_id for take in first_removed)
    deleted_pool = tuple(take for take in source_takes if take.clip_id in deleted_pool_ids)

    kept, second_removed, second_diagnostics = suppress_selected_prefixes_with_failed_suffixes(
        kept, deleted_pool, result.semantic_decisions,
    )
    if not first_diagnostics and not second_diagnostics:
        return result

    deleted_ids = set(deleted_pool_ids)
    deleted_ids.update(take.clip_id for take in second_removed)
    deleted = tuple(take for take in source_takes if take.clip_id in deleted_ids)
    diagnostics = tuple(result.diagnostics) + ({
        "hybrid_failed_continuation_integrity": [*list(first_diagnostics), *list(second_diagnostics)],
        "deleted_ids": sorted(take.clip_id for take in (*first_removed, *second_removed)),
    },)
    return type(result)(
        kept=kept, deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=diagnostics, semantic_decisions=result.semantic_decisions,
    )


# --- Step 7: hybrid_retry_winner_authority -----------------------------------

def _step_retry_winner_authority(result, source_takes, context):
    if not result.kept or not result.semantic_decisions:
        return result
    kept, extra_deleted, authority_diagnostics = enforce_proven_retry_winners(
        result.kept, result.semantic_decisions, context,
    )
    if not authority_diagnostics:
        return result
    deleted_ids = {take.clip_id for take in result.deleted}
    deleted_ids.update(take.clip_id for take in extra_deleted)
    deleted = tuple(take for take in source_takes if take.clip_id in deleted_ids)
    diagnostics = tuple(result.diagnostics) + ({
        "hybrid_retry_winner_authority": list(authority_diagnostics),
        "deleted_ids": [item["clip_id"] for item in authority_diagnostics],
    },)
    return type(result)(
        kept=kept, deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=diagnostics, semantic_decisions=result.semantic_decisions,
    )


# --- Step 11: hybrid_complementary_delivery_guard (two-part repair) ---------

def _step_complementary_delivery_guard(result, source_takes):
    semantic = _guard_semantic_map(result.semantic_decisions)
    cross_group_deleted = _cross_group_deleted_ids(result.diagnostics)
    restore_ids, restore_rows = _restore_complementary_cross_group_deletions(
        tuple(result.kept), tuple(result.deleted), semantic, cross_group_deleted,
    )
    kept_ids = {take.clip_id for take in result.kept} | restore_ids
    kept = tuple(take for take in source_takes if take.clip_id in kept_ids)

    delete_ids: set[str] = set()
    delete_rows: list[dict] = []
    if result.requested_chunk_count > result.available_chunk_count:
        delete_ids, delete_rows = _delete_unavailable_prior_restarts(kept, semantic)
        if delete_ids:
            kept = tuple(take for take in kept if take.clip_id not in delete_ids)

    if not restore_rows and not delete_rows:
        return result

    final_kept_ids = {take.clip_id for take in kept}
    deleted = tuple(take for take in source_takes if take.clip_id not in final_kept_ids)
    diagnostics = tuple(result.diagnostics) + ({
        "hybrid_complementary_delivery_guard": {
            "restored": restore_rows,
            "deleted_unavailable_prior_restarts": delete_rows,
        },
        "restored_ids": sorted(restore_ids),
        "deleted_ids": sorted(delete_ids),
    },)
    return type(result)(
        kept=kept, deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=diagnostics, semantic_decisions=result.semantic_decisions,
    )


# --- Step 12: hybrid_semantic_complementary_rescue --------------------------
# Returns (result, split_ids) instead of setting a module ContextVar.

def _step_semantic_complementary_rescue(result, source_takes):
    pairs = _completion_removed_pairs(result.diagnostics)
    if not pairs:
        return result, frozenset()
    by_id = {take.clip_id: take for take in source_takes}
    semantic = _rescue_semantic_map(result.semantic_decisions)
    restore_ids: set[str] = set()
    split_ids: set[str] = set()
    audit: list[dict] = []
    for alternate_id, winner_id in pairs:
        alternate = by_id.get(alternate_id)
        winner = by_id.get(winner_id)
        if alternate is None or winner is None:
            continue
        relation = complementary_relation(alternate, winner, semantic)
        if relation is None:
            continue
        restore_ids.add(alternate_id)
        split_ids.update((alternate_id, winner_id))
        audit.append(relation)

    if not restore_ids:
        return result, frozenset()
    kept_ids = {take.clip_id for take in result.kept} | restore_ids
    kept = tuple(take for take in source_takes if take.clip_id in kept_ids)
    deleted = tuple(take for take in source_takes if take.clip_id not in kept_ids)
    diagnostics = tuple(result.diagnostics) + ({
        "hybrid_semantic_complementary_rescue": audit,
        "restored_ids": sorted(restore_ids),
        "split_group_clip_ids": sorted(split_ids),
    },)
    new_result = type(result)(
        kept=kept, deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=diagnostics, semantic_decisions=result.semantic_decisions,
    )
    return new_result, frozenset(split_ids)


# --- Step 13: hybrid_semantic_composite_bridge ------------------------------
# Takes the pending split_ids explicitly instead of reaching into another
# module's ContextVar.

def _step_semantic_composite_bridge(result, source_takes, pending_split_ids):
    kept, revoked, normalized = reconcile_semantic_rescues(
        source_takes, tuple(result.kept), result.diagnostics,
    )
    if not revoked and not normalized:
        return result, pending_split_ids

    split_ids = set(pending_split_ids)
    split_ids.difference_update(str(row["clip_id"]) for row in revoked)

    kept_ids = {take.clip_id for take in kept}
    deleted = tuple(take for take in source_takes if take.clip_id not in kept_ids)
    extra = {
        "hybrid_semantic_composite_bridge": {
            "revoked_same_opening_rescues": list(revoked),
            "normalized_composite_rescues": list(normalized),
        },
        # Composite Best Take already consumes this canonical guard shape.
        "hybrid_complementary_delivery_guard": {
            "restored": list(normalized),
            "deleted_unavailable_prior_restarts": [],
        },
        "deleted_ids": [str(row["clip_id"]) for row in revoked],
        "restored_ids": [str(row["clip_id"]) for row in normalized],
    }
    new_result = type(result)(
        kept=kept, deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=tuple(result.diagnostics) + (extra,),
        semantic_decisions=result.semantic_decisions,
    )
    return new_result, frozenset(split_ids)


# --- Step 14: hybrid_composite_best_take ------------------------------------

def _step_composite_best_take(result, source_takes):
    semantic = _composite_semantic_map(result.semantic_decisions)
    decisions = _decision_map(result.diagnostics)
    perf_restore_ids, perf_restore_rows = _restore_performance_only_unique_deliveries(
        tuple(result.kept), tuple(result.deleted), semantic, decisions,
    )

    kept_ids = {take.clip_id for take in result.kept} | perf_restore_ids
    kept = tuple(take for take in source_takes if take.clip_id in kept_ids)

    strong_delete_ids: set[str] = set()
    strong_delete_rows: list[dict] = []
    if result.requested_chunk_count > result.available_chunk_count:
        strong_delete_ids, strong_delete_rows = _delete_strong_prefix_prior_restarts(kept, semantic)
        if strong_delete_ids:
            kept = tuple(take for take in kept if take.clip_id not in strong_delete_ids)

    restored_rows = [*_existing_restored_rows(result.diagnostics), *perf_restore_rows]
    suppress_ids, split_ids, composite_rows = _choose_composite_replacements(
        kept, semantic, restored_rows,
    )
    if suppress_ids:
        kept = tuple(take for take in kept if take.clip_id not in suppress_ids)

    if not perf_restore_rows and not strong_delete_rows and not composite_rows:
        return result, frozenset(split_ids)

    final_kept_ids = {take.clip_id for take in kept}
    deleted = tuple(take for take in source_takes if take.clip_id not in final_kept_ids)
    diagnostics = tuple(result.diagnostics) + ({
        "hybrid_composite_best_take": {
            "restored_performance_only": perf_restore_rows,
            "deleted_strong_prefix_unavailable_restarts": strong_delete_rows,
            "composite_replacements": composite_rows,
            "split_group_clip_ids": sorted(split_ids),
        },
        "restored_ids": sorted(perf_restore_ids),
        "deleted_ids": sorted(strong_delete_ids | suppress_ids),
    },)
    new_result = type(result)(
        kept=kept, deleted=deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=diagnostics, semantic_decisions=result.semantic_decisions,
    )
    return new_result, frozenset(split_ids)


def apply_composite_resolution(
    takes: Iterable[CandidateTake], context, editorial_judge,
) -> tuple[HybridSessionCleanupResult, frozenset[str]]:
    """The ONE CompositeResolver step for take-level restoration/rescue/composite
    marking. Returns ``(result, composite_split_ids)`` -- the split ids are
    for ``apply_composite_group_split`` to apply to the grouping result that
    runs immediately after this in ``pipeline.py``.

    Same 15 algorithms, same order, as the old 14-file monkeypatch chain
    (step 1 is the base itself). See this module's docstring and D-023 for
    the classification and equivalence rationale.
    """
    source_takes = tuple(takes)
    result = apply_hybrid_session_cleanup(source_takes, context, editorial_judge)
    result = apply_hybrid_retry_completion_integrity(result, source_takes, context)
    result = restore_hybrid_story_coverage(source_takes, result, context)
    result = _step_alternate_integrity(result, source_takes)
    result = _step_cross_group_retry_integrity(result, source_takes)
    result = _step_failed_continuation_integrity(result, source_takes)
    result = _step_retry_winner_authority(result, source_takes, context)
    result = reconcile_human_gold_hybrid(result, source_takes, context)
    result = restore_weak_failed_cross_group_deletions(result, source_takes)
    result = apply_unavailable_retry_fallback(result, source_takes)
    result = _step_complementary_delivery_guard(result, source_takes)
    result, split_ids_rescue = _step_semantic_complementary_rescue(result, source_takes)
    result, split_ids_rescue = _step_semantic_composite_bridge(result, source_takes, split_ids_rescue)
    result, split_ids_composite = _step_composite_best_take(result, source_takes)
    result = reconcile_semantic_conflicts(result, source_takes)

    split_ids = frozenset(split_ids_rescue) | frozenset(split_ids_composite)
    return result, split_ids


def _split_groups_for_composite(
    groups: Iterable[Iterable[str]],
    split_ids: set[str] | frozenset[str],
    natural_ids: Iterable[str],
) -> tuple[tuple[str, ...], ...]:
    """Force each composite-marked clip into its own singleton group so
    BestTakeResolver's one-winner competition cannot re-collapse an intended
    composite. One shared implementation -- the old chain had two functions
    (``hybrid_composite_best_take._split_groups_for_composite`` and
    ``hybrid_semantic_complementary_rescue._split_groups``) that were
    already byte-for-byte identical in logic."""
    split_ids = set(split_ids)
    order = {clip_id: index for index, clip_id in enumerate(natural_ids)}
    out: list[tuple[str, ...]] = []
    for raw_group in groups:
        group = tuple(str(item) for item in raw_group)
        hits = tuple(clip_id for clip_id in group if clip_id in split_ids)
        remainder = tuple(clip_id for clip_id in group if clip_id not in split_ids)
        if remainder:
            out.append(remainder)
        out.extend((clip_id,) for clip_id in hits)
    out.sort(key=lambda group: min(order.get(clip_id, 10**9) for clip_id in group))
    return tuple(out)


def apply_composite_group_split(grouping_result, takes: Iterable[CandidateTake], split_ids: frozenset[str]):
    """Apply composite split ids to a grouping result. Called directly by
    pipeline.py right after ``safe_group_takes_by_sessions`` -- no
    ContextVar, no monkeypatch of ``session_boundaries``."""
    takes = tuple(takes)
    if not split_ids or not takes:
        return grouping_result
    natural_ids = tuple(take.clip_id for take in takes)
    relevant = split_ids & set(natural_ids)
    if not relevant:
        return grouping_result
    groups = _split_groups_for_composite(grouping_result.groups, relevant, natural_ids)
    if groups == tuple(grouping_result.groups):
        return grouping_result
    return type(grouping_result)(
        groups=groups,
        status=grouping_result.status,
        reason="; ".join(
            part for part in (
                grouping_result.reason,
                f"composite_resolver_group_split:{len(relevant)}",
            ) if part
        ),
    )


def apply_composite_family_stabilization(draft):
    """Step 16: the one genuinely downstream CompositeResolver extension --
    operates on the built DraftTimeline, not raw takes. Called explicitly by
    pipeline.py at the end of build_flow_b_draft; no longer installed as a
    monkeypatch on ``pipeline.build_flow_b_draft`` itself (see D-023)."""
    return apply_post_selection_complementary_family_stabilizer(draft)
