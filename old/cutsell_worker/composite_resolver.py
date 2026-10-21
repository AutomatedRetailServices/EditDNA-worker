"""CompositeResolver -- Clean Cut Core V1's single, directly-callable authority
for delivery-attempt restoration, semantic rescue, and composite construction.

## Why this module exists (D-023)

Before this module, the same responsibility -- deciding which candidate
deliveries survive Hybrid's initial failed/BTS classification, which
deleted deliveries get restored as complementary, and which pairs of
deliveries get marked for a composite -- was implemented as a run of
separate ``install_*()`` functions (see D-021's CompositeResolver row and
D-022/D-023), each called once at ``cutsell_worker`` import time, each
monkeypatching the SAME mutable module attribute
(``hybrid_session_cleanup.apply_hybrid_session_cleanup``, and in two cases
also ``session_boundaries.safe_group_takes_by_sessions``) on top of
whichever wrapper the previous one had already installed. The net behavior
was one long, implicit, import-order-dependent chain with no single owner
and no explicit decision record.

**The full set turned out to be 19 hooks, not 14.** An earlier version of
this module hand-transcribed 14 of them (the ones named `install_hybrid_*`/
`install_post_selection_*`, matching the original audit) directly into
composed step functions here, in what was believed to be their exact
execution order. That transcription MISSED five more hooks that also wrap
the identical function under different naming (`semantic_fragment_guard`,
`incomplete_bridge_retry_authority`, `failed_prefix_completion_rescue`,
`final_delivery_integrity`, `terminal_delivery_reconciliation`) -- found
only when a differential test written against this module's own claimed
"pure base" reference failed, because the base was not actually pure. Worse,
those five were INTERLEAVED with the original 14 in `__init__.py`'s
historical order, not merely appended before or after them, so a hand
composition that only reordered the 14 would have run several of them in
the wrong relative order relative to the five it did not know existed.

Given real evidence that hand-transcribing this many interacting closures
is error-prone even with careful reading, this module now builds the chain
a fundamentally safer way: it calls each hook's own real, already-tested
`install_*()` function -- unmodified, verbatim -- exactly once, in the
exact historical order `cutsell_worker/__init__.py` used to call them in,
against private scratch module state, then restores the two shared module
attributes to what they already were before this ran. This reuses every
hook's real closure directly (zero risk of a transcription error changing
any threshold, condition, or diagnostics key) while still turning 19
scattered import-time side effects into one composed, directly-callable,
private reference this module owns and pipeline.py calls explicitly.

None of the 19 hooks' own `install_*()` functions are called anywhere else
any more -- `cutsell_worker/__init__.py` no longer calls any of them, so
`hybrid_session_cleanup.apply_hybrid_session_cleanup` and
`session_boundaries.safe_group_takes_by_sessions` stay their pure,
unwrapped selves for the process lifetime; only this module's private
`_TAKE_LEVEL_CHAIN` (built once, lazily, on first use) carries the composed
behavior. Each hook's own file, its own pure/glue functions, and its own
monkeypatch-based tests are all completely unchanged.

## Canonical order (the historical `__init__.py` order, verbatim)

1.  hybrid_session_cleanup.apply_hybrid_session_cleanup   (base: LLM classify + corroborated delete)
2.  semantic_fragment_guard                                (textual-structure corroboration for tiny/open failed fragments)
3.  hybrid_retry_completion_integrity                       (cross-group retry completion + parallel-clause rollback)
4.  hybrid_story_guard                                      (restore unique story paragraphs from non-authoritative deletes)
5.  hybrid_alternate_integrity                               (suppress stranded short alternates beside a clear winner)
6.  hybrid_cross_group_retry_integrity                       (collapse retries stranded across deterministic groups)
7.  incomplete_bridge_retry_authority                        (protect a completed clause's bridge continuation)
8.  hybrid_failed_continuation_integrity                     (repair split-fragment failed retries, both directions)
9.  hybrid_retry_winner_authority                            (drop a proven failed attempt superseded by a later clean winner)
10. hybrid_gold_reconciliation                                (two narrow Human-Gold-exposed repairs)
11. failed_prefix_completion_rescue                          (rescue a clean completion prefix from a failed tail)
12. final_delivery_integrity                                 (three global delivery-integrity repairs)
13. terminal_delivery_reconciliation                         (two terminal boundary/attempt repairs)
14. hybrid_failed_soft_restore                                (undo weak cross-group "failed" deletes lacking real authority)
15. hybrid_unavailable_retry_fallback                        (delete undecided incomplete retries when Hybrid was unavailable)
16. hybrid_complementary_delivery_guard                       (restore a complementary tail; delete unavailable prior restarts)
17. hybrid_semantic_complementary_rescue                      (restore a complete alternate with material unique content)
18. hybrid_semantic_composite_bridge                          (revoke same-opening rescues; normalize for composite matching)
19. hybrid_composite_best_take                                (restore performance-only deletes; build two-piece composites)
20. hybrid_semantic_conflict_arbitration                      (resolve label conflicts across overlapping windows)

Step 21, ``apply_post_selection_complementary_family_stabilizer`` (from
``post_selection_complementary_family_stabilizer.py``), is CompositeResolver's
one downstream extension: it operates on the already-built ``DraftTimeline``
(after grouping/ranking), not on raw takes, so it cannot be folded into the
take-level chain above. It is still owned and called explicitly by this
module (``apply_composite_family_stabilization``) rather than by monkeypatching
``pipeline.build_flow_b_draft``.

## Composite group-splitting

Only a true two-piece composite built by ``hybrid_composite_best_take`` is
allowed to bypass Best-Take's one-winner competition. A semantic
complementary rescue may restore a delivery that would otherwise be lost,
but restoration is not itself proof that two complete realizations should
co-exist. Rescue ids are therefore cleared here after the chain runs but are
not promoted into ``split_ids``. Restored deliveries remain eligible to
compete normally downstream; true composite ids remain split into singleton
groups so their complementary pieces survive together.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Iterable

from .contracts import CandidateTake
from .hybrid_session_cleanup import HybridSessionCleanupResult
from .post_selection_complementary_family_stabilizer import (
    apply_post_selection_complementary_family_stabilizer,
)

_CHAIN_SPEC: tuple[tuple[str, str], ...] = (
    ("semantic_fragment_guard", "install_semantic_fragment_guard"),
    ("hybrid_retry_completion_integrity", "install_hybrid_retry_completion_integrity"),
    ("hybrid_story_guard", "install_hybrid_story_coverage_guard"),
    ("hybrid_alternate_integrity", "install_hybrid_alternate_integrity"),
    ("hybrid_cross_group_retry_integrity", "install_hybrid_cross_group_retry_integrity"),
    ("incomplete_bridge_retry_authority", "install_incomplete_bridge_retry_authority"),
    ("hybrid_failed_continuation_integrity", "install_hybrid_failed_continuation_integrity"),
    ("hybrid_retry_winner_authority", "install_hybrid_retry_winner_authority"),
    ("hybrid_gold_reconciliation", "install_hybrid_gold_reconciliation"),
    ("failed_prefix_completion_rescue", "install_failed_prefix_completion_rescue"),
    ("final_delivery_integrity", "install_final_delivery_integrity"),
    ("terminal_delivery_reconciliation", "install_terminal_delivery_reconciliation"),
    ("hybrid_failed_soft_restore", "install_hybrid_failed_soft_restore"),
    ("hybrid_unavailable_retry_fallback", "install_hybrid_unavailable_retry_fallback"),
    ("hybrid_complementary_delivery_guard", "install_hybrid_complementary_delivery_guard"),
    ("hybrid_semantic_complementary_rescue", "install_hybrid_semantic_complementary_rescue"),
    ("hybrid_semantic_composite_bridge", "install_hybrid_semantic_composite_bridge"),
    ("hybrid_composite_best_take", "install_hybrid_composite_best_take"),
    ("hybrid_semantic_conflict_arbitration", "install_hybrid_semantic_conflict_arbitration"),
)

_take_level_chain: Callable | None = None


def _build_take_level_chain() -> Callable:
    """Build the take-level chain once using each hook's own installer."""
    import importlib

    from . import hybrid_session_cleanup, session_boundaries

    base_cleanup = hybrid_session_cleanup.apply_hybrid_session_cleanup
    base_grouping = session_boundaries.safe_group_takes_by_sessions

    for module_name, install_name in _CHAIN_SPEC:
        module = importlib.import_module(f".{module_name}", __package__)
        getattr(module, install_name)()

    chain = hybrid_session_cleanup.apply_hybrid_session_cleanup
    session_boundaries.safe_group_takes_by_sessions = base_grouping
    hybrid_session_cleanup.apply_hybrid_session_cleanup = base_cleanup
    return chain


def _get_take_level_chain() -> Callable:
    global _take_level_chain
    if _take_level_chain is None:
        _take_level_chain = _build_take_level_chain()
    return _take_level_chain


def _composite_split_ids() -> frozenset[str]:
    """Return only ids proven to be a true multi-piece composite.

    Semantic complementary rescue remains a restoration authority, not a
    co-keep authority. Its ContextVar is still consumed/cleared here because
    the semantic composite bridge may use it inside the take-level chain, but
    rescue ids no longer receive singleton immunity from Best Take.
    """
    from . import hybrid_composite_best_take, hybrid_semantic_complementary_rescue

    composite_ids = frozenset(hybrid_composite_best_take._COMPOSITE_SPLIT_IDS.get())
    hybrid_semantic_complementary_rescue._SPLIT_IDS.set(frozenset())
    hybrid_composite_best_take._COMPOSITE_SPLIT_IDS.set(frozenset())
    return composite_ids


def apply_composite_resolution(
    takes: Iterable[CandidateTake], context, editorial_judge,
) -> tuple[HybridSessionCleanupResult, frozenset[str]]:
    """Run take-level restoration/rescue/composite marking."""
    chain = _get_take_level_chain()
    result = chain(tuple(takes), context, editorial_judge)
    split_ids = _composite_split_ids()
    result = _restore_semantic_compute_plan(result)
    return result, split_ids


def _restore_semantic_compute_plan(result: HybridSessionCleanupResult) -> HybridSessionCleanupResult:
    """D-053 Section 11: several of the 19 chain hooks reconstruct
    HybridSessionCleanupResult via a keyword constructor call that never
    named ``semantic_compute_plan`` (written before that field existed),
    silently reverting it to the dataclass default (None) the moment such
    a hook actually fires -- discovered live in the D-052 stability
    battery (top-level diagnostics showed "absent_flag_off" even though
    the flag was on and the per-window planner fields proved the planner
    had genuinely run). Read the same base-call plan back via the
    ContextVar side-channel hybrid_session_cleanup.py sets unconditionally
    (mirroring _composite_split_ids's own pattern above) and reattach it
    only if the chain's own result lost it -- never overwrites a plan a
    hook actually preserved correctly, and never touches kept/deleted/
    diagnostics/semantic_decisions or any other field, so this cannot
    change which takes were kept, discarded, or how they got there."""
    from . import hybrid_session_cleanup

    plan = hybrid_session_cleanup._LAST_SEMANTIC_COMPUTE_PLAN.get()
    hybrid_session_cleanup._LAST_SEMANTIC_COMPUTE_PLAN.set(None)
    if result.semantic_compute_plan is None and plan is not None:
        result = dataclasses.replace(result, semantic_compute_plan=plan)
    return result


def _split_groups_for_composite(
    groups: Iterable[Iterable[str]],
    split_ids: set[str] | frozenset[str],
    natural_ids: Iterable[str],
) -> tuple[tuple[str, ...], ...]:
    """Force true composite-marked clips into singleton Best-Take groups."""
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
    """Apply true composite split ids to a grouping result."""
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
    """Apply the downstream complementary-family stabilizer."""
    return apply_post_selection_complementary_family_stabilizer(draft)
