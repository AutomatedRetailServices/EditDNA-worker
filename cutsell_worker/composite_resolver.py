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

Two of the 19 take-level hooks (17 and 19) can mark a pair of deliveries as
a composite: pieces that must survive Best-Take's one-winner competition
together rather than be collapsed back into a single retry contest. Each
tracks this via its own private ``ContextVar`` and its own monkeypatch of
``safe_group_takes_by_sessions``. This module reads both ContextVars right
after invoking the chain (rather than letting either hook's grouping
monkeypatch actually apply), combines them into one explicit ``split_ids``
return value, and exposes ONE ``apply_composite_group_split`` function
``pipeline.py`` calls directly after grouping -- no monkeypatch of
``session_boundaries`` survives this module's own setup.
"""
from __future__ import annotations

from typing import Callable, Iterable

from .contracts import CandidateTake
from .hybrid_session_cleanup import HybridSessionCleanupResult
from .post_selection_complementary_family_stabilizer import (
    apply_post_selection_complementary_family_stabilizer,
)

# Historical cutsell_worker/__init__.py order for every hook that wraps
# hybrid_session_cleanup.apply_hybrid_session_cleanup. See this module's
# docstring for how this list was actually verified (grep for every file
# referencing apply_hybrid_session_cleanup, not just files named
# hybrid_*/post_selection_*), not merely assumed from naming convention.
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
    """Build the take-level chain ONCE by calling each hook's own real
    install_*() function, unmodified, in the exact historical order, against
    the shared module attributes -- then restore those attributes to what
    they already were, so nothing global leaks beyond this module's own
    private cached reference. See module docstring for the full rationale.
    """
    import importlib

    from . import hybrid_session_cleanup, session_boundaries

    base_cleanup = hybrid_session_cleanup.apply_hybrid_session_cleanup
    base_grouping = session_boundaries.safe_group_takes_by_sessions

    for module_name, install_name in _CHAIN_SPEC:
        module = importlib.import_module(f".{module_name}", __package__)
        getattr(module, install_name)()

    chain = hybrid_session_cleanup.apply_hybrid_session_cleanup

    # Composite split-marking is owned explicitly by apply_composite_group_
    # split below (called by pipeline.py after grouping), not by either
    # hook's own grouping monkeypatch -- restore grouping to what it
    # already was (which still includes any OTHER, out-of-this-module's-
    # scope wrap, e.g. global_session_sibling_bridge's, applied earlier by
    # cutsell_worker/__init__.py before this function ever runs).
    session_boundaries.safe_group_takes_by_sessions = base_grouping
    hybrid_session_cleanup.apply_hybrid_session_cleanup = base_cleanup

    return chain


def _get_take_level_chain() -> Callable:
    global _take_level_chain
    if _take_level_chain is None:
        _take_level_chain = _build_take_level_chain()
    return _take_level_chain


def _composite_split_ids() -> frozenset[str]:
    """Read both hooks' private split-id ContextVars right after invoking
    the chain, and clear them so a later, unrelated call doesn't inherit a
    stale value. Reaching into another module's "private" ContextVar this
    way is the same pattern the original hybrid_semantic_composite_bridge.py
    already used to read hybrid_semantic_complementary_rescue's -- not a new
    coupling this module introduces."""
    from . import hybrid_composite_best_take, hybrid_semantic_complementary_rescue

    rescue_ids = frozenset(hybrid_semantic_complementary_rescue._SPLIT_IDS.get())
    composite_ids = frozenset(hybrid_composite_best_take._COMPOSITE_SPLIT_IDS.get())
    hybrid_semantic_complementary_rescue._SPLIT_IDS.set(frozenset())
    hybrid_composite_best_take._COMPOSITE_SPLIT_IDS.set(frozenset())
    return rescue_ids | composite_ids


def apply_composite_resolution(
    takes: Iterable[CandidateTake], context, editorial_judge,
) -> tuple[HybridSessionCleanupResult, frozenset[str]]:
    """The ONE CompositeResolver step for take-level restoration/rescue/composite
    marking. Returns ``(result, composite_split_ids)`` -- the split ids are
    for ``apply_composite_group_split`` to apply to the grouping result that
    runs immediately after this in ``pipeline.py``.

    Same 20 algorithms (base + 19 hooks), same historical order, as the old
    scattered monkeypatch chain -- see this module's docstring and D-023.
    """
    chain = _get_take_level_chain()
    result = chain(tuple(takes), context, editorial_judge)
    split_ids = _composite_split_ids()
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
    """Step 21: the one genuinely downstream CompositeResolver extension --
    operates on the built DraftTimeline, not raw takes. Called explicitly by
    pipeline.py at the end of build_flow_b_draft; no longer installed as a
    monkeypatch on ``pipeline.build_flow_b_draft`` itself (see D-023)."""
    return apply_post_selection_complementary_family_stabilizer(draft)
