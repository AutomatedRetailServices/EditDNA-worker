"""Reconcile sibling retry families globally after session-scoped grouping.

Session boundaries protect compilation-style uploads by preventing ordinary retry reasoning
from crossing hard internal edits. That is correct for the local grouper, but it can also
strand two recordings of the same creator idea in separate mini-sessions. Benchmark 51
proved this with Video 00: the hereditary-cancer delivery and its split retry remained in
three independent groups even though final sibling reconciliation worked in unit tests.

This bridge keeps session grouping as the first authority, then performs one conservative
whole-source sibling reconciliation over the *final groups* immediately before Pipeline
imports/uses the result. It never deletes takes and never selects a winner; it only makes
true competing deliveries visible to the existing TakeJudge/Hybrid Best Take authority.
"""
from __future__ import annotations

from .final_sibling_grouping import reconcile_final_sibling_groups
from .take_grouping_provider import TakeGroupingProviderResult


def install_global_session_sibling_bridge() -> None:
    from . import session_boundaries

    original = session_boundaries.safe_group_takes_by_sessions
    if getattr(original, "_cutsell_global_session_sibling_bridge", False):
        return

    def safe_group_takes_by_sessions_with_global_siblings(
        provider,
        takes,
        context,
        *,
        context_text="",
    ):
        take_tuple = tuple(takes)
        result = original(
            provider,
            take_tuple,
            context,
            context_text=context_text,
        )
        groups, changed = reconcile_final_sibling_groups(result.groups, take_tuple)
        if not changed:
            return result
        reason = (result.reason + "; " if result.reason else "") + "global_post_session_sibling_reconciled"
        return TakeGroupingProviderResult(
            groups=groups,
            status=result.status,
            reason=reason,
        )

    safe_group_takes_by_sessions_with_global_siblings._cutsell_global_session_sibling_bridge = True
    session_boundaries.safe_group_takes_by_sessions = safe_group_takes_by_sessions_with_global_siblings
