"""Bind session-scoped grouping to the installed local retry reconciler.

``session_boundaries`` imports ``safe_group_takes`` by value. If that module is imported
before ``install_local_retry_grouping`` runs, it keeps the old baseline function even
after ``take_grouping_provider.safe_group_takes`` is wrapped. The production pipeline
then silently skips the local retry reconciliation that CI tests exercise directly.

This installer runs after the local reconciler and rebinds the session module to the
already-installed function. It changes no grouping thresholds and creates no deletions;
it only makes session-scoped production behavior match the intended local grouping path.
"""
from __future__ import annotations


def install_session_grouping_bridge() -> None:
    from . import session_boundaries, take_grouping_provider

    session_boundaries.safe_group_takes = take_grouping_provider.safe_group_takes
