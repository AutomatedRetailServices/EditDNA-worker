"""Enable full deterministic retry reconciliation on the RunPod-local path.

The provider boundary already contains conservative repair stages for missed retries,
adjacent retake extension, and interstitial false-start debris. Historically those
stages only ran after an external grouping provider returned output. Clean Cut now
runs with ``provider=None`` by design, so the local path stopped at seed lexical
groups and Best Take never got a chance to compare many obvious retakes.

This installer wraps the grouping boundary before ``pipeline`` imports it. It does not
change deletion thresholds and it never drops a candidate: it only makes retry groups
more complete so the existing Best Take ranking can choose one representative.
"""
from __future__ import annotations


def install_local_retry_grouping() -> None:
    from . import take_grouping_provider as grouping

    original = grouping.safe_group_takes
    if getattr(original, "_cutsell_local_retry_grouping", False):
        return

    def safe_group_takes_with_local_reconciliation(provider, takes, context_text=""):
        result = original(provider, takes, context_text=context_text)
        if provider is not None or len(takes) <= 1:
            return result

        groups, reconciled = grouping._reconcile_missed_retries(result.groups, takes)
        groups, extended = grouping._extend_adjacent_retry_groups(groups, takes)
        groups, debris_absorbed = grouping._absorb_interstitial_retry_debris(groups, takes)

        reasons = ["baseline_local"]
        if reconciled:
            reasons.append("local_retry_reconciled")
        if extended:
            reasons.append("adjacent_retry_extended")
        if debris_absorbed:
            reasons.append("interstitial_retry_debris_absorbed")

        return grouping.TakeGroupingProviderResult(
            groups=groups,
            status=grouping.ProviderStatus(
                "baseline",
                False,
                True,
                "local_retry_reconciled",
            ),
            reason="; ".join(reasons),
        )

    safe_group_takes_with_local_reconciliation._cutsell_local_retry_grouping = True
    grouping.safe_group_takes = safe_group_takes_with_local_reconciliation
