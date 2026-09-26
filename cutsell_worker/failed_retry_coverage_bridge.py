"""Find failed-attempt/clean-retry pairs worth a bounded semantic comparison.

This module grants no deletion or grouping authority.  It only recovers a
candidate pair that ordinary temporal discovery can miss when several pieces
of retry debris separate a failed delivery from its later clean realization.
The semantic arbiter still has to prove directional coverage and the absence
of a meaning conflict before the pair can become one retry family.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from .contracts import CandidateTake
from .hybrid_retry_winner_authority import _same_retry_attempt


def failed_retry_coverage_pairs(
    takes: Iterable[CandidateTake],
    session_diagnostics: Iterable[dict],
    *,
    partition_by_id: Mapping[str, int],
) -> frozenset[tuple[str, str]]:
    """Return directed ``(failed, later_delivery)`` comparison candidates.

    Every returned pair already has strong *eligibility* evidence, but no
    authority: the earlier take is consistently failed, locally corroborated
    and has at least one explicit delete recommendation; the later complete take is consistently proposed
    as the winner, carries no semantic deletion or word-scoped recording
    process finding, and both belong to the same creator-session partition.
    A conservative exact-attempt relation remains mandatory.  The caller must
    still obtain explicit semantic no-conflict + directional-coverage proof.
    """
    take_tuple = tuple(takes)
    by_id = {take.clip_id: take for take in take_tuple}
    rows_by_id: dict[str, list[dict]] = defaultdict(list)
    recorded_partitions: dict[str, set[int]] = defaultdict(set)
    for window in session_diagnostics:
        partition = window.get("partition_index")
        if type(partition) is int:
            for clip_id in window.get("member_ids") or ():
                if str(clip_id) in by_id:
                    recorded_partitions[str(clip_id)].add(partition)
        for row in window.get("decisions") or ():
            clip_id = str(row.get("clip_id") or "")
            if clip_id in by_id:
                rows_by_id[clip_id].append(row)

    failed_ids: list[str] = []
    delivery_ids: list[str] = []
    for clip_id, rows in rows_by_id.items():
        # Overlapping whole-session windows may legitimately vary by a few
        # confidence points or omit the destructive recommendation in one
        # view.  Eligibility therefore follows the stable evidence: every
        # view calls the take failed and corroborates the same dense local
        # failure, at least one view reaches the established 0.85 semantic
        # floor and explicitly recommends deletion, and no view contradicts
        # that classification.  This only buys a comparison; directional
        # coverage from the semantic arbiter is still mandatory for removal.
        if (
            rows
            and all(
                row.get("label") == "failed"
                and float(row.get("confidence") or 0.0) >= 0.80
                and row.get("local_failure_corroborated") is True
                and row.get("dense_semantic_failure_cluster") is True
                for row in rows
            )
            and any(
                float(row.get("confidence") or 0.0) >= 0.85
                and row.get("semantic_delete_recommended") is True
                for row in rows
            )
        ):
            failed_ids.append(clip_id)
        if rows and all(
            row.get("proposed_label") == "winner"
            and float(row.get("confidence") or 0.0) >= 0.95
            and row.get("semantic_delete_recommended") is not True
            and not (row.get("recording_word_ranges") or ())
            and int(row.get("recording_prefix_words") or 0) == 0
            and int(row.get("recording_suffix_words") or 0) == 0
            and row.get("label") not in {"failed", "bts"}
            for row in rows
        ):
            delivery_ids.append(clip_id)

    pairs: set[tuple[str, str]] = set()
    for failed_id in failed_ids:
        failed = by_id[failed_id]
        for delivery_id in delivery_ids:
            delivery = by_id[delivery_id]
            failed_recorded = recorded_partitions.get(failed_id, set())
            delivery_recorded = recorded_partitions.get(delivery_id, set())
            if (
                failed.source_asset_id != delivery.source_asset_id
                or delivery.start < failed.end
                or not delivery.complete_idea
                or partition_by_id.get(failed_id) is None
                or partition_by_id.get(failed_id) != partition_by_id.get(delivery_id)
                or (bool(failed_recorded or delivery_recorded) and (
                    len(failed_recorded) != 1
                    or failed_recorded != delivery_recorded
                ))
            ):
                continue
            same_attempt, _ = _same_retry_attempt(failed, delivery)
            if same_attempt:
                pairs.add((failed_id, delivery_id))
    return frozenset(pairs)
