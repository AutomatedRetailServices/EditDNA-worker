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
    retry_groups: Iterable[Iterable[str]] = (),
) -> frozenset[tuple[str, str]]:
    """Return directed ``(failed, later_delivery)`` comparison candidates.

    Every returned pair already has strong *eligibility* evidence, but no
    authority: the earlier take has a dominant failed decision, is locally
    corroborated and has at least one explicit delete recommendation; the
    later complete take is consistently proposed as the winner, carries no
    semantic deletion or word-scoped recording
    process finding, and both belong to the same creator-session partition.
    A conservative exact-attempt relation remains mandatory.  The caller must
    still obtain explicit semantic no-conflict + directional-coverage proof.
    """
    take_tuple = tuple(takes)
    by_id = {take.clip_id: take for take in take_tuple}
    group_by_id: dict[str, frozenset[str]] = {}
    for group in retry_groups:
        known = frozenset(str(clip_id) for clip_id in group if str(clip_id) in by_id)
        for clip_id in known:
            group_by_id[clip_id] = known
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
        # Overlapping whole-session windows can see the same take with
        # different context.  A lower-confidence KEEP/ALTERNATE observation
        # must not erase a stronger FAILED+delete observation. Equal or
        # stronger protective evidence still wins, as do WINNER/BTS or an
        # uncorroborated view. This only buys a comparison; directional
        # coverage from the semantic arbiter remains mandatory for removal.
        failed_votes = [
            row for row in rows
            if row.get("label") == "failed"
            and row.get("semantic_delete_recommended") is True
            and row.get("local_failure_corroborated") is True
            and row.get("dense_semantic_failure_cluster") is True
            and float(row.get("confidence") or 0.0) >= 0.85
        ]
        strongest_failed = max(
            (float(row.get("confidence") or 0.0) for row in failed_votes),
            default=0.0,
        )
        protective_confidence = max(
            (
                float(row.get("confidence") or 0.0)
                for row in rows
                if row.get("label") in {"keep", "alternate", "uncertain", "winner"}
                or row.get("proposed_label") == "winner"
            ),
            default=0.0,
        )
        unsafe_view = any(
            row.get("label") in {"winner", "bts"}
            or (
                row.get("label") == "failed"
                and float(row.get("confidence") or 0.0) < 0.80
            )
            or row.get("local_failure_corroborated") is not True
            or row.get("dense_semantic_failure_cluster") is not True
            for row in rows
        )
        if failed_votes and not unsafe_view and strongest_failed > protective_confidence:
            failed_ids.append(clip_id)
        if rows and all(
            row.get("proposed_label") == "winner"
            and float(row.get("confidence") or 0.0) >= 0.90
            and row.get("semantic_delete_recommended") is not True
            and not (row.get("recording_word_ranges") or ())
            and int(row.get("recording_prefix_words") or 0) == 0
            and int(row.get("recording_suffix_words") or 0) == 0
            and row.get("label") not in {"failed", "bts"}
            for row in rows
        ) and any(float(row.get("confidence") or 0.0) >= 0.95 for row in rows):
            delivery_ids.append(clip_id)

    pairs: set[tuple[str, str]] = set()
    for failed_id in failed_ids:
        failed = by_id[failed_id]
        failed_recorded = recorded_partitions.get(failed_id, set())
        # Failure evidence and lexical relation need not land on the same
        # member of an already-established retry family. One window may see a
        # long member as FAILED while a shorter sibling preserves the literal
        # phrase that matches the later clean delivery. The family supplies
        # eligibility; a locally corroborated, non-winner/non-BTS member may
        # supply the exact-attempt anchor. This still grants comparison only.
        anchor_ids = group_by_id.get(failed_id, frozenset({failed_id}))
        safe_anchor_ids = [
            anchor_id for anchor_id in anchor_ids
            if rows_by_id.get(anchor_id)
            and all(
                row.get("label") not in {"winner", "bts"}
                and row.get("proposed_label") != "winner"
                and row.get("local_failure_corroborated") is True
                and row.get("dense_semantic_failure_cluster") is True
                for row in rows_by_id[anchor_id]
            )
        ]
        for delivery_id in delivery_ids:
            delivery = by_id[delivery_id]
            delivery_recorded = recorded_partitions.get(delivery_id, set())
            if (
                failed.source_asset_id != delivery.source_asset_id
                or delivery.start < failed.end
                or partition_by_id.get(failed_id) is None
                or partition_by_id.get(failed_id) != partition_by_id.get(delivery_id)
                or (bool(failed_recorded or delivery_recorded) and (
                    len(failed_recorded) != 1
                    or failed_recorded != delivery_recorded
                ))
            ):
                continue
            for anchor_id in safe_anchor_ids:
                anchor = by_id[anchor_id]
                anchor_recorded = recorded_partitions.get(anchor_id, set())
                if (
                    anchor.source_asset_id != delivery.source_asset_id
                    or anchor.source_asset_id != failed.source_asset_id
                    or delivery.start < anchor.end
                    or not delivery.complete_idea
                    or partition_by_id.get(anchor_id) is None
                    or partition_by_id.get(anchor_id) != partition_by_id.get(delivery_id)
                    or partition_by_id.get(anchor_id) != partition_by_id.get(failed_id)
                    or (bool(anchor_recorded or delivery_recorded) and (
                        len(anchor_recorded) != 1
                        or anchor_recorded != delivery_recorded
                    ))
                    or (bool(anchor_recorded or failed_recorded) and (
                        len(anchor_recorded) != 1
                        or anchor_recorded != failed_recorded
                    ))
                ):
                    continue
                same_attempt, _ = _same_retry_attempt(anchor, delivery)
                if same_attempt:
                    pairs.add((anchor_id, delivery_id))
    return frozenset(pairs)
