"""Find failed-attempt/clean-retry pairs worth a bounded semantic comparison.

This module grants no deletion or grouping authority.  It only recovers a
candidate pair that ordinary temporal discovery can miss when several pieces
of retry debris separate a failed delivery from its later clean realization.
The semantic arbiter still has to prove directional coverage and the absence
of a meaning conflict before the pair can become one retry family.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable, Mapping

from .contracts import CandidateTake
from .hybrid_retry_winner_authority import _same_retry_attempt


def _internal_restart_suffix_covered(anchor: CandidateTake, delivery: CandidateTake) -> bool:
    """Recognize a spoken restart inside one ASR take, for comparison only.

    A long pause/restart is sometimes retained in the same candidate and the
    candidate is consequently marked ``complete_idea=True`` even though its
    final clause is an abandoned attempt.  When an explicit ellipsis marks
    that restart, admit the trailing clause only if nearly its entire word
    sequence occurs, in order, inside the later complete delivery.  This
    grants no deletion authority; the semantic directional-coverage arbiter
    still has to prove that the later delivery preserves the failed attempt.
    """
    text = str(anchor.text or "")
    marker = max(text.rfind("..."), text.rfind("…"))
    if marker < 0 or not delivery.complete_idea:
        return False
    from .hybrid_retry_winner_authority import _content, _numbers, _tokens
    suffix_words = _tokens(text[marker + (1 if text[marker] == "…" else 3):])
    delivery_words = _tokens(delivery.text)
    suffix_content = _content(" ".join(suffix_words))
    if len(suffix_words) < 6 or len(suffix_content) < 2:
        return False
    left_numbers = _numbers(" ".join(suffix_words))
    right_numbers = _numbers(delivery.text)
    if left_numbers and right_numbers and left_numbers != right_numbers:
        return False
    # Ordered-subsequence coverage tolerates words inserted by the cleaner,
    # fuller retake but never reordered or substituted words.
    cursor = 0
    matched = 0
    for word in suffix_words:
        while cursor < len(delivery_words) and delivery_words[cursor] != word:
            cursor += 1
        if cursor >= len(delivery_words):
            continue
        matched += 1
        cursor += 1
    return matched >= 6 and matched / len(suffix_words) >= 0.80 and suffix_content.issubset(_content(delivery.text))


def failed_retry_coverage_pairs(
    takes: Iterable[CandidateTake],
    session_diagnostics: Iterable[dict],
    *,
    partition_by_id: Mapping[str, int],
    retry_groups: Iterable[Iterable[str]] = (),
    relation_takes: Iterable[CandidateTake] = (),
    relation_partition_by_id: Mapping[str, int] | None = None,
) -> frozenset[tuple[str, str]]:
    """Return directed ``(failed, later_delivery)`` comparison candidates.

    Every returned pair already has strong *eligibility* evidence, but no
    authority: the earlier take has a dominant failed decision and is locally
    corroborated; the
    later complete take is either consistently proposed as the winner or is
    a substantially fuller, non-failed delivery candidate. It carries no
    semantic deletion or word-scoped recording process finding, and both
    belong to the same creator-session partition.
    A conservative exact-attempt relation remains mandatory.  The caller must
    still obtain explicit semantic no-conflict + directional-coverage proof.
    """
    take_tuple = tuple(takes)
    by_id = {take.clip_id: take for take in take_tuple}
    relation_by_id = dict(by_id)
    relation_by_id.update({take.clip_id: take for take in relation_takes})
    relation_partitions = relation_partition_by_id or partition_by_id
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
                if str(clip_id) in relation_by_id:
                    recorded_partitions[str(clip_id)].add(partition)
        for row in window.get("decisions") or ():
            clip_id = str(row.get("clip_id") or "")
            if clip_id in relation_by_id:
                rows_by_id[clip_id].append(row)

    def same_creator_session(
        clip_ids: Iterable[str],
        heuristic_partitions: Mapping[str, int],
    ) -> bool:
        """Resolve session identity without letting a reduced-pool split win.

        Whole-session hybrid diagnostics preserve the creator-session identity
        observed before discarded takes reshape the retry pool.  When every
        member has one identical recorded partition, that evidence is more
        authoritative than a later heuristic repartition.  Ambiguous,
        incomplete, or contradictory recorded evidence never overrides the
        heuristic and therefore fails closed when the heuristic also splits.
        """
        ids = tuple(clip_ids)
        recorded = tuple(recorded_partitions.get(clip_id, set()) for clip_id in ids)
        if any(recorded):
            if not all(len(partitions) == 1 for partitions in recorded):
                return False
            if len({next(iter(partitions)) for partitions in recorded}) != 1:
                return False
            return True
        heuristic = tuple(heuristic_partitions.get(clip_id) for clip_id in ids)
        return all(partition is not None for partition in heuristic) and len(set(heuristic)) == 1

    failed_ids: list[str] = []
    delivery_ids: list[str] = []
    fallback_delivery_ids: set[str] = set()
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
            and row.get("local_failure_corroborated") is True
            and row.get("dense_semantic_failure_cluster") is True
            and float(row.get("confidence") or 0.0) >= 0.80
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
        if (
            clip_id in by_id
            and failed_votes
            and not unsafe_view
            and strongest_failed > protective_confidence
        ):
            failed_ids.append(clip_id)
        strict_winner = rows and all(
            row.get("proposed_label") == "winner"
            and float(row.get("confidence") or 0.0) >= 0.90
            and row.get("semantic_delete_recommended") is not True
            and not (row.get("recording_word_ranges") or ())
            and int(row.get("recording_prefix_words") or 0) == 0
            and int(row.get("recording_suffix_words") or 0) == 0
            and row.get("label") not in {"failed", "bts"}
            for row in rows
        ) and clip_id in by_id and any(
            float(row.get("confidence") or 0.0) >= 0.95 for row in rows
        )
        # D-303: a noisy contextual view may call the genuinely complete
        # later delivery ALTERNATE even while another view correctly marks
        # the earlier fragment FAILED.  Permit that later take to be *asked
        # about* only when every view remains non-destructive and free of
        # scoped recording-process words.  Substantial fullness, exact retry
        # relation, chronology, session identity and semantic directional
        # coverage are still checked below/beyond this function.
        fallback_delivery = (
            clip_id in by_id
            and by_id[clip_id].complete_idea
            and rows
            and all(
                row.get("label") not in {"failed", "bts"}
                and row.get("content_role") != "recording_only"
                and row.get("semantic_delete_recommended") is not True
                and not (row.get("recording_word_ranges") or ())
                and int(row.get("recording_prefix_words") or 0) == 0
                and int(row.get("recording_suffix_words") or 0) == 0
                for row in rows
            )
        )
        if strict_winner or fallback_delivery:
            delivery_ids.append(clip_id)
            if fallback_delivery and not strict_winner:
                fallback_delivery_ids.add(clip_id)

    pairs: set[tuple[str, str]] = set()
    for failed_id in failed_ids:
        failed = by_id[failed_id]
        # Failure evidence and lexical relation need not land on the same
        # member of an already-established retry family. One window may see a
        # long member as FAILED while a shorter sibling preserves the literal
        # phrase that matches the later clean delivery. The family supplies
        # eligibility; a locally corroborated, non-winner/non-BTS member may
        # supply the exact-attempt anchor. This still grants comparison only.
        group_anchor_ids = group_by_id.get(failed_id, frozenset({failed_id}))
        anchor_ids = set(group_anchor_ids)
        if not failed.complete_idea:
            anchor_ids.update(
                anchor_id for anchor_id, anchor in relation_by_id.items()
                if anchor_id != failed_id
                and anchor.source_asset_id == failed.source_asset_id
                and 0.0 <= anchor.start - failed.end <= 4.0
            )
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
            if (
                failed.source_asset_id != delivery.source_asset_id
                or delivery.start < failed.end
                or not same_creator_session(
                    (failed_id, delivery_id), partition_by_id,
                )
            ):
                continue
            for anchor_id in safe_anchor_ids:
                anchor = relation_by_id[anchor_id]
                if (
                    anchor.source_asset_id != delivery.source_asset_id
                    or anchor.source_asset_id != failed.source_asset_id
                    or delivery.start < anchor.end
                    or not delivery.complete_idea
                    or not same_creator_session(
                        (failed_id, anchor_id, delivery_id), relation_partitions,
                    )
                ):
                    continue
                if delivery_id in fallback_delivery_ids:
                    anchor_words = len(anchor.text.split())
                    delivery_words = len(delivery.text.split())
                    if (
                        (delivery.end - delivery.start) < max(
                            (anchor.end - anchor.start) * 1.5,
                            (anchor.end - anchor.start) + 3.0,
                        )
                        or delivery_words < max(anchor_words + 5, math.ceil(anchor_words * 1.5))
                    ):
                        continue
                same_attempt, relation_evidence = _same_retry_attempt(anchor, delivery)
                # D-304: a complete failed pitch can be substantially shorter
                # than its later clean realization. The ordinary literal
                # relation floor can then prevent even asking the structured
                # directional-coverage question. Admit only a large shared
                # proposition core with a materially fuller later delivery.
                shared_core_candidate = bool(
                    not same_attempt
                    and anchor.complete_idea
                    and delivery.complete_idea
                    and relation_evidence.get("numbers_ok") is True
                    and int(relation_evidence.get("shared_count") or 0) >= 8
                    and float(relation_evidence.get("failed_coverage") or 0.0) >= 0.30
                    and (
                        (delivery.end - delivery.start) >= (anchor.end - anchor.start) * 1.35
                        or len(delivery.text.split()) >= len(anchor.text.split()) + 12
                    )
                )
                internal_restart_candidate = bool(
                    not same_attempt
                    and _internal_restart_suffix_covered(anchor, delivery)
                )
                if same_attempt or shared_core_candidate or internal_restart_candidate:
                    left_id = anchor_id if anchor_id in group_anchor_ids else failed_id
                    pairs.add((left_id, delivery_id))
    return frozenset(pairs)
