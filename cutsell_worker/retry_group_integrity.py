"""Preserve proven retry envelopes after overbroad-story splitting.

The Benchmark 44 story-chain guard correctly prevents substantive story paragraphs from
becoming one transitive Best-Take family.  A later regression showed that the same split
can strand tiny false starts which earlier reconciliation had already placed around a
proven retry family.  This final grouping pass is deliberately asymmetric: it may attach
weak/restart-only groups to an already-proven multi-member retry family, but weak speech
can never merge two substantive families together.
"""
from __future__ import annotations


def _group_members(group, take_map):
    return tuple(
        sorted(
            (take_map[clip_id] for clip_id in group if clip_id in take_map),
            key=lambda take: (take.source_order, take.start, take.end, take.clip_id),
        )
    )


def _temporal_gap(left, right) -> float:
    if left.source_asset_id != right.source_asset_id:
        return float("inf")
    if left.end < right.start:
        return max(0.0, right.start - left.end)
    if right.end < left.start:
        return max(0.0, left.start - right.end)
    return 0.0


def _restore_weak_retry_envelopes(groups, takes, *, maximum_gap_sec: float = 30.0):
    """Attach weak debris only to one already-proven retry family.

    A target must already contain at least two members and at least one substantive
    delivery.  A weak group can attach when it lies inside that family's time envelope
    (classic interstitial false-start debris), or when it has at least two meaningful
    shared/fuzzy content tokens within the normal long-form retry window.  If two target
    families tie, fail open and keep the weak group separate rather than bridging them.
    """
    from .local_retry_grouping import _restart_heavy, _shared_content_strength

    if len(groups) <= 1:
        return tuple(groups), False

    take_map = {take.clip_id: take for take in takes}
    work = [list(group) for group in groups]
    changed = False

    def members(index):
        return _group_members(work[index], take_map)

    weak_indices = []
    for index in range(len(work)):
        group_members = members(index)
        if group_members and all(_restart_heavy(item.text) for item in group_members):
            weak_indices.append(index)

    assignments = []
    for weak_index in weak_indices:
        weak_members = members(weak_index)
        if not weak_members:
            continue
        candidates = []
        for target_index in range(len(work)):
            if target_index == weak_index:
                continue
            target_members = members(target_index)
            if len(target_members) < 2:
                continue
            if not any(not _restart_heavy(item.text) for item in target_members):
                continue
            if weak_members[0].source_asset_id != target_members[0].source_asset_id:
                continue

            target_start = min(item.start for item in target_members)
            target_end = max(item.end for item in target_members)
            weak_start = min(item.start for item in weak_members)
            weak_end = max(item.end for item in weak_members)
            interstitial = target_start <= weak_start and weak_end <= target_end
            gap = min(
                _temporal_gap(weak_item, target_item)
                for weak_item in weak_members
                for target_item in target_members
            )
            shared = max(
                _shared_content_strength(weak_item.text, target_item.text)
                for weak_item in weak_members
                for target_item in target_members
            )
            if not interstitial and (gap > maximum_gap_sec or shared < 2):
                continue
            candidates.append(((1 if interstitial else 0, shared, -gap), target_index))

        if not candidates:
            continue
        candidates.sort(reverse=True)
        best_score, best_target = candidates[0]
        if len(candidates) > 1 and candidates[1][0] == best_score:
            continue
        assignments.append((weak_index, best_target))

    if not assignments:
        return tuple(tuple(group) for group in work), False

    # Apply from the original index space, allowing several weak groups to join one
    # target without ever joining target families to each other.
    absorbed = set()
    for weak_index, target_index in assignments:
        if weak_index == target_index or weak_index in absorbed:
            continue
        work[target_index].extend(work[weak_index])
        absorbed.add(weak_index)
        changed = True

    normalized = []
    for index, group in enumerate(work):
        if index in absorbed:
            continue
        ordered = tuple(
            sorted(
                set(group),
                key=lambda clip_id: (
                    take_map[clip_id].source_order,
                    take_map[clip_id].start,
                    take_map[clip_id].end,
                    clip_id,
                ),
            )
        )
        if ordered:
            normalized.append(ordered)
    normalized.sort(
        key=lambda group: (
            take_map[group[0]].source_order,
            take_map[group[0]].start,
            take_map[group[0]].end,
            group[0],
        )
    )
    return tuple(normalized), changed


def install_retry_group_integrity() -> None:
    from . import take_grouping_provider as grouping

    original = grouping.safe_group_takes
    if getattr(original, "_cutsell_retry_group_integrity", False):
        return

    def safe_group_takes_with_retry_integrity(provider, takes, context_text=""):
        result = original(provider, takes, context_text=context_text)
        if provider is not None or len(takes) <= 1:
            return result
        groups, changed = _restore_weak_retry_envelopes(result.groups, takes)
        if not changed:
            return result
        return grouping.TakeGroupingProviderResult(
            groups=groups,
            status=result.status,
            reason="; ".join(part for part in (result.reason, "weak_retry_envelope_restored") if part),
        )

    safe_group_takes_with_retry_integrity._cutsell_retry_group_integrity = True
    grouping.safe_group_takes = safe_group_takes_with_retry_integrity
