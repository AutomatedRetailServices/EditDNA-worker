"""Preserve proven retry envelopes after overbroad-story splitting.

The Benchmark 44 story-chain guard correctly prevents substantive story paragraphs from
becoming one transitive Best-Take family.  A later regression showed that the same split
can strand tiny false starts which earlier reconciliation had already placed around a
proven retry family.  This final grouping pass is deliberately asymmetric: it may attach
weak/restart-only groups to an already-proven retry family, or to one clearly fuller
delivery under strict direct retry evidence, but weak speech can never merge two
substantive families together.
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


def _single_fuller_retry_target(
    weak_members,
    target_members,
    *,
    maximum_gap_sec: float = 15.0,
) -> bool:
    """Allow weak false starts to join one unmistakably fuller delivery.

    Story-chain splitting can leave a clean final delivery as a singleton while the
    preceding false starts remain in one or more weak-only groups. Requiring the target
    to already contain two members then strands those false starts and the composer must
    preserve them as independent speech.

    A singleton target is therefore eligible only when every weak member is materially
    shorter and all of its meaningful content is covered by the fuller delivery (fuzzy
    one-character recovery is allowed for ASR slips such as ``croc`` -> ``crop``). This
    is direct member-to-target evidence, not transitive grouping, so it cannot bridge two
    substantive story families.
    """
    from .local_retry_grouping import _content_tokens, _restart_heavy, _shared_content_strength

    if len(target_members) != 1 or not weak_members:
        return False
    target = target_members[0]
    if _restart_heavy(target.text):
        return False
    target_content = set(_content_tokens(target.text))
    if len(target_content) < 5:
        return False

    for weak in weak_members:
        if weak.source_asset_id != target.source_asset_id:
            return False
        if _temporal_gap(weak, target) > maximum_gap_sec:
            return False
        weak_content = set(_content_tokens(weak.text))
        if not 2 <= len(weak_content) <= 4:
            return False
        if len(target_content) < len(weak_content) + 3:
            return False
        if target.duration_sec < weak.duration_sec + 0.75:
            return False
        # Require coverage of every meaningful weak token. _shared_content_strength
        # counts exact tokens plus one-edit fuzzy matches, which is intentionally useful
        # for short ASR false starts while remaining much stricter than topic overlap.
        if _shared_content_strength(weak.text, target.text) < len(weak_content):
            return False
    return True


def _restore_weak_retry_envelopes(groups, takes, *, maximum_gap_sec: float = 30.0):
    """Attach weak debris only to one strongly supported retry family.

    Normal targets are already-proven retry families with at least two members and at
    least one substantive delivery. A weak group can attach when it lies inside that
    family's time envelope, or when it has at least two meaningful shared/fuzzy content
    tokens within the normal long-form retry window.

    A singleton substantive target is allowed only through ``_single_fuller_retry_target``
    so a clean final delivery can absorb its obvious false starts after story-chain
    splitting. If two targets tie, fail open and keep the weak group separate rather than
    bridging them.
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
            if not target_members:
                continue
            if not any(not _restart_heavy(item.text) for item in target_members):
                continue
            if weak_members[0].source_asset_id != target_members[0].source_asset_id:
                continue

            proven_family = len(target_members) >= 2
            direct_singleton = (
                len(target_members) == 1
                and _single_fuller_retry_target(weak_members, target_members)
            )
            if not proven_family and not direct_singleton:
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
            if proven_family and not interstitial and (gap > maximum_gap_sec or shared < 2):
                continue
            candidates.append((
                (
                    1 if interstitial else 0,
                    shared,
                    1 if proven_family else 0,
                    -gap,
                ),
                target_index,
            ))

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
