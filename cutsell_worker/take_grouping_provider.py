"""Provider boundary for semantic retry/take grouping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

from .contracts import CandidateTake
from .providers import ProviderStatus
from .semantic_idea_equivalence import (
    IdeaEquivalencePair,
    IdeaEquivalenceRequest,
    SemanticEquivalenceArbiter,
    SemanticEquivalenceGatePolicy,
    safe_check_idea_equivalence,
    same_idea_by_pair_index,
)
from .take_grouping import group_takes, retry_similarity, semantic_key


@dataclass(frozen=True)
class TakeGroupingProviderResult:
    groups: Tuple[Tuple[str, ...], ...]
    status: ProviderStatus
    reason: str = ""


class TakeGroupingProvider(Protocol):
    def group(
        self,
        takes: Tuple[CandidateTake, ...],
        context_text: str = "",
    ) -> TakeGroupingProviderResult: ...


def _baseline_groups(takes: Tuple[CandidateTake, ...]) -> Tuple[Tuple[str, ...], ...]:
    grouped = group_takes(takes)
    return tuple(tuple(item.clip_id for item in members) for members in grouped.values())


def _provider_members_compatible(left: CandidateTake, right: CandidateTake) -> bool:
    if left.source_asset_id != right.source_asset_id:
        return False
    score = retry_similarity(left.text, right.text)
    gap = max(0.0, max(left.start, right.start) - min(left.end, right.end))
    if gap <= 8.0:
        return score >= 0.72
    return score >= 0.82


def _constrain_provider_group(
    group: Tuple[str, ...],
    take_map: dict[str, CandidateTake],
) -> Tuple[Tuple[str, ...], ...]:
    """Split provider groups using complete-link retry compatibility."""
    members = [take_map[clip_id] for clip_id in group if clip_id in take_map]
    members.sort(key=lambda take: (take.source_order, take.start, take.end, take.clip_id))
    if len(members) <= 1:
        return (tuple(take.clip_id for take in members),) if members else ()

    clusters: list[list[CandidateTake]] = []
    for take in members:
        placed = False
        for cluster in clusters:
            if all(_provider_members_compatible(take, existing) for existing in cluster):
                cluster.append(take)
                placed = True
                break
        if not placed:
            clusters.append([take])
    return tuple(tuple(take.clip_id for take in cluster) for cluster in clusters)


def _repair_groups(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
) -> tuple[Tuple[Tuple[str, ...], ...], bool]:
    natural_ids = tuple(take.clip_id for take in takes)
    take_map = {take.clip_id: take for take in takes}
    allowed = set(natural_ids)
    seen: set[str] = set()
    repaired = False
    normalized: list[Tuple[str, ...]] = []

    for raw_group in groups:
        kept: list[str] = []
        for raw_id in raw_group:
            clip_id = str(raw_id)
            if clip_id not in allowed or clip_id in seen:
                repaired = True
                continue
            seen.add(clip_id)
            kept.append(clip_id)
        if kept:
            constrained = _constrain_provider_group(tuple(kept), take_map)
            if len(constrained) > 1:
                repaired = True
            normalized.extend(group for group in constrained if group)
        elif raw_group:
            repaired = True

    for clip_id in natural_ids:
        if clip_id not in seen:
            normalized.append((clip_id,))
            seen.add(clip_id)
            repaired = True

    return tuple(normalized), repaired


def _group_gap(
    left_group: Tuple[str, ...],
    right_group: Tuple[str, ...],
    take_map: dict[str, CandidateTake],
) -> float:
    gaps = []
    for left_id in left_group:
        left = take_map[left_id]
        for right_id in right_group:
            right = take_map[right_id]
            if left.source_asset_id != right.source_asset_id:
                continue
            gaps.append(max(0.0, max(left.start, right.start) - min(left.end, right.end)))
    return min(gaps) if gaps else float("inf")


def _reconcile_similarity_threshold(group_gap_sec: float) -> float:
    if group_gap_sec <= 8.0:
        return 0.80
    if group_gap_sec <= 30.0:
        return 0.90
    return 0.97


def _is_prefix_fragment(fragment: CandidateTake, reference: CandidateTake) -> bool:
    fragment_tokens = semantic_key(fragment.text).split()
    reference_tokens = semantic_key(reference.text).split()
    if not fragment_tokens or len(fragment_tokens) > 8:
        return False
    return reference_tokens[: len(fragment_tokens)] == fragment_tokens


def _is_material_prefix_fragment(fragment: CandidateTake, reference: CandidateTake) -> bool:
    """Recognize exact false-start prefixes even when completeness heuristics overstate them.

    ``complete_idea`` intentionally fails open for longer speech, so a seven-word
    false start can be marked complete merely because it crossed the length threshold.
    For retry reconciliation only, an exact prefix can be treated as non-substantive
    when it is materially shorter in both words and time. The fragment remains in the
    group for Best Take; this only prevents it from vetoing two strong full retries.
    """
    if not _is_prefix_fragment(fragment, reference):
        return False
    fragment_tokens = semantic_key(fragment.text).split()
    reference_tokens = semantic_key(reference.text).split()
    return (
        len(reference_tokens) - len(fragment_tokens) >= 3
        and fragment.duration_sec + 0.75 <= reference.duration_sec
    )


def _substantive_reconcile_members(members: list[CandidateTake]) -> list[CandidateTake]:
    """Exclude only structural prefix debris from complete-link retry comparison.

    A provider may already place a false-start prefix beside its full attempt. That
    debris should remain in the group for Best Take, but must not veto reconciliation
    with another near-identical full attempt. Because ``complete_idea`` is deliberately
    fail-open for longer speech, a materially shorter exact prefix may be neutralized
    here even when that heuristic marked it complete.
    """
    substantive: list[CandidateTake] = []
    for candidate in members:
        prefix_debris = any(
            other.clip_id != candidate.clip_id
            and (
                (not candidate.complete_idea and _is_prefix_fragment(candidate, other))
                or _is_material_prefix_fragment(candidate, other)
            )
            for other in members
        )
        if not prefix_debris:
            substantive.append(candidate)
    return substantive or members


def _groups_should_reconcile(
    left_group: Tuple[str, ...],
    right_group: Tuple[str, ...],
    take_map: dict[str, CandidateTake],
) -> bool:
    """Use group-level timing and complete-link over substantive retry attempts."""
    left_members = []
    right_members = []
    for clip_id in left_group:
        take = take_map.get(clip_id)
        if take is None:
            return False
        left_members.append(take)
    for clip_id in right_group:
        take = take_map.get(clip_id)
        if take is None:
            return False
        right_members.append(take)
    if not left_members or not right_members:
        return False
    source_ids = {take.source_asset_id for take in (*left_members, *right_members)}
    if len(source_ids) != 1:
        return False

    threshold = _reconcile_similarity_threshold(_group_gap(left_group, right_group, take_map))
    left_substantive = _substantive_reconcile_members(left_members)
    right_substantive = _substantive_reconcile_members(right_members)
    return all(
        retry_similarity(left.text, right.text) >= threshold
        for left in left_substantive
        for right in right_substantive
    )


def _reconcile_missed_retries(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
) -> tuple[Tuple[Tuple[str, ...], ...], bool]:
    if len(groups) <= 1:
        return groups, False
    take_map = {take.clip_id: take for take in takes}
    merged: list[list[str]] = []
    changed = False

    for group in groups:
        target_index = None
        for index, existing in enumerate(merged):
            if _groups_should_reconcile(tuple(existing), group, take_map):
                target_index = index
                break
        if target_index is None:
            merged.append(list(group))
        else:
            merged[target_index].extend(group)
            changed = True

    ordered_groups: list[Tuple[str, ...]] = []
    for group in merged:
        unique = {clip_id for clip_id in group}
        ordered = sorted(
            unique,
            key=lambda clip_id: (
                take_map[clip_id].source_order,
                take_map[clip_id].start,
                take_map[clip_id].end,
                clip_id,
            ),
        )
        ordered_groups.append(tuple(ordered))
    ordered_groups.sort(
        key=lambda group: (
            take_map[group[0]].source_order,
            take_map[group[0]].start,
            take_map[group[0]].end,
            group[0],
        )
    )
    return tuple(ordered_groups), changed


def _extend_adjacent_retry_groups(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
    *,
    max_gap_sec: float = 8.0,
    minimum_similarity: float = 0.93,
) -> tuple[Tuple[Tuple[str, ...], ...], bool]:
    """Extend a validated retry group only into the next near-verbatim attempt.

    This is intentionally narrower than general reconciliation. It may look past one
    singleton false-start when that false-start is an exact lexical prefix of the
    group's latest substantive attempt. The next substantive singleton must then be
    highly similar (>=0.93) and within eight seconds. This recovers serial retries
    without reintroducing broad transitive/topic chaining.
    """
    take_map = {take.clip_id: take for take in takes}
    ordered = tuple(sorted(takes, key=lambda item: (item.source_order, item.start, item.end, item.clip_id)))
    position = {take.clip_id: index for index, take in enumerate(ordered)}
    group_lists = [list(group) for group in groups]
    membership = {clip_id: index for index, group in enumerate(group_lists) for clip_id in group}
    changed = False

    for group_index, group in enumerate(group_lists):
        if len(group) < 2:
            continue
        members = sorted(
            (take_map[clip_id] for clip_id in group if clip_id in take_map),
            key=lambda item: (item.source_order, item.start, item.end, item.clip_id),
        )
        if not members:
            continue
        anchor = members[-1]
        anchor_pos = position.get(anchor.clip_id)
        if anchor_pos is None:
            continue

        pending_prefix: CandidateTake | None = None
        for candidate in ordered[anchor_pos + 1 :]:
            if candidate.source_asset_id != anchor.source_asset_id:
                break
            if candidate.start - anchor.end > max_gap_sec:
                break
            candidate_group_index = membership.get(candidate.clip_id)
            if candidate_group_index == group_index:
                continue
            if candidate_group_index is None or len(group_lists[candidate_group_index]) != 1:
                break

            if pending_prefix is None and len(candidate.text.split()) <= 3 and _is_prefix_fragment(candidate, anchor):
                pending_prefix = candidate
                continue

            if retry_similarity(anchor.text, candidate.text) < minimum_similarity:
                break

            if pending_prefix is not None:
                prefix_group_index = membership[pending_prefix.clip_id]
                group_lists[prefix_group_index].remove(pending_prefix.clip_id)
                group.append(pending_prefix.clip_id)
                membership[pending_prefix.clip_id] = group_index
            group_lists[candidate_group_index].remove(candidate.clip_id)
            group.append(candidate.clip_id)
            membership[candidate.clip_id] = group_index
            changed = True
            break

    normalized = []
    for group in group_lists:
        if not group:
            continue
        unique = sorted(
            set(group),
            key=lambda clip_id: (
                take_map[clip_id].source_order,
                take_map[clip_id].start,
                take_map[clip_id].end,
                clip_id,
            ),
        )
        normalized.append(tuple(unique))
    normalized.sort(
        key=lambda group: (
            take_map[group[0]].source_order,
            take_map[group[0]].start,
            take_map[group[0]].end,
            group[0],
        )
    )
    return tuple(normalized), changed


def _absorb_interstitial_retry_debris(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
    *,
    max_retry_span_sec: float = 15.0,
    max_fragment_sec: float = 2.5,
    max_fragment_words: int = 5,
    max_edge_gap_sec: float = 3.0,
) -> tuple[Tuple[Tuple[str, ...], ...], bool]:
    """Fold short incomplete speech trapped inside or directly beside a retry envelope."""
    take_map = {take.clip_id: take for take in takes}
    ordered = tuple(sorted(takes, key=lambda item: (item.source_order, item.start, item.end, item.clip_id)))
    group_lists = [list(group) for group in groups]
    membership = {clip_id: index for index, group in enumerate(group_lists) for clip_id in group}
    changed = False

    for group_index, group in enumerate(tuple(tuple(item) for item in group_lists)):
        if len(group) < 2:
            continue
        members = [take_map[clip_id] for clip_id in group if clip_id in take_map]
        members.sort(key=lambda item: (item.source_order, item.start, item.end, item.clip_id))
        for left, right in zip(members, members[1:]):
            if left.source_asset_id != right.source_asset_id:
                continue
            if right.start - left.end > max_retry_span_sec:
                continue
            for candidate in ordered:
                if membership.get(candidate.clip_id) == group_index:
                    continue
                if candidate.source_asset_id != left.source_asset_id:
                    continue
                if candidate.start < left.end or candidate.end > right.start:
                    continue
                if candidate.duration_sec > max_fragment_sec:
                    continue
                if len(candidate.text.split()) > max_fragment_words or candidate.complete_idea:
                    continue
                old_index = membership.get(candidate.clip_id)
                if old_index is None or len(group_lists[old_index]) != 1:
                    continue
                group_lists[old_index].remove(candidate.clip_id)
                group_lists[group_index].append(candidate.clip_id)
                membership[candidate.clip_id] = group_index
                changed = True

        first = members[0]
        first_pos = next((index for index, item in enumerate(ordered) if item.clip_id == first.clip_id), None)
        if first_pos is not None and first_pos > 0:
            candidate = ordered[first_pos - 1]
            old_index = membership.get(candidate.clip_id)
            gap = first.start - candidate.end
            if (
                old_index is not None
                and old_index != group_index
                and len(group_lists[old_index]) == 1
                and candidate.source_asset_id == first.source_asset_id
                and 0.0 <= gap <= max_edge_gap_sec
                and candidate.duration_sec <= max_fragment_sec
                and len(candidate.text.split()) <= max_fragment_words + 2
                and _is_prefix_fragment(candidate, first)
            ):
                group_lists[old_index].remove(candidate.clip_id)
                group_lists[group_index].append(candidate.clip_id)
                membership[candidate.clip_id] = group_index
                changed = True

    normalized = []
    for group in group_lists:
        if not group:
            continue
        unique = sorted(
            set(group),
            key=lambda clip_id: (
                take_map[clip_id].source_order,
                take_map[clip_id].start,
                take_map[clip_id].end,
                clip_id,
            ),
        )
        normalized.append(tuple(unique))
    normalized.sort(
        key=lambda group: (
            take_map[group[0]].source_order,
            take_map[group[0]].start,
            take_map[group[0]].end,
            group[0],
        )
    )
    return tuple(normalized), changed


def _cross_group_candidate_pairs(
    groups: Tuple[Tuple[str, ...], ...],
    take_map: dict[str, CandidateTake],
    *,
    maximum_gap_sec: float,
) -> tuple[tuple[int, int, str, str], ...]:
    pairs: list[tuple[int, int, str, str]] = []
    for left_index in range(len(groups)):
        for right_index in range(left_index + 1, len(groups)):
            left_group, right_group = groups[left_index], groups[right_index]
            if _group_gap(left_group, right_group, take_map) > maximum_gap_sec:
                continue
            for left_id in left_group:
                left_take = take_map.get(left_id)
                if left_take is None or len(semantic_key(left_take.text).split()) <= 3:
                    continue
                for right_id in right_group:
                    right_take = take_map.get(right_id)
                    if right_take is None or len(semantic_key(right_take.text).split()) <= 3:
                        continue
                    if left_take.source_asset_id != right_take.source_asset_id:
                        continue
                    pairs.append((left_index, right_index, left_id, right_id))
    return tuple(pairs)


def _raw_content_overlap(left_text: str, right_text: str) -> float:
    """Unfloored word-containment overlap, for RANKING only -- never a hard
    accept/reject gate. retry_similarity() deliberately floors to 0.0 below
    0.60 containment (see reconcile_semantic_idea_equivalence's docstring for
    why that makes it useless as an eligibility gate); this raw score keeps
    the same low-overlap paraphrases distinguishable from zero-overlap
    unrelated text purely as a priority signal."""
    left_tokens = set(semantic_key(left_text).split())
    right_tokens = set(semantic_key(right_text).split())
    if not left_tokens or not right_tokens:
        return 0.0
    shared = len(left_tokens & right_tokens)
    return shared / max(1, min(len(left_tokens), len(right_tokens)))


def _continuation_or_restart_bonus(left_take: CandidateTake, right_take: CandidateTake) -> float:
    """Boost pairs carrying existing continuation/restart evidence this
    codebase already computes on every CandidateTake: an incomplete delivery
    (complete_idea=False) or an exact lexical prefix relationship is a strong,
    general prior that two takes are the same attempt at different points of
    completion -- not a new heuristic, just reusing fields/helpers this module
    already has for other purposes."""
    bonus = 0.0
    if not left_take.complete_idea or not right_take.complete_idea:
        bonus += 0.25
    if _is_prefix_fragment(left_take, right_take) or _is_prefix_fragment(right_take, left_take):
        bonus += 0.25
    return bonus


def _pair_priority_score(
    left_take: CandidateTake, right_take: CandidateTake, *, gap_sec: float,
) -> float:
    """Composite ranking score for one candidate pair: temporal proximity +
    raw lexical/topical overlap + continuation/restart evidence. General and
    reusable -- no per-video tuning, no hardcoded thresholds beyond what the
    eligibility gate already enforces. Used only to decide WHICH eligible
    pairs get asked about first when there are more than the batch budget
    allows; never used to decide same_idea itself (that stays the arbiter's
    job, or the existing lexical reconciliation's)."""
    proximity = 1.0 / (1.0 + max(0.0, gap_sec))
    overlap = _raw_content_overlap(left_take.text, right_take.text)
    return proximity + overlap + _continuation_or_restart_bonus(left_take, right_take)


def _rank_candidate_pairs(
    pairs: tuple[tuple[int, int, str, str], ...],
    take_map: dict[str, CandidateTake],
) -> tuple[tuple[int, int, str, str], ...]:
    """Sort eligible candidate pairs by priority, highest first, so a fixed
    per-request pair budget spends its slots on the pairs most likely to be
    real retries instead of whichever happened to be enumerated first.

    This directly addresses the root cause an offline audit of a real run
    found: _cross_group_candidate_pairs enumerates ALL eligible group-index
    pairs in plain chronological order with no priority, so on any video
    dense enough to exceed the batch cap, coverage became a function of
    "where in iteration order did this pair land" rather than "how likely is
    this to be a real duplicate" -- pairs later in a long video were
    systematically less likely to ever be proposed to the arbiter at all,
    regardless of how obvious a retry they were. Ranking does not remove the
    batch cap or make this pairwise discovery exhaustive; it makes the
    truncation that DOES happen non-arbitrary.
    """
    scored = [
        (_pair_priority_score(take_map[left_id], take_map[right_id], gap_sec=_group_gap(
            (left_id,), (right_id,), take_map,
        )), pair)
        for pair in pairs
        for left_index, right_index, left_id, right_id in (pair,)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return tuple(pair for _, pair in scored)


def reconcile_semantic_idea_equivalence(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
    arbiter: SemanticEquivalenceArbiter | None,
    *,
    policy: SemanticEquivalenceGatePolicy = SemanticEquivalenceGatePolicy(),
    maximum_gap_sec: float = 30.0,
) -> tuple[Tuple[Tuple[str, ...], ...], dict]:
    """Merge groups the lexical layer left separate only when a narrow
    semantic arbiter is confident they are recording attempts of the same
    intended idea. Phase 2 of the architecture rebalance.

    Eligibility is temporal/structural, not a retry_similarity score band:
    a genuine paraphrase pair can score exactly 0.0 on that function's
    word-containment floor -- identical to genuinely unrelated text -- so
    no numeric similarity threshold reliably separates "ambiguous" from
    "definitely distinct" here (confirmed against real paraphrase fixtures;
    see semantic_idea_equivalence tests). A pair is eligible when both
    groups are (a) still separate after the full existing lexical
    reconciliation above, (b) from the same source, (c) within this
    module's own existing 30-second outer reconcile breakpoint
    (_reconcile_similarity_threshold's widest tier -- reused, not
    invented), and (d) both sides longer than retry_similarity's own
    existing short-phrase floor (<=3 tokens is already "not fuzzy-
    comparable" there).

    Fails open throughout: a pair the arbiter did not confidently confirm
    as the same idea leaves both groups exactly as they were.
    """
    if len(groups) < 2 or arbiter is None:
        return groups, {"status": "not_requested", "candidate_pair_count": 0, "merged_pair_count": 0}

    take_map = {take.clip_id: take for take in takes}
    candidate_pairs = _cross_group_candidate_pairs(groups, take_map, maximum_gap_sec=maximum_gap_sec)
    if not candidate_pairs:
        return groups, {"status": "no_eligible_pairs", "candidate_pair_count": 0, "merged_pair_count": 0}

    # Priority-ranked, not appearance-ordered: see _rank_candidate_pairs's
    # docstring for the root-cause finding this fixes. The full eligible set
    # is still bounded by the same structural gates above; only the order in
    # which the batch budget below gets spent changes.
    ranked_pairs = _rank_candidate_pairs(candidate_pairs, take_map)
    truncated = ranked_pairs[: policy.max_pairs_per_request]
    request = IdeaEquivalenceRequest(pairs=tuple(
        IdeaEquivalencePair(left_text=take_map[left_id].text, right_text=take_map[right_id].text)
        for _, _, left_id, right_id in truncated
    ))
    result = safe_check_idea_equivalence(arbiter, request, policy)
    decisions = same_idea_by_pair_index(result)

    # Union-find over group indices: if any member of group A is confirmed
    # the same idea as any member of group B, the two contests are one
    # retry family and their whole groups merge.
    parent = list(range(len(groups)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    audit: list[dict] = []
    merged_count = 0
    for pair_index, (left_group_index, right_group_index, left_id, right_id) in enumerate(truncated):
        decision = decisions.get(pair_index)
        if decision is None:
            continue  # fail-open: arbiter unavailable/declined -> preserve separate
        same_idea, confidence, reason = decision
        if not same_idea:
            continue
        union(left_group_index, right_group_index)
        merged_count += 1
        audit.append({
            "left_clip_id": left_id,
            "right_clip_id": right_id,
            "confidence": round(confidence, 4),
            "reason": reason,
        })

    if merged_count == 0:
        return groups, {
            "status": "checked_no_merge" if result.available else "arbiter_unavailable",
            "provider": result.provider,
            "candidate_pair_count": len(candidate_pairs),
            "checked_pair_count": len(truncated),
            "merged_pair_count": 0,
        }

    clusters: dict[int, list[str]] = {}
    for index, group in enumerate(groups):
        clusters.setdefault(find(index), []).extend(group)
    merged_groups = tuple(tuple(members) for members in clusters.values())

    return merged_groups, {
        "status": "applied",
        "provider": result.provider,
        "model": result.model,
        "candidate_pair_count": len(candidate_pairs),
        "checked_pair_count": len(truncated),
        "merged_pair_count": merged_count,
        "merges": audit,
    }


def safe_group_takes(
    provider: TakeGroupingProvider | None,
    takes: Tuple[CandidateTake, ...],
    context_text: str = "",
) -> TakeGroupingProviderResult:
    """Use semantic grouping while preserving every real candidate exactly once.

    Phase 2's semantic idea-equivalence pass (reconcile_semantic_idea_equivalence,
    below) deliberately runs OUTSIDE this function rather than being threaded
    through it: this codebase already layers several production monkeypatch
    wrappers over safe_group_takes and safe_group_takes_by_sessions (see
    final_sibling_grouping.py, session_grouping_bridge.py,
    global_session_sibling_bridge.py, local_retry_grouping.py,
    retry_group_integrity.py, hybrid_composite_best_take.py and friends), each
    hardcoding this function's current signature. Adding a new keyword here
    would silently stop propagating through every one of those wrappers in the
    real production call path -- exactly the class of regression several of
    those files' own docstrings describe fixing. pipeline.py instead calls
    reconcile_semantic_idea_equivalence directly on the final resolved groups,
    a single well-defined choke point immune to that layering.
    """
    baseline = _baseline_groups(takes)
    if provider is None or len(takes) <= 1:
        return TakeGroupingProviderResult(
            baseline,
            ProviderStatus("baseline", False, True, "lexical_fallback"),
            "baseline",
        )
    try:
        result = provider.group(takes, context_text=context_text)
        if not result.groups:
            raise ValueError("take grouping returned no groups")
        normalized_input = tuple(tuple(str(item) for item in group) for group in result.groups if group)
        repaired_groups, repaired = _repair_groups(normalized_input, takes)
        if not repaired_groups:
            raise ValueError("take grouping produced no valid candidates")
        reconciled_groups, reconciled = _reconcile_missed_retries(repaired_groups, takes)
        extended_groups, extended = _extend_adjacent_retry_groups(reconciled_groups, takes)
        final_groups, debris_absorbed = _absorb_interstitial_retry_debris(extended_groups, takes)
        reason = result.reason
        if repaired:
            reason = (reason + "; " if reason else "") + "provider_output_repaired"
        if reconciled:
            reason = (reason + "; " if reason else "") + "local_retry_reconciled"
        if extended:
            reason = (reason + "; " if reason else "") + "adjacent_retry_extended"
        if debris_absorbed:
            reason = (reason + "; " if reason else "") + "interstitial_retry_debris_absorbed"
        return TakeGroupingProviderResult(
            final_groups,
            ProviderStatus("openai", True, True, "applied"),
            reason,
        )
    except Exception as exc:
        return TakeGroupingProviderResult(
            baseline,
            ProviderStatus(
                provider=provider.__class__.__name__,
                requested=True,
                available=False,
                status="provider_error_fallback",
                reason=f"{exc.__class__.__name__}:{str(exc)[:160]}",
            ),
            "baseline_fallback",
        )
