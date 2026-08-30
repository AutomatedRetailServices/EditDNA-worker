"""Final Story / Coherence Validation -- Clean Cut Core V1.

Runs after Best-Take authority resolves decisive retry-family contests and
before Selection Freeze. It is the last semantic authority allowed to touch
membership; Boundary must never repair a semantic membership mistake (see
CLAUDE.md / docs/CUTSELL_DECISIONS.md).

Clean Cut Core V1 product scope: SELECT/KEEP vs DISCARD only. SWAP is out of
scope until explicitly reintroduced. This module is also where that becomes
final and irreversible for the winning timeline: whatever the upstream
authorities left in ``alternates`` is folded into ``discarded`` here, because
nothing that isn't SELECT belongs in the one winning edit.

What this module checks, deterministically, from evidence the pipeline
already computed (no new heuristic invented for this pass):

  - KEEP/DISCARD only: alternates always folds into discarded.
  - Unresolved retry families: a take_judge_groups entry (a genuine retry
    contest -- 2+ ranked members) that STILL has 2+ members in the final
    selected set means Best-Take authority did not resolve it (the
    score gap was too thin to be decisive). This is exactly the "unresolved
    final-story coherence" case the architecture reserves for a bounded
    semantic arbiter, not a new guard: if a semantic_equivalence_arbiter is
    available, the residual members are asked pairwise whether they are the
    same intended idea; a confirmed match keeps only the take_judge's own
    top-ranked member and discards the rest. Fails open -- no arbiter, or an
    arbiter that cannot confidently confirm sameness, leaves the family
    exactly as it was, flagged in diagnostics for human review rather than
    silently resolved without evidence.
  - Missing story ending: flags (does not auto-restore) when the
    chronologically-last kept take in a source was discarded and nothing
    selected follows it in that source -- a possible dropped CTA/closing
    beat. Observability only; auto-restoring here would risk overriding a
    legitimate composer/review trim on no stronger evidence than position.

Not implemented in V1 (documented gap, not silently skipped): general
contradiction detection and exhaustive unique-fact-loss detection. Both need
capability this deterministic pass does not have; flagging that honestly is
preferable to a heuristic that would look like coverage it doesn't have.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations

from .semantic_idea_equivalence import (
    IdeaEquivalencePair,
    IdeaEquivalenceRequest,
    SemanticEquivalenceArbiter,
    same_idea_by_pair_index,
    safe_check_idea_equivalence,
)


def _fold_alternates_into_discarded(draft):
    if not draft.alternates:
        return draft
    discarded = tuple(
        sorted(
            (*draft.discarded, *(replace(clip, selected=False) for clip in draft.alternates)),
            key=lambda clip: (clip.source_order, clip.start, clip.end, clip.clip_id),
        )
    )
    return replace(draft, alternates=(), discarded=discarded)


def _residual_multi_select_groups(draft) -> list[dict]:
    selected_ids = {clip.clip_id for clip in draft.selected}
    groups = list((draft.diagnostics or {}).get("take_judge_groups") or ())
    residual = []
    for group in groups:
        ranked = list(group.get("ranked") or ())
        still_selected = [row for row in ranked if str(row.get("clip_id") or "") in selected_ids]
        if len(still_selected) >= 2:
            residual.append({**group, "ranked": ranked, "still_selected": still_selected})
    return residual


def _resolve_residual_family(
    group: dict,
    take_by_id: dict[str, object],
    arbiter: SemanticEquivalenceArbiter | None,
) -> tuple[list[str], list[dict]]:
    """Return (clip_ids_to_discard, audit_rows) for one still-ambiguous
    retry family. Empty on fail-open (no arbiter, or nothing confirmed)."""
    still_selected = group["still_selected"]
    if arbiter is None or len(still_selected) < 2:
        return [], []

    ordered = sorted(still_selected, key=lambda row: -float(row.get("score") or 0.0))
    pairs_meta = list(combinations(range(len(ordered)), 2))
    request = IdeaEquivalenceRequest(pairs=tuple(
        IdeaEquivalencePair(
            left_text=str(take_by_id.get(ordered[i]["clip_id"]).text if take_by_id.get(ordered[i]["clip_id"]) else ""),
            right_text=str(take_by_id.get(ordered[j]["clip_id"]).text if take_by_id.get(ordered[j]["clip_id"]) else ""),
        )
        for i, j in pairs_meta
    ))
    if not request.pairs:
        return [], []
    result = safe_check_idea_equivalence(arbiter, request)
    decisions = same_idea_by_pair_index(result)
    if not decisions:
        return [], []

    # Union-find over the ordered members: any confirmed same-idea pair
    # collapses to keeping only the higher-ranked (index 0 after sort) member
    # of that connected component.
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    audit: list[dict] = []
    any_confirmed = False
    for pair_index, (i, j) in enumerate(pairs_meta):
        decision = decisions.get(pair_index)
        if decision is None:
            continue
        same_idea, confidence, reason = decision
        if not same_idea:
            continue
        any_confirmed = True
        ra, rb = find(i), find(j)
        if ra != rb:
            keeper = min(ra, rb)
            loser = max(ra, rb)
            parent[loser] = keeper
        audit.append({
            "left_clip_id": ordered[i]["clip_id"],
            "right_clip_id": ordered[j]["clip_id"],
            "confidence": round(confidence, 4),
            "reason": reason,
        })

    if not any_confirmed:
        return [], []

    clusters: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        clusters.setdefault(find(index), []).append(index)

    to_discard: list[str] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        # ordered is already sorted by score descending, so the
        # lowest-index member of each cluster is the take_judge's own
        # top-ranked pick within it.
        keeper_index = min(members)
        for member_index in members:
            if member_index != keeper_index:
                to_discard.append(ordered[member_index]["clip_id"])

    return to_discard, audit


def apply_final_story_coherence_validation(
    draft, *, semantic_equivalence_arbiter: SemanticEquivalenceArbiter | None = None,
):
    """Last semantic authority before Selection Freeze. See module docstring."""
    draft = _fold_alternates_into_discarded(draft)

    residual = _residual_multi_select_groups(draft)
    take_by_id = {clip.clip_id: clip for clip in (*draft.selected, *draft.discarded)}

    resolved_families: list[dict] = []
    unresolved_families: list[dict] = []
    discard_ids: set[str] = set()

    for group in residual:
        to_discard, audit = _resolve_residual_family(group, take_by_id, semantic_equivalence_arbiter)
        if to_discard:
            discard_ids.update(to_discard)
            resolved_families.append({
                "group_id": group.get("group_id"),
                "discarded_clip_ids": to_discard,
                "merges": audit,
            })
        else:
            unresolved_families.append({
                "group_id": group.get("group_id"),
                "still_selected_clip_ids": [row["clip_id"] for row in group["still_selected"]],
            })

    if discard_ids:
        keep_selected = tuple(
            clip for clip in draft.selected if clip.clip_id not in discard_ids
        )
        newly_discarded = tuple(
            replace(clip, selected=False)
            for clip in draft.selected
            if clip.clip_id in discard_ids
        )
        discarded = tuple(sorted(
            (*draft.discarded, *newly_discarded),
            key=lambda clip: (clip.source_order, clip.start, clip.end, clip.clip_id),
        ))
        draft = replace(draft, selected=keep_selected, discarded=discarded)

    # Missing-story-ending observability check: the chronologically-last kept
    # take (by source_order/start across everything that survived attempt
    # reconstruction) was discarded and nothing selected follows it in that
    # source. Flag only -- never auto-restore on position evidence alone.
    all_takes = sorted(
        (*draft.selected, *draft.discarded),
        key=lambda clip: (clip.source_order, clip.start, clip.end, clip.clip_id),
    )
    possible_missing_ending = False
    if all_takes:
        selected_ids = {clip.clip_id for clip in draft.selected}
        last_by_source: dict[str, object] = {}
        for clip in all_takes:
            last_by_source[clip.source_asset_id] = clip
        for last_clip in last_by_source.values():
            if last_clip.clip_id not in selected_ids:
                possible_missing_ending = True
                break

    diagnostics = dict(draft.diagnostics or {})
    diagnostics["final_story_coherence_validation"] = {
        "status": "applied",
        "alternates_folded_into_discard": True,
        "residual_family_count": len(residual),
        "resolved_family_count": len(resolved_families),
        "resolved_families": resolved_families,
        "unresolved_family_count": len(unresolved_families),
        "unresolved_families": unresolved_families,
        "possible_missing_story_ending": possible_missing_ending,
        "not_implemented": [
            "general_contradiction_detection",
            "exhaustive_unique_fact_loss_detection",
        ],
    }
    return replace(draft, diagnostics=diagnostics)
