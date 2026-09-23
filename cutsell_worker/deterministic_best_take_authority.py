"""Promote the deterministic take_judge Best-Take ranking to real authority for
clear-cut retry-family contests.

Phase 0/1 of the CutSell hybrid Selection rebalance (see the architecture map
in this session's transcript / docs/claude-handoff). Before this module, once
Unified Selection ran, the local retry-family ranking pipeline.py's
take_judge.rank_takes already computed for every multi-member group became
mere "evidence" the whole-video reasoner was free to overturn arbitrarily --
including selecting more than one member of the same contest, or keeping an
already-evidenced failed/incomplete fragment. This module makes Unified
Selection and the deterministic ranker sequential rather than the reasoner
having unconditional final say: for a retry-family contest the ranker was
genuinely decisive about, its verdict now becomes the final bucket
assignment; for a contest it was NOT decisive about (thin score gap), the
reasoner's decision is left completely untouched.

This module invents no new similarity/grouping/completeness heuristic of its
own -- Phase 2 (semantic idea-equivalence) is explicitly out of scope here.
It only reads:
  - diagnostics["take_judge_groups"], the RankedTake scores pipeline.py's
    take_judge.rank_takes already computed per retry-family group (only
    present for groups with 2+ members -- i.e. genuine retry contests,
    singletons are never touched);
  - the exact "decisive" threshold hybrid_take_judge.py already uses
    (top-two score gap >= 0.30 is exactly the gap at which that module's own
    conflict_score reaches 0, i.e. "no ambiguity") -- reused verbatim, not
    reinvented;
  - take_judge.rank_takes' own reason strings for a fragment it already
    proved is a failed/incomplete/abandoned delivery
    (material_prefix_fragment_penalty, repetitive_restart_fragment_penalty,
    restart_tail_fragment_penalty) -- reused verbatim as the evidence for
    "Failed/incomplete deliveries remain DISCARD when existing evidence
    supports that."

Ambiguity fails open throughout: any contest this module is not confident
about is left exactly as the upstream semantic authority decided it.
"""
from __future__ import annotations

from dataclasses import replace

# Matches hybrid_take_judge.build_editorial_session_from_group's own existing
# definition of "decisive": conflict_score there is 0 (no ambiguity) exactly
# when the top-two RankedTake score gap is >= 0.30. Reused verbatim rather
# than invented, per the architecture map's "do not add a new guard" rule.
CLEAR_WINNER_MINIMUM_GAP = 0.30

# take_judge.rank_takes' own reason strings for a fragment it already proved
# is a failed/incomplete/abandoned delivery. This module adds no new failure
# heuristic of its own -- it only reads this existing evidence.
_FAILURE_EVIDENCE_MARKERS = (
    "material_prefix_fragment_penalty",
    "repetitive_restart_fragment_penalty",
    "restart_tail_fragment_penalty",
)


def _looks_failed_or_incomplete(reason: object) -> bool:
    text = str(reason or "")
    return any(marker in text for marker in _FAILURE_EVIDENCE_MARKERS)


def clear_retry_family_winner(ranked: list[dict]) -> dict | None:
    """Return the winning ranked-take row iff this contest is decisive and its
    own winner is not itself evidenced as a failed/incomplete fragment.

    A "contest" requires at least two ranked members -- a singleton group is
    not a retry-family contest and is never touched by this module.
    """
    if len(ranked) < 2:
        return None
    ordered = sorted(ranked, key=lambda row: -float(row.get("score") or 0.0))
    gap = float(ordered[0].get("score") or 0.0) - float(ordered[1].get("score") or 0.0)
    if gap < CLEAR_WINNER_MINIMUM_GAP:
        return None
    if _looks_failed_or_incomplete(ordered[0].get("reason")):
        return None
    return ordered[0]


def apply_deterministic_best_take_authority(draft, *, swap_enabled: bool = False):
    """Lock in the deterministic ranker's verdict for every clear retry-family
    contest; leave every ambiguous (thin-gap) contest exactly as the upstream
    semantic authority decided it.

    ``swap_enabled`` controls where a legitimate (not failed/incomplete) losing
    retry goes. Clean Cut Core V1's product scope decision is KEEP/DISCARD
    only -- SWAP is out of scope until explicitly reintroduced (see CLAUDE.md
    and docs/CUTSELL_DECISIONS.md) -- so the default is False: a legitimate
    loser is DISCARDed, same as a failed/incomplete one, because it lost a
    decisive contest and does not belong in the one winning edit. Passing
    swap_enabled=True restores the pre-V1 behavior (legitimate losers become
    SWAP/alternates) for rollback/testing of the retired architecture only.
    """
    groups = list((draft.diagnostics or {}).get("take_judge_groups") or ())
    if not groups:
        return draft

    selected_by_id = {clip.clip_id: clip for clip in draft.selected}
    alternates_by_id = {clip.clip_id: clip for clip in draft.alternates}
    discarded_by_id = {clip.clip_id: clip for clip in draft.discarded}
    all_clips = {**selected_by_id, **alternates_by_id, **discarded_by_id}

    def bucket_of(clip_id: str) -> str:
        if clip_id in selected_by_id:
            return "select"
        if clip_id in alternates_by_id:
            return "swap"
        return "discard"

    moves: list[dict] = []
    new_selected = dict(selected_by_id)
    new_alternates = dict(alternates_by_id)
    new_discarded = dict(discarded_by_id)

    def move(clip_id: str, target: str, reason: str, extra: dict) -> None:
        origin = bucket_of(clip_id)
        if origin == target:
            return
        clip = all_clips[clip_id]
        new_selected.pop(clip_id, None)
        new_alternates.pop(clip_id, None)
        new_discarded.pop(clip_id, None)
        updated = replace(clip, selected=(target == "select"))
        {"select": new_selected, "swap": new_alternates, "discard": new_discarded}[target][clip_id] = updated
        moves.append({"clip_id": clip_id, "from_bucket": origin, "to_bucket": target, "reason": reason, **extra})

    for group in groups:
        if group.get("no_usable_realization"):
            # D-097.B: Best Take refused to force a winner for this family --
            # a clear local score gap among failed members is not one either.
            continue
        ranked = list(group.get("ranked") or ())
        winner_row = clear_retry_family_winner(ranked)
        if winner_row is None:
            continue
        winner_id = str(winner_row.get("clip_id") or "")
        if winner_id not in all_clips:
            continue
        group_id = group.get("group_id")

        move(winner_id, "select", "deterministic_clear_retry_family_winner", {"group_id": group_id})

        for row in ranked:
            clip_id = str(row.get("clip_id") or "")
            if clip_id == winner_id or clip_id not in all_clips:
                continue
            if bucket_of(clip_id) == "discard":
                # Already discarded by the upstream authority; never resurrect it.
                continue
            if _looks_failed_or_incomplete(row.get("reason")):
                move(
                    clip_id, "discard", "deterministic_failed_or_incomplete_evidence",
                    {"group_id": group_id, "winner_clip_id": winner_id, "ranker_reason": row.get("reason")},
                )
            elif swap_enabled:
                move(
                    clip_id, "swap", "deterministic_legitimate_alternate_not_additional_select",
                    {"group_id": group_id, "winner_clip_id": winner_id},
                )
            else:
                # Clean Cut Core V1: KEEP/DISCARD only. A legitimate loser of a
                # decisive retry-family contest still loses -- it does not
                # belong in the one winning edit, so it is discarded rather
                # than parked as an alternate/SWAP.
                move(
                    clip_id, "discard", "deterministic_legitimate_loser_no_swap_scope",
                    {"group_id": group_id, "winner_clip_id": winner_id},
                )

    if not moves:
        return draft

    def _order(clip):
        return (clip.source_order, float(clip.start), float(clip.end), clip.clip_id)

    selected = tuple(sorted(new_selected.values(), key=_order))
    alternates = tuple(sorted(new_alternates.values(), key=_order))
    discarded = tuple(sorted(new_discarded.values(), key=_order))

    diagnostics = dict(draft.diagnostics or {})
    diagnostics["deterministic_best_take_authority"] = {
        "status": "applied",
        "clear_winner_minimum_gap": CLEAR_WINNER_MINIMUM_GAP,
        "swap_enabled": swap_enabled,
        "moves": moves,
    }

    # D-122 (advisory/diagnostics only): a `deterministic_clear_retry_
    # family_winner` move proves this authority actually changed a
    # group's bucket assignment relative to whatever `_semantic_best_
    # take` decided upstream (`move()` is a no-op when the bucket was
    # already correct) -- the second, independent override path D-121
    # Section 5 identified. Annotate ONLY those groups' existing
    # `take_judge_groups` diagnostics rows (built additively by
    # pipeline.py) so `winner_path` reflects the real final path;
    # every other group's row is left byte-identical. This never reads
    # or changes `ranked`/`selected`/membership -- it only overwrites
    # two diagnostic fields on rows this same function's own `moves`
    # already proves it changed.
    overridden_group_ids = {
        str(m.get("group_id")) for m in moves
        if m.get("reason") == "deterministic_clear_retry_family_winner" and m.get("group_id") is not None
    }
    if overridden_group_ids:
        # D-123 (docs/CUTSELL_DECISIONS.md D-123): `winner_path_after`/
        # `final_winner` are the same "what actually happened" fields
        # `winner_path` already tracked for D-122 -- this authority is a
        # LATER stage than `_semantic_best_take`, so its own move is the
        # true final answer. `winner_path_before` is untouched: it is the
        # semantic-only counterfactual computed upstream and is unaffected
        # by this authority running afterward.
        winner_by_group = {
            str(m.get("group_id")): str(m.get("clip_id"))
            for m in moves
            if m.get("reason") == "deterministic_clear_retry_family_winner" and m.get("group_id") is not None
        }
        existing_groups = list(diagnostics.get("take_judge_groups") or ())
        patched_groups = []
        for row in existing_groups:
            if isinstance(row, dict) and str(row.get("group_id")) in overridden_group_ids:
                gid_key = str(row.get("group_id"))
                row = {
                    **row,
                    "winner_path": "DETERMINISTIC_OVERRIDE",
                    "performance_consulted_before_winner": True,
                    "winner_path_after": "DETERMINISTIC_OVERRIDE",
                    "final_winner": winner_by_group.get(gid_key, row.get("final_winner")),
                }
            patched_groups.append(row)
        diagnostics["take_judge_groups"] = patched_groups

    return replace(draft, selected=selected, alternates=alternates, discarded=discarded, diagnostics=diagnostics)
