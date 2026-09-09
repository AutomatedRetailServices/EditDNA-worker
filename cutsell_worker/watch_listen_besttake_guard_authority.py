"""D-174 -- Watch+Listen BestTake Guard Authority.

Per docs/CUTSELL_DECISIONS.md D-123/D-163/D-167/D-170/D-172/D-173 and this
task's own Product Owner decision: a BOUNDED authority, authorized for the
first time in this session's history to actually change a family's
`selected_clip_id`, layered on top of D-172's own diagnostic-only V2
comparison.

## Core principle (this task's own binding instruction)

    WATCH+LISTEN MAY VETO A BAD WINNER.
    WATCH+LISTEN DOES NOT BECOME THE WINNER SELECTOR.

Exactly D-123's own philosophy, restated for this one additional step:
evidence may BLOCK an unsafe/poor structured winner, then EXISTING
structured authority (`deterministic_best_take_authority.clear_retry_
family_winner`, D-123's own already-CLOSED ladder, reused verbatim, never
re-derived) resumes to pick the actual replacement among the remaining
eligible candidates. This module never writes `winner_after = <V2 dominant
candidate>` directly -- see `apply_watch_listen_besttake_guard_authority`
below for the structural proof.

## Two-phase design (one canonical seam, no duplicated resolver)

Phase 1 -- `evaluate_watch_listen_besttake_guard_authority` (pure,
diagnostic, called from `pipeline.py`'s existing per-family loop,
immediately after D-172's own `evaluate_watch_listen_besttake_guard_v2`):
decides WHETHER this family's current winner should be rejected, from
fields D-163/D-167/D-172/D-123 ALREADY compute -- no new perception, no new
score, no new dominance algorithm. Never mutates anything.

Phase 2 -- `apply_watch_listen_besttake_guard_authority` (the ONE real
winner-mutation seam, called from `universal_clean_cut.py` immediately
after `deterministic_best_take_authority.apply_deterministic_best_take_
authority` -- the SAME existing place a family's real bucket assignment is
already finalized): for every family Phase 1 marked `GUARD_REJECT_CURRENT_
WINNER`, excludes the rejected winner from that family's own already-
computed `ranked` list and calls `deterministic_best_take_authority.
clear_retry_family_winner` -- the EXACT SAME existing, unchanged, already-
CLOSED deterministic ladder D-123 already runs -- on the remainder. If (and
only if) that ladder independently, decisively picks a replacement, this
module moves the rejected winner to `discard` and the ladder's own pick to
`select`. If the ladder does not decisively resolve the remainder (thin
gap, or its own pick is itself evidenced failed/incomplete), NOTHING is
mutated -- fail-open, original winner preserved, no partial mutation.

## Authority entry conditions (this task's own numbered list, all required)

  1. current winner exists.
  2. family has >= 2 real candidates.
  3. at least one alternative is meaning-sufficient.
  4. V2 evidence exists (a REAL `CandidateZoneUsabilityV2`, not V1_FALLBACK
     -- V1-only evidence never authorizes rejection in this task).
  5. V2 comparison is non-conflicted (winner carries no `zone_conflict` /
     V1 `conflict_flags` -- `GUARD_UNCERTAIN` fails open, exactly like
     D-163's own base guard).
  6. DELIVERY-owned performance evidence exists (winner's own delivery
     usability is UNUSABLE/IMPAIRED -- CASE_B_DELIVERY_OWNED; guaranteed
     by construction, since `zone_usability_v2_dominates`/`_performance_
     dominates` never fire otherwise).
  7. current winner is materially worse than at least one eligible
     alternative -- the MATERIALITY floor below, reusing D-167's own
     existing `SEVERITY_MATERIAL`/`SEVERITY_SEVERE` categorical vocabulary,
     never a new numeric threshold.
  8. CASE A does not own the defect (`case_classification !=
     CASE_A_BOUNDARY_ONLY` -- guaranteed by construction; `zone_usability_
     v2_dominates` refuses to fire for a CASE-A winner. Checked again here
     defensively, never re-derived).
  9. CASE C ambiguity is not present (guaranteed by construction: a
     `zone_conflict`/`conflict_flags` winner never reaches a dominance
     finding at all).
  10. D-123 does not already own an actionable disagreement (`case_b_
      conflict_present` on this family's own row, computed by `pipeline.
      py`'s existing D-123 fast-path-conflict check, reused verbatim).
  11. no semantic/meaning firewall violation (`dominant_candidate_id` is
      ALWAYS drawn from the meaning-sufficient set only -- guaranteed by
      construction in `evaluate_watch_listen_besttake_guard_v2`).

If any condition fails: `NO_ACTION` (or a more specific `BLOCKED_*` state
below) -- never a forced outcome, never an exception.

## Materiality (condition 7, this task's own explicit instruction)

No new threshold is invented. Authority is considered only when the
CURRENT WINNER's own DELIVERY zone severity (D-167's `CandidateZone
UsabilityV2.delivery.zone_severity`, already computed) is `MATERIAL` or
`SEVERE` -- never `MILD`/`NONE`/`UNKNOWN`/`MIXED`. This is the SAME
existing categorical severity table D-167 already computes (`_SEVERITY_
TABLE`); this module adds no numeric fraction cutoff, no weighted score.
A pimples-like near-equal defect (both candidates `MILD` or better) never
reaches `MATERIAL`/`SEVERE` and therefore never authorizes rejection --
this is also, incidentally, why D-173's own real `tg_8cae696f55d852a3e5`
family (winner severity `MILD`) does NOT authorize a rejection under this
module even though V1 already found `PERFORMANCE_DOMINANT_ALTERNATIVE`
there: a real ordinal dominance finding is a necessary but not sufficient
condition for THIS authority to act -- MATERIALITY is the additional bar
D-163/D-172's own diagnostic-only guards never had to clear.

## Authority states (bounded, no state zoo)

    NO_ACTION                    -- nothing to do (no winner, no V2
                                     evidence, winner already acceptable,
                                     no dominant alternative found, or
                                     materiality floor not met).
    BLOCKED_BY_MEANING_FIREWALL   -- V2 found a dominant alternative but it
                                     is meaning-insufficient (D-170/D-172's
                                     own real replay shape).
    BLOCKED_BY_CASE_OWNERSHIP     -- the defect is CASE_A_BOUNDARY_ONLY
                                     (Boundary's own territory; defensive,
                                     structurally unreachable in practice).
    BLOCKED_BY_D123               -- D-123 already owns an actionable
                                     semantic-vs-performance disagreement
                                     for this family (`case_b_conflict_
                                     present`); D-174 never duplicates or
                                     overrides that decision.
    BLOCKED_BY_CONFLICT           -- the winner carries a material
                                     modality conflict (CASE C); fails
                                     open, exactly like D-163's own base
                                     guard's `UNCERTAIN`.
    GUARD_REJECT_CURRENT_WINNER   -- Phase 1's own verdict: this family's
                                     current winner should be rejected: the
                                     Phase-2 ladder reevaluation decides
                                     whether that verdict is actually
                                     applied.
    LADDER_REEVALUATED            -- Phase 2 ran the existing deterministic
                                     ladder on the remainder and it
                                     decisively picked a replacement --
                                     the winner mutation was actually
                                     applied.

## Feature flag

`CUTSELL_WATCH_LISTEN_BESTTAKE_GUARD_AUTHORITY_ENABLED`, default OFF, a
SEPARATE flag from D-163's `CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED`
and D-172's `CUTSELL_WATCH_LISTEN_ZONE_USABILITY_V2_BESTTAKE_ENABLED` --
independently rollbackable. OFF: `evaluate_watch_listen_besttake_guard_
authority` is never called from `pipeline.py` (Phase 1 diagnostics are
absent from the row, exactly like D-163/D-172's own off-state), and
`apply_watch_listen_besttake_guard_authority` (Phase 2) returns its input
`draft` completely unchanged -- byte-identical to pre-D-174 in every
respect, including when D-163/D-172's own flags are ON.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Tuple
import os

from .watch_listen_besttake_evidence import (
    GUARD_NO_ACTION,
    GUARD_PRESERVE_STRUCTURED_WINNER,
    GUARD_UNCERTAIN,
)
from .watch_listen_besttake_v2_evidence import (
    EVIDENCE_SOURCE_V2,
    GUARD_BYPASS_POOR_USABILITY_WINNER,
    GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE,
    WatchListenBestTakeV2GuardResult,
)
from .watch_listen_zone_usability_v2 import (
    CASE_A_BOUNDARY_ONLY,
    CandidateZoneUsabilityV2,
    SEVERITY_MATERIAL,
    SEVERITY_SEVERE,
)

SCHEMA_VERSION = "cutsell.watch_listen_besttake_guard_authority.v1"

_GUARD_AUTHORITY_ENV = "CUTSELL_WATCH_LISTEN_BESTTAKE_GUARD_AUTHORITY_ENABLED"

# ---------------------------------------------------------------------------
# Authority states (bounded, module docstring).
# ---------------------------------------------------------------------------
AUTHORITY_NO_ACTION = "NO_ACTION"
AUTHORITY_BLOCKED_BY_MEANING_FIREWALL = "BLOCKED_BY_MEANING_FIREWALL"
AUTHORITY_BLOCKED_BY_CASE_OWNERSHIP = "BLOCKED_BY_CASE_OWNERSHIP"
AUTHORITY_BLOCKED_BY_D123 = "BLOCKED_BY_D123"
AUTHORITY_BLOCKED_BY_CONFLICT = "BLOCKED_BY_CONFLICT"
AUTHORITY_GUARD_REJECT_CURRENT_WINNER = "GUARD_REJECT_CURRENT_WINNER"
AUTHORITY_LADDER_REEVALUATED = "LADDER_REEVALUATED"

# Materiality floor (condition 7) -- reuses D-167's OWN existing severity
# vocabulary verbatim; no new numeric threshold.
_MATERIAL_SEVERITIES = frozenset({SEVERITY_MATERIAL, SEVERITY_SEVERE})

# Winner-source authority provenance vocabulary (module docstring's
# "Winner provenance" contract) -- distinct labels so a diagnostic reader
# can always tell WHICH authority actually moved a clip.
AUTHORITY_SOURCE_WATCH_LISTEN_GUARD_REJECTION = "WATCH_LISTEN_GUARD_REJECTION"
AUTHORITY_SOURCE_DETERMINISTIC_BESTTAKE_LADDER = "DETERMINISTIC_BESTTAKE_LADDER"


def _env_true_default_false(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def watch_listen_besttake_guard_authority_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_GUARD_AUTHORITY_ENV))


@dataclass(frozen=True)
class WatchListenBestTakeGuardAuthorityResult:
    """Phase 1's own pure verdict. No mutation capability -- see module
    docstring's two-phase design."""
    authority_state: str
    rejection_reason: str
    winner_id: str | None
    rejected_winner_id: str | None
    eligible_alternative_ids: Tuple[str, ...]
    v2_dominant_candidates: Tuple[str, ...]
    meaning_firewall_blocked: bool
    d123_blocked: bool
    case_owner: str | None


def evaluate_watch_listen_besttake_guard_authority(
    *,
    winner_id: str | None,
    meaning_sufficient_ids: Iterable[str],
    member_count: int,
    v2_result: WatchListenBestTakeV2GuardResult | None,
    v2_evidence_by_id: Mapping[str, CandidateZoneUsabilityV2 | None],
    case_b_conflict_present: bool,
) -> WatchListenBestTakeGuardAuthorityResult:
    """Phase 1 -- the bounded, pure eligibility decision. Every one of this
    task's 11 numbered entry conditions is checked (module docstring);
    failure of any one routes to `NO_ACTION` or a specific `BLOCKED_*`
    state, never an exception, never a forced outcome."""
    meaning_sufficient_ids = frozenset(meaning_sufficient_ids)

    def _none(reason: str) -> WatchListenBestTakeGuardAuthorityResult:
        return WatchListenBestTakeGuardAuthorityResult(
            AUTHORITY_NO_ACTION, reason, winner_id, None, (), (), False, False, None,
        )

    # Conditions 1/2: a current winner and a real >=2-candidate contest.
    if not winner_id or member_count < 2 or v2_result is None:
        return _none("missing_winner_or_insufficient_candidates_or_no_v2_result")

    # Condition 4: REAL V2 evidence required -- V1_FALLBACK never authorizes
    # rejection in this task (materiality below needs D-167's own severity
    # table, which only a real CandidateZoneUsabilityV2 carries).
    if v2_result.evidence_source != EVIDENCE_SOURCE_V2:
        return _none("v2_evidence_absent_v1_fallback_never_authorizes")

    # Condition 5/9: CASE C / material modality conflict fails open.
    if v2_result.guard_status == GUARD_UNCERTAIN:
        return WatchListenBestTakeGuardAuthorityResult(
            AUTHORITY_BLOCKED_BY_CONFLICT, "winner_has_material_conflict_flags",
            winner_id, None, (), (), False, False, None,
        )

    # Winner's own delivery usability is acceptable, or evidence missing --
    # nothing to reject.
    if v2_result.guard_status in (GUARD_NO_ACTION, GUARD_PRESERVE_STRUCTURED_WINNER):
        return _none("winner_usability_acceptable_or_evidence_missing")

    if v2_result.guard_status == GUARD_BYPASS_POOR_USABILITY_WINNER:
        # Condition 3/11: V2 found no meaning-sufficient dominant
        # alternative at all -- either genuinely none exists, or one was
        # found but is meaning-insufficient (D-170/D-172's own real replay
        # shape) and the Meaning Firewall already blocked it upstream in
        # `evaluate_watch_listen_besttake_guard_v2`.
        if v2_result.meaning_firewall_blocked:
            return WatchListenBestTakeGuardAuthorityResult(
                AUTHORITY_BLOCKED_BY_MEANING_FIREWALL,
                "v2_dominant_alternative_meaning_insufficient_blocked",
                winner_id, None, (), (), True, False, None,
            )
        return _none("no_dominant_meaning_sufficient_alternative")

    if v2_result.guard_status != GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE:
        # Defensive: no other guard_status value exists today.
        return _none("unrecognized_v1_v2_guard_status")

    dominant_id = v2_result.dominant_candidate_id
    if dominant_id is None:
        # Defensive: PERFORMANCE_DOMINANT_ALTERNATIVE always carries a
        # dominant_candidate_id by construction; never reached in practice.
        return _none("dominant_candidate_missing_defensive")

    winner_v2 = v2_evidence_by_id.get(winner_id)
    case_owner = winner_v2.case_classification if winner_v2 is not None else None

    # Condition 8: CASE A ownership -- structurally unreachable (a CASE-A
    # winner never produces a dominance finding at all, see module
    # docstring), checked again here defensively, never re-derived.
    if case_owner == CASE_A_BOUNDARY_ONLY:
        return WatchListenBestTakeGuardAuthorityResult(
            AUTHORITY_BLOCKED_BY_CASE_OWNERSHIP, "case_a_boundary_only_never_besttake_rejection",
            winner_id, None, (dominant_id,), (dominant_id,), v2_result.meaning_firewall_blocked, False, case_owner,
        )

    # Condition 10: D-123 already owns an actionable disagreement for this
    # family -- D-174 never duplicates or overrides that decision.
    if case_b_conflict_present:
        return WatchListenBestTakeGuardAuthorityResult(
            AUTHORITY_BLOCKED_BY_D123, "d123_actionable_disagreement_owns_this_family",
            winner_id, None, (dominant_id,), (dominant_id,), v2_result.meaning_firewall_blocked, True, case_owner,
        )

    # Condition 7: materiality floor -- reuses D-167's own existing
    # categorical severity table; no new threshold.
    winner_severity = winner_v2.delivery.zone_severity if winner_v2 is not None else None
    if winner_severity not in _MATERIAL_SEVERITIES:
        return WatchListenBestTakeGuardAuthorityResult(
            AUTHORITY_NO_ACTION, "winner_severity_below_material_floor",
            winner_id, None, (dominant_id,), (dominant_id,), v2_result.meaning_firewall_blocked, False, case_owner,
        )

    eligible_alternative_ids = tuple(sorted(meaning_sufficient_ids - {winner_id}))
    return WatchListenBestTakeGuardAuthorityResult(
        AUTHORITY_GUARD_REJECT_CURRENT_WINNER,
        "winner_delivery_materially_dominated_by_meaning_sufficient_alternative",
        winner_id, winner_id, eligible_alternative_ids, (dominant_id,),
        v2_result.meaning_firewall_blocked, False, case_owner,
    )


def watch_listen_besttake_guard_authority_row(result: WatchListenBestTakeGuardAuthorityResult) -> dict:
    """The exact compact per-family diagnostic fields this task's own
    directive names, bounded and JSON-safe -- no transcript dump. Phase-2
    outcome fields (`ladder_reevaluated`/`ladder_replacement_id`/
    `winner_after`/`authority_applied`) default to their fail-open state
    here; `apply_watch_listen_besttake_guard_authority` patches them onto
    this same row in place when it actually runs (mirroring D-123's own
    `winner_path`/`final_winner` patch-after-the-fact idiom)."""
    return {
        "guard_authority_enabled": True,
        "guard_authority_evaluated": True,
        "guard_authority_state": result.authority_state,
        "guard_authority_rejection_reason": result.rejection_reason,
        "guard_authority_winner_before": result.winner_id,
        "guard_authority_rejected_winner_id": result.rejected_winner_id,
        "guard_authority_eligible_alternative_ids": list(result.eligible_alternative_ids),
        "guard_authority_v2_dominant_candidates": list(result.v2_dominant_candidates),
        "guard_authority_meaning_firewall_blocked": result.meaning_firewall_blocked,
        "guard_authority_d123_blocked": result.d123_blocked,
        "guard_authority_case_owner": result.case_owner,
        "guard_authority_ladder_reevaluated": False,
        "guard_authority_ladder_replacement_id": None,
        "guard_authority_winner_after": result.winner_id,
        "guard_authority_applied": False,
        "guard_authority_source": (
            AUTHORITY_SOURCE_WATCH_LISTEN_GUARD_REJECTION
            if result.authority_state == AUTHORITY_GUARD_REJECT_CURRENT_WINNER
            else None
        ),
    }


def watch_listen_besttake_guard_authority_diagnostics(
    rows: Iterable[WatchListenBestTakeGuardAuthorityResult],
) -> dict:
    """Tail-safe, counts-only CI summary (module docstring's own field
    list) -- same pattern as D-163/D-172's compact summaries."""
    rows = tuple(rows)
    counts = {
        "guard_authority_evaluated_count": len(rows),
        "guard_authority_no_action_count": 0,
        "guard_authority_meaning_block_count": 0,
        "guard_authority_case_block_count": 0,
        "guard_authority_d123_block_count": 0,
        "guard_authority_conflict_block_count": 0,
        "guard_authority_rejection_count": 0,
        "guard_authority_ladder_reselection_count": 0,
        "guard_authority_winner_changed_count": 0,
    }
    key_for_state = {
        AUTHORITY_NO_ACTION: "guard_authority_no_action_count",
        AUTHORITY_BLOCKED_BY_MEANING_FIREWALL: "guard_authority_meaning_block_count",
        AUTHORITY_BLOCKED_BY_CASE_OWNERSHIP: "guard_authority_case_block_count",
        AUTHORITY_BLOCKED_BY_D123: "guard_authority_d123_block_count",
        AUTHORITY_BLOCKED_BY_CONFLICT: "guard_authority_conflict_block_count",
        AUTHORITY_GUARD_REJECT_CURRENT_WINNER: "guard_authority_rejection_count",
    }
    for row in rows:
        key = key_for_state.get(row.authority_state)
        if key:
            counts[key] += 1
    return counts


# ---------------------------------------------------------------------------
# Phase 2 -- the ONE real winner-mutation seam (module docstring). Called
# from universal_clean_cut.py, immediately after `deterministic_best_take_
# authority.apply_deterministic_best_take_authority` -- the SAME existing
# place a family's real bucket assignment is already finalized.
# ---------------------------------------------------------------------------
def _order(clip):
    return (clip.source_order, float(clip.start), float(clip.end), clip.clip_id)


def apply_watch_listen_besttake_guard_authority(draft):
    """Fail-open throughout: any family Phase 1 did not mark `GUARD_REJECT_
    CURRENT_WINNER` is untouched; a family it did mark is mutated ONLY if
    the EXISTING deterministic ladder (`deterministic_best_take_authority.
    clear_retry_family_winner`, reused verbatim, never re-derived) itself
    decisively picks a replacement from the remainder. No exception path
    here can raise past this function -- a malformed group is simply
    skipped, the same defensive posture `apply_deterministic_best_take_
    authority` itself already uses.

    Structural guarantee (this task's own explicit "no direct V2 winner
    selection" requirement): the ONLY candidate this function can ever
    write into `winner_after` is whatever `clear_retry_family_winner`
    independently returns from the rejected-winner-excluded remainder --
    this function never reads or writes `guard_authority_v2_dominant_
    candidates` when choosing a replacement."""
    if not watch_listen_besttake_guard_authority_enabled():
        return draft

    groups = list((draft.diagnostics or {}).get("take_judge_groups") or ())
    if not groups:
        return draft

    from .deterministic_best_take_authority import clear_retry_family_winner

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

    new_selected = dict(selected_by_id)
    new_alternates = dict(alternates_by_id)
    new_discarded = dict(discarded_by_id)
    moves: list[dict] = []

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

    patched_groups = []
    any_patch = False
    for row in groups:
        if not isinstance(row, dict) or row.get("guard_authority_state") != AUTHORITY_GUARD_REJECT_CURRENT_WINNER:
            patched_groups.append(row)
            continue

        rejected_id = row.get("guard_authority_rejected_winner_id")
        ranked = list(row.get("ranked") or ())
        winner_before = row.get("guard_authority_winner_before")

        if not rejected_id or rejected_id not in all_clips:
            # Fail-open: cannot locate the rejected clip in this draft's
            # clip universe (should not happen; defensive only).
            patched_groups.append(row)
            continue

        remaining_ranked = [r for r in ranked if str(r.get("clip_id") or "") != str(rejected_id)]
        if len(remaining_ranked) == 1:
            # Not a contest to resolve -- exactly one candidate remains once
            # the rejected winner is excluded, so it is trivially the
            # replacement (nothing for `clear_retry_family_winner`'s own
            # >=2-member decisive-gap contract to decide between). This is
            # NOT a second resolver: it is the same "a singleton is never a
            # contest" boundary `clear_retry_family_winner` itself already
            # documents, restated for the post-rejection remainder.
            replacement_row = remaining_ranked[0]
        else:
            replacement_row = clear_retry_family_winner(remaining_ranked)

        if replacement_row is None:
            # Fail-open (this task's own explicit requirement): the
            # remainder is not decisive, or its own top pick is itself
            # evidenced failed/incomplete -- preserve the original winner,
            # no partial mutation.
            patched_groups.append(row)
            continue

        replacement_id = str(replacement_row.get("clip_id") or "")
        if not replacement_id or replacement_id not in all_clips or replacement_id == rejected_id:
            patched_groups.append(row)
            continue

        group_id = row.get("group_id")
        move(
            str(rejected_id), "discard", "watch_listen_guard_rejected_current_winner",
            {"group_id": group_id, "replacement_clip_id": replacement_id},
        )
        move(
            replacement_id, "select", "watch_listen_guard_ladder_replacement",
            {"group_id": group_id, "rejected_winner_id": str(rejected_id)},
        )
        any_patch = True
        patched_groups.append({
            **row,
            "guard_authority_state": AUTHORITY_LADDER_REEVALUATED,
            "guard_authority_ladder_reevaluated": True,
            "guard_authority_ladder_replacement_id": replacement_id,
            "guard_authority_winner_after": replacement_id,
            "guard_authority_applied": True,
            "guard_authority_replacement_source": AUTHORITY_SOURCE_DETERMINISTIC_BESTTAKE_LADDER,
            # D-123 (docs/CUTSELL_DECISIONS.md D-123)'s own field names,
            # kept consistent so any existing consumer of `winner_path_
            # after`/`final_winner` sees the true final answer -- this
            # authority is a LATER stage than both `_semantic_best_take`
            # and `deterministic_best_take_authority`.
            "winner_path_after": "WATCH_LISTEN_GUARD_LADDER_REPLACEMENT",
            "final_winner": replacement_id,
        })

    if not any_patch:
        return draft

    def _clip_order(clip):
        return _order(clip)

    selected = tuple(sorted(new_selected.values(), key=_clip_order))
    alternates = tuple(sorted(new_alternates.values(), key=_clip_order))
    discarded = tuple(sorted(new_discarded.values(), key=_clip_order))

    diagnostics = dict(draft.diagnostics or {})
    diagnostics["take_judge_groups"] = patched_groups
    diagnostics["watch_listen_besttake_guard_authority_applied"] = {
        "status": "applied",
        "moves": moves,
    }
    return replace(draft, selected=selected, alternates=alternates, discarded=discarded, diagnostics=diagnostics)
