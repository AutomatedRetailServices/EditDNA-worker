"""D-163 Phase D -- Watch+Listen PERFORMANCE / USABILITY evidence for BestTake.

Per docs/CUTSELL_DECISIONS.md D-148 through D-162. D-162's own real-media
qualification proved the general shape this module addresses:

    CORRECT COMPETITOR FAMILY
    + MEANING-SUFFICIENT FINALISTS
    + SEMANTIC/STRUCTURED SYSTEM DOES NOT RESOLVE QUALITY WELL
    -> DeliveryScorer may choose a visibly/perceptually inferior take.

## Core principle

    WATCH+LISTEN PERFORMANCE EVIDENCE CAN DISQUALIFY OR DEMOTE A POORLY-
    USABLE REALIZATION. IT DOES NOT AUTOMATICALLY SELECT THE MOST ACTIVE
    OR MOST EXPRESSIVE TAKE. CLARITY AND USABILITY BEFORE ENERGY.

This module is DIAGNOSTIC-ONLY in this task (D-163's own "offline
implementation first" scope): `evaluate_watch_listen_besttake_guard`
computes a bounded, structured judgment about whether the existing
BestTake winner for a family has materially worse DELIVERY usability
than a meaning-sufficient alternative, and (when its own downstream
deterministic ladder -- `deterministic_best_take_authority.py`'s
existing, unchanged `clear_retry_family_winner`, D-123's own
`CLEAR_WINNER_MINIMUM_GAP` -- ALREADY independently supports that same
alternative) reports that the two authorities agree. It never mutates
`selected_clip_id`, `ranked`, membership, grouping, or Boundary itself --
see the module's own "AUTHORITY ACTION" contract below for why this is
the deliberately conservative choice for this task, not a limitation
overlooked.

## No new opaque score

Per this task's own explicit instruction, this module invents no
`watch_listen_score = 0.37 * visual + ...` composite. Every comparison is
a partial-order dominance check over the SAME bounded categorical
usability vocabulary `watch_listen_understanding.py` already computes
(`USABLE`/`QUESTIONABLE`/`UNUSABLE`/`UNKNOWN`), reusing the EXISTING
`RankedTake` scores (`take_judge.py`, unchanged) only as a tie-break among
candidates this module's own dominance check already narrowed to.

## Evidence contract: `WatchListenBestTakeEvidence`

One normalized, per-candidate structure built PURELY from an already-
computed `watch_listen_understanding.UnderstandingSpan` (D-157, unchanged)
plus one boolean this module's caller already knows (`meaning_sufficient`,
read from the EXISTING `meaning_sufficient_candidates` field `pipeline.py`
already computes -- D-089, unchanged). No perception is recomputed; no new
detector, threshold, or provider call is introduced.

## Double-counting audit (this task's own explicit requirement)

The four real Track C event kinds (`camera_disengagement_candidate`,
`facial_expression_shift_candidate`, `body_reset_candidate`, `hand_
motion_reset_candidate`) are the SAME underlying signal that already
independently feeds THREE existing consumers before this module ever
runs:

    local_performance.apply_local_performance_to_takes
        -> MediaSignals (visual_fumble, expression_naturalness,
           gesture_naturalness, distraction_risk) -> take_judge.score_take
           -> the DeliveryScorer composite blended into `ranked`.
    take_judge.delivery_cleanliness_evidence (D-097)
        -> a confidence-floored (>=0.88 reset AND >=0.76 break, BOTH
           required together), interior-window-gated penalty already
           subtracted from `ranked`'s own score.
    case_b_performance_evidence.py (D-122/D-123)
        -> a DELIVERY-zone-filtered factual re-projection, advisory only,
           never itself decisive.

`watch_listen_understanding.py`'s own `entry_usability`/`delivery_
usability`/`exit_usability` (D-157, unchanged) are ALSO derived from
these same four kinds (plus D-100's `wrong_take`/`retry_setup`
corroboration and the recording-process/false-start/breaking-character
families) via `_usability_for_zone` -- but that function applies NO
confidence floor and NO "two independent signals required" rule; it
fires categorically (UNUSABLE) on a single defect-kind event of any
confidence inside the DELIVERY zone. This is classified below as
`PARTIALLY_CORRELATED` with the DeliveryScorer/D-097 signal (same root
events, different aggregation shape), never `INDEPENDENT` -- see
`DOUBLE_COUNTING_AUDIT` below, consulted directly by this module's own
guard to avoid treating agreement between these two views as two
independent confirmations.

Genuinely `INDEPENDENT` evidence this module adds: `conflict_flags`
(`MEANING_COMPLETE_VS_ABANDONED_ATTEMPT_EVIDENCE`, `EXIT_RESET_VS_
MEANING_COMPLETE`) cross-check meaning completion against behavior
evidence -- a comparison NONE of DeliveryScorer/D-097/D-122 perform
today -- and the explicit ENTRY-vs-DELIVERY-vs-EXIT zone split itself,
which the DeliveryScorer's own blended score never exposes separately
(D-107/D-115's CASE A/B/C doctrine has no equivalent in `score_take`'s
single composite number).

## Meaning firewall

`build_watch_listen_besttake_evidence` takes `meaning_sufficient` as a
caller-supplied fact (from the EXISTING `meaning_sufficient_candidates`
field) and the guard NEVER treats a meaning-insufficient candidate as an
eligible alternative or as a valid winner to demote -- see `_eligible_
alternatives`. No W+L performance signal here can override meaning/
safety; this module has no code path capable of marking a meaning-
insufficient candidate the guard's `dominant_candidate_id`.

## CASE A / B / C ownership (D-107/D-115, reused, never re-derived)

    CASE_B_DELIVERY_OWNED  -- defect overlaps DELIVERY -> legitimate
                              BestTake performance evidence (this module's
                              own territory).
    CASE_A_BOUNDARY_ONLY   -- defect is ENTRY/EXIT only -> Boundary's
                              concern; NEVER sufficient alone for this
                              module to demote a candidate.
    CASE_C_AMBIGUOUS       -- delivery usability UNKNOWN or conflicting ->
                              preserve/uncertain, never an aggressive
                              penalty.

## Performance dominance contract

Candidate B may performance-dominate A only if ALL hold:
  1. B is no worse than A on every trusted usability dimension (entry,
     delivery, exit -- ordinal USABLE > QUESTIONABLE > UNUSABLE, UNKNOWN
     treated as neutral, never worse, never better);
  2. B is strictly better than A on the DELIVERY dimension specifically
     (never on entry/exit alone -- see CASE A ownership above);
  3. A's own inferiority is not explained ONLY by ENTRY/EXIT-removable
     debris (`case_for(A) != CASE_A_BOUNDARY_ONLY`);
  4. neither A nor B carries a `conflict_flags` entry (a material
     modality disagreement invalidates the comparison outright).
No arbitrary numeric threshold is introduced; the ordinal comparison
above is the entire rule.

## Authority action (deliberately conservative for this offline task)

Per this task's own "Preferred safe behavior" and "No new terminal
absolute authority unless code architecture proves necessary": when the
guard finds a performance-dominant alternative, it does NOT itself
reassign `selected_clip_id`. It instead checks whether `deterministic_
best_take_authority.clear_retry_family_winner` (D-123's own existing,
unchanged, already-CLOSED deterministic ladder) ALREADY independently
picks that SAME candidate as its own decisive winner. When it does, the
guard reports `PERFORMANCE_DOMINANT_ALTERNATIVE` with `existing_ladder_
agrees=True` -- a confirmation, not a new action (that ladder's own pick
already stood on its own before this module ever ran). When the existing
ladder does NOT independently support it (D-162's own real-media shape:
DeliveryScorer's blended score already agreed with the poorly-usable
take), the guard reports `PERFORMANCE_DOMINANT_ALTERNATIVE` with
`existing_ladder_agrees=False` -- visible, structured QA evidence for the
Product Owner to weigh a future, explicitly-authorized action gate on,
never a silent auto-correction in THIS task.

## Feature flag

`CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED`, default OFF, separate
from D-158's `CUTSELL_WATCH_LISTEN_FAMILY_EVIDENCE_ENABLED` and D-161's
`CUTSELL_WATCH_LISTEN_RELATION_DISCOVERY_ENABLED` -- three different
authorities (merge-veto, relation-discovery, BestTake-evidence). OFF:
`pipeline.py` never calls into this module; behavior (including
`selected_clip_id`, `ranked`, membership, Boundary, Pacing, render) is
byte-identical to pre-D-163. ON: this module's own diagnostics populate
alongside the existing `take_judge_groups` row; `selected_clip_id`
remains untouched either way in this task.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple
import os

from .raw_understanding_map import (
    BEHAVIOR_ABANDONED_ATTEMPT,
    BEHAVIOR_BREAKING_CHARACTER,
    BEHAVIOR_FALSE_START,
    BEHAVIOR_POST_TAKE_RESET,
    BEHAVIOR_RECORDING_PROCESS,
)
from .watch_listen_understanding import (
    USABILITY_QUESTIONABLE,
    USABILITY_UNKNOWN,
    USABILITY_UNUSABLE,
    USABILITY_USABLE,
    UnderstandingSpan,
)

SCHEMA_VERSION = "cutsell.watch_listen_besttake_evidence.v1"

_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENV = "CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED"

# ---------------------------------------------------------------------------
# CASE A/B/C ownership (D-107/D-115, reused vocabulary).
# ---------------------------------------------------------------------------
CASE_A_BOUNDARY_ONLY = "CASE_A_BOUNDARY_ONLY"
CASE_B_DELIVERY_OWNED = "CASE_B_DELIVERY_OWNED"
CASE_C_AMBIGUOUS = "CASE_C_AMBIGUOUS"
CASE_CLEAN = "CASE_CLEAN"

# ---------------------------------------------------------------------------
# Editability / continuity vocabulary (repo-conventional shape: bounded
# categorical states, never a score).
# ---------------------------------------------------------------------------
EDITABILITY_CLEAN = "CLEAN"
EDITABILITY_BOUNDARY_ONLY = "BOUNDARY_ONLY"
EDITABILITY_DELIVERY_OWNED = "DELIVERY_OWNED"
EDITABILITY_AMBIGUOUS = "AMBIGUOUS"

CONTINUITY_CONTINUOUS = "CONTINUOUS"
CONTINUITY_INTERRUPTED_AT_EDGE = "INTERRUPTED_AT_EDGE"
CONTINUITY_INTERRUPTED_DURING_DELIVERY = "INTERRUPTED_DURING_DELIVERY"
CONTINUITY_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Double-counting audit (this task's own explicit requirement) -- a static,
# code-derived classification of every dimension this module reads against
# the existing DeliveryScorer/D-097/D-122 evidence chain. Never re-derived
# per call; read directly by tests as the audit's own source of truth.
# ---------------------------------------------------------------------------
DOUBLE_COUNTING_AUDIT: Mapping[str, str] = {
    # entry/delivery/exit usability share their root events (the 4 real
    # Track C kinds) with MediaSignals/D-097/D-122, but apply a DIFFERENT,
    # un-floored, categorical aggregation -- correlated, never identical.
    "entry_usability": "PARTIALLY_CORRELATED",
    "delivery_usability": "PARTIALLY_CORRELATED",
    "exit_usability": "PARTIALLY_CORRELATED",
    "overall_performance_usability": "PARTIALLY_CORRELATED",
    "reset_or_fumble_during_delivery": "PARTIALLY_CORRELATED",
    # BEHAVIOR_BREAKING_CHARACTER/BEHAVIOR_ABANDONED_ATTEMPT (D-100's own
    # wrong_take/retry_setup corroboration) are NOT consumed by case_b_
    # performance_evidence.py at all (it filters strictly to the 4
    # LOCAL_PERFORMANCE_EVENT_KINDS) and are not folded into MediaSignals
    # either -- genuinely new evidence for BestTake at this layer.
    "breaking_character_during_delivery": "INDEPENDENT",
    # conflict_flags compare meaning completion against behavior evidence
    # -- no existing BestTake-adjacent consumer performs this cross-check.
    "conflict_flags": "INDEPENDENT",
    # No live producer feeds pure audio-signal behavior hypotheses into
    # WatchListenUnderstanding today (see module docstring's Audio Honesty
    # section) -- there is nothing here to double-count.
    "audio_signal_usability": "INDEPENDENT",
    # visual_signal_usability is a direct relabeling of the SAME zone
    # usability fields above.
    "visual_signal_usability": "SAME_SOURCE_DUPLICATE",
}


def _env_true_default_false(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def watch_listen_besttake_evidence_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENV))


@dataclass(frozen=True)
class WatchListenBestTakeEvidence:
    """One normalized per-candidate BestTake-evidence record. No final-
    winner field -- this is evidence, never a decision (see module
    docstring)."""
    candidate_id: str
    meaning_sufficient: bool
    overall_performance_usability: str
    entry_usability: str
    delivery_usability: str
    exit_usability: str
    delivery_defect_present: bool
    entry_only_defect: bool
    exit_only_defect: bool
    breaking_character_during_delivery: bool
    reset_or_fumble_during_delivery: bool
    performance_continuity_status: str
    audio_signal_usability: str
    visual_signal_usability: str
    editability_status: str
    conflict_flags: Tuple[str, ...]
    evidence_provenance: Mapping[str, str]
    case_classification: str


def _has_label(span: UnderstandingSpan, label: str) -> bool:
    return any(h.label == label for h in span.behavior_state_hypotheses)


def _case_for(entry: str, delivery: str, exit_: str) -> str:
    if delivery == USABILITY_UNUSABLE:
        return CASE_B_DELIVERY_OWNED
    if delivery == USABILITY_UNKNOWN:
        return CASE_C_AMBIGUOUS
    if entry == USABILITY_QUESTIONABLE or exit_ == USABILITY_QUESTIONABLE:
        return CASE_A_BOUNDARY_ONLY
    return CASE_CLEAN


_EDITABILITY_FOR_CASE = {
    CASE_CLEAN: EDITABILITY_CLEAN,
    CASE_A_BOUNDARY_ONLY: EDITABILITY_BOUNDARY_ONLY,
    CASE_B_DELIVERY_OWNED: EDITABILITY_DELIVERY_OWNED,
    CASE_C_AMBIGUOUS: EDITABILITY_AMBIGUOUS,
}
_CONTINUITY_FOR_CASE = {
    CASE_CLEAN: CONTINUITY_CONTINUOUS,
    CASE_A_BOUNDARY_ONLY: CONTINUITY_INTERRUPTED_AT_EDGE,
    CASE_B_DELIVERY_OWNED: CONTINUITY_INTERRUPTED_DURING_DELIVERY,
    CASE_C_AMBIGUOUS: CONTINUITY_UNKNOWN,
}


def build_watch_listen_besttake_evidence(
    candidate_id: str, span: UnderstandingSpan | None, *, meaning_sufficient: bool,
) -> WatchListenBestTakeEvidence | None:
    """Pure projection from an already-computed `UnderstandingSpan` (D-157,
    unchanged). Returns None when no span exists for this candidate --
    fail-open, never a fabricated default (per D-163's own fail-open
    requirement; the caller treats None exactly like missing evidence)."""
    if span is None:
        return None

    case = _case_for(span.entry_usability, span.delivery_usability, span.exit_usability)
    delivery_unusable = span.delivery_usability == USABILITY_UNUSABLE
    breaking_character = _has_label(span, BEHAVIOR_BREAKING_CHARACTER) and delivery_unusable
    reset_or_fumble = (
        _has_label(span, BEHAVIOR_POST_TAKE_RESET)
        or _has_label(span, BEHAVIOR_RECORDING_PROCESS)
        or _has_label(span, BEHAVIOR_FALSE_START)
        or _has_label(span, BEHAVIOR_ABANDONED_ATTEMPT)
    ) and delivery_unusable

    return WatchListenBestTakeEvidence(
        candidate_id=candidate_id,
        meaning_sufficient=meaning_sufficient,
        overall_performance_usability=span.performance_usability_hypothesis,
        entry_usability=span.entry_usability,
        delivery_usability=span.delivery_usability,
        exit_usability=span.exit_usability,
        delivery_defect_present=delivery_unusable,
        entry_only_defect=bool(case == CASE_A_BOUNDARY_ONLY and span.entry_usability == USABILITY_QUESTIONABLE),
        exit_only_defect=bool(case == CASE_A_BOUNDARY_ONLY and span.exit_usability == USABILITY_QUESTIONABLE),
        breaking_character_during_delivery=breaking_character,
        reset_or_fumble_during_delivery=reset_or_fumble,
        performance_continuity_status=_CONTINUITY_FOR_CASE[case],
        # Audio honesty (this task's own explicit requirement): no live
        # producer feeds pure audio-signal behavior hypotheses into
        # WatchListenUnderstanding today -- reporting anything else here
        # would be an overclaim (tone/prosody/emotion-from-voice), which
        # this task explicitly forbids.
        audio_signal_usability=USABILITY_UNKNOWN,
        visual_signal_usability=span.performance_usability_hypothesis,
        editability_status=_EDITABILITY_FOR_CASE[case],
        conflict_flags=span.conflict_flags,
        evidence_provenance=span.evidence_provenance,
        case_classification=case,
    )


# ---------------------------------------------------------------------------
# Guard outcomes (bounded, never a numeric score).
# ---------------------------------------------------------------------------
GUARD_NO_ACTION = "NO_ACTION"
GUARD_PRESERVE_STRUCTURED_WINNER = "PRESERVE_STRUCTURED_WINNER"
GUARD_BYPASS_POOR_USABILITY_WINNER = "BYPASS_POOR_USABILITY_WINNER"
GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE = "PERFORMANCE_DOMINANT_ALTERNATIVE"
GUARD_UNCERTAIN = "UNCERTAIN"

_USABILITY_RANK = {USABILITY_UNUSABLE: 0, USABILITY_UNKNOWN: 1, USABILITY_QUESTIONABLE: 1, USABILITY_USABLE: 2}


@dataclass(frozen=True)
class WatchListenBestTakeGuardResult:
    guard_status: str
    guard_reason: str
    winner_id: str | None
    dominant_candidate_id: str | None
    existing_ladder_agrees: bool | None
    candidate_usability_summary: Mapping[str, str]


def _no_worse(a: str, b: str) -> bool:
    """True when `b`'s rank is >= `a`'s rank on the ordinal usability
    scale (UNKNOWN/QUESTIONABLE treated as neutral -- never worse, never
    strictly better than each other)."""
    return _USABILITY_RANK.get(b, 1) >= _USABILITY_RANK.get(a, 1)


def _strictly_better_delivery(a: WatchListenBestTakeEvidence, b: WatchListenBestTakeEvidence) -> bool:
    return _USABILITY_RANK.get(b.delivery_usability, 1) > _USABILITY_RANK.get(a.delivery_usability, 1)


def _performance_dominates(
    alternative: WatchListenBestTakeEvidence, winner: WatchListenBestTakeEvidence,
) -> bool:
    """The Performance Dominance contract (module docstring). Pure; no
    numeric threshold."""
    if alternative.conflict_flags or winner.conflict_flags:
        return False
    if winner.case_classification == CASE_A_BOUNDARY_ONLY:
        # Winner's own inferiority (if any) is entry/exit-removable debris
        # only -- Boundary's territory, never sufficient for this module.
        return False
    if not (
        _no_worse(winner.entry_usability, alternative.entry_usability)
        and _no_worse(winner.delivery_usability, alternative.delivery_usability)
        and _no_worse(winner.exit_usability, alternative.exit_usability)
    ):
        return False
    return _strictly_better_delivery(winner, alternative)


def _existing_ladder_pick(ranked: Iterable[Mapping[str, object]]) -> str | None:
    """Reuses `deterministic_best_take_authority.clear_retry_family_
    winner` verbatim -- the SAME existing, unchanged, already-CLOSED
    deterministic ladder D-123 already runs -- never a re-derived copy of
    its own >=0.30 gap rule."""
    from .deterministic_best_take_authority import clear_retry_family_winner
    winner_row = clear_retry_family_winner(list(ranked))
    if winner_row is None:
        return None
    return str(winner_row.get("clip_id") or "") or None


def evaluate_watch_listen_besttake_guard(
    *,
    winner_id: str | None,
    meaning_sufficient_ids: Iterable[str],
    evidence_by_id: Mapping[str, WatchListenBestTakeEvidence | None],
    ranked: Iterable[Mapping[str, object]] = (),
) -> WatchListenBestTakeGuardResult:
    """The bounded guard. Never mutates a selection -- see module
    docstring's "Authority action" section for why. Fail-open throughout:
    missing evidence, an unsupported case, or a modality conflict always
    routes to NO_ACTION/UNCERTAIN, never a forced outcome."""
    meaning_sufficient_ids = frozenset(meaning_sufficient_ids)
    usability_summary = {
        cid: (ev.delivery_usability if ev is not None else "MISSING")
        for cid, ev in evidence_by_id.items()
    }

    if not winner_id or winner_id not in evidence_by_id:
        return WatchListenBestTakeGuardResult(
            GUARD_NO_ACTION, "missing_watch_listen_evidence_for_winner",
            winner_id, None, None, usability_summary,
        )
    winner_evidence = evidence_by_id[winner_id]
    if winner_evidence is None:
        return WatchListenBestTakeGuardResult(
            GUARD_NO_ACTION, "missing_watch_listen_evidence_for_winner",
            winner_id, None, None, usability_summary,
        )
    if winner_id not in meaning_sufficient_ids:
        # Meaning firewall: this module never evaluates a meaning-
        # insufficient winner (upstream authorities own that decision).
        return WatchListenBestTakeGuardResult(
            GUARD_NO_ACTION, "winner_not_meaning_sufficient_upstream_owned",
            winner_id, None, None, usability_summary,
        )
    if winner_evidence.conflict_flags:
        return WatchListenBestTakeGuardResult(
            GUARD_UNCERTAIN, "winner_has_material_conflict_flags",
            winner_id, None, None, usability_summary,
        )
    if winner_evidence.delivery_usability != USABILITY_UNUSABLE:
        return WatchListenBestTakeGuardResult(
            GUARD_PRESERVE_STRUCTURED_WINNER, "winner_delivery_usability_acceptable",
            winner_id, None, None, usability_summary,
        )
    if winner_evidence.case_classification == CASE_A_BOUNDARY_ONLY:
        # Unreachable given delivery_usability == UNUSABLE (CASE A requires
        # delivery != UNUSABLE by construction) -- defensive, never raises.
        return WatchListenBestTakeGuardResult(
            GUARD_PRESERVE_STRUCTURED_WINNER, "boundary_only_defect_never_demotes",
            winner_id, None, None, usability_summary,
        )

    dominant_candidates = [
        cid for cid, ev in evidence_by_id.items()
        if cid != winner_id and ev is not None and cid in meaning_sufficient_ids
        and _performance_dominates(ev, winner_evidence)
    ]
    if not dominant_candidates:
        return WatchListenBestTakeGuardResult(
            GUARD_BYPASS_POOR_USABILITY_WINNER,
            "winner_delivery_unusable_no_dominant_meaning_sufficient_alternative",
            winner_id, None, None, usability_summary,
        )

    # Deterministic tie-break among dominant candidates: reuse the EXISTING
    # `ranked` score (never a new number), then clip_id for a total order.
    ranked_score_by_id = {str(row.get("clip_id") or ""): float(row.get("score") or 0.0) for row in ranked}
    dominant_candidates.sort(key=lambda cid: (-ranked_score_by_id.get(cid, 0.0), cid))
    dominant_id = dominant_candidates[0]

    existing_pick = _existing_ladder_pick(ranked)
    return WatchListenBestTakeGuardResult(
        GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE,
        "winner_delivery_unusable_dominant_alternative_found",
        winner_id, dominant_id, existing_pick == dominant_id, usability_summary,
    )


def watch_listen_besttake_diagnostics(rows: Iterable[WatchListenBestTakeGuardResult]) -> dict:
    """Tail-safe, counts-only CI summary (same pattern as D-158/D-161's
    own compact summaries) -- never dumps a transcript or per-candidate
    basis string beyond the bounded usability-state map already on each
    result."""
    rows = tuple(rows)
    counts = {
        "watch_listen_besttake_evaluated_count": len(rows),
        "watch_listen_besttake_no_action_count": 0,
        "watch_listen_besttake_preserved_count": 0,
        "watch_listen_besttake_bypass_count": 0,
        "watch_listen_besttake_dominance_count": 0,
        "watch_listen_besttake_uncertain_count": 0,
    }
    key_for_status = {
        GUARD_NO_ACTION: "watch_listen_besttake_no_action_count",
        GUARD_PRESERVE_STRUCTURED_WINNER: "watch_listen_besttake_preserved_count",
        GUARD_BYPASS_POOR_USABILITY_WINNER: "watch_listen_besttake_bypass_count",
        GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE: "watch_listen_besttake_dominance_count",
        GUARD_UNCERTAIN: "watch_listen_besttake_uncertain_count",
    }
    for row in rows:
        key = key_for_status.get(row.guard_status)
        if key:
            counts[key] += 1
    return counts


def watch_listen_besttake_group_row(result: WatchListenBestTakeGuardResult, winner_before: str | None) -> dict:
    """The exact compact family-level diagnostic fields this task's own
    directive names, bounded and JSON-safe -- no transcript dump."""
    return {
        "watch_listen_besttake_evaluated": True,
        "watch_listen_besttake_candidate_count": len(result.candidate_usability_summary),
        "watch_listen_besttake_current_winner": result.winner_id,
        "watch_listen_besttake_guard_status": result.guard_status,
        "watch_listen_besttake_guard_reason": result.guard_reason,
        "watch_listen_besttake_dominant_candidate": result.dominant_candidate_id,
        "watch_listen_besttake_winner_before": winner_before,
        "watch_listen_besttake_winner_after": winner_before,  # D-163: never mutated in this task
        "watch_listen_besttake_action_applied": False,
        "watch_listen_besttake_existing_ladder_agrees": result.existing_ladder_agrees,
        "candidate_usability_summary": dict(result.candidate_usability_summary),
    }
