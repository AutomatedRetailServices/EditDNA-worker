"""D-188: Prosodic Audio V2 -> Bounded Finalist Arbiter -- PHASE B,
DIAGNOSTIC FUSION ONLY. Offline. No winner authority. No RAW. No
provider.

This module answers ONE bounded question for a small (2-3) finalist set
that D-187's `ProsodicDeliveryEvidence` has already been computed for:
"does the OBSERVABLE VOCAL DELIVERY safely distinguish these finalists,
in a direction any editor would agree is safe, or not?" It NEVER decides
who wins -- `compare_prosodic_finalists` returns a diagnostic comparison
object with no winner/action field, exactly like D-184's own
`BoundedFinalistArbiterResult` (which separately consumes this module's
output as ONE additional evidence dimension -- see
`bounded_finalist_arbiter.py`'s new `prosodic_comparison` field).

## Core principle (this task's own, binding)

PROSODY MAY BREAK A TRUE LOCAL BESTTAKE TIE. PROSODY MUST NOT CREATE A
PREFERENCE FROM DESCRIPTIVE DIFFERENCES THAT HAVE NO SAFE EDITORIAL
DIRECTION.

## Two evidence categories (binding, structurally enforced)

**A. DIRECTIONALLY SAFE delivery-quality evidence** -- MAY support a
bounded preference: vocal continuity (`CONTINUOUS` > `MILDLY_
INTERRUPTED` > `FRAGMENTED`), acoustic hesitation (`NOT_OBSERVED` >
`PRESENT`), vocal restart/interruption (`NOT_OBSERVED` > `SUPPORTED`).
These three categorical D-187 states are the ONLY inputs to the
dominance test below -- no numeric threshold is invented; each is
already a categorical judgment D-187 itself produced.

**B. DESCRIPTIVE / context-dependent evidence** -- speech rate, energy
mean/variation, emphasis dynamics, pitch. These NEVER independently
create a preference (no `HIGH_ENERGY > LOW_ENERGY`, no `FAST > SLOW`,
no `MORE_PITCH_VARIATION > LESS` rule exists anywhere in this module --
tested by source-scan). They are recorded on `ProsodicFinalistComparison`
as `descriptive_*_relation` fields purely for future context-aware
reasoning (D-098 15.4's Editorial Moment/Sequence Understanding, not
this task) -- never consulted by `comparison_state`/`preferred_
candidate_id`.

## Partial-order dominance rule (this task's own, binding)

Candidate B prosodically dominates A only if (1) at least one safe
dimension materially favors B, AND (2) no safe dimension materially
favors A, AND (3) both A and B have REAL, `EVALUATED` acoustic evidence
(never a transcript-only guess). (Meaning parity -- "candidates must be
meaning-sufficient" -- is D-184's own P0 gate, checked BEFORE this
module is ever consulted; this module is purely about delivery.) If
safe dimensions disagree across a pair (one favors A, another favors
B) that pair is CONFLICTED, never resolved by picking one. NO
`ProsodyScore` -- no numeric weight is ever summed across dimensions;
this is a structural partial order over categorical states only.

## Double-counting audit (this task's own explicit requirement)

`vocal_continuity_state` and `pause_structure_state` are the SAME
value in D-187 Phase A (`prosodic_audio_v2.analyze_prosodic_delivery`
sets both from one `_continuity_state(...)` computation) -- so
`continuity_comparison` and `pause_structure_comparison` on
`ProsodicFinalistComparison` are DERIVED FROM ONE UNDERLYING SIGNAL and
are counted as ONE vote (not two) in the dominance test below (see
`DOUBLE_COUNTING_AUDIT`). Separately, Prosodic pause evidence itself is
already D-187's own reuse of Audio V1's dead-air evidence (never an
independent recomputation) -- this module adds no new silence
detection of any kind.

## Firewalls (restates D-187's, binding here too)

No psychological/demographic/identity inference. No master prosody
score. No winner authority -- `ProsodicFinalistComparison` has no
`selected_clip_id`/`winner`/`action` field, and this module imports
nothing from `pipeline.py`/`canonical_edit_plan.py`/any render module.
No provider/network call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Iterable, Mapping, Sequence, Tuple
import os

from .prosodic_audio_v2 import (
    CONTINUITY_CONTINUOUS,
    CONTINUITY_FRAGMENTED,
    CONTINUITY_MILDLY_INTERRUPTED,
    EMPHASIS_PRESENT,
    HESITATION_NOT_OBSERVED,
    HESITATION_PRESENT,
    PITCH_NOT_IMPLEMENTED,
    ProsodicDeliveryEvidence,
    RESTART_NOT_OBSERVED,
    RESTART_SUPPORTED,
    STATUS_EVALUATED,
    UNKNOWN,
)

SCHEMA_VERSION = "cutsell.prosodic_finalist_comparison.v1"

_PROSODIC_FINALIST_ARBITER_DIAGNOSTICS_ENV = "CUTSELL_PROSODIC_FINALIST_ARBITER_DIAGNOSTICS_ENABLED"
_MIN_FINALISTS = 2
_MAX_FINALISTS = 3

# ---------------------------------------------------------------------------
# Comparison states (compact, no state zoo -- mirrors D-184's own vocabulary).
# ---------------------------------------------------------------------------
COMPARISON_DOMINANT = "DOMINANT"
COMPARISON_NEAR_EQUAL = "NEAR_EQUAL"
COMPARISON_CONFLICTED = "CONFLICTED"
COMPARISON_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
COMPARISON_NOT_EVALUABLE = "NOT_EVALUABLE"

_RELATION_NEAR_EQUAL = "NEAR_EQUAL"
_RELATION_CONFLICTED = "CONFLICTED"
_RELATION_UNKNOWN = "UNKNOWN"

# Directionally-safe categorical ranks (higher = more favorable delivery
# quality). Only these three source fields ever participate in the
# dominance test -- everything else on ProsodicDeliveryEvidence is
# descriptive-only and structurally excluded below.
_CONTINUITY_RANK: Mapping[str, int] = {
    CONTINUITY_CONTINUOUS: 2, CONTINUITY_MILDLY_INTERRUPTED: 1, CONTINUITY_FRAGMENTED: 0,
}
_HESITATION_RANK: Mapping[str, int] = {HESITATION_NOT_OBSERVED: 1, HESITATION_PRESENT: 0}
_RESTART_RANK: Mapping[str, int] = {RESTART_NOT_OBSERVED: 1, RESTART_SUPPORTED: 0}

# This task's own explicit double-counting audit requirement: a static,
# code-derived record of how the diagnostic fields on
# ProsodicFinalistComparison relate to each other and to D-187/Audio V1.
DOUBLE_COUNTING_AUDIT: Mapping[str, str] = {
    "continuity_comparison": "DECISION_SOURCE (one of three independent safe votes)",
    "pause_structure_comparison": "SAME_UNDERLYING_SIGNAL_AS_continuity_comparison_NEVER_AN_INDEPENDENT_VOTE",
    "hesitation_comparison": "DECISION_SOURCE (one of three independent safe votes)",
    "restart_comparison": "DECISION_SOURCE (one of three independent safe votes)",
    "descriptive_rate_relation": "OBSERVATIONAL_ONLY_NEVER_A_VOTE",
    "descriptive_energy_relation": "OBSERVATIONAL_ONLY_NEVER_A_VOTE",
    "descriptive_emphasis_relation": "OBSERVATIONAL_ONLY_NEVER_A_VOTE",
    "pitch_status": "ALWAYS_NOT_IMPLEMENTED_THIS_PHASE_NEVER_A_VOTE",
    "audio_v1_pause_evidence": "REUSED_BY_D187_NEVER_RECOMPUTED_HERE_NEVER_AN_INDEPENDENT_SOURCE",
}


def _env_true_default_false(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def prosodic_finalist_arbiter_diagnostics_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Reserved for a future LIVE pipeline-wiring task (not D-188 itself --
    this task adds no call site in `pipeline.py`). Default OFF, mirroring
    `bounded_finalist_arbiter_enabled`'s own convention. Not referenced by
    any production code path in this task."""
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_PROSODIC_FINALIST_ARBITER_DIAGNOSTICS_ENV))


@dataclass(frozen=True)
class ProsodicFinalistComparison:
    """One family's finalist-set Prosodic delivery comparison. NO winner/
    action field -- this is consumed by D-184 as ONE additional diagnostic
    evidence dimension, never a decision by itself."""
    candidate_ids: Tuple[str, ...]
    comparison_state: str  # COMPARISON_*
    preferred_candidate_id: str | None  # set ONLY when comparison_state == DOMINANT

    continuity_comparison: str  # candidate_id, NEAR_EQUAL, CONFLICTED, or UNKNOWN
    hesitation_comparison: str
    restart_comparison: str
    pause_structure_comparison: str  # ALWAYS == continuity_comparison (see DOUBLE_COUNTING_AUDIT)

    descriptive_rate_relation: str  # candidate_id, NEAR_EQUAL, or UNKNOWN -- NEVER a vote
    descriptive_energy_relation: str
    descriptive_emphasis_relation: str
    pitch_status: str  # always PITCH_NOT_IMPLEMENTED this phase

    directional_evidence_present: bool  # True iff >=1 safe dimension differentiated any pair
    conflict_present: bool  # True iff comparison_state == CONFLICTED
    missing_evidence: Tuple[str, ...]

    evidence_sources: Tuple[str, ...]
    provenance: str


def _relation_for_dimension(
    ids: Sequence[str], rank_map: Mapping[str, int], evidences: Mapping[str, ProsodicDeliveryEvidence],
    state_attr: str,
) -> str:
    """Aggregate one safe dimension's pairwise votes into a single compact
    relation string: a candidate_id if every pair that produces a vote
    agrees on the SAME favored candidate, CONFLICTED if pairs disagree,
    NEAR_EQUAL if evaluated but no pair differentiates, UNKNOWN if the
    dimension's own state is unavailable on any candidate."""
    votes: set[str] = set()
    any_state_known = False
    for a_id, b_id in combinations(ids, 2):
        a_state = getattr(evidences[a_id], state_attr)
        b_state = getattr(evidences[b_id], state_attr)
        if a_state in rank_map and b_state in rank_map:
            any_state_known = True
        vote = _rank_favor(rank_map, a_state, b_state, a_id, b_id)
        if vote is not None:
            votes.add(vote)
    if not any_state_known:
        return _RELATION_UNKNOWN
    if len(votes) == 0:
        return _RELATION_NEAR_EQUAL
    if len(votes) == 1:
        return next(iter(votes))
    return _RELATION_CONFLICTED


def _rank_favor(rank_map: Mapping[str, int], a_state: str, b_state: str, a_id: str, b_id: str) -> str | None:
    ra = rank_map.get(a_state)
    rb = rank_map.get(b_state)
    if ra is None or rb is None:
        return None
    if ra > rb:
        return a_id
    if rb > ra:
        return b_id
    return None


def _safe_votes_for_pair(a: ProsodicDeliveryEvidence, b: ProsodicDeliveryEvidence) -> Tuple[str | None, ...]:
    """The three INDEPENDENT safe-dimension votes for one pair (continuity/
    pause-structure counted ONCE, per DOUBLE_COUNTING_AUDIT)."""
    return (
        _rank_favor(_CONTINUITY_RANK, a.vocal_continuity_state, b.vocal_continuity_state, a.candidate_id, b.candidate_id),
        _rank_favor(_HESITATION_RANK, a.hesitation_state, b.hesitation_state, a.candidate_id, b.candidate_id),
        _rank_favor(_RESTART_RANK, a.restart_or_interruption_state, b.restart_or_interruption_state, a.candidate_id, b.candidate_id),
    )


def _prosodic_dominates(a: ProsodicDeliveryEvidence, b: ProsodicDeliveryEvidence) -> bool:
    """True iff `a` prosodically dominates `b`: >=1 safe dimension favors
    `a`, AND no safe dimension favors `b`, AND both carry real, EVALUATED
    acoustic evidence."""
    if a.analysis_status != STATUS_EVALUATED or b.analysis_status != STATUS_EVALUATED:
        return False
    votes = _safe_votes_for_pair(a, b)
    favors_a = any(v == a.candidate_id for v in votes)
    favors_b = any(v == b.candidate_id for v in votes)
    return favors_a and not favors_b


def _pair_has_internal_conflict(a: ProsodicDeliveryEvidence, b: ProsodicDeliveryEvidence) -> bool:
    if a.analysis_status != STATUS_EVALUATED or b.analysis_status != STATUS_EVALUATED:
        return False
    votes = _safe_votes_for_pair(a, b)
    distinct = {v for v in votes if v is not None}
    return len(distinct) >= 2


def _dominant_candidate(
    evidences: Mapping[str, ProsodicDeliveryEvidence], ids: Tuple[str, ...],
) -> Tuple[str | None, bool]:
    """Mirrors `bounded_finalist_arbiter._v2_preferred_candidate`'s own
    generic N-candidate (2 or 3) dominance-search + cycle-detection shape
    -- reused pattern, not a new algorithm family. Returns
    `(preferred_id_or_None, internally_conflicted)`."""
    def dominates(x: str, y: str) -> bool:
        return _prosodic_dominates(evidences[x], evidences[y])

    dominant = [cid for cid in ids if all(dominates(cid, other) for other in ids if other != cid)]
    if len(dominant) == 1:
        return dominant[0], False
    if len(dominant) == 0:
        if len(ids) >= 3:
            for a, b, c in permutations(ids, 3):
                if dominates(a, b) and dominates(b, c) and dominates(c, a):
                    return None, True
        return None, False
    # Defensive: strict dominance should make >1 simultaneous full-
    # dominators impossible, but report conflict rather than pick.
    return None, True


def _descriptive_relation(ids: Sequence[str], evidences: Mapping[str, ProsodicDeliveryEvidence], value_attr: str) -> str:
    """Purely observational -- reports which candidate has the (unique)
    higher raw value, or NEAR_EQUAL/UNKNOWN. NEVER consulted by
    `comparison_state`/`preferred_candidate_id` (see module docstring's
    Category B doctrine and `DOUBLE_COUNTING_AUDIT`)."""
    values = {cid: getattr(evidences[cid], value_attr) for cid in ids}
    known = {cid: v for cid, v in values.items() if v is not None}
    if not known:
        return _RELATION_UNKNOWN
    max_v = max(known.values())
    top = [cid for cid, v in known.items() if v == max_v]
    if len(top) != 1 or len(known) < len(ids):
        # A genuine tie among the known values, OR some candidate's value
        # is unknown -- never guess a direction from partial data.
        if len(top) != 1:
            return _RELATION_NEAR_EQUAL
        return _RELATION_UNKNOWN
    return top[0]


def _emphasis_relation(ids: Sequence[str], evidences: Mapping[str, ProsodicDeliveryEvidence]) -> str:
    states = {cid: evidences[cid].emphasis_dynamics_state for cid in ids}
    if any(s == UNKNOWN for s in states.values()):
        return _RELATION_UNKNOWN
    present = [cid for cid, s in states.items() if s == EMPHASIS_PRESENT]
    if len(present) == 1:
        return present[0]
    return _RELATION_NEAR_EQUAL  # both/none show emphasis -- no distinguishing signal


def compare_prosodic_finalists(
    evidence_by_id: Mapping[str, ProsodicDeliveryEvidence | None],
    candidate_ids: Sequence[str],
) -> ProsodicFinalistComparison:
    """Pure, deterministic: the SAME evidence map always yields the SAME
    comparison, independent of candidate order, clip ids, or dict
    ordering. Never mutates anything; there is no `selected_clip_id`/
    `winner`/`action` field on the return type at all."""
    ids = tuple(candidate_ids)
    provenance = "prosodic_finalist_comparison_v1"

    if not (_MIN_FINALISTS <= len(ids) <= _MAX_FINALISTS):
        return _abstain_comparison(
            ids, state=COMPARISON_NOT_EVALUABLE, missing=("candidate_count_out_of_range",), provenance=provenance,
        )

    evidences_raw = {cid: evidence_by_id.get(cid) for cid in ids}
    if any(evidences_raw[cid] is None for cid in ids):
        return _abstain_comparison(
            ids, state=COMPARISON_NOT_EVALUABLE, missing=("prosodic_delivery_evidence",), provenance=provenance,
        )
    evidences: Mapping[str, ProsodicDeliveryEvidence] = evidences_raw  # type: ignore[assignment]

    evaluated_flags = [evidences[cid].analysis_status == STATUS_EVALUATED for cid in ids]
    if not any(evaluated_flags):
        return _abstain_comparison(
            ids, state=COMPARISON_NOT_EVALUABLE, missing=("acoustic_evidence",), provenance=provenance,
        )
    if not all(evaluated_flags):
        return _abstain_comparison(
            ids, state=COMPARISON_INSUFFICIENT_EVIDENCE, missing=("acoustic_evidence_partial",), provenance=provenance,
        )

    pref, cycle_conflict = _dominant_candidate(evidences, ids)
    pair_conflict = any(
        _pair_has_internal_conflict(evidences[a], evidences[b]) for a, b in combinations(ids, 2)
    )
    if cycle_conflict or (pref is None and pair_conflict):
        comparison_state = COMPARISON_CONFLICTED
        preferred = None
    elif pref is not None:
        comparison_state = COMPARISON_DOMINANT
        preferred = pref
    else:
        comparison_state = COMPARISON_NEAR_EQUAL
        preferred = None

    continuity_comparison = _relation_for_dimension(ids, _CONTINUITY_RANK, evidences, "vocal_continuity_state")
    hesitation_comparison = _relation_for_dimension(ids, _HESITATION_RANK, evidences, "hesitation_state")
    restart_comparison = _relation_for_dimension(ids, _RESTART_RANK, evidences, "restart_or_interruption_state")
    # Same underlying D-187 signal as continuity -- reported identically,
    # never independently derived (DOUBLE_COUNTING_AUDIT).
    pause_structure_comparison = continuity_comparison

    directional_evidence_present = any(
        rel not in (_RELATION_NEAR_EQUAL, _RELATION_UNKNOWN)
        for rel in (continuity_comparison, hesitation_comparison, restart_comparison)
    )

    descriptive_rate_relation = _descriptive_relation(ids, evidences, "speech_rate")
    descriptive_energy_relation = _descriptive_relation(ids, evidences, "energy_variation")
    descriptive_emphasis_relation = _emphasis_relation(ids, evidences)

    missing_evidence = ["pitch_analysis"]  # never available this phase, honest always

    return ProsodicFinalistComparison(
        candidate_ids=ids,
        comparison_state=comparison_state,
        preferred_candidate_id=preferred,
        continuity_comparison=continuity_comparison,
        hesitation_comparison=hesitation_comparison,
        restart_comparison=restart_comparison,
        pause_structure_comparison=pause_structure_comparison,
        descriptive_rate_relation=descriptive_rate_relation,
        descriptive_energy_relation=descriptive_energy_relation,
        descriptive_emphasis_relation=descriptive_emphasis_relation,
        pitch_status=PITCH_NOT_IMPLEMENTED,
        directional_evidence_present=directional_evidence_present,
        conflict_present=comparison_state == COMPARISON_CONFLICTED,
        missing_evidence=tuple(missing_evidence),
        evidence_sources=("prosodic_audio_v2",),
        provenance=provenance,
    )


def _abstain_comparison(
    ids: Tuple[str, ...], *, state: str, missing: Tuple[str, ...], provenance: str,
) -> ProsodicFinalistComparison:
    return ProsodicFinalistComparison(
        candidate_ids=ids,
        comparison_state=state,
        preferred_candidate_id=None,
        continuity_comparison=_RELATION_UNKNOWN,
        hesitation_comparison=_RELATION_UNKNOWN,
        restart_comparison=_RELATION_UNKNOWN,
        pause_structure_comparison=_RELATION_UNKNOWN,
        descriptive_rate_relation=_RELATION_UNKNOWN,
        descriptive_energy_relation=_RELATION_UNKNOWN,
        descriptive_emphasis_relation=_RELATION_UNKNOWN,
        pitch_status=PITCH_NOT_IMPLEMENTED,
        directional_evidence_present=False,
        conflict_present=False,
        missing_evidence=tuple(missing) + ("pitch_analysis",),
        evidence_sources=(),
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Compact per-candidate-set diagnostics row (this task's own required shape,
# 13 keys -- separate from D-184's own closed 13-key
# `bounded_finalist_arbiter_diagnostics` contract, never merged into it).
# ---------------------------------------------------------------------------
def prosodic_finalist_diagnostics(comparison: ProsodicFinalistComparison) -> dict:
    return {
        "prosodic_finalist_evaluated": comparison.comparison_state not in (
            COMPARISON_NOT_EVALUABLE, COMPARISON_INSUFFICIENT_EVIDENCE,
        ),
        "prosodic_finalist_state": comparison.comparison_state,
        "prosodic_finalist_preferred_candidate_id": comparison.preferred_candidate_id,
        "prosodic_finalist_continuity_relation": comparison.continuity_comparison,
        "prosodic_finalist_hesitation_relation": comparison.hesitation_comparison,
        "prosodic_finalist_restart_relation": comparison.restart_comparison,
        "prosodic_finalist_pause_relation": comparison.pause_structure_comparison,
        "prosodic_finalist_descriptive_rate_relation": comparison.descriptive_rate_relation,
        "prosodic_finalist_descriptive_energy_relation": comparison.descriptive_energy_relation,
        "prosodic_finalist_descriptive_emphasis_relation": comparison.descriptive_emphasis_relation,
        "prosodic_finalist_directional_evidence_present": comparison.directional_evidence_present,
        "prosodic_finalist_conflict": comparison.conflict_present,
        "prosodic_finalist_missing_evidence": comparison.missing_evidence,
    }


# D-188 run-level tail-safe summary -- a pure aggregator over already-
# computed comparisons, never a recomputation of any family's own verdict.
# Mirrors D-183/D-184's own `*_run_summary` pattern.
def prosodic_finalist_run_summary(comparisons: Iterable[ProsodicFinalistComparison]) -> dict:
    counts = {
        "prosodic_finalist_evaluated_count": 0,
        "prosodic_finalist_dominance_count": 0,
        "prosodic_finalist_near_equal_count": 0,
        "prosodic_finalist_conflicted_count": 0,
        "prosodic_finalist_insufficient_count": 0,
    }
    for c in comparisons:
        if not isinstance(c, ProsodicFinalistComparison):
            continue
        if c.comparison_state == COMPARISON_DOMINANT:
            counts["prosodic_finalist_evaluated_count"] += 1
            counts["prosodic_finalist_dominance_count"] += 1
        elif c.comparison_state == COMPARISON_NEAR_EQUAL:
            counts["prosodic_finalist_evaluated_count"] += 1
            counts["prosodic_finalist_near_equal_count"] += 1
        elif c.comparison_state == COMPARISON_CONFLICTED:
            counts["prosodic_finalist_evaluated_count"] += 1
            counts["prosodic_finalist_conflicted_count"] += 1
        else:  # INSUFFICIENT_EVIDENCE or NOT_EVALUABLE -- folded together here
            counts["prosodic_finalist_insufficient_count"] += 1
    return counts
