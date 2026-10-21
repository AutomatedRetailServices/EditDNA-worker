"""D-188 comparison, D-290.1 safety correction: Phase-A descriptors only.

An interior pause can be a rhetorical beat or a failed delivery. D-187
derives continuity, hesitation and restart categories from the same pause
evidence. Its optional language-restart flag is not scoped to an in-span
error either: it may identify a clean retry after an abandoned attempt.
These fields do not independently prove which realization is better.

Keep raw descriptor relations visible, but never promote their differences
to DOMINANT or CONFLICTED quality evidence. Differing or unknown categories
yield INSUFFICIENT_EVIDENCE with an explicit missing-proof reason; equal
known categories yield NEAR_EQUAL. No threshold or silence detector is
added. D-184 may still use its independently supported V2 evidence.

DOMINANT and CONFLICTED remain part of the typed consumer contract, not
verdicts that this Phase-A producer can certify. A future qualified source
of independent, source-local delivery-error evidence needs its own review
before enabling a prosodic preference. No psychological inference, master
score, provider call, selection mutation or render authority lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence, Tuple
import os
from .watch_listen_runtime import capability_enabled

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

# Descriptor ordering only. Phase A derives these categories from pauses
# or an unscoped language-restart flag, not independently verified in-span
# speech errors. Ordering them must NOT produce an editorial preference.
_CONTINUITY_RANK: Mapping[str, int] = {
    CONTINUITY_CONTINUOUS: 2, CONTINUITY_MILDLY_INTERRUPTED: 1, CONTINUITY_FRAGMENTED: 0,
}
_HESITATION_RANK: Mapping[str, int] = {HESITATION_NOT_OBSERVED: 1, HESITATION_PRESENT: 0}
_RESTART_RANK: Mapping[str, int] = {RESTART_NOT_OBSERVED: 1, RESTART_SUPPORTED: 0}

# This task's own explicit double-counting audit requirement: a static,
# code-derived record of how the diagnostic fields on
# ProsodicFinalistComparison relate to each other and to D-187/Audio V1.
DOUBLE_COUNTING_AUDIT: Mapping[str, str] = {
    "continuity_comparison": "PAUSE_DERIVED_OBSERVATION_NEVER_AN_INDEPENDENT_SAFE_VOTE",
    "pause_structure_comparison": "SAME_UNDERLYING_SIGNAL_AS_continuity_comparison_NEVER_AN_INDEPENDENT_VOTE",
    "hesitation_comparison": "PAUSE_DERIVED_OBSERVATION_NEVER_AN_INDEPENDENT_SAFE_VOTE",
    "restart_comparison": "PAUSE_OR_UNSCOPED_LANGUAGE_OBSERVATION_NEVER_AN_INDEPENDENT_SAFE_VOTE",
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
    return capability_enabled(_PROSODIC_FINALIST_ARBITER_DIAGNOSTICS_ENV, values)


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

    directional_evidence_present: bool  # Phase-A descriptors alone never set this
    conflict_present: bool  # True iff comparison_state == CONFLICTED
    missing_evidence: Tuple[str, ...]

    evidence_sources: Tuple[str, ...]
    provenance: str


def _relation_for_dimension(
    ids: Sequence[str], rank_map: Mapping[str, int], evidences: Mapping[str, ProsodicDeliveryEvidence],
    state_attr: str,
) -> str:
    """Aggregate one descriptive dimension into a single compact
    relation string: a candidate_id if every pair that produces a vote
    agrees on the SAME favored candidate, CONFLICTED if pairs disagree,
    NEAR_EQUAL if evaluated but no pair differentiates, UNKNOWN if the
    dimension's own state is unavailable on any candidate."""
    if any(getattr(evidences[cid], state_attr) not in rank_map for cid in ids):
        return _RELATION_UNKNOWN
    votes: set[str] = set()
    for a_id, b_id in combinations(ids, 2):
        a_state = getattr(evidences[a_id], state_attr)
        b_state = getattr(evidences[b_id], state_attr)
        vote = _rank_favor(rank_map, a_state, b_state, a_id, b_id)
        if vote is not None:
            votes.add(vote)
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

    continuity_comparison = _relation_for_dimension(ids, _CONTINUITY_RANK, evidences, "vocal_continuity_state")
    hesitation_comparison = _relation_for_dimension(ids, _HESITATION_RANK, evidences, "hesitation_state")
    restart_comparison = _relation_for_dimension(ids, _RESTART_RANK, evidences, "restart_or_interruption_state")
    # Same underlying D-187 signal as continuity -- reported identically,
    # never independently derived (DOUBLE_COUNTING_AUDIT).
    pause_structure_comparison = continuity_comparison

    # These are correlated descriptors, not evidence that a pause or a
    # restart belongs to a failed delivery. Even a language restart may
    # mark the clean retry after a preceding abandoned attempt. Phase A
    # carries no source-local error proof to authorize a preference.
    missing_disruption_proof = any(
        rel != _RELATION_NEAR_EQUAL
        for rel in (continuity_comparison, hesitation_comparison, restart_comparison)
    )
    comparison_state = (
        COMPARISON_INSUFFICIENT_EVIDENCE if missing_disruption_proof else COMPARISON_NEAR_EQUAL
    )

    descriptive_rate_relation = _descriptive_relation(ids, evidences, "speech_rate")
    descriptive_energy_relation = _descriptive_relation(ids, evidences, "energy_variation")
    descriptive_emphasis_relation = _emphasis_relation(ids, evidences)

    missing_evidence = ["pitch_analysis"]  # never available this phase, honest always
    if missing_disruption_proof:
        missing_evidence.append("independent_in_span_disruption_evidence")

    return ProsodicFinalistComparison(
        candidate_ids=ids,
        comparison_state=comparison_state,
        preferred_candidate_id=None,
        continuity_comparison=continuity_comparison,
        hesitation_comparison=hesitation_comparison,
        restart_comparison=restart_comparison,
        pause_structure_comparison=pause_structure_comparison,
        descriptive_rate_relation=descriptive_rate_relation,
        descriptive_energy_relation=descriptive_energy_relation,
        descriptive_emphasis_relation=descriptive_emphasis_relation,
        pitch_status=PITCH_NOT_IMPLEMENTED,
        directional_evidence_present=False,
        conflict_present=False,
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
