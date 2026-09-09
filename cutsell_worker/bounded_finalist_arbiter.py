"""D-184: Bounded Finalist Arbiter -- OFFLINE / DIAGNOSTIC ONLY.

D-183 (`pipeline.py`'s `TerminalBestTakeConfidence`) answers "do we
actually know who is better?" for the terminal Steps 6-9 comparison. This
module answers the SEPARATE, narrower question D-183 explicitly left
open: "if the normal terminal ladder does NOT know
(`terminal_besttake_confidence_state` is `NON_DECISIVE`, `TIED`, or
`CONFLICTED`), do our existing independent structured signals provide a
SAFE preference among a small (2-3) meaning-sufficient finalist set?"

THE ARBITER MAY PREFER. THE ARBITER MAY ABSTAIN. THE ARBITER MUST NOT BE
FORCED TO PICK. There is no `max(score)`, no fallback-to-first, no
source-order pick, no clip-id/family-id pick anywhere in this module.
`action_applied` is `False` on every result this task produces --
`evaluate_bounded_finalist_arbiter` never mutates `selected_clip_id`,
`preferred_id`, `DraftClip` membership, family winner, retry family, or
render plan; it returns a diagnostic verdict only.

NO RAW-SCORE REPACKAGING: `FinalistArbiterInput.terminal_scores` exists
so a diagnostics consumer can SEE the raw scores that already produced
D-183's own non-decisive state, but this module's decision logic never
reads that field. Since raw score already failed to decide the terminal
comparison, it cannot be smuggled back in here as decisive evidence under
a different name.

Evidence hierarchy (highest authority first):
    P0. Meaning parity/safety (never overridden by anything below).
    1.  Factual performance dominance (D-172 Zone-Usability V2 --
        `zone_usability_v2_dominates`, REUSED verbatim, never a new
        severity-ranking algorithm).
    2.  Editability/Boundary-safe advantage (consumed only if an
        independent, already-canonical comparator supplies one; none
        exists in this codebase today outside V2's own Boundary firewall,
        so this is realistically empty in live wiring and exists only so
        a genuinely independent future signal, or a synthetic test, has
        somewhere principled to plug in).
    3.  Existing structured comparative evidence (the same D-172 V2
        result IS this dimension -- see the double-counting audit below).

Double-counting audit (this task's own explicit requirement): D-163
(`case_b_performance_evidence.py`) and D-172 (`watch_listen_zone_
usability_v2.py`) evidence are PARTIALLY_CORRELATED / the SAME underlying
Track-C behavior events (see `watch_listen_besttake_evidence.py`'s own
`DOUBLE_COUNTING_AUDIT` and `watch_listen_besttake_v2_evidence.py`'s own
module docstring). This module therefore treats them as ONE performance
comparison dimension: `performance_evidence_by_id` (D-163's raw CASE B
evidence objects) is carried on `FinalistArbiterInput` for provenance/
audit visibility ONLY -- the decision logic never reads it. Only
`v2_evidence_by_id` (D-172) is ever consulted to produce a performance
preference. D-163 and D-172 are never independently voted as two sources.

Prosodic Audio: `PROSODIC_AUDIO_AVAILABLE = False` (module-level flag,
UNCHANGED since D-184 -- restates the DEFAULT posture when no Prosodic
comparison is supplied; every pre-D-188 test that never sets
`FinalistArbiterInput.prosodic_comparison` sees byte-identical behavior).
This module never itself infers energy/confidence/hesitation/emphasis/
cadence from transcript or visual evidence -- ALL such inference happens
exclusively in `prosodic_audio_v2.py` (D-187) and
`prosodic_finalist_comparison.py` (D-188), never here.

D-188 (POST D-184, ADDITIVE, OFFLINE, DIAGNOSTIC FUSION ONLY): when a
caller supplies `FinalistArbiterInput.prosodic_comparison` (a
`prosodic_finalist_comparison.ProsodicFinalistComparison`, default
`None`), it is consulted as a FOURTH independent evidence dimension
(`PROSODIC_DELIVERY`) alongside `MEANING` (P0), `VISUAL/PERFORMANCE`
(D-172 V2), and `EDITABILITY` -- using the SAME unanimous-agreement-or-
conflict merge this module already used for those three (never majority
voting, never a new algorithm). A `ProsodicFinalistComparison` in its
own internal `CONFLICTED` state aborts immediately to `ABSTAIN`/
`CONFLICTED`, mirroring D-172's own `v2_conflict` early-return. Omitting
`prosodic_comparison` entirely (the default) reproduces D-184's own
original, closed behavior exactly -- `missing_evidence` still contains
the literal string `"prosodic_audio"` and `bounded_finalist_arbiter_
prosodic_audio_status` in `bounded_finalist_arbiter_diagnostics` stays
the historical, always-`"NOT_AVAILABLE"` value it always was (that 13-
key diagnostics contract is CLOSED and untouched by D-188 -- see the new,
separate `bounded_finalist_arbiter_prosodic_fusion_diagnostics`/
`prosodic_comparison_status` field instead). When candidates remain
indistinguishable even with Prosodic evidence consulted, `ABSTAIN`
(state `NEAR_EQUAL` or `INSUFFICIENT_EVIDENCE`) is still the correct,
expected result -- not a defect to be tuned away.

P1 / global context: NOT consulted, NOT implemented here. This is LOCAL
FINALIST arbitration only -- no sequence-position role, no commercial
role, no upstream sales-conversion logic, no moment-level story
understanding, no whole-video reasoning.

No provider: this module never calls any external LLM or network
provider. Phase A uses only already-computed, in-process structured
evidence.

Eligibility (checked FIRST, before any evidence is even read):
    `terminal_confidence_state` must be one of D-183's own public states
    `NON_DECISIVE` / `TIED` / `CONFLICTED` (the exact string literals
    `pipeline.terminal_besttake_confidence_diagnostics` already emits --
    reused verbatim, never re-declared as a private duplicate ontology),
    AND `2 <= len(candidate_ids) <= 3`. Otherwise: `NOT_ELIGIBLE` /
    `ABSTAIN`, and nothing below is evaluated.

Decision vocabulary: `PREFER_CANDIDATE` / `ABSTAIN` (no separate
`PREFER_A`/`PREFER_B`/`PREFER_C` enum values -- `preferred_candidate_id`
already identifies which candidate, so a parallel per-letter vocabulary
would be a duplicate ontology).

Arbiter states (compact, no state zoo): `PREFERENCE_SUPPORTED`,
`NEAR_EQUAL`, `CONFLICTED`, `INSUFFICIENT_EVIDENCE`, `NOT_ELIGIBLE`.

Feature flag: `CUTSELL_BOUNDED_FINALIST_ARBITER_ENABLED`, default OFF,
mirroring `watch_listen_besttake_evidence.py`'s own
`CUTSELL_WATCH_LISTEN_BESTTAKE_EVIDENCE_ENABLED` convention. OFF: no
pipeline wiring calls into this module at all. ON: diagnostics-only --
`selected_clip_id`/`ranked`/membership/Boundary/Pacing/Renderer stay
byte-identical either way, in this task, always (`action_applied` is
hardcoded `False`, never read as a mutation signal by any caller here).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Iterable, Mapping, Sequence, Tuple
import os

from .language_proposition_relation import build_claim_signature, claim_signatures_conflict
from .prosodic_finalist_comparison import (
    COMPARISON_CONFLICTED,
    COMPARISON_DOMINANT,
    COMPARISON_NEAR_EQUAL,
    ProsodicFinalistComparison,
)
from .watch_listen_zone_usability_v2 import CandidateZoneUsabilityV2, zone_usability_v2_dominates

SCHEMA_VERSION = "cutsell.bounded_finalist_arbiter.v1"

PROSODIC_AUDIO_AVAILABLE = False

_BOUNDED_FINALIST_ARBITER_ENV = "CUTSELL_BOUNDED_FINALIST_ARBITER_ENABLED"

# D-183's own public confidence-state vocabulary (the exact strings
# `terminal_besttake_confidence_diagnostics` emits under
# `terminal_besttake_confidence_state`). Reused verbatim as the
# eligibility gate's input -- never re-declared privately here, and never
# imported from `pipeline.py`'s own private `_TERMINAL_CONFIDENCE_*`
# constants (importing pipeline back from here would risk a load-order
# cycle once pipeline.py imports this module for wiring).
_ELIGIBLE_TERMINAL_CONFIDENCE_STATES = frozenset({"NON_DECISIVE", "TIED", "CONFLICTED"})
_MIN_FINALISTS = 2
_MAX_FINALISTS = 3

DECISION_PREFER_CANDIDATE = "PREFER_CANDIDATE"
DECISION_ABSTAIN = "ABSTAIN"

STATE_PREFERENCE_SUPPORTED = "PREFERENCE_SUPPORTED"
STATE_NEAR_EQUAL = "NEAR_EQUAL"
STATE_CONFLICTED = "CONFLICTED"
STATE_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
STATE_NOT_ELIGIBLE = "NOT_ELIGIBLE"

_MEANING_PARITY_CONSISTENT = "CONSISTENT"
_MEANING_PARITY_CONFLICT = "CONFLICT"
_MEANING_PARITY_UNKNOWN = "UNKNOWN"

_STATUS_DOMINANT = "DOMINANT"
_STATUS_NEAR_EQUAL = "NEAR_EQUAL"
_STATUS_CONFLICTED = "CONFLICTED"
_STATUS_NO_EVIDENCE = "NO_EVIDENCE"
_STATUS_NOT_EVALUATED = "NOT_EVALUATED"

# This task's own explicit double-counting audit requirement: a static,
# code-derived record of how `performance_evidence_by_id` (D-163) relates
# to `v2_evidence_by_id` (D-172) inside this module. Read directly by
# tests as the audit's own source of truth -- never re-derived per call.
DOUBLE_COUNTING_AUDIT: Mapping[str, str] = {
    "v2_evidence_by_id": "DECISION_SOURCE",
    "performance_evidence_by_id": "PARTIALLY_CORRELATED_WITH_v2_evidence_by_id_NEVER_INDEPENDENTLY_VOTED",
    "terminal_scores": "VISIBLE_ONLY_NEVER_DECISIVE",
    "editability_preferred_candidate_id": "INDEPENDENT_WHEN_SUPPLIED_NONE_EXISTS_TODAY_IN_LIVE_WIRING",
    "prosodic_comparison": (
        "D188_INDEPENDENT_FOURTH_DIMENSION_WHEN_SUPPLIED_DEFAULT_NONE_"
        "PRESERVES_D184_ORIGINAL_BEHAVIOR_EXACTLY_SEE_prosodic_finalist_"
        "comparison_py_OWN_DOUBLE_COUNTING_AUDIT_FOR_CONTINUITY_VS_PAUSE"
    ),
    "prosodic_audio": "LEGACY_ALWAYS_NOT_AVAILABLE_STRING_UNCHANGED_SINCE_D184_SEE_prosodic_comparison_FOR_D188",
    "p1_global_context": "NOT_AVAILABLE_NOT_CONSULTED",
}


def _env_true_default_false(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def bounded_finalist_arbiter_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_BOUNDED_FINALIST_ARBITER_ENV))


@dataclass(frozen=True)
class FinalistArbiterInput:
    """Everything the arbiter is allowed to look at for one family's
    finalist set. No QA-oracle-reference fields appear here -- those
    references are oracle-only, never fed into runtime logic)."""
    family_id: str
    candidate_ids: Tuple[str, ...]
    meaning_sufficient_candidate_ids: Tuple[str, ...]
    terminal_confidence_state: str | None
    # D-183's own raw terminal scores, carried for DIAGNOSTIC VISIBILITY
    # ONLY -- see module docstring's "NO RAW-SCORE REPACKAGING". Never
    # read by `evaluate_bounded_finalist_arbiter`'s decision logic.
    terminal_scores: Mapping[str, float | None] = field(default_factory=dict)
    # P0 language/proposition evidence: candidate_id -> normalized text.
    # Used ONLY to run `language_proposition_relation.py`'s OWN existing
    # negation/number/claim-type conflict primitives
    # (`build_claim_signature` / `claim_signatures_conflict`) -- no new
    # NLP/LLM engine. Omit a candidate (or supply none at all) rather
    # than fabricate text; an unverified pair is reported `UNKNOWN`, never
    # silently assumed `CONSISTENT`.
    candidate_texts: Mapping[str, str] = field(default_factory=dict)
    # D-172 Zone-Usability V2 evidence -- the ONE performance-comparison
    # dimension this arbiter consults (see module docstring's
    # double-counting audit).
    v2_evidence_by_id: Mapping[str, "CandidateZoneUsabilityV2 | None"] = field(default_factory=dict)
    # D-163 CASE B performance evidence objects -- carried for provenance/
    # audit visibility only. PARTIALLY_CORRELATED with `v2_evidence_by_id`
    # (see `DOUBLE_COUNTING_AUDIT`); never independently scored here.
    performance_evidence_by_id: Mapping[str, object] = field(default_factory=dict)
    # Boundary/editability evidence, ONLY where an independent, already-
    # canonical comparator supplies it (none exists in live wiring today
    # -- see module docstring). `None` is the honest, expected default.
    editability_preferred_candidate_id: str | None = None
    # D-188: the FOURTH independent evidence dimension (PROSODIC_DELIVERY),
    # a `prosodic_finalist_comparison.ProsodicFinalistComparison` built
    # from D-187's `ProsodicDeliveryEvidence` for this same finalist set.
    # `None` (the default) reproduces D-184's own original behavior
    # exactly -- no existing caller/test that never sets this field is
    # affected in any way.
    prosodic_comparison: "ProsodicFinalistComparison | None" = None
    provenance: str = "bounded_finalist_arbiter_v1"


@dataclass(frozen=True)
class BoundedFinalistArbiterResult:
    """One family's bounded-arbiter verdict. `action_applied` is always
    `False` in this task -- this is a diagnostic verdict, never a winner
    mutation."""
    candidate_ids: Tuple[str, ...]
    decision: str
    preferred_candidate_id: str | None
    arbiter_state: str
    reason: str
    meaning_parity_status: str
    performance_comparison_status: str
    editability_comparison_status: str
    structured_conflict: bool
    missing_evidence: Tuple[str, ...]
    evidence_sources: Tuple[str, ...]
    provenance: str
    action_applied: bool
    # D-188: per-call Prosodic fusion status -- NOT_AVAILABLE (no
    # `prosodic_comparison` supplied, the D-184-original default),
    # AVAILABLE (a real DOMINANT/NEAR_EQUAL comparison was consulted),
    # INSUFFICIENT (supplied but NOT_EVALUABLE/INSUFFICIENT_EVIDENCE), or
    # CONFLICTED (the comparison's own internal state was CONFLICTED).
    # Deliberately NOT merged into the closed 13-key
    # `bounded_finalist_arbiter_diagnostics` contract -- see
    # `bounded_finalist_arbiter_prosodic_fusion_diagnostics` instead.
    prosodic_comparison_status: str = "NOT_AVAILABLE"


def _abstain(
    *,
    candidate_ids: Tuple[str, ...],
    state: str,
    reason: str,
    meaning_parity_status: str = _MEANING_PARITY_UNKNOWN,
    performance_comparison_status: str = _STATUS_NOT_EVALUATED,
    editability_comparison_status: str = _STATUS_NOT_EVALUATED,
    structured_conflict: bool = False,
    missing_evidence: Tuple[str, ...] = (),
    evidence_sources: Tuple[str, ...] = (),
    preferred_candidate_id: str | None = None,
    provenance: str,
    prosodic_comparison_status: str = "NOT_AVAILABLE",
) -> BoundedFinalistArbiterResult:
    return BoundedFinalistArbiterResult(
        candidate_ids=candidate_ids,
        decision=DECISION_ABSTAIN,
        preferred_candidate_id=preferred_candidate_id,
        arbiter_state=state,
        reason=reason,
        meaning_parity_status=meaning_parity_status,
        performance_comparison_status=performance_comparison_status,
        editability_comparison_status=editability_comparison_status,
        structured_conflict=structured_conflict,
        missing_evidence=missing_evidence,
        evidence_sources=evidence_sources,
        provenance=provenance,
        action_applied=False,
        prosodic_comparison_status=prosodic_comparison_status,
    )


def _meaning_conflict_pairs_from_texts(
    candidate_ids: Sequence[str],
    candidate_texts: Mapping[str, str],
) -> bool:
    """True iff two DIFFERENT finalists' supplied texts materially
    conflict (negation polarity flip, differing critical numbers/claim
    type -- see `language_proposition_relation.claim_signatures_conflict`
    for the exact, already-existing, deterministic definition). Pairs
    missing text on either side are never compared (never fabricated)."""
    signatures = {
        cid: build_claim_signature(cid, candidate_texts[cid])
        for cid in candidate_ids
        if cid in candidate_texts and str(candidate_texts[cid] or "").strip()
    }
    ids_with_text = list(signatures.keys())
    for i, left_id in enumerate(ids_with_text):
        for right_id in ids_with_text[i + 1:]:
            if claim_signatures_conflict(signatures[left_id], signatures[right_id]):
                return True
    return False


def _v2_preferred_candidate(
    candidate_ids: Sequence[str],
    v2_evidence_by_id: Mapping[str, "CandidateZoneUsabilityV2 | None"],
) -> Tuple[str | None, bool, bool]:
    """Returns `(preferred_id_or_None, internally_conflicted, has_evidence)`.

    Reuses `zone_usability_v2_dominates` (D-172) verbatim as the ONLY
    dominance test -- no new severity-ranking algorithm is implemented
    here. `has_evidence` is True only when EVERY finalist has a real V2
    projection; a missing projection for any finalist means this source
    has nothing safe to say about the full comparison (a partial
    comparison is never guessed at). A directed 3-cycle among the
    pairwise dominance edges (A>B>C>A) is reported as an internal
    conflict, never resolved by an arbitrary pick."""
    ids = tuple(candidate_ids)
    evidences = {cid: v2_evidence_by_id.get(cid) for cid in ids}
    if any(evidences[cid] is None for cid in ids):
        return None, False, False

    def dominates(a: str, b: str) -> bool:
        return zone_usability_v2_dominates(evidences[a], evidences[b])

    dominant = [
        cid for cid in ids
        if all(dominates(cid, other) for other in ids if other != cid)
    ]
    if len(dominant) == 1:
        return dominant[0], False, True
    if len(dominant) == 0:
        if len(ids) >= 3:
            for a, b, c in permutations(ids, 3):
                if dominates(a, b) and dominates(b, c) and dominates(c, a):
                    return None, True, True
        return None, False, True
    # Defensive: strict pairwise dominance should make >1 simultaneous
    # "dominates every other finalist" impossible, but a genuine conflict
    # is reported rather than picking arbitrarily if it ever occurs.
    return None, True, True


def evaluate_bounded_finalist_arbiter(data: FinalistArbiterInput) -> BoundedFinalistArbiterResult:
    """Core D-184 classifier. Deterministic: the SAME evidence always
    yields the SAME result, independent of candidate input order, clip
    ids, family ids, or dict/set ordering. Never mutates anything on
    `data` or elsewhere -- `action_applied` is always `False`."""
    candidate_ids = tuple(data.candidate_ids)
    n = len(candidate_ids)
    provenance = data.provenance

    # --- Eligibility gate (FIRST; nothing below is evaluated otherwise) ---
    if n < _MIN_FINALISTS or n > _MAX_FINALISTS:
        return _abstain(
            candidate_ids=candidate_ids,
            state=STATE_NOT_ELIGIBLE,
            reason="candidate_count_not_eligible",
            missing_evidence=("prosodic_audio", "p1_global_context"),
            provenance=provenance,
        )
    if data.terminal_confidence_state not in _ELIGIBLE_TERMINAL_CONFIDENCE_STATES:
        return _abstain(
            candidate_ids=candidate_ids,
            state=STATE_NOT_ELIGIBLE,
            reason="terminal_confidence_state_not_eligible",
            missing_evidence=("prosodic_audio", "p1_global_context"),
            provenance=provenance,
        )

    # --- P0 meaning parity/safety (SECOND; overrides everything below) ---
    meaning_sufficient = set(data.meaning_sufficient_candidate_ids)
    outside_meaning_sufficient = any(cid not in meaning_sufficient for cid in candidate_ids)
    texts_for_finalists = {cid: data.candidate_texts[cid] for cid in candidate_ids if cid in data.candidate_texts}
    meaning_conflict = outside_meaning_sufficient or _meaning_conflict_pairs_from_texts(
        candidate_ids, texts_for_finalists,
    )
    if meaning_conflict:
        return _abstain(
            candidate_ids=candidate_ids,
            state=STATE_CONFLICTED,
            reason=(
                "candidate_not_meaning_sufficient" if outside_meaning_sufficient
                else "meaning_parity_conflict_detected"
            ),
            meaning_parity_status=_MEANING_PARITY_CONFLICT,
            structured_conflict=True,
            missing_evidence=("prosodic_audio", "p1_global_context"),
            evidence_sources=("language_proposition_relation",),
            provenance=provenance,
        )
    meaning_parity_status = (
        _MEANING_PARITY_CONSISTENT if len(texts_for_finalists) == n else _MEANING_PARITY_UNKNOWN
    )

    # --- Evidence gathering ---
    missing_evidence: list[str] = []
    evidence_sources: list[str] = []
    sources: list[Tuple[str, str | None]] = []

    v2_pref, v2_conflict, v2_has_evidence = _v2_preferred_candidate(candidate_ids, data.v2_evidence_by_id)
    if v2_conflict:
        performance_comparison_status = _STATUS_CONFLICTED
    elif v2_has_evidence:
        performance_comparison_status = _STATUS_DOMINANT if v2_pref is not None else _STATUS_NEAR_EQUAL
    else:
        performance_comparison_status = _STATUS_NO_EVIDENCE
    if v2_has_evidence:
        evidence_sources.append("zone_usability_v2")
        if not v2_conflict:
            sources.append(("zone_usability_v2", v2_pref))
    else:
        missing_evidence.append("zone_usability_v2")

    editability_pref = data.editability_preferred_candidate_id
    if editability_pref is not None and editability_pref in candidate_ids:
        editability_comparison_status = _STATUS_DOMINANT
        evidence_sources.append("editability_evidence")
        sources.append(("editability_evidence", editability_pref))
    else:
        editability_comparison_status = _STATUS_NO_EVIDENCE
        missing_evidence.append("editability_evidence")

    # D-188: PROSODIC_DELIVERY, the fourth independent evidence dimension.
    # `prosodic_comparison` defaults to `None` -- every caller/test that
    # never sets it reproduces D-184's own original hardcoded
    # `missing_evidence.append("prosodic_audio")` behavior exactly.
    prosodic_comparison = data.prosodic_comparison
    prosodic_pref: str | None = None
    prosodic_internal_conflict = False
    if prosodic_comparison is None:
        prosodic_comparison_status = "NOT_AVAILABLE"
        missing_evidence.append("prosodic_audio")
    elif set(prosodic_comparison.candidate_ids) != set(candidate_ids):
        # Safety guard: a comparison built for a DIFFERENT finalist set is
        # never trusted -- treated exactly like "not supplied".
        prosodic_comparison_status = "NOT_AVAILABLE"
        missing_evidence.append("prosodic_audio")
    elif prosodic_comparison.comparison_state == COMPARISON_CONFLICTED:
        prosodic_comparison_status = "CONFLICTED"
        prosodic_internal_conflict = True
    elif prosodic_comparison.comparison_state == COMPARISON_DOMINANT:
        prosodic_comparison_status = "AVAILABLE"
        evidence_sources.append("prosodic_delivery")
        prosodic_pref = (
            prosodic_comparison.preferred_candidate_id
            if prosodic_comparison.preferred_candidate_id in candidate_ids else None
        )
        sources.append(("prosodic_delivery", prosodic_pref))
    elif prosodic_comparison.comparison_state == COMPARISON_NEAR_EQUAL:
        prosodic_comparison_status = "AVAILABLE"
        evidence_sources.append("prosodic_delivery")
        sources.append(("prosodic_delivery", None))
    else:  # INSUFFICIENT_EVIDENCE / NOT_EVALUABLE
        prosodic_comparison_status = "INSUFFICIENT"
        missing_evidence.append("prosodic_audio")

    missing_evidence.append("p1_global_context")

    # --- Merge (mirrors D-183's own structured-signal aggregation
    # contract -- unanimous non-None preference = supported, disagreement
    # = conflict -- implemented independently here because D-183's own
    # function also folds in raw score, which this arbiter must never
    # treat as decisive input; see module docstring). D-188 extends this
    # SAME merge with a fourth `sources` entry -- no new algorithm, no
    # majority voting, no score. ---
    if v2_conflict:
        return _abstain(
            candidate_ids=candidate_ids,
            state=STATE_CONFLICTED,
            reason="zone_usability_v2_internal_conflict",
            meaning_parity_status=meaning_parity_status,
            performance_comparison_status=performance_comparison_status,
            editability_comparison_status=editability_comparison_status,
            structured_conflict=True,
            missing_evidence=tuple(missing_evidence),
            evidence_sources=tuple(evidence_sources),
            provenance=provenance,
            prosodic_comparison_status=prosodic_comparison_status,
        )

    if prosodic_internal_conflict:
        return _abstain(
            candidate_ids=candidate_ids,
            state=STATE_CONFLICTED,
            reason="prosodic_delivery_internal_conflict",
            meaning_parity_status=meaning_parity_status,
            performance_comparison_status=performance_comparison_status,
            editability_comparison_status=editability_comparison_status,
            structured_conflict=True,
            missing_evidence=tuple(missing_evidence),
            evidence_sources=tuple(evidence_sources),
            provenance=provenance,
            prosodic_comparison_status=prosodic_comparison_status,
        )

    if not sources:
        return _abstain(
            candidate_ids=candidate_ids,
            state=STATE_INSUFFICIENT_EVIDENCE,
            reason="no_structured_evidence_available",
            meaning_parity_status=meaning_parity_status,
            performance_comparison_status=performance_comparison_status,
            editability_comparison_status=editability_comparison_status,
            missing_evidence=tuple(missing_evidence),
            evidence_sources=tuple(evidence_sources),
            provenance=provenance,
            prosodic_comparison_status=prosodic_comparison_status,
        )

    distinct_preferences = {pref for _name, pref in sources if pref is not None}
    if len(distinct_preferences) >= 2:
        return _abstain(
            candidate_ids=candidate_ids,
            state=STATE_CONFLICTED,
            reason="structured_evidence_sources_disagree",
            meaning_parity_status=meaning_parity_status,
            performance_comparison_status=performance_comparison_status,
            editability_comparison_status=editability_comparison_status,
            structured_conflict=True,
            missing_evidence=tuple(missing_evidence),
            evidence_sources=tuple(evidence_sources),
            provenance=provenance,
            prosodic_comparison_status=prosodic_comparison_status,
        )
    if len(distinct_preferences) == 0:
        return _abstain(
            candidate_ids=candidate_ids,
            state=STATE_NEAR_EQUAL,
            reason="structured_evidence_near_equal_no_dominance_found",
            meaning_parity_status=meaning_parity_status,
            performance_comparison_status=performance_comparison_status,
            editability_comparison_status=editability_comparison_status,
            missing_evidence=tuple(missing_evidence),
            evidence_sources=tuple(evidence_sources),
            provenance=provenance,
            prosodic_comparison_status=prosodic_comparison_status,
        )

    preferred = next(iter(distinct_preferences))
    return BoundedFinalistArbiterResult(
        candidate_ids=candidate_ids,
        decision=DECISION_PREFER_CANDIDATE,
        preferred_candidate_id=preferred,
        arbiter_state=STATE_PREFERENCE_SUPPORTED,
        reason="structured_evidence_sources_unanimously_prefer_one_candidate",
        meaning_parity_status=meaning_parity_status,
        performance_comparison_status=performance_comparison_status,
        editability_comparison_status=editability_comparison_status,
        structured_conflict=False,
        missing_evidence=tuple(missing_evidence),
        evidence_sources=tuple(evidence_sources),
        provenance=provenance,
        action_applied=False,
        prosodic_comparison_status=prosodic_comparison_status,
    )


def bounded_finalist_arbiter_diagnostics(result: BoundedFinalistArbiterResult) -> dict:
    """JSON-safe, bounded per-family diagnostics row -- no transcript
    dump, no QA-reference info. Field names match this task's own
    directive verbatim (13 fields)."""
    return {
        "bounded_finalist_arbiter_eligible": result.arbiter_state != STATE_NOT_ELIGIBLE,
        "bounded_finalist_arbiter_candidate_count": len(result.candidate_ids),
        "bounded_finalist_arbiter_state": result.arbiter_state,
        "bounded_finalist_arbiter_decision": result.decision,
        "bounded_finalist_arbiter_preferred_candidate_id": result.preferred_candidate_id,
        "bounded_finalist_arbiter_reason": result.reason,
        "bounded_finalist_arbiter_meaning_parity": result.meaning_parity_status,
        "bounded_finalist_arbiter_performance_status": result.performance_comparison_status,
        "bounded_finalist_arbiter_editability_status": result.editability_comparison_status,
        "bounded_finalist_arbiter_prosodic_audio_status": "NOT_AVAILABLE",
        "bounded_finalist_arbiter_conflict": result.structured_conflict,
        "bounded_finalist_arbiter_missing_evidence": result.missing_evidence,
        "bounded_finalist_arbiter_action_applied": result.action_applied,
    }


# D-184 run-level tail-safe summary -- a pure aggregator over already-
# computed per-family diagnostics rows, never a recomputation of any
# family's own verdict. Mirrors D-183's own
# `terminal_besttake_confidence_run_summary` pattern (7 counts).
def bounded_finalist_arbiter_run_summary(rows: Iterable[Mapping]) -> dict:
    counts = {
        "arbiter_evaluated_count": 0,
        "arbiter_preference_supported_count": 0,
        "arbiter_abstain_count": 0,
        "arbiter_near_equal_count": 0,
        "arbiter_conflicted_count": 0,
        "arbiter_insufficient_evidence_count": 0,
        "arbiter_not_eligible_count": 0,
    }
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        state = row.get("bounded_finalist_arbiter_state")
        if state is None:
            continue
        counts["arbiter_evaluated_count"] += 1
        if row.get("bounded_finalist_arbiter_decision") == DECISION_ABSTAIN:
            counts["arbiter_abstain_count"] += 1
        if state == STATE_PREFERENCE_SUPPORTED:
            counts["arbiter_preference_supported_count"] += 1
        elif state == STATE_NEAR_EQUAL:
            counts["arbiter_near_equal_count"] += 1
        elif state == STATE_CONFLICTED:
            counts["arbiter_conflicted_count"] += 1
        elif state == STATE_INSUFFICIENT_EVIDENCE:
            counts["arbiter_insufficient_evidence_count"] += 1
        elif state == STATE_NOT_ELIGIBLE:
            counts["arbiter_not_eligible_count"] += 1
    return counts


# ---------------------------------------------------------------------------
# D-188: additive Prosodic-fusion diagnostics -- deliberately SEPARATE from
# the closed 13-key `bounded_finalist_arbiter_diagnostics` contract above
# (which stays byte-identical, including its historical always-
# `"NOT_AVAILABLE"` `bounded_finalist_arbiter_prosodic_audio_status` key).
# ---------------------------------------------------------------------------
def bounded_finalist_arbiter_prosodic_fusion_diagnostics(result: BoundedFinalistArbiterResult) -> dict:
    """One additional, bounded per-family row: the D-188 fusion's own
    per-call status plus whether Prosodic evidence contributed to this
    call's `evidence_sources` (regardless of whether it alone decided the
    outcome -- see module docstring)."""
    return {
        "bounded_finalist_arbiter_prosodic_status": result.prosodic_comparison_status,
        "bounded_finalist_arbiter_prosodic_contributed": "prosodic_delivery" in result.evidence_sources,
    }


def bounded_finalist_arbiter_prosodic_fusion_run_summary(
    results: Iterable[BoundedFinalistArbiterResult],
) -> dict:
    """Tail-safe run-level count: how many `PREFER_CANDIDATE` diagnostic
    preferences this run had Prosodic evidence contributing to (inclusive
    of cases where Visual/Editability also agreed -- a simple, honest
    definition, never a claim that Prosody was the SOLE cause)."""
    count = 0
    for result in results:
        if not isinstance(result, BoundedFinalistArbiterResult):
            continue
        if result.decision == DECISION_PREFER_CANDIDATE and "prosodic_delivery" in result.evidence_sources:
            count += 1
    return {"arbiter_preferences_due_to_prosody_count": count}
