"""D-235Q: COMPLETE EDITORIAL-REQUIREMENT + LOST-ATOM MATERIALITY
INTEGRATION -- OFFLINE ONLY.

Post D-235P (docs/CUTSELL_DECISIONS.md, VERDICT A -- SHARED WORD-
MEMBERSHIP IDENTITY SEAM OFFLINE PROVEN): this module is the ONE bounded
decision object that COMBINES, never replaces:

    D-235L (`lost_semantic_atom_materiality.py`)     -- meaning-critical
        safety floor, retry/recording-residue, redundant-equivalent,
        non-material real content, all reused by DIRECT IMPORT, unmodified.
    D-235M (`lost_atom_editorial_requirement_evidence.py`) -- editorial
        requirement (story completeness) evidence, reused by DIRECT
        IMPORT, unmodified. Its own `identity_mapping_status`/
        `editorial_slot_evidence`/`slot_evidence_source` parameters
        already anticipated exact identity -- this task is simply the
        FIRST real caller able to supply `IDENTITY_MAPPING_EXACT` with a
        genuinely exact slot value, computed from D-235P.
    D-235P (`shared_attempt_word_identity.py`)       -- exact canonical
        word-membership identity (clip -> exact LanguageAttempt set ->
        exact PropositionCandidate set), reused by DIRECT IMPORT,
        unmodified.

Neither `lost_semantic_atom_materiality.py` nor `lost_atom_editorial_
requirement_evidence.py` nor `shared_attempt_word_identity.py` is
modified by this task (zero diff, confirmed by this module's own
regression tests) -- "upgrading D-235M" per this task's own directive
means constructing its EXISTING `EXACT` identity-mapping inputs from a
real D-235P match for the first time, never touching D-235M's own file.

## What this module is NOT (binding, restated from this task's own scope)

- No live Freeze authority. `assess_complete_lost_semantic_atom_
  materiality()` never reads or writes `freeze_blocked`, `lost_atom.
  blocking`, `FinalEditReviewer` findings, or `RepairLoop` attempts, and
  is not imported by `universal_clean_cut.py`, `final_story_coherence_
  validation.py`, `final_edit_reviewer.py`, `repair_loop.py`, or any
  resolver module. This gate is decision FOUNDATION only -- see D-235R
  (named, not implemented here) for the future bounded Freeze adapter.
- No new semantic engine, no numeric master score. `final_materiality_
  status` and `blocking_recommendation` reuse D-235L's own vocabulary
  constants verbatim (imported, never redefined) -- the SAME 7/3-value
  bounded categorical vocabularies, because D-235L's own directive
  already specified exactly the 7 values this task's own directive
  requires again.
- No atom-ownership guessing. When D-235P's exact proposition SET
  contains more than one proposition, this module NEVER attributes a
  lost atom to one specific proposition. See "Multi-proposition safety"
  below.

## Decision precedence (binding, this task's own required order)

Implemented as a strict, ordered if/elif chain -- each step tested only
after every earlier step's condition has been checked and found false,
matching the directive's own "the order matters" instruction literally:

  1. Meaning safety first (`materiality.materiality_status ==
     MEANING_CRITICAL`) -> BLOCK. Never downgraded by any later step --
     structurally guaranteed by being the FIRST branch in the chain.
  2. Editorial requirement second (`requirement.editorial_requirement_
     status == REQUIRED`) -> BLOCK.
  3. Conflict / unknown safety -- `materiality.materiality_status ==
     CONFLICTED`, OR `requirement.editorial_requirement_status ==
     CONFLICTED`, OR an unresolved multi-proposition ownership ambiguity
     with no independent redundancy proof -> ABSTAIN (`CONFLICTED`).
  4. Retry / recording residue (`materiality.materiality_status ==
     RETRY_OR_RECORDING_RESIDUE`) -> DO_NOT_BLOCK.
  5. Redundant equivalent (`materiality.materiality_status ==
     REDUNDANT_EQUIVALENT` OR `requirement.editorial_requirement_status
     == REDUNDANT_REQUIRED_FUNCTION_PRESERVED`) -> DO_NOT_BLOCK.
  6. Non-material real content -- `materiality.materiality_status ==
     NON_MATERIAL_REAL_CONTENT` AND the editorial-requirement side is
     genuinely clear (see "Identity-sufficiency gate" below) ->
     DO_NOT_BLOCK.
  7. Otherwise -> ABSTAIN (`INSUFFICIENT_EVIDENCE`).

## Identity-sufficiency gate for step 6 (the D-235K real-shape distinction)

Step 6 requires the editorial-requirement side to be genuinely clear,
which means ONE of:

  - `requirement.editorial_requirement_status == NOT_REQUIRED` (D-235M
    reached this via a firewall/explicit-clearance branch that never
    depended on slot evidence at all -- always safe, identity-independent);
  - `requirement.editorial_requirement_status == INSUFFICIENT_EVIDENCE`
    **AND** `exact_identity_available` is True (D-235M's own generic
    "nothing found" default is only trustworthy here when the slot
    evidence it was given was genuinely EXACT and simply empty -- never
    when the slot channel itself was unavailable or heuristic-only).

This is the exact mechanism distinguishing the D-235K real-shape fixture
(exact identity available, no requirement proven -> DO_NOT_BLOCK) from
fixtures 19/20 (identity missing/heuristic-only -> ABSTAIN, even though
D-235M's own output looks identical, `INSUFFICIENT_EVIDENCE`, in both
cases) -- "requirement depends on identity" is operationalized as
`exact_identity_available`, never inferred from D-235M's own status
value alone.

## Multi-proposition safety (binding, restated from this task's own
## "Do not guess atom ownership" / "fail closed" instructions)

When D-235P's exact match resolves to a SINGLE proposition id, its
`editorial_slot_evidence` is supplied to D-235M directly (unambiguous
ownership). When it resolves to MULTIPLE proposition ids (an exact 1->N
partition), this module NEVER attributes the lost atom to one of them:
if ANY proposition in the exact set carries a story-function slot
(`SLOT_HOOK`/`SLOT_CTA`/`SLOT_CONCLUSION` -- the SAME `_STORY_FUNCTION_
SLOTS` set D-235M's own module already defines, mirrored here verbatim
from the public `SLOT_*` constants, never importing a private name),
`ownership_ambiguous` is set and NO slot evidence at all is passed to
D-235M for this row (never a guessed single value) -- the ambiguity is
instead resolved at THIS module's own step 3, and only escapes ABSTAIN
when an INDEPENDENT redundancy signal (`materiality.materiality_status
== REDUNDANT_EQUIVALENT`, `requirement.editorial_requirement_status ==
REDUNDANT_REQUIRED_FUNCTION_PRESERVED`, or `replacement_function_
preserved is True`) already proves the function is preserved elsewhere.
If NO proposition in the exact multi-set carries a story-function slot
at all, there is nothing to be ambiguous about -- `ownership_ambiguous`
stays False and the row proceeds normally.

## Exact identity use (binding)

`exact_match`'s `relationship_status` is checked against D-235P's own
`AUTHORITATIVE_RELATIONSHIP_STATUSES` (imported, never re-derived) before
ANY slot evidence is read from it -- `HEURISTIC_OVERLAP`/non-authoritative
containment/partial-overlap matches are NEVER used to supply
`editorial_slot_evidence`, per this task's own "do not use time-overlap
heuristic or text fuzzy matching to upgrade evidence to authoritative"
instruction, structurally enforced (grep-verified: `editorial_slot_
evidence=` is only ever assigned inside the `if exact_identity_available:`
branch).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .language_proposition_relation import SLOT_CONCLUSION, SLOT_CTA, SLOT_HOOK
from .lost_atom_editorial_requirement_evidence import (
    IDENTITY_MAPPING_EXACT,
    IDENTITY_MAPPING_HEURISTIC_OVERLAP,
    IDENTITY_MAPPING_NONE,
    REQUIREMENT_CONFLICTED,
    REQUIREMENT_INSUFFICIENT_EVIDENCE,
    REQUIREMENT_NOT_REQUIRED,
    REQUIREMENT_REDUNDANT_REQUIRED_FUNCTION_PRESERVED,
    REQUIREMENT_REQUIRED,
    LostAtomEditorialRequirementEvidence,
    assess_editorial_requirement_evidence,
)
from .lost_semantic_atom_materiality import (
    MATERIALITY_CONFLICTED,
    MATERIALITY_EDITORIALLY_REQUIRED,
    MATERIALITY_INSUFFICIENT_EVIDENCE,
    MATERIALITY_MEANING_CRITICAL,
    MATERIALITY_NON_MATERIAL_REAL_CONTENT,
    MATERIALITY_REDUNDANT_EQUIVALENT,
    MATERIALITY_RETRY_OR_RECORDING_RESIDUE,
    RECOMMEND_ABSTAIN,
    RECOMMEND_BLOCK,
    RECOMMEND_DO_NOT_BLOCK,
    STATE_FOUND,
    LostSemanticAtomMateriality,
    assess_lost_semantic_atom_materiality,
)
from .shared_attempt_word_identity import (
    AUTHORITATIVE_RELATIONSHIP_STATUSES,
    AttemptLanguageIdentityMatch,
    exact_proposition_candidate_ids_for_match,
)

SCHEMA_VERSION = "cutsell.complete_lost_semantic_atom_materiality.v1"

# Final materiality vocabulary -- reused VERBATIM from D-235L's own 7-value
# set (the directive's own required vocabulary is identical to D-235L's).
FINAL_MATERIALITY_VOCABULARY = frozenset({
    MATERIALITY_MEANING_CRITICAL, MATERIALITY_EDITORIALLY_REQUIRED,
    MATERIALITY_NON_MATERIAL_REAL_CONTENT, MATERIALITY_RETRY_OR_RECORDING_RESIDUE,
    MATERIALITY_REDUNDANT_EQUIVALENT, MATERIALITY_INSUFFICIENT_EVIDENCE,
    MATERIALITY_CONFLICTED,
})

# Blocking vocabulary -- reused verbatim from D-235L.
_VALID_BLOCKING = frozenset({RECOMMEND_BLOCK, RECOMMEND_DO_NOT_BLOCK, RECOMMEND_ABSTAIN})

# Mirrors lost_atom_editorial_requirement_evidence.py's own private
# `_STORY_FUNCTION_SLOTS` verbatim, built from the SAME public SLOT_*
# constants -- never a new slot invented, never importing a private
# cross-module name (same precedent D-235M itself used for D-235's own
# `_PROCESS_SHAPED_ROLES`).
_STORY_FUNCTION_SLOTS = frozenset({SLOT_HOOK, SLOT_CTA, SLOT_CONCLUSION})


@dataclass(frozen=True)
class CompleteLostSemanticAtomMateriality:
    clip_id: str
    exact_identity_available: bool
    exact_language_attempt_ids: Tuple[str, ...]
    exact_proposition_candidate_ids: Tuple[str, ...]
    meaning_materiality_status: str
    editorial_requirement_status: str
    retry_or_process_status: str
    redundancy_status: str
    final_materiality_status: str
    blocking_recommendation: str
    reason_codes: Tuple[str, ...]
    provenance: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.final_materiality_status not in FINAL_MATERIALITY_VOCABULARY:
            raise ValueError(f"invalid final_materiality_status: {self.final_materiality_status!r}")
        if self.blocking_recommendation not in _VALID_BLOCKING:
            raise ValueError(f"invalid blocking_recommendation: {self.blocking_recommendation!r}")

    def as_dict(self) -> dict:
        """JSON-safe, no transcript dump."""
        return {
            "clip_id": self.clip_id,
            "exact_identity_available": self.exact_identity_available,
            "exact_language_attempt_ids": list(self.exact_language_attempt_ids),
            "exact_proposition_candidate_ids": list(self.exact_proposition_candidate_ids),
            "meaning_materiality_status": self.meaning_materiality_status,
            "editorial_requirement_status": self.editorial_requirement_status,
            "retry_or_process_status": self.retry_or_process_status,
            "redundancy_status": self.redundancy_status,
            "final_materiality_status": self.final_materiality_status,
            "blocking_recommendation": self.blocking_recommendation,
            "reason_codes": list(self.reason_codes),
            "provenance": list(self.provenance),
        }


def _exact_slot_for_proposition_set(
    proposition_ids: Tuple[str, ...],
    proposition_slot_evidence_by_id: Mapping[str, str],
) -> Tuple[Optional[str], bool]:
    """Returns (safe_single_slot_or_None, ownership_ambiguous). Never
    guesses which proposition in a multi-member exact set owns the lost
    atom -- see module docstring's "Multi-proposition safety" section."""
    if not proposition_ids:
        return None, False
    if len(proposition_ids) == 1:
        return proposition_slot_evidence_by_id.get(proposition_ids[0]), False
    slots_present = {proposition_slot_evidence_by_id.get(pid) for pid in proposition_ids}
    slots_present.discard(None)
    if slots_present & _STORY_FUNCTION_SLOTS:
        return None, True
    return None, False


def assess_complete_lost_semantic_atom_materiality(
    row: Mapping,
    *,
    critical_claim_conflict: Optional[bool] = None,
    recording_process_evidence: Optional[bool] = None,
    exact_match: Optional[AttemptLanguageIdentityMatch] = None,
    proposition_candidate_ids_by_attempt_id: Optional[Mapping[str, Tuple[str, ...]]] = None,
    proposition_slot_evidence_by_id: Optional[Mapping[str, str]] = None,
    heuristic_identity_available: bool = False,
    recording_process_status: Optional[str] = None,
    audience_delivery_status: Optional[str] = None,
    idea_coverage_status: Optional[bool] = None,
    replacement_function_preserved: Optional[bool] = None,
    downstream_dependency_present: Optional[bool] = None,
) -> CompleteLostSemanticAtomMateriality:
    """The one D-235Q entry point. Pure function of one already-computed
    `_lost_semantic_atoms()` row, D-235L's own two context flags, D-235P's
    own optional exact identity match, and D-235M's own remaining
    optional structured signals. Calls `assess_lost_semantic_atom_
    materiality` and `assess_editorial_requirement_evidence` internally
    (both unmodified imports) and combines their outputs per the module
    docstring's decision precedence. Mints nothing, mutates nothing,
    never called from any live Freeze/repair/resolver module.

    `exact_match`: a `shared_attempt_word_identity.AttemptLanguageIdentityMatch`
    for this row's own `clip_id`, when the caller has already built one
    (D-235P). `proposition_candidate_ids_by_attempt_id` /
    `proposition_slot_evidence_by_id`: the SAME lookup tables D-235P's own
    `exact_proposition_candidate_ids_for_match` and a real `Proposition
    Candidate.editorial_slot_evidence` projection already provide.

    `heuristic_identity_available`: `True` only when the caller
    positively confirms a HEURISTIC_OVERLAP bridge existed for this
    clip_id (diagnostic labelling only -- never upgrades anything).
    """
    clip_id = str(row.get("clip_id") or "")
    reason_codes: list = []

    materiality = assess_lost_semantic_atom_materiality(
        row, critical_claim_conflict=critical_claim_conflict,
        recording_process_evidence=recording_process_evidence,
    )

    proposition_candidate_ids_by_attempt_id = proposition_candidate_ids_by_attempt_id or {}
    proposition_slot_evidence_by_id = proposition_slot_evidence_by_id or {}

    exact_identity_available = bool(
        exact_match is not None and exact_match.relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES
    )
    exact_language_attempt_ids: Tuple[str, ...] = exact_match.language_attempt_ids if exact_identity_available else ()
    exact_proposition_ids: Tuple[str, ...] = (
        exact_proposition_candidate_ids_for_match(exact_match, proposition_candidate_ids_by_attempt_id)
        if exact_identity_available else ()
    )

    ownership_ambiguous = False
    exact_slot_evidence: Optional[str] = None
    if exact_identity_available:
        exact_slot_evidence, ownership_ambiguous = _exact_slot_for_proposition_set(
            exact_proposition_ids, proposition_slot_evidence_by_id,
        )
        if ownership_ambiguous:
            reason_codes.append("multi_proposition_exact_set_unresolved_atom_ownership")

    if exact_identity_available:
        identity_mapping_status = IDENTITY_MAPPING_EXACT
    elif heuristic_identity_available:
        identity_mapping_status = IDENTITY_MAPPING_HEURISTIC_OVERLAP
    else:
        identity_mapping_status = IDENTITY_MAPPING_NONE

    requirement = assess_editorial_requirement_evidence(
        row,
        identity_mapping_status=identity_mapping_status,
        recording_process_status=recording_process_status,
        audience_delivery_status=audience_delivery_status,
        idea_coverage_status=idea_coverage_status,
        editorial_slot_evidence=exact_slot_evidence,
        slot_evidence_source=(IDENTITY_MAPPING_EXACT if exact_slot_evidence is not None else None),
        replacement_function_preserved=replacement_function_preserved,
        downstream_dependency_present=downstream_dependency_present,
    )

    redundancy_proven = bool(
        materiality.materiality_status == MATERIALITY_REDUNDANT_EQUIVALENT
        or requirement.editorial_requirement_status == REQUIREMENT_REDUNDANT_REQUIRED_FUNCTION_PRESERVED
        or replacement_function_preserved is True
    )

    # Step 6's own identity-sufficiency gate -- see module docstring.
    requirement_genuinely_clear = (
        requirement.editorial_requirement_status == REQUIREMENT_NOT_REQUIRED
        or (requirement.editorial_requirement_status == REQUIREMENT_INSUFFICIENT_EVIDENCE and exact_identity_available)
    )

    # ================= PRECEDENCE (order matters; first match wins) =================
    if materiality.materiality_status == MATERIALITY_MEANING_CRITICAL:
        reason_codes.append("meaning_critical_floor")
        final_status, blocking = MATERIALITY_MEANING_CRITICAL, RECOMMEND_BLOCK

    elif requirement.editorial_requirement_status == REQUIREMENT_REQUIRED:
        reason_codes.append("editorially_required_evidence")
        final_status, blocking = MATERIALITY_EDITORIALLY_REQUIRED, RECOMMEND_BLOCK

    elif (
        materiality.materiality_status == MATERIALITY_CONFLICTED
        or requirement.editorial_requirement_status == REQUIREMENT_CONFLICTED
        or (ownership_ambiguous and not redundancy_proven)
    ):
        reason_codes.append("conflicting_or_unresolved_evidence")
        final_status, blocking = MATERIALITY_CONFLICTED, RECOMMEND_ABSTAIN

    elif materiality.materiality_status == MATERIALITY_RETRY_OR_RECORDING_RESIDUE:
        reason_codes.append("retry_or_recording_residue")
        final_status, blocking = MATERIALITY_RETRY_OR_RECORDING_RESIDUE, RECOMMEND_DO_NOT_BLOCK

    elif redundancy_proven:
        reason_codes.append("redundant_equivalent_function_preserved")
        final_status, blocking = MATERIALITY_REDUNDANT_EQUIVALENT, RECOMMEND_DO_NOT_BLOCK

    elif materiality.materiality_status == MATERIALITY_NON_MATERIAL_REAL_CONTENT and requirement_genuinely_clear:
        reason_codes.append("non_material_real_content_requirement_cleared")
        final_status, blocking = MATERIALITY_NON_MATERIAL_REAL_CONTENT, RECOMMEND_DO_NOT_BLOCK

    else:
        reason_codes.append("insufficient_evidence_default")
        final_status, blocking = MATERIALITY_INSUFFICIENT_EVIDENCE, RECOMMEND_ABSTAIN

    return CompleteLostSemanticAtomMateriality(
        clip_id=clip_id,
        exact_identity_available=exact_identity_available,
        exact_language_attempt_ids=exact_language_attempt_ids,
        exact_proposition_candidate_ids=exact_proposition_ids,
        meaning_materiality_status=materiality.materiality_status,
        editorial_requirement_status=requirement.editorial_requirement_status,
        retry_or_process_status=materiality.retry_or_process_status,
        redundancy_status=(
            STATE_FOUND if redundancy_proven else materiality.redundancy_status
        ),
        final_materiality_status=final_status,
        blocking_recommendation=blocking,
        reason_codes=tuple(reason_codes),
        provenance=(SCHEMA_VERSION, "assess_complete_lost_semantic_atom_materiality"),
    )


def assess_many(
    rows: Sequence[Mapping],
    *,
    critical_claim_conflict_by_clip_id: Optional[Mapping[str, bool]] = None,
    recording_process_evidence_by_clip_id: Optional[Mapping[str, bool]] = None,
    exact_match_by_clip_id: Optional[Mapping[str, AttemptLanguageIdentityMatch]] = None,
    proposition_candidate_ids_by_attempt_id: Optional[Mapping[str, Tuple[str, ...]]] = None,
    proposition_slot_evidence_by_id: Optional[Mapping[str, str]] = None,
) -> tuple:
    """Batch convenience wrapper -- assesses each row independently, never
    letting one row's evidence leak into another's classification (same
    isolation discipline as D-235L's own `assess_many`)."""
    critical_claim_conflict_by_clip_id = critical_claim_conflict_by_clip_id or {}
    recording_process_evidence_by_clip_id = recording_process_evidence_by_clip_id or {}
    exact_match_by_clip_id = exact_match_by_clip_id or {}
    results = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        clip_id = str(row.get("clip_id") or "")
        results.append(assess_complete_lost_semantic_atom_materiality(
            row,
            critical_claim_conflict=critical_claim_conflict_by_clip_id.get(clip_id),
            recording_process_evidence=recording_process_evidence_by_clip_id.get(clip_id),
            exact_match=exact_match_by_clip_id.get(clip_id),
            proposition_candidate_ids_by_attempt_id=proposition_candidate_ids_by_attempt_id,
            proposition_slot_evidence_by_id=proposition_slot_evidence_by_id,
        ))
    return tuple(results)


def complete_lost_semantic_atom_materiality_diagnostics(
    result: CompleteLostSemanticAtomMateriality,
) -> dict:
    """Per-row compact diagnostics, exactly the directive's own required
    field list. No transcript dump."""
    return {
        "clip_id": result.clip_id,
        "exact_identity_status": "AVAILABLE" if result.exact_identity_available else "UNAVAILABLE",
        "language_attempt_ids": list(result.exact_language_attempt_ids),
        "proposition_candidate_ids": list(result.exact_proposition_candidate_ids),
        "meaning_materiality_status": result.meaning_materiality_status,
        "editorial_requirement_status": result.editorial_requirement_status,
        "retry_or_process_status": result.retry_or_process_status,
        "redundancy_status": result.redundancy_status,
        "final_materiality_status": result.final_materiality_status,
        "blocking_recommendation": result.blocking_recommendation,
        "reason_codes": list(result.reason_codes),
        "provenance": list(result.provenance),
    }


def complete_lost_semantic_atom_materiality_batch_diagnostics(
    results: Sequence[CompleteLostSemanticAtomMateriality],
) -> dict:
    """Batch, counts-only CI summary -- same pattern as every other D-19x/
    D-235x compact summary in this codebase."""
    results = tuple(results)
    status_counts: dict = {}
    blocking_counts: dict = {}
    for r in results:
        status_counts[r.final_materiality_status] = status_counts.get(r.final_materiality_status, 0) + 1
        blocking_counts[r.blocking_recommendation] = blocking_counts.get(r.blocking_recommendation, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(results),
        "final_materiality_status_counts": status_counts,
        "blocking_recommendation_counts": blocking_counts,
        "exact_identity_available_count": sum(1 for r in results if r.exact_identity_available),
    }
