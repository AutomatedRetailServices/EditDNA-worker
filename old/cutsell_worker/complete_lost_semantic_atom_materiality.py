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

## D-238: bounded lost-atom ownership as a SECOND, narrower identity-
## sufficiency source (offline, additive, docs/CUTSELL_DECISIONS.md D-238)

D-237M recovered a real target relationship that is genuinely, provably
owned by exactly one canonical proposition context (`exact_lost_atom_
ownership.py`'s own `EXACT_SINGLETON_OWNERSHIP`) but whose D-235P
`relationship_status` is `RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_
RECONSTRUCTED` -- correctly NON-authoritative for FULL ATTEMPT IDENTITY
(a 9-word fragment is not "the same delivery" as a 221-word attempt).
Before D-238, such a row's `exact_identity_available` was `False`, so
step 6's identity-sufficiency gate could never fire even when D-235L
independently found `NON_MATERIAL_REAL_CONTENT` -- the row abstained
(`INSUFFICIENT_EVIDENCE`) purely for lack of a full-attempt-identity
proof it structurally could never have (see D-237K's own forensic).

`lost_atom_ownership` (new, optional parameter below) supplies a SECOND,
narrower and DISJOINT identity-sufficiency source: a real `exact_lost_
atom_ownership.ExactLostAtomOwnership` for this row's own `clip_id`. Its
own boolean, `exact_ownership_available` (new field on `Complete
LostSemanticAtomMateriality`, computed as `lost_atom_ownership is not
None and lost_atom_ownership.is_exact_singleton`), extends the step-6
identity-sufficiency gate's own `OR` condition (`exact_identity_available
OR exact_ownership_available`) AND (D-239I Seam A, below) the slot-
evidence lookup `identity_mapping_status` feeds `assess_editorial_
requirement_evidence` -- it participates in NOTHING else. In particular:

  - it NEVER changes step order. Meaning-critical (step 1), editorial-
    required (step 2), conflicted/unresolved (step 3), retry/process
    (step 4), and redundant (step 5) are all evaluated FIRST, using
    D-235L/D-235M's own completely independent evidence -- a lost atom
    with `EXACT_SINGLETON_OWNERSHIP` but `MEANING_CRITICAL`/`REQUIRED`/
    `CONFLICTED` evidence elsewhere is BLOCKed/ABSTAINed exactly as it
    would be with no ownership at all (this module's own counterexample
    tests 1-6 prove this directly);
  - the size of the containing `LanguageAttempt` (9 of 221 words, or 221
    of 252 source words, in the real D-237M shape) is NEVER read here --
    `exact_ownership_available` is a bare boolean off `is_exact_
    singleton`, structurally blind to word counts;
  - `exact_identity_available` itself (D-235P's own full-attempt-
    identity flag) is completely UNCHANGED by this addition -- it is
    still computed exactly as before, from `exact_match` alone, and
    `lost_atom_ownership` never substitutes for it anywhere else in this
    module (`exact_language_attempt_ids`/`exact_proposition_candidate_
    ids` and the multi-proposition ambiguity check still read `exact_
    match`/`exact_identity_available` only).

`lost_atom_ownership_status` (new field) reports the raw ownership
status for diagnostics (`None` when no ownership object was supplied),
independent of whether it was ever consulted by the gate.

## D-239I Seam A: ownership unlocks the SAME existing slot-evidence lookup
## (offline, additive, docs/CUTSELL_DECISIONS.md D-239I)

D-239H's own forensic (docs/CUTSELL_DECISIONS.md D-239H) proved
`proposition_slot_evidence_by_id` already contains the owned
`PropositionCandidate`'s own slot value and already reaches this exact
function -- the ONLY gap was that `_exact_slot_for_proposition_set` was
called exclusively inside `if exact_identity_available:`, never for the
ownership-only case. Seam A closes exactly that gap and NOTHING more:
when `exact_identity_available` is `False` but `exact_ownership_
available` is `True`, this function now ALSO calls `_exact_slot_for_
proposition_set` -- the SAME unmodified function, doing the SAME lookup
against the SAME `proposition_slot_evidence_by_id` map -- using D-238's
own `lost_atom_ownership.proposition_candidate_ids` (by construction of
`is_exact_singleton`, always exactly one id) in place of D-235P's
`exact_proposition_ids`. `identity_mapping_status` is correspondingly
upgraded to `IDENTITY_MAPPING_EXACT` whenever EITHER source is exact
(describing the QUALITY of the identity mapping, never whether a slot
value was actually found for it -- an owned proposition carrying no
story-function slot still correctly reports `slot_evidence_status:
NOT_FOUND` downstream in D-235M, never a fabricated one). Never a new
slot interpretation (`_STORY_FUNCTION_SLOTS`/`_exact_slot_for_
proposition_set` are byte-for-byte unmodified), never a guess (a
singleton-owned set has exactly one id by D-238's own 9-condition gate,
so the multi-id ambiguity branch inside `_exact_slot_for_proposition_
set` is structurally unreachable through this path), never a change to
step order or to any OTHER evidence dimension (meaning-critical,
critical-claim-conflict, retry/process, and redundancy are all seams
D-239I addresses separately, in `final_story_coherence_validation.py`'s
own orchestration layer -- see D-239I's own decision-log entry).

## D-239L: atom-granular refinement of the ownership-only REQUIRED bridge
## (offline, additive, docs/CUTSELL_DECISIONS.md D-239K/D-239L)

D-239K's own forensic proved that Seam A's `slot_is_exact` promotion,
when reached through `exact_ownership_available` alone (never through
`exact_identity_available`), rests on evidence that is PROPOSITION-level
(a `PropositionCandidate`'s own `editorial_slot_evidence` is minted once
per whole `LanguageAttempt`, from that attempt's own ordinal position
among source siblings and its own whole-text claim signature -- see
`language_proposition_relation.py::_slot_evidence`) and NEVER proves
that the lost atom's own specific word span carries the required
function. Before D-239I this never mattered: the only identity source
that could ever make `slot_is_exact` true was `exact_identity_
available` (D-235P's own full-attempt identity), under which the clip
being evaluated effectively WAS the whole attempt -- "the proposition is
required" and "this exact realization must survive" were the same
claim by construction. D-238/D-239I's ownership bridge introduced a
narrower, disjoint identity source (`exact_ownership_available`) whose
own proof is WEAKER on this exact dimension: it proves only "these N
words belong, unambiguously, to one attempt that owns one proposition"
(word-SET containment, D-238's own scope), never "this attempt's
required realization IS this N-word span" (function/realization
equivalence) -- D-238's own module docstring says this explicitly.

This section closes exactly that gap, for the ownership-only path only:

  - `exact_identity_available` is TRUE (D-235P full-attempt identity):
    **completely unchanged.** Every branch below is gated on `not
    exact_identity_available` -- a row with full-attempt identity never
    reaches this refinement, regardless of whether ownership ALSO
    happens to resolve exactly for the same clip. This is the literal
    "D-235P full-attempt exact identity behavior must remain unchanged"
    requirement.
  - `exact_identity_available` is FALSE and `exact_ownership_available`
    is TRUE, and the ONLY reason `assess_editorial_requirement_evidence`
    reached `REQUIRED` was the inherited proposition-level slot evidence
    (detected the same way this function already reports it: a
    `"exact_story_function_slot:..."` reason code in `requirement.
    reason_codes`, with NEITHER `idea_coverage_status is True` NOR
    `downstream_dependency_present is True` -- both already-atom/idea-
    scoped, independent signals this refinement never touches): the
    verdict is trusted ONLY when the SAME exact, already-computed,
    target-clip_id-keyed atom-level evidence D-239I's own Seam C
    already threads into this same call
    (`recording_process_status`/`audience_delivery_status`, sourced
    from `editorial_moment_sequence_integration.p1_moment_role_and_
    audience_status_by_clip_id_for`, itself gated to `CONFIDENCE_
    SUPPORTED` and non-`UNCERTAIN` roles -- never a new lookup, never a
    new classifier) actually CONFIRMS it: `recording_process_status`
    resolved (a confirmed, non-process-shaped role for the TARGET's own
    clip_id -- a process-shaped role would already have been caught by
    `assess_editorial_requirement_evidence`'s own PRE-EXISTING retry/
    process firewall, unchanged) AND `audience_delivery_status` resolved
    to `AUDIENCE_DELIVERY_SUPPORTED`/`AUDIENCE_DELIVERY_PARTIAL` (never
    `UNCERTAIN`, never absent -- "proves required audience-delivery
    function", not merely "not disproven"). When BOTH hold, `REQUIRED`
    stands untouched -- the target's own exact evidence, not the
    proposition's, now corroborates it.
  - When that exact target-level corroboration is absent (the target's
    own P1 role never resolved at all, or resolved but its own
    audience-delivery status is `UNCERTAIN`/absent): the row is
    downgraded from `REQUIREMENT_REQUIRED` to `REQUIREMENT_INSUFFICIENT_
    EVIDENCE` -- an EXISTING vocabulary value, never a new one, and
    never `REQUIREMENT_NOT_REQUIRED` (this task's own explicit "NOT:
    NOT_REQUIRED" instruction). `reason_codes` gains one new, honestly-
    named code (`ownership_only_slot_evidence_lacks_atom_level_target_
    corroboration`) rather than silently losing the original slot
    reason code, which is kept for audit.
  - This NEVER fires when `idea_coverage_status is True` or
    `downstream_dependency_present is True` also contributed to
    `REQUIRED` -- those are already atom/idea-scoped, independent
    evidence classes this refinement has no reason to distrust; only
    the SLOT-EVIDENCE-ONLY case is in scope, per this task's own "refine
    ONLY... ownership-based editorial-requirement bridge" instruction.
  - Meaning-critical (step 1) and the conflicted/CONFLICTED branch
    (step 3) of `assess_complete_lost_semantic_atom_materiality`'s own
    precedence chain are evaluated on the (possibly downgraded)
    `requirement` object exactly as before -- this section changes
    WHICH value `requirement.editorial_requirement_status` holds when
    the ownership-only slot path was the sole reason it became
    `REQUIRED`, never the precedence chain itself, never any other
    evidence dimension, never `assess_editorial_requirement_evidence`'s
    own general contract (called with the SAME arguments as before;
    its return value is post-processed here, in the ONE caller,
    never inside that function itself).
  - Diagnostics-only additions: `editorial_requirement_granularity`
    (`FULL_IDENTITY_PROPOSITION` / `ATOM_EXACT_P1` / `PROPOSITION_ONLY_
    INSUFFICIENT` / `AMBIGUOUS` / `MISSING` / `None`) and `editorial_
    requirement_target_evidence_source` (`FULL_ATTEMPT_IDENTITY` /
    `EXACT_P1_MOMENT` / `NONE` / `None`) -- both pure labels over
    already-computed booleans/values, never consulted by the precedence
    chain itself, added purely so a caller (D-239J/D-239M's own
    real-media reporting) can see WHY a row landed where it did.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence, Tuple

from .editorial_moment_sequence import (
    AUDIENCE_DELIVERY_PARTIAL,
    AUDIENCE_DELIVERY_SUPPORTED,
)
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
from .exact_lost_atom_ownership import ExactLostAtomOwnership

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
    # D-238: the bounded lost-atom ownership seam's own verdict for this
    # row, disjoint from and never overloading `exact_identity_available`
    # above -- see module docstring's "D-238" section.
    exact_ownership_available: bool = False
    lost_atom_ownership_status: Optional[str] = None
    # D-239L: diagnostics-only labels over already-computed values -- see
    # module docstring's own "D-239L" section. Never consulted by the
    # precedence chain; `None` whenever neither identity path applies.
    editorial_requirement_granularity: Optional[str] = None
    editorial_requirement_target_evidence_source: Optional[str] = None

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
            "exact_ownership_available": self.exact_ownership_available,
            "lost_atom_ownership_status": self.lost_atom_ownership_status,
            "editorial_requirement_granularity": self.editorial_requirement_granularity,
            "editorial_requirement_target_evidence_source": self.editorial_requirement_target_evidence_source,
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
    lost_atom_ownership: Optional[ExactLostAtomOwnership] = None,
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

    `lost_atom_ownership` (D-238): an optional, already-computed
    `exact_lost_atom_ownership.ExactLostAtomOwnership` for this row's own
    `clip_id`. Consulted at the step-6 identity-sufficiency gate (see
    module docstring's "D-238" section) AND, as of D-239I, at the SAME
    slot-evidence lookup `exact_identity_available` already feeds when
    `exact_identity_available` is `False` (see module docstring's "D-239I
    Seam A" section) -- never changes step order, never substitutes for
    `exact_match`/`exact_identity_available` anywhere ELSE in this
    function. `None` (the default, every pre-D-238 caller) reproduces
    byte-identical pre-D-238 behavior.
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

    # D-238's own bare boolean, needed here (ahead of its own "D-238"
    # section below) so Seam A can consult it for the SAME slot-evidence
    # lookup exact_identity_available already uses -- structurally blind
    # to the containing LanguageAttempt's own size, exactly as elsewhere.
    exact_ownership_available = bool(
        lost_atom_ownership is not None and lost_atom_ownership.is_exact_singleton
    )

    ownership_ambiguous = False
    exact_slot_evidence: Optional[str] = None
    if exact_identity_available:
        exact_slot_evidence, ownership_ambiguous = _exact_slot_for_proposition_set(
            exact_proposition_ids, proposition_slot_evidence_by_id,
        )
        if ownership_ambiguous:
            reason_codes.append("multi_proposition_exact_set_unresolved_atom_ownership")
    elif exact_ownership_available:
        # D-239I Seam A: D-238's own exact singleton ownership names EXACTLY
        # one proposition id (by construction of `is_exact_singleton` --
        # see exact_lost_atom_ownership.py's own condition 6). Feed that
        # SAME id, through the SAME existing, unmodified `_exact_slot_for_
        # proposition_set` lookup, into the SAME already-populated, already-
        # threaded `proposition_slot_evidence_by_id` map D-235X already
        # builds -- never a new slot interpretation, never a guess: this is
        # the identical function call `exact_identity_available` already
        # makes, reached through D-238's own narrower, disjoint identity
        # source instead of D-235P's full-attempt one. `ownership_ambiguous`
        # can never be set True by this branch (a singleton-owned set has
        # exactly one id, so `_exact_slot_for_proposition_set`'s own multi-
        # id ambiguity branch is structurally unreachable here) -- included
        # only for defensive symmetry with the `exact_identity_available`
        # branch above, never expected to fire.
        exact_slot_evidence, ownership_ambiguous = _exact_slot_for_proposition_set(
            lost_atom_ownership.proposition_candidate_ids, proposition_slot_evidence_by_id,
        )
        if ownership_ambiguous:
            reason_codes.append("multi_proposition_exact_set_unresolved_atom_ownership")

    if exact_identity_available or exact_ownership_available:
        # D-239I: ownership's own identity mapping is exact (a narrower,
        # disjoint, but equally fail-closed structural proof -- see
        # exact_lost_atom_ownership.py's own 9-condition gate), so it earns
        # the SAME `IDENTITY_MAPPING_EXACT` label D-235P's full-attempt
        # match already does. This describes the QUALITY of the identity
        # mapping, never whether a slot value was actually found for it --
        # `exact_slot_evidence` above independently stays `None` when the
        # owned proposition simply carries no story-function slot.
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

    # D-239L: refine the ownership-only REQUIRED bridge -- see module
    # docstring's own "D-239L" section. Gated STRICTLY on `not exact_
    # identity_available` so a row with full-attempt D-235P identity is
    # byte-identical regardless of what ownership independently resolves
    # to for the same clip (this task's own "D-235P full-attempt exact
    # identity behavior must remain unchanged" requirement).
    ownership_only_identity_path = bool((not exact_identity_available) and exact_ownership_available)
    slot_was_sole_required_reason = bool(
        requirement.editorial_requirement_status == REQUIREMENT_REQUIRED
        and any(str(code).partition(":")[0] == "exact_story_function_slot" for code in requirement.reason_codes)
        and idea_coverage_status is not True
        and downstream_dependency_present is not True
    )
    # Exact, already-computed, target-clip_id-keyed atom-level evidence
    # (D-239I Seam C's own `recording_process_status`/`audience_delivery_
    # status` for THIS row's own clip_id -- never re-derived, never a new
    # lookup). A resolved (non-`None`) `recording_process_status` at this
    # point already means "confirmed, non-process-shaped" -- a process-
    # shaped role would have already returned `NOT_REQUIRED` via `assess_
    # editorial_requirement_evidence`'s own PRE-EXISTING retry/process
    # firewall, unchanged by this task, before `requirement.editorial_
    # requirement_status` could ever be `REQUIRED` here.
    atom_level_role_resolved = recording_process_status is not None
    atom_level_audience_delivery_proven = audience_delivery_status in (
        AUDIENCE_DELIVERY_SUPPORTED, AUDIENCE_DELIVERY_PARTIAL,
    )
    atom_level_corroborated = bool(atom_level_role_resolved and atom_level_audience_delivery_proven)

    if ownership_only_identity_path and slot_was_sole_required_reason and not atom_level_corroborated:
        # Proposition-level slot evidence + EXACT_SINGLETON_OWNERSHIP is
        # NOT sufficient by itself for REQUIRED (this task's own "CRITICAL
        # RULE"). Fail closed to INSUFFICIENT_EVIDENCE -- an EXISTING
        # vocabulary value -- never NOT_REQUIRED (this task's own explicit
        # "NOT: NOT_REQUIRED" instruction). The original slot reason code
        # is kept (never dropped) alongside the new one, for audit.
        requirement = replace(
            requirement,
            editorial_requirement_status=REQUIREMENT_INSUFFICIENT_EVIDENCE,
            reason_codes=requirement.reason_codes + (
                "ownership_only_slot_evidence_lacks_atom_level_target_corroboration",
            ),
        )
        # This module's own `reason_codes` list (distinct from `requirement.
        # reason_codes` above) is what `CompleteLostSemanticAtomMateriality.
        # reason_codes` is actually built from below -- surface the SAME
        # honest code there too, so a caller reading the final result object
        # (rather than the intermediate `requirement`) still sees WHY.
        reason_codes.append("ownership_only_slot_evidence_lacks_atom_level_target_corroboration")

    # D-239L diagnostics-only labels (never consulted by the precedence
    # chain below) -- see module docstring's own "D-239L" section.
    if exact_identity_available:
        editorial_requirement_granularity = "FULL_IDENTITY_PROPOSITION"
        editorial_requirement_target_evidence_source = "FULL_ATTEMPT_IDENTITY"
    elif exact_ownership_available:
        editorial_requirement_target_evidence_source = "EXACT_P1_MOMENT" if atom_level_role_resolved else "NONE"
        if slot_was_sole_required_reason and not atom_level_corroborated:
            editorial_requirement_granularity = "PROPOSITION_ONLY_INSUFFICIENT"
        elif not atom_level_role_resolved:
            editorial_requirement_granularity = "MISSING"
        elif not atom_level_corroborated:
            editorial_requirement_granularity = "AMBIGUOUS"
        else:
            editorial_requirement_granularity = "ATOM_EXACT_P1"
    else:
        editorial_requirement_granularity = None
        editorial_requirement_target_evidence_source = None

    redundancy_proven = bool(
        materiality.materiality_status == MATERIALITY_REDUNDANT_EQUIVALENT
        or requirement.editorial_requirement_status == REQUIREMENT_REDUNDANT_REQUIRED_FUNCTION_PRESERVED
        or replacement_function_preserved is True
    )

    # D-238/D-239I: `exact_ownership_available` was already computed above
    # (needed there for Seam A's own slot-evidence lookup) -- never touches
    # `exact_identity_available`/`exact_language_attempt_ids`/`exact_
    # proposition_candidate_ids` above regardless. As of D-239I Seam A it
    # DOES feed `editorial_slot_evidence` (see that section above), but
    # ONLY through the SAME existing lookup `exact_identity_available`
    # already used -- never a new interpretation.

    # Step 6's own identity-sufficiency gate -- see module docstring.
    # D-238 extends this gate's own OR-condition with a SECOND, disjoint
    # sufficiency source (`exact_ownership_available`) -- never replacing
    # `exact_identity_available`, never widening any OTHER step.
    requirement_genuinely_clear = (
        requirement.editorial_requirement_status == REQUIREMENT_NOT_REQUIRED
        or (
            requirement.editorial_requirement_status == REQUIREMENT_INSUFFICIENT_EVIDENCE
            and (exact_identity_available or exact_ownership_available)
        )
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
        exact_ownership_available=exact_ownership_available,
        lost_atom_ownership_status=(
            lost_atom_ownership.ownership_status if lost_atom_ownership is not None else None
        ),
        editorial_requirement_granularity=editorial_requirement_granularity,
        editorial_requirement_target_evidence_source=editorial_requirement_target_evidence_source,
    )


def assess_many(
    rows: Sequence[Mapping],
    *,
    critical_claim_conflict_by_clip_id: Optional[Mapping[str, bool]] = None,
    recording_process_evidence_by_clip_id: Optional[Mapping[str, bool]] = None,
    exact_match_by_clip_id: Optional[Mapping[str, AttemptLanguageIdentityMatch]] = None,
    proposition_candidate_ids_by_attempt_id: Optional[Mapping[str, Tuple[str, ...]]] = None,
    proposition_slot_evidence_by_id: Optional[Mapping[str, str]] = None,
    lost_atom_ownership_by_clip_id: Optional[Mapping[str, ExactLostAtomOwnership]] = None,
) -> tuple:
    """Batch convenience wrapper -- assesses each row independently, never
    letting one row's evidence leak into another's classification (same
    isolation discipline as D-235L's own `assess_many`).

    `lost_atom_ownership_by_clip_id` (D-238): optional, mirrors `exact_
    match_by_clip_id`'s own per-clip lookup shape. `None`/absent-entry
    reproduces byte-identical pre-D-238 behavior for that row."""
    critical_claim_conflict_by_clip_id = critical_claim_conflict_by_clip_id or {}
    recording_process_evidence_by_clip_id = recording_process_evidence_by_clip_id or {}
    exact_match_by_clip_id = exact_match_by_clip_id or {}
    lost_atom_ownership_by_clip_id = lost_atom_ownership_by_clip_id or {}
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
            lost_atom_ownership=lost_atom_ownership_by_clip_id.get(clip_id),
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
        "exact_ownership_available": result.exact_ownership_available,
        "lost_atom_ownership_status": result.lost_atom_ownership_status,
        "editorial_requirement_granularity": result.editorial_requirement_granularity,
        "editorial_requirement_target_evidence_source": result.editorial_requirement_target_evidence_source,
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
