"""D-235N: EXACT PROPOSITION IDENTITY BRIDGE FORENSIC -- OFFLINE ONLY,
FORENSIC ONLY.

Post D-235M (docs/CUTSELL_DECISIONS.md): D-235M's own verdict was B,
naming one missing identity seam -- `PropositionCandidate.editorial_slot_
evidence` (the only structured story-FUNCTION signal in the codebase) has
no `clip_id` field, and the only bridge from a lost-atom row's own
`clip_id` to it is a "deterministic MAXIMUM-OVERLAP match", which D-235M's
own scope disqualified as authority. This task's job is to FORENSICALLY
TRACE each seam of that chain mechanically -- no new ID, no schema
change, no live wiring -- and determine precisely which seams are EXACT,
ONE_TO_MANY_EXACT, AMBIGUOUS, HEURISTIC_ONLY, or MISSING.

## The five seams, traced through the real, current, unmodified code

**Seam 1 -- `lost_semantic_atom.clip_id` -> canonical/selected clip:
EXACT.** `final_story_coherence_validation.py::_lost_semantic_atoms()`
copies `"clip_id": clip.clip_id` verbatim from the real, discarded
`DraftClip` -- no transform, no re-mint. `pipeline.py::_draft_clip` itself
sets `clip_id=take.clip_id` verbatim from the originating `CandidateTake`
(`pipeline.py` line ~768). This is a plain field copy at every hop -- not
computed by this module, a pure code-reading fact.

**Seam 2 -- canonical clip -> P1 `EditorialMoment`: EXACT.**
`editorial_moment_sequence_integration.py::build_editorial_moments_for_
source` sets `EditorialMoment.source_span_id = take.clip_id` verbatim
(the SAME field, no transform) for every take with a matching
`UnderstandingSpan`, over the FULL per-source candidate pool (including
discarded clips). `classify_seam2_clip_to_p1_moment()` below verifies
this mechanically against a real `EditorialMoment`.

**Seam 3 -- P1 `EditorialMoment.proposition_candidate_ids`: NEVER EXACT
(HEURISTIC_ONLY when populated, MISSING otherwise) -- the decisive
finding.** Traced precisely at `editorial_moment_sequence_integration.py`
lines ~453-460 and ~512:

    real_attempt = language_attempts_by_span_id.get(take.clip_id)
    if real_attempt is not None:
        attempt = real_attempt
        attempt_source_by_position[position] = LANGUAGE_EVIDENCE_CANONICAL
    else:
        attempt = _derive_language_attempt(take, span, d157_relation)
        attempt_source_by_position[position] = LANGUAGE_EVIDENCE_D157_FALLBACK
    ...
    proposition_candidate_ids=proposition_candidate_ids_by_attempt_id.get(attempt.attempt_id, ())

`language_attempts_by_span_id` is itself the output of `language_spine_
live_integration.py::language_attempts_by_span_id_for_source` -- that
module's OWN docstring states plainly: the Language Spine's independently
segmented `LanguageAttempt` boundaries "will not, in general, align
exactly with" P1's clip-keyed spans, and the bridge is "a deterministic
MAXIMUM-OVERLAP match" (never an identity equality). So even in the
`LANGUAGE_EVIDENCE_CANONICAL` case -- where `proposition_candidate_ids`
genuinely gets populated -- the very identification of WHICH real
`LanguageAttempt` corresponds to this clip_id is itself the product of
that overlap match. In the `LANGUAGE_EVIDENCE_D157_FALLBACK` case,
`attempt.attempt_id` is a locally-derived placeholder id that (per that
same call site's own comment) "never appears in a real
PropositionCandidate's attempt_ids", so the lookup is guaranteed empty.
**There is no third case where `proposition_candidate_ids` is populated
via anything other than the overlap match.** Confirmed further: this
entire construction path is gated behind `live_language_spine_
diagnostics_enabled()` (`pipeline.py` ~line 2480) -- an explicit, off-by-
default DIAGNOSTICS flag (D-199's own naming) -- so on the engine's
STANDARD/default run (flag off), `proposition_candidate_ids` is
unconditionally `()` for every moment, independent of the heuristic's own
quality.

Separately confirmed (independent proof the two ID spaces cannot coincide
by construction, not merely by observed behavior): `DraftClip.attempt_id`/
`CandidateTake.attempt_id` are minted by `attempt_reconstruction.py`'s own
`_merge_attempt` via `canonical_identity.mint_attempt_id(member_source_
span_ids)`, while `LanguageAttempt.attempt_id` (D-168) is minted by an
entirely separate function, `language_utterance_attempt.py::_attempt_id
(member_utterance_ids)` -- different module, different input space,
different hash. These are structurally distinct identifier spaces; only
a real-world hash collision could make them equal, never a design
guarantee.

**Seam 4 -- `proposition_candidate_id` -> `PropositionCandidate` object:
EXACT, once you have a real id.** `language_proposition_relation.py::
_proposition_id` is a deterministic mint (`source_asset_id` + claim-
signature hash + timing), and `build_proposition_candidates` produces
exactly ONE `PropositionCandidate` per `LanguageAttempt` (a strict 1:1,
never duplicated). A plain dict lookup by this id is unambiguous. This
seam was never the problem -- it only matters once Seam 3 has already
supplied a real id, which it cannot do exactly.

**Seam 5 -- `PropositionCandidate` -> `editorial_slot_evidence`: EXACT.**
`editorial_slot_evidence: str` is a direct, always-present dataclass
field -- no lookup, no ambiguity, once you have the object.

## One-to-many / atom-ownership finding (fixtures 8, 12)

`language_attempts_by_span_id.get(take.clip_id)` returns AT MOST ONE
`LanguageAttempt` (a `.get()` on a dict, never a set), and `build_
proposition_candidates` is a strict 1:1 `LanguageAttempt -> Proposition
Candidate` mapping. **Under the current architecture, one clip can never
resolve to more than one `proposition_candidate_id` at all** -- so the
"atom ownership ambiguity inside a multi-proposition clip" scenario the
directive asks to investigate (fixture 12) is honestly MOOT for this
codebase as it exists today: the one-to-many case cannot structurally
occur, not because it was solved, but because the current single-attempt-
per-clip overlap match never produces more than one candidate to begin
with. `EditorialMoment.proposition_candidate_ids` being a `Tuple[str,
...]` (plural-shaped) is schema headroom for a FUTURE multi-attempt
widening (`PropositionCandidate.attempt_ids` is itself already a tuple,
per that dataclass's own docstring, "so a future phase MAY widen this to
a multi-attempt proposition without a schema break") -- not evidence that
multiple propositions are produced today.

## What this module is NOT (binding, restated from this task's own scope)

- No new canonical ID, no schema migration, no new lost-atom field. Every
  function below CLASSIFIES already-existing objects/values the caller
  supplies; it mints nothing.
- No live wiring. Nothing here is imported by `pipeline.py`, `universal_
  clean_cut.py`, `final_story_coherence_validation.py`,
  `lost_semantic_atom_materiality.py`, or `lost_atom_editorial_
  requirement_evidence.py`.
- No promotion of the overlap match. `classify_seam3_moment_to_
  proposition_ids()` NEVER returns `EXACT` or `ONE_TO_MANY_EXACT` for the
  canonical case -- by design, it can only return `HEURISTIC_ONLY` or
  `MISSING`, mechanically enforcing D-235M's own "no fuzzy matching as
  authority" instruction rather than merely restating it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .language_spine_live_integration import (
    LANGUAGE_EVIDENCE_CANONICAL,
    LANGUAGE_EVIDENCE_D157_FALLBACK,
)

SCHEMA_VERSION = "cutsell.lost_atom_proposition_identity_forensic.v1"

# The directive's own five-value seam-classification vocabulary.
SEAM_EXACT = "EXACT"
SEAM_ONE_TO_MANY_EXACT = "ONE_TO_MANY_EXACT"
SEAM_AMBIGUOUS = "AMBIGUOUS"
SEAM_HEURISTIC_ONLY = "HEURISTIC_ONLY"
SEAM_MISSING = "MISSING"

_VALID_SEAM_STATUS = frozenset({
    SEAM_EXACT, SEAM_ONE_TO_MANY_EXACT, SEAM_AMBIGUOUS, SEAM_HEURISTIC_ONLY, SEAM_MISSING,
})


def classify_seam2_clip_to_p1_moment(
    *, clip_id: str, editorial_moment_source_span_id: Optional[str],
) -> str:
    """EXACT iff the P1 moment's own `source_span_id` equals this clip_id
    verbatim (the SAME field, no transform -- see module docstring, Seam
    2). `MISSING` when no moment was found for this clip_id at all (an
    unresolved take, or no matching `UnderstandingSpan`)."""
    if editorial_moment_source_span_id is None:
        return SEAM_MISSING
    return SEAM_EXACT if editorial_moment_source_span_id == clip_id else SEAM_AMBIGUOUS


def classify_seam3_moment_to_proposition_ids(
    *, language_evidence_source: Optional[str], proposition_candidate_ids: Sequence[str],
) -> str:
    """The decisive seam. Reuses ONLY the already-computed, already-
    exposed per-moment provenance (`LANGUAGE_EVIDENCE_CANONICAL`/
    `LANGUAGE_EVIDENCE_D157_FALLBACK`, D-199's own `moment_language_
    evidence_source`) -- never a new detector, never a re-derivation of
    the overlap match itself. By design this function can NEVER return
    `EXACT` or `ONE_TO_MANY_EXACT` for the canonical source, regardless of
    how many ids are present -- the canonical source's own identity is
    itself established via a maximum-overlap heuristic (see module
    docstring), so any ids it supplies are `HEURISTIC_ONLY`, never
    upgraded. `MISSING` under the fallback source (guaranteed empty per
    that call site's own code) or when genuinely empty under either
    source."""
    ids = tuple(proposition_candidate_ids or ())
    if language_evidence_source == LANGUAGE_EVIDENCE_D157_FALLBACK:
        return SEAM_MISSING
    if language_evidence_source == LANGUAGE_EVIDENCE_CANONICAL and ids:
        return SEAM_HEURISTIC_ONLY
    return SEAM_MISSING


def classify_seam4_proposition_id_to_candidate(
    *, proposition_candidate_id: Optional[str], proposition_candidate_lookup: Mapping[str, object],
) -> str:
    """EXACT plain-dict lookup once a real id is in hand -- `build_
    proposition_candidates` mints exactly one `PropositionCandidate` per
    id, never duplicated (Seam 4 was never the weak link; it only matters
    once Seam 3 has already supplied a real id, which it structurally
    cannot do exactly)."""
    if not proposition_candidate_id:
        return SEAM_MISSING
    return SEAM_EXACT if proposition_candidate_id in proposition_candidate_lookup else SEAM_MISSING


def classify_seam5_candidate_to_slot_evidence(candidate: object) -> str:
    """EXACT -- `editorial_slot_evidence` is a direct, always-present
    dataclass field on `PropositionCandidate`, no lookup, no ambiguity,
    once the object itself is in hand."""
    if candidate is None:
        return SEAM_MISSING
    return SEAM_EXACT if getattr(candidate, "editorial_slot_evidence", None) is not None else SEAM_MISSING


@dataclass(frozen=True)
class PropositionIdentityBridgeForensic:
    clip_id: str
    seam2_clip_to_p1_moment: str
    seam3_moment_to_proposition_ids: str
    seam4_proposition_id_to_candidate: str
    seam5_candidate_to_slot_evidence: str
    end_to_end_status: str
    reason_codes: tuple
    provenance: tuple

    def __post_init__(self) -> None:
        for field_value in (
            self.seam2_clip_to_p1_moment, self.seam3_moment_to_proposition_ids,
            self.seam4_proposition_id_to_candidate, self.seam5_candidate_to_slot_evidence,
            self.end_to_end_status,
        ):
            if field_value not in _VALID_SEAM_STATUS:
                raise ValueError(f"invalid seam status: {field_value!r}")

    def as_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "seam2_clip_to_p1_moment": self.seam2_clip_to_p1_moment,
            "seam3_moment_to_proposition_ids": self.seam3_moment_to_proposition_ids,
            "seam4_proposition_id_to_candidate": self.seam4_proposition_id_to_candidate,
            "seam5_candidate_to_slot_evidence": self.seam5_candidate_to_slot_evidence,
            "end_to_end_status": self.end_to_end_status,
            "reason_codes": list(self.reason_codes),
            "provenance": list(self.provenance),
        }


def _weakest_seam(*seams: str) -> str:
    """The end-to-end chain's authority is exactly as strong as its
    weakest link -- `MISSING` dominates, then `AMBIGUOUS`, then
    `HEURISTIC_ONLY`; only when every seam is `EXACT`/`ONE_TO_MANY_EXACT`
    does the chain itself qualify as exact."""
    if SEAM_MISSING in seams:
        return SEAM_MISSING
    if SEAM_AMBIGUOUS in seams:
        return SEAM_AMBIGUOUS
    if SEAM_HEURISTIC_ONLY in seams:
        return SEAM_HEURISTIC_ONLY
    if SEAM_ONE_TO_MANY_EXACT in seams:
        return SEAM_ONE_TO_MANY_EXACT
    return SEAM_EXACT


def classify_proposition_identity_bridge(
    *,
    clip_id: str,
    editorial_moment_source_span_id: Optional[str],
    language_evidence_source: Optional[str],
    proposition_candidate_ids: Sequence[str] = (),
    proposition_candidate_lookup: Optional[Mapping[str, object]] = None,
    proposition_candidate: Optional[object] = None,
) -> PropositionIdentityBridgeForensic:
    """The one D-235N entry point -- classifies Seams 2-5 end to end for
    ONE clip_id, from already-computed, caller-supplied real objects/
    values (Seam 1, the lost-atom-row-to-clip-id copy, is a static code
    fact established in the module docstring, not a runtime input here).
    Mints nothing, discovers nothing itself, never invoked live."""
    reason_codes: list = []
    seam2 = classify_seam2_clip_to_p1_moment(clip_id=clip_id, editorial_moment_source_span_id=editorial_moment_source_span_id)
    ids = tuple(proposition_candidate_ids or ())
    seam3 = classify_seam3_moment_to_proposition_ids(language_evidence_source=language_evidence_source, proposition_candidate_ids=ids)
    if seam3 == SEAM_HEURISTIC_ONLY:
        reason_codes.append("proposition_ids_sourced_via_max_overlap_match_never_authoritative")
    if language_evidence_source == LANGUAGE_EVIDENCE_D157_FALLBACK:
        reason_codes.append("d157_fallback_attempt_id_never_appears_in_real_proposition_attempt_ids")

    proposition_candidate_id = ids[0] if ids else None
    seam4 = classify_seam4_proposition_id_to_candidate(
        proposition_candidate_id=proposition_candidate_id,
        proposition_candidate_lookup=proposition_candidate_lookup or {},
    )
    seam5 = classify_seam5_candidate_to_slot_evidence(proposition_candidate)

    end_to_end = _weakest_seam(seam2, seam3, seam4, seam5)
    if end_to_end != SEAM_EXACT:
        reason_codes.append("end_to_end_bridge_not_authoritative_for_required_determination")

    return PropositionIdentityBridgeForensic(
        clip_id=clip_id, seam2_clip_to_p1_moment=seam2, seam3_moment_to_proposition_ids=seam3,
        seam4_proposition_id_to_candidate=seam4, seam5_candidate_to_slot_evidence=seam5,
        end_to_end_status=end_to_end, reason_codes=tuple(reason_codes),
        provenance=(SCHEMA_VERSION, "classify_proposition_identity_bridge"),
    )
