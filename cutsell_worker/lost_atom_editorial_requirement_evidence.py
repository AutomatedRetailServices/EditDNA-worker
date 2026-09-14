"""D-235M: Lost Semantic Atom EDITORIAL-REQUIREMENT EVIDENCE FOUNDATION --
OFFLINE ONLY.

Post D-235L (docs/CUTSELL_DECISIONS.md): D-235L's own verdict was B --
partially proven, with one honestly-documented gap: `EDITORIALLY_REQUIRED`
(content that is not meaning-critical at the claim level but may still be
necessary for story completeness -- a setup, a consequence, a causal
bridge, a required CTA, a required conclusion) had no reachable path from
existing evidence. This module is the AUDIT and, where a real bridge
exists, the FOUNDATION for that missing dimension -- never a live Freeze
authority, never a second story engine.

## What this module is NOT (binding, restated from this task's own scope)

- No Freeze/repair/resolver authority change. `assess_editorial_
  requirement_evidence()` never reads or writes `freeze_blocked`,
  `lost_atom.blocking`, `FinalEditReviewer` findings, or `RepairLoop`
  attempts, and is not called from any of those modules.
- No second story engine. Every signal this module can consume is read
  from an ALREADY-EXISTING structured object this task audited (P1
  `EditorialMoment` roles/`audience_delivery_status`, P2
  `WholeVideoEditorialRegion` aggregates, the coverage ledger's own
  family-level `missing_idea_coverage`, causal-order dependency evidence)
  -- never a new whole-video scorer, never a keyword classifier.
- No fabricated identity. D-235J already established that a
  `_lost_semantic_atoms()` row retains no `atom_id`/`source_span_id`/
  `source_proposition_id`/`semantic_role`/`required_or_optional`. This
  module does not invent them. It audits, and where useful, DOCUMENTS
  exactly which existing structured objects the row's own real `clip_id`
  CAN and CANNOT deterministically resolve to.

## THE IDENTITY AUDIT (the core finding of this task)

Traced end to end through the real, current code (no modification):

1. **`clip_id` -> P1 `EditorialMoment` is EXACT.** `editorial_moment_
   sequence_integration.py`'s own `build_editorial_moments_for_source`
   sets `EditorialMomentUnderstanding.attempt_id = take.clip_id` and
   `EditorialMoment.source_span_id = take.clip_id` verbatim ("reuse the
   REAL existing canonical clip identity -- no new id minted", per that
   module's own comment) -- built over the FULL per-source candidate pool
   (`takes_for_source: Iterable[CandidateTake]`), not only the selected
   subset, so a discarded clip genuinely has its own `EditorialMoment`
   when a matching `UnderstandingSpan` exists. This module reuses this
   bridge DIRECTLY and EXACTLY for `MOMENT_ROLE_*` (recording-process/
   retry/false-start/abandoned-attempt/post-take-reset shape -- the SAME
   evidence class D-235L's own `recording_process_evidence` parameter
   already consumes) and `audience_delivery_status`
   (SUPPORTED/PARTIAL/NOT_SUPPORTED/UNCERTAIN).
2. **`clip_id` -> P2 `WholeVideoEditorialRegion` is EXACT for
   `moment_ids`, but only INHERITED (not independently exact) for
   `proposition_candidate_ids`.** `WholeVideoEditorialRegion.moment_ids`
   is built directly from the same clip_id-exact P1 moments (item 1
   above) via `EditorialLocalGroup` membership -- exact. Its
   `proposition_candidate_ids` field, however, is populated from the SAME
   overlap bridge named in item 3 below, so it carries the identical
   caveat.
3. **`clip_id` -> `PropositionCandidate.editorial_slot_evidence` (the
   ONLY structured object in the codebase carrying a story-FUNCTION/slot
   signal -- `SLOT_HOOK`/`SLOT_SETUP`/`SLOT_PROBLEM`/`SLOT_FEATURE`/
   `SLOT_PROOF`/`SLOT_CONCLUSION`/`SLOT_CTA`/`SLOT_OTHER`, distinct from
   meaning-criticality) has NO EXACT IDENTITY BRIDGE.** `PropositionCandidate`
   (`language_proposition_relation.py`, D-169) has no `clip_id` field at
   all -- it is keyed by `source_asset_id` + `attempt_ids` (Language
   Spine's OWN, INDEPENDENT word-timing-gap + structural-boundary
   segmentation, D-166/D-168) which does NOT, in general, align with
   `take.clip_id`'s own, separately-segmented boundaries
   (`attempt_reconstruction.py`'s semantic merge). `language_spine_live_
   integration.py`'s own module docstring says this explicitly: "will
   not, in general, align exactly with those spans" -- and the ONLY
   bridge that exists, `language_attempts_by_span_id_for_source`, is a
   "deterministic MAXIMUM-OVERLAP match", not an identity equality. This
   task's own directive is explicit: "No fuzzy matching as authority" and
   "chronological proximity is NOT proof of editorial requirement" -- a
   maximum-overlap time-based join is exactly that kind of proximity
   evidence, not identity. **This is the one identity seam this task
   names as still missing** (see D-235M's own verdict).
4. **Family-level coverage (`missing_idea_coverage`, D-235L's own
   `critical_claim_conflict` concept) is real and reusable, but IDEA-
   scoped, not ATOM-scoped** -- it tells us whether a whole retry
   family/idea vanished, never whether one specific atom inside an
   otherwise-complete idea's own DISCARDED alternate carried a distinct,
   required function. This module accepts it as an optional, honestly-
   labelled corroborating signal, never a proxy for atom-level
   uniqueness.
5. **Downstream causal dependency** (`causal_order_validator.py`'s own
   `find_causal_order_breaks`, keyed by real `clip_id` pairs
   `required_clip_id`/`dependent_clip_id`) is a genuine, exact,
   clip_id-based dependency signal -- but it is a VALIDATION check (fires
   only when an already-FROZEN order is broken) with an optional,
   provider-backed `CausalOrderArbiter`, not a general "does anything
   depend on this DISCARDED clip" query this module can invoke itself
   without a provider call. Accepted here only as an optional,
   caller-supplied boolean the caller already established from real
   evidence -- never computed or guessed by this module.

## Critical/retry firewalls (binding, restated from the directive)

- **Criticality firewall**: this module never reads or overrides a
  MEANING_CRITICAL verdict from `lost_semantic_atom_materiality.py` --
  it answers only the editorial-requirement dimension, and the intended
  calling convention is to consult it only once that module's own
  critical-safety floor has NOT already fired.
- **Retry/process firewall**: a clip whose EXACT P1 role is
  recording-process-shaped (`RECORDING_PROCESS`/`FALSE_START`/
  `ABANDONED_ATTEMPT`/`RETRY`/`CORRECTION`/`POST_TAKE_RESET`/
  `PRE_TAKE_SETUP`/`BREAKING_CHARACTER` -- the SAME set `whole_video_
  editorial_reasoning.py`'s own `_PROCESS_ROLES` already defines, mirrored
  here verbatim from the public `MOMENT_ROLE_*` constants rather than
  importing a private name) is NEVER promoted to `REQUIRED`, regardless
  of any other signal -- it occupying a unique time span is never, by
  itself, proof of editorial necessity.
- **Chronology firewall**: this module accepts no raw timing/ordering
  parameter at all -- only already-resolved structured signals. Bare
  chronological adjacency is structurally unreachable as an input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .editorial_moment_sequence import (
    MOMENT_ROLE_ABANDONED_ATTEMPT,
    MOMENT_ROLE_BREAKING_CHARACTER,
    MOMENT_ROLE_CORRECTION,
    MOMENT_ROLE_FALSE_START,
    MOMENT_ROLE_POST_TAKE_RESET,
    MOMENT_ROLE_PRE_TAKE_SETUP,
    MOMENT_ROLE_RECORDING_PROCESS,
    MOMENT_ROLE_RETRY,
    AUDIENCE_DELIVERY_NOT_SUPPORTED,
)
from .language_proposition_relation import (
    SLOT_CONCLUSION,
    SLOT_CTA,
    SLOT_HOOK,
    SLOT_OTHER,
)

SCHEMA_VERSION = "cutsell.lost_atom_editorial_requirement_evidence.v1"

STATE_FOUND = "FOUND"
STATE_NOT_FOUND = "NOT_FOUND"
STATE_UNKNOWN = "UNKNOWN"

# Editorial-requirement vocabulary -- exactly the directive's own five
# bounded states.
REQUIREMENT_REQUIRED = "REQUIRED"
REQUIREMENT_NOT_REQUIRED = "NOT_REQUIRED"
REQUIREMENT_REDUNDANT_REQUIRED_FUNCTION_PRESERVED = "REDUNDANT_REQUIRED_FUNCTION_PRESERVED"
REQUIREMENT_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
REQUIREMENT_CONFLICTED = "CONFLICTED"

_VALID_REQUIREMENT_STATUS = frozenset({
    REQUIREMENT_REQUIRED, REQUIREMENT_NOT_REQUIRED,
    REQUIREMENT_REDUNDANT_REQUIRED_FUNCTION_PRESERVED,
    REQUIREMENT_INSUFFICIENT_EVIDENCE, REQUIREMENT_CONFLICTED,
})

# Mirrors whole_video_editorial_reasoning.py's own private `_PROCESS_ROLES`
# verbatim, built from the SAME public MOMENT_ROLE_* constants -- never a
# new role invented, never importing a private cross-module name.
_PROCESS_SHAPED_ROLES = frozenset({
    MOMENT_ROLE_RECORDING_PROCESS, MOMENT_ROLE_FALSE_START, MOMENT_ROLE_ABANDONED_ATTEMPT,
    MOMENT_ROLE_RETRY, MOMENT_ROLE_CORRECTION, MOMENT_ROLE_POST_TAKE_RESET,
    MOMENT_ROLE_BREAKING_CHARACTER, MOMENT_ROLE_PRE_TAKE_SETUP,
})

# Only these editorial_slot_evidence values are treated as carrying a
# distinct story FUNCTION (never SLOT_OTHER, which carries no function
# signal at all by construction).
_STORY_FUNCTION_SLOTS = frozenset({SLOT_HOOK, SLOT_CTA, SLOT_CONCLUSION})

# Identity-mapping-quality vocabulary the caller reports about its own
# supplied evidence -- never inferred by this module.
IDENTITY_MAPPING_EXACT = "EXACT"
IDENTITY_MAPPING_HEURISTIC_OVERLAP = "HEURISTIC_OVERLAP"
IDENTITY_MAPPING_AMBIGUOUS = "AMBIGUOUS"
IDENTITY_MAPPING_NONE = "NONE"


@dataclass(frozen=True)
class LostAtomEditorialRequirementEvidence:
    clip_id: str
    coverage_status: str
    story_role_status: str
    slot_evidence_status: str
    unique_story_function_status: str
    replacement_coverage_status: str
    downstream_dependency_status: str
    editorial_requirement_status: str
    reason_codes: tuple
    provenance: tuple

    def __post_init__(self) -> None:
        if self.editorial_requirement_status not in _VALID_REQUIREMENT_STATUS:
            raise ValueError(f"invalid editorial_requirement_status: {self.editorial_requirement_status!r}")

    def as_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "coverage_status": self.coverage_status,
            "story_role_status": self.story_role_status,
            "slot_evidence_status": self.slot_evidence_status,
            "unique_story_function_status": self.unique_story_function_status,
            "replacement_coverage_status": self.replacement_coverage_status,
            "downstream_dependency_status": self.downstream_dependency_status,
            "editorial_requirement_status": self.editorial_requirement_status,
            "reason_codes": list(self.reason_codes),
            "provenance": list(self.provenance),
        }


def assess_editorial_requirement_evidence(
    row: Mapping,
    *,
    identity_mapping_status: str = IDENTITY_MAPPING_NONE,
    recording_process_status: Optional[str] = None,
    audience_delivery_status: Optional[str] = None,
    idea_coverage_status: Optional[bool] = None,
    editorial_slot_evidence: Optional[str] = None,
    slot_evidence_source: Optional[str] = None,
    replacement_function_preserved: Optional[bool] = None,
    downstream_dependency_present: Optional[bool] = None,
) -> LostAtomEditorialRequirementEvidence:
    """The one D-235M entry point. Pure function of one already-computed
    `_lost_semantic_atoms()` row plus a set of OPTIONAL, explicitly
    caller-supplied structured signals -- this module discovers none of
    them itself (no P1/P2 construction call, no proposition-candidate
    lookup, no causal-order arbiter invocation).

    `identity_mapping_status` (default `NONE`, meaning "the caller
    supplied nothing"): the caller's own honest report of how much of the
    identity bridge it actually resolved for this clip_id --
    `EXACT` (e.g. a real P1 `EditorialMoment` was found for this clip_id),
    `HEURISTIC_OVERLAP` (only an overlap-matched Language Spine link was
    available), `AMBIGUOUS` (multiple, disagreeing candidates), or `NONE`.
    This module never upgrades a `HEURISTIC_OVERLAP` source into decisive
    evidence for `REQUIRED` -- per the directive's own "no fuzzy matching
    as authority."

    `recording_process_status` / `audience_delivery_status`: the SAME
    exact, clip_id-keyed P1 `EditorialMoment` fields D-235L's own
    `recording_process_evidence` parameter already reuses.

    `idea_coverage_status`: `True` only when the caller has established
    (from the SAME family-level `missing_idea_coverage`/coverage-ledger
    evidence D-235L's own `critical_claim_conflict` concept reuses) that
    removing this atom leaves a required proposition/idea uncovered.

    `editorial_slot_evidence` / `slot_evidence_source`: a
    `PropositionCandidate.editorial_slot_evidence` value the caller
    already resolved, PLUS an honest label of how ('EXACT' or
    'HEURISTIC_OVERLAP') -- required whenever `editorial_slot_evidence`
    is supplied; missing/unlabelled is treated defensively as
    `HEURISTIC_OVERLAP` (fail-safe, never assumed exact).

    `replacement_function_preserved`: `True` only when the caller has
    established the SAME required function is already covered by kept
    content (distinct from literal-text redundancy, though this module
    also treats the row's own `content_loss_suppressed_by`/
    `preserving_realization_id` as the closest existing structural proxy
    when no dedicated function-level signal is supplied).

    `downstream_dependency_present`: `True` only when the caller has
    established real causal-order dependency evidence
    (`causal_order_validator.py`) naming this clip as a
    `required_clip_id` for some later dependent -- never guessed from
    chronology.
    """
    clip_id = str(row.get("clip_id") or "")
    reason_codes: list = []

    # --- 0. No identity mapping at all -- the honest default when the
    # caller supplied nothing (fixture 12). ---
    if identity_mapping_status == IDENTITY_MAPPING_NONE and all(
        v is None for v in (
            recording_process_status, audience_delivery_status, idea_coverage_status,
            editorial_slot_evidence, replacement_function_preserved, downstream_dependency_present,
        )
    ) and not row.get("content_loss_suppressed_by") and not row.get("preserving_realization_id"):
        reason_codes.append("no_identity_mapping_supplied")
        return LostAtomEditorialRequirementEvidence(
            clip_id=clip_id, coverage_status=STATE_UNKNOWN, story_role_status=STATE_UNKNOWN,
            slot_evidence_status=STATE_UNKNOWN, unique_story_function_status=STATE_UNKNOWN,
            replacement_coverage_status=STATE_UNKNOWN, downstream_dependency_status=STATE_UNKNOWN,
            editorial_requirement_status=REQUIREMENT_INSUFFICIENT_EVIDENCE,
            reason_codes=tuple(reason_codes), provenance=(SCHEMA_VERSION, "assess_editorial_requirement_evidence"),
        )

    # --- 1. Ambiguous mapping (fixture 13) -- the caller itself reports
    # it could not resolve one confident identity. ---
    if identity_mapping_status == IDENTITY_MAPPING_AMBIGUOUS:
        reason_codes.append("identity_mapping_ambiguous")
        return LostAtomEditorialRequirementEvidence(
            clip_id=clip_id, coverage_status=STATE_UNKNOWN, story_role_status=STATE_UNKNOWN,
            slot_evidence_status=STATE_UNKNOWN, unique_story_function_status=STATE_UNKNOWN,
            replacement_coverage_status=STATE_UNKNOWN, downstream_dependency_status=STATE_UNKNOWN,
            editorial_requirement_status=REQUIREMENT_CONFLICTED,
            reason_codes=tuple(reason_codes), provenance=(SCHEMA_VERSION, "assess_editorial_requirement_evidence"),
        )

    story_role_status = STATE_FOUND if recording_process_status is not None else STATE_UNKNOWN
    coverage_status = STATE_FOUND if idea_coverage_status is not None else STATE_UNKNOWN

    slot_is_exact = editorial_slot_evidence is not None and slot_evidence_source == IDENTITY_MAPPING_EXACT
    slot_evidence_status = STATE_FOUND if slot_is_exact else (STATE_UNKNOWN if editorial_slot_evidence is not None else STATE_NOT_FOUND)
    if editorial_slot_evidence is not None and not slot_is_exact:
        reason_codes.append("slot_evidence_present_but_not_exact_never_authoritative")

    downstream_dependency_status = STATE_FOUND if downstream_dependency_present is not None else STATE_UNKNOWN

    # --- 2. Retry/recording-process firewall (fixtures 9, 10) -- never
    # promoted to REQUIRED regardless of any other signal. ---
    if recording_process_status in _PROCESS_SHAPED_ROLES:
        reason_codes.append("retry_or_process_firewall_never_promoted")
        return LostAtomEditorialRequirementEvidence(
            clip_id=clip_id, coverage_status=coverage_status, story_role_status=story_role_status,
            slot_evidence_status=slot_evidence_status, unique_story_function_status=STATE_NOT_FOUND,
            replacement_coverage_status=STATE_UNKNOWN, downstream_dependency_status=downstream_dependency_status,
            editorial_requirement_status=REQUIREMENT_NOT_REQUIRED,
            reason_codes=tuple(reason_codes), provenance=(SCHEMA_VERSION, "assess_editorial_requirement_evidence"),
        )

    # --- 3. Not even a supported audience delivery -- cannot carry a
    # required audience-facing function. ---
    if audience_delivery_status == AUDIENCE_DELIVERY_NOT_SUPPORTED:
        reason_codes.append("not_a_supported_audience_delivery")
        return LostAtomEditorialRequirementEvidence(
            clip_id=clip_id, coverage_status=coverage_status, story_role_status=story_role_status,
            slot_evidence_status=slot_evidence_status, unique_story_function_status=STATE_NOT_FOUND,
            replacement_coverage_status=STATE_UNKNOWN, downstream_dependency_status=downstream_dependency_status,
            editorial_requirement_status=REQUIREMENT_NOT_REQUIRED,
            reason_codes=tuple(reason_codes), provenance=(SCHEMA_VERSION, "assess_editorial_requirement_evidence"),
        )

    # --- Compute the two independent signal groups. ---
    required_signal = bool(
        idea_coverage_status is True
        or downstream_dependency_present is True
        or (slot_is_exact and editorial_slot_evidence in _STORY_FUNCTION_SLOTS)
    )
    redundant_signal = bool(
        replacement_function_preserved is True
        or row.get("content_loss_suppressed_by") is not None
        or row.get("preserving_realization_id") is not None
    )
    if redundant_signal and (row.get("content_loss_suppressed_by") is not None or row.get("preserving_realization_id") is not None) and replacement_function_preserved is None:
        reason_codes.append("redundancy_via_row_level_suppression_proxy_not_dedicated_function_check")

    # --- 4. Genuine internal conflict between the two signal groups. ---
    if required_signal and redundant_signal:
        reason_codes.append("required_and_redundant_signals_both_present")
        return LostAtomEditorialRequirementEvidence(
            clip_id=clip_id, coverage_status=coverage_status, story_role_status=story_role_status,
            slot_evidence_status=slot_evidence_status, unique_story_function_status=STATE_FOUND,
            replacement_coverage_status=STATE_FOUND, downstream_dependency_status=downstream_dependency_status,
            editorial_requirement_status=REQUIREMENT_CONFLICTED,
            reason_codes=tuple(reason_codes), provenance=(SCHEMA_VERSION, "assess_editorial_requirement_evidence"),
        )

    if required_signal:
        if idea_coverage_status is True:
            reason_codes.append("idea_coverage_would_be_left_uncovered")
        if downstream_dependency_present is True:
            reason_codes.append("downstream_dependency_present")
        if slot_is_exact and editorial_slot_evidence in _STORY_FUNCTION_SLOTS:
            reason_codes.append(f"exact_story_function_slot:{editorial_slot_evidence}")
        return LostAtomEditorialRequirementEvidence(
            clip_id=clip_id, coverage_status=coverage_status, story_role_status=story_role_status,
            slot_evidence_status=slot_evidence_status, unique_story_function_status=STATE_FOUND,
            replacement_coverage_status=STATE_NOT_FOUND, downstream_dependency_status=downstream_dependency_status,
            editorial_requirement_status=REQUIREMENT_REQUIRED,
            reason_codes=tuple(reason_codes), provenance=(SCHEMA_VERSION, "assess_editorial_requirement_evidence"),
        )

    if redundant_signal:
        return LostAtomEditorialRequirementEvidence(
            clip_id=clip_id, coverage_status=coverage_status, story_role_status=story_role_status,
            slot_evidence_status=slot_evidence_status, unique_story_function_status=STATE_NOT_FOUND,
            replacement_coverage_status=STATE_FOUND, downstream_dependency_status=downstream_dependency_status,
            editorial_requirement_status=REQUIREMENT_REDUNDANT_REQUIRED_FUNCTION_PRESERVED,
            reason_codes=tuple(reason_codes), provenance=(SCHEMA_VERSION, "assess_editorial_requirement_evidence"),
        )

    # --- 5. Explicit clearance -- every reused signal that was actually
    # checked says "nothing required here." ---
    explicitly_cleared = (
        idea_coverage_status is False
        and downstream_dependency_present is not True
        and (recording_process_status is None or recording_process_status not in _PROCESS_SHAPED_ROLES)
        and audience_delivery_status != AUDIENCE_DELIVERY_NOT_SUPPORTED
    )
    if explicitly_cleared:
        reason_codes.append("all_reused_signals_explicitly_cleared")
        return LostAtomEditorialRequirementEvidence(
            clip_id=clip_id, coverage_status=coverage_status, story_role_status=story_role_status,
            slot_evidence_status=slot_evidence_status, unique_story_function_status=STATE_NOT_FOUND,
            replacement_coverage_status=STATE_NOT_FOUND, downstream_dependency_status=downstream_dependency_status,
            editorial_requirement_status=REQUIREMENT_NOT_REQUIRED,
            reason_codes=tuple(reason_codes), provenance=(SCHEMA_VERSION, "assess_editorial_requirement_evidence"),
        )

    # --- 6. Honest default: some signals were supplied, but not enough to
    # confidently clear OR require. ---
    reason_codes.append("insufficient_reused_evidence_to_determine_editorial_requirement")
    return LostAtomEditorialRequirementEvidence(
        clip_id=clip_id, coverage_status=coverage_status, story_role_status=story_role_status,
        slot_evidence_status=slot_evidence_status, unique_story_function_status=STATE_UNKNOWN,
        replacement_coverage_status=STATE_UNKNOWN, downstream_dependency_status=downstream_dependency_status,
        editorial_requirement_status=REQUIREMENT_INSUFFICIENT_EVIDENCE,
        reason_codes=tuple(reason_codes), provenance=(SCHEMA_VERSION, "assess_editorial_requirement_evidence"),
    )
