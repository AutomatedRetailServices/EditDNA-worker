"""D-235L: Lost Semantic Atom MATERIALITY DISCRIMINATOR -- OFFLINE ONLY.

Post D-235K (docs/CUTSELL_DECISIONS.md): the D-235K real-media evidence
showed a `REAL_CONTENT_LOSS` blocking row ("oh too many people ready set
these are the") with `missing_critical_atom_count=0`,
`preserved_claim_count=0`, no contradiction, no global idea loss, no lost
critical claim, no integrity failure -- yet the coarse content-vocabulary
signal alone still blocked Selection Freeze. This module answers a
narrower, offline-only question the D-235K gate could not: GIVEN one
already-computed `_lost_semantic_atoms()` row (and a small set of already-
computed, caller-supplied plan-level context flags), what MATERIALITY
classification does the EXISTING evidence already support?

## What this module is NOT (binding, restated from this task's own scope)

- No new semantic engine. Every signal consumed here is a field the real
  engine already computes elsewhere: `atom_classifications` (D-031,
  `semantic_atom_importance.py`'s own `CRITICAL`/`CONTEXTUAL`/`UNCERTAIN`
  vocabulary), `pre_group_restart_consultations` (D-097.1's own retry-
  adjacency arbiter record), `content_loss_suppressed_by`/
  `preserving_realization_id`/`preserved_claim_ids` (D-061/D-076's own
  suppression/preservation evidence). No provider/LLM call is made or
  imported here; no new lexical/keyword classifier is defined.
- No Freeze/authority change. `assess_lost_semantic_atom_materiality()`
  never mutates the caller's row, never reads or writes `diagnostics`,
  `freeze_blocked`, `repair_loop`, or `resolver_status`, and is not called
  from `universal_clean_cut.py`, `final_story_coherence_validation.py`,
  `final_edit_reviewer.py`, or `repair_loop.py` -- this task wires nothing
  into the live path. A future, separately-authorized D-235M is the
  earliest point any of this could influence a real Freeze decision, and
  only ever as a bounded SUPPRESSION of the coarse content-loss signal on
  a subset of the vocabulary below (`NON_MATERIAL_REAL_CONTENT`,
  `RETRY_OR_RECORDING_RESIDUE`, `REDUNDANT_EQUIVALENT`), never as a
  general Freeze relaxation.
- No master score. Every field below is a bounded categorical status or a
  short reason-code tuple -- never a single numeric confidence blended
  across evidence sources.

## Honest gap this module surfaces (not hidden)

`EDITORIALLY_REQUIRED` is a member of `MATERIALITY_VOCABULARY` (the
directive's own requested vocabulary), but this module's decision logic
has NO REACHABLE PATH into it: the current engine has no atom-scoped
"this fragment is necessary for story completeness" signal distinct from
`missing_critical_atom_count` (atom-level, already covered by
`MEANING_CRITICAL`) and idea/family-level completeness checks
(`missing_idea_coverage`, `dropped_no_usable_realization` -- both scoped to
whole retry families, not to one discarded clip's own content). Inventing
one here would be exactly the new heuristic this task's own "reuse
existing evidence first" / "do not build a second semantic engine"
instructions forbid. See the D-235L decision-log entry for the resulting
verdict.

## Critical safety floor (binding)

Any of the following, when present, forces `blocking_recommendation` to
`BLOCK` or `ABSTAIN` -- **never** `DO_NOT_BLOCK`:
- a `CRITICAL`-importance atom classification on the row (negation, or a
  number carrying a percentage/price/measurement/dose/correction-language/
  chronology-relation marker -- D-031's own deterministic rules);
- a caller-supplied `critical_claim_conflict=True` (this row's own clip/
  idea is also implicated in a separate `lost_critical_claims`/
  `contradiction_findings`/`missing_idea_coverage` finding elsewhere in
  the SAME `CanonicalEditPlan` -- a plan-level fact this module never
  discovers itself, only consumes when the caller supplies it);
- malformed/missing atom-importance data on a row that DOES list missing
  critical atoms (schema drift -- ABSTAIN, never silently treated as
  safe);
- an `UNCERTAIN`-importance atom with no `CRITICAL` present (D-031's own
  "WHEN UNCERTAIN, KEEP" -- genuine ambiguity, ABSTAIN, never DO_NOT_BLOCK).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .semantic_atom_importance import CRITICAL, UNCERTAIN

SCHEMA_VERSION = "cutsell.lost_semantic_atom_materiality.v1"

# Tri-state vocabulary, reused verbatim from selection_freeze_diagnostics.py's
# own discipline: UNKNOWN when a signal was never checked/supplied, never
# fabricated as NOT_FOUND from mere absence.
STATE_FOUND = "FOUND"
STATE_NOT_FOUND = "NOT_FOUND"
STATE_UNKNOWN = "UNKNOWN"

# Materiality vocabulary -- bounded, exactly the directive's own seven
# categories. See module docstring for the one category
# (EDITORIALLY_REQUIRED) this module's own evidence cannot reach.
MATERIALITY_MEANING_CRITICAL = "MEANING_CRITICAL"
MATERIALITY_EDITORIALLY_REQUIRED = "EDITORIALLY_REQUIRED"
MATERIALITY_NON_MATERIAL_REAL_CONTENT = "NON_MATERIAL_REAL_CONTENT"
MATERIALITY_RETRY_OR_RECORDING_RESIDUE = "RETRY_OR_RECORDING_RESIDUE"
MATERIALITY_REDUNDANT_EQUIVALENT = "REDUNDANT_EQUIVALENT"
MATERIALITY_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
MATERIALITY_CONFLICTED = "CONFLICTED"

_VALID_MATERIALITY = frozenset({
    MATERIALITY_MEANING_CRITICAL, MATERIALITY_EDITORIALLY_REQUIRED,
    MATERIALITY_NON_MATERIAL_REAL_CONTENT, MATERIALITY_RETRY_OR_RECORDING_RESIDUE,
    MATERIALITY_REDUNDANT_EQUIVALENT, MATERIALITY_INSUFFICIENT_EVIDENCE,
    MATERIALITY_CONFLICTED,
})

# Blocking-recommendation vocabulary -- OFFLINE ADVISORY ONLY. Never wired
# to `blocking`/`freeze_blocked` by this module.
RECOMMEND_BLOCK = "BLOCK"
RECOMMEND_DO_NOT_BLOCK = "DO_NOT_BLOCK"
RECOMMEND_ABSTAIN = "ABSTAIN"

_EXCERPT_MAX_CHARS = 160


def _bounded_text(text: Optional[str], limit: int = _EXCERPT_MAX_CHARS) -> Optional[str]:
    """Same bounding discipline as D-235J's own helper -- smallest bounded
    phrase, hard cap, never a transcript dump."""
    if not text:
        return None
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


@dataclass(frozen=True)
class LostSemanticAtomMateriality:
    clip_id: str
    bounded_excerpt: Optional[str]
    existing_loss_classification: str
    materiality_status: str
    meaning_critical_status: str
    retry_or_process_status: str
    redundancy_status: str
    critical_atom_count: int
    preserved_claim_count: int
    blocking_recommendation: str
    reason_codes: tuple
    confidence_state: str
    provenance: tuple

    def __post_init__(self) -> None:
        if self.materiality_status not in _VALID_MATERIALITY:
            raise ValueError(f"invalid materiality_status: {self.materiality_status!r}")
        if self.blocking_recommendation not in (RECOMMEND_BLOCK, RECOMMEND_DO_NOT_BLOCK, RECOMMEND_ABSTAIN):
            raise ValueError(f"invalid blocking_recommendation: {self.blocking_recommendation!r}")

    def as_dict(self) -> dict:
        """JSON-safe projection for diagnostics serialization -- no
        transcript dump beyond the already-bounded excerpt."""
        return {
            "clip_id": self.clip_id,
            "bounded_excerpt": self.bounded_excerpt,
            "existing_loss_classification": self.existing_loss_classification,
            "materiality_status": self.materiality_status,
            "meaning_critical_status": self.meaning_critical_status,
            "retry_or_process_status": self.retry_or_process_status,
            "redundancy_status": self.redundancy_status,
            "critical_atom_count": self.critical_atom_count,
            "preserved_claim_count": self.preserved_claim_count,
            "blocking_recommendation": self.blocking_recommendation,
            "reason_codes": list(self.reason_codes),
            "confidence_state": self.confidence_state,
            "provenance": list(self.provenance),
        }


def _valid_atom_classifications(row: Mapping) -> list:
    entries = row.get("atom_classifications") or ()
    return [a for a in entries if isinstance(a, Mapping) and "importance" in a]


def assess_lost_semantic_atom_materiality(
    row: Mapping,
    *,
    critical_claim_conflict: Optional[bool] = None,
    recording_process_evidence: Optional[bool] = None,
) -> LostSemanticAtomMateriality:
    """The one D-235L entry point. Pure function of one already-computed
    `_lost_semantic_atoms()` row plus two optional, explicitly-supplied
    plan-level context flags. Recomputes no semantic atom, invents no new
    evidence source, and never mutates its input.

    `critical_claim_conflict`: `True` only when the CALLER has already
    established (from `CanonicalEditPlan.lost_critical_claims`/
    `contradiction_findings`/`missing_idea_coverage`, none of which this
    module reads itself) that this row's own clip/idea is implicated in a
    separate always-critical finding. `False` only when the caller has
    POSITIVELY checked and found none. `None` (default) means "not
    checked" -- never treated as either a floor trigger or a clearance.

    `recording_process_evidence`: `True` only when the caller has already
    established (from `editorial_moment_sequence`'s own per-clip
    `recording_process_status`/`FALSE_START`/`ABANDONED_ATTEMPT`/
    `POST_TAKE_RESET` role classification, D-196) that this SAME clip is a
    confirmed recording-process/false-start/abandoned-attempt/take-reset
    moment. `False` only when the caller has positively confirmed it is
    NOT. `None` (default) means "not checked."
    """
    clip_id = str(row.get("clip_id") or "")
    excerpt = _bounded_text(row.get("text"))
    kind = row.get("kind")
    existing_loss_classification = str(row.get("classification") or kind or "UNKNOWN")
    preserved_claim_count = len(row.get("preserved_claim_ids") or ())
    suppressed_by = row.get("content_loss_suppressed_by")
    preserving_id = row.get("preserving_realization_id")
    restart_consultations = [
        c for c in (row.get("pre_group_restart_consultations") or ()) if isinstance(c, Mapping)
    ]
    missing_critical_atoms_listed = bool(row.get("missing_critical_atoms"))

    valid_atoms = _valid_atom_classifications(row)
    critical_present = any(a.get("importance") == CRITICAL for a in valid_atoms)
    uncertain_present = (not critical_present) and any(a.get("importance") == UNCERTAIN for a in valid_atoms)
    critical_atom_count = sum(1 for a in valid_atoms if a.get("importance") in (CRITICAL, UNCERTAIN))
    malformed_atom_data = missing_critical_atoms_listed and not valid_atoms

    reason_codes: list = []

    # --- 1. Critical safety floor -- always evaluated first, always wins. ---
    if critical_claim_conflict is True:
        reason_codes.append("critical_claim_conflict_present")
        return LostSemanticAtomMateriality(
            clip_id=clip_id, bounded_excerpt=excerpt,
            existing_loss_classification=existing_loss_classification,
            materiality_status=MATERIALITY_MEANING_CRITICAL,
            meaning_critical_status=STATE_FOUND,
            retry_or_process_status=STATE_UNKNOWN, redundancy_status=STATE_UNKNOWN,
            critical_atom_count=critical_atom_count, preserved_claim_count=preserved_claim_count,
            blocking_recommendation=RECOMMEND_BLOCK, reason_codes=tuple(reason_codes),
            confidence_state=STATE_FOUND, provenance=(SCHEMA_VERSION, "assess_lost_semantic_atom_materiality"),
        )

    if critical_present:
        reason_codes.append("critical_atom_present")
        return LostSemanticAtomMateriality(
            clip_id=clip_id, bounded_excerpt=excerpt,
            existing_loss_classification=existing_loss_classification,
            materiality_status=MATERIALITY_MEANING_CRITICAL,
            meaning_critical_status=STATE_FOUND,
            retry_or_process_status=STATE_UNKNOWN, redundancy_status=STATE_UNKNOWN,
            critical_atom_count=critical_atom_count, preserved_claim_count=preserved_claim_count,
            blocking_recommendation=RECOMMEND_BLOCK, reason_codes=tuple(reason_codes),
            confidence_state=STATE_FOUND, provenance=(SCHEMA_VERSION, "assess_lost_semantic_atom_materiality"),
        )

    if malformed_atom_data:
        reason_codes.append("atom_importance_data_missing_or_malformed")
        return LostSemanticAtomMateriality(
            clip_id=clip_id, bounded_excerpt=excerpt,
            existing_loss_classification=existing_loss_classification,
            materiality_status=MATERIALITY_INSUFFICIENT_EVIDENCE,
            meaning_critical_status=STATE_UNKNOWN,
            retry_or_process_status=STATE_UNKNOWN, redundancy_status=STATE_UNKNOWN,
            critical_atom_count=critical_atom_count, preserved_claim_count=preserved_claim_count,
            blocking_recommendation=RECOMMEND_ABSTAIN, reason_codes=tuple(reason_codes),
            confidence_state=STATE_UNKNOWN, provenance=(SCHEMA_VERSION, "assess_lost_semantic_atom_materiality"),
        )

    if uncertain_present:
        reason_codes.append("uncertain_atom_present_when_uncertain_keep")
        return LostSemanticAtomMateriality(
            clip_id=clip_id, bounded_excerpt=excerpt,
            existing_loss_classification=existing_loss_classification,
            materiality_status=MATERIALITY_INSUFFICIENT_EVIDENCE,
            meaning_critical_status=STATE_UNKNOWN,
            retry_or_process_status=STATE_UNKNOWN, redundancy_status=STATE_UNKNOWN,
            critical_atom_count=critical_atom_count, preserved_claim_count=preserved_claim_count,
            blocking_recommendation=RECOMMEND_ABSTAIN, reason_codes=tuple(reason_codes),
            confidence_state=STATE_UNKNOWN, provenance=(SCHEMA_VERSION, "assess_lost_semantic_atom_materiality"),
        )

    # --- Floor clear: no critical/uncertain atom, no critical-claim conflict. ---
    meaning_critical_status = STATE_NOT_FOUND

    # --- 2. Conflicting evidence, reused directly from the row's own
    # already-recorded consultations -- never a new heuristic. ---
    same_idea_values = {
        bool(c.get("same_idea")) for c in restart_consultations if "same_idea" in c
    }
    if len(same_idea_values) > 1:
        reason_codes.append("conflicting_retry_relation_consultations")
        return LostSemanticAtomMateriality(
            clip_id=clip_id, bounded_excerpt=excerpt,
            existing_loss_classification=existing_loss_classification,
            materiality_status=MATERIALITY_CONFLICTED,
            meaning_critical_status=meaning_critical_status,
            retry_or_process_status=STATE_UNKNOWN, redundancy_status=STATE_UNKNOWN,
            critical_atom_count=critical_atom_count, preserved_claim_count=preserved_claim_count,
            blocking_recommendation=RECOMMEND_ABSTAIN, reason_codes=tuple(reason_codes),
            confidence_state=STATE_UNKNOWN, provenance=(SCHEMA_VERSION, "assess_lost_semantic_atom_materiality"),
        )

    # --- 3. Retry / recording-process residue -- structural adjacency
    # (D-097.1's own _pre_group_retry_relation match) or an explicit,
    # caller-supplied editorial_moment_sequence role. ---
    if restart_consultations or recording_process_evidence is True:
        reason_codes.append(
            "pre_group_retry_relation_present" if restart_consultations
            else "recording_process_evidence_confirmed"
        )
        return LostSemanticAtomMateriality(
            clip_id=clip_id, bounded_excerpt=excerpt,
            existing_loss_classification=existing_loss_classification,
            materiality_status=MATERIALITY_RETRY_OR_RECORDING_RESIDUE,
            meaning_critical_status=meaning_critical_status,
            retry_or_process_status=STATE_FOUND, redundancy_status=STATE_UNKNOWN,
            critical_atom_count=critical_atom_count, preserved_claim_count=preserved_claim_count,
            blocking_recommendation=RECOMMEND_DO_NOT_BLOCK, reason_codes=tuple(reason_codes),
            confidence_state=STATE_FOUND, provenance=(SCHEMA_VERSION, "assess_lost_semantic_atom_materiality"),
        )

    # --- 4. Redundant/equivalent -- only ever from already-structurally-
    # supported evidence (a real suppression credit or a verified
    # SemanticPreservationProof), never invented fuzzy equivalence. ---
    if suppressed_by is not None or preserving_id is not None:
        reason_codes.append("content_loss_suppressed_or_preserved" if suppressed_by is not None else "verified_preservation_proof_present")
        return LostSemanticAtomMateriality(
            clip_id=clip_id, bounded_excerpt=excerpt,
            existing_loss_classification=existing_loss_classification,
            materiality_status=MATERIALITY_REDUNDANT_EQUIVALENT,
            meaning_critical_status=meaning_critical_status,
            retry_or_process_status=STATE_NOT_FOUND, redundancy_status=STATE_FOUND,
            critical_atom_count=critical_atom_count, preserved_claim_count=preserved_claim_count,
            blocking_recommendation=RECOMMEND_DO_NOT_BLOCK, reason_codes=tuple(reason_codes),
            confidence_state=STATE_FOUND, provenance=(SCHEMA_VERSION, "assess_lost_semantic_atom_materiality"),
        )

    # --- 5. Non-material -- only when the caller has POSITIVELY confirmed
    # no critical-claim conflict exists (never inferred from absence, and
    # never from short length / incomplete-fragment shape alone -- those
    # are corroboration-only reason codes, added but never load-bearing). ---
    if critical_claim_conflict is False and recording_process_evidence is not True:
        reason_codes.append("critical_claim_conflict_explicitly_cleared")
        own_tokens = row.get("own_content_token_count")
        coverage = row.get("coverage_against_final_keep")
        if isinstance(own_tokens, int) and own_tokens <= 8:
            reason_codes.append("short_fragment_corroboration_only")
        if isinstance(coverage, (int, float)) and coverage < 0.45:
            reason_codes.append("low_coverage_corroboration_only")
        return LostSemanticAtomMateriality(
            clip_id=clip_id, bounded_excerpt=excerpt,
            existing_loss_classification=existing_loss_classification,
            materiality_status=MATERIALITY_NON_MATERIAL_REAL_CONTENT,
            meaning_critical_status=meaning_critical_status,
            retry_or_process_status=STATE_NOT_FOUND, redundancy_status=STATE_NOT_FOUND,
            critical_atom_count=critical_atom_count, preserved_claim_count=preserved_claim_count,
            blocking_recommendation=RECOMMEND_DO_NOT_BLOCK, reason_codes=tuple(reason_codes),
            confidence_state=STATE_FOUND, provenance=(SCHEMA_VERSION, "assess_lost_semantic_atom_materiality"),
        )

    # --- 6. Honest default: no reusable evidence distinguishes this row
    # from a genuinely material loss. This is the D-235K real-shape
    # outcome -- WHEN UNCERTAIN, ABSTAIN, never DO_NOT_BLOCK by default. ---
    reason_codes.append("no_reusable_evidence_beyond_coarse_content_loss_signal")
    return LostSemanticAtomMateriality(
        clip_id=clip_id, bounded_excerpt=excerpt,
        existing_loss_classification=existing_loss_classification,
        materiality_status=MATERIALITY_INSUFFICIENT_EVIDENCE,
        meaning_critical_status=meaning_critical_status,
        retry_or_process_status=STATE_UNKNOWN, redundancy_status=STATE_UNKNOWN,
        critical_atom_count=critical_atom_count, preserved_claim_count=preserved_claim_count,
        blocking_recommendation=RECOMMEND_ABSTAIN, reason_codes=tuple(reason_codes),
        confidence_state=STATE_UNKNOWN, provenance=(SCHEMA_VERSION, "assess_lost_semantic_atom_materiality"),
    )


def assess_many(
    rows: Sequence[Mapping],
    *,
    critical_claim_conflict_by_clip_id: Optional[Mapping[str, bool]] = None,
    recording_process_evidence_by_clip_id: Optional[Mapping[str, bool]] = None,
) -> tuple:
    """Batch convenience wrapper -- assesses each row independently (never
    lets one row's evidence leak into another's classification). Per-clip
    context maps are optional and looked up by `clip_id`; a clip absent
    from a map is treated as `None` (not checked), same as the single-row
    function's own default."""
    critical_claim_conflict_by_clip_id = critical_claim_conflict_by_clip_id or {}
    recording_process_evidence_by_clip_id = recording_process_evidence_by_clip_id or {}
    results = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        clip_id = str(row.get("clip_id") or "")
        results.append(assess_lost_semantic_atom_materiality(
            row,
            critical_claim_conflict=critical_claim_conflict_by_clip_id.get(clip_id),
            recording_process_evidence=recording_process_evidence_by_clip_id.get(clip_id),
        ))
    return tuple(results)
