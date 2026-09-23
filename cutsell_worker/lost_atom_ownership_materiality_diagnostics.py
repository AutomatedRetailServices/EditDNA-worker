"""D-239F: BOUNDED LOST-ATOM OWNERSHIP -> MATERIALITY -> FREEZE
OBSERVABILITY -- OFFLINE ONLY.

## Why this module exists

D-239's own real-media gate proved (via externally recovered artifact
data) that the structural input D-238's `assess_exact_lost_atom_
ownership()` needs genuinely exists on real media for the historical
blocking atom (`clip_1d9d2cebaf8ed3004836`: candidate words 41-48,
uniquely contained by `LanguageAttempt latt_be3141887f7b37d1bb3a`
(31-258), one proposition) -- but NEITHER the D-238 `ExactLostAtom
Ownership` result NOR the D-235Q/`CompleteLostSemanticAtomMateriality`
result for that exact atom was ever serialized into any artifact this
session's own sandbox tooling could read. Freeze stayed blocked and it
was impossible to determine, from any retrievable evidence, WHERE in
the ownership -> D-235Q -> D-235R -> D-235T chain the atom stopped
short of suppression (ownership itself non-singleton? D-235Q abstained
despite exact ownership? D-235R's own firewall correctly preserved a
BLOCK? D-235T never reached this atom at all?).

This module closes exactly that gap: a pure, additive, BEHAVIOR-NEUTRAL
projection over objects `final_story_coherence_validation.py`'s own
Freeze-composition seam ALREADY computes (`lost_atom_ownership_by_
clip_id` -- D-238/D-239's own live wiring; `materiality_by_clip_id` --
D-235Q, unchanged; `critical_claim_conflict_by_clip_id` -- D-235W's own
existing classifier, unchanged) plus ONE additional pure re-derivation
this codebase's OWN existing precedent already performs for diagnostics
(`decide_lost_semantic_atom_freeze_authority(row, materiality)` -- the
SAME call `_lost_atom_materiality_orchestration_diagnostics` in this
same file already makes on the SAME already-computed `materiality`
object; see that function's own docstring/precedent). NEVER calls
`assess_exact_lost_atom_ownership`, NEVER calls `assess_complete_lost_
semantic_atom_materiality`, NEVER calls `decide_lost_atom_repair_
suppression` -- this module reads results, it does not produce them.

## What this module is NOT

- NOT a second ownership/materiality/Freeze/repair policy. Every status
  string in its output is copied verbatim from an object a production
  call site already built; this module's own code contains zero
  containment/materiality-precedence/Freeze-firewall/repair-suppression
  logic of its own.
- NOT a change to `AUTHORITATIVE_RELATIONSHIP_STATUSES`, `exact_lost_
  atom_ownership.py`'s own gate, D-235Q's precedence, D-235R's policy,
  or D-235T's policy. Diff-proven (see this task's own test suite).
- NOT a transcript dump. `bounded_excerpt` reuses the SAME `_bounded_
  text`-style truncation precedent already used throughout the D-235-
  series (see e.g. `lost_semantic_atom_materiality.py`'s own
  `_bounded_text`) -- capped, never the full row.

## Two-part construction (see module callers)

D-235S/D-235T's own real decision objects (`LostAtomRepairSuppressionDecision`,
`RepairAttempt`) are only available AFTER `repair_loop.py::run_repair_loop`
executes, which happens strictly later than `final_story_coherence_
validation.py`'s own diagnostics-building pass. Rather than recompute
anything to force them into the same call, this module's own primary
entry point (`build_lost_atom_ownership_materiality_diagnostics`)
produces the OWNERSHIP/D-235Q/D-235R portion only (called from `final_
story_coherence_validation.py`, where those objects are already in
scope); a SEPARATE, later merge step (a workflow-level pure JSON join by
`lost_atom_provenance_id`, never a Python re-import) attaches the D-235S/
D-235T fields a sibling function
(`lost_atom_repair_suppression_by_provenance_diagnostics`, this same
module) projects from `RepairLoopResult.suppression_decisions`/
`.attempts` (see `repair_loop.py`'s own D-239F addition). One source of
truth per field; two harmless, additive projection passes over it.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .complete_lost_semantic_atom_materiality import CompleteLostSemanticAtomMateriality
from .exact_lost_atom_ownership import ExactLostAtomOwnership
from .lost_semantic_atom_freeze_authority import decide_lost_semantic_atom_freeze_authority

SCHEMA_VERSION = "cutsell.lost_atom_ownership_materiality_diagnostics.v1"

_BOUNDED_EXCERPT_MAX_CHARS = 160


def _bounded_text(value: object) -> Optional[str]:
    if not value:
        return None
    text = str(value)
    return text[:_BOUNDED_EXCERPT_MAX_CHARS]


def build_lost_atom_ownership_materiality_diagnostics(
    lost_semantic_atoms: Sequence[Mapping],
    *,
    lost_atom_ownership_by_clip_id: Optional[Mapping[str, ExactLostAtomOwnership]] = None,
    materiality_by_clip_id: Optional[Mapping[str, CompleteLostSemanticAtomMateriality]] = None,
    critical_claim_conflict_by_clip_id: Optional[Mapping[str, Optional[bool]]] = None,
) -> dict:
    """Ownership (D-238) + materiality (D-235Q) + Freeze-authority (D-235R)
    per-atom projection. Pure; mutates nothing, calls only the ONE already-
    precedented pure re-derivation (`decide_lost_semantic_atom_freeze_
    authority`, reused verbatim on the SAME row + SAME already-computed
    materiality every other call site in this codebase already uses).
    `{}`/`None` inputs degrade to explicit `None`/`False` fields per atom,
    never a fabricated guess."""
    lost_atom_ownership_by_clip_id = lost_atom_ownership_by_clip_id or {}
    materiality_by_clip_id = materiality_by_clip_id or {}
    critical_claim_conflict_by_clip_id = critical_claim_conflict_by_clip_id or {}

    atoms = []
    for row in lost_semantic_atoms:
        if not isinstance(row, Mapping):
            continue
        clip_id = str(row.get("clip_id") or "")
        if not clip_id:
            continue
        provenance_id = row.get("lost_atom_provenance_id")
        ownership = lost_atom_ownership_by_clip_id.get(clip_id)
        materiality = materiality_by_clip_id.get(clip_id)
        freeze_decision = (
            decide_lost_semantic_atom_freeze_authority(row, materiality)
            if materiality is not None else None
        )

        atoms.append({
            "clip_id": clip_id,
            "lost_atom_provenance_id": provenance_id,
            "bounded_excerpt": _bounded_text(row.get("text")),
            "classification": row.get("classification"),
            "blocking": bool(row.get("blocking", True)),
            # -- D-238 ownership --
            "ownership_input_present": ownership is not None,
            "ownership_status": ownership.ownership_status if ownership is not None else None,
            "containing_language_attempt_id": (
                ownership.containing_language_attempt_id if ownership is not None else None
            ),
            "proposition_candidate_ids": (
                list(ownership.proposition_candidate_ids) if ownership is not None else []
            ),
            "ownership_ambiguity_reason": (
                list(ownership.reason_codes)
                if ownership is not None and not ownership.is_exact_singleton
                else []
            ),
            # -- D-235Q --
            "exact_ownership_available": (
                bool(materiality.exact_ownership_available) if materiality is not None else None
            ),
            "critical_claim_conflict_state": critical_claim_conflict_by_clip_id.get(clip_id),
            "editorial_requirement_state": (
                materiality.editorial_requirement_status if materiality is not None else None
            ),
            "meaning_critical_state": (
                materiality.meaning_materiality_status if materiality is not None else None
            ),
            "retry_process_state": (
                materiality.retry_or_process_status if materiality is not None else None
            ),
            "redundancy_state": (
                materiality.redundancy_status if materiality is not None else None
            ),
            # -- D-239I: which evidence source actually reached each
            # dimension's own gate -- pure re-projection of already-
            # computed booleans on `materiality`/`ownership`, never a new
            # computation. "Eligible" (not "used") for the critical-
            # context/redundancy bridges: those two seams resolve at
            # `final_story_coherence_validation.py`'s own orchestration
            # layer, strictly before this row's own `materiality` object
            # exists, so this module (which never sees that layer's own
            # intermediate lookups) can only honestly report whether
            # D-238 ownership was exact enough to make the bridge
            # possible, not whether it actually fired for this row.
            "editorial_requirement_evidence_source": (
                "EXACT_IDENTITY" if materiality is not None and materiality.exact_identity_available
                else "EXACT_OWNERSHIP" if materiality is not None and materiality.exact_ownership_available
                else "NONE" if materiality is not None else None
            ),
            "critical_context_ownership_bridge_eligible": (
                bool(ownership is not None and ownership.is_exact_singleton)
            ),
            "retry_process_ownership_bridge_eligible": (
                bool(ownership is not None and ownership.is_exact_singleton)
            ),
            "redundancy_ownership_bridge_eligible": (
                bool(ownership is not None and ownership.is_exact_singleton)
            ),
            # -- D-239L: atom-granular refinement of the ownership-only
            # REQUIRED bridge -- pure re-projection of already-computed
            # fields on `materiality` (see complete_lost_semantic_atom_
            # materiality.py's own "D-239L" docstring section), never a
            # new computation.
            "editorial_requirement_granularity": (
                materiality.editorial_requirement_granularity if materiality is not None else None
            ),
            "editorial_requirement_target_evidence_source": (
                materiality.editorial_requirement_target_evidence_source if materiality is not None else None
            ),
            "final_materiality_status": (
                materiality.final_materiality_status if materiality is not None else None
            ),
            "blocking_recommendation": (
                materiality.blocking_recommendation if materiality is not None else None
            ),
            # -- D-235R --
            "freeze_materiality_received": materiality is not None,
            "freeze_authority_status": (
                freeze_decision.authority_status if freeze_decision is not None else None
            ),
            "freeze_effective_blocking": (
                freeze_decision.effective_blocking if freeze_decision is not None else None
            ),
            "freeze_reason": (
                freeze_decision.safety_block_reason if freeze_decision is not None else None
            ),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "atom_count": len(atoms),
        "atoms": atoms,
        "provenance": (SCHEMA_VERSION, "build_lost_atom_ownership_materiality_diagnostics"),
    }


def lost_atom_repair_suppression_by_provenance_diagnostics(
    *,
    suppression_decisions: Sequence = (),
    repair_attempts: Sequence = (),
) -> dict:
    """D-235S/D-235T per-provenance-id projection. Pure; reads ONLY the
    ALREADY-COMPUTED `RepairLoopResult.suppression_decisions` (D-239F's own
    new capture -- see `repair_loop.py`, previously discarded) and
    `.attempts` (D-235S, unchanged since D-050B). Never calls `decide_
    lost_atom_repair_suppression` itself -- that would be the exact
    recomputation this task's own directive forbids."""
    attempts_by_provenance: dict = {}
    for attempt in repair_attempts:
        pid = getattr(attempt, "source_lost_atom_provenance_id", None)
        if pid and pid not in attempts_by_provenance:
            attempts_by_provenance[pid] = attempt

    entries = []
    for decision in suppression_decisions:
        pid = decision.lost_atom_provenance_id
        attempt = attempts_by_provenance.get(pid) if pid else None
        entries.append({
            "lost_atom_provenance_id": pid,
            "d235s_reviewer_finding_kind": decision.reviewer_finding_kind,
            "d235s_repair_attempt_provenance_confirmed": attempt is not None,
            "d235s_repair_attempt_reason": getattr(attempt, "reason", None),
            "d235s_exact_link_status": (
                decision.reason if str(decision.reason).startswith(
                    ("provenance_link_", "lost_atom_provenance_id_missing"),
                ) else "EXACT_MATCH"
            ),
            "d235t_precomputed_materiality_received": decision.materiality_status is not None,
            "d235t_suppression_status": decision.suppression_status,
            "d235t_suppress_repair_escalation": decision.suppress_repair_escalation,
            "d235t_reason": decision.reason,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "entry_count": len(entries),
        "entries": entries,
        "suppressed_count": sum(1 for e in entries if e["d235t_suppress_repair_escalation"]),
        "preserved_count": sum(1 for e in entries if not e["d235t_suppress_repair_escalation"]),
        "provenance": (SCHEMA_VERSION, "lost_atom_repair_suppression_by_provenance_diagnostics"),
    }
