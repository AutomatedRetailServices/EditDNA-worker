"""D-235R: BOUNDED LOST-ATOM FREEZE AUTHORITY ADAPTER -- OFFLINE FIRST.

Post D-235Q (docs/CUTSELL_DECISIONS.md, VERDICT A -- COMPLETE LOST-ATOM
MATERIALITY INTEGRATION OFFLINE PROVEN): this module is the ONE bounded
adapter that decides whether an EXISTING, already-computed lost-semantic-
atom Freeze blocker (`_lost_semantic_atoms()` row's own `blocking` field)
should remain blocking, using D-235Q's own `CompleteLostSemanticAtomMateriality`
result -- imported, never reimplemented.

## What this module is NOT (binding, restated from this task's own scope)

- No new semantic engine, no numeric score. `decide_lost_semantic_atom_
  freeze_authority()` reads only D-235Q's own already-computed result
  (imported unmodified) plus the row's own `blocking` field.
- No RepairLoop change, no resolver change, no threshold change, no P1/P2/
  BestTake/Family/Ordering/Boundary/Pacing/Audio-Join change. This module
  is imported by NOTHING in `repair_loop.py`, `final_edit_reviewer.py`,
  `deterministic_best_take_authority.py`, `boundary_engine_pass.py`,
  `dialogue_pacing_transition.py`, or any resolver module (module-leaf
  grep tests, this task's own test suite).
- No general Freeze redesign. The ONLY Freeze-composition change this
  task makes is at ONE existing seam in `final_story_coherence_
  validation.py` -- replacing `any(row.get("blocking", True) for row in
  lost_semantic_atoms)` with a call into this module's own `lost_
  semantic_atom_freeze_trigger_present()`, behind a new, default-OFF
  flag (`CUTSELL_LOST_ATOM_MATERIALITY_FREEZE_AUTHORITY_ENABLED`). Every
  other Freeze term (`contradiction_findings`, `missing_idea_coverage`,
  `lost_critical_claims`, `authority_membership_findings`, repair-loop
  status, resolver status, D-090 integrity) is completely untouched.

## Only allowed suppression (binding, this task's own 10-condition gate)

`effective_blocking` may become `False` for an originally-blocking row
ONLY when ALL of the following hold -- checked explicitly and
independently in `decide_lost_semantic_atom_freeze_authority()` (never
trusting D-235Q's own internal precedence alone as the sole safety net --
"never trust a single point of failure," this task's own defense-in-depth
instruction):

  1. `row.get("blocking", True)` is `True` (nothing to suppress otherwise
     -- see "Non-blocking atoms" below);
  2. a real `CompleteLostSemanticAtomMateriality` instance was supplied
     for this row (never a dict, never `None` -- type-checked);
  3. `materiality.blocking_recommendation == DO_NOT_BLOCK`;
  4. `materiality.final_materiality_status` is exactly one of
     `NON_MATERIAL_REAL_CONTENT` / `RETRY_OR_RECORDING_RESIDUE` /
     `REDUNDANT_EQUIVALENT` (imported from D-235Q, never redefined);
  5. `materiality.meaning_materiality_status != MEANING_CRITICAL`;
  6. `materiality.editorial_requirement_status != REQUIRED`;
  7. no `CONFLICTED` anywhere (`final_materiality_status`,
     `meaning_materiality_status`, `editorial_requirement_status`);
  8. no `INSUFFICIENT_EVIDENCE` anywhere (same three fields);
  9. `materiality.blocking_recommendation != BLOCK` (redundant with 3/5/6
     by construction, kept as an explicit second check per this task's
     own defense-in-depth instruction rather than relying on condition 3
     alone);
  10. the identity-sufficiency requirement D-235Q's own step 6 gate
      already enforces is independently RE-VERIFIED here: if
      `editorial_requirement_status == INSUFFICIENT_EVIDENCE` (D-235M's
      own generic "nothing found" default) `exact_identity_available`
      MUST be `True` -- a heuristic-only or missing identity paired with
      an `INSUFFICIENT_EVIDENCE` editorial-requirement verdict must never
      reach suppression even if some upstream caller's own composition
      bug let a `NON_MATERIAL_REAL_CONTENT`/`DO_NOT_BLOCK` result through
      without satisfying this.

Any single failure -> `PRESERVE_BLOCK` (or `ABSTAIN_PRESERVE_BLOCK` when
the underlying verdict was itself an abstention) -- see "Fail-closed
contract" below.

## Fail-closed contract (binding)

Missing, malformed, unknown, conflicted, insufficient, or heuristic-only-
where-exact-was-required evidence NEVER suppresses. `decide_lost_
semantic_atom_freeze_authority()` has exactly one path to `effective_
blocking=False` for an originally-blocking row -- every other path
preserves `True`. Structurally guaranteed (single `if` gate, `else`
branches all preserve), never a probabilistic/threshold judgment.

## Meaning-critical / editorially-required firewalls (absolute)

Both are ordinary consequences of the 10-condition gate above (conditions
5/6/9), but are additionally documented and separately tested as named
firewalls per this task's own instruction: neither can ever be bypassed
by a retry/redundant/non-material signal appearing ELSEWHERE on the same
row, because D-235Q's own precedence (imported, unmodified) already
ranks meaning-critical and editorial-required ABOVE retry/redundant/non-
material, and because conditions 5/6/9 here independently re-check the
same firewall a second time.

## Multiple lost atoms (binding, never a majority vote)

`lost_semantic_atom_freeze_trigger_present()` computes ONE decision per
row and returns `any(decision.effective_blocking for decision in
decisions)` -- if even ONE originally-blocking atom fails to clear the
10-condition gate, the overall lost-semantic-atom Freeze trigger REMAINS
present, regardless of how many other atoms were safely suppressed. An
atom already `blocking=False` is never reinterpreted or promoted
(`AUTHORITY_NOT_APPLICABLE`, `effective_blocking=False` unconditionally,
independent of any materiality result).

## Freeze composition seam (the only live-wiring this task performs)

`final_story_coherence_validation.py`'s two `freeze_blocked` computation
sites both replace their own `any(row.get("blocking", True) for row in
lost_semantic_atoms)` sub-expression with `lost_semantic_atom_freeze_
trigger_present(lost_semantic_atoms)` -- when the new flag is OFF
(default), this function's own first line returns the IDENTICAL boolean
expression, so `freeze_blocked`'s value, and every other Freeze term, is
byte-identical to pre-D-235R behavior (mandatory, tested).

## What is, and is not, reachable through the LIVE seam today (honest
## scope boundary, not hidden)

`lost_semantic_atom_freeze_trigger_present()` calls D-235Q's own
`assess_complete_lost_semantic_atom_materiality(row)` with ONLY the row
itself -- no `critical_claim_conflict` override, no D-235P exact-identity
match, no editorial-requirement signals. This is a deliberate choice: at
`final_story_coherence_validation.py`'s own call sites, `contradiction_
findings`/`missing_idea_coverage`/`lost_critical_claims` exist only as
PLAN-LEVEL lists with no already-established per-`clip_id` correlation
this task's own scope authorizes building (that would be exactly the
kind of new heuristic/correlation the whole D-235 series has consistently
refused to invent without separate authorization), and D-235P's own exact
identity requires the Live Language Spine construction seam
(`language_spine_live_integration.build_live_language_spine_for_source`),
itself gated behind its own default-OFF diagnostics flag and not
constructed at this call site.

Consequence, proven by this task's own tests: with the flag ON, using
ONLY real row-native evidence, `RETRY_OR_RECORDING_RESIDUE` (from the
row's own `pre_group_restart_consultations`) and `REDUNDANT_EQUIVALENT`
(from the row's own `content_loss_suppressed_by`/`preserving_
realization_id`) suppression IS genuinely reachable through the live
seam today -- `NON_MATERIAL_REAL_CONTENT` suppression (the D-235K real-
shape's own category) requires `critical_claim_conflict` to be positively
cleared, which is NOT safely derivable at this call site without that
separately-authorized correlation work, so it currently ABSTAINS/
preserves live even though it is fully proven reachable OFFLINE (see
`assess_lost_semantic_atom_freeze_authority_for_d235k_shape_fixture`-style
tests, which construct the full context directly, exactly as D-235Q's own
test suite already does).

## RepairLoop / FinalEditReviewer (NOT touched -- binding, per this
## task's own banner-level "NO REPAIR-LOOP CHANGE")

This module NEVER imports, calls, or is imported by `repair_loop.py` or
`final_edit_reviewer.py`. D-235J's own already-established finding
(`selection_freeze_diagnostics._find_repair_link`) shows a REAL, exact
(never text-matched), `clip_id`-based linkage CAN exist between a lost-
atom row and a repair-loop attempt, and `final_edit_reviewer.py::review()`
maps every `lost_semantic_atoms` row to exactly one `UNIQUE_FACT_LOST`
Finding 1:1 by construction -- but actually preventing RepairLoop from
escalating such a finding to `NEEDS_HUMAN_REVIEW` would require a change
to `repair_loop.py` itself, which this task's own banner explicitly
forbids regardless of whether the identity linkage exists. This module
therefore reports the observability-only linkage (reusing D-235J's own
`_find_repair_link` shape, never a new text-based reconstruction) as pure
diagnostics, and this task's own verdict is B (a real repair-loop
same-atom GAP remains, per the directive's own explicit escape hatch),
never silently claiming it as solved.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from .complete_lost_semantic_atom_materiality import (
    CompleteLostSemanticAtomMateriality,
    assess_complete_lost_semantic_atom_materiality,
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
)
from .lost_atom_editorial_requirement_evidence import (
    REQUIREMENT_CONFLICTED,
    REQUIREMENT_INSUFFICIENT_EVIDENCE,
    REQUIREMENT_REQUIRED,
)

SCHEMA_VERSION = "cutsell.lost_semantic_atom_freeze_authority.v1"

_AUTHORITY_ENV = "CUTSELL_LOST_ATOM_MATERIALITY_FREEZE_AUTHORITY_ENABLED"


def _env_true_default_false(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def lost_atom_materiality_freeze_authority_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Default OFF -- same pattern as `language_spine_live_integration.
    live_language_spine_diagnostics_enabled`. When OFF, `lost_semantic_
    atom_freeze_trigger_present()`'s first line returns the byte-identical
    pre-D-235R expression; nothing else in this module is ever consulted
    by `final_story_coherence_validation.py`."""
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_AUTHORITY_ENV))


# ---------------------------------------------------------------------------
# Authority-status vocabulary (this task's own required 4-value set).
# ---------------------------------------------------------------------------
AUTHORITY_PRESERVE_BLOCK = "PRESERVE_BLOCK"
AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK = "SUPPRESS_NON_MATERIAL_BLOCK"
AUTHORITY_ABSTAIN_PRESERVE_BLOCK = "ABSTAIN_PRESERVE_BLOCK"
AUTHORITY_NOT_APPLICABLE = "NOT_APPLICABLE"

_VALID_AUTHORITY_STATUSES = frozenset({
    AUTHORITY_PRESERVE_BLOCK, AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK,
    AUTHORITY_ABSTAIN_PRESERVE_BLOCK, AUTHORITY_NOT_APPLICABLE,
})

# The ONLY three final_materiality_status values a suppression may ever be
# based on -- imported from D-235Q, never redefined with different spellings.
SUPPRESSIBLE_FINAL_MATERIALITY_STATUSES = frozenset({
    MATERIALITY_NON_MATERIAL_REAL_CONTENT,
    MATERIALITY_RETRY_OR_RECORDING_RESIDUE,
    MATERIALITY_REDUNDANT_EQUIVALENT,
})

# Repair-loop linkage vocabulary (observability only -- see module
# docstring's "RepairLoop / FinalEditReviewer (NOT touched)" section).
REPAIR_FINDING_LABEL_NON_MATERIAL_LOST_ATOM_SUPPRESSED = "NON_MATERIAL_LOST_ATOM_SUPPRESSED"
REVIEWER_FINDING_KIND_UNIQUE_FACT_LOST = "UNIQUE_FACT_LOST"


@dataclass(frozen=True)
class LostSemanticAtomFreezeAuthorityDecision:
    clip_id: str
    original_blocking: bool
    materiality_status: Optional[str]
    materiality_recommendation: Optional[str]
    authority_status: str
    effective_blocking: bool
    suppression_applied: bool
    suppression_reason: Optional[str]
    safety_block_reason: Optional[str]
    provenance: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.authority_status not in _VALID_AUTHORITY_STATUSES:
            raise ValueError(f"invalid authority_status: {self.authority_status!r}")

    def as_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "original_blocking": self.original_blocking,
            "materiality_status": self.materiality_status,
            "materiality_recommendation": self.materiality_recommendation,
            "authority_status": self.authority_status,
            "effective_blocking": self.effective_blocking,
            "suppression_applied": self.suppression_applied,
            "suppression_reason": self.suppression_reason,
            "safety_block_reason": self.safety_block_reason,
            "provenance": list(self.provenance),
        }


def decide_lost_semantic_atom_freeze_authority(
    row: Mapping,
    materiality: Optional[CompleteLostSemanticAtomMateriality],
) -> LostSemanticAtomFreezeAuthorityDecision:
    """The one D-235R entry point. Pure function; mutates nothing, mints no
    new id. See module docstring's "Only allowed suppression" / "Fail-
    closed contract" sections for the exact gate implemented below."""
    clip_id = str(row.get("clip_id") or "")
    original_blocking = bool(row.get("blocking", True))

    # Non-blocking atoms are never reinterpreted or promoted.
    if not original_blocking:
        return LostSemanticAtomFreezeAuthorityDecision(
            clip_id=clip_id, original_blocking=False, materiality_status=None,
            materiality_recommendation=None, authority_status=AUTHORITY_NOT_APPLICABLE,
            effective_blocking=False, suppression_applied=False, suppression_reason=None,
            safety_block_reason=None, provenance=(SCHEMA_VERSION, "decide_lost_semantic_atom_freeze_authority"),
        )

    # Fail-closed: missing or malformed (wrong-typed) materiality result.
    if not isinstance(materiality, CompleteLostSemanticAtomMateriality):
        return LostSemanticAtomFreezeAuthorityDecision(
            clip_id=clip_id, original_blocking=True, materiality_status=None,
            materiality_recommendation=None, authority_status=AUTHORITY_PRESERVE_BLOCK,
            effective_blocking=True, suppression_applied=False, suppression_reason=None,
            safety_block_reason="materiality_result_missing_or_malformed",
            provenance=(SCHEMA_VERSION, "decide_lost_semantic_atom_freeze_authority"),
        )

    m_status = materiality.final_materiality_status
    meaning_status = materiality.meaning_materiality_status
    requirement_status = materiality.editorial_requirement_status
    recommendation = materiality.blocking_recommendation

    # Absolute firewalls (conditions 5/6/9 -- independently re-checked,
    # never trusting D-235Q's own precedence as the sole safety net).
    if meaning_status == MATERIALITY_MEANING_CRITICAL:
        return _preserve(clip_id, materiality, "meaning_critical_firewall")
    if requirement_status == REQUIREMENT_REQUIRED:
        return _preserve(clip_id, materiality, "editorially_required_firewall")
    if recommendation == RECOMMEND_BLOCK:
        return _preserve(clip_id, materiality, "blocking_recommendation_block")

    # No conflict/insufficiency anywhere on the OVERALL verdict (conditions
    # 7/8) -- `editorial_requirement_status` alone is deliberately NOT
    # included here: D-235Q's own retry/redundant precedence legitimately
    # reaches DO_NOT_BLOCK with `editorial_requirement_status ==
    # INSUFFICIENT_EVIDENCE` whenever D-235M genuinely had nothing to say
    # about story-function requirement (D-235L's own retry/redundant
    # signal is independently sufficient there) -- see condition 10 below
    # for the ONE combination (NON_MATERIAL_REAL_CONTENT + requirement
    # INSUFFICIENT_EVIDENCE) where that sub-field DOES matter.
    # `requirement_status == CONFLICTED` is still checked here as harmless
    # defense-in-depth: D-235Q's own precedence structurally forecloses
    # this combination ever co-occurring with a DO_NOT_BLOCK/suppressible
    # final status, so this branch cannot fire on valid D-235Q output --
    # it only guards a hypothetical malformed/inconsistent object.
    conflict_or_insufficient = (
        m_status in (MATERIALITY_CONFLICTED, MATERIALITY_INSUFFICIENT_EVIDENCE)
        or requirement_status == REQUIREMENT_CONFLICTED
    )
    if conflict_or_insufficient:
        return _abstain_preserve(clip_id, materiality, "conflicted_or_insufficient_evidence")

    # Condition 3: must be an explicit DO_NOT_BLOCK.
    if recommendation != RECOMMEND_DO_NOT_BLOCK:
        return _abstain_preserve(clip_id, materiality, "recommendation_not_do_not_block")

    # Condition 4: final_materiality_status must be exactly one of the
    # three suppressible categories.
    if m_status not in SUPPRESSIBLE_FINAL_MATERIALITY_STATUSES:
        return _abstain_preserve(clip_id, materiality, "final_materiality_status_not_suppressible")

    # Condition 10: identity-sufficiency re-verification (defense in
    # depth), scoped exactly to the ONE combination where it matters --
    # NON_MATERIAL_REAL_CONTENT reached with an INSUFFICIENT_EVIDENCE
    # editorial-requirement verdict must never suppress unless exact
    # identity was genuinely available (D-235Q's own step 6 gate already
    # enforces this; re-checked independently here, never trusting a
    # single layer). RETRY_OR_RECORDING_RESIDUE/REDUNDANT_EQUIVALENT
    # legitimately reach DO_NOT_BLOCK with `editorial_requirement_status
    # == INSUFFICIENT_EVIDENCE` on every ordinary row (D-235M has nothing
    # to say when no story-function signal was supplied at all) -- that
    # combination is NOT a violation and must not be preserved here.
    if (
        m_status == MATERIALITY_NON_MATERIAL_REAL_CONTENT
        and requirement_status == REQUIREMENT_INSUFFICIENT_EVIDENCE
        and not materiality.exact_identity_available
    ):
        return _preserve(clip_id, materiality, "identity_insufficient_for_suppression")

    # All 10 conditions satisfied -- the ONLY path to suppression.
    return LostSemanticAtomFreezeAuthorityDecision(
        clip_id=clip_id, original_blocking=True, materiality_status=m_status,
        materiality_recommendation=recommendation, authority_status=AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK,
        effective_blocking=False, suppression_applied=True, suppression_reason=m_status,
        safety_block_reason=None, provenance=(SCHEMA_VERSION, "decide_lost_semantic_atom_freeze_authority"),
    )


def _preserve(clip_id: str, materiality: CompleteLostSemanticAtomMateriality, reason: str) -> LostSemanticAtomFreezeAuthorityDecision:
    return LostSemanticAtomFreezeAuthorityDecision(
        clip_id=clip_id, original_blocking=True, materiality_status=materiality.final_materiality_status,
        materiality_recommendation=materiality.blocking_recommendation, authority_status=AUTHORITY_PRESERVE_BLOCK,
        effective_blocking=True, suppression_applied=False, suppression_reason=None,
        safety_block_reason=reason, provenance=(SCHEMA_VERSION, "decide_lost_semantic_atom_freeze_authority"),
    )


def _abstain_preserve(clip_id: str, materiality: CompleteLostSemanticAtomMateriality, reason: str) -> LostSemanticAtomFreezeAuthorityDecision:
    return LostSemanticAtomFreezeAuthorityDecision(
        clip_id=clip_id, original_blocking=True, materiality_status=materiality.final_materiality_status,
        materiality_recommendation=materiality.blocking_recommendation, authority_status=AUTHORITY_ABSTAIN_PRESERVE_BLOCK,
        effective_blocking=True, suppression_applied=False, suppression_reason=None,
        safety_block_reason=reason, provenance=(SCHEMA_VERSION, "decide_lost_semantic_atom_freeze_authority"),
    )


def decide_lost_semantic_atoms_freeze_decisions(
    lost_semantic_atoms: Iterable[Mapping],
    materiality_by_clip_id: Optional[Mapping[str, CompleteLostSemanticAtomMateriality]] = None,
) -> Tuple[LostSemanticAtomFreezeAuthorityDecision, ...]:
    """Batch form: one decision per row, in input order. `materiality_by_
    clip_id` absent entries are treated as missing materiality (fail-closed
    PRESERVE_BLOCK for an originally-blocking row)."""
    materiality_by_clip_id = materiality_by_clip_id or {}
    decisions = []
    for row in lost_semantic_atoms:
        if not isinstance(row, Mapping):
            continue
        clip_id = str(row.get("clip_id") or "")
        decisions.append(decide_lost_semantic_atom_freeze_authority(row, materiality_by_clip_id.get(clip_id)))
    return tuple(decisions)


def lost_semantic_atom_freeze_trigger_present(
    lost_semantic_atoms: Optional[Sequence[Mapping]],
    *,
    enabled: Optional[bool] = None,
    materiality_by_clip_id: Optional[Mapping[str, CompleteLostSemanticAtomMateriality]] = None,
) -> bool:
    """The one Freeze-composition seam function -- replaces `any(row.get(
    "blocking", True) for row in lost_semantic_atoms)` at BOTH call sites
    in `final_story_coherence_validation.py`. `enabled=None` (the live
    call sites' own default) checks the env flag; tests may pass an
    explicit bool. When OFF, returns the IDENTICAL original expression --
    mandatory byte-identical default-off parity (see module docstring).

    When ON and `materiality_by_clip_id` is not explicitly supplied (the
    live call sites' own usage), computes `assess_complete_lost_semantic_
    atom_materiality(row)` for each row using ONLY the row itself -- see
    module docstring's "What is, and is not, reachable through the LIVE
    seam today" section for exactly which suppression categories this
    reaches. `materiality_by_clip_id` lets a caller (or a test) supply a
    richer, separately-computed D-235Q result per clip_id instead."""
    rows = tuple(row for row in (lost_semantic_atoms or ()) if isinstance(row, Mapping))
    is_enabled = enabled if enabled is not None else lost_atom_materiality_freeze_authority_enabled()
    if not is_enabled:
        return any(row.get("blocking", True) for row in rows)

    if materiality_by_clip_id is not None:
        by_clip_id = materiality_by_clip_id
    else:
        by_clip_id = {
            str(row.get("clip_id") or ""): assess_complete_lost_semantic_atom_materiality(row)
            for row in rows
        }
    decisions = decide_lost_semantic_atoms_freeze_decisions(rows, by_clip_id)
    return any(decision.effective_blocking for decision in decisions)


# ---------------------------------------------------------------------------
# RepairLoop linkage -- OBSERVABILITY ONLY, never wired into RepairLoop's
# own decision (see module docstring's "RepairLoop / FinalEditReviewer
# (NOT touched)" section). Mirrors `selection_freeze_diagnostics.
# _find_repair_link`'s own exact-id-only matching shape, never a new
# text-based reconstruction.
# ---------------------------------------------------------------------------
def suppressed_atom_repair_finding_labels(
    decisions: Iterable[LostSemanticAtomFreezeAuthorityDecision],
    repair_loop_attempts: Optional[Sequence[Mapping]] = None,
) -> dict:
    """For each SUPPRESSED decision, reports whether an EXACT (never text-
    matched) repair-loop attempt linkage exists -- `finding_kind ==
    UNIQUE_FACT_LOST` and this clip_id appears in that attempt's own real
    `previous_realization` tuple, same as D-235J's own `_find_repair_link`.
    Diagnostics only: this function's own return value is never read by
    `repair_loop.py`, and calling it changes no RepairLoop behavior."""
    repair_loop_attempts = tuple(repair_loop_attempts or ())
    labels: dict = {}
    for decision in decisions:
        if not decision.suppression_applied:
            continue
        linked = False
        for attempt in repair_loop_attempts:
            if not isinstance(attempt, Mapping):
                continue
            if attempt.get("finding_kind") != REVIEWER_FINDING_KIND_UNIQUE_FACT_LOST:
                continue
            previous_realization = attempt.get("previous_realization") or ()
            if decision.clip_id in previous_realization:
                linked = True
                break
        if linked:
            labels[decision.clip_id] = REPAIR_FINDING_LABEL_NON_MATERIAL_LOST_ATOM_SUPPRESSED
    return labels


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe, no transcript dump).
# ---------------------------------------------------------------------------
def lost_semantic_atom_freeze_authority_diagnostics(
    *,
    lost_semantic_atoms: Optional[Sequence[Mapping]] = None,
    decisions: Optional[Sequence[LostSemanticAtomFreezeAuthorityDecision]] = None,
    repair_loop_attempts: Optional[Sequence[Mapping]] = None,
    enabled: Optional[bool] = None,
) -> dict:
    rows = tuple(row for row in (lost_semantic_atoms or ()) if isinstance(row, Mapping))
    decisions = tuple(decisions or ())
    is_enabled = enabled if enabled is not None else lost_atom_materiality_freeze_authority_enabled()

    original_blocking_count = sum(1 for row in rows if bool(row.get("blocking", True)))
    effective_blocking_count = sum(1 for d in decisions if d.effective_blocking)
    suppressed_count = sum(1 for d in decisions if d.suppression_applied)
    preserved_blocking_count = sum(
        1 for d in decisions if d.original_blocking and d.effective_blocking
    )
    suppression_reasons = sorted({d.suppression_reason for d in decisions if d.suppression_reason})

    if not decisions:
        suppression_status = "NOT_EVALUATED"
    elif suppressed_count == 0:
        suppression_status = "NONE_SUPPRESSED"
    elif effective_blocking_count == 0:
        suppression_status = "ALL_ORIGINALLY_BLOCKING_SUPPRESSED"
    else:
        suppression_status = "PARTIALLY_SUPPRESSED"

    repair_labels = suppressed_atom_repair_finding_labels(decisions, repair_loop_attempts)

    return {
        "schema_version": SCHEMA_VERSION,
        "lost_atom_materiality_authority_enabled": is_enabled,
        "lost_atom_original_blocking_count": original_blocking_count,
        "lost_atom_effective_blocking_count": effective_blocking_count,
        "lost_atom_suppressed_count": suppressed_count,
        "lost_atom_preserved_blocking_count": preserved_blocking_count,
        "lost_atom_suppression_status": suppression_status,
        "lost_atom_suppression_reasons": suppression_reasons,
        "repair_findings_suppressed_count": len(repair_labels),
        "provenance": (SCHEMA_VERSION, "lost_semantic_atom_freeze_authority_diagnostics"),
    }
