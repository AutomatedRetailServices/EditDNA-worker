"""D-235T: BOUNDED SAME-ATOM REPAIR-LOOP SUPPRESSION ADAPTER -- OFFLINE FIRST.

## Core principle (restated from this task's own directive)

One non-material lost semantic atom must not be allowed to block Freeze
TWICE through two independent paths that both ultimately read the SAME
`_lost_semantic_atoms()` row:

  PATH A -- `final_story_coherence_validation.py`'s own `freeze_blocked`
      aggregate, via `lost_semantic_atom.blocking` -- already fixed,
      D-235R (`lost_semantic_atom_freeze_authority.py`, default-OFF flag
      `CUTSELL_LOST_ATOM_MATERIALITY_FREEZE_AUTHORITY_ENABLED`).
  PATH B -- `final_edit_reviewer.py`'s `UNIQUE_FACT_LOST` Finding (built
      from the exact SAME row, `detail=dict(row)`, D-235S) reaching
      `repair_loop.py::run_repair_loop()`'s "no repair strategy exists"
      branch, which (pre-D-235T) unconditionally escalates to
      `NEEDS_HUMAN_REVIEW` -- THIS is the path this module closes.

D-235R deliberately did not touch RepairLoop ("NO REPAIR-LOOP CHANGE"
banner). D-235S proved the exact provenance link survives into a
`RepairAttempt` (`source_lost_atom_provenance_id`) but performed no
suppression. This module is the first to actually suppress PATH B, and
ONLY when it can independently, deterministically re-prove -- using the
SAME pure D-235Q/D-235R functions, on the SAME row data -- that the
specific atom behind a `UNIQUE_FACT_LOST` Finding is the exact atom
D-235R's own gate would (or does) safely suppress at the Freeze seam.

## Why re-computation, not a passed-in decision, proves "the SAME atom"

`run_repair_loop()` never receives `final_story_coherence_validation.py`'s
own `LostSemanticAtomFreezeAuthorityDecision` objects -- there is no live
call path carrying them that far downstream, and building one would be a
new cross-module correlation channel this task's own scope forbids
("general RepairLoop redesign"). Instead: `final_edit_reviewer.py::
review()`'s own `UNIQUE_FACT_LOST` construction already does `detail=
dict(row)` (D-235S's own confirmed finding) -- so `finding.detail` IS,
byte-for-byte, the exact same row `lost_semantic_atom_freeze_trigger_
present()` would compute a materiality/freeze-authority verdict from at
the coherence seam. Calling the SAME pure, deterministic functions
(`assess_complete_lost_semantic_atom_materiality`, `decide_lost_semantic_
atom_freeze_authority`) again on that identical row produces the
identical verdict -- there is no risk of two independent judgment calls
disagreeing, because it is structurally the same computation on the same
data, not a second opinion. `lost_atom_provenance_id` (D-235S) is used
here as an ADDITIONAL integrity check (via `classify_lost_atom_reviewer_
finding_link`), never as the suppression basis by itself: a missing,
ambiguous, or mismatched provenance id fails closed (ABSTAIN), it never
substitutes for a real materiality/freeze-authority re-verification.

## Absolute firewalls (never suppress)

Reused verbatim from D-235R's own gate (never re-implemented with
different thresholds) via `decide_lost_semantic_atom_freeze_authority`:
`MEANING_CRITICAL`, `EDITORIALLY_REQUIRED`, any `CONFLICTED`/
`INSUFFICIENT_EVIDENCE` verdict, `RECOMMEND_BLOCK`. This module adds its
own, independent firewalls on top: a missing `lost_atom_provenance_id`,
an `AMBIGUOUS`/`NO_MATCH`/`MISSING_PROVENANCE` provenance-link result
(D-235S's own classifier, called here for the first time as an actual
gating input rather than pure observability), a malformed (non-mapping)
`Finding.detail`, and any finding kind other than `UNIQUE_FACT_LOST`
(scoped exactly like D-235S's own bridge).

## Multiple-findings semantics (no majority voting)

`all_blocking_findings_safely_suppressed()` is the ONE entry point
`repair_loop.py` calls. It requires EVERY current blocking finding --
not a majority, not "the first one" -- to independently qualify for
suppression before the loop is allowed to finish without
`NEEDS_HUMAN_REVIEW`. One unrelated or non-qualifying blocking finding
(a different kind, a `MEANING_CRITICAL` atom, an ambiguous provenance
link, ...) preserves the original escalation for the WHOLE batch, exactly
as the directive's own item 19 ("one suppressible + one critical atom ->
preserve overall escalation") requires. This module never attributes
suppression to "most findings agree" -- see `all_blocking_findings_
safely_suppressed`'s own `all(...)` (never `any(...)` or a threshold).

## RepairLoop terminal-vocabulary audit (required before touching the seam)

`repair_loop.py::RepairLoopResult.status` has, and had before this task,
EXACTLY two values: `"PASS"` and `"NEEDS_HUMAN_REVIEW"` (confirmed by
direct code read, not assumed). No third value is invented here. When
every current blocking finding is safely suppressed, the loop reuses the
existing `"PASS"` value -- the narrowest existing non-human-review
terminal status, exactly as the directive requires -- rather than minting
a new status string. Because reusing an existing value for a genuinely
different situation (suppressed vs a real reviewer PASS) would otherwise
be indistinguishable and dishonest (CLAUDE.md: never declare success from
green CI/duration/counts alone), `repair_loop.py` also gains ONE new,
purely additive field, `RepairLoopResult.blocking_findings_suppressed:
bool = False`, defaulting to `False` for every pre-existing code path.
`RepairLoopResult.final_review` (the actual `FinalEditReviewResult`
`review()` returned) is NEVER mutated or replaced by this task -- it
still, honestly, reports `status == "FAIL"` with its own real findings
when suppression fires; only the LOOP's own convenience `status` field
changes. A caller that wants the honest underlying reviewer verdict reads
`final_review.status`; a caller that wants "does a human need to look at
this" reads the loop's own `status` (now correctly `PASS` in this case)
or, for the finer distinction, `blocking_findings_suppressed`.

## No fake repair

Every `RepairAttempt` this module's own seam produces keeps `repaired=
False` -- suppression is never recorded as a repair. `reason` is the
descriptive string `"finding_safely_suppressed_as_non_material_same_
atom"`, distinct from the pre-existing `"no_repair_strategy_exists_for_
this_finding_kind"` used when suppression does not apply.

## Feature flag (SAME as D-235R, no new flag)

`CUTSELL_LOST_ATOM_MATERIALITY_FREEZE_AUTHORITY_ENABLED`, default OFF.
`decide_lost_atom_repair_suppression()` self-gates on this flag (same
idiom as `lost_semantic_atom_freeze_trigger_present`): when OFF, it
ALWAYS returns `NOT_APPLICABLE` / `suppress_repair_escalation=False`,
regardless of the row's own content -- so `all_blocking_findings_safely_
suppressed()` can never return `True` when the flag is off, which is what
keeps `repair_loop.py`'s own seam byte-identical to its pre-D-235T
behavior by construction (proven by this module's own parity tests, not
merely asserted).

## What this module is NOT

- No general Freeze/RepairLoop redesign. Only the exact "no repair
  strategy exists" branch of `run_repair_loop()` is touched; the
  repair-strategy branch (`STORY_ORDER_BREAK`), `CausalOrderArbiter`
  wiring, `max_attempts` bound, and `CanonicalEditPlan`/`review()` calls
  are all unchanged.
- No threshold change, no resolver change, no P1/P2 change, no BestTake/
  Family/Ordering/Boundary change, no Pacing/Audio-Join change. This
  module imports only D-235Q/D-235R/D-235S's own already-existing pure
  functions and `final_edit_reviewer.Finding`/`UNIQUE_FACT_LOST`.
- No new master score. `LostAtomRepairSuppressionDecision` is a
  categorical decision object, exactly like every other D-235 adapter in
  this family -- no numeric aggregate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .complete_lost_semantic_atom_materiality import (
    CompleteLostSemanticAtomMateriality,
    assess_complete_lost_semantic_atom_materiality,
)
from .final_edit_reviewer import UNIQUE_FACT_LOST, Finding
from .lost_atom_reviewer_finding_provenance import (
    LINK_EXACT_MATCH,
    PROVENANCE_FIELD_NAME,
    classify_lost_atom_reviewer_finding_link,
)
from .lost_semantic_atom_freeze_authority import (
    AUTHORITY_ABSTAIN_PRESERVE_BLOCK,
    AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK,
    decide_lost_semantic_atom_freeze_authority,
    lost_atom_materiality_freeze_authority_enabled,
)

SCHEMA_VERSION = "cutsell.lost_atom_repair_suppression.v1"

# ---------------------------------------------------------------------------
# Suppression-status vocabulary (this task's own required 4-value set).
# ---------------------------------------------------------------------------
SUPPRESS_SAME_NON_MATERIAL_ATOM = "SUPPRESS_SAME_NON_MATERIAL_ATOM"
PRESERVE_REPAIR_ESCALATION = "PRESERVE_REPAIR_ESCALATION"
ABSTAIN_PRESERVE_ESCALATION = "ABSTAIN_PRESERVE_ESCALATION"
NOT_APPLICABLE = "NOT_APPLICABLE"

_VALID_SUPPRESSION_STATUSES = frozenset({
    SUPPRESS_SAME_NON_MATERIAL_ATOM, PRESERVE_REPAIR_ESCALATION,
    ABSTAIN_PRESERVE_ESCALATION, NOT_APPLICABLE,
})

# `RepairAttempt`-facing reason strings -- reused verbatim at the
# `repair_loop.py` call site, never re-spelled there.
REASON_SUPPRESSED_SAME_ATOM = "finding_safely_suppressed_as_non_material_same_atom"
REASON_NO_REPAIR_STRATEGY = "no_repair_strategy_exists_for_this_finding_kind"

# `repair_attempt_status` vocabulary -- documents the (today, always true)
# precondition that this adapter is only ever consulted for a finding kind
# with no automatic repair strategy in `_REPAIR_STRATEGIES`.
NO_REPAIR_STRATEGY_EXISTS = "NO_REPAIR_STRATEGY_EXISTS"


@dataclass(frozen=True)
class LostAtomRepairSuppressionDecision:
    lost_atom_provenance_id: Optional[str]
    reviewer_finding_kind: Optional[str]
    repair_attempt_status: Optional[str]
    freeze_authority_status: Optional[str]
    materiality_status: Optional[str]
    suppression_status: str
    suppress_repair_escalation: bool
    reason: str
    provenance: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.suppression_status not in _VALID_SUPPRESSION_STATUSES:
            raise ValueError(f"invalid suppression_status: {self.suppression_status!r}")

    def as_dict(self) -> dict:
        return {
            "lost_atom_provenance_id": self.lost_atom_provenance_id,
            "reviewer_finding_kind": self.reviewer_finding_kind,
            "repair_attempt_status": self.repair_attempt_status,
            "freeze_authority_status": self.freeze_authority_status,
            "materiality_status": self.materiality_status,
            "suppression_status": self.suppression_status,
            "suppress_repair_escalation": self.suppress_repair_escalation,
            "reason": self.reason,
            "provenance": list(self.provenance),
        }


def _not_applicable(finding: Finding, *, reason: str) -> LostAtomRepairSuppressionDecision:
    row = finding.detail if isinstance(finding.detail, Mapping) else {}
    return LostAtomRepairSuppressionDecision(
        lost_atom_provenance_id=row.get(PROVENANCE_FIELD_NAME),
        reviewer_finding_kind=finding.kind,
        repair_attempt_status=None,
        freeze_authority_status=None,
        materiality_status=None,
        suppression_status=NOT_APPLICABLE,
        suppress_repair_escalation=False,
        reason=reason,
        provenance=(SCHEMA_VERSION, "decide_lost_atom_repair_suppression"),
    )


def _abstain(
    finding: Finding,
    *,
    provenance_id: Optional[str],
    materiality_status: Optional[str],
    freeze_authority_status: Optional[str] = None,
    reason: str,
) -> LostAtomRepairSuppressionDecision:
    return LostAtomRepairSuppressionDecision(
        lost_atom_provenance_id=provenance_id,
        reviewer_finding_kind=finding.kind,
        repair_attempt_status=NO_REPAIR_STRATEGY_EXISTS,
        freeze_authority_status=freeze_authority_status,
        materiality_status=materiality_status,
        suppression_status=ABSTAIN_PRESERVE_ESCALATION,
        suppress_repair_escalation=False,
        reason=reason,
        provenance=(SCHEMA_VERSION, "decide_lost_atom_repair_suppression"),
    )


def decide_lost_atom_repair_suppression(
    finding: Finding,
    *,
    all_findings: Optional[Sequence[Finding]] = None,
    enabled: Optional[bool] = None,
    materiality_by_provenance_id: Optional[Mapping[str, CompleteLostSemanticAtomMateriality]] = None,
) -> LostAtomRepairSuppressionDecision:
    """The one D-235T entry point for a single Finding. Pure; mutates
    nothing, mints nothing, calls only already-existing D-235Q/D-235R/
    D-235S pure functions. See module docstring for the full gate.

    `all_findings`: the SAME `result.findings` tuple the caller is
    iterating (used only for the D-235S provenance-uniqueness re-check,
    `classify_lost_atom_reviewer_finding_link`'s own AMBIGUOUS detection).
    Defaults to `(finding,)` when omitted -- correct for a lone finding,
    but a real caller with multiple findings MUST pass the full set, or
    an accidental duplicate provenance id across findings would not be
    caught.

    `materiality_by_provenance_id` (D-235X): an optional, caller-supplied
    `lost_atom_provenance_id -> CompleteLostSemanticAtomMateriality` map --
    the SAME already-computed D-235Q result `final_story_coherence_
    validation.py`'s own Freeze-composition seam produced (see
    `DraftTimeline.lost_atom_materiality_by_provenance_id`), so D-235R and
    D-235T read the identical result instead of D-235T independently
    recomputing an incomplete, row-only materiality. `None` (the default,
    every pre-D-235X caller) preserves the exact prior behavior: fresh
    row-only recomputation below. Purely an INPUT-SOURCE substitution --
    the suppression GATE itself (which statuses suppress, the firewalls,
    the feature flag, the provenance-uniqueness requirement) is completely
    unchanged either way; this task's own directive explicitly authorizes
    only this substitution, nothing else."""
    is_enabled = enabled if enabled is not None else lost_atom_materiality_freeze_authority_enabled()
    if not is_enabled:
        return _not_applicable(finding, reason="lost_atom_materiality_freeze_authority_disabled")

    if finding.kind != UNIQUE_FACT_LOST:
        return _not_applicable(
            finding,
            reason=f"suppression_authorized_only_for_{UNIQUE_FACT_LOST}_not_{finding.kind}",
        )

    row = finding.detail
    if not isinstance(row, Mapping):
        return _abstain(finding, provenance_id=None, materiality_status=None,
                         reason="finding_detail_malformed_not_a_mapping")

    provenance_id = row.get(PROVENANCE_FIELD_NAME)
    if not provenance_id:
        return _abstain(finding, provenance_id=None, materiality_status=None,
                         reason="lost_atom_provenance_id_missing")

    clip_id = str(row.get("clip_id") or (finding.clip_ids[0] if finding.clip_ids else ""))
    candidates = tuple(all_findings) if all_findings is not None else (finding,)
    link = classify_lost_atom_reviewer_finding_link(
        lost_atom_provenance_id=provenance_id,
        clip_id=clip_id,
        findings=candidates,
        target_finding_kind=UNIQUE_FACT_LOST,
    )
    if link.link_status != LINK_EXACT_MATCH:
        return _abstain(
            finding, provenance_id=provenance_id, materiality_status=None,
            reason=f"provenance_link_{link.link_status.lower()}",
        )

    # D-235X: prefer the SAME already-computed D-235Q result the Freeze-
    # composition seam produced (keyed by this exact `lost_atom_
    # provenance_id`) over an incomplete row-only recomputation -- an
    # INPUT-SOURCE substitution only, never a gate change (see this
    # function's own docstring). Falls through to the unchanged pre-
    # D-235X recompute whenever no map was supplied, or it has no entry
    # for this provenance id (never a partial/best-effort merge of the
    # two sources).
    precomputed = (materiality_by_provenance_id or {}).get(provenance_id)
    if isinstance(precomputed, CompleteLostSemanticAtomMateriality):
        materiality = precomputed
    else:
        try:
            materiality = assess_complete_lost_semantic_atom_materiality(row)
        except Exception:
            # Fail-closed: never suppress on a malformed/exception-raising row.
            return _abstain(finding, provenance_id=provenance_id, materiality_status=None,
                             reason="materiality_assessment_raised_fail_closed")

    if not isinstance(materiality, CompleteLostSemanticAtomMateriality):
        return _abstain(finding, provenance_id=provenance_id, materiality_status=None,
                         reason="materiality_result_missing_or_malformed")

    freeze_decision = decide_lost_semantic_atom_freeze_authority(row, materiality)

    if freeze_decision.authority_status == AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK:
        return LostAtomRepairSuppressionDecision(
            lost_atom_provenance_id=provenance_id,
            reviewer_finding_kind=finding.kind,
            repair_attempt_status=NO_REPAIR_STRATEGY_EXISTS,
            freeze_authority_status=freeze_decision.authority_status,
            materiality_status=materiality.final_materiality_status,
            suppression_status=SUPPRESS_SAME_NON_MATERIAL_ATOM,
            suppress_repair_escalation=True,
            reason="same_atom_already_qualifies_for_freeze_authority_suppression",
            provenance=(SCHEMA_VERSION, "decide_lost_atom_repair_suppression"),
        )

    if freeze_decision.authority_status == AUTHORITY_ABSTAIN_PRESERVE_BLOCK:
        return _abstain(
            finding, provenance_id=provenance_id,
            materiality_status=materiality.final_materiality_status,
            freeze_authority_status=freeze_decision.authority_status,
            reason=freeze_decision.safety_block_reason or "freeze_authority_abstained",
        )

    # AUTHORITY_PRESERVE_BLOCK (the meaning-critical/editorially-required/
    # BLOCK firewalls), or AUTHORITY_NOT_APPLICABLE (defensive only -- a
    # Finding present in `result.findings` at all implies its own row's
    # `blocking` was True, per `final_edit_reviewer.review()`'s own
    # findings/warnings split, so this branch should not occur in practice
    # but is never treated as a suppression path if it somehow does).
    return LostAtomRepairSuppressionDecision(
        lost_atom_provenance_id=provenance_id,
        reviewer_finding_kind=finding.kind,
        repair_attempt_status=NO_REPAIR_STRATEGY_EXISTS,
        freeze_authority_status=freeze_decision.authority_status,
        materiality_status=materiality.final_materiality_status,
        suppression_status=PRESERVE_REPAIR_ESCALATION,
        suppress_repair_escalation=False,
        reason=freeze_decision.safety_block_reason or "freeze_authority_preserved_block",
        provenance=(SCHEMA_VERSION, "decide_lost_atom_repair_suppression"),
    )


def all_blocking_findings_safely_suppressed(
    findings: Sequence[Finding],
    *,
    enabled: Optional[bool] = None,
    materiality_by_provenance_id: Optional[Mapping[str, CompleteLostSemanticAtomMateriality]] = None,
) -> Tuple[bool, Tuple[LostAtomRepairSuppressionDecision, ...]]:
    """The one entry point `repair_loop.py` calls. Requires EVERY finding
    in `findings` (never a majority, never just the first) to
    independently qualify for suppression -- see module docstring's
    "Multiple-findings semantics" section. Returns `(False, ())` for an
    empty `findings` (nothing to suppress, nothing to preserve either --
    the caller's own pre-existing "no findings" handling is untouched).

    `materiality_by_provenance_id` (D-235X): forwarded unchanged to every
    `decide_lost_atom_repair_suppression` call below -- see that
    function's own identically-named parameter docstring. `None` (the
    default, every pre-D-235X caller) preserves the exact prior
    behavior."""
    findings = tuple(findings)
    if not findings:
        return False, ()
    decisions = tuple(
        decide_lost_atom_repair_suppression(
            f, all_findings=findings, enabled=enabled,
            materiality_by_provenance_id=materiality_by_provenance_id,
        )
        for f in findings
    )
    return all(d.suppress_repair_escalation for d in decisions), decisions


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe, no transcript dump).
# ---------------------------------------------------------------------------
def lost_atom_repair_suppression_diagnostics(
    *,
    decisions: Optional[Sequence[LostAtomRepairSuppressionDecision]] = None,
    all_suppressed: Optional[bool] = None,
    enabled: Optional[bool] = None,
    materiality_by_provenance_id: Optional[Mapping[str, CompleteLostSemanticAtomMateriality]] = None,
) -> dict:
    decisions = tuple(decisions or ())
    return {
        "enabled": enabled if enabled is not None else lost_atom_materiality_freeze_authority_enabled(),
        "decision_count": len(decisions),
        "suppressed_count": sum(1 for d in decisions if d.suppress_repair_escalation),
        "all_suppressed": bool(all_suppressed) if all_suppressed is not None else (
            bool(decisions) and all(d.suppress_repair_escalation for d in decisions)
        ),
        "decisions": [d.as_dict() for d in decisions],
        # D-235X diagnostics (tail-safe counts only, no transcript dump):
        # whether this call was fed the D-235X precomputed compute-once
        # context at all, and how many of THIS batch's decisions actually
        # suppressed via `SUPPRESS_SAME_NON_MATERIAL_ATOM` (a strict
        # subset of `suppressed_count` above, which also counts pre-D-235X
        # row-only suppressions).
        "repair_context_received_count": len(materiality_by_provenance_id or {}),
        "repair_same_atom_suppressed_count": sum(
            1 for d in decisions if d.suppression_status == SUPPRESS_SAME_NON_MATERIAL_ATOM
        ),
        "provenance": (SCHEMA_VERSION, "lost_atom_repair_suppression_diagnostics"),
    }
