"""D-235G: Selection Freeze Blocker Observability, OBSERVABILITY ONLY.

Surfaces the ALREADY-COMPUTED Selection Freeze decision from
`universal_clean_cut.py` -- and the already-computed downstream Pacing V2
/ Audio Join Treatment seam reachability -- into one small, bounded,
JSON-safe diagnostic block, `diagnostics["selection_freeze_diagnostics"]`.

## What this module is NOT (binding, restated from this task's own scope)

- No Freeze decision logic. `build_selection_freeze_diagnostics()` takes
  the CALLER'S already-final `freeze_blocked` boolean and the caller's
  already-computed evidence dicts/statuses as plain arguments; it never
  re-derives, re-evaluates, or second-guesses whether Freeze SHOULD have
  blocked. Changing this module can never change whether a real run's
  Freeze blocks, what constitutes a contradiction or idea loss, the
  repair loop's outcome, resolver authority, or D-090 integrity behavior.
- No Video00-specific content. Every field here is generic across any
  source: no hardcoded expected-selection-count, no golden-file semantic
  comparison, no Video00 transcript assertion. Contrast this deliberately
  with `benchmarks/validate_video00_selection_lock.py` (compares against
  Video00's own golden `segments`), `benchmarks/validate_video00_
  architecture.py` (gated behind a Video00-tuned `source_duration_sec >
  350` threshold in the calling workflow step), and `benchmarks/
  validate_video00_regression_qa.py` (an 18-check Video00 regression
  manifest) -- all three are VIDEO00_SPECIFIC_ORACLEs (D-235F's own
  finding) that this module never touches, never replaces, and is never
  confused with. They remain useful for Video00; this module is the
  sibling-safe complement.
- No transcript/large-object dump. Only bounded counts, booleans, and
  short category-code strings are ever included.

## The real trigger vocabulary (D-235F's own audit, unchanged here)

`universal_clean_cut.py`'s `freeze_blocked` is the OR of four top-level
trigger families, each surfaced here under its OWN real code name/status,
never renamed to fit this task's own vocabulary:

1. `coherence_diag["freeze_blocked"]` (`final_story_coherence_validation`
   -- StoryValidator, D-020/D-050C2), itself the OR of five sub-reasons
   surfaced individually below: `contradiction_findings`, `missing_idea_
   coverage`, a blocking row in `lost_semantic_atoms`, `lost_critical_
   claims`, and `authority_membership_findings` -- plus a `status ==
   "integrity_failure"` shape (missing post-authority context).
2. `repair_result.status == "NEEDS_HUMAN_REVIEW"` (the bounded repair
   loop's own terminal status).
3. `authoritative_result.status == AUTHORITATIVE_REVIEW_REQUIRED`
   (`realization_resolver.py`'s own literal `"REVIEW_REQUIRED"`).
4. `post_authority_integrity_failed` (D-090), with its own failure codes
   (e.g. `POST_AUTHORITY_CONTEXT_MISSING`, `POST_AUTHORITY_SELECTION_
   MUTATION` from `post_authority_validation.py`) surfaced verbatim.

## Downstream Pacing seam reachability

`pacing_seam_reached` is simply `not freeze_blocked` -- the SAME already-
computed boolean the caller passes in, never a second computation of the
same question. `pacing_v2_serialized` / `pacing_v2_handle_aware_
serialized` / `audio_join_treatment_v2_serialized` are pure KEY-PRESENCE
observations on the caller's own already-finalized `draft.diagnostics`
dict -- this module never imports or calls any of D-216/D-217/D-224/D-234's
own construction functions, and never influences whether those keys
exist.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

SCHEMA_VERSION = "cutsell.selection_freeze_diagnostics.v1"

# Trigger category codes -- real code names, not renamed for this task.
TRIGGER_COHERENCE_CONTRADICTION = "COHERENCE_CONTRADICTION_FINDINGS"
TRIGGER_COHERENCE_MISSING_IDEA_COVERAGE = "COHERENCE_MISSING_IDEA_COVERAGE"
TRIGGER_COHERENCE_BLOCKING_LOST_SEMANTIC_ATOM = "COHERENCE_BLOCKING_LOST_SEMANTIC_ATOM"
TRIGGER_COHERENCE_LOST_CRITICAL_CLAIM = "COHERENCE_LOST_CRITICAL_CLAIM"
TRIGGER_COHERENCE_AUTHORITY_MEMBERSHIP_FINDING = "COHERENCE_AUTHORITY_MEMBERSHIP_FINDING"
TRIGGER_COHERENCE_INTEGRITY_FAILURE = "COHERENCE_VALIDATION_INTEGRITY_FAILURE"
TRIGGER_REPAIR_LOOP_NEEDS_HUMAN_REVIEW = "REPAIR_LOOP_NEEDS_HUMAN_REVIEW"
TRIGGER_RESOLVER_AUTHORITATIVE_REVIEW_REQUIRED = "RESOLVER_AUTHORITATIVE_REVIEW_REQUIRED"
TRIGGER_POST_AUTHORITY_INTEGRITY_FAILURE = "POST_AUTHORITY_INTEGRITY_FAILURE"

_ALL_TRIGGER_CODES = (
    TRIGGER_COHERENCE_CONTRADICTION,
    TRIGGER_COHERENCE_MISSING_IDEA_COVERAGE,
    TRIGGER_COHERENCE_BLOCKING_LOST_SEMANTIC_ATOM,
    TRIGGER_COHERENCE_LOST_CRITICAL_CLAIM,
    TRIGGER_COHERENCE_AUTHORITY_MEMBERSHIP_FINDING,
    TRIGGER_COHERENCE_INTEGRITY_FAILURE,
    TRIGGER_REPAIR_LOOP_NEEDS_HUMAN_REVIEW,
    TRIGGER_RESOLVER_AUTHORITATIVE_REVIEW_REQUIRED,
    TRIGGER_POST_AUTHORITY_INTEGRITY_FAILURE,
)

# Tri-state vocabulary for each sub-reason -- UNKNOWN when the supporting
# detail was never retained/observed, never fabricated as False.
STATE_FOUND = "FOUND"
STATE_NOT_FOUND = "NOT_FOUND"
STATE_UNKNOWN = "UNKNOWN"

RESOLVER_STATUS_AUTHORITATIVE_REVIEW_REQUIRED = "REVIEW_REQUIRED"
REPAIR_STATUS_NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"
COHERENCE_STATUS_INTEGRITY_FAILURE = "integrity_failure"

# First-missing-link bounded vocabulary.
LINK_FREEZE_BLOCKED_BEFORE_PACING = "FREEZE_BLOCKED_BEFORE_PACING"
LINK_PACING_SEAM_REACHED_DIAGNOSTIC_MISSING = "PACING_SEAM_REACHED_DIAGNOSTIC_MISSING"
LINK_PACING_DIAGNOSTIC_SERIALIZED = "PACING_DIAGNOSTIC_SERIALIZED"


def _tri_state(value) -> str:
    """`value` is `None`/absent -> UNKNOWN; a real bool -> FOUND/NOT_FOUND.
    Never collapses "never observed" into "not found"."""
    if value is None:
        return STATE_UNKNOWN
    return STATE_FOUND if bool(value) else STATE_NOT_FOUND


def _coherence_sub_reasons(coherence_diag: Mapping) -> dict:
    """Reads ONLY already-computed fields from `final_story_coherence_
    validation`'s own real diagnostics dict (`final_story_coherence_
    validation.py`'s `_apply_post_authority_validation_only` /
    `apply_post_authority_story_validation`'s integrity-failure shape).
    A field's ABSENCE from a non-empty `coherence_diag` (schema drift) or
    a completely empty `coherence_diag` (coherence validation never ran
    at this seam, or Freeze was blocked for a reason outside coherence
    entirely) both report UNKNOWN, never NOT_FOUND -- this task's own
    "report UNKNOWN, not False" instruction."""
    if not coherence_diag:
        return {
            "coherence_status": None,
            "coherence_contradiction_status": STATE_UNKNOWN,
            "idea_loss_status": STATE_UNKNOWN,
            "lost_semantic_atom_status": STATE_UNKNOWN,
            "lost_critical_claim_status": STATE_UNKNOWN,
            "authority_membership_finding_status": STATE_UNKNOWN,
            "coherence_integrity_failure_status": STATE_UNKNOWN,
        }
    status = coherence_diag.get("status")
    contradiction_findings = coherence_diag.get("contradiction_findings")
    missing_idea_coverage = coherence_diag.get("missing_idea_coverage")
    lost_semantic_atoms = coherence_diag.get("lost_semantic_atoms")
    lost_critical_claims = coherence_diag.get("lost_critical_claims")
    authority_membership_findings = coherence_diag.get("authority_membership_findings")

    blocking_atom = None
    if lost_semantic_atoms is not None:
        blocking_atom = any(
            isinstance(row, Mapping) and row.get("blocking", True) for row in lost_semantic_atoms
        )

    return {
        "coherence_status": status,
        "coherence_contradiction_status": (
            _tri_state(bool(contradiction_findings)) if contradiction_findings is not None else STATE_UNKNOWN
        ),
        "idea_loss_status": (
            _tri_state(bool(missing_idea_coverage)) if missing_idea_coverage is not None else STATE_UNKNOWN
        ),
        "lost_semantic_atom_status": _tri_state(blocking_atom),
        "lost_critical_claim_status": (
            _tri_state(bool(lost_critical_claims)) if lost_critical_claims is not None else STATE_UNKNOWN
        ),
        "authority_membership_finding_status": (
            _tri_state(bool(authority_membership_findings)) if authority_membership_findings is not None else STATE_UNKNOWN
        ),
        "coherence_integrity_failure_status": _tri_state(status == COHERENCE_STATUS_INTEGRITY_FAILURE),
    }


def build_selection_freeze_diagnostics(
    *,
    freeze_blocked: bool,
    coherence_diag: Optional[Mapping] = None,
    repair_loop_status: Optional[str] = None,
    resolver_status: Optional[str] = None,
    post_authority_integrity_failed: Optional[bool] = None,
    post_authority_integrity_failure_codes: Sequence[str] = (),
    selected_count_before_freeze: Optional[int] = None,
    pacing_v2_serialized: Optional[bool] = None,
    pacing_v2_handle_aware_serialized: Optional[bool] = None,
    audio_join_treatment_v2_serialized: Optional[bool] = None,
) -> dict:
    """The one D-235G entry point. Pure function of already-computed real
    Freeze-decision state -- builds no new evidence, recomputes no
    boolean, decides nothing. `freeze_blocked` is the SAME already-final
    value `universal_clean_cut.py` already uses to gate the Pacing seam;
    `pacing_seam_reached` below is its plain negation, not a second
    computation of the same question."""
    coherence_diag = coherence_diag or {}
    sub_reasons = _coherence_sub_reasons(coherence_diag)

    trigger_categories = []
    if sub_reasons["coherence_contradiction_status"] == STATE_FOUND:
        trigger_categories.append(TRIGGER_COHERENCE_CONTRADICTION)
    if sub_reasons["idea_loss_status"] == STATE_FOUND:
        trigger_categories.append(TRIGGER_COHERENCE_MISSING_IDEA_COVERAGE)
    if sub_reasons["lost_semantic_atom_status"] == STATE_FOUND:
        trigger_categories.append(TRIGGER_COHERENCE_BLOCKING_LOST_SEMANTIC_ATOM)
    if sub_reasons["lost_critical_claim_status"] == STATE_FOUND:
        trigger_categories.append(TRIGGER_COHERENCE_LOST_CRITICAL_CLAIM)
    if sub_reasons["authority_membership_finding_status"] == STATE_FOUND:
        trigger_categories.append(TRIGGER_COHERENCE_AUTHORITY_MEMBERSHIP_FINDING)
    if sub_reasons["coherence_integrity_failure_status"] == STATE_FOUND:
        trigger_categories.append(TRIGGER_COHERENCE_INTEGRITY_FAILURE)
    if repair_loop_status == REPAIR_STATUS_NEEDS_HUMAN_REVIEW:
        trigger_categories.append(TRIGGER_REPAIR_LOOP_NEEDS_HUMAN_REVIEW)
    if resolver_status == RESOLVER_STATUS_AUTHORITATIVE_REVIEW_REQUIRED:
        trigger_categories.append(TRIGGER_RESOLVER_AUTHORITATIVE_REVIEW_REQUIRED)
    if post_authority_integrity_failed:
        trigger_categories.append(TRIGGER_POST_AUTHORITY_INTEGRITY_FAILURE)

    pacing_seam_reached = not freeze_blocked

    if freeze_blocked:
        first_missing_link = LINK_FREEZE_BLOCKED_BEFORE_PACING
    elif pacing_v2_serialized is False or audio_join_treatment_v2_serialized is False:
        # Seam was reachable (Freeze not blocked) but a diagnostic block
        # this run expected is nonetheless absent -- e.g. the relevant
        # CUTSELL_*_DIAGNOSTICS_ENABLED flag was off for this run, or a
        # real gap. Never conflated with a Freeze block.
        first_missing_link = LINK_PACING_SEAM_REACHED_DIAGNOSTIC_MISSING
    else:
        first_missing_link = LINK_PACING_DIAGNOSTIC_SERIALIZED

    return {
        "schema_version": SCHEMA_VERSION,
        "freeze_blocked": bool(freeze_blocked),
        "trigger_count": len(trigger_categories),
        "trigger_categories": trigger_categories,
        **sub_reasons,
        "repair_loop_status": repair_loop_status,
        "resolver_status": resolver_status,
        "post_authority_integrity_status": _tri_state(post_authority_integrity_failed),
        "post_authority_integrity_failure_codes": list(post_authority_integrity_failure_codes or ()),
        "selected_count_before_freeze": selected_count_before_freeze,
        "pacing_seam_reached": pacing_seam_reached,
        "pacing_v2_serialized": pacing_v2_serialized,
        "pacing_v2_handle_aware_serialized": pacing_v2_handle_aware_serialized,
        "audio_join_treatment_v2_serialized": audio_join_treatment_v2_serialized,
        "first_missing_link": first_missing_link,
        "provenance": (SCHEMA_VERSION, "build_selection_freeze_diagnostics"),
    }
