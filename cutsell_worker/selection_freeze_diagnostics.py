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

# D-235J: sibling schema version for the lost-semantic-atom DETAIL block.
# Separate top-level diagnostics key (`lost_semantic_atom_diagnostics`),
# never nested inside `selection_freeze_diagnostics` -- keeps that block's
# own existing small-size test unaffected and keeps the two concerns
# (Freeze-trigger shape vs. lost-atom CONTENT) independently readable.
LOST_ATOM_SCHEMA_VERSION = "cutsell.lost_semantic_atom_diagnostics.v1"

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


# ---------------------------------------------------------------------------
# D-235J: Lost Semantic Atom DETAIL observability, OBSERVABILITY ONLY.
# ---------------------------------------------------------------------------
#
# `build_lost_semantic_atom_diagnostics()` exposes the exact ALREADY-COMPUTED
# per-atom rows `final_story_coherence_validation.py::_lost_semantic_atoms()`
# already built and `canonical_edit_plan.py` already threaded onto
# `CanonicalEditPlan.lost_semantic_atoms` -- the SAME rows `final_edit_
# reviewer.py::review()` reads to mint `UNIQUE_FACT_LOST` Findings. This
# function recomputes NOTHING: no semantic-atom detection, no materiality
# judgment (reserved for a future, separately-authorized D-235K), no
# `blocking` flag change. It is a pure, bounded, JSON-safe re-projection of
# fields that already exist on the caller's own already-final data.
#
# Real, retained fields this function reads verbatim from each row (see
# `_lost_semantic_atoms()`'s own two row shapes):
#   - the general coverage-ledger shape: `clip_id`, `text`, `missing_
#     critical_atoms`, `atom_classifications` (`atom`/`atom_type`/
#     `importance`/`evidence`/`resolved_by` each), `missing_content_token_
#     count`, `own_content_token_count`, `coverage_against_final_keep`,
#     `blocking`, `classification`, optional `content_loss_suppressed_by`,
#     `pre_group_restart_consultations`, `preserving_realization_id`,
#     `preserved_claim_ids`, `nonrequired_omissions`;
#   - the no-usable-realization shape: `clip_id`, `text` (already
#     pre-truncated to 200 chars by the source function), `kind` ==
#     "LOST_IN_NO_USABLE_REALIZATION_FAMILY", `basis`, `blocking` (always
#     False for this shape).
#
# Fields the D-235J directive suggested that this engine's CURRENT data
# model does NOT retain -- listed honestly rather than invented, per the
# directive's own "Do NOT invent fields that do not exist" instruction.
# Their absence is a genuine finding of this task, surfaced verbatim in
# every returned block as `absent_fields_not_retained_by_engine` so a
# reader never has to re-derive it from a docstring:
ABSENT_FIELDS_NOT_RETAINED_BY_ENGINE = (
    "atom_id",              # no stable per-atom identifier is minted; a row
                             # is identified only by its (clip_id, atom text)
                             # pair, reconstructed positionally, never a
                             # persisted id.
    "source_span_id",       # no span/offset identity into the source
                             # transcript is retained -- only the clip's own
                             # `text` (full utterance) and, for missing
                             # critical atoms, the literal missing token/
                             # phrase string itself.
    "source_proposition_id", # no link to a proposition/claim identity object
                             # is retained for THIS ledger (contrast
                             # `_lost_critical_claims`, a separate, claim-
                             # scoped check with its own `idea_id`, not
                             # threaded through this ledger's own rows).
    "semantic_role",         # no explicit semantic-role tag (e.g. subject/
                             # predicate/qualifier) is retained; the closest
                             # real field is `atom_type` (NUMBER/NEGATION)
                             # on `atom_classifications` entries, which is a
                             # narrower, syntactic-not-semantic distinction.
    "required_or_optional",  # no explicit REQUIRED/OPTIONAL tag is stored;
                             # the closest real fields are `importance`
                             # (CRITICAL/UNCERTAIN/CONTEXTUAL, on missing
                             # critical atoms only) and the row-level
                             # `blocking` boolean (which folds importance
                             # and the broader content-loss signal into one
                             # bit) -- neither is a direct required/optional
                             # classification.
)

_TEXT_EXCERPT_MAX_CHARS = 160
_ATOM_TEXT_MAX_CHARS = 40
_MAX_ATOMS_SERIALIZED = 25

# Repair-loop linkage vocabulary -- never reconstructed from text similarity,
# only ever a real id match: `finding_kind == "UNIQUE_FACT_LOST"` and this
# row's own real `clip_id` appearing in that attempt's real `previous_
# realization` tuple (both already-persisted, exact identifiers).
REPAIR_LINK_NOT_DIRECTLY_ATTEMPTED = "NOT_DIRECTLY_ATTEMPTED_THIS_RUN"
REVIEWER_FINDING_KIND_UNIQUE_FACT_LOST = "UNIQUE_FACT_LOST"


def _bounded_text(text: Optional[str], limit: int) -> Optional[str]:
    """Smallest bounded phrase, hard length cap, never a transcript dump.
    `None`/empty input stays `None` -- never fabricated as an empty string
    standing in for "no text retained"."""
    if not text:
        return None
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"  # single-char ellipsis, not "..."


def _atom_classification_row(entry: Mapping) -> dict:
    """Bounds one `atom_classifications` entry. `atom` (the literal missing
    token/phrase, e.g. a number or negation word) is short by construction
    but still capped; `evidence` is free text from the classifier and is
    reported as presence-only (`evidence_present`), never dumped verbatim --
    consistent with this module's "no transcript/large-object dump" rule."""
    return {
        "atom": _bounded_text(entry.get("atom"), _ATOM_TEXT_MAX_CHARS),
        "atom_type": entry.get("atom_type"),
        "importance": entry.get("importance"),
        "resolved_by": entry.get("resolved_by"),
        "evidence_present": bool(entry.get("evidence")),
    }


def _find_repair_link(clip_id: str, repair_loop_attempts: Sequence[Mapping]) -> dict:
    """Real-id-only linkage to a repair-loop attempt: `finding_kind ==
    UNIQUE_FACT_LOST` AND `clip_id` appears in that attempt's own real
    `previous_realization` tuple. Never a text-similarity reconstruction.
    `repair_loop.py::run_repair_loop` only ever records ONE attempt row per
    loop iteration when no repairable finding exists (`result.findings[0]`)
    -- so most `UNIQUE_FACT_LOST` rows genuinely have NO matching attempt
    even on a run where the repair loop ran and returned NEEDS_HUMAN_REVIEW;
    that is reported honestly as `NOT_DIRECTLY_ATTEMPTED_THIS_RUN`, never
    inferred or defaulted to the one attempt that does exist for an
    unrelated clip."""
    for index, attempt in enumerate(repair_loop_attempts):
        if not isinstance(attempt, Mapping):
            continue
        if attempt.get("finding_kind") != REVIEWER_FINDING_KIND_UNIQUE_FACT_LOST:
            continue
        previous_realization = attempt.get("previous_realization") or ()
        if clip_id in previous_realization:
            return {
                "repair_loop_attempt_status": "MATCHED_BY_CLIP_ID",
                "repair_loop_attempt_index": index,
                "repair_loop_reason": attempt.get("reason"),
                "repair_loop_repaired": bool(attempt.get("repaired")),
            }
    return {
        "repair_loop_attempt_status": REPAIR_LINK_NOT_DIRECTLY_ATTEMPTED,
        "repair_loop_attempt_index": None,
        "repair_loop_reason": None,
        "repair_loop_repaired": None,
    }


def _one_lost_atom_row(row: Mapping, repair_loop_attempts: Sequence[Mapping]) -> dict:
    clip_id = str(row.get("clip_id") or "")
    blocking = bool(row.get("blocking", True))
    kind = row.get("kind")  # "LOST_IN_NO_USABLE_REALIZATION_FAMILY" or absent
    atom_classifications = row.get("atom_classifications") or ()
    preserved_claim_ids = row.get("preserved_claim_ids") or ()
    nonrequired_omissions = row.get("nonrequired_omissions") or ()

    out = {
        "clip_id": clip_id,
        "row_kind": kind if kind else "COVERAGE_LEDGER_CONTENT_LOSS",
        "blocking": blocking,
        "classification": row.get("classification"),
        "text_excerpt": _bounded_text(row.get("text"), _TEXT_EXCERPT_MAX_CHARS),
        "missing_critical_atom_count": len(row.get("missing_critical_atoms") or ()),
        "atom_classifications": [
            _atom_classification_row(entry) for entry in atom_classifications if isinstance(entry, Mapping)
        ],
        "own_content_token_count": row.get("own_content_token_count"),
        "missing_content_token_count": row.get("missing_content_token_count"),
        "coverage_against_final_keep": row.get("coverage_against_final_keep"),
        "content_loss_suppressed_by": row.get("content_loss_suppressed_by"),
        "preserving_realization_id": row.get("preserving_realization_id"),
        "preserved_claim_count": len(preserved_claim_ids),
        "nonrequired_omission_count": len(nonrequired_omissions),
        "no_usable_realization_basis": row.get("basis") if kind else None,
        # Structurally true/false BY CONSTRUCTION for every row in this
        # ledger -- every row originates from `draft.discarded` (a clip
        # that existed as a real candidate before Selection Freeze ran and
        # was not carried into the final KEEP timeline). Never a second
        # membership check; restated here only because the directive asked
        # for it explicitly.
        "present_before_selection": True,
        "present_after_selection": False,
        # `final_edit_reviewer.py::review()` maps EVERY row in
        # `edit_plan.lost_semantic_atoms` to exactly one `UNIQUE_FACT_LOST`
        # Finding (1:1, by construction, never conditional) -- so this
        # linkage is a structural fact about the current code, not a
        # runtime lookup that could fail to match.
        "reviewer_finding_kind": REVIEWER_FINDING_KIND_UNIQUE_FACT_LOST,
    }
    out.update(_find_repair_link(clip_id, repair_loop_attempts))
    return out


def build_lost_semantic_atom_diagnostics(
    *,
    lost_semantic_atoms: Optional[Sequence[Mapping]] = None,
    repair_loop_attempts: Optional[Sequence[Mapping]] = None,
) -> dict:
    """The one D-235J entry point. Pure function of the caller's own
    already-computed `coherence_diag.get("lost_semantic_atoms")` rows and
    the caller's own already-serialized `diagnostics["repair_loop"]["
    attempts"]` list -- both already-final data this module never
    recomputes, reorders, or reinterprets. Bounded to at most
    `_MAX_ATOMS_SERIALIZED` rows (real overflow is reported via
    `atoms_truncated` / `atom_count`, never silently dropped without a
    marker); no full transcript text is ever included, only short bounded
    excerpts. `lost_semantic_atoms=None` (the coherence stage never ran, or
    Freeze was blocked before this seam) reports `atom_count=0` with an
    explicit `ledger_status=UNKNOWN` rather than fabricating "no atoms
    lost" -- consistent with this module's own tri-state discipline."""
    repair_loop_attempts = tuple(repair_loop_attempts or ())
    if lost_semantic_atoms is None:
        return {
            "schema_version": LOST_ATOM_SCHEMA_VERSION,
            "ledger_status": STATE_UNKNOWN,
            "atom_count": 0,
            "blocking_atom_count": 0,
            "atoms_truncated": False,
            "atoms": [],
            "absent_fields_not_retained_by_engine": list(ABSENT_FIELDS_NOT_RETAINED_BY_ENGINE),
            "provenance": (LOST_ATOM_SCHEMA_VERSION, "build_lost_semantic_atom_diagnostics"),
        }

    rows = [row for row in lost_semantic_atoms if isinstance(row, Mapping)]
    atom_count = len(rows)
    blocking_atom_count = sum(1 for row in rows if bool(row.get("blocking", True)))
    truncated_rows = rows[:_MAX_ATOMS_SERIALIZED]
    atoms_truncated = atom_count > len(truncated_rows)

    return {
        "schema_version": LOST_ATOM_SCHEMA_VERSION,
        "ledger_status": STATE_FOUND if atom_count else STATE_NOT_FOUND,
        "atom_count": atom_count,
        "blocking_atom_count": blocking_atom_count,
        "atoms_truncated": atoms_truncated,
        "atoms": [_one_lost_atom_row(row, repair_loop_attempts) for row in truncated_rows],
        "absent_fields_not_retained_by_engine": list(ABSENT_FIELDS_NOT_RETAINED_BY_ENGINE),
        "provenance": (LOST_ATOM_SCHEMA_VERSION, "build_lost_semantic_atom_diagnostics"),
    }
