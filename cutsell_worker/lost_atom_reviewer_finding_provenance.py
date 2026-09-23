"""D-235S: EXACT LOST-ATOM -> REVIEWER FINDING PROVENANCE LINK, OFFLINE ONLY.

Post D-235R (docs/CUTSELL_DECISIONS.md, VERDICT B -- FREEZE AUTHORITY
OFFLINE PROVEN, REPAIR-LOOP SAME-ATOM LINKAGE GAP REMAINS): this module
establishes IDENTITY only -- it never suppresses anything, never touches
Freeze/materiality/RepairLoop behavior. It answers, exactly, "did THIS
`UNIQUE_FACT_LOST` Finding come from THIS EXACT `_lost_semantic_atoms()`
row" -- never "same clip maybe", "same text approximately", "same
timestamp region", or "same family".

## The audit this task's own directive required (a genuine code-truth
## finding, not an assumption -- see `final_story_coherence_validation.
## py::_lost_semantic_atoms()`)

The directive's own "IMPORTANT EXISTING LIMITATION" section assumed one
clip could yield two lost-atom rows ("lost atom A, lost atom B"), making
bare `clip_id` insufficient. Traced mechanically through the real,
current, unmodified code: `_lost_semantic_atoms()`'s own row-building
loop is `for clip in draft.discarded:` -- ONE iteration per discarded
clip, and each iteration produces AT MOST ONE row (either the early
`LOST_IN_NO_USABLE_REALIZATION_FAMILY` row via `continue`, or the general
coverage-ledger row at the loop's tail, never both, never more than one
of either). **Under the current, unmodified data model, one `clip_id`
cannot structurally produce two `_lost_semantic_atoms()` rows** -- this
is a code fact, not an assumption, and it directly contradicts the
directive's own stated premise. Reported honestly here rather than
building a heuristic to match an incorrect assumption, per this whole
D-235 series' established discipline.

This does NOT mean `clip_id` is trusted blindly forever, or that this
module assumes the invariant without checking it every time it runs: a
future code change (or a hand-built/malformed row list passed to this
module by a test or a future caller) could break it, so `classify_lost_
atom_reviewer_finding_link()` below STILL independently detects and
reports `AMBIGUOUS` whenever more than one candidate genuinely exists in
the data it is actually given -- it never assumes the invariant holds,
it PROVES uniqueness from the supplied evidence every single call.

## Provenance key design (Option A + B, smallest safe combination)

`lost_atom_provenance_id` (this task's own new, additive field, minted in
`_lost_semantic_atoms()` -- see that function's own new inline comment)
is `f"latom_{clip_id}_{ordinal}"`, where `ordinal` is the 0-based count of
prior rows THIS SAME BUILD already produced for the same `clip_id`
(deterministic list order, never dict iteration order, `hash()`, memory
address, or provider ordering) -- it degenerates to `latom_{clip_id}_0`
for every row today (per the audit above), but is already correctly
shaped for a hypothetical future multi-row-per-clip case without any
further design change. It is NOT a new canonical/semantic/attempt/
proposition/clip id -- a narrow, LOCAL provenance handle for exactly one
purpose: lost atom -> reviewer finding -> repair-loop attempt linkage.

## Propagation (zero code change needed at the reviewer stage -- a real
## finding, not merely a design choice)

`final_edit_reviewer.py::review()`'s own `UNIQUE_FACT_LOST` construction
already does `detail=dict(row)` -- a full, verbatim, key-for-key copy of
the ORIGINAL row dict. This means `lost_atom_provenance_id` reaches
`Finding.detail["lost_atom_provenance_id"]` automatically, with NO
modification to `final_edit_reviewer.py` required at all (confirmed by
this module's own regression tests, which assert that file's byte
content is unchanged). `repair_loop.py::RepairAttempt` did NOT already
carry this (its own fields are `previous_realization`/`replacement_
realization`, both plain clip-id tuples, never a `detail` passthrough) --
so ONE additive field, `source_lost_atom_provenance_id: str | None =
None`, was added there and populated (via `finding.detail.get(...)`,
never re-derived or re-matched) at all three `RepairAttempt` construction
sites. No decision/termination logic in either file was touched.

## What this module is NOT (binding, restated from this task's own scope)

- No suppression. `classify_lost_atom_reviewer_finding_link()` and
  `lost_atom_provenance_survived_into_repair_attempt()` are pure
  classifiers -- neither is imported by `repair_loop.py`, `final_edit_
  reviewer.py`, `lost_semantic_atom_freeze_authority.py`, or
  `complete_lost_semantic_atom_materiality.py`, and neither mutates
  anything.
- No Freeze/materiality authority expansion. This module reads only
  already-computed `Finding`/`RepairAttempt` objects and the row's own
  `lost_atom_provenance_id`; it makes no editorial judgment.
- Only `UNIQUE_FACT_LOST`. `classify_lost_atom_reviewer_finding_link()`
  takes an explicit `target_finding_kind` (default `UNIQUE_FACT_LOST`,
  imported from `final_edit_reviewer.py`, never redefined) and returns
  `UNSUPPORTED_FINDING_KIND` immediately for anything else -- `STORY_
  ORDER_BREAK`, `CONTRADICTION`, `IDEA_COVERAGE_LOST`, `CRITICAL_CLAIM_
  LOST`, `CAUSAL_ORDER_BREAK`, and every other kind are never linked or
  suppressed here.
- No fuzzy text, no timestamp heuristic. Matching is exact string
  equality on `lost_atom_provenance_id` (or, when that field is entirely
  absent -- a historical payload -- a defensive `clip_id`-based
  ambiguity check), never `SequenceMatcher`/`difflib`, never a time-
  window/overlap computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .final_edit_reviewer import UNIQUE_FACT_LOST, Finding

SCHEMA_VERSION = "cutsell.lost_atom_reviewer_finding_provenance.v1"

# ---------------------------------------------------------------------------
# Link-status vocabulary (this task's own required 5-value set).
# ---------------------------------------------------------------------------
LINK_EXACT_MATCH = "EXACT_MATCH"
LINK_NO_MATCH = "NO_MATCH"
LINK_AMBIGUOUS = "AMBIGUOUS"
LINK_MISSING_PROVENANCE = "MISSING_PROVENANCE"
LINK_UNSUPPORTED_FINDING_KIND = "UNSUPPORTED_FINDING_KIND"

_VALID_LINK_STATUSES = frozenset({
    LINK_EXACT_MATCH, LINK_NO_MATCH, LINK_AMBIGUOUS,
    LINK_MISSING_PROVENANCE, LINK_UNSUPPORTED_FINDING_KIND,
})

PROVENANCE_FIELD_NAME = "lost_atom_provenance_id"


@dataclass(frozen=True)
class LostAtomReviewerFindingLink:
    lost_atom_provenance_id: Optional[str]
    clip_id: str
    reviewer_finding_kind: Optional[str]
    # `final_edit_reviewer.Finding` has no dedicated id field of its own
    # (confirmed: `plan_id`/`plan_version`/`idea_id`/`clip_ids`/`detail`/
    # `owning_authority`/`blocking`, no `finding_id`) -- honestly `None`
    # always, never fabricated, matching D-235J's own "ABSENT_FIELDS_NOT_
    # RETAINED_BY_ENGINE" precedent for an evidence gap that is real,
    # not this module's own omission.
    reviewer_finding_id: Optional[str]
    link_status: str
    reason: str
    provenance: tuple

    def __post_init__(self) -> None:
        if self.link_status not in _VALID_LINK_STATUSES:
            raise ValueError(f"invalid link_status: {self.link_status!r}")

    def as_dict(self) -> dict:
        return {
            "lost_atom_provenance_id": self.lost_atom_provenance_id,
            "clip_id": self.clip_id,
            "reviewer_finding_kind": self.reviewer_finding_kind,
            "reviewer_finding_id": self.reviewer_finding_id,
            "link_status": self.link_status,
            "reason": self.reason,
            "provenance": list(self.provenance),
        }


def classify_lost_atom_reviewer_finding_link(
    *,
    lost_atom_provenance_id: Optional[str],
    clip_id: str,
    findings: Sequence[Finding],
    target_finding_kind: str = UNIQUE_FACT_LOST,
) -> LostAtomReviewerFindingLink:
    """The one D-235S entry point for reviewer-side linkage. Pure; mutates
    nothing, mints nothing. Scoped exactly to `target_finding_kind`
    (default `UNIQUE_FACT_LOST`) -- see module docstring's "Only
    UNIQUE_FACT_LOST" section."""
    if target_finding_kind != UNIQUE_FACT_LOST:
        return LostAtomReviewerFindingLink(
            lost_atom_provenance_id=lost_atom_provenance_id, clip_id=clip_id,
            reviewer_finding_kind=None, reviewer_finding_id=None,
            link_status=LINK_UNSUPPORTED_FINDING_KIND,
            reason=f"bridge_authorized_only_for_{UNIQUE_FACT_LOST}_not_{target_finding_kind}",
            provenance=(SCHEMA_VERSION, "classify_lost_atom_reviewer_finding_link"),
        )

    candidates = tuple(f for f in findings if f.kind == target_finding_kind)

    if not lost_atom_provenance_id:
        # Historical payload / caller never supplied provenance -- fall
        # back to a DEFENSIVE clip_id-only check: never claim EXACT_MATCH
        # without a real provenance id, but still report whether clip_id
        # itself would be ambiguous among the supplied findings.
        clip_matches = tuple(f for f in candidates if clip_id in f.clip_ids)
        if len(clip_matches) > 1:
            return LostAtomReviewerFindingLink(
                lost_atom_provenance_id=None, clip_id=clip_id,
                reviewer_finding_kind=target_finding_kind, reviewer_finding_id=None,
                link_status=LINK_AMBIGUOUS,
                reason="no_provenance_id_and_clip_id_matches_multiple_findings",
                provenance=(SCHEMA_VERSION, "classify_lost_atom_reviewer_finding_link"),
            )
        return LostAtomReviewerFindingLink(
            lost_atom_provenance_id=None, clip_id=clip_id,
            reviewer_finding_kind=target_finding_kind if clip_matches else None, reviewer_finding_id=None,
            link_status=LINK_MISSING_PROVENANCE,
            reason="lost_atom_provenance_id_absent_historical_payload",
            provenance=(SCHEMA_VERSION, "classify_lost_atom_reviewer_finding_link"),
        )

    exact_matches = tuple(
        f for f in candidates if f.detail.get(PROVENANCE_FIELD_NAME) == lost_atom_provenance_id
    )
    if len(exact_matches) > 1:
        # Defensive: never assumed impossible, always independently
        # verified from the actual supplied evidence -- see module
        # docstring's audit section.
        return LostAtomReviewerFindingLink(
            lost_atom_provenance_id=lost_atom_provenance_id, clip_id=clip_id,
            reviewer_finding_kind=target_finding_kind, reviewer_finding_id=None,
            link_status=LINK_AMBIGUOUS,
            reason="multiple_findings_share_the_same_lost_atom_provenance_id",
            provenance=(SCHEMA_VERSION, "classify_lost_atom_reviewer_finding_link"),
        )
    if len(exact_matches) == 1:
        return LostAtomReviewerFindingLink(
            lost_atom_provenance_id=lost_atom_provenance_id, clip_id=clip_id,
            reviewer_finding_kind=target_finding_kind, reviewer_finding_id=None,
            link_status=LINK_EXACT_MATCH,
            reason="provenance_id_matches_exactly_one_finding",
            provenance=(SCHEMA_VERSION, "classify_lost_atom_reviewer_finding_link"),
        )

    # No exact provenance match -- check whether a same-clip_id finding
    # exists without a provenance id of its own (a partially-historical
    # mix, or a mismatched clip_id/provenance pair supplied in error).
    clip_matches_without_provenance = tuple(
        f for f in candidates if clip_id in f.clip_ids and not f.detail.get(PROVENANCE_FIELD_NAME)
    )
    if len(clip_matches_without_provenance) > 1:
        return LostAtomReviewerFindingLink(
            lost_atom_provenance_id=lost_atom_provenance_id, clip_id=clip_id,
            reviewer_finding_kind=target_finding_kind, reviewer_finding_id=None,
            link_status=LINK_AMBIGUOUS,
            reason="provenance_id_absent_on_candidates_and_clip_id_matches_multiple",
            provenance=(SCHEMA_VERSION, "classify_lost_atom_reviewer_finding_link"),
        )
    if len(clip_matches_without_provenance) == 1:
        return LostAtomReviewerFindingLink(
            lost_atom_provenance_id=lost_atom_provenance_id, clip_id=clip_id,
            reviewer_finding_kind=target_finding_kind, reviewer_finding_id=None,
            link_status=LINK_MISSING_PROVENANCE,
            reason="matching_clip_id_finding_lacks_its_own_provenance_id",
            provenance=(SCHEMA_VERSION, "classify_lost_atom_reviewer_finding_link"),
        )

    return LostAtomReviewerFindingLink(
        lost_atom_provenance_id=lost_atom_provenance_id, clip_id=clip_id,
        reviewer_finding_kind=None, reviewer_finding_id=None,
        link_status=LINK_NO_MATCH,
        reason="no_finding_carries_this_provenance_id_or_clip_id",
        provenance=(SCHEMA_VERSION, "classify_lost_atom_reviewer_finding_link"),
    )


def lost_atom_provenance_survived_into_repair_attempt(
    lost_atom_provenance_id: Optional[str],
    repair_attempts: Sequence,
) -> bool:
    """Repair-loop-side propagation check: `True` iff at least one
    `RepairAttempt` in `repair_attempts` carries the SAME `source_lost_
    atom_provenance_id` -- exact string equality only, never a clip_id
    fallback (unlike the reviewer-side classifier above, this check is
    only ever meaningful once a real provenance id exists to compare)."""
    if not lost_atom_provenance_id:
        return False
    return any(
        getattr(attempt, "source_lost_atom_provenance_id", None) == lost_atom_provenance_id
        for attempt in repair_attempts
    )


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe, no transcript dump).
# ---------------------------------------------------------------------------
def lost_atom_reviewer_finding_provenance_diagnostics(
    *,
    lost_atom_provenance_id: Optional[str],
    reviewer_finding_kind: Optional[str],
    reviewer_source_lost_atom_provenance_id: Optional[str],
    repair_source_lost_atom_provenance_id: Optional[str],
    link_status: str,
) -> dict:
    return {
        "lost_atom_provenance_id": lost_atom_provenance_id,
        "reviewer_finding_kind": reviewer_finding_kind,
        "reviewer_source_lost_atom_provenance_id": reviewer_source_lost_atom_provenance_id,
        "repair_source_lost_atom_provenance_id": repair_source_lost_atom_provenance_id,
        "link_status": link_status,
        "provenance": (SCHEMA_VERSION, "lost_atom_reviewer_finding_provenance_diagnostics"),
    }
