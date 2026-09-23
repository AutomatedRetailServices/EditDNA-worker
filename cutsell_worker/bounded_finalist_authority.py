"""D-191: Bounded Finalist Authority -- the FIRST gate where the bounded
finalist arbiter (D-184) MAY change the terminal BestTake winner.

This is NOT a second BestTake engine and NOT "Prosody picks the winner."
It is a narrow AUTHORITY GATE around D-184's own, already-closed
`BoundedFinalistArbiterResult`: when the normal terminal ladder (D-183's
`TerminalBestTakeConfidence`) has already said "I do not have decisive
evidence" (`NON_DECISIVE` / `TIED` / `CONFLICTED`) for a small (2-3),
meaning-sufficient finalist set, and the bounded finalist arbiter
subsequently reaches `PREFERENCE_SUPPORTED` from non-conflicting,
meaning-safe structured evidence (D-172 Visual/Performance and/or D-188
Prosodic Delivery -- whichever the arbiter's own unanimous-agreement
merge actually consulted; this module never re-derives that merge), the
forced raw-score winner MAY be replaced by the arbiter-supported finalist.

Prosody is evidence, not authority. `authority_source` is always
`"bounded_finalist_arbiter"` -- never `"prosodic_audio"` -- because the
bounded finalist arbiter (D-184), not Prosodic Audio V2 (D-187/D-188)
directly, is what this module consults. D-188's Prosodic comparison is
simply one of the (up to four) independent evidence dimensions D-184's
own merge may have already found unanimous; see `bounded_finalist_
arbiter.py`'s own module docstring for that merge's full contract,
reused here verbatim, never re-implemented.

NO SECOND BESTTAKE ENGINE: this module performs zero scoring, zero
ranking, and zero evidence re-derivation. It consumes exactly one
already-computed `bounded_finalist_arbiter.BoundedFinalistArbiterResult`
per family and applies a small, additional set of REQUIRED, ALREADY-
EXISTING signals (meaning parity from that same result, an already-
computed D-123 actionable-conflict flag, an optional Boundary-ownership
flag) as gates on whether that arbiter's own supported preference may be
trusted as a winner replacement.

D-183 FIREWALL (never weakened): eligibility requires `terminal_
confidence_state` to be exactly one of D-183's own `NON_DECISIVE` /
`TIED` / `CONFLICTED` public states (the SAME frozenset `bounded_
finalist_arbiter.py` already gates on -- reused verbatim). `DECISIVE`,
`DECISIVE_BY_ELIMINATION`, and `UNKNOWN` are never eligible: D-191 must
never reopen a decisive structured decision (D-150's own authoritative
`single_semantic_winner` fast path -- which produces D-183 `DECISIVE`
with reason `single_semantic_winner` -- is therefore already firewalled
by this SAME check; no separate D-150-specific gate is implemented,
because D-150's authority is expressed entirely through the terminal
confidence state it produces, never through a second signal this module
would need to re-consult). The D-190/D-186B Gynecologist family (D-183
`DECISIVE`) is the canonical negative control -- see the D-191 test
suite's own generic replay of that shape.

MEANING P0: `meaning_parity_status` is read directly off the ALREADY-
COMPUTED `BoundedFinalistArbiterResult` (D-184's own P0 gate, which
already aborts to `CONFLICTED`/`ABSTAIN` on any negation/number/claim-
type conflict or any candidate outside the meaning-sufficient set --
see `bounded_finalist_arbiter.language_proposition_relation` reuse).
This module never re-runs that check; it only refuses to apply
authority when that field is not `CONSISTENT`.

D-123 OWNERSHIP: `d123_actionable_conflict` is a caller-supplied boolean
-- in live wiring, `pipeline.py` passes its own already-computed `case_
b_conflict_present` (D-123's own actionable-disagreement signal, built
by `_case_b_fast_path_conflict`, never re-derived here). When True, this
module refuses to apply authority regardless of the arbiter's own
verdict: D-123 already owns an actionable disagreement, and D-191 does
not duplicate or override that ownership.

BOUNDARY FIREWALL: `boundary_only_difference` is a caller-supplied
boolean, honestly `False` in ALL live wiring today -- there is no
independent, already-canonical "the only difference between these
finalists is removable ENTRY/EXIT edge debris" comparator implemented
anywhere in this codebase (mirroring `bounded_finalist_arbiter.py`'s own
disclosed "editability_preferred_candidate_id ... realistically empty
in live wiring" precedent). The parameter exists, and `BLOCKED_BY_
BOUNDARY` is a real, tested state, so a genuinely independent future
Boundary-ownership signal has somewhere principled to plug in without
requiring a new authority state or a second implementation -- it is
never fabricated or inferred here.

Feature flag: `CUTSELL_BOUNDED_FINALIST_ARBITER_AUTHORITY_ENABLED`,
default OFF. Independent of, and layered strictly ON TOP OF, D-184's own
`CUTSELL_BOUNDED_FINALIST_ARBITER_ENABLED` -- this module's authority
evaluation only ever receives a real, non-`None` `BoundedFinalistArbiterResult`
when the caller (`pipeline.py`) already ran that flag's own diagnostic
block; with the authority flag OFF, or with no arbiter result at all,
`evaluate_bounded_finalist_authority` always returns `NOT_ENABLED`/
`NOT_ELIGIBLE` and `winner_after == winner_before` -- selected_clip_id,
ranked, membership, Boundary, Pacing, and Renderer stay byte-identical
to every pre-D-191 run. Diagnostic flags (`CUTSELL_BOUNDED_FINALIST_
ARBITER_ENABLED`, `CUTSELL_PROSODIC_FINALIST_ARBITER_DIAGNOSTICS_
ENABLED`) are NOT implicitly turned on by this flag and this module
never reads them itself -- the caller decides what evidence the arbiter
result it hands in was built from.

Fail-open (this task's own explicit requirement): missing Prosodic
evidence, a missing arbiter row, a malformed/absent preferred candidate
id, a preferred candidate outside the eligible finalist set, an
exception anywhere in evaluation, or missing audio/local path all
resolve to the ORIGINAL winner being preserved (`winner_after ==
winner_before`, `authority_applied=False`) -- never a crash, never a
forced pick.

No provider, no network, no score/weight/threshold, no P1/global
context, no QA-oracle reference of any kind is read or written by this
module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple
import os

from .bounded_finalist_arbiter import (
    DECISION_PREFER_CANDIDATE,
    STATE_CONFLICTED,
    STATE_NOT_ELIGIBLE,
    BoundedFinalistArbiterResult,
)

SCHEMA_VERSION = "cutsell.bounded_finalist_authority.v1"

_AUTHORITY_ENV = "CUTSELL_BOUNDED_FINALIST_ARBITER_AUTHORITY_ENABLED"

# D-183's own public eligible-confidence-state vocabulary, reused
# verbatim (the SAME frozenset `bounded_finalist_arbiter.py` gates its
# own eligibility on) -- never re-declared as a private duplicate
# ontology, and never imported from `pipeline.py` to avoid a load-order
# cycle (matching `bounded_finalist_arbiter.py`'s own precedent).
_ELIGIBLE_TERMINAL_CONFIDENCE_STATES = frozenset({"NON_DECISIVE", "TIED", "CONFLICTED"})
_MIN_FINALISTS = 2
_MAX_FINALISTS = 3

_MEANING_PARITY_CONSISTENT = "CONSISTENT"

# Compact authority-state vocabulary -- no state zoo.
STATE_NOT_ENABLED = "NOT_ENABLED"
STATE_NOT_ELIGIBLE = "NOT_ELIGIBLE"
STATE_BLOCKED_BY_MEANING = "BLOCKED_BY_MEANING"
STATE_BLOCKED_BY_D123 = "BLOCKED_BY_D123"
STATE_BLOCKED_BY_BOUNDARY = "BLOCKED_BY_BOUNDARY"
STATE_BLOCKED_BY_CONFLICT = "BLOCKED_BY_CONFLICT"
STATE_NO_SUPPORTED_PREFERENCE = "NO_SUPPORTED_PREFERENCE"
STATE_APPLIED = "APPLIED"

AUTHORITY_SOURCE_BOUNDED_FINALIST_ARBITER = "bounded_finalist_arbiter"


def _env_true_default_false(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def bounded_finalist_arbiter_authority_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_AUTHORITY_ENV))


@dataclass(frozen=True)
class BoundedFinalistAuthorityResult:
    """One family's bounded-authority verdict. `winner_after` is the
    ONLY field downstream callers may use to (optionally) replace a
    `TakeGroup.selected_clip_id` -- see this module's own docstring for
    the exact, single mutation seam this is designed for."""
    evaluated: bool
    authority_state: str
    winner_before: str
    supported_candidate_id: str | None
    winner_after: str
    terminal_confidence_state: str | None
    arbiter_state: str | None
    arbiter_decision: str | None
    meaning_firewall_passed: bool
    d123_blocked: bool
    boundary_blocked: bool
    structured_conflict: bool
    authority_applied: bool
    authority_source: str | None
    reason: str
    provenance: str = "bounded_finalist_authority_v1"


def _result(
    *,
    evaluated: bool,
    authority_state: str,
    winner_before: str,
    reason: str,
    supported_candidate_id: str | None = None,
    winner_after: str | None = None,
    terminal_confidence_state: str | None = None,
    arbiter_state: str | None = None,
    arbiter_decision: str | None = None,
    meaning_firewall_passed: bool = False,
    d123_blocked: bool = False,
    boundary_blocked: bool = False,
    structured_conflict: bool = False,
    authority_applied: bool = False,
    authority_source: str | None = None,
    provenance: str = "bounded_finalist_authority_v1",
) -> BoundedFinalistAuthorityResult:
    return BoundedFinalistAuthorityResult(
        evaluated=evaluated,
        authority_state=authority_state,
        winner_before=winner_before,
        supported_candidate_id=supported_candidate_id,
        winner_after=winner_after if winner_after is not None else winner_before,
        terminal_confidence_state=terminal_confidence_state,
        arbiter_state=arbiter_state,
        arbiter_decision=arbiter_decision,
        meaning_firewall_passed=meaning_firewall_passed,
        d123_blocked=d123_blocked,
        boundary_blocked=boundary_blocked,
        structured_conflict=structured_conflict,
        authority_applied=authority_applied,
        authority_source=authority_source,
        reason=reason,
        provenance=provenance,
    )


def evaluate_bounded_finalist_authority(
    *,
    enabled: bool,
    winner_before: str,
    candidate_ids: Tuple[str, ...],
    terminal_confidence_state: str | None,
    arbiter_result: "BoundedFinalistArbiterResult | None",
    d123_actionable_conflict: bool = False,
    boundary_only_difference: bool = False,
    provenance: str = "bounded_finalist_authority_v1",
) -> BoundedFinalistAuthorityResult:
    """Core D-191 authority gate. Deterministic: the SAME inputs always
    yield the SAME result, independent of candidate order, clip ids, or
    family ids. Never raises -- any caller-side exception around this
    call is a caller bug, not something this function needs to guard
    with its own try/except, since every branch here is a pure,
    exception-free comparison over already-validated inputs; callers in
    live wiring wrap the WHOLE authority attempt (including arbiter
    evaluation) in a fail-open `try/except` per this module's own
    docstring."""
    try:
        candidate_ids = tuple(candidate_ids)
    except TypeError:
        candidate_ids = ()

    if not enabled:
        return _result(
            evaluated=False,
            authority_state=STATE_NOT_ENABLED,
            winner_before=winner_before,
            reason="authority_flag_disabled",
            terminal_confidence_state=terminal_confidence_state,
            provenance=provenance,
        )

    # --- Eligibility (mirrors D-184's own gate exactly; never a second,
    # differently-tuned eligibility rule). DECISIVE / DECISIVE_BY_
    # ELIMINATION / UNKNOWN are never eligible -- the D-183 firewall. ---
    n = len(candidate_ids)
    if n < _MIN_FINALISTS or n > _MAX_FINALISTS:
        return _result(
            evaluated=True,
            authority_state=STATE_NOT_ELIGIBLE,
            winner_before=winner_before,
            reason="candidate_count_not_eligible",
            terminal_confidence_state=terminal_confidence_state,
            provenance=provenance,
        )
    if terminal_confidence_state not in _ELIGIBLE_TERMINAL_CONFIDENCE_STATES:
        return _result(
            evaluated=True,
            authority_state=STATE_NOT_ELIGIBLE,
            winner_before=winner_before,
            reason="terminal_confidence_state_not_eligible",
            terminal_confidence_state=terminal_confidence_state,
            provenance=provenance,
        )

    if arbiter_result is None or not isinstance(arbiter_result, BoundedFinalistArbiterResult):
        return _result(
            evaluated=True,
            authority_state=STATE_NO_SUPPORTED_PREFERENCE,
            winner_before=winner_before,
            reason="arbiter_result_not_available",
            terminal_confidence_state=terminal_confidence_state,
            provenance=provenance,
        )
    if arbiter_result.arbiter_state == STATE_NOT_ELIGIBLE:
        return _result(
            evaluated=True,
            authority_state=STATE_NOT_ELIGIBLE,
            winner_before=winner_before,
            reason="arbiter_not_eligible",
            terminal_confidence_state=terminal_confidence_state,
            arbiter_state=arbiter_result.arbiter_state,
            arbiter_decision=arbiter_result.decision,
            provenance=provenance,
        )

    # --- P0 meaning firewall (SECOND; overrides everything below). Read
    # directly off D-184's own already-computed field -- never re-run. ---
    meaning_firewall_passed = arbiter_result.meaning_parity_status == _MEANING_PARITY_CONSISTENT
    if not meaning_firewall_passed:
        return _result(
            evaluated=True,
            authority_state=STATE_BLOCKED_BY_MEANING,
            winner_before=winner_before,
            reason="meaning_parity_not_consistent",
            terminal_confidence_state=terminal_confidence_state,
            arbiter_state=arbiter_result.arbiter_state,
            arbiter_decision=arbiter_result.decision,
            meaning_firewall_passed=False,
            structured_conflict=arbiter_result.structured_conflict,
            provenance=provenance,
        )

    # --- D-123 ownership (THIRD; a distinct, higher-priority actionable
    # owner blocks D-191 outright, regardless of the arbiter's verdict). ---
    if d123_actionable_conflict:
        return _result(
            evaluated=True,
            authority_state=STATE_BLOCKED_BY_D123,
            winner_before=winner_before,
            reason="d123_actionable_conflict_present",
            terminal_confidence_state=terminal_confidence_state,
            arbiter_state=arbiter_result.arbiter_state,
            arbiter_decision=arbiter_result.decision,
            meaning_firewall_passed=True,
            d123_blocked=True,
            structured_conflict=arbiter_result.structured_conflict,
            provenance=provenance,
        )

    # --- Boundary ownership (FOURTH; honestly always False in live
    # wiring today -- see module docstring). ---
    if boundary_only_difference:
        return _result(
            evaluated=True,
            authority_state=STATE_BLOCKED_BY_BOUNDARY,
            winner_before=winner_before,
            reason="boundary_only_difference_present",
            terminal_confidence_state=terminal_confidence_state,
            arbiter_state=arbiter_result.arbiter_state,
            arbiter_decision=arbiter_result.decision,
            meaning_firewall_passed=True,
            boundary_blocked=True,
            structured_conflict=arbiter_result.structured_conflict,
            provenance=provenance,
        )

    # --- The arbiter's own verdict (FIFTH). ABSTAIN in any of its own
    # states (NEAR_EQUAL / CONFLICTED / INSUFFICIENT_EVIDENCE) is a
    # first-class, non-forced result -- never a forced pick. ---
    if arbiter_result.structured_conflict or arbiter_result.arbiter_state == STATE_CONFLICTED:
        return _result(
            evaluated=True,
            authority_state=STATE_BLOCKED_BY_CONFLICT,
            winner_before=winner_before,
            reason="arbiter_structured_conflict",
            terminal_confidence_state=terminal_confidence_state,
            arbiter_state=arbiter_result.arbiter_state,
            arbiter_decision=arbiter_result.decision,
            meaning_firewall_passed=True,
            structured_conflict=True,
            provenance=provenance,
        )

    if (
        arbiter_result.decision != DECISION_PREFER_CANDIDATE
        or arbiter_result.preferred_candidate_id is None
    ):
        return _result(
            evaluated=True,
            authority_state=STATE_NO_SUPPORTED_PREFERENCE,
            winner_before=winner_before,
            reason="arbiter_abstained_or_no_preference",
            terminal_confidence_state=terminal_confidence_state,
            arbiter_state=arbiter_result.arbiter_state,
            arbiter_decision=arbiter_result.decision,
            meaning_firewall_passed=True,
            provenance=provenance,
        )

    supported = arbiter_result.preferred_candidate_id
    # Fail-open: a malformed/out-of-family preferred id is never applied.
    if supported not in candidate_ids:
        return _result(
            evaluated=True,
            authority_state=STATE_NO_SUPPORTED_PREFERENCE,
            winner_before=winner_before,
            reason="preferred_candidate_outside_eligible_finalist_set",
            supported_candidate_id=supported,
            terminal_confidence_state=terminal_confidence_state,
            arbiter_state=arbiter_result.arbiter_state,
            arbiter_decision=arbiter_result.decision,
            meaning_firewall_passed=True,
            provenance=provenance,
        )

    # --- All conditions pass: the bounded finalist arbiter's own
    # supported preference becomes the terminal winner. Prosody is
    # evidence, not authority_source. ---
    return _result(
        evaluated=True,
        authority_state=STATE_APPLIED,
        winner_before=winner_before,
        winner_after=supported,
        supported_candidate_id=supported,
        reason=arbiter_result.reason,
        terminal_confidence_state=terminal_confidence_state,
        arbiter_state=arbiter_result.arbiter_state,
        arbiter_decision=arbiter_result.decision,
        meaning_firewall_passed=True,
        structured_conflict=False,
        authority_applied=True,
        authority_source=AUTHORITY_SOURCE_BOUNDED_FINALIST_ARBITER,
        provenance=provenance,
    )


def bounded_finalist_authority_diagnostics(result: "BoundedFinalistAuthorityResult | None") -> dict:
    """JSON-safe, bounded per-family diagnostics row. `None` (authority
    never evaluated at all, e.g. a true single-member family) yields the
    same honest empty-ish row `NOT_ENABLED`/never-evaluated callers get,
    so this key set is always populated identically in shape."""
    if result is None:
        return {
            "bounded_finalist_authority_enabled": False,
            "bounded_finalist_authority_evaluated": False,
            "bounded_finalist_authority_state": STATE_NOT_ENABLED,
            "bounded_finalist_authority_winner_before": None,
            "bounded_finalist_authority_supported_candidate_id": None,
            "bounded_finalist_authority_winner_after": None,
            "bounded_finalist_authority_terminal_confidence": None,
            "bounded_finalist_authority_arbiter_state": None,
            "bounded_finalist_authority_arbiter_decision": None,
            "bounded_finalist_authority_meaning_passed": False,
            "bounded_finalist_authority_d123_blocked": False,
            "bounded_finalist_authority_boundary_blocked": False,
            "bounded_finalist_authority_conflict": False,
            "bounded_finalist_authority_applied": False,
            "bounded_finalist_authority_source": None,
            "bounded_finalist_authority_reason": None,
        }
    return {
        "bounded_finalist_authority_enabled": result.authority_state != STATE_NOT_ENABLED,
        "bounded_finalist_authority_evaluated": result.evaluated,
        "bounded_finalist_authority_state": result.authority_state,
        "bounded_finalist_authority_winner_before": result.winner_before,
        "bounded_finalist_authority_supported_candidate_id": result.supported_candidate_id,
        "bounded_finalist_authority_winner_after": result.winner_after,
        "bounded_finalist_authority_terminal_confidence": result.terminal_confidence_state,
        "bounded_finalist_authority_arbiter_state": result.arbiter_state,
        "bounded_finalist_authority_arbiter_decision": result.arbiter_decision,
        "bounded_finalist_authority_meaning_passed": result.meaning_firewall_passed,
        "bounded_finalist_authority_d123_blocked": result.d123_blocked,
        "bounded_finalist_authority_boundary_blocked": result.boundary_blocked,
        "bounded_finalist_authority_conflict": result.structured_conflict,
        "bounded_finalist_authority_applied": result.authority_applied,
        "bounded_finalist_authority_source": result.authority_source,
        "bounded_finalist_authority_reason": result.reason,
    }


def bounded_finalist_authority_run_summary(rows) -> dict:
    """Pure aggregator over already-computed per-family diagnostics rows
    -- mirrors D-183/D-184's own tail-safe run-summary pattern. Never a
    recomputation of any family's own verdict."""
    counts = {
        "finalist_authority_evaluated_count": 0,
        "finalist_authority_applied_count": 0,
        "finalist_authority_no_action_count": 0,
        "finalist_authority_meaning_block_count": 0,
        "finalist_authority_d123_block_count": 0,
        "finalist_authority_boundary_block_count": 0,
        "finalist_authority_conflict_block_count": 0,
        "finalist_authority_winner_changed_count": 0,
    }
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if not row.get("bounded_finalist_authority_evaluated"):
            continue
        counts["finalist_authority_evaluated_count"] += 1
        state = row.get("bounded_finalist_authority_state")
        if state == STATE_APPLIED:
            counts["finalist_authority_applied_count"] += 1
        elif state == STATE_BLOCKED_BY_MEANING:
            counts["finalist_authority_meaning_block_count"] += 1
        elif state == STATE_BLOCKED_BY_D123:
            counts["finalist_authority_d123_block_count"] += 1
        elif state == STATE_BLOCKED_BY_BOUNDARY:
            counts["finalist_authority_boundary_block_count"] += 1
        elif state == STATE_BLOCKED_BY_CONFLICT:
            counts["finalist_authority_conflict_block_count"] += 1
        else:
            counts["finalist_authority_no_action_count"] += 1
        before = row.get("bounded_finalist_authority_winner_before")
        after = row.get("bounded_finalist_authority_winner_after")
        if row.get("bounded_finalist_authority_applied") and before != after:
            counts["finalist_authority_winner_changed_count"] += 1
    return counts
