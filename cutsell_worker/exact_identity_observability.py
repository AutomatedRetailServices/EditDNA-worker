"""D-237G: EXACT IDENTITY REAL-MEDIA OBSERVABILITY FOUNDATION.

See docs/CUTSELL_DECISIONS.md D-237/D-237F for the forensic finding this
task closes the observability gap for: `live_language_spine_diagnostics()`
only ever serializes COUNTS (`language_attempt_count`, etc.), never the
per-object `CandidateTake`/`LanguageAttempt`/relationship data a future
real-media run needs to PROVE (not merely hypothesize) why a specific
lost-atom clip does or does not reach an `AUTHORITATIVE_RELATIONSHIP_
STATUSES` verdict.

## What this module is

A pure, additive, DIAGNOSTICS-ONLY re-projection layer over data
`pipeline.py` (D-235X) and `shared_attempt_word_identity.py` (D-235P)
ALREADY compute -- it recomputes NOTHING: no new word-index matching, no
new set-arithmetic, no new relationship classification. Every function
here takes an already-built `shared_attempt_word_identity.
AttemptLanguageIdentityMatch` (or the underlying `WordMembership`/
`LanguageAttempt`/`CandidateTake` objects that produced it) and reads
its already-computed fields into a bounded, JSON-safe dict.

## What this module is NOT

It mints no id, classifies no new relationship, and NEVER promotes
`RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED` (or any other non-
authoritative relationship) to authoritative status -- `relationship_
is_authoritative` is a bare re-projection of `shared_attempt_word_
identity.AUTHORITATIVE_RELATIONSHIP_STATUSES` membership, that frozenset
itself untouched, imported verbatim, never redefined here. This module
is never called by any Freeze/materiality/repair/resolver authority
module to make a KEEP/DISCARD/suppress/block decision -- only to REPORT
one that was already made elsewhere.

## Bounded output (this task's own explicit requirement)

No ASR word text, no transcript, no audio. Word indices (plain integers,
already-computed, source-scoped) are the one exception this task's own
directive explicitly authorizes ("this gate specifically audits
identity") -- every other field is an id, a count, a span (start/end
float), or a categorical status string. Scope is bounded to whichever
`CandidateTake`s and `LanguageAttempt`s a caller already had reason to
compare (this module builds nothing new to iterate; it only serializes
what `pipeline.py`'s own existing per-source loop already produced) --
never an unbounded all-vs-all corpus matrix.

## Source scoping (binding, tested)

Every row this module builds carries its own `source_asset_id`; a
correlation function is never given cross-source inputs to begin with
(the caller's own per-source loop already enforces this), and this
module's own tests additionally prove a poisoned cross-source input
never silently produces a row.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple

from .shared_attempt_word_identity import (
    AttemptLanguageIdentityMatch,
    AUTHORITATIVE_RELATIONSHIP_STATUSES,
    WordMembership,
    exact_proposition_candidate_ids_for_match,
)

SCHEMA_VERSION = "cutsell.exact_identity_observability.v1"


# ---------------------------------------------------------------------------
# Per-object bounded rows.
# ---------------------------------------------------------------------------
def _word_membership_row(membership: WordMembership) -> dict:
    """Bounded, JSON-safe re-projection of one already-built `WordMembership`
    -- word indices only (already-computed integers), never text."""
    indices = tuple(sorted(membership.word_indices))
    return {
        "word_index_count": len(indices),
        "word_index_min": indices[0] if indices else None,
        "word_index_max": indices[-1] if indices else None,
        "word_indices": list(indices),
        "identity_status": membership.identity_status,
    }


def candidate_take_identity_diagnostic_row(
    *, clip_id: str, source_asset_id: str, source_start: Optional[float], source_end: Optional[float],
    membership: WordMembership,
) -> dict:
    """Bounded row for one `CandidateTake`'s own exact word membership --
    Stage 1 of D-237F's own audit. Pure re-projection; never re-derives
    the membership itself (the caller already built it via `shared_
    attempt_word_identity.build_reconstructed_attempt_word_membership`,
    reused verbatim, never reimplemented here)."""
    return {
        "clip_id": clip_id,
        "source_asset_id": source_asset_id,
        "source_start": None if source_start is None else float(source_start),
        "source_end": None if source_end is None else float(source_end),
        **_word_membership_row(membership),
    }


def language_attempt_identity_diagnostic_row(
    *, attempt_id: str, source_asset_id: str, source_start: Optional[float], source_end: Optional[float],
    membership: WordMembership, proposition_candidate_ids: Tuple[str, ...] = (),
    attempt_state: Optional[str] = None,
) -> dict:
    """Bounded row for one `LanguageAttempt`'s own exact word membership
    plus its already-linked `PropositionCandidate` ids (D-169's own V1
    one-per-attempt mapping, read here, never re-derived) -- Stage 2/6 of
    D-237F's own audit."""
    return {
        "attempt_id": attempt_id,
        "source_asset_id": source_asset_id,
        "source_start": None if source_start is None else float(source_start),
        "source_end": None if source_end is None else float(source_end),
        **_word_membership_row(membership),
        "proposition_candidate_ids": list(proposition_candidate_ids),
        "attempt_state": attempt_state,
    }


def identity_match_diagnostic_row(
    match: AttemptLanguageIdentityMatch,
    *,
    candidate_take_row: dict,
    language_attempt_rows: Sequence[dict],
    exact_match_by_clip_id: Optional[Mapping[str, AttemptLanguageIdentityMatch]] = None,
    proposition_candidate_ids_by_attempt_id: Optional[Mapping[str, Tuple[str, ...]]] = None,
) -> dict:
    """THE one bounded, per-clip diagnostic row this task requires --
    Stage 3/4 of D-237F's own audit. Reports the EXACT `relationship_
    status` `shared_attempt_word_identity.classify_word_membership_
    relationship`/`match_reconstructed_attempt_against_language_attempts`
    already computed (D-235P, reused verbatim, never reimplemented) plus
    the derived set-arithmetic quantities already available on `match`
    itself -- recomputes no set operation of its own beyond a bare
    frozenset difference/intersection over ALREADY-COMPUTED index sets
    (never a new relationship classification)."""
    exact_match_by_clip_id = exact_match_by_clip_id or {}
    proposition_candidate_ids_by_attempt_id = proposition_candidate_ids_by_attempt_id or {}

    reconstructed_indices = frozenset(match.reconstructed_word_membership.word_indices)
    language_indices: frozenset = frozenset().union(
        *(frozenset(m.word_indices) for m in match.language_word_memberships)
    ) if match.language_word_memberships else frozenset()

    is_authoritative = match.relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES
    exact_entry = exact_match_by_clip_id.get(match.reconstructed_attempt_id)
    exact_present = exact_entry is not None
    exact_proposition_ids = (
        exact_proposition_candidate_ids_for_match(exact_entry, proposition_candidate_ids_by_attempt_id)
        if exact_present else ()
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": match.reconstructed_attempt_id,
        "source_asset_id": match.source_asset_id,
        "candidate_take": candidate_take_row,
        "language_attempts": list(language_attempt_rows),
        "relationship_status": match.relationship_status,
        "relationship_is_authoritative": is_authoritative,
        "reconstructed_only_word_index_count": len(reconstructed_indices - language_indices),
        "reconstructed_only_word_indices": sorted(reconstructed_indices - language_indices),
        "language_only_word_index_count": len(language_indices - reconstructed_indices),
        "language_only_word_indices": sorted(language_indices - reconstructed_indices),
        "intersection_count": len(reconstructed_indices & language_indices),
        "exact_match_by_clip_id_present": exact_present,
        "exact_match_attempt_ids": (
            list(exact_entry.language_attempt_ids) if exact_present else []
        ),
        "exact_match_proposition_candidate_ids": list(exact_proposition_ids),
    }


# ---------------------------------------------------------------------------
# Batch builder -- bounded to exactly the (candidate, attempt-set) pairs a
# caller already compared (this module iterates nothing of its own).
# ---------------------------------------------------------------------------
def identity_observability_rows_for_source(
    *,
    matches: Sequence[AttemptLanguageIdentityMatch],
    takes_by_clip_id: Mapping[str, object],
    attempts_by_id: Mapping[str, object],
    proposition_candidate_ids_by_attempt_id: Optional[Mapping[str, Tuple[str, ...]]] = None,
    exact_match_by_clip_id: Optional[Mapping[str, AttemptLanguageIdentityMatch]] = None,
) -> dict[str, dict]:
    """Batch form: one bounded row per already-computed `match`, keyed by
    `clip_id`. `takes_by_clip_id`/`attempts_by_id` supply the source_
    asset_id/source_start/source_end/attempt_state fields the `match`
    object itself does not carry (it only carries word-index sets) --
    read-only lookups, never a new comparison. A `match` whose own
    `reconstructed_attempt_id` is missing from `takes_by_clip_id`, or
    whose language attempt ids are missing from `attempts_by_id`, still
    produces a row (fail-open, honest) with `None` span fields rather
    than raising or silently dropping the row."""
    proposition_candidate_ids_by_attempt_id = proposition_candidate_ids_by_attempt_id or {}
    exact_match_by_clip_id = exact_match_by_clip_id or {}
    rows: dict[str, dict] = {}
    for match in matches:
        take = takes_by_clip_id.get(match.reconstructed_attempt_id)
        candidate_row = candidate_take_identity_diagnostic_row(
            clip_id=match.reconstructed_attempt_id,
            source_asset_id=match.source_asset_id,
            source_start=getattr(take, "start", None) if take is not None else None,
            source_end=getattr(take, "end", None) if take is not None else None,
            membership=match.reconstructed_word_membership,
        )
        language_rows: list[dict] = []
        for attempt_id, membership in zip(match.language_attempt_ids, match.language_word_memberships):
            attempt = attempts_by_id.get(attempt_id)
            language_rows.append(language_attempt_identity_diagnostic_row(
                attempt_id=attempt_id,
                source_asset_id=match.source_asset_id,
                source_start=getattr(attempt, "source_start", None) if attempt is not None else None,
                source_end=getattr(attempt, "source_end", None) if attempt is not None else None,
                membership=membership,
                proposition_candidate_ids=proposition_candidate_ids_by_attempt_id.get(attempt_id, ()),
                attempt_state=getattr(attempt, "attempt_state", None) if attempt is not None else None,
            ))
        rows[match.reconstructed_attempt_id] = identity_match_diagnostic_row(
            match,
            candidate_take_row=candidate_row,
            language_attempt_rows=language_rows,
            exact_match_by_clip_id=exact_match_by_clip_id,
            proposition_candidate_ids_by_attempt_id=proposition_candidate_ids_by_attempt_id,
        )
    return rows


# ---------------------------------------------------------------------------
# Lost-atom correlation (this task's own "LOST-ATOM CORRELATION" section).
# ---------------------------------------------------------------------------
def lost_atom_identity_correlation(
    lost_semantic_atoms: Sequence[Mapping],
    identity_rows_by_clip_id: Mapping[str, dict],
) -> list[dict]:
    """Correlates each `lost_semantic_atoms` row (already computed
    elsewhere) with its own bounded identity-observability row by
    `clip_id` -- a pure lookup/join, no new classification, no behavior
    change. A row with no matching `clip_id` entry is simply omitted
    (never a guessed/synthetic identity row)."""
    correlated: list[dict] = []
    seen_provenance_ids: set[str] = set()
    for row in lost_semantic_atoms:
        if not isinstance(row, Mapping):
            continue
        provenance_id = row.get("lost_atom_provenance_id")
        clip_id = str(row.get("clip_id") or "")
        if not provenance_id or not clip_id or str(provenance_id) in seen_provenance_ids:
            continue
        identity_row = identity_rows_by_clip_id.get(clip_id)
        if identity_row is None:
            continue
        seen_provenance_ids.add(str(provenance_id))
        correlated.append({
            "lost_atom_provenance_id": str(provenance_id),
            "clip_id": clip_id,
            "identity": identity_row,
        })
    return correlated


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe compact counts -- same pattern as every other
# D-19x/D-235x summary in this codebase). No transcript dump.
# ---------------------------------------------------------------------------
def exact_identity_observability_diagnostics(rows: Iterable[dict]) -> dict:
    rows = tuple(rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = row.get("relationship_status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(rows),
        "authoritative_count": sum(1 for r in rows if r.get("relationship_is_authoritative")),
        "relationship_status_counts": status_counts,
        "exact_match_present_count": sum(1 for r in rows if r.get("exact_match_by_clip_id_present")),
    }
