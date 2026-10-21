"""D-238: BOUNDED EXACT SINGLETON LOST-ATOM OWNERSHIP -- OFFLINE ONLY.

See docs/CUTSELL_DECISIONS.md D-237K/D-237L/D-237M: D-237L fixed the
observability-layer clip_id namespace bug; D-237M then recovered the
real target relationship for the historical blocking lost atom
(`clip_3b74992d8a5a6cb08b31`, RAW 34673899271) -- a genuine, exact
`RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED` shape: the target's
9 words (indices 41-49) are a strict subset of exactly ONE real
`LanguageAttempt` (`latt_939d22f452dbf84d81cb`, 221 of 252 source words),
which owns exactly ONE `PropositionCandidate`. `relationship_is_
authoritative` is `False` for this shape by design (D-235P's own
`AUTHORITATIVE_RELATIONSHIP_STATUSES` never includes containment) --
correctly so, because "this reconstructed take IS this LanguageAttempt"
would be a FALSE claim for a 9/221-word containment. But "this lost
atom's 9 words are UNAMBIGUOUSLY OWNED by that one attempt's one
proposition" is a narrower, TRUE, and separately provable claim.

## What this module is

A pure, additive, STRUCTURAL-ONLY seam answering exactly one question:
"does a lost atom's own exact word-index set belong, unambiguously and
exclusively, to ONE `LanguageAttempt` that itself owns exactly ONE
`PropositionCandidate`?" Inputs are exactly the ones D-235P/D-237G/L
already compute and expose (candidate word indices, LanguageAttempt word
indices, `source_asset_id`, `proposition_candidate_ids`) -- no text
similarity, no timestamp/overlap heuristic, no fuzzy matching, no new
threshold. Every check is bare frozenset containment/intersection over
already-computed integer index sets, mirroring `exact_identity_
observability.py`'s own "recomputes zero set arithmetic beyond a bare
frozenset difference/intersection" discipline exactly.

## What this module is NOT (binding, restated from this task's own scope)

- NOT a change to `shared_attempt_word_identity.py`. `AUTHORITATIVE_
  RELATIONSHIP_STATUSES` is never imported, never redefined, never
  consulted here -- this module answers a DIFFERENT, narrower question
  than D-235P's own full-attempt-identity contract, and never promotes
  `EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED` (or any other D-235P
  relationship) to authoritative status for FULL ATTEMPT IDENTITY. See
  module docstring's "Two separate contracts" section below.
- NOT a materiality/Freeze/repair decision. `assess_exact_lost_atom_
  ownership()` never reads or writes `blocking`, `freeze_blocked`, a
  `CompleteLostSemanticAtomMateriality`, a `LostSemanticAtomFreezeAuthority
  Decision`, or a `RepairAttempt`. It is imported ONLY by `complete_
  lost_semantic_atom_materiality.py` (D-235Q), as one additional,
  OPTIONAL evidence source consulted at exactly ONE existing gate (the
  step-6 identity-sufficiency gate) -- see that module's own updated
  docstring.
- NOT a numeric/size threshold. A containing `LanguageAttempt` spanning
  221 of 252 source words (the real D-237M shape) is treated IDENTICALLY
  to one spanning 2 of 10 -- ownership_status depends ONLY on set
  membership/exclusivity, never on a size ratio, count, or percentage.
  See `test_15_huge_containing_attempt_ownership_proven_size_irrelevant`.

## Two separate contracts (binding, this task's own explicit instruction)

  ATTEMPT IDENTITY (`shared_attempt_word_identity.py`, D-235P, UNCHANGED):
      "is this reconstructed CandidateTake THE SAME DELIVERY as this
      LanguageAttempt?" -- a full-attempt question, answered by
      `AUTHORITATIVE_RELATIONSHIP_STATUSES` membership (currently
      `RELATIONSHIP_EXACT_SAME_MEMBERSHIP` and the 1-to-N exact
      partition). A containment relationship is correctly NEVER
      authoritative here -- a 9-word fragment is not "the same delivery"
      as a 221-word attempt.
  LOST-ATOM OWNERSHIP (this module, D-238, NEW):
      "do THESE SPECIFIC WORDS (a lost atom's own exact index set)
      belong, unambiguously, to ONE canonical ownership context?" -- a
      narrower, source-scoped question about a SUBSET of words, never
      about whether two whole objects are "the same attempt."

These are never merged. `ExactLostAtomOwnership` has its own status
vocabulary, disjoint from `AUTHORITATIVE_RELATIONSHIP_STATUSES`, and its
own dataclass, never a field bolted onto `AttemptLanguageIdentityMatch`.

## Minimum structural requirements (binding, this task's own 9-condition
## gate -- ALL must hold, or the row ABSTAINs)

  1. the candidate's own word-index set is non-empty;
  2. same `source_asset_id` (the candidate and the containing attempt);
  3. the candidate's words are a STRICT SUBSET of the containing
     attempt's own word set (equivalently: zero "reconstructed-only"
     words -- the candidate's own word set, minus the containing
     attempt's, is empty);
  4. exactly ONE `LanguageAttempt` (from the same source) contains ALL
     target words;
  5. the target words overlap NO OTHER `LanguageAttempt` at all (not
     even partially) -- checked independently of condition 4, so an
     attempt that merely PARTIALLY overlaps the target (without fully
     containing it) still blocks ownership;
  6. that ONE containing attempt owns exactly ONE `PropositionCandidate`
     id;
  7. no cross-source attempt is ever considered for containment (every
     attempt not sharing the candidate's own `source_asset_id` is
     filtered out before any word-set comparison, never merely
     de-prioritized);
  8. no missing word provenance (an attempt with an empty word-index set
     is never eligible to "contain" anything, and never silently
     dropped -- it participates in the overlap check like any other,
     just cannot satisfy containment);
  9. no unresolved ownership ambiguity of any kind (multiple containing
     attempts, any overlap with a second attempt, zero or multiple
     proposition ids on the sole containing attempt).

Any single failure -> a specific, named ABSTAIN-family status (never a
generic catch-all, never a best-effort guess) -- see the 8-value status
vocabulary below. Fail-closed by construction: the function has exactly
ONE path to `EXACT_SINGLETON_OWNERSHIP`, every other branch returns a
distinct non-ownership status.

## Semantic firewall (binding, restated from this task's own directive)

`EXACT_SINGLETON_OWNERSHIP` alone means NOTHING about whether the atom
is safe to drop. It is consumed by D-235Q (`complete_lost_semantic_atom_
materiality.py`) as ONE additional way to satisfy the SAME identity-
sufficiency gate `exact_identity_available` already gates -- meaning-
critical, editorial-required, conflicted, retry/process, and redundant
evidence are all evaluated by D-235L/D-235M's OWN independent logic
BEFORE that gate is ever reached, completely unaffected by this module's
own verdict. This module NEVER computes materiality, editorial
requirement, or a blocking recommendation itself, and is never called by
Freeze/repair/resolver authority directly -- only by D-235Q, as
described in that module's own updated docstring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

SCHEMA_VERSION = "cutsell.exact_lost_atom_ownership.v1"

# ---------------------------------------------------------------------------
# Ownership-status vocabulary (this task's own required 8-value set, plus
# the explicit non-ownership terminal `ABSTAIN`). Disjoint from D-235P's
# own `AUTHORITATIVE_RELATIONSHIP_STATUSES` -- never merged, never reused
# as a synonym for any D-235P relationship status.
# ---------------------------------------------------------------------------
OWNERSHIP_EXACT_SINGLETON = "EXACT_SINGLETON_OWNERSHIP"
OWNERSHIP_AMBIGUOUS_MULTIPLE_ATTEMPTS = "AMBIGUOUS_MULTIPLE_ATTEMPTS"
OWNERSHIP_AMBIGUOUS_MULTIPLE_PROPOSITIONS = "AMBIGUOUS_MULTIPLE_PROPOSITIONS"
OWNERSHIP_SOURCE_MISMATCH = "SOURCE_MISMATCH"
OWNERSHIP_MISSING_WORD_PROVENANCE = "MISSING_WORD_PROVENANCE"
OWNERSHIP_NO_CONTAINING_ATTEMPT = "NO_CONTAINING_ATTEMPT"
OWNERSHIP_PARTIAL_OVERLAP = "PARTIAL_OVERLAP"
OWNERSHIP_ABSTAIN = "ABSTAIN"

_VALID_OWNERSHIP_STATUSES = frozenset({
    OWNERSHIP_EXACT_SINGLETON, OWNERSHIP_AMBIGUOUS_MULTIPLE_ATTEMPTS,
    OWNERSHIP_AMBIGUOUS_MULTIPLE_PROPOSITIONS, OWNERSHIP_SOURCE_MISMATCH,
    OWNERSHIP_MISSING_WORD_PROVENANCE, OWNERSHIP_NO_CONTAINING_ATTEMPT,
    OWNERSHIP_PARTIAL_OVERLAP, OWNERSHIP_ABSTAIN,
})

# The ONE status that ever satisfies D-235Q's identity-sufficiency gate --
# every other status is a non-ownership, fail-closed terminal.
OWNERSHIP_STATUSES_SUFFICIENT_FOR_IDENTITY_GATE = frozenset({OWNERSHIP_EXACT_SINGLETON})


@dataclass(frozen=True)
class LanguageAttemptWordEvidence:
    """Bounded, read-only per-attempt input row -- exactly the fields this
    task's own directive names ("Use exact existing evidence only"): a
    `LanguageAttempt`'s own id, source, already-computed word-index set,
    and already-linked `PropositionCandidate` id set (D-169's own
    one-per-attempt mapping, read here, never re-derived). No text, no
    timestamps."""
    attempt_id: str
    source_asset_id: str
    word_indices: Tuple[int, ...]
    proposition_candidate_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ExactLostAtomOwnership:
    clip_id: str
    source_asset_id: str
    candidate_word_indices: Tuple[int, ...]
    containing_language_attempt_id: Optional[str]
    proposition_candidate_ids: Tuple[str, ...]
    ownership_status: str
    reason_codes: Tuple[str, ...]
    provenance: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.ownership_status not in _VALID_OWNERSHIP_STATUSES:
            raise ValueError(f"invalid ownership_status: {self.ownership_status!r}")

    @property
    def is_exact_singleton(self) -> bool:
        """The ONE boolean D-235Q's own identity-sufficiency gate reads --
        never inferred from any other field, always this exact comparison."""
        return self.ownership_status in OWNERSHIP_STATUSES_SUFFICIENT_FOR_IDENTITY_GATE

    def as_dict(self) -> dict:
        """JSON-safe, no transcript dump -- word indices (already-computed
        integers) are the one bounded exception, same precedent as
        `exact_identity_observability.py`."""
        return {
            "clip_id": self.clip_id,
            "source_asset_id": self.source_asset_id,
            "candidate_word_indices": list(self.candidate_word_indices),
            "containing_language_attempt_id": self.containing_language_attempt_id,
            "proposition_candidate_ids": list(self.proposition_candidate_ids),
            "ownership_status": self.ownership_status,
            "is_exact_singleton": self.is_exact_singleton,
            "reason_codes": list(self.reason_codes),
            "provenance": list(self.provenance),
        }


def _abstain(
    *, clip_id: str, source_asset_id: str, candidate_word_indices: Tuple[int, ...],
    status: str, reason: str,
    containing_language_attempt_id: Optional[str] = None,
    proposition_candidate_ids: Tuple[str, ...] = (),
) -> ExactLostAtomOwnership:
    return ExactLostAtomOwnership(
        clip_id=clip_id, source_asset_id=source_asset_id,
        candidate_word_indices=candidate_word_indices,
        containing_language_attempt_id=containing_language_attempt_id,
        proposition_candidate_ids=proposition_candidate_ids,
        ownership_status=status, reason_codes=(reason,),
        provenance=(SCHEMA_VERSION, "assess_exact_lost_atom_ownership"),
    )


def assess_exact_lost_atom_ownership(
    *,
    clip_id: str,
    candidate_source_asset_id: str,
    candidate_word_indices: Sequence[int],
    language_attempts: Sequence[LanguageAttemptWordEvidence],
) -> ExactLostAtomOwnership:
    """The one D-238 entry point. Pure function; mutates nothing, mints no
    id, recomputes no D-235P relationship. See module docstring for the
    full 9-condition gate. `language_attempts` should be every real
    `LanguageAttempt` this source has (the SAME source-scoped population
    D-237G/L's own `identity_observability_rows_for_source` already
    iterates) -- this function does its OWN source filtering (condition
    7), so passing a caller's full multi-source list is safe."""
    target = frozenset(candidate_word_indices)
    sorted_target = tuple(sorted(target))

    if not target:
        return _abstain(
            clip_id=clip_id, source_asset_id=candidate_source_asset_id,
            candidate_word_indices=sorted_target, status=OWNERSHIP_MISSING_WORD_PROVENANCE,
            reason="candidate_word_indices_empty",
        )

    same_source = [a for a in language_attempts if a.source_asset_id == candidate_source_asset_id]
    cross_source = [a for a in language_attempts if a.source_asset_id != candidate_source_asset_id]

    if not same_source:
        status = OWNERSHIP_SOURCE_MISMATCH if cross_source else OWNERSHIP_NO_CONTAINING_ATTEMPT
        reason = (
            "no_same_source_language_attempt_available"
            if cross_source else "no_language_attempts_supplied"
        )
        return _abstain(
            clip_id=clip_id, source_asset_id=candidate_source_asset_id,
            candidate_word_indices=sorted_target, status=status, reason=reason,
        )

    containing: list = []
    any_overlap = False
    for attempt in same_source:
        words = frozenset(attempt.word_indices)
        if not words:
            continue
        if target <= words:
            containing.append(attempt)
            any_overlap = True
        elif target & words:
            any_overlap = True

    if not containing:
        status = OWNERSHIP_PARTIAL_OVERLAP if any_overlap else OWNERSHIP_NO_CONTAINING_ATTEMPT
        reason = "target_overlaps_but_no_full_containment" if any_overlap else "no_attempt_contains_any_target_word"
        return _abstain(
            clip_id=clip_id, source_asset_id=candidate_source_asset_id,
            candidate_word_indices=sorted_target, status=status, reason=reason,
        )

    if len(containing) > 1:
        return _abstain(
            clip_id=clip_id, source_asset_id=candidate_source_asset_id,
            candidate_word_indices=sorted_target, status=OWNERSHIP_AMBIGUOUS_MULTIPLE_ATTEMPTS,
            reason="multiple_language_attempts_each_fully_contain_target",
        )

    sole = containing[0]

    # Condition 5, independent of condition 4: the target must not touch
    # any SECOND attempt at all, even one that does not fully contain it
    # (a fully-contained-by-A-but-also-partially-overlapping-B shape is
    # still an ownership ambiguity, never silently resolved to A).
    for other in same_source:
        if other.attempt_id == sole.attempt_id:
            continue
        if target & frozenset(other.word_indices):
            return _abstain(
                clip_id=clip_id, source_asset_id=candidate_source_asset_id,
                candidate_word_indices=sorted_target, status=OWNERSHIP_AMBIGUOUS_MULTIPLE_ATTEMPTS,
                reason="target_also_overlaps_a_second_language_attempt",
                containing_language_attempt_id=sole.attempt_id,
            )

    props = tuple(sole.proposition_candidate_ids)
    if len(props) > 1:
        return _abstain(
            clip_id=clip_id, source_asset_id=candidate_source_asset_id,
            candidate_word_indices=sorted_target, status=OWNERSHIP_AMBIGUOUS_MULTIPLE_PROPOSITIONS,
            reason="containing_attempt_owns_more_than_one_proposition",
            containing_language_attempt_id=sole.attempt_id, proposition_candidate_ids=props,
        )
    if len(props) == 0:
        return _abstain(
            clip_id=clip_id, source_asset_id=candidate_source_asset_id,
            candidate_word_indices=sorted_target, status=OWNERSHIP_ABSTAIN,
            reason="containing_attempt_owns_no_proposition_candidate",
            containing_language_attempt_id=sole.attempt_id,
        )

    return ExactLostAtomOwnership(
        clip_id=clip_id, source_asset_id=candidate_source_asset_id,
        candidate_word_indices=sorted_target,
        containing_language_attempt_id=sole.attempt_id,
        proposition_candidate_ids=props,
        ownership_status=OWNERSHIP_EXACT_SINGLETON,
        reason_codes=("exact_singleton_containment_and_proposition_proven",),
        provenance=(SCHEMA_VERSION, "assess_exact_lost_atom_ownership"),
    )


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe, no transcript dump -- same pattern as every
# other D-19x/D-235x/D-237x compact summary in this codebase).
# ---------------------------------------------------------------------------
def exact_lost_atom_ownership_diagnostics(results: Sequence[ExactLostAtomOwnership]) -> dict:
    results = tuple(results)
    status_counts: dict = {}
    for r in results:
        status_counts[r.ownership_status] = status_counts.get(r.ownership_status, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(results),
        "exact_singleton_count": sum(1 for r in results if r.is_exact_singleton),
        "ownership_status_counts": status_counts,
        "provenance": (SCHEMA_VERSION, "exact_lost_atom_ownership_diagnostics"),
    }
