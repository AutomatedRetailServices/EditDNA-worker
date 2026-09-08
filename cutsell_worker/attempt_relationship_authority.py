"""D-158 Phase C -- Attempt Relationship structured authority.

Per docs/CUTSELL_DECISIONS.md D-148/D-154/D-155/D-156/D-157/D-158. This is
the FIRST editorial consumer of Watch+Listen Understanding V1 (D-157,
unchanged): it lets `take_grouping_provider.reconcile_semantic_idea_
equivalence`'s own pairwise merge decision (Proposition Identity evidence,
via the existing semantic-equivalence arbiter or deterministic restart
evidence -- both unchanged) be corroborated or CONFLICTED by real Watch+
Listen attempt-relation hypotheses for that SAME pair.

## Core principle (this task's own instruction, enforced structurally)

    PERCEPTION PROPOSES EVIDENCE.       (Tracks A-D)
    UNDERSTANDING FORMS HYPOTHESES.     (watch_listen_understanding.py, D-157)
    STRUCTURED EDITORIAL AUTHORITIES DECIDE.  (this module, for ONE narrow
                                          question; Family Formation/D-150/
                                          BestTake for everything else)

This module NEVER sets family membership from Watch+Listen evidence ALONE.
`resolve_final_attempt_relation`'s only possible EFFECT on `would_merge` is
to turn a `True` (the pre-D-158 pipeline decided to merge) into `False`
when real, `SUPPORTED`-confidence Watch+Listen evidence for that EXACT
pair materially disagrees -- it can never turn a `False` into `True`. A
merge Watch+Listen support alone would argue for is never created; only a
merge the existing Proposition Identity evidence already decided on can be
withheld. This is the literal implementation of "Do NOT directly set
family membership from one hypothesis alone" and "Watch+Listen does not
automatically override strong semantic evidence" (this task's own text) --
the asymmetry is deliberate, not an oversight.

## Authority order (this task's own instruction, implemented literally)

1. meaning/safety invariants           -- untouched, upstream (D-089 etc.),
   never read or altered here.
2. Proposition Identity evidence       -- the caller's own already-decided
   `would_merge`/`would_merge_source` (the semantic-equivalence arbiter's
   `same_idea` decision, OR `take_grouping.py`'s deterministic restart-
   evidence rules -- both unchanged, computed entirely before this module
   is ever called).
3. Watch+Listen attempt evidence       -- `AttemptRelationHypothesis`
   tuples for this exact pair (D-157, unchanged).
4. semantic/provider relation evidence -- restated: step 2's own decision
   IS this input; this module never calls a provider itself.
5. structured conflict resolution      -- `resolve_final_attempt_relation`.
6. final attempt relationship          -- `FinalAttemptRelationship`.
7. family formation                    -- the caller applies `would_merge`
   (`take_grouping_provider.reconcile_semantic_idea_equivalence`, unchanged
   elsewhere) -- family formation never runs before step 6 resolves.

## Final relation -> family-membership-action mapping (this task's own
## table, implemented verbatim)

RETRY               -> ELIGIBLE_RETRY_FAMILY (merge proceeds)
CORRECTION          -> SEPARATE_CORRECTION (not a competing retry)
CONTINUATION        -> SEPARATE_NOT_COMPETING
COMPLEMENTARY       -> SEPARATE_NOT_COMPETING
NEW_AUDIENCE_BEAT   -> SEPARATE_BEAT
DISTINCT_PROPOSITION-> SEPARATE_DISTINCT (never actually reachable above
                        UNCERTAIN from `watch_listen_understanding.py`'s
                        own honesty ceiling -- handled defensively)
UNCERTAIN           -> ABSTAIN_UNCERTAIN (preserve separate; never a
                        forced merge, per this task's own UNCERTAIN
                        section)

Only `RETRY` ever yields a merge; every other final relation withholds one
-- this collapses cleanly onto `reconcile_semantic_idea_equivalence`'s own
existing boolean union-find API without requiring any change to its
merge/no-merge machinery.

## D-150 compatibility (restated, unaffected)

This module operates strictly BEFORE any family exists (it decides whether
two groups merge into one family at all). D-150's semantic-authority gate
(`family_complete_context`/`complete_context_conflict`) operates strictly
AFTER a family already exists, deciding comparative-winner authority
WITHIN it. The two stages read disjoint state and this module never
imports, calls, or is called by `semantic_authority_observability.py` or
`pipeline.py`'s D-150 call site -- confirmed by the module-leaf tests in
`tests/test_cutsell_d158_attempt_relationship_authority.py`.

## Fail-open / rollback

`watch_listen_family_evidence_enabled()` (env `CUTSELL_WATCH_LISTEN_
FAMILY_EVIDENCE_ENABLED`, default **OFF** -- this is a genuinely new
editorial-adjacent evidence PATH, unlike D-155's byte-identical scheduling
change, so it stays opt-in until qualified on real media) is the one
capability flag, matching this task's own "do not create a complex flag
framework" instruction. When OFF, or when no Watch+Listen evidence is
available for a given pair (`watch_listen_relations` empty/absent), every
function in this module is a total no-op pass-through: `resolve_final_
attempt_relation` returns exactly `would_merge`'s own pre-D-158 decision,
unchanged, with `conflict=False`. No exception is ever raised for missing
or malformed Watch+Listen data -- absence is treated as "no evidence",
never as an error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

from .watch_listen_understanding import (
    CONFIDENCE_SUPPORTED,
    RELATION_COMPLEMENTARY,
    RELATION_CONTINUATION,
    RELATION_CORRECTION,
    RELATION_DISTINCT_PROPOSITION,
    RELATION_NEW_AUDIENCE_BEAT,
    RELATION_RETRY,
    RELATION_UNCERTAIN,
    AttemptRelationHypothesis,
    UnderstandingSpan,
    WatchListenUnderstanding,
)

import os

SCHEMA_VERSION = "cutsell.attempt_relationship_authority.v1"

_WATCH_LISTEN_FAMILY_EVIDENCE_ENV = "CUTSELL_WATCH_LISTEN_FAMILY_EVIDENCE_ENABLED"

# ---------------------------------------------------------------------------
# Final relation vocabulary -- reuses D-157's own relation labels verbatim
# (never a second vocabulary); re-exported here for callers that only need
# this module.
# ---------------------------------------------------------------------------
RELATION_SOURCE_SEMANTIC_PROVIDER = "SEMANTIC_PROVIDER"
RELATION_SOURCE_DETERMINISTIC_RESTART = "DETERMINISTIC_RESTART"
RELATION_SOURCE_WATCH_LISTEN = "WATCH_LISTEN"
RELATION_SOURCE_CONFLICT_RESOLUTION = "CONFLICT_RESOLUTION"
RELATION_SOURCE_UNCERTAIN = "UNCERTAIN"
# D-161: a pair the PRE-EXISTING semantic candidate-pair path never
# generated or evaluated at all -- distinct from RELATION_SOURCE_WATCH_
# LISTEN (which marks AGREEMENT with an existing would_merge=False
# decision the semantic path actively made). "No semantic pair" is the
# discovery module's own honest label for "nothing to agree or disagree
# with existed" -- see resolve_final_attempt_relation's own docstring.
RELATION_SOURCE_NO_SEMANTIC_PAIR = "NO_SEMANTIC_PAIR"
RELATION_SOURCE_WATCH_LISTEN_DISCOVERY = "WATCH_LISTEN_DISCOVERY"

FAMILY_ACTION_ELIGIBLE_RETRY_FAMILY = "ELIGIBLE_RETRY_FAMILY"
FAMILY_ACTION_SEPARATE_NOT_COMPETING = "SEPARATE_NOT_COMPETING"
FAMILY_ACTION_SEPARATE_CORRECTION = "SEPARATE_CORRECTION"
FAMILY_ACTION_SEPARATE_BEAT = "SEPARATE_BEAT"
FAMILY_ACTION_SEPARATE_DISTINCT = "SEPARATE_DISTINCT"
FAMILY_ACTION_ABSTAIN_UNCERTAIN = "ABSTAIN_UNCERTAIN"

_FAMILY_ACTION_FOR_RELATION: Mapping[str, str] = {
    RELATION_RETRY: FAMILY_ACTION_ELIGIBLE_RETRY_FAMILY,
    RELATION_CORRECTION: FAMILY_ACTION_SEPARATE_CORRECTION,
    RELATION_CONTINUATION: FAMILY_ACTION_SEPARATE_NOT_COMPETING,
    RELATION_COMPLEMENTARY: FAMILY_ACTION_SEPARATE_NOT_COMPETING,
    RELATION_NEW_AUDIENCE_BEAT: FAMILY_ACTION_SEPARATE_BEAT,
    RELATION_DISTINCT_PROPOSITION: FAMILY_ACTION_SEPARATE_DISTINCT,
    RELATION_UNCERTAIN: FAMILY_ACTION_ABSTAIN_UNCERTAIN,
}

# Relations that agree with "these are NOT one competing retry family" --
# used to detect AGREEMENT (not conflict) when the semantic/deterministic
# side already said would_merge=False.
_NOT_COMPETING_RELATIONS: frozenset[str] = frozenset({
    RELATION_CONTINUATION, RELATION_COMPLEMENTARY, RELATION_NEW_AUDIENCE_BEAT,
    RELATION_DISTINCT_PROPOSITION,
})


def _env_true_default_false(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def watch_listen_family_evidence_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_WATCH_LISTEN_FAMILY_EVIDENCE_ENV))


@dataclass(frozen=True)
class FinalAttemptRelationship:
    """The one structured decision this module produces per candidate
    pair. `would_merge` is the ONLY field `reconcile_semantic_idea_
    equivalence` needs to act on; everything else is diagnostics."""
    relation: str
    would_merge: bool
    source: str
    conflict: bool
    family_membership_action: str
    reason: str
    watch_listen_relation_evaluated: bool
    watch_listen_supported: bool


def build_understanding_span_index(
    understandings: Iterable[WatchListenUnderstanding],
) -> Mapping[str, UnderstandingSpan]:
    """Flattens one or more per-source `WatchListenUnderstanding` objects
    into ONE `{span_id: UnderstandingSpan}` lookup -- mirrors the existing
    `take_map = {take.clip_id: take for take in takes}` convention already
    used everywhere else in this pipeline (one global clip_id namespace
    across all hydrated sources). Pure re-indexing; computes nothing new."""
    index: dict[str, UnderstandingSpan] = {}
    for understanding in understandings:
        for span in understanding.understanding_spans:
            index[span.span_id] = span
    return index


def attempt_relation_hypotheses_for_pair(
    understanding_spans_by_id: Mapping[str, UnderstandingSpan] | None,
    left_id: str,
    right_id: str,
) -> Tuple[AttemptRelationHypothesis, ...]:
    """Looks up the real D-157 relation hypotheses this SPECIFIC pair
    already carries (`right_id`'s own span, filtered to entries whose
    `left_span_id == left_id`). Returns an empty tuple -- treated as "no
    Watch+Listen evidence for this pair", never an error -- whenever the
    index is missing, the span is absent, or (the common case for a
    non-adjacent candidate pair) D-157 never computed a relation between
    these two specific spans at all (D-157 only computes relations between
    IMMEDIATE map-order neighbors; `reconcile_semantic_idea_equivalence`'s
    own candidate pairs can be non-adjacent -- this module does not widen
    D-157's own scope to cover that, it only reuses what already exists)."""
    if not understanding_spans_by_id:
        return ()
    right_span = understanding_spans_by_id.get(right_id)
    if right_span is None:
        return ()
    return tuple(
        relation for relation in right_span.attempt_relation_hypotheses
        if relation.left_span_id == left_id
    )


def _best_supported_relation(
    relations: Tuple[AttemptRelationHypothesis, ...],
) -> AttemptRelationHypothesis | None:
    """First SUPPORTED-confidence entry, in D-157's own deterministic
    append order -- WEAK/UNKNOWN entries are treated as "no material
    evidence" (see module docstring's UNCERTAIN-is-not-a-conflict note),
    never escalated to a conflict on their own."""
    for relation in relations:
        if relation.confidence == CONFIDENCE_SUPPORTED and relation.relation != RELATION_UNCERTAIN:
            return relation
    return None


def _resolve_discovery_relation(
    watch_listen_relations: Tuple[AttemptRelationHypothesis, ...],
    proposition_evidence_sufficient: bool,
) -> FinalAttemptRelationship:
    """D-161 Phase C.2: the DISCOVERY half of this authority's truth
    table -- reached only when the caller explicitly passes
    `semantic_path_evaluated=False` (i.e. this pair was never generated
    or evaluated by the pre-existing semantic candidate-pair path at
    all; there is no `would_merge` decision to agree or conflict with).
    `resolve_final_attempt_relation`'s own False->True prohibition still
    holds here in spirit: a discovered RETRY relation ALONE is never
    sufficient -- `proposition_evidence_sufficient` (independent
    deterministic restart/completion evidence, computed by the caller,
    e.g. `watch_listen_relation_discovery.proposition_evidence_for_pair`)
    must ALSO be true before `would_merge` is ever set. Every other
    relation (CORRECTION/CONTINUATION/COMPLEMENTARY/NEW_AUDIENCE_BEAT/
    DISTINCT_PROPOSITION) is surfaced structurally but never merges,
    exactly as the non-discovery truth table already does for those
    labels."""
    best = _best_supported_relation(watch_listen_relations)
    if best is None:
        return FinalAttemptRelationship(
            RELATION_UNCERTAIN, False, RELATION_SOURCE_UNCERTAIN, False,
            FAMILY_ACTION_ABSTAIN_UNCERTAIN,
            "no material watch-listen evidence for this discovered pair; no discovery action",
            bool(watch_listen_relations), False,
        )

    wl_relation = best.relation
    if wl_relation == RELATION_RETRY:
        if proposition_evidence_sufficient:
            return FinalAttemptRelationship(
                RELATION_RETRY, True, RELATION_SOURCE_WATCH_LISTEN_DISCOVERY, False,
                FAMILY_ACTION_ELIGIBLE_RETRY_FAMILY,
                "watch-listen discovery: SUPPORTED retry relation + independent proposition evidence",
                True, True,
            )
        return FinalAttemptRelationship(
            RELATION_UNCERTAIN, False, RELATION_SOURCE_WATCH_LISTEN_DISCOVERY, False,
            FAMILY_ACTION_ABSTAIN_UNCERTAIN,
            "watch-listen discovery: SUPPORTED retry relation but proposition evidence unresolved; not forcing a merge",
            True, True,
        )
    # CORRECTION/COMPLEMENTARY/NEW_AUDIENCE_BEAT/DISTINCT_PROPOSITION:
    # surfaced into their own structured family-membership action, never
    # a retry-family merge -- these relations do not compete for
    # membership, so no proposition-evidence gate applies to them.
    return FinalAttemptRelationship(
        wl_relation, False, RELATION_SOURCE_WATCH_LISTEN_DISCOVERY, False,
        _FAMILY_ACTION_FOR_RELATION[wl_relation],
        f"watch-listen discovery: surfaced {wl_relation} relation (not a retry-family merge)",
        True, True,
    )


def resolve_final_attempt_relation(
    *,
    would_merge: bool,
    would_merge_source: str,
    watch_listen_relations: Tuple[AttemptRelationHypothesis, ...] = (),
    semantic_path_evaluated: bool = True,
    proposition_evidence_sufficient: bool = False,
) -> FinalAttemptRelationship:
    """The one structured-conflict-resolution function this module
    provides (Authority Order steps 5-6). Pure; no I/O, no provider call,
    no recomputation of perception. See module docstring for the full
    truth table and its justification.

    D-161: `semantic_path_evaluated=False` (default `True`, so every
    existing D-158 call site is byte-identical unchanged) routes to
    `_resolve_discovery_relation` -- the ONLY way this function's
    `would_merge` can ever become `True` from Watch+Listen evidence when
    no pre-existing semantic decision exists at all, and only when
    `proposition_evidence_sufficient` is ALSO true (see that function's
    own docstring)."""
    if not semantic_path_evaluated:
        return _resolve_discovery_relation(watch_listen_relations, proposition_evidence_sufficient)

    best = _best_supported_relation(watch_listen_relations)

    if best is None:
        # No material Watch+Listen evidence for this exact pair (missing,
        # WEAK/UNKNOWN only, or UNCERTAIN) -- fail-open: pass the pre-D-158
        # decision through byte-identical.
        relation = RELATION_RETRY if would_merge else RELATION_UNCERTAIN
        return FinalAttemptRelationship(
            relation=relation,
            would_merge=would_merge,
            source=would_merge_source if would_merge else RELATION_SOURCE_UNCERTAIN,
            conflict=False,
            family_membership_action=_FAMILY_ACTION_FOR_RELATION[relation],
            reason="no material watch-listen evidence for this pair; pre-D-158 decision unchanged",
            watch_listen_relation_evaluated=bool(watch_listen_relations),
            watch_listen_supported=False,
        )

    wl_relation = best.relation

    if would_merge:
        if wl_relation == RELATION_RETRY:
            return FinalAttemptRelationship(
                RELATION_RETRY, True, would_merge_source, False,
                FAMILY_ACTION_ELIGIBLE_RETRY_FAMILY,
                "semantic/deterministic and watch-listen evidence agree: RETRY",
                True, True,
            )
        if wl_relation in _NOT_COMPETING_RELATIONS or wl_relation == RELATION_CORRECTION:
            # Conflict: the pre-D-158 evidence said "merge", real, SUPPORTED
            # Watch+Listen evidence for this exact pair says otherwise --
            # withhold the merge (never force it), per this task's own
            # instruction that Watch+Listen evidence must not be blindly
            # inherited over by the provider/deterministic path either way.
            return FinalAttemptRelationship(
                wl_relation, False, RELATION_SOURCE_CONFLICT_RESOLUTION, True,
                _FAMILY_ACTION_FOR_RELATION[wl_relation],
                f"conflict: {would_merge_source} proposed a merge but watch-listen evidence supports {wl_relation}",
                True, True,
            )
        # Defensive: RELATION_UNCERTAIN can never reach here (excluded by
        # `_best_supported_relation`); any other value is unreachable given
        # D-157's own bounded vocabulary. Fail open rather than raise.
        return FinalAttemptRelationship(
            RELATION_RETRY, True, would_merge_source, False,
            FAMILY_ACTION_ELIGIBLE_RETRY_FAMILY,
            "unrecognized watch-listen relation; pre-D-158 decision unchanged",
            True, True,
        )

    # would_merge is False.
    if wl_relation in _NOT_COMPETING_RELATIONS:
        return FinalAttemptRelationship(
            wl_relation, False, RELATION_SOURCE_WATCH_LISTEN, False,
            _FAMILY_ACTION_FOR_RELATION[wl_relation],
            "semantic/deterministic and watch-listen evidence agree: not a competing retry",
            True, True,
        )
    # wl_relation in {RETRY, CORRECTION}: material conflict the other way
    # (pre-D-158 said "not the same idea", Watch+Listen says restart/
    # correction evidence exists) -- never FORCE a merge from Watch+Listen
    # alone (this task's own hard rule); report UNCERTAIN instead.
    return FinalAttemptRelationship(
        RELATION_UNCERTAIN, False, RELATION_SOURCE_CONFLICT_RESOLUTION, True,
        FAMILY_ACTION_ABSTAIN_UNCERTAIN,
        f"conflict: {would_merge_source} found no shared idea but watch-listen evidence supports {wl_relation}; not forcing a merge",
        True, True,
    )


def attempt_relationship_diagnostics(rows: Iterable[FinalAttemptRelationship]) -> dict:
    """Tail-safe, counts-only CI summary (same pattern as D-119/D-125/
    D-152/D-155/D-157's own compact summaries) -- never dumps a transcript
    or a per-pair reason string."""
    rows = tuple(rows)
    counts = {
        "retry_count": 0, "continuation_count": 0, "correction_count": 0,
        "complementary_count": 0, "new_beat_count": 0,
        "distinct_proposition_count": 0, "uncertain_relation_count": 0,
    }
    label_to_key = {
        RELATION_RETRY: "retry_count", RELATION_CONTINUATION: "continuation_count",
        RELATION_CORRECTION: "correction_count", RELATION_COMPLEMENTARY: "complementary_count",
        RELATION_NEW_AUDIENCE_BEAT: "new_beat_count",
        RELATION_DISTINCT_PROPOSITION: "distinct_proposition_count",
        RELATION_UNCERTAIN: "uncertain_relation_count",
    }
    watch_listen_supported_count = 0
    watch_listen_conflict_count = 0
    for row in rows:
        counts[label_to_key.get(row.relation, "uncertain_relation_count")] += 1
        if row.watch_listen_supported:
            watch_listen_supported_count += 1
        if row.conflict:
            watch_listen_conflict_count += 1
    return {
        "watch_listen_relation_evaluated_count": sum(1 for r in rows if r.watch_listen_relation_evaluated),
        "watch_listen_supported_count": watch_listen_supported_count,
        "watch_listen_conflict_count": watch_listen_conflict_count,
        "pair_count": len(rows),
        **counts,
    }
