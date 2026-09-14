"""D-200.3: DIMENSION-AWARE EDITORIAL RELATION EVIDENCE -- BOUNDED
IMPLEMENTATION.

See ``docs/CUTSELL_CANONICAL_ENGINE_ARCHITECTURE_D098.md`` Section 10
(D-111) and ``docs/CUTSELL_DECISIONS.md`` D-200/D-200.1/D-200.2/D-200.3.
This module is the ONE small, new, narrow adapter D-200.2 authorized: it
DECOMPOSES the two existing flat 7-label relation vocabularies D-157
(``watch_listen_understanding.AttemptRelationHypothesis``) and D-169
(``language_proposition_relation.RelationEvidence``) already independently
produce into THREE non-mutually-exclusive evidence dimensions --

    attempt_relation        -- how does this realization relate to the
                                preceding realization?
    proposition_relation    -- how does the informational claim relate?
    editorial_beat_relation -- same local audience-facing editorial beat,
                                or a distinct one?

-- per D-200.2's own code-verified decomposition table (``docs/
CUTSELL_DECISIONS.md`` D-200.2 Section 1). It mints NOTHING new: every
value here is read directly off an ALREADY-COMPUTED D-157
``AttemptRelationHypothesis`` tuple or D-169 ``RelationEvidence`` object,
never a new classifier, never a new NLP/behavioral signal.

## D-157/D-169 are EVIDENCE PRODUCERS, never rewritten (binding, restated)

This module imports from ``watch_listen_understanding.py`` and
``language_proposition_relation.py`` for their VOCABULARY CONSTANTS and
TYPES only -- it calls no function in either module and modifies neither
(D-200.3's own directive: "D-157 and D-169 are EVIDENCE PRODUCERS. Do NOT
rewrite them.").

## The decomposition table (D-200.2 Section 1, restated exactly)

    RETRY                  -> ATTEMPT dimension (both D-157, D-169)
    CORRECTION             -> ATTEMPT dimension (both D-157, D-169)
    CONTINUATION           -> ATTEMPT dimension (both D-157, D-169)
    COMPLEMENTARY          -> PROPOSITION dimension (D-169's real editorial-
                               slot-equality path only -- see "D-157 MUST
                               NOT BECOME PROPOSITION AUTHORITY" below)
    NEW_AUDIENCE_BEAT      -> EDITORIAL-BEAT dimension (both D-157, D-169)
    DISTINCT_PROPOSITION   -> PROPOSITION dimension (D-169 only -- D-157
                               structurally never emits this label)
    UNCERTAIN              -> no dimension (absence-of-evidence marker)

Each flat label maps to EXACTLY ONE dimension (D-200.2's own audited
conclusion) -- this module performs no further semantic inference beyond
that direct one-to-one mapping.

## D-157 MUST NOT BECOME PROPOSITION AUTHORITY (binding, this task's own
## instruction)

D-157's own ``COMPLEMENTARY`` hypothesis is emitted from a content-blind,
evidence-free default ("no semantic corroboration available to confirm
non-duplicative content" -- its own docstring) -- never claim-content
evidence. This module therefore NEVER reads a D-157 ``COMPLEMENTARY``
hypothesis as ``proposition_relation`` evidence; only D-169's
``RelationEvidence.relation_candidate`` (which reaches ``COMPLEMENTARY``
via a real ``editorial_slot_evidence`` equality check) may populate the
``proposition_relation`` dimension.

## No new evidence source (binding, this task's own instruction)

``proposition_relation``'s ``SAME_PROPOSITION``/``PROPOSITION_PROGRESSION``
values and ``editorial_beat_relation``'s ``SAME_EDITORIAL_BEAT`` value are
declared in the vocabulary (D-200.2 Section 2) but are NEVER PRODUCED by
this module -- no existing evidence source in this codebase positively
asserts any of the three today (D-200.2's own audited finding). This is
an honest, documented gap, not a bug: a run's
``proposition_same_count``/``proposition_progression_count``/
``same_editorial_beat_count`` remaining ``0`` is CORRECT, never "fixed" by
inventing evidence here (this task's own "NO SAME_EDITORIAL_BEAT FICTION"
section).

## Cross-dimension vs. same-dimension conflict (binding, restated)

Different SUPPORTED values in DIFFERENT dimensions are never a conflict
(e.g. ``attempt_relation=RETRY`` + ``proposition_relation=
DISTINCT_PROPOSITION`` coexist freely -- D-111's own "many-signals... not
one-signal-to-one-rule" doctrine, applied here as "one-signal-can-serve-
several-questions"). Only two sources disagreeing WITHIN THE SAME
dimension (e.g. D-157 says ``attempt_relation=RETRY``, D-169 says
``attempt_relation=CORRECTION`` for the same edge) is a real conflict --
that dimension alone collapses to its own ``UNKNOWN`` value with
``CONFIDENCE_MIXED`` status and a ``<DIMENSION>_SAME_DIMENSION_CONFLICT``
flag; the OTHER two dimensions are never erased by it (the "UNKNOWN
FIREWALL", D-200.2 Section 16).

## No weights, no thresholds (binding, restated)

Confidence/support is categorical only (``CONFIDENCE_SUPPORTED``/
``CONFIDENCE_WEAK``/``CONFIDENCE_MIXED``/``CONFIDENCE_UNKNOWN`` --
``language_utterance_attempt.py``'s own already-vetted vocabulary, reused
verbatim, never redefined). There is no numeric master score anywhere in
this module.

## No authority (restated)

This module produces EVIDENCE only. ``grouping_effective_relation`` below
is the one authorized D-197 GROUPING CONSUMER translation (D-200.3's own
"D-197 GROUPING CONSUMER" section) -- it decides ONLY which already-
existing JOIN/SPLIT rule ``editorial_moment_sequence_integration.
build_editorial_local_groups`` (unchanged) should apply, never a Family/
BestTake/D-191/Ordering/Boundary/Pacing/Renderer decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .language_proposition_relation import RelationEvidence
from .language_utterance_attempt import (
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
)
from .watch_listen_understanding import (
    RELATION_COMPLEMENTARY,
    RELATION_CONTINUATION,
    RELATION_CORRECTION,
    RELATION_DISTINCT_PROPOSITION,
    RELATION_NEW_AUDIENCE_BEAT,
    RELATION_RETRY,
    AttemptRelationHypothesis,
)

SCHEMA_VERSION = "cutsell.structured_editorial_relation.v1"

# ---------------------------------------------------------------------------
# Dimension 1 -- ATTEMPT / REALIZATION RELATION (D-200.2 Section 2.1).
# NEW_ATTEMPT/NO_ATTEMPT_RELATION are declared for schema completeness (a
# future evidence source may populate them) but are NEVER PRODUCED by this
# module today -- no existing signal cleanly distinguishes "positively a
# fresh, unrelated attempt" from "no attempt-relation evidence at all";
# both collapse honestly into ATTEMPT_UNKNOWN here, never guessed.
# ---------------------------------------------------------------------------
ATTEMPT_RETRY = "RETRY"
ATTEMPT_CORRECTION = "CORRECTION"
ATTEMPT_CONTINUATION = "CONTINUATION"
ATTEMPT_NEW_ATTEMPT = "NEW_ATTEMPT"
ATTEMPT_NO_ATTEMPT_RELATION = "NO_ATTEMPT_RELATION"
ATTEMPT_UNKNOWN = "UNKNOWN"
ALLOWED_ATTEMPT_RELATIONS: frozenset[str] = frozenset({
    ATTEMPT_RETRY, ATTEMPT_CORRECTION, ATTEMPT_CONTINUATION,
    ATTEMPT_NEW_ATTEMPT, ATTEMPT_NO_ATTEMPT_RELATION, ATTEMPT_UNKNOWN,
})

# ---------------------------------------------------------------------------
# Dimension 2 -- PROPOSITION / CONTENT RELATION (D-200.2 Section 2.2).
# SAME_PROPOSITION/PROPOSITION_PROGRESSION are declared for schema
# completeness but NEVER PRODUCED here today -- see module docstring's
# "no new evidence source" section.
# ---------------------------------------------------------------------------
PROPOSITION_SAME = "SAME_PROPOSITION"
PROPOSITION_DISTINCT = "DISTINCT_PROPOSITION"
PROPOSITION_COMPLEMENTARY = "COMPLEMENTARY_PROPOSITION"
PROPOSITION_PROGRESSION = "PROPOSITION_PROGRESSION"
PROPOSITION_UNKNOWN = "UNKNOWN"
ALLOWED_PROPOSITION_RELATIONS: frozenset[str] = frozenset({
    PROPOSITION_SAME, PROPOSITION_DISTINCT, PROPOSITION_COMPLEMENTARY,
    PROPOSITION_PROGRESSION, PROPOSITION_UNKNOWN,
})

# ---------------------------------------------------------------------------
# Dimension 3 -- EDITORIAL-BEAT RELATION (D-200.2 Section 2.3).
# SAME_EDITORIAL_BEAT is declared for schema completeness but NEVER
# PRODUCED here today -- see module docstring's "no new evidence source"
# section and this task's own "NO SAME_EDITORIAL_BEAT DETECTOR" directive.
# ---------------------------------------------------------------------------
BEAT_SAME = "SAME_EDITORIAL_BEAT"
BEAT_NEW_AUDIENCE = "NEW_AUDIENCE_BEAT"
BEAT_UNKNOWN = "UNKNOWN"
ALLOWED_EDITORIAL_BEAT_RELATIONS: frozenset[str] = frozenset({
    BEAT_SAME, BEAT_NEW_AUDIENCE, BEAT_UNKNOWN,
})

# ---------------------------------------------------------------------------
# Provenance tags -- which already-existing evidence PRODUCER supported a
# dimension's value. Never a new evidence class of its own.
# ---------------------------------------------------------------------------
PROVENANCE_D157_WATCH_LISTEN = "D157_WATCH_LISTEN"
PROVENANCE_D169_LANGUAGE_PROPOSITION = "D169_LANGUAGE_PROPOSITION"

# ---------------------------------------------------------------------------
# Decomposition table -- the ONE authoritative label->dimension mapping
# (D-200.2 Section 1). Every flat 7-label relation value maps to EXACTLY
# ONE dimension; RELATION_UNCERTAIN maps to none (absence-of-evidence).
# ---------------------------------------------------------------------------
_ATTEMPT_LABELS: frozenset[str] = frozenset({RELATION_RETRY, RELATION_CORRECTION, RELATION_CONTINUATION})
_PROPOSITION_LABELS: frozenset[str] = frozenset({RELATION_COMPLEMENTARY, RELATION_DISTINCT_PROPOSITION})
_BEAT_LABELS: frozenset[str] = frozenset({RELATION_NEW_AUDIENCE_BEAT})

_ATTEMPT_VALUE_MAP: dict[str, str] = {
    RELATION_RETRY: ATTEMPT_RETRY, RELATION_CORRECTION: ATTEMPT_CORRECTION, RELATION_CONTINUATION: ATTEMPT_CONTINUATION,
}
_PROPOSITION_VALUE_MAP: dict[str, str] = {
    RELATION_COMPLEMENTARY: PROPOSITION_COMPLEMENTARY, RELATION_DISTINCT_PROPOSITION: PROPOSITION_DISTINCT,
}
_BEAT_VALUE_MAP: dict[str, str] = {
    RELATION_NEW_AUDIENCE_BEAT: BEAT_NEW_AUDIENCE,
}

_UNKNOWN_FOR_DIMENSION: dict[str, str] = {
    "ATTEMPT": ATTEMPT_UNKNOWN, "PROPOSITION": PROPOSITION_UNKNOWN, "EDITORIAL_BEAT": BEAT_UNKNOWN,
}

# Categorical rank, reused verbatim in shape from every other D-19x module's
# own local ranking constant (e.g. editorial_moment_sequence_integration.
# _CONFIDENCE_RANK) -- never a numeric score, only an ordering used to pick
# the single most-decisive ALREADY-COMPUTED hypothesis when D-157 supplied
# more than one for the same dimension.
_CONFIDENCE_RANK: dict[str, int] = {
    CONFIDENCE_SUPPORTED: 3, CONFIDENCE_MIXED: 2, CONFIDENCE_WEAK: 1, CONFIDENCE_UNKNOWN: 0,
}


@dataclass(frozen=True)
class StructuredEditorialRelationEvidence:
    """One predecessor edge's dimension-aware relation evidence -- EVIDENCE
    only, never a final editing authority (see module docstring). No
    transcript field, no numeric master confidence. Attached to real
    ``UnderstandingSpan.span_id`` values only (D-200.3's own "SOURCE
    IDENTITY" contract) -- never a family/BestTake/rendered-timeline id."""
    left_source_span_id: str | None
    right_source_span_id: str
    attempt_relation: str
    attempt_relation_status: str
    proposition_relation: str
    proposition_relation_status: str
    editorial_beat_relation: str
    editorial_beat_relation_status: str
    attempt_relation_provenance: Tuple[str, ...]
    proposition_relation_provenance: Tuple[str, ...]
    editorial_beat_relation_provenance: Tuple[str, ...]
    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]


def _d157_dimension_vote(
    hypotheses: Tuple[AttemptRelationHypothesis, ...], labels: frozenset[str], value_map: dict[str, str],
) -> Tuple[str | None, str]:
    """Picks the single most-decisive D-157 hypothesis whose ``relation``
    falls in this dimension's label set -- never a new relation-detection
    heuristic, only a selection among evidence D-157 already produced
    (same selection discipline as ``editorial_moment_sequence_integration.
    _dominant_relation``, reused in spirit, not imported, to keep this
    module free of any dependency on that integration module)."""
    candidates = [h for h in hypotheses if h.relation in labels]
    if not candidates:
        return None, CONFIDENCE_UNKNOWN
    best = max(candidates, key=lambda h: _CONFIDENCE_RANK.get(h.confidence, 0))
    return value_map[best.relation], best.confidence


def _d169_dimension_vote(
    relation_evidence: RelationEvidence | None, labels: frozenset[str], value_map: dict[str, str],
) -> Tuple[str | None, str]:
    """Reads D-169's own already-computed ``relation_candidate``/
    ``confidence`` -- never re-derives claim-signature overlap or any
    other D-169-internal computation."""
    if relation_evidence is None:
        return None, CONFIDENCE_UNKNOWN
    label = relation_evidence.relation_candidate
    if label not in labels:
        return None, CONFIDENCE_UNKNOWN
    return value_map[label], relation_evidence.confidence


def _fuse_dimension(
    d157_vote: Tuple[str | None, str], d169_vote: Tuple[str | None, str], dimension_name: str,
) -> Tuple[str, str, Tuple[str, ...], Tuple[str, ...]]:
    """SUPPORT/CONFLICT-style fusion, never a weighted score (mirrors
    ``language_proposition_relation._fuse_support``'s own posture, reused
    in spirit). Returns ``(value, status, provenance, conflict_flags)``.
    Agreement between two independent sources raises status to
    ``CONFIDENCE_SUPPORTED`` (two independent confirmations); a genuine
    same-dimension disagreement collapses the value to this dimension's
    own ``UNKNOWN`` with ``CONFIDENCE_MIXED`` and an explicit conflict
    flag -- the OTHER dimensions are never touched by this (see module
    docstring's "UNKNOWN FIREWALL")."""
    d157_value, d157_status = d157_vote
    d169_value, d169_status = d169_vote
    provenance: list[str] = []
    if d157_value is not None:
        provenance.append(PROVENANCE_D157_WATCH_LISTEN)
    if d169_value is not None:
        provenance.append(PROVENANCE_D169_LANGUAGE_PROPOSITION)

    if d157_value is None and d169_value is None:
        return _UNKNOWN_FOR_DIMENSION[dimension_name], CONFIDENCE_UNKNOWN, tuple(provenance), ()
    if d157_value is not None and d169_value is None:
        return d157_value, d157_status, tuple(provenance), ()
    if d157_value is None and d169_value is not None:
        return d169_value, d169_status, tuple(provenance), ()
    if d157_value == d169_value:
        return d157_value, CONFIDENCE_SUPPORTED, tuple(provenance), ()
    conflict_flag = f"{dimension_name}_SAME_DIMENSION_CONFLICT"
    return _UNKNOWN_FOR_DIMENSION[dimension_name], CONFIDENCE_MIXED, tuple(provenance), (conflict_flag,)


def build_structured_editorial_relation(
    *,
    left_source_span_id: str | None,
    right_source_span_id: str,
    d157_hypotheses: Tuple[AttemptRelationHypothesis, ...] = (),
    canonical_relation_evidence: RelationEvidence | None = None,
) -> StructuredEditorialRelationEvidence:
    """The one canonical adapter this task requires. Pure; no I/O, no
    provider call, no re-derivation of any D-157/D-169-internal
    computation -- reads only their own already-produced hypotheses/
    RelationEvidence. Deterministic: same inputs always produce the same
    output."""
    attempt_value, attempt_status, attempt_prov, attempt_conflict = _fuse_dimension(
        _d157_dimension_vote(d157_hypotheses, _ATTEMPT_LABELS, _ATTEMPT_VALUE_MAP),
        _d169_dimension_vote(canonical_relation_evidence, _ATTEMPT_LABELS, _ATTEMPT_VALUE_MAP),
        "ATTEMPT",
    )
    # D-157 MUST NOT BECOME PROPOSITION AUTHORITY (module docstring) --
    # its own COMPLEMENTARY hypothesis is a content-blind default, never
    # claim-content evidence, so D-157 contributes no vote here at all.
    proposition_value, proposition_status, proposition_prov, proposition_conflict = _fuse_dimension(
        (None, CONFIDENCE_UNKNOWN),
        _d169_dimension_vote(canonical_relation_evidence, _PROPOSITION_LABELS, _PROPOSITION_VALUE_MAP),
        "PROPOSITION",
    )
    beat_value, beat_status, beat_prov, beat_conflict = _fuse_dimension(
        _d157_dimension_vote(d157_hypotheses, _BEAT_LABELS, _BEAT_VALUE_MAP),
        _d169_dimension_vote(canonical_relation_evidence, _BEAT_LABELS, _BEAT_VALUE_MAP),
        "EDITORIAL_BEAT",
    )

    conflict_flags = attempt_conflict + proposition_conflict + beat_conflict
    provenance = tuple(sorted(set(attempt_prov) | set(proposition_prov) | set(beat_prov)))

    return StructuredEditorialRelationEvidence(
        left_source_span_id=left_source_span_id,
        right_source_span_id=right_source_span_id,
        attempt_relation=attempt_value,
        attempt_relation_status=attempt_status,
        proposition_relation=proposition_value,
        proposition_relation_status=proposition_status,
        editorial_beat_relation=beat_value,
        editorial_beat_relation_status=beat_status,
        attempt_relation_provenance=tuple(attempt_prov),
        proposition_relation_provenance=tuple(proposition_prov),
        editorial_beat_relation_provenance=tuple(beat_prov),
        conflict_flags=conflict_flags,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# D-197 GROUPING CONSUMER translation (D-200.2 Sections 11-15, D-200.3's
# own "D-197 GROUPING CONSUMER"/"POSITIVE JOIN CONTRACT"/"EXPLICIT SPLIT
# CONTRACT"/"PROPOSITION DIMENSION MUST NOT CONTROL GROUPING" sections).
#
# This is the ONLY place this module makes an editorial-adjacent decision,
# and it decides NOTHING beyond "which value, if any, should reach
# `editorial_moment_sequence_integration.build_editorial_local_groups`'s
# OWN, UNCHANGED `_JOIN_RELATIONS` membership test" -- that test itself
# (RETRY/CORRECTION/CONTINUATION join, everything else boundary) is never
# modified here or anywhere in this task.
# ---------------------------------------------------------------------------
GROUPING_ACTION_JOIN = "JOIN"
GROUPING_ACTION_SPLIT = "SPLIT"
ALLOWED_GROUPING_ACTIONS: frozenset[str] = frozenset({GROUPING_ACTION_JOIN, GROUPING_ACTION_SPLIT})

GROUPING_REASON_NO_PREDECESSOR = "NO_PREDECESSOR"
GROUPING_REASON_EXPLICIT_BEAT_BOUNDARY = "EXPLICIT_NEW_AUDIENCE_BEAT_BOUNDARY"
GROUPING_REASON_ATTEMPT_JOIN = "ATTEMPT_RELATION_JOIN"
GROUPING_REASON_NO_POSITIVE_JOIN_EVIDENCE = "NO_POSITIVE_JOIN_EVIDENCE"
ALLOWED_GROUPING_REASONS: frozenset[str] = frozenset({
    GROUPING_REASON_NO_PREDECESSOR, GROUPING_REASON_EXPLICIT_BEAT_BOUNDARY,
    GROUPING_REASON_ATTEMPT_JOIN, GROUPING_REASON_NO_POSITIVE_JOIN_EVIDENCE,
})

# The SAME three values `build_editorial_local_groups._JOIN_RELATIONS`
# already treats as JOIN -- never a new vocabulary reaching that unchanged
# rule set (D-200.3's own "D-197 GROUPING CONSUMER" instruction).
_JOIN_ATTEMPT_RELATIONS: frozenset[str] = frozenset({ATTEMPT_RETRY, ATTEMPT_CORRECTION, ATTEMPT_CONTINUATION})

# A supported OR weak NEW_AUDIENCE_BEAT already split under the pre-D-200.3
# flat rule set (D-157's own WEAK "fresh-start evidence inconclusive"
# hypothesis was never a member of `_JOIN_RELATIONS` either) -- preserving
# that same threshold here keeps this an explicit split contract, never a
# behavior change for the degenerate D-157-only case.
_BEAT_BOUNDARY_STATUSES: frozenset[str] = frozenset({CONFIDENCE_SUPPORTED, CONFIDENCE_WEAK})


def grouping_effective_relation(
    evidence: StructuredEditorialRelationEvidence | None,
) -> Tuple[str | None, str, str]:
    """Returns ``(grouping_effective_relation_value_or_None, grouping_
    action, grouping_reason)``. ``grouping_effective_relation_value`` is
    always either one of RETRY/CORRECTION/CONTINUATION (the exact
    vocabulary ``build_editorial_local_groups`` already understands) or
    ``None`` (boundary) -- never a new relation vocabulary reaching that
    unchanged consumer.

    PROPOSITION DIMENSION MUST NOT CONTROL GROUPING (binding): this
    function never reads ``proposition_relation`` at all -- neither
    ``DISTINCT_PROPOSITION`` nor ``COMPLEMENTARY_PROPOSITION`` can join or
    split a group by themselves (D-200.2 Sections 14-15's own binding
    decision).

    EXPLICIT SPLIT CONTRACT (binding): a real ``editorial_beat_relation
    == NEW_AUDIENCE_BEAT`` finding is checked FIRST and wins even when the
    attempt dimension would otherwise suggest a join -- the structural
    conflict is preserved (never silently overridden) by simply returning
    the conservative boundary."""
    if evidence is None:
        return None, GROUPING_ACTION_SPLIT, GROUPING_REASON_NO_PREDECESSOR
    if evidence.editorial_beat_relation == BEAT_NEW_AUDIENCE and evidence.editorial_beat_relation_status in _BEAT_BOUNDARY_STATUSES:
        return None, GROUPING_ACTION_SPLIT, GROUPING_REASON_EXPLICIT_BEAT_BOUNDARY
    if evidence.attempt_relation in _JOIN_ATTEMPT_RELATIONS:
        return evidence.attempt_relation, GROUPING_ACTION_JOIN, GROUPING_REASON_ATTEMPT_JOIN
    return None, GROUPING_ACTION_SPLIT, GROUPING_REASON_NO_POSITIVE_JOIN_EVIDENCE


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe, counts/fields-only -- same pattern as every other
# D-19x diagnostic function in this codebase). No transcript.
# ---------------------------------------------------------------------------
def structured_editorial_relation_diagnostics(evidence: StructuredEditorialRelationEvidence | None) -> dict:
    if evidence is None:
        return {
            "left_source_span_id": None, "right_source_span_id": None,
            "attempt_relation": None, "attempt_relation_status": None,
            "proposition_relation": None, "proposition_relation_status": None,
            "editorial_beat_relation": None, "editorial_beat_relation_status": None,
            "attempt_relation_provenance": [], "proposition_relation_provenance": [],
            "editorial_beat_relation_provenance": [], "conflict_flags": [], "provenance": [],
        }
    return {
        "left_source_span_id": evidence.left_source_span_id,
        "right_source_span_id": evidence.right_source_span_id,
        "attempt_relation": evidence.attempt_relation,
        "attempt_relation_status": evidence.attempt_relation_status,
        "proposition_relation": evidence.proposition_relation,
        "proposition_relation_status": evidence.proposition_relation_status,
        "editorial_beat_relation": evidence.editorial_beat_relation,
        "editorial_beat_relation_status": evidence.editorial_beat_relation_status,
        "attempt_relation_provenance": list(evidence.attempt_relation_provenance),
        "proposition_relation_provenance": list(evidence.proposition_relation_provenance),
        "editorial_beat_relation_provenance": list(evidence.editorial_beat_relation_provenance),
        "conflict_flags": list(evidence.conflict_flags),
        "provenance": list(evidence.provenance),
    }
