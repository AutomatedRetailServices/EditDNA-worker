"""D-202: P2 Whole-Video Editorial Reasoning -- Phase A, TYPED FOUNDATION +
DETERMINISTIC WHOLE-VIDEO HYPOTHESES.

See ``docs/CUTSELL_DECISIONS.md`` D-201 (Phase 0 architecture/forensic) and
D-202 (this module) for full context. This module answers P2's own core
question, and NOTHING else:

    HOW DO LOCAL EDITORIAL MOMENTS / GROUPS / SEQUENCES ACROSS THE ENTIRE
    RAW RELATE TO ONE ANOTHER?

It never decides what to delete, what take wins, final sequence order, a
render plan, a physical boundary, or a pacing transition (Family/BestTake/
Ordering/Boundary/Pacing/Renderer territory, all untouched). It produces
HYPOTHESES ONLY -- see "NO AUTHORITY" below.

## What this module is NOT (D-201's own authority boundary, restated)

This module NEVER decides Family membership, a BestTake winner, a Boundary
trim, a Pacing transition, or a render plan. It is not called by any
production call site as of this task: ``pipeline.py``, ``flow_b.py``,
``take_grouping.py``, ``take_grouping_provider.py``, ``composite_
resolver.py``, ``realization_resolver.py``, ``bounded_finalist_arbiter.py``,
``bounded_finalist_authority.py``, ``boundary_engine_pass.py``, ``dialogue_
pacing_transition.py``, ``semantic_ledger.py`` are all confirmed unaware of
this module's existence (module-leaf grep tests,
``tests/test_cutsell_d202_whole_video_editorial_reasoning.py``), and it
imports nothing from any of them. There is no feature flag: nothing gates
this module because nothing calls it yet.

## NO SECOND ONTOLOGY (this task's own instruction, enforced structurally)

Every input this module's builders consume is an already-built P1/D-169
object, referenced by its own stable id, never copied or recomputed:
``EditorialMoment``/``EditorialLocalGroup``/``EditorialSequenceHypothesis``
(D-194/D-195/D-197, ``editorial_moment_sequence*.py``, unchanged) and
``PropositionCandidate``/``ClaimSignature`` (D-169,
``language_proposition_relation.py``, unchanged). This module mints exactly
two NEW id namespaces (``wver_`` for a ``WholeVideoEditorialRegion``,
``wvsup_`` for a ``WholeVideoSupersessionHypothesis``) -- both membership-
anchored (sorted stable-id hashes), never order- or timestamp-anchored,
mirroring D-194/D-197's own id-minting shape exactly.

## Distant proposition matching reuses D-169 verbatim (no second semantic
## engine)

The ONLY comparison this module performs between two propositions is
``language_proposition_relation.signatures_describe_same_proposition``/
``claim_signatures_conflict`` -- the SAME deterministic, marker-based
``ClaimSignature`` comparison D-169 already uses for ADJACENT pairs,
applied here across arbitrary DISTANT pairs. No new lexical similarity
metric, no Family membership used as proposition truth, no QA label
consulted. When a proposition cannot be safely compared (no claim
signature overlap at all), the result is honestly ``UNKNOWN`` -- never
invented.

## Meaning firewall (binding, structurally enforced)

A ``claim_signatures_conflict`` hit (negation polarity flip, or both sides
state a differing number) on ANY earlier/later proposition pair forces the
whole hypothesis's ``meaning_conflict_status`` to ``CONFLICTED`` and its
``supersession_status`` to ``CONFLICTED`` -- unconditionally, before any
coverage/role reasoning runs. This is a short-circuit, never a tie-break
(see ``_supersession_status`` below).

## Unique-information firewall (binding, structurally enforced)

``coverage_status`` can only reach ``FULL_COVERAGE`` when EVERY earlier
proposition candidate this hypothesis claims is redundant is actually
matched, non-conflicting, and COMPLETE on the later side --
``uncovered_earlier_proposition_candidate_ids`` is computed FIRST and
``supersession_status`` can never reach ``SUPPORTED_SUPERSESSION`` while it
is non-empty (see ``_supersession_status``: ``PARTIAL_COVERAGE``/``NO_
COVERAGE`` never resolve to ``SUPPORTED_SUPERSESSION``).

## Chronology firewall (binding, structurally enforced)

``source_start``/``source_end`` decide ONLY which region is labeled
``earlier_region_ids`` vs ``later_region_ids`` in the output (a reporting
convenience -- one canonical ordering per pair, independent of input list
order, closing input-order-independence at the same time). The VERDICT
itself never reads position as evidence: ``recording_process_support`` is
computed strictly from the EARLIER region's OWN ``dominant_process_status``
and ``audience_delivery_support`` strictly from the LATER region's OWN
``dominant_process_status`` -- a region being chronologically later never by
itself credits it with delivery evidence it does not structurally have
(see ``test_chronology_firewall_role_not_position`` in the test file: two
regions with roles reversed relative to their timestamps produce NO
``SUPPORTED_SUPERSESSION`` verdict despite one genuinely being later).

## Region formation (deterministic, no invented segmentation)

Phase A forms exactly one ``WholeVideoEditorialRegion`` per already-computed
``EditorialLocalGroup`` (D-197) -- a thin, referencing WRAPPER, never a
re-derivation of local-group membership and never a new numeric time-gap
segmentation (no ``REGION_GAP_SECONDS`` constant anywhere in this module).
``dominant_process_status``/``audience_delivery_status`` are aggregated
directly from the group's own member moments' already-classified
``moment_role``/``audience_delivery_status`` fields -- the SAME categorical
composition style D-194's own sequence-level aggregators already use, never
a new detector.

## SAME_EDITORIAL_BEAT independence (binding, restated from D-201)

This module never reads or requires ``StructuredEditorialRelationEvidence.
editorial_beat_relation == BEAT_SAME`` (P1's known, honestly-absent-today
positive evidence, see D-193/D-200.2/D-201) for any of its own reasoning --
region formation, the proposition realization map, and supersession
hypotheses all function entirely from moment role / local group / sequence
/ proposition-candidate evidence already proven sufficient in D-201.

## PREASSEMBLED_FINAL_SEQUENCE independence (binding, restated from D-201)

This module never requires ``EditorialSequenceHypothesis.sequence_kind ==
PREASSEMBLED_FINAL_SEQUENCE`` for anything -- a region's
``audience_delivery_status`` is derived from its own moments' roles, not
from whether P1 happened to detect that stronger sequence-level label.

## Confidence / conflict preservation (no invented scores, no averaging)

Categorical only: ``SUPPORTED``/``WEAK``/``MIXED``/``UNKNOWN`` -- the exact
string values ``language_utterance_attempt.py`` already uses (imported,
never redefined). Conflicts are never resolved by majority vote, "latest
wins", "most complete wins", or "longest wins" -- every detected conflict is
recorded as an explicit ``conflict_flags`` entry and forces the relevant
``confidence``/status field to a non-positive value, never averaged away.

## NO AUTHORITY (restated, binding)

No type defined here carries a delete/select/reorder/winner field. No
function in this module mutates a Family/BestTake/Ordering/Boundary/
Pacing/Renderer object, calls ``composite_resolver.py``, or constructs a
runtime realization winner. **P2 HYPOTHESES DO NOT ALTER THE EDIT.**

## No provider, no pipeline wiring (restated, binding)

No OpenAI/Gemini/``whole_video_openai`` call anywhere in this module -- it
is a pure, deterministic, offline classifier over already-computed typed
evidence. ``pipeline.py`` does not import this module (confirmed by grep
test); this module is imported by its own test file only, per this task's
own "no pipeline wiring" scope.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence, Tuple

from .editorial_moment_sequence import (
    AUDIENCE_DELIVERY_NOT_SUPPORTED,
    AUDIENCE_DELIVERY_PARTIAL,
    AUDIENCE_DELIVERY_SUPPORTED,
    AUDIENCE_DELIVERY_UNCERTAIN,
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
    MOMENT_ROLE_ABANDONED_ATTEMPT,
    MOMENT_ROLE_BREAKING_CHARACTER,
    MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
    MOMENT_ROLE_CORRECTION,
    MOMENT_ROLE_FALSE_START,
    MOMENT_ROLE_NEW_AUDIENCE_BEAT,
    MOMENT_ROLE_POST_TAKE_RESET,
    MOMENT_ROLE_PRE_TAKE_SETUP,
    MOMENT_ROLE_RECORDING_PROCESS,
    MOMENT_ROLE_RETRY,
    MOMENT_ROLE_UNCERTAIN,
    EditorialMoment,
    EditorialSequenceHypothesis,
)
from .editorial_moment_sequence_integration import EditorialLocalGroup
from .language_proposition_relation import (
    MEANING_COMPLETE,
    ClaimSignature,
    PropositionCandidate,
    claim_signatures_conflict,
    signatures_describe_same_proposition,
)

SCHEMA_VERSION = "cutsell.whole_video_editorial_reasoning.v1"

# ---------------------------------------------------------------------------
# Region-role vocabulary (small, per D-201 Section 20 -- deliberately does
# NOT include PREASSEMBLED_SEQUENCE_REGION/UNIQUE_CONTENT_REGION; see module
# docstring's PREASSEMBLED independence section).
# ---------------------------------------------------------------------------
REGION_RECORDING_PROCESS = "RECORDING_PROCESS_REGION"
REGION_TAKE_SERIES = "TAKE_SERIES_REGION"
REGION_CLEAN_DELIVERY = "CLEAN_DELIVERY_REGION"
REGION_MIXED = "MIXED_REGION"
REGION_UNKNOWN = "UNKNOWN"
ALLOWED_REGION_PROCESS_STATUSES: frozenset[str] = frozenset({
    REGION_RECORDING_PROCESS, REGION_TAKE_SERIES, REGION_CLEAN_DELIVERY, REGION_MIXED, REGION_UNKNOWN,
})

# Roles that structurally indicate recording-process / retry-family content
# (reused verbatim from editorial_moment_sequence.py's own role vocabulary --
# never a new role invented here).
_PROCESS_ROLES: frozenset[str] = frozenset({
    MOMENT_ROLE_RECORDING_PROCESS, MOMENT_ROLE_FALSE_START, MOMENT_ROLE_ABANDONED_ATTEMPT,
    MOMENT_ROLE_RETRY, MOMENT_ROLE_CORRECTION, MOMENT_ROLE_POST_TAKE_RESET, MOMENT_ROLE_BREAKING_CHARACTER,
    MOMENT_ROLE_PRE_TAKE_SETUP,
})
_TAKE_SERIES_ONLY_ROLES: frozenset[str] = frozenset({MOMENT_ROLE_RETRY, MOMENT_ROLE_CORRECTION})
_CLEAN_ROLES: frozenset[str] = frozenset({MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, MOMENT_ROLE_NEW_AUDIENCE_BEAT})

# ---------------------------------------------------------------------------
# Proposition-realization relationship vocabulary.
# ---------------------------------------------------------------------------
REALIZATION_SINGLE = "SINGLE_REALIZATION"
REALIZATION_MULTIPLE_LOCAL = "MULTIPLE_LOCAL_REALIZATIONS"
REALIZATION_MULTIPLE_DISTANT = "MULTIPLE_DISTANT_REALIZATIONS"
REALIZATION_UNKNOWN = "UNKNOWN"
ALLOWED_REALIZATION_STATUSES: frozenset[str] = frozenset({
    REALIZATION_SINGLE, REALIZATION_MULTIPLE_LOCAL, REALIZATION_MULTIPLE_DISTANT, REALIZATION_UNKNOWN,
})

# ---------------------------------------------------------------------------
# Coverage vocabulary (supersession's own per-pair evidence, before the
# final categorical supersession_status is derived).
# ---------------------------------------------------------------------------
COVERAGE_FULL = "FULL_COVERAGE"
COVERAGE_PARTIAL = "PARTIAL_COVERAGE"
COVERAGE_NONE = "NO_COVERAGE"
COVERAGE_UNKNOWN = "UNKNOWN"
ALLOWED_COVERAGE_STATUSES: frozenset[str] = frozenset({
    COVERAGE_FULL, COVERAGE_PARTIAL, COVERAGE_NONE, COVERAGE_UNKNOWN,
})

# ---------------------------------------------------------------------------
# Role-support vocabulary (categorical, reuses the same SUPPORTED/WEAK/
# UNKNOWN style as everything else in this module -- no numeric score).
# ---------------------------------------------------------------------------
SUPPORT_SUPPORTED = "SUPPORTED"
SUPPORT_WEAK = "WEAK"
SUPPORT_UNKNOWN = "UNKNOWN"
ALLOWED_SUPPORT_STATUSES: frozenset[str] = frozenset({SUPPORT_SUPPORTED, SUPPORT_WEAK, SUPPORT_UNKNOWN})

# ---------------------------------------------------------------------------
# Meaning-conflict vocabulary.
# ---------------------------------------------------------------------------
MEANING_CONFLICT_NONE = "NO_CONFLICT"
MEANING_CONFLICT_PRESENT = "CONFLICTED"
MEANING_CONFLICT_UNKNOWN = "UNKNOWN"
ALLOWED_MEANING_CONFLICT_STATUSES: frozenset[str] = frozenset({
    MEANING_CONFLICT_NONE, MEANING_CONFLICT_PRESENT, MEANING_CONFLICT_UNKNOWN,
})

# ---------------------------------------------------------------------------
# Supersession-status vocabulary (D-201 Section 12, categorical -- never a
# numeric score).
# ---------------------------------------------------------------------------
SUPERSESSION_SUPPORTED = "SUPPORTED_SUPERSESSION"
SUPERSESSION_PARTIAL = "PARTIAL_SUPERSESSION"
SUPERSESSION_NO_SAFE = "NO_SAFE_SUPERSESSION"
SUPERSESSION_CONFLICTED = "CONFLICTED"
SUPERSESSION_UNKNOWN = "UNKNOWN"
ALLOWED_SUPERSESSION_STATUSES: frozenset[str] = frozenset({
    SUPERSESSION_SUPPORTED, SUPERSESSION_PARTIAL, SUPERSESSION_NO_SAFE,
    SUPERSESSION_CONFLICTED, SUPERSESSION_UNKNOWN,
})

# ---------------------------------------------------------------------------
# Global continuity / capability vocabulary (reuses the same categorical
# style; capability values mirror editorial_moment_sequence_integration.py's
# own CAPABILITY_* constants verbatim in spirit, redeclared here to avoid an
# import-time coupling to that module's diagnostics-only constants).
# ---------------------------------------------------------------------------
CONTINUITY_COHERENT = "COHERENT"
CONTINUITY_PARTIAL = "PARTIAL"
CONTINUITY_CONFLICTED = "CONFLICTED"
CONTINUITY_UNKNOWN = "UNKNOWN"
ALLOWED_CONTINUITY_STATUSES: frozenset[str] = frozenset({
    CONTINUITY_COHERENT, CONTINUITY_PARTIAL, CONTINUITY_CONFLICTED, CONTINUITY_UNKNOWN,
})

CAPABILITY_AVAILABLE = "AVAILABLE"
CAPABILITY_PARTIAL = "PARTIAL"
CAPABILITY_NOT_EVALUABLE = "NOT_EVALUABLE"
ALLOWED_CAPABILITY_STATUSES: frozenset[str] = frozenset({
    CAPABILITY_AVAILABLE, CAPABILITY_PARTIAL, CAPABILITY_NOT_EVALUABLE,
})

# Cross-source proposition correspondence vocabulary (D-201 Section "multi-
# source support" -- source identity stays explicit on both sides; never a
# merged identity).
CROSS_SOURCE_SAME_PROPOSITION = "SAME_PROPOSITION"
CROSS_SOURCE_CONFLICTED = "CONFLICTED"
ALLOWED_CROSS_SOURCE_RELATIONSHIPS: frozenset[str] = frozenset({
    CROSS_SOURCE_SAME_PROPOSITION, CROSS_SOURCE_CONFLICTED,
})


def _region_id(source_asset_id: str, moment_ids: Sequence[str]) -> str:
    """Deterministic, MEMBERSHIP-anchored id -- same shape as D-194/D-197's
    own ``_editorial_sequence_id``/``_local_group_id`` (sorted member id
    set, never order- or timestamp-anchored) under a distinct ``wver_``
    prefix."""
    raw = "|".join((source_asset_id, "|".join(sorted(str(v) for v in moment_ids if v)))).encode("utf-8")
    return "wver_" + hashlib.sha256(raw).hexdigest()[:20]


def _supersession_id(source_asset_id: str, earlier_region_ids: Sequence[str], later_region_ids: Sequence[str]) -> str:
    """Deterministic, MEMBERSHIP-anchored id -- same shape as ``_region_id``
    above under a distinct ``wvsup_`` prefix. Both sides are sorted
    independently and joined in a fixed (earlier, later) slot order so the
    id itself never encodes which region happened to be listed first by a
    caller -- only which side the CONTENT (source_start ordering) placed
    earlier/later."""
    raw = "|".join((
        source_asset_id,
        "|".join(sorted(str(v) for v in earlier_region_ids if v)),
        "|".join(sorted(str(v) for v in later_region_ids if v)),
    )).encode("utf-8")
    return "wvsup_" + hashlib.sha256(raw).hexdigest()[:20]


def _categorical_confidence(confidences: Sequence[str], has_conflict: bool) -> str:
    """The one canonical categorical-confidence aggregator this module uses
    everywhere -- never averaged, any conflict forces MIXED. Mirrors
    editorial_moment_sequence.py's/editorial_moment_sequence_integration.
    py's own aggregation style exactly (same four-value vocabulary, same
    precedence), applied here at the whole-video layer."""
    if has_conflict:
        return CONFIDENCE_MIXED
    values = set(confidences)
    if not values:
        return CONFIDENCE_UNKNOWN
    if values == {CONFIDENCE_SUPPORTED}:
        return CONFIDENCE_SUPPORTED
    if CONFIDENCE_MIXED in values:
        return CONFIDENCE_MIXED
    return CONFIDENCE_WEAK


# ---------------------------------------------------------------------------
# TYPE 1 -- WholeVideoEditorialRegion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WholeVideoEditorialRegion:
    """One bounded whole-video HYPOTHESIS region -- a thin, referencing
    wrapper over exactly one already-computed ``EditorialLocalGroup``
    (D-197), never a source-file segmentation authority and never a
    re-derivation of that group's own membership. See module docstring's
    "Region formation" section."""
    source_asset_id: str
    region_id: str
    moment_ids: Tuple[str, ...]
    local_group_ids: Tuple[str, ...]
    sequence_ids: Tuple[str, ...]
    source_start: float
    source_end: float
    dominant_process_status: str
    audience_delivery_status: str
    proposition_candidate_ids: Tuple[str, ...]
    confidence: str
    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.source_end - self.source_start)


def _dominant_process_status(roles: frozenset[str]) -> str:
    """Deterministic, precedence-ordered aggregation over already-classified
    ``EditorialMoment.moment_role`` values -- no new detector, same
    composition style D-194's own sequence-kind classifier uses.
    ``TAKE_SERIES_REGION`` is the narrower case (only RETRY/CORRECTION
    roles present, ignoring UNCERTAIN) inside the broader
    ``RECORDING_PROCESS_REGION`` case, matching D-201's own five-way
    region-role classification fixture set."""
    process_roles = roles & _PROCESS_ROLES
    clean_roles = roles & _CLEAN_ROLES
    non_uncertain = roles - {MOMENT_ROLE_UNCERTAIN}
    if process_roles and clean_roles:
        return REGION_MIXED
    if process_roles:
        if non_uncertain and non_uncertain <= _TAKE_SERIES_ONLY_ROLES:
            return REGION_TAKE_SERIES
        return REGION_RECORDING_PROCESS
    if clean_roles and roles <= clean_roles:
        return REGION_CLEAN_DELIVERY
    return REGION_UNKNOWN


def _audience_delivery_status_for_region(moments: Sequence[EditorialMoment]) -> str:
    """Same categorical aggregation rule as ``editorial_moment_sequence.
    _aggregate_audience_delivery`` (D-194), replayed here at the region
    level over the SAME already-computed per-moment ``audience_delivery_
    status`` field -- never a new detector."""
    statuses = {m.audience_delivery_status for m in moments}
    if statuses == {AUDIENCE_DELIVERY_SUPPORTED}:
        return AUDIENCE_DELIVERY_SUPPORTED
    if AUDIENCE_DELIVERY_SUPPORTED in statuses:
        return AUDIENCE_DELIVERY_PARTIAL
    if statuses == {AUDIENCE_DELIVERY_UNCERTAIN}:
        return AUDIENCE_DELIVERY_UNCERTAIN
    return AUDIENCE_DELIVERY_NOT_SUPPORTED


def build_whole_video_editorial_regions(
    *,
    source_asset_id: str,
    moments: Sequence[EditorialMoment],
    local_groups: Sequence[EditorialLocalGroup],
    sequences: Sequence[EditorialSequenceHypothesis] = (),
) -> Tuple[WholeVideoEditorialRegion, ...]:
    """The one canonical region builder. Pure; no I/O, no provider call, no
    numeric time-gap segmentation. One region per already-computed
    ``EditorialLocalGroup``, deterministic ``(source_start, region_id)``
    output order regardless of input list order (input-order
    independence)."""
    moments_by_id: dict[str, EditorialMoment] = {m.editorial_moment_id: m for m in moments}
    regions: list[WholeVideoEditorialRegion] = []
    for group in local_groups:
        if group.source_asset_id != source_asset_id:
            continue
        group_moments = [moments_by_id[mid] for mid in group.moment_ids if mid in moments_by_id]
        if not group_moments:
            continue
        roles = frozenset(m.moment_role for m in group_moments)
        proposition_candidate_ids = tuple(dict.fromkeys(
            pid for m in group_moments for pid in m.proposition_candidate_ids
        ))
        sequence_ids = tuple(
            s.sequence_id for s in sequences
            if s.source_asset_id == source_asset_id and set(s.moment_ids) <= set(group.moment_ids)
        )
        conflict_flags = tuple(sorted(set(group.conflict_flags) | {f for m in group_moments for f in m.conflict_flags}))
        regions.append(WholeVideoEditorialRegion(
            source_asset_id=source_asset_id,
            region_id=_region_id(source_asset_id, group.moment_ids),
            moment_ids=group.moment_ids,
            local_group_ids=(group.group_id,),
            sequence_ids=sequence_ids,
            source_start=group.source_start,
            source_end=group.source_end,
            dominant_process_status=_dominant_process_status(roles),
            audience_delivery_status=_audience_delivery_status_for_region(group_moments),
            proposition_candidate_ids=proposition_candidate_ids,
            confidence=_categorical_confidence([m.confidence for m in group_moments], bool(conflict_flags)),
            conflict_flags=conflict_flags,
            provenance=("EDITORIAL_LOCAL_GROUP_REFERENCE",),
        ))
    return tuple(sorted(regions, key=lambda r: (r.source_start, r.region_id)))


# ---------------------------------------------------------------------------
# TYPE 2 -- WholeVideoPropositionRealizationMap
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WholeVideoPropositionRealizationMap:
    """One proposition candidate -> every source-real P1 realization that
    may express it, across the WHOLE per-source region set. References
    existing ids only -- never copies a moment/region/sequence object."""
    source_asset_id: str
    proposition_candidate_id: str
    moment_ids: Tuple[str, ...]
    local_group_ids: Tuple[str, ...]
    sequence_ids: Tuple[str, ...]
    region_ids: Tuple[str, ...]
    realization_count: int
    relationship_status: str
    confidence: str
    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]


def build_whole_video_proposition_realization_maps(
    *,
    source_asset_id: str,
    moments: Sequence[EditorialMoment],
    regions: Sequence[WholeVideoEditorialRegion],
) -> Tuple[WholeVideoPropositionRealizationMap, ...]:
    """The one canonical proposition-realization-map builder. Pure; keys
    strictly on ``proposition_candidate_id`` (D-169's own identity) --
    never on ``semantic_idea_id``/``retry_family_id``/``take_group_id``
    (D-201's own "do not equate proposition id with retry family id"
    instruction). Deterministic output order (sorted by
    ``proposition_candidate_id``) regardless of input order."""
    moment_id_to_region: dict[str, WholeVideoEditorialRegion] = {}
    for region in regions:
        if region.source_asset_id != source_asset_id:
            continue
        for mid in region.moment_ids:
            moment_id_to_region[mid] = region

    moments_by_proposition: dict[str, list[EditorialMoment]] = {}
    for m in moments:
        if m.source_asset_id != source_asset_id:
            continue
        for pid in m.proposition_candidate_ids:
            moments_by_proposition.setdefault(pid, []).append(m)

    rows: list[WholeVideoPropositionRealizationMap] = []
    for proposition_candidate_id, prop_moments in moments_by_proposition.items():
        moment_ids = tuple(dict.fromkeys(m.editorial_moment_id for m in prop_moments))
        touched_regions = tuple(dict.fromkeys(
            moment_id_to_region[mid].region_id for mid in moment_ids if mid in moment_id_to_region
        ))
        local_group_ids = tuple(dict.fromkeys(
            gid for mid in moment_ids if mid in moment_id_to_region
            for gid in moment_id_to_region[mid].local_group_ids
        ))
        sequence_ids = tuple(dict.fromkeys(
            sid for mid in moment_ids if mid in moment_id_to_region
            for sid in moment_id_to_region[mid].sequence_ids
        ))
        realization_count = len(moment_ids)
        if realization_count <= 1:
            relationship_status = REALIZATION_SINGLE
        elif len(set(touched_regions)) <= 1:
            relationship_status = REALIZATION_MULTIPLE_LOCAL
        else:
            relationship_status = REALIZATION_MULTIPLE_DISTANT
        conflict_flags = tuple(sorted({f for m in prop_moments for f in m.conflict_flags}))
        rows.append(WholeVideoPropositionRealizationMap(
            source_asset_id=source_asset_id,
            proposition_candidate_id=proposition_candidate_id,
            moment_ids=moment_ids,
            local_group_ids=local_group_ids,
            sequence_ids=sequence_ids,
            region_ids=touched_regions,
            realization_count=realization_count,
            relationship_status=relationship_status,
            confidence=_categorical_confidence([m.confidence for m in prop_moments], bool(conflict_flags)),
            conflict_flags=conflict_flags,
            provenance=("EDITORIAL_MOMENT_PROPOSITION_REFERENCE",),
        ))
    return tuple(sorted(rows, key=lambda r: r.proposition_candidate_id))


# ---------------------------------------------------------------------------
# Cross-source proposition correspondence (multi-source support -- source
# identity stays explicit on both sides, never a merged identity).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CrossSourcePropositionLink:
    left_proposition_candidate_id: str
    left_source_asset_id: str
    right_proposition_candidate_id: str
    right_source_asset_id: str
    relationship: str
    provenance: Tuple[str, ...]


def build_cross_source_proposition_links(
    proposition_candidates_by_id: Mapping[str, PropositionCandidate],
) -> Tuple[CrossSourcePropositionLink, ...]:
    """Pure; O(n^2) over the supplied candidates (bounded per D-201 Section
    36's own graph-vs-typed decision -- no indexing structure needed at
    Phase-A scale). Only compares candidates from DIFFERENT
    ``source_asset_id``s -- same-source comparison is already covered by
    ``build_whole_video_proposition_realization_maps`` grouping by shared
    ``proposition_candidate_id``, which never happens across sources
    (D-169's own id includes ``source_asset_id``). Deterministic output
    order (sorted by the pair's own two ids)."""
    candidates = sorted(proposition_candidates_by_id.values(), key=lambda p: p.proposition_candidate_id)
    links: list[CrossSourcePropositionLink] = []
    for i, left in enumerate(candidates):
        for right in candidates[i + 1:]:
            if left.source_asset_id == right.source_asset_id:
                continue
            if not signatures_describe_same_proposition(left.claim_signature, right.claim_signature):
                continue
            relationship = (
                CROSS_SOURCE_CONFLICTED if claim_signatures_conflict(left.claim_signature, right.claim_signature)
                else CROSS_SOURCE_SAME_PROPOSITION
            )
            links.append(CrossSourcePropositionLink(
                left_proposition_candidate_id=left.proposition_candidate_id,
                left_source_asset_id=left.source_asset_id,
                right_proposition_candidate_id=right.proposition_candidate_id,
                right_source_asset_id=right.source_asset_id,
                relationship=relationship,
                provenance=("CLAIM_SIGNATURE_CROSS_SOURCE_COMPARISON",),
            ))
    return tuple(links)


# ---------------------------------------------------------------------------
# TYPE 3 -- WholeVideoSupersessionHypothesis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WholeVideoSupersessionHypothesis:
    """A later or otherwise distinct source-real region MAY make earlier
    recording-process realizations redundant because it covers the same
    required proposition content -- a HYPOTHESIS only. No delete action, no
    ``selected_clip_id`` field, no field any downstream code could mistake
    for an authority. See module docstring's firewalls."""
    source_asset_id: str
    supersession_id: str
    earlier_region_ids: Tuple[str, ...]
    later_region_ids: Tuple[str, ...]
    covered_proposition_candidate_ids: Tuple[str, ...]
    uncovered_earlier_proposition_candidate_ids: Tuple[str, ...]
    coverage_status: str
    recording_process_support: str
    audience_delivery_support: str
    meaning_conflict_status: str
    supersession_status: str
    confidence: str
    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]


_PROCESS_SUPPORT_BY_STATUS: Mapping[str, str] = {
    REGION_RECORDING_PROCESS: SUPPORT_SUPPORTED,
    REGION_TAKE_SERIES: SUPPORT_SUPPORTED,
    REGION_MIXED: SUPPORT_WEAK,
    REGION_CLEAN_DELIVERY: SUPPORT_UNKNOWN,
    REGION_UNKNOWN: SUPPORT_UNKNOWN,
}
_DELIVERY_SUPPORT_BY_STATUS: Mapping[str, str] = {
    REGION_CLEAN_DELIVERY: SUPPORT_SUPPORTED,
    REGION_MIXED: SUPPORT_WEAK,
    REGION_RECORDING_PROCESS: SUPPORT_UNKNOWN,
    REGION_TAKE_SERIES: SUPPORT_UNKNOWN,
    REGION_UNKNOWN: SUPPORT_UNKNOWN,
}


def _coverage_for_pair(
    earlier_prop_ids: Sequence[str],
    later_prop_ids: Sequence[str],
    proposition_candidates_by_id: Mapping[str, PropositionCandidate],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], str, str]:
    """Returns (covered, uncovered, coverage_status, meaning_conflict_status)
    for one earlier/later region pair. The ONLY comparison used is D-169's
    own ``signatures_describe_same_proposition``/``claim_signatures_
    conflict`` -- never a new similarity metric, never Family membership,
    never a QA label. A later match only counts as coverage when it is
    ALSO ``MEANING_COMPLETE`` (module docstring's "sequence coherence
    sufficient" conjunction item)."""
    later_candidates = [proposition_candidates_by_id[pid] for pid in later_prop_ids if pid in proposition_candidates_by_id]
    covered: list[str] = []
    uncovered: list[str] = []
    any_conflict = False
    any_comparable = False
    for eid in earlier_prop_ids:
        earlier = proposition_candidates_by_id.get(eid)
        if earlier is None:
            continue
        matched = False
        conflicted = False
        for later in later_candidates:
            if claim_signatures_conflict(earlier.claim_signature, later.claim_signature):
                conflicted = True
                any_comparable = True
                continue
            if signatures_describe_same_proposition(earlier.claim_signature, later.claim_signature):
                any_comparable = True
                if later.meaning_completion == MEANING_COMPLETE:
                    matched = True
        if conflicted:
            any_conflict = True
        if matched:
            covered.append(eid)
        else:
            uncovered.append(eid)

    if not earlier_prop_ids:
        coverage_status = COVERAGE_UNKNOWN
    elif not any_comparable and not covered:
        coverage_status = COVERAGE_UNKNOWN
    elif not uncovered:
        coverage_status = COVERAGE_FULL
    elif covered:
        coverage_status = COVERAGE_PARTIAL
    else:
        coverage_status = COVERAGE_NONE

    if any_conflict:
        meaning_conflict_status = MEANING_CONFLICT_PRESENT
    elif any_comparable:
        meaning_conflict_status = MEANING_CONFLICT_NONE
    else:
        meaning_conflict_status = MEANING_CONFLICT_UNKNOWN

    return tuple(covered), tuple(uncovered), coverage_status, meaning_conflict_status


def _supersession_status(
    *, coverage_status: str, meaning_conflict_status: str, recording_process_support: str, audience_delivery_support: str,
) -> str:
    """The one canonical supersession-status decision table. Deterministic,
    precedence-ordered, never averaged. Meaning conflict is checked FIRST
    and unconditionally (meaning firewall); coverage never reaches
    ``SUPPORTED_SUPERSESSION`` unless BOTH role-support signals are
    positive (unique-information + chronology-as-role, not position,
    firewalls)."""
    if meaning_conflict_status == MEANING_CONFLICT_PRESENT:
        return SUPERSESSION_CONFLICTED
    if coverage_status == COVERAGE_UNKNOWN:
        return SUPERSESSION_UNKNOWN
    if coverage_status == COVERAGE_NONE:
        return SUPERSESSION_NO_SAFE
    if coverage_status == COVERAGE_PARTIAL:
        return SUPERSESSION_PARTIAL
    # coverage_status == COVERAGE_FULL from here.
    if recording_process_support == SUPPORT_SUPPORTED and audience_delivery_support == SUPPORT_SUPPORTED:
        return SUPERSESSION_SUPPORTED
    if recording_process_support == SUPPORT_UNKNOWN:
        # No process/retry evidence on the earlier side AT ALL -- this is
        # not "recording process replaced by delivery", it is just two
        # deliveries that happen to share content. Never a supersession
        # claim regardless of how strong the later side's delivery
        # evidence is (recording-process support is the PRIMARY gate).
        return SUPERSESSION_NO_SAFE
    return SUPERSESSION_PARTIAL


def build_whole_video_supersession_hypotheses(
    *,
    source_asset_id: str,
    regions: Sequence[WholeVideoEditorialRegion],
    proposition_candidates_by_id: Mapping[str, PropositionCandidate],
) -> Tuple[WholeVideoSupersessionHypothesis, ...]:
    """The one canonical supersession-hypothesis builder. Pure; O(n^2) over
    the supplied regions of ONE source (bounded per D-201 Section 36).
    Every pair is canonically ordered by ``source_start`` (ties broken by
    ``region_id``) BEFORE any evidence is read -- this fixes which region is
    labeled earlier/later independently of the caller's own input order
    (input-order independence) and is the ONLY use chronology has in this
    function (see module docstring's chronology firewall). Pairs where
    NEITHER side has any proposition_candidate_ids are skipped -- nothing
    to hypothesize about."""
    same_source_regions = [r for r in regions if r.source_asset_id == source_asset_id]
    ordered = sorted(same_source_regions, key=lambda r: (r.source_start, r.region_id))
    hypotheses: list[WholeVideoSupersessionHypothesis] = []
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            if left.source_start <= right.source_start:
                earlier, later = left, right
            else:
                earlier, later = right, left
            if not earlier.proposition_candidate_ids and not later.proposition_candidate_ids:
                continue

            covered, uncovered, coverage_status, meaning_conflict_status = _coverage_for_pair(
                earlier.proposition_candidate_ids, later.proposition_candidate_ids, proposition_candidates_by_id,
            )
            recording_process_support = _PROCESS_SUPPORT_BY_STATUS.get(earlier.dominant_process_status, SUPPORT_UNKNOWN)
            audience_delivery_support = _DELIVERY_SUPPORT_BY_STATUS.get(later.dominant_process_status, SUPPORT_UNKNOWN)
            status = _supersession_status(
                coverage_status=coverage_status, meaning_conflict_status=meaning_conflict_status,
                recording_process_support=recording_process_support, audience_delivery_support=audience_delivery_support,
            )

            conflict_flags: list[str] = []
            if meaning_conflict_status == MEANING_CONFLICT_PRESENT:
                conflict_flags.append(f"MEANING_CONFLICT_PROPOSITION_PAIR:{earlier.region_id}:{later.region_id}")
            confidence = (
                CONFIDENCE_MIXED if conflict_flags
                else CONFIDENCE_SUPPORTED if status == SUPERSESSION_SUPPORTED
                else CONFIDENCE_WEAK if status in (SUPERSESSION_PARTIAL, SUPERSESSION_NO_SAFE)
                else CONFIDENCE_UNKNOWN
            )

            hypotheses.append(WholeVideoSupersessionHypothesis(
                source_asset_id=source_asset_id,
                supersession_id=_supersession_id(source_asset_id, (earlier.region_id,), (later.region_id,)),
                earlier_region_ids=(earlier.region_id,),
                later_region_ids=(later.region_id,),
                covered_proposition_candidate_ids=covered,
                uncovered_earlier_proposition_candidate_ids=uncovered,
                coverage_status=coverage_status,
                recording_process_support=recording_process_support,
                audience_delivery_support=audience_delivery_support,
                meaning_conflict_status=meaning_conflict_status,
                supersession_status=status,
                confidence=confidence,
                conflict_flags=tuple(conflict_flags),
                provenance=("CLAIM_SIGNATURE_COMPARISON", "REGION_PROCESS_STATUS_REFERENCE"),
            ))
    return tuple(sorted(hypotheses, key=lambda h: h.supersession_id))


# ---------------------------------------------------------------------------
# TYPE 4 -- WholeVideoEditorialUnderstanding (aggregate root)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WholeVideoEditorialUnderstanding:
    """Whole-video P2 aggregate -- bounded, diagnostics-only, zero
    authority. No final edit plan, no winners, no ordering plan (see module
    docstring's "NO AUTHORITY" section)."""
    source_asset_ids: Tuple[str, ...]
    regions: Tuple[WholeVideoEditorialRegion, ...]
    proposition_realization_maps: Tuple[WholeVideoPropositionRealizationMap, ...]
    cross_source_proposition_links: Tuple[CrossSourcePropositionLink, ...]
    supersession_hypotheses: Tuple[WholeVideoSupersessionHypothesis, ...]
    global_continuity_status: str
    unresolved_conflicts: Tuple[str, ...]
    capability_status: str
    confidence: str
    provenance: Tuple[str, ...]


def _global_continuity_status(
    maps: Sequence[WholeVideoPropositionRealizationMap],
    hypotheses: Sequence[WholeVideoSupersessionHypothesis],
) -> str:
    if not maps and not hypotheses:
        return CONTINUITY_UNKNOWN
    if any(h.supersession_status == SUPERSESSION_CONFLICTED for h in hypotheses):
        return CONTINUITY_CONFLICTED
    if any(m.conflict_flags for m in maps) or any(h.supersession_status == SUPERSESSION_PARTIAL for h in hypotheses):
        return CONTINUITY_PARTIAL
    return CONTINUITY_COHERENT


def build_whole_video_editorial_understanding(
    *,
    moments_by_source: Mapping[str, Sequence[EditorialMoment]],
    local_groups_by_source: Mapping[str, Sequence[EditorialLocalGroup]],
    sequences_by_source: Mapping[str, Sequence[EditorialSequenceHypothesis]] | None = None,
    proposition_candidates_by_id: Mapping[str, PropositionCandidate] | None = None,
) -> WholeVideoEditorialUnderstanding:
    """The one canonical whole-video P2 Phase-A builder. Pure; no I/O, no
    provider call, no pipeline coupling. Supports one source or multiple
    sources (D-201's own multi-source-support requirement) -- supersession
    hypotheses are computed WITHIN each source only (Phase A scope, never
    merging identities across ``source_asset_id``); cross-source
    correspondence is represented separately, with both sides' source
    identity kept explicit (``CrossSourcePropositionLink``)."""
    sequences_by_source = sequences_by_source or {}
    proposition_candidates_by_id = proposition_candidates_by_id or {}

    all_regions: list[WholeVideoEditorialRegion] = []
    all_maps: list[WholeVideoPropositionRealizationMap] = []
    all_hypotheses: list[WholeVideoSupersessionHypothesis] = []
    source_asset_ids = tuple(sorted(moments_by_source.keys()))

    for source_asset_id in source_asset_ids:
        moments = tuple(moments_by_source.get(source_asset_id, ()))
        local_groups = tuple(local_groups_by_source.get(source_asset_id, ()))
        sequences = tuple(sequences_by_source.get(source_asset_id, ()))
        regions = build_whole_video_editorial_regions(
            source_asset_id=source_asset_id, moments=moments, local_groups=local_groups, sequences=sequences,
        )
        all_regions.extend(regions)
        maps = build_whole_video_proposition_realization_maps(
            source_asset_id=source_asset_id, moments=moments, regions=regions,
        )
        all_maps.extend(maps)
        all_hypotheses.extend(build_whole_video_supersession_hypotheses(
            source_asset_id=source_asset_id, regions=regions, proposition_candidates_by_id=proposition_candidates_by_id,
        ))

    cross_source_links = build_cross_source_proposition_links(proposition_candidates_by_id)

    unresolved_conflicts = tuple(sorted(set(
        [h.supersession_id for h in all_hypotheses if h.supersession_status == SUPERSESSION_CONFLICTED]
        + [m.proposition_candidate_id for m in all_maps if m.conflict_flags]
        + [f"{link.left_proposition_candidate_id}:{link.right_proposition_candidate_id}"
           for link in cross_source_links if link.relationship == CROSS_SOURCE_CONFLICTED]
    )))

    if not all_regions:
        capability_status = CAPABILITY_NOT_EVALUABLE
    elif any(r.confidence == CONFIDENCE_UNKNOWN for r in all_regions):
        capability_status = CAPABILITY_PARTIAL
    else:
        capability_status = CAPABILITY_AVAILABLE

    confidence = _categorical_confidence(
        [r.confidence for r in all_regions] + [m.confidence for m in all_maps] + [h.confidence for h in all_hypotheses],
        bool(unresolved_conflicts),
    )

    return WholeVideoEditorialUnderstanding(
        source_asset_ids=source_asset_ids,
        regions=tuple(all_regions),
        proposition_realization_maps=tuple(all_maps),
        cross_source_proposition_links=cross_source_links,
        supersession_hypotheses=tuple(all_hypotheses),
        global_continuity_status=_global_continuity_status(all_maps, all_hypotheses),
        unresolved_conflicts=unresolved_conflicts,
        capability_status=capability_status,
        confidence=confidence,
        provenance=("WHOLE_VIDEO_EDITORIAL_REGION", "WHOLE_VIDEO_PROPOSITION_REALIZATION_MAP", "WHOLE_VIDEO_SUPERSESSION_HYPOTHESIS"),
    )


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe, counts/status-only -- same pattern as every prior
# P1 phase's own compact summaries). No transcript dump.
# ---------------------------------------------------------------------------
def whole_video_editorial_region_diagnostics(region: WholeVideoEditorialRegion) -> dict:
    return {
        "region_id": region.region_id,
        "source_asset_id": region.source_asset_id,
        "source_start": region.source_start,
        "source_end": region.source_end,
        "moment_ids": list(region.moment_ids),
        "local_group_ids": list(region.local_group_ids),
        "sequence_ids": list(region.sequence_ids),
        "proposition_candidate_ids": list(region.proposition_candidate_ids),
        "dominant_process_status": region.dominant_process_status,
        "audience_delivery_status": region.audience_delivery_status,
        "confidence": region.confidence,
        "conflict": list(region.conflict_flags),
        "provenance": list(region.provenance),
    }


def whole_video_proposition_realization_diagnostics(row: WholeVideoPropositionRealizationMap) -> dict:
    return {
        "proposition_candidate_id": row.proposition_candidate_id,
        "source_asset_id": row.source_asset_id,
        "moment_count": len(row.moment_ids),
        "region_count": len(row.region_ids),
        "realization_count": row.realization_count,
        "relationship_status": row.relationship_status,
        "confidence": row.confidence,
        "conflict": list(row.conflict_flags),
    }


def whole_video_supersession_diagnostics(hypothesis: WholeVideoSupersessionHypothesis) -> dict:
    return {
        "supersession_id": hypothesis.supersession_id,
        "earlier_region_ids": list(hypothesis.earlier_region_ids),
        "later_region_ids": list(hypothesis.later_region_ids),
        "covered_proposition_candidate_ids": list(hypothesis.covered_proposition_candidate_ids),
        "uncovered_earlier_proposition_candidate_ids": list(hypothesis.uncovered_earlier_proposition_candidate_ids),
        "coverage_status": hypothesis.coverage_status,
        "recording_process_support": hypothesis.recording_process_support,
        "audience_delivery_support": hypothesis.audience_delivery_support,
        "meaning_conflict_status": hypothesis.meaning_conflict_status,
        "supersession_status": hypothesis.supersession_status,
        "confidence": hypothesis.confidence,
        "conflicts": list(hypothesis.conflict_flags),
    }


def whole_video_editorial_understanding_run_summary(understanding: WholeVideoEditorialUnderstanding) -> dict:
    """Pure aggregator -- the exact bounded field list this task's own
    "RUN SUMMARY" section requires. No master score."""
    regions = understanding.regions
    hypotheses = understanding.supersession_hypotheses
    maps = understanding.proposition_realization_maps

    process_counts = {status: 0 for status in ALLOWED_REGION_PROCESS_STATUSES}
    for r in regions:
        process_counts[r.dominant_process_status] = process_counts.get(r.dominant_process_status, 0) + 1

    supersession_counts = {status: 0 for status in ALLOWED_SUPERSESSION_STATUSES}
    for h in hypotheses:
        supersession_counts[h.supersession_status] = supersession_counts.get(h.supersession_status, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "whole_video_region_count": len(regions),
        "recording_process_region_count": process_counts[REGION_RECORDING_PROCESS] + process_counts[REGION_TAKE_SERIES],
        "audience_delivery_region_count": process_counts[REGION_CLEAN_DELIVERY],
        "mixed_region_count": process_counts[REGION_MIXED],
        "proposition_realization_map_count": len(maps),
        "multi_realization_proposition_count": sum(1 for m in maps if m.realization_count > 1),
        "supersession_hypothesis_count": len(hypotheses),
        "supported_supersession_count": supersession_counts[SUPERSESSION_SUPPORTED],
        "partial_supersession_count": supersession_counts[SUPERSESSION_PARTIAL],
        "no_safe_supersession_count": supersession_counts[SUPERSESSION_NO_SAFE],
        "conflicted_supersession_count": supersession_counts[SUPERSESSION_CONFLICTED],
        "unknown_supersession_count": supersession_counts[SUPERSESSION_UNKNOWN],
        "uncovered_unique_proposition_count": sum(len(h.uncovered_earlier_proposition_candidate_ids) for h in hypotheses),
        "global_conflict_count": len(understanding.unresolved_conflicts),
        "global_continuity_status": understanding.global_continuity_status,
        "capability_status": understanding.capability_status,
    }
