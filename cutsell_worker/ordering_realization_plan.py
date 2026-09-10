"""D-206: Ordering Consolidation -- Phase A, TYPED ORDERING EVIDENCE /
CONSTRAINT LAYER OVER THE EXISTING COMPOSER CONTRACT.

See ``docs/CUTSELL_DECISIONS.md`` D-205 (architecture/forensic) and D-206
(this module) for full context. D-205 proved real editorial-ordering
infrastructure already exists (``composer.py``'s ``compose_selected``,
always live; ``composer_provider.py``/``composer_openai.py``'s
``OpenAIComposerProvider``, fully built but confirmed DORMANT on every
live path) and recommended CONSOLIDATION, never a parallel engine
(D-205 Verdict C). This module is exactly that consolidation's missing
piece: a typed, deterministic, P1/P2-evidence-aware EVIDENCE/CONSTRAINT
layer that could later inform or wrap the existing composer's own
already-proven reorder-only safety contract (``composer_provider.
_repair_order``) -- it does NOT reimplement, call, or activate that
contract itself.

## What this module is NOT (binding, restated from this task's own scope)

No story scoring. No hook/problem/benefit/proof/CTA ranking. No LLM
prompt. No narrative-quality score. No provider client of any kind --
``OpenAIComposerProvider``/``safe_compose_order`` remain completely
untouched (zero import, zero call) and stay dormant on every live path
exactly as D-205 found them. No pipeline wiring: ``pipeline.py``,
``universal_clean_cut.py``, and ``brain_runtime.py`` are all unmodified
by this task. No RAW, no Family/BestTake/D-191 mutation, no Boundary/
Pacing/Renderer change, no Commercial Moment/Sales Funnel field.
**ORDERING AUTHORITY: NONE.** Every function here is a pure, offline
builder producing a diagnostics-only ``OrderedRealizationPlan`` for
qualification -- nothing calls these functions from any live pipeline
path.

## Reorder-only invariant (the existing composer's own strongest
## guarantee, reused as a structural contract here too)

Exactly like ``composer_provider._repair_order``, every builder in this
module treats the caller-supplied realization-id set as FIXED: no
function here can add, drop, or duplicate a unit. ``validate_reorder_
only_invariant`` makes this explicit and testable (set equality +
multiplicity-one on both sides).

## Atomic unit (D-205 Section 7's own recommendation, adopted as-is)

The REALIZATION, keyed by ``realization_id`` when present, falling back
to ``clip_id`` -- the EXACT SAME convention ``semantic_ledger.
_clip_realization_id`` already established (D-050B), reused verbatim,
never re-derived differently here.

## Composite representation (D-205 Section 15/24's own named gap, honest)

``DraftClip``/``CandidateTake`` carry no POSITIVE "this realization is
one piece of a 2+-piece composite, and here is its internal position"
field today (confirmed by direct code read, D-205 Section 15) --
``composite_resolver.py``'s own split-marking is a private, run-scoped
``ContextVar`` signal, never serialized onto the clip. ``OrderingUnit.
composite_component_ids`` is therefore an OPTIONAL, caller-supplied
reference (default empty) rather than something this module derives --
honestly reported as absent/unknown today, never guessed, exactly per
this task's own "audit current shapes and adapt" instruction.

## P1/P2 evidence reuse (never recomputed)

Every relation this module's builders can produce is read directly off
already-computed D-194/D-197/D-202 objects (``EditorialMoment.moment_
role``, ``EditorialLocalGroup.moment_ids`` order, ``WholeVideoSupersession
Hypothesis.supersession_status``/``meaning_conflict_status``) -- this
module imports no ASR/Language-Spine/Watch+Listen/visual/Prosodic
builder and calls no provider (module-leaf grep tests in this task's own
test file confirm this).

## Causal Order Validator (future seam only, per this task's own
## "if premature, document future seam only" instruction)

``causal_order_validator.find_causal_order_breaks`` operates on a
``CanonicalEditPlan`` (a different, richer input shape than this
module's flat ``OrderingUnit`` list) and is a VALIDATOR, never a
constructor (D-205 Section 3). Wrapping its cross-idea dependency
findings into ``OrderingRelationEvidence`` would require a real
``CanonicalEditPlan``-to-``OrderingUnit`` adapter this task does not
build -- documented here as the CAUSAL_ORDER_VALIDATOR provenance value
existing in the vocabulary for a future producer, never emitted by any
builder in this module today (module-leaf grep test confirms zero
import of ``causal_order_validator.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence, Tuple

from .editorial_moment_sequence import (
    EditorialMoment,
    MOMENT_ROLE_CONTINUATION,
    MOMENT_ROLE_CORRECTION,
)
from .editorial_moment_sequence_integration import EditorialLocalGroup
from .language_utterance_attempt import (
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
)
from .whole_video_editorial_reasoning import (
    SUPERSESSION_CONFLICTED,
    SUPERSESSION_PARTIAL,
    SUPERSESSION_SUPPORTED,
    WholeVideoSupersessionHypothesis,
)

SCHEMA_VERSION = "cutsell.ordering_realization_plan.v1"

# ---------------------------------------------------------------------------
# Ordering-relation vocabulary (D-206's own, small, closed set).
# ---------------------------------------------------------------------------
RELATION_MUST_PRECEDE = "MUST_PRECEDE"
RELATION_MUST_FOLLOW = "MUST_FOLLOW"
RELATION_PRESERVE_INTERNAL_ORDER = "PRESERVE_INTERNAL_ORDER"
RELATION_NO_CONSTRAINT = "NO_ORDER_CONSTRAINT"
RELATION_CONFLICTED = "CONFLICTED"
RELATION_UNKNOWN = "UNKNOWN"
ALLOWED_ORDERING_RELATIONS: frozenset[str] = frozenset({
    RELATION_MUST_PRECEDE, RELATION_MUST_FOLLOW, RELATION_PRESERVE_INTERNAL_ORDER,
    RELATION_NO_CONSTRAINT, RELATION_CONFLICTED, RELATION_UNKNOWN,
})

# ---------------------------------------------------------------------------
# Ordering-reason (provenance) vocabulary -- this task's own bounded list,
# never over-expanded.
# ---------------------------------------------------------------------------
REASON_P1_LOCAL_SEQUENCE = "P1_LOCAL_SEQUENCE"
REASON_P1_CONTINUATION = "P1_CONTINUATION"
REASON_P1_CORRECTION = "P1_CORRECTION"
REASON_P2_PROPOSITION_PROGRESSION = "P2_PROPOSITION_PROGRESSION"
REASON_P2_GLOBAL_CONTINUITY = "P2_GLOBAL_CONTINUITY"
REASON_COMPOSITE_INTERNAL_ORDER = "COMPOSITE_INTERNAL_ORDER"
REASON_SOURCE_CHRONOLOGY_FALLBACK = "SOURCE_CHRONOLOGY_FALLBACK"
# Declared for a future producer only -- never emitted by any builder in
# this module today (module docstring's own "Causal Order Validator"
# section; confirmed by module-leaf grep test).
REASON_CAUSAL_ORDER_VALIDATOR = "CAUSAL_ORDER_VALIDATOR"
REASON_UNIQUE_INFORMATION = "UNIQUE_INFORMATION"
REASON_MEANING_FIREWALL = "MEANING_FIREWALL"

# ---------------------------------------------------------------------------
# Ordering-status vocabulary (plan-level).
# ---------------------------------------------------------------------------
ORDERING_STATUS_ORDERED = "ORDERED"
ORDERING_STATUS_PARTIALLY_ORDERED = "PARTIALLY_ORDERED"
ORDERING_STATUS_CONFLICTED = "CONFLICTED"
ORDERING_STATUS_UNKNOWN = "UNKNOWN"
ALLOWED_ORDERING_STATUSES: frozenset[str] = frozenset({
    ORDERING_STATUS_ORDERED, ORDERING_STATUS_PARTIALLY_ORDERED,
    ORDERING_STATUS_CONFLICTED, ORDERING_STATUS_UNKNOWN,
})

# Categorical confidence -- reused verbatim from language_utterance_attempt.py
# (the same vocabulary every P1/P2 type already uses). No numeric score
# anywhere in this module.
_CONFIDENCE_RANK: Mapping[str, int] = {
    CONFIDENCE_SUPPORTED: 3, CONFIDENCE_WEAK: 2, CONFIDENCE_MIXED: 1, CONFIDENCE_UNKNOWN: 0,
}


def _categorical_confidence(conflict_present: bool, has_positive_evidence: bool) -> str:
    if conflict_present:
        return CONFIDENCE_MIXED
    if has_positive_evidence:
        return CONFIDENCE_SUPPORTED
    return CONFIDENCE_UNKNOWN


# ---------------------------------------------------------------------------
# TYPE 1 -- OrderingUnit
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrderingUnit:
    """One frozen selected realization -- a thin, referencing wrapper,
    never a copy of any upstream object. ``realization_id`` is the SAME
    identity ``semantic_ledger._clip_realization_id`` already uses
    (``realization_id`` if set, else ``clip_id``) -- never re-derived
    differently here."""
    realization_id: str
    source_asset_id: str
    source_span_ids: Tuple[str, ...]
    proposition_candidate_ids: Tuple[str, ...]
    p1_local_group_id: str | None
    p1_sequence_ids: Tuple[str, ...]
    p2_region_ids: Tuple[str, ...]
    # D-205 Section 15/24's own named gap (module docstring): empty unless
    # a caller explicitly supplies it -- no live signal exists today.
    composite_component_ids: Tuple[str, ...]
    source_start: float
    source_end: float
    source_order: int
    provenance: Tuple[str, ...]


def _realization_identity(realization: object) -> str:
    """The exact ``semantic_ledger._clip_realization_id`` convention,
    reused verbatim (never redefined)."""
    return str(getattr(realization, "realization_id", None) or realization.clip_id)


def build_ordering_units(
    *,
    realizations: Sequence[object],
    moments_by_source: Mapping[str, Sequence[EditorialMoment]] | None = None,
    local_groups_by_source: Mapping[str, Sequence[EditorialLocalGroup]] | None = None,
    sequence_ids_by_moment_id: Mapping[str, Tuple[str, ...]] | None = None,
    region_ids_by_moment_id: Mapping[str, Tuple[str, ...]] | None = None,
    composite_component_ids_by_realization: Mapping[str, Tuple[str, ...]] | None = None,
) -> Tuple[OrderingUnit, ...]:
    """The one canonical realization -> ``OrderingUnit`` builder. Pure; no
    I/O, no provider call, no numeric time-gap segmentation. Every P1/P2
    reference is a lookup into ALREADY-COMPUTED objects the caller
    supplies -- nothing here recomputes a moment role, a local group, a
    sequence, or a region. Deterministic, input-order-independent output
    (sorted by ``(source_order, source_start, realization_id)``)."""
    moments_by_source = moments_by_source or {}
    local_groups_by_source = local_groups_by_source or {}
    sequence_ids_by_moment_id = sequence_ids_by_moment_id or {}
    region_ids_by_moment_id = region_ids_by_moment_id or {}
    composite_component_ids_by_realization = composite_component_ids_by_realization or {}

    units: list[OrderingUnit] = []
    for realization in realizations:
        source_asset_id = str(realization.source_asset_id)
        source_span_id = getattr(realization, "source_span_id", None)
        source_span_ids = (str(source_span_id),) if source_span_id else ()
        realization_id = _realization_identity(realization)

        moments = [
            m for m in moments_by_source.get(source_asset_id, ())
            if source_span_id and m.source_span_id == source_span_id
        ]
        proposition_candidate_ids = tuple(dict.fromkeys(
            pid for m in moments for pid in m.proposition_candidate_ids
        ))
        local_group_id: str | None = None
        for group in local_groups_by_source.get(source_asset_id, ()):
            if any(m.editorial_moment_id in group.moment_ids for m in moments):
                local_group_id = group.group_id
                break
        sequence_ids = tuple(dict.fromkeys(
            sid for m in moments for sid in sequence_ids_by_moment_id.get(m.editorial_moment_id, ())
        ))
        region_ids = tuple(dict.fromkeys(
            rid for m in moments for rid in region_ids_by_moment_id.get(m.editorial_moment_id, ())
        ))

        units.append(OrderingUnit(
            realization_id=realization_id,
            source_asset_id=source_asset_id,
            source_span_ids=source_span_ids,
            proposition_candidate_ids=proposition_candidate_ids,
            p1_local_group_id=local_group_id,
            p1_sequence_ids=sequence_ids,
            p2_region_ids=region_ids,
            composite_component_ids=tuple(composite_component_ids_by_realization.get(realization_id, ())),
            source_start=float(realization.start),
            source_end=float(realization.end),
            source_order=int(getattr(realization, "source_order", 0)),
            provenance=("FROZEN_REALIZATION_REFERENCE",),
        ))
    return tuple(sorted(units, key=lambda u: (u.source_order, u.source_start, u.realization_id)))


# ---------------------------------------------------------------------------
# TYPE 2 -- OrderingRelationEvidence
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrderingRelationEvidence:
    """One pairwise ordering constraint -- EVIDENCE only, never an
    authority. ``relation_id`` is deterministic (never randomly
    generated): minted from ``(left, right, relation, reason)`` --
    direction matters, so this is never sorted/canonicalized the way an
    undirected id would be."""
    relation_id: str
    left_realization_id: str
    right_realization_id: str
    ordering_relation: str
    ordering_reason: str
    confidence: str
    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]


def _relation_id(left: str, right: str, relation: str, reason: str) -> str:
    raw = "|".join((left, right, relation, reason)).encode("utf-8")
    return "ordrel_" + hashlib.sha256(raw).hexdigest()[:20]


def _units_by_moment_id(units: Sequence[OrderingUnit], moments_by_id: Mapping[str, EditorialMoment]) -> Mapping[str, str]:
    """moment_id -> owning realization_id, for every unit whose source_span_id
    matches a real moment. A moment with no owning unit (the common case --
    most P1 candidates never survive to Freeze) simply has no entry."""
    span_to_unit: dict[str, str] = {}
    for unit in units:
        for span_id in unit.source_span_ids:
            span_to_unit[span_id] = unit.realization_id
    result: dict[str, str] = {}
    for moment_id, moment in moments_by_id.items():
        if moment.source_span_id and moment.source_span_id in span_to_unit:
            result[moment_id] = span_to_unit[moment.source_span_id]
    return result


def build_ordering_relation_evidence(
    *,
    units: Sequence[OrderingUnit],
    moments_by_id: Mapping[str, EditorialMoment] | None = None,
    local_groups_by_id: Mapping[str, EditorialLocalGroup] | None = None,
    supersession_hypotheses: Sequence[WholeVideoSupersessionHypothesis] = (),
    region_ids_by_unit: Mapping[str, Tuple[str, ...]] | None = None,
    composite_group_by_realization: Mapping[str, str] | None = None,
) -> Tuple[OrderingRelationEvidence, ...]:
    """The one canonical relation-evidence builder. Pure; reads only
    ALREADY-COMPUTED P1 moment roles / local-group order and P2
    supersession hypotheses -- never re-derives a relation, never calls a
    provider. Deterministic, input-order-independent (a stable sort at
    the end).

    ``composite_group_by_realization`` is OPTIONAL, caller-supplied
    evidence (no live field carries this today, D-205 Section 15/24's own
    named gap, module docstring) -- when given, members sharing one
    composite-group id are chained with ``PRESERVE_INTERNAL_ORDER``
    relations in their own ``source_start`` order, so a future global
    reorder can move the WHOLE composite without scrambling its pieces."""
    moments_by_id = moments_by_id or {}
    local_groups_by_id = local_groups_by_id or {}
    region_ids_by_unit = region_ids_by_unit or {}
    composite_group_by_realization = composite_group_by_realization or {}
    relations: list[OrderingRelationEvidence] = []
    seen: set[tuple] = set()

    def _emit(left: str, right: str, relation: str, reason: str, *, conflict_flags: Tuple[str, ...] = ()) -> None:
        key = (left, right, relation, reason)
        if key in seen:
            return
        seen.add(key)
        confidence = _categorical_confidence(bool(conflict_flags), True)
        relations.append(OrderingRelationEvidence(
            relation_id=_relation_id(left, right, relation, reason),
            left_realization_id=left, right_realization_id=right,
            ordering_relation=relation, ordering_reason=reason, confidence=confidence,
            conflict_flags=conflict_flags, provenance=("D206_DETERMINISTIC_BUILDER",),
        ))

    # --- 1. Local-sequence / composite internal order: consecutive
    # moments within one already-computed EditorialLocalGroup, restricted
    # to moments whose owning unit is actually present. ---
    moment_owner = _units_by_moment_id(units, moments_by_id)
    for group in local_groups_by_id.values():
        ordered_owned = [
            (mid, moment_owner[mid]) for mid in group.moment_ids if mid in moment_owner
        ]
        for (left_mid, left_unit), (right_mid, right_unit) in zip(ordered_owned, ordered_owned[1:]):
            if left_unit == right_unit:
                continue
            left_role = moments_by_id[left_mid].moment_role
            right_role = moments_by_id[right_mid].moment_role
            if right_role == MOMENT_ROLE_CONTINUATION:
                _emit(left_unit, right_unit, RELATION_MUST_PRECEDE, REASON_P1_CONTINUATION)
            elif right_role == MOMENT_ROLE_CORRECTION:
                _emit(left_unit, right_unit, RELATION_MUST_PRECEDE, REASON_P1_CORRECTION)
            else:
                _emit(left_unit, right_unit, RELATION_PRESERVE_INTERNAL_ORDER, REASON_P1_LOCAL_SEQUENCE)
            del left_role  # read for clarity only; not otherwise used

    # --- 1b. Composite internal order (caller-supplied evidence only,
    # module docstring's own honest gap). ---
    unit_by_id = {u.realization_id: u for u in units}
    groups: dict[str, list[str]] = {}
    for realization_id, group_id in composite_group_by_realization.items():
        if realization_id in unit_by_id:
            groups.setdefault(group_id, []).append(realization_id)
    for members in groups.values():
        ordered_members = sorted(members, key=lambda rid: (unit_by_id[rid].source_start, rid))
        for left_unit, right_unit in zip(ordered_members, ordered_members[1:]):
            _emit(left_unit, right_unit, RELATION_PRESERVE_INTERNAL_ORDER, REASON_COMPOSITE_INTERNAL_ORDER)

    # --- 2. P2 supersession-derived conflicts. Never a deletion signal --
    # only ever a CONFLICTED relation naming the affected units, per this
    # task's own "return conflict / duplicate-presence diagnostic" rule. ---
    for hyp in supersession_hypotheses:
        earlier_units = tuple(dict.fromkeys(
            u for rid in hyp.earlier_region_ids for u, unit_regions in region_ids_by_unit.items()
            if rid in unit_regions
        ))
        later_units = tuple(dict.fromkeys(
            u for rid in hyp.later_region_ids for u, unit_regions in region_ids_by_unit.items()
            if rid in unit_regions
        ))
        if not earlier_units or not later_units:
            continue
        if hyp.supersession_status == SUPERSESSION_CONFLICTED:
            for left in earlier_units:
                for right in later_units:
                    _emit(left, right, RELATION_CONFLICTED, REASON_MEANING_FIREWALL,
                          conflict_flags=("MEANING_CONFLICT_BOTH_SURVIVED",))
        elif hyp.supersession_status in (SUPERSESSION_SUPPORTED, SUPERSESSION_PARTIAL):
            # Both the "superseded" and "superseding" side legitimately
            # remain as frozen units -- this module has no delete
            # authority (module docstring), so this is reported as an
            # upstream duplicate/supersession inconsistency, never
            # silently resolved.
            for left in earlier_units:
                for right in later_units:
                    _emit(left, right, RELATION_CONFLICTED, REASON_P2_GLOBAL_CONTINUITY,
                          conflict_flags=("SUPERSESSION_SURVIVAL_CONFLICT",))

    return tuple(sorted(relations, key=lambda r: (r.left_realization_id, r.right_realization_id, r.ordering_relation)))


# ---------------------------------------------------------------------------
# TYPE 3 -- OrderedRealizationPlan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrderedUnitPlacement:
    """One unit's final placement -- position + why, never a copy of the
    unit itself (reference by ``realization_id`` only)."""
    realization_id: str
    ordering_position: int
    ordering_reason: str
    continuity_status: str  # "CONSTRAINT_SUPPORTED" | "FALLBACK" | "UNKNOWN"
    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]


@dataclass(frozen=True)
class OrderedRealizationPlan:
    """The final, bounded plan -- diagnostics/qualification only. No cut
    frames, no pacing instructions, no render instructions, no
    ``selected_clip_id``/winner/delete field anywhere on this type."""
    source_asset_ids: Tuple[str, ...]
    ordered_realization_ids: Tuple[str, ...]
    ordered_units: Tuple[OrderedUnitPlacement, ...]
    ordering_status: str
    fallback_used: bool
    fallback_reason: str
    unresolved_relation_ids: Tuple[str, ...]
    conflict_flags: Tuple[str, ...]
    confidence: str
    provenance: Tuple[str, ...]


CONTINUITY_CONSTRAINT_SUPPORTED = "CONSTRAINT_SUPPORTED"
CONTINUITY_FALLBACK = "FALLBACK"
CONTINUITY_UNKNOWN = "UNKNOWN"


def build_deterministic_ordering_plan(
    *,
    units: Sequence[OrderingUnit],
    relations: Sequence[OrderingRelationEvidence],
) -> OrderedRealizationPlan:
    """The one canonical deterministic ordering-plan builder. Pure; no
    provider, no numeric optimization score. Algorithm (this task's own
    principle order):

    1. Build a precedence graph from ``MUST_PRECEDE``/``PRESERVE_INTERNAL_
       ORDER`` edges (``MUST_FOLLOW(left, right)`` is normalized to the
       identical constraint ``MUST_PRECEDE(right, left)`` -- both express
       the same directed edge from a different side).
    2. Kahn's-algorithm topological placement: at each step, among the
       units with zero remaining unsatisfied incoming constraint, pick
       the STABLE-FALLBACK-lowest one (``source_order``, ``source_start``,
       ``realization_id``) -- this is what makes chronology a TIE-BREAK
       only, never authority, and guarantees input-order independence.
    3. A genuine cycle (contradictory constraints, e.g. A MUST_PRECEDE B
       and B MUST_PRECEDE A) cannot be resolved by construction: the
       whole cyclic subset falls back to stable order, is recorded in
       ``unresolved_relation_ids``, and forces ``ordering_status ==
       CONFLICTED`` -- a deterministic TOTAL list is still emitted
       (downstream renderer contract, D-205 Section 29), but the semantic
       status is never upgraded past the truth.
    4. An explicit ``CONFLICTED`` relation (supersession-survival /
       meaning-firewall, never a precedence contradiction) never blocks
       placing OTHER, unrelated units -- it only marks its own two units'
       relation as unresolved and forces the overall status to at least
       ``CONFLICTED`` (this task's own "no forced topological result
       presented as valid" instruction, applied narrowly).

    Structurally guarantees the reorder-only invariant: every input
    ``realization_id`` appears in the output exactly once, never more,
    never fewer (see ``validate_reorder_only_invariant``).
    """
    unit_by_id = {u.realization_id: u for u in units}
    ids = tuple(u.realization_id for u in units)

    # --- Step 1: precedence graph. ---
    precedes: dict[str, set[str]] = {rid: set() for rid in ids}   # a -> {b, ...} means a precedes b
    edge_reason: dict[tuple[str, str], str] = {}
    unresolved: list[str] = []
    conflict_flags: set[str] = set()
    has_conflicted_relation = False

    for rel in relations:
        if rel.left_realization_id not in unit_by_id or rel.right_realization_id not in unit_by_id:
            continue
        if rel.ordering_relation == RELATION_MUST_PRECEDE or rel.ordering_relation == RELATION_PRESERVE_INTERNAL_ORDER:
            precedes[rel.left_realization_id].add(rel.right_realization_id)
            edge_reason[(rel.left_realization_id, rel.right_realization_id)] = rel.ordering_reason
        elif rel.ordering_relation == RELATION_MUST_FOLLOW:
            precedes[rel.right_realization_id].add(rel.left_realization_id)
            edge_reason[(rel.right_realization_id, rel.left_realization_id)] = rel.ordering_reason
        elif rel.ordering_relation == RELATION_CONFLICTED:
            has_conflicted_relation = True
            unresolved.append(rel.relation_id)
            conflict_flags.update(rel.conflict_flags or (rel.ordering_reason,))

    # --- Step 2: Kahn's algorithm with stable-fallback tie-break. ---
    in_degree: dict[str, int] = {rid: 0 for rid in ids}
    for src, dsts in precedes.items():
        for dst in dsts:
            in_degree[dst] += 1

    fallback_key = {rid: (unit_by_id[rid].source_order, unit_by_id[rid].source_start, rid) for rid in ids}
    remaining = set(ids)
    placed: list[str] = []
    placement_reason: dict[str, str] = {}
    placement_status: dict[str, str] = {}
    any_fallback_break = False

    while remaining:
        ready = sorted((rid for rid in remaining if in_degree[rid] == 0), key=lambda rid: fallback_key[rid])
        if not ready:
            # A genuine cycle among the remaining nodes: break it
            # deterministically via stable fallback, record every
            # cyclic edge as unresolved.
            cyclic = sorted(remaining, key=lambda rid: fallback_key[rid])
            for a in cyclic:
                for b in precedes.get(a, ()):
                    if b in remaining:
                        unresolved.append(f"CYCLE:{a}->{b}")
            any_fallback_break = True
            pick = cyclic[0]
        else:
            pick = ready[0]
        placed.append(pick)
        remaining.discard(pick)
        # A node is CONSTRAINT_SUPPORTED if it participates in ANY real
        # precedence edge -- as the edge's source (it precedes something)
        # or its destination (something precedes it). Looking at incoming
        # edges alone would wrongly mark the very FIRST node of a chain
        # (which has no incoming edge by definition) as unconstrained.
        reasons_incoming = {edge_reason[(src, pick)] for src in ids if pick in precedes.get(src, ()) and (src, pick) in edge_reason}
        reasons_outgoing = {edge_reason[(pick, dst)] for dst in precedes.get(pick, ()) if (pick, dst) in edge_reason}
        reasons = reasons_incoming | reasons_outgoing
        if reasons:
            placement_reason[pick] = sorted(reasons)[0]
            placement_status[pick] = CONTINUITY_CONSTRAINT_SUPPORTED
        else:
            placement_reason[pick] = REASON_SOURCE_CHRONOLOGY_FALLBACK
            placement_status[pick] = CONTINUITY_FALLBACK if len(ids) > 1 else CONTINUITY_UNKNOWN
        for dst in precedes.get(pick, ()):
            if dst in in_degree:
                in_degree[dst] -= 1

    fallback_used = any_fallback_break or any(status == CONTINUITY_FALLBACK for status in placement_status.values())
    has_any_constraint = any(precedes[rid] for rid in ids)

    if any_fallback_break or has_conflicted_relation:
        ordering_status = ORDERING_STATUS_CONFLICTED
    elif not has_any_constraint:
        ordering_status = ORDERING_STATUS_UNKNOWN
    elif fallback_used:
        ordering_status = ORDERING_STATUS_PARTIALLY_ORDERED
    else:
        ordering_status = ORDERING_STATUS_ORDERED

    fallback_reason = (
        "constraint_cycle_detected" if any_fallback_break
        else "no_positive_editorial_relation" if fallback_used
        else ""
    )

    ordered_units = tuple(
        OrderedUnitPlacement(
            realization_id=rid, ordering_position=index, ordering_reason=placement_reason[rid],
            continuity_status=placement_status[rid],
            conflict_flags=tuple(sorted(conflict_flags)) if rid in {
                r.left_realization_id for r in relations if r.ordering_relation == RELATION_CONFLICTED
            } | {r.right_realization_id for r in relations if r.ordering_relation == RELATION_CONFLICTED} else (),
            provenance=("D206_DETERMINISTIC_BUILDER",),
        )
        for index, rid in enumerate(placed)
    )

    return OrderedRealizationPlan(
        source_asset_ids=tuple(sorted({u.source_asset_id for u in units})),
        ordered_realization_ids=tuple(placed),
        ordered_units=ordered_units,
        ordering_status=ordering_status,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        unresolved_relation_ids=tuple(dict.fromkeys(unresolved)),
        conflict_flags=tuple(sorted(conflict_flags)),
        confidence=_categorical_confidence(bool(conflict_flags), has_any_constraint),
        provenance=("D206_DETERMINISTIC_BUILDER",),
    )


def validate_reorder_only_invariant(units: Sequence[OrderingUnit], plan: OrderedRealizationPlan) -> bool:
    """The one canonical reorder-only proof -- exactly this task's own
    wording: ``set(input realization ids) == set(output realization ids)``
    and multiplicity exactly 1 on both sides. Mirrors ``composer_
    provider._repair_order``'s own already-proven invariant, restated as
    an independently-callable check rather than duplicated logic."""
    input_ids = [u.realization_id for u in units]
    output_ids = list(plan.ordered_realization_ids)
    if len(input_ids) != len(set(input_ids)) or len(output_ids) != len(set(output_ids)):
        return False
    return set(input_ids) == set(output_ids) and len(input_ids) == len(output_ids)


# ---------------------------------------------------------------------------
# Diagnostics / run summary -- tail-safe, counts/status-only, no transcript,
# no QA reference, no master score.
# ---------------------------------------------------------------------------
def ordering_unit_diagnostics(unit: OrderingUnit) -> dict:
    return {
        "realization_id": unit.realization_id,
        "source_asset_id": unit.source_asset_id,
        "source_span_ids": list(unit.source_span_ids),
        "proposition_candidate_ids": list(unit.proposition_candidate_ids),
        "p1_local_group_id": unit.p1_local_group_id,
        "p1_sequence_ids": list(unit.p1_sequence_ids),
        "p2_region_ids": list(unit.p2_region_ids),
        "composite_component_ids": list(unit.composite_component_ids),
        "source_start": unit.source_start,
        "source_end": unit.source_end,
        "source_order": unit.source_order,
        "provenance": list(unit.provenance),
    }


def ordering_relation_diagnostics(relation: OrderingRelationEvidence) -> dict:
    return {
        "relation_id": relation.relation_id,
        "left_realization_id": relation.left_realization_id,
        "right_realization_id": relation.right_realization_id,
        "ordering_relation": relation.ordering_relation,
        "ordering_reason": relation.ordering_reason,
        "confidence": relation.confidence,
        "conflict_flags": list(relation.conflict_flags),
        "provenance": list(relation.provenance),
    }


def ordered_realization_plan_diagnostics(plan: OrderedRealizationPlan) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_asset_ids": list(plan.source_asset_ids),
        "ordered_realization_ids": list(plan.ordered_realization_ids),
        "ordered_units": [
            {
                "realization_id": u.realization_id, "ordering_position": u.ordering_position,
                "ordering_reason": u.ordering_reason, "continuity_status": u.continuity_status,
                "conflict_flags": list(u.conflict_flags), "provenance": list(u.provenance),
            }
            for u in plan.ordered_units
        ],
        "ordering_status": plan.ordering_status,
        "fallback_used": plan.fallback_used,
        "fallback_reason": plan.fallback_reason,
        "unresolved_relation_ids": list(plan.unresolved_relation_ids),
        "conflict_flags": list(plan.conflict_flags),
        "confidence": plan.confidence,
        "provenance": list(plan.provenance),
    }


def ordered_realization_plan_run_summary(plan: OrderedRealizationPlan) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_count": len(plan.ordered_realization_ids),
        "ordering_status": plan.ordering_status,
        "fallback_used": plan.fallback_used,
        "constraint_supported_count": sum(
            1 for u in plan.ordered_units if u.continuity_status == CONTINUITY_CONSTRAINT_SUPPORTED
        ),
        "fallback_placement_count": sum(
            1 for u in plan.ordered_units if u.continuity_status == CONTINUITY_FALLBACK
        ),
        "unresolved_relation_count": len(plan.unresolved_relation_ids),
        "conflict_flag_count": len(plan.conflict_flags),
        "confidence": plan.confidence,
    }
