"""D-207: Ordering Consolidation -- Phase B, EXISTING COMPOSER ADAPTER +
TYPED PROPOSAL VALIDATION. OFFLINE / MOCKED ONLY.

See ``docs/CUTSELL_DECISIONS.md`` D-205 (forensic), D-206 (typed ordering
foundation) and D-207 (this module) for full context. D-206 built the
typed evidence/constraint layer (``OrderingUnit``/``OrderingRelation
Evidence``/``OrderedRealizationPlan``) over the EXISTING composer
contract without touching it. This module is the next, still purely
offline, consolidation step: an ADAPTER between D-206's typed objects and
the existing composer's real input/output shape, plus a VALIDATOR that
decides whether a composer-generated reorder proposal may be ACCEPTED or
must FALL BACK to D-206's own deterministic baseline plan.

## What this module is NOT (binding, restated from this task's own scope)

No story scoring. No hook/problem/benefit/proof/CTA ranking. No new
narrative model. No new provider prompt. No new LLM abstraction. No
second composer -- ``composer.py``'s ``compose_selected`` and
``composer_provider.py``'s ``safe_compose_order``/``ComposerProvider``
Protocol are used EXACTLY as they already exist, unmodified, imported
only. ``composer_openai.py``'s ``OpenAIComposerProvider`` is never
imported here at all (grep-checked by this module's own test file) --
every test exercises the ``ComposerProvider`` Protocol with a MOCK/FAKE
implementation only, never a real network call, never an API key.
No pipeline wiring: ``pipeline.py``, ``universal_clean_cut.py``, and
``brain_runtime.py`` are all unmodified by this task. No RAW, no Family/
BestTake/D-191 mutation, no Boundary/Pacing/Renderer change, no
Commercial Moment/Sales Funnel field. **ORDERING AUTHORITY: NONE.**
Every function here is a pure, offline validator producing a
diagnostics-only ``OrderingComposerProposalResult`` -- nothing calls
these functions from any live pipeline path, and an "ACCEPTED" proposal
here mutates no real edit; it only reports that a proposal SATISFIES
D-206's own typed constraints.

## Composer input identity (the adapter's actual job)

The existing composer speaks ``clip_id`` only: ``ComposerProvider.order``
takes/returns clip ids, and ``composer_provider.safe_compose_order``'s
own reorder-only repair (``_repair_order``) operates on the SAME
``clip_id`` domain. D-206's ``OrderingUnit`` is keyed by
``realization_id`` (``realization_id`` if set on the underlying take,
else its ``clip_id`` -- the ``semantic_ledger._clip_realization_id``
convention). This module's ``_build_identity_map`` is the one place that
resolves the two domains: it looks up, for every ``OrderingUnit``, the
REAL underlying take object the caller supplies (the same objects passed
to ``ordering_realization_plan.build_ordering_units``), and builds an
explicit ``realization_id <-> clip_id`` map -- never assuming the two
strings are equal, and treating two distinct realizations resolving to
the same ``clip_id`` as an identity gap rather than silently picking one.
No change to ``composer.py``/``composer_provider.py`` was needed or made
-- pure adapter-side translation, exactly as this task instructed.

## Composite internal order (honest gap, restated from D-206)

The existing composer has no concept of a "composite" at all -- it is
handed every component realization as an ordinary, independent
``CandidateTake`` and is free to reorder them arbitrarily. This module
never asks the composer to treat a composite atomically; instead, it
VALIDATES after the fact that any ``PRESERVE_INTERNAL_ORDER`` relation
D-206 already encoded (whether from local-sequence or, when the caller
supplied composite-group evidence, from ``REASON_COMPOSITE_INTERNAL_
ORDER``) is still satisfied in the proposed order, rejecting the whole
proposal if not.

## Baseline conflict gate (never let the composer "resolve" a conflict)

If D-206's own deterministic baseline plan is already ``CONFLICTED`` (a
genuine precedence cycle, or any ``CONFLICTED`` relation such as a
meaning-firewall or supersession-survival conflict), this module never
even invokes the composer: ``NOT_EVALUABLE`` is returned immediately with
the baseline retained, exactly per this task's own "composer is NOT
allowed to magically resolve semantic conflict" instruction. No P2/
meaning/unique-information/supersession-survival logic is recomputed
here -- this module only reads the ``ordering_relation``/``ordering_
status`` fields D-206's builders already computed.

## Causal Order Validator (future seam only, unchanged from D-206)

``causal_order_validator.find_causal_order_breaks`` operates on a
``CanonicalEditPlan`` (a different, richer input shape than this
module's flat ``OrderingUnit``/proposal). No adapter from a composer
proposal to a ``CanonicalEditPlan`` is built this task -- documented as a
future seam (``CAUSAL_VALIDATOR_NOT_INTEGRATED`` in every diagnostic
result); ``causal_order_validator.py`` is never imported here (module-
leaf grep test confirms this).

## One proposal, no retry (restated from this task's own instruction)

``validate_composer_ordering_proposal`` calls the supplied
``ComposerProvider`` at most ONCE per invocation (via the existing,
unmodified ``safe_compose_order``, which already fails open to the
natural/baseline order on any provider exception). A rejected proposal
is never retried, re-prompted, or "asked to try again" -- it falls back
to D-206's own already-computed baseline plan, returned verbatim (never
recomputed through a different algorithm).

## No repair that changes membership (stricter than the existing composer)

``composer_provider.safe_compose_order`` already has its own reorder-only
repair (``_repair_order``): a raw provider proposal that dropped, added,
duplicated, or invented a clip id is silently corrected before it is
even returned. This task's own instruction is stricter for THIS module:
"No repair that changes membership" -- so this validator never accepts a
silently-repaired result as if it were a clean proposal. It reads
``ComposerProviderResult.reason`` for the ``"provider_output_repaired"``
marker ``safe_compose_order`` already sets whenever its own repair had to
run, and treats that marker itself as a membership violation --
``REJECTED_REORDER_ONLY``, falling back to D-206's baseline, never to the
lower layer's repaired list. Dropped/added/duplicated/unknown-id cases
all surface through this one shared, already-proven signal; this module
does not re-implement (and cannot reliably re-derive, since the raw
pre-repair proposal is not exposed) a finer-grained split between them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from .composer import compose_selected
from .composer_provider import ComposerProvider, safe_compose_order
from .contracts import EditStrategy, SemanticLabel
from .ordering_realization_plan import (
    ORDERING_STATUS_CONFLICTED,
    REASON_COMPOSITE_INTERNAL_ORDER,
    RELATION_CONFLICTED,
    RELATION_MUST_FOLLOW,
    RELATION_MUST_PRECEDE,
    RELATION_PRESERVE_INTERNAL_ORDER,
    OrderedRealizationPlan,
    OrderingRelationEvidence,
    OrderingUnit,
)

SCHEMA_VERSION = "cutsell.ordering_composer_adapter.v1"

# ---------------------------------------------------------------------------
# Proposal-status vocabulary (this task's own closed set).
# ---------------------------------------------------------------------------
PROPOSAL_ACCEPTED = "ACCEPTED"
PROPOSAL_REJECTED_CONSTRAINT = "REJECTED_CONSTRAINT"
PROPOSAL_REJECTED_REORDER_ONLY = "REJECTED_REORDER_ONLY"
PROPOSAL_REJECTED_IDENTITY = "REJECTED_IDENTITY"
PROPOSAL_REJECTED_CONFLICT = "REJECTED_CONFLICT"
PROPOSAL_FALLBACK_BASELINE = "FALLBACK_BASELINE"
PROPOSAL_NOT_EVALUABLE = "NOT_EVALUABLE"
ALLOWED_PROPOSAL_STATUSES: frozenset[str] = frozenset({
    PROPOSAL_ACCEPTED, PROPOSAL_REJECTED_CONSTRAINT, PROPOSAL_REJECTED_REORDER_ONLY,
    PROPOSAL_REJECTED_IDENTITY, PROPOSAL_REJECTED_CONFLICT, PROPOSAL_FALLBACK_BASELINE,
    PROPOSAL_NOT_EVALUABLE,
})

# Which real path produced (or declined to produce) a proposal.
COMPOSER_PATH_MOCK_PROVIDER = "MOCK_PROVIDER"
COMPOSER_PATH_NO_PROVIDER_NATURAL = "NO_PROVIDER_NATURAL"
COMPOSER_PATH_NOT_EVALUATED_IDENTITY_GAP = "NOT_EVALUATED_IDENTITY_GAP"
COMPOSER_PATH_NOT_EVALUATED_BASELINE_CONFLICTED = "NOT_EVALUATED_BASELINE_CONFLICTED"

# Never anything but this today -- causal_order_validator.py is not
# imported/invoked anywhere in this module (see module docstring).
CAUSAL_VALIDATOR_NOT_INTEGRATED = "NOT_INTEGRATED_INPUT_SHAPE_MISMATCH"

_HARD_ORDER_RELATIONS: frozenset[str] = frozenset({
    RELATION_MUST_PRECEDE, RELATION_MUST_FOLLOW, RELATION_PRESERVE_INTERNAL_ORDER,
})


# ---------------------------------------------------------------------------
# Result type -- diagnostics only, no edit-action field anywhere.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrderingComposerProposalResult:
    """One validated (or declined) composer reorder proposal. Never
    carries a cut/render/select/delete field -- this is a proposal-
    VALIDATION record, not an edit action."""
    proposal_status: str
    composer_path: str
    proposed_realization_ids: Tuple[str, ...]
    accepted_realization_ids: Tuple[str, ...]
    identity_valid: bool
    membership_valid: bool
    composite_order_valid: bool
    violated_relation_ids: Tuple[str, ...]
    constraint_violation_count: int
    validation_flags: Tuple[str, ...]
    causal_validator_status: str
    fallback_used: bool
    fallback_reason: str
    underlying_ordering_status: str
    baseline_realization_ids: Tuple[str, ...]
    provenance: Tuple[str, ...]


@dataclass(frozen=True)
class _IdentityMap:
    take_by_realization_id: Mapping[str, object]
    clip_id_to_realization_id: Mapping[str, str]


def _composer_identity(realization: object) -> str:
    """The EXACT SAME identity convention ``ordering_realization_plan.
    _realization_identity``/``semantic_ledger._clip_realization_id``
    already use -- reused verbatim (never re-derived differently here)."""
    return str(getattr(realization, "realization_id", None) or realization.clip_id)


def _build_identity_map(
    units: Sequence[OrderingUnit], realizations: Sequence[object],
) -> Tuple[_IdentityMap, bool, Tuple[str, ...]]:
    """The one canonical realization_id<->clip_id resolver. Pure; no
    provider, no recomputation of any P1/P2 field. An ``OrderingUnit``
    whose underlying take object is missing, or two distinct units
    resolving to the same ``clip_id``, is reported as an identity gap --
    never silently guessed."""
    take_by_realization_id: dict[str, object] = {}
    for obj in realizations:
        take_by_realization_id[_composer_identity(obj)] = obj

    flags: list[str] = []
    missing = [u.realization_id for u in units if u.realization_id not in take_by_realization_id]
    if missing:
        flags.append("MISSING_REALIZATION_OBJECT")

    clip_id_to_realization_id: dict[str, str] = {}
    collision = False
    for unit in units:
        take = take_by_realization_id.get(unit.realization_id)
        if take is None:
            continue
        clip_id = str(take.clip_id)
        existing = clip_id_to_realization_id.get(clip_id)
        if existing is not None and existing != unit.realization_id:
            collision = True
        clip_id_to_realization_id[clip_id] = unit.realization_id
    if collision:
        flags.append("AMBIGUOUS_CLIP_ID_MAPPING")

    valid = not missing and not collision
    return _IdentityMap(take_by_realization_id, clip_id_to_realization_id), valid, tuple(flags)


def _fallback_result(
    *,
    status: str,
    composer_path: str,
    baseline_plan: OrderedRealizationPlan,
    fallback_reason: str,
    proposed_realization_ids: Tuple[str, ...] = (),
    violated_relation_ids: Tuple[str, ...] = (),
    identity_valid: bool = True,
    membership_valid: bool = True,
    composite_order_valid: bool = True,
    flags: Tuple[str, ...] = (),
) -> OrderingComposerProposalResult:
    """Every rejection/decline path returns D-206's own deterministic
    baseline order VERBATIM -- never recomputed through a different
    algorithm (this task's own 'baseline fallback' instruction)."""
    return OrderingComposerProposalResult(
        proposal_status=status,
        composer_path=composer_path,
        proposed_realization_ids=proposed_realization_ids,
        accepted_realization_ids=baseline_plan.ordered_realization_ids,
        identity_valid=identity_valid,
        membership_valid=membership_valid,
        composite_order_valid=composite_order_valid,
        violated_relation_ids=violated_relation_ids,
        constraint_violation_count=len(violated_relation_ids),
        validation_flags=flags,
        causal_validator_status=CAUSAL_VALIDATOR_NOT_INTEGRATED,
        fallback_used=True,
        fallback_reason=fallback_reason,
        underlying_ordering_status=baseline_plan.ordering_status,
        baseline_realization_ids=baseline_plan.ordered_realization_ids,
        provenance=("D207_ADAPTER_VALIDATOR",),
    )


def validate_composer_ordering_proposal(
    *,
    units: Sequence[OrderingUnit],
    realizations: Sequence[object],
    relations: Sequence[OrderingRelationEvidence],
    baseline_plan: OrderedRealizationPlan,
    provider: ComposerProvider | None = None,
    labels: Sequence[SemanticLabel] = (),
    strategy: EditStrategy = EditStrategy.MIXED,
    context_text: str = "",
) -> OrderingComposerProposalResult:
    """The one canonical adapter+validator entry point. Pure aside from
    the single, non-retried ``provider`` call this task explicitly
    authorizes (mock/fake only -- never ``OpenAIComposerProvider``, never
    a network call). Validation order (this task's own numbered list):

    1. identity validity (both of the INPUT map and of whatever the
       composer actually returned)
    2. reorder-only membership invariant
    3. baseline conflict gate (checked before invoking the composer at
       all -- see module docstring: a genuinely conflicted baseline is
       never even offered to the composer to "resolve")
    4. hard ``OrderingRelation`` constraints (``MUST_PRECEDE``/
       ``MUST_FOLLOW``/``PRESERVE_INTERNAL_ORDER``, covering P1 local-
       sequence/continuation/correction AND any caller-supplied
       composite-internal-order evidence)
    5. composite internal order (isolated for its own diagnostic flag,
       already covered by step 4's general check)
    6. causal order validator -- never invoked (future seam only,
       ``CAUSAL_VALIDATOR_NOT_INTEGRATED``, module docstring)
    7. accept
    """
    identity_map, identity_ok, identity_flags = _build_identity_map(units, realizations)
    if not identity_ok:
        return _fallback_result(
            status=PROPOSAL_REJECTED_IDENTITY,
            composer_path=COMPOSER_PATH_NOT_EVALUATED_IDENTITY_GAP,
            baseline_plan=baseline_plan,
            fallback_reason="identity_gap",
            identity_valid=False, membership_valid=False, composite_order_valid=False,
            flags=identity_flags,
        )

    conflicted_relation_ids = tuple(r.relation_id for r in relations if r.ordering_relation == RELATION_CONFLICTED)
    if baseline_plan.ordering_status == ORDERING_STATUS_CONFLICTED or conflicted_relation_ids:
        return _fallback_result(
            status=PROPOSAL_NOT_EVALUABLE,
            composer_path=COMPOSER_PATH_NOT_EVALUATED_BASELINE_CONFLICTED,
            baseline_plan=baseline_plan,
            fallback_reason="baseline_conflicted",
            violated_relation_ids=conflicted_relation_ids or baseline_plan.unresolved_relation_ids,
            flags=identity_flags + ("BASELINE_CONFLICTED",),
        )

    takes = tuple(
        identity_map.take_by_realization_id[rid]
        for rid in baseline_plan.ordered_realization_ids
        if rid in identity_map.take_by_realization_id
    )
    composer_path = COMPOSER_PATH_MOCK_PROVIDER if provider is not None else COMPOSER_PATH_NO_PROVIDER_NATURAL

    # Exactly one provider invocation. composer_provider.safe_compose_order
    # (unmodified) already fails open to natural order on any provider
    # exception -- this module adds no retry of its own.
    composer_result = safe_compose_order(provider, takes, tuple(labels), strategy, context_text=context_text)
    proposed_realization_ids = tuple(
        identity_map.clip_id_to_realization_id.get(str(clip_id), "") for clip_id in composer_result.ordered_clip_ids
    )

    # composer_provider's own repair already silently fixed a dropped/
    # added/duplicated/unknown-id raw proposal (see module docstring's
    # "No repair that changes membership" section) -- this module never
    # accepts that repaired list as a clean proposal.
    if "provider_output_repaired" in (composer_result.reason or ""):
        return _fallback_result(
            status=PROPOSAL_REJECTED_REORDER_ONLY,
            composer_path=composer_path,
            baseline_plan=baseline_plan,
            fallback_reason="provider_proposal_membership_violation",
            proposed_realization_ids=proposed_realization_ids,
            membership_valid=False,
            flags=identity_flags + ("PROVIDER_OUTPUT_REPAIRED_REJECTED",),
        )

    if any(rid == "" for rid in proposed_realization_ids) or len(set(proposed_realization_ids)) != len(proposed_realization_ids):
        return _fallback_result(
            status=PROPOSAL_REJECTED_IDENTITY,
            composer_path=composer_path,
            baseline_plan=baseline_plan,
            fallback_reason="proposal_identity_gap",
            proposed_realization_ids=proposed_realization_ids,
            identity_valid=False,
            flags=identity_flags + ("PROPOSAL_IDENTITY_GAP",),
        )

    input_ids = tuple(u.realization_id for u in units)
    membership_valid = (
        len(input_ids) == len(set(input_ids))
        and set(proposed_realization_ids) == set(input_ids)
        and len(proposed_realization_ids) == len(input_ids)
    )
    if not membership_valid:
        return _fallback_result(
            status=PROPOSAL_REJECTED_REORDER_ONLY,
            composer_path=composer_path,
            baseline_plan=baseline_plan,
            fallback_reason="membership_violated",
            proposed_realization_ids=proposed_realization_ids,
            membership_valid=False,
            flags=identity_flags + ("MEMBERSHIP_VIOLATED",),
        )

    position = {rid: index for index, rid in enumerate(proposed_realization_ids)}
    violated: list[str] = []
    for rel in relations:
        if rel.ordering_relation not in _HARD_ORDER_RELATIONS:
            continue
        left, right = rel.left_realization_id, rel.right_realization_id
        if left not in position or right not in position:
            continue
        if rel.ordering_relation == RELATION_MUST_FOLLOW:
            if position[right] >= position[left]:
                violated.append(rel.relation_id)
        else:
            if position[left] >= position[right]:
                violated.append(rel.relation_id)

    composite_relation_ids = {
        r.relation_id for r in relations if r.ordering_reason == REASON_COMPOSITE_INTERNAL_ORDER
    }
    composite_order_valid = not (set(violated) & composite_relation_ids)

    if violated:
        return _fallback_result(
            status=PROPOSAL_REJECTED_CONSTRAINT,
            composer_path=composer_path,
            baseline_plan=baseline_plan,
            fallback_reason="ordering_relation_violated",
            proposed_realization_ids=proposed_realization_ids,
            violated_relation_ids=tuple(violated),
            composite_order_valid=composite_order_valid,
            flags=identity_flags + ("CONSTRAINT_VIOLATED",),
        )

    # Step 6 (causal order validator) is a documented future seam only --
    # never invoked (module docstring). Step 7: accept.
    return OrderingComposerProposalResult(
        proposal_status=PROPOSAL_ACCEPTED,
        composer_path=composer_path,
        proposed_realization_ids=proposed_realization_ids,
        accepted_realization_ids=proposed_realization_ids,
        identity_valid=True,
        membership_valid=True,
        composite_order_valid=True,
        violated_relation_ids=(),
        constraint_violation_count=0,
        validation_flags=identity_flags,
        causal_validator_status=CAUSAL_VALIDATOR_NOT_INTEGRATED,
        fallback_used=False,
        fallback_reason="",
        underlying_ordering_status=baseline_plan.ordering_status,
        baseline_realization_ids=baseline_plan.ordered_realization_ids,
        provenance=("D207_ADAPTER_VALIDATOR",),
    )


def run_existing_compose_selected_compat(
    units: Sequence[OrderingUnit], realizations: Sequence[object],
) -> Tuple[str, ...]:
    """Read-only compatibility probe (this task's own 'EXISTING
    DETERMINISTIC COMPOSER: audit compose_selected(...), exercise as one
    compatibility path' instruction). Calls ``composer.py``'s real,
    always-live ``compose_selected`` directly -- completely unmodified,
    with no retry-family groups (every unit is therefore 'ungrouped
    valid' and kept) -- to prove today's existing chronological composer
    is already callable offline on this module's own identity-mapped
    takes with ZERO changes to ``composer.py`` itself. Diagnostic-only:
    never called from ``validate_composer_ordering_proposal`` above,
    never treated as an alternative ordering authority. Returns
    realization ids (not raw clip ids) so callers can compare directly
    against a ``OrderedRealizationPlan.ordered_realization_ids``."""
    identity_map, identity_ok, _ = _build_identity_map(units, realizations)
    if not identity_ok:
        return ()
    takes = tuple(
        identity_map.take_by_realization_id[u.realization_id]
        for u in units
        if u.realization_id in identity_map.take_by_realization_id
    )
    selected = compose_selected(takes, groups=(), labels=())
    return tuple(
        identity_map.clip_id_to_realization_id.get(str(take.clip_id), "") for take in selected
    )


# ---------------------------------------------------------------------------
# Diagnostics / run summary -- tail-safe, counts/status-only, no
# transcript, no QA reference, no master score.
# ---------------------------------------------------------------------------
def ordering_composer_proposal_diagnostics(result: OrderingComposerProposalResult) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "composer_path": result.composer_path,
        "proposal_status": result.proposal_status,
        "proposed_realization_ids": list(result.proposed_realization_ids),
        "accepted_realization_ids": list(result.accepted_realization_ids),
        "identity_valid": result.identity_valid,
        "membership_valid": result.membership_valid,
        "composite_order_valid": result.composite_order_valid,
        "constraint_violation_count": result.constraint_violation_count,
        "violated_relation_ids": list(result.violated_relation_ids),
        "causal_validator_status": result.causal_validator_status,
        "fallback_used": result.fallback_used,
        "fallback_reason": result.fallback_reason,
        "underlying_ordering_status": result.underlying_ordering_status,
        "baseline_realization_ids": list(result.baseline_realization_ids),
        "validation_flags": list(result.validation_flags),
        "provenance": list(result.provenance),
    }


def ordering_composer_run_summary(results: Sequence[OrderingComposerProposalResult]) -> dict:
    results = tuple(results)
    return {
        "schema_version": SCHEMA_VERSION,
        "proposal_count": len(results),
        "accepted_count": sum(1 for r in results if r.proposal_status == PROPOSAL_ACCEPTED),
        "rejected_identity_count": sum(1 for r in results if r.proposal_status == PROPOSAL_REJECTED_IDENTITY),
        "rejected_membership_count": sum(1 for r in results if r.proposal_status == PROPOSAL_REJECTED_REORDER_ONLY),
        "rejected_constraint_count": sum(1 for r in results if r.proposal_status == PROPOSAL_REJECTED_CONSTRAINT),
        "rejected_conflict_count": sum(
            1 for r in results if r.proposal_status in (PROPOSAL_REJECTED_CONFLICT, PROPOSAL_NOT_EVALUABLE)
        ),
        "fallback_count": sum(1 for r in results if r.fallback_used),
        "constraint_violation_count": sum(r.constraint_violation_count for r in results),
        # causal_order_validator.py is never integrated in this module
        # (module docstring) -- always 0, never omitted, so a reader never
        # has to guess whether it silently PASSed.
        "causal_validator_failure_count": 0,
    }
