"""D-208: Ordering Live Diagnostic Integration -- DETERMINISTIC / NO LIVE
PROVIDER / NO AUTHORITY. A DIAGNOSTIC SIDE-CHANNEL over the real pipeline's
already-computed frozen/selected realizations, D-193-D-200.4B P1 evidence,
and D-201-D-204 P2 evidence -- answering "what ordering constraints and
deterministic safe ordering plan would CutSell construct for the current
selected realizations?" without ever changing the final edit.

See ``docs/CUTSELL_DECISIONS.md`` D-205 (forensic), D-206 (typed
foundation), D-207 (composer adapter/proposal validation) and D-208
(this module). D-206/D-207 are OFFLINE-proven typed builders/validators
with zero live callers; this module is the thin, pure adapter that maps
the REAL live objects the pipeline already computes into their inputs --
it adds no new construction logic of its own beyond that mapping.

## What this module is NOT (binding, restated from this task's own scope)

No Ordering authority: nothing here can alter membership, selection, or
the final rendered edit. No pipeline stage reorder: the EXISTING
composer (``composer.py``/``composer_provider.py``) still runs exactly
where D-205 found it (before the point this module's own diagnostics
run) -- this module never moves it, and this module's own diagnostics
run in a strictly read-only, discard-the-result side channel. No live
provider: ``OpenAIComposerProvider`` is never imported by this module,
never instantiated, never called -- every composer-shaped diagnostic
this module produces goes through the EXISTING, unmodified, provider-
free ``composer.py::compose_selected`` (see ``_DeterministicCompose
SelectedProvider`` below), never a network call. No RAW. No re-
computation of P1 (``editorial_moment_sequence[_integration].py``),
P2 (``whole_video_editorial_reasoning[_integration].py``), Language,
Family, BestTake, ``CompositeResolver``, or ``RealizationResolver`` --
every P1/P2 object this module reads is passed in by the caller
(``pipeline.py``), already built by those modules' own existing call
sites, by reference only.

## Mechanical pipeline-position honesty (D-205's own finding, restated)

D-205/D-207 already established that the existing composer
(``compose_selected``/``safe_compose_order``) mechanically runs BEFORE
the conceptual post-Freeze "Ordering" stage this whole consolidation
effort is named after -- Selection/Composition, then Freeze (a SEPARATE
gate in ``canonical_edit_plan.py``/``universal_clean_cut.py``, not
reached inside ``pipeline.py``'s own draft-construction function), then
Boundary/Pacing/Renderer. **The ideal, mechanically-real, POST-FREEZE
seam this task's directive describes does not exist inside
``pipeline.py`` today** -- this function returns its draft before Freeze
ever runs. This module's live diagnostics therefore run at the ONLY real
in-memory point available inside this function: immediately after this
function's own already-computed `selected` bucket (the CURRENT,
post-composition, post-review selected ``DraftClip``s -- the same
objects D-195/D-199/D-203's own diagnostic blocks already sit beside),
never a genuinely-verified post-Freeze set. This is reported honestly in
every diagnostic result and this document, per this task's own "if the
ideal seam does not mechanically exist, state that explicitly, do not
fix stage order" instruction -- D-208 does not, and must not, move
Family/BestTake/the existing composer/Freeze/Boundary/Pacing.

## Capability status ladder (mirrors D-203's own established pattern)

``AVAILABLE`` only when a non-empty selected-realization set exists AND
both P1 (``editorial_moment_understandings``) and P2 (a real
``WholeVideoEditorialUnderstanding``) evidence are genuinely present.
``PARTIAL`` when a non-empty selection exists but P1 and/or P2 evidence
is unavailable (flag off, or empty) -- Ordering can still report a
source-order-only, constraint-free diagnostic view in that case (never
silently upgraded to AVAILABLE). ``NOT_EVALUABLE`` only when there is no
selected realization to build anything from at all. ``DISABLED`` is
reported by the caller (``pipeline.py``) when the flag itself is off --
this module's own builder is never even invoked in that case (zero
compute, not merely zero extra output).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Sequence, Tuple

from .composer import compose_selected
from .composer_provider import ComposerProviderResult
from .editorial_moment_sequence_integration import EditorialMomentUnderstanding
from .ordering_composer_adapter import (
    CAUSAL_VALIDATOR_NOT_INTEGRATED,
    OrderingComposerProposalResult,
    ordering_composer_proposal_diagnostics,
    validate_composer_ordering_proposal,
)
from .ordering_realization_plan import (
    ORDERING_STATUS_CONFLICTED,
    REASON_COMPOSITE_INTERNAL_ORDER,
    REASON_MEANING_FIREWALL,
    REASON_P1_CONTINUATION,
    REASON_P1_CORRECTION,
    REASON_P1_LOCAL_SEQUENCE,
    REASON_P2_GLOBAL_CONTINUITY,
    REASON_P2_PROPOSITION_PROGRESSION,
    REASON_UNIQUE_INFORMATION,
    RELATION_CONFLICTED,
    RELATION_MUST_FOLLOW,
    RELATION_MUST_PRECEDE,
    RELATION_PRESERVE_INTERNAL_ORDER,
    RELATION_UNKNOWN,
    OrderedRealizationPlan,
    OrderingRelationEvidence,
    OrderingUnit,
    build_deterministic_ordering_plan,
    build_ordering_relation_evidence,
    build_ordering_units,
    ordered_realization_plan_diagnostics,
    ordering_relation_diagnostics,
    ordering_unit_diagnostics,
)
from .providers import ProviderStatus
from .whole_video_editorial_reasoning import WholeVideoEditorialUnderstanding

SCHEMA_VERSION = "cutsell.ordering_live_diagnostics_integration.v1"

_DIAGNOSTICS_ENV = "CUTSELL_ORDERING_DIAGNOSTICS_ENABLED"

# ---------------------------------------------------------------------------
# Capability-status ladder (mirrors D-203's own vocabulary; this module
# mints its own constants rather than importing another module's, exactly
# per that module's own precedent -- see docstring).
# ---------------------------------------------------------------------------
CAPABILITY_AVAILABLE = "AVAILABLE"
CAPABILITY_PARTIAL = "PARTIAL"
CAPABILITY_NOT_EVALUABLE = "NOT_EVALUABLE"
CAPABILITY_DISABLED = "DISABLED"

# ---------------------------------------------------------------------------
# Missing-evidence vocabulary (explicit, never silent auto-enable/upgrade).
# ---------------------------------------------------------------------------
MISSING_NO_SELECTED_REALIZATIONS = "NO_SELECTED_REALIZATIONS"
MISSING_P1_UNAVAILABLE = "P1_EDITORIAL_MOMENT_UNDERSTANDING_UNAVAILABLE"
MISSING_P2_UNAVAILABLE = "P2_WHOLE_VIDEO_EDITORIAL_REASONING_UNAVAILABLE"


def _env_true_default_false(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def ordering_diagnostics_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Default OFF. When OFF, nothing in this module is ever called by
    ``pipeline.py`` -- zero Ordering compute, selection/Family/BestTake/
    D-191/Boundary/Pacing/render output stays byte-identical. There is no
    authority flag anywhere in this module."""
    values = env if env is not None else os.environ
    return _env_true_default_false(values.get(_DIAGNOSTICS_ENV))


# ---------------------------------------------------------------------------
# Deterministic, provider-free "existing composer" compatibility path.
# ---------------------------------------------------------------------------
class _DeterministicComposeSelectedProvider:
    """Wraps the EXISTING, unmodified, always-live, provider-free
    ``composer.py::compose_selected`` as a ``ComposerProvider``-shaped
    diagnostic compatibility path -- never a second composer, never a
    network call, never a story model. Calling it with ``groups=()``
    treats every already-selected realization as independent (retry-
    family collapsing already happened upstream, before this diagnostic
    side-channel runs), so the only thing it can ever do is restate
    stable chronological order -- exactly D-207's own ``run_existing_
    compose_selected_compat`` framing, reused here as a real
    ``ComposerProvider`` so D-207's own validator can be exercised on it
    unmodified (this task's own "pass it through D-207 proposal
    validation" instruction)."""

    def order(self, takes, labels, strategy, context_text=""):
        selected = compose_selected(takes, groups=(), labels=labels)
        return ComposerProviderResult(
            tuple(take.clip_id for take in selected),
            ProviderStatus(
                provider="deterministic_compose_selected_compat", requested=True, available=True,
                status="diagnostic_compatibility_only",
            ),
            "chronological_compose_selected_compatibility_probe",
        )


# ---------------------------------------------------------------------------
# Result type.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrderingLiveDiagnosticsResult:
    """Bounded, diagnostics-only. ``baseline_plan``/``composer_proposal``
    are ``None`` only when ``capability_status == NOT_EVALUABLE`` (no
    selected realization to build anything from) -- never a half-built
    object standing in for a real one."""
    capability_status: str
    missing_evidence: Tuple[str, ...]
    units: Tuple[OrderingUnit, ...]
    relations: Tuple[OrderingRelationEvidence, ...]
    baseline_plan: OrderedRealizationPlan | None
    composer_proposal: OrderingComposerProposalResult | None
    causal_validator_status: str
    provenance: Tuple[str, ...]


def _not_evaluable_result(missing_evidence: Tuple[str, ...]) -> OrderingLiveDiagnosticsResult:
    return OrderingLiveDiagnosticsResult(
        capability_status=CAPABILITY_NOT_EVALUABLE,
        missing_evidence=missing_evidence,
        units=(), relations=(), baseline_plan=None, composer_proposal=None,
        causal_validator_status=CAUSAL_VALIDATOR_NOT_INTEGRATED,
        provenance=("D208_LIVE_DIAGNOSTIC_INTEGRATION",),
    )


def build_ordering_live_diagnostics(
    *,
    selected_realizations: Sequence[object],
    editorial_moment_understandings: Sequence[EditorialMomentUnderstanding] = (),
    whole_video_editorial_reasoning_result: WholeVideoEditorialUnderstanding | None = None,
    p1_diagnostics_enabled: bool,
    p2_diagnostics_enabled: bool,
) -> OrderingLiveDiagnosticsResult:
    """The one canonical live-integration entrypoint. Pure; consumes ONLY
    already-built P1/P2 objects the caller passes in -- never recomputes
    P1/P2/Language/Family/BestTake/CompositeResolver/RealizationResolver
    itself. ``p1_diagnostics_enabled``/``p2_diagnostics_enabled`` are
    passed explicitly by the caller (rather than read from the
    environment here) so this function stays a pure, deterministic
    transform of its own arguments -- the caller (``pipeline.py``) is the
    one place flag state is actually read, exactly mirroring D-195/D-199/
    D-203's own call-site pattern. Reuses D-206's ``build_ordering_units``/
    ``build_ordering_relation_evidence``/``build_deterministic_ordering_
    plan`` and D-207's ``validate_composer_ordering_proposal`` verbatim --
    no sorting/topological/validation logic is reimplemented here."""
    if not selected_realizations:
        return _not_evaluable_result((MISSING_NO_SELECTED_REALIZATIONS,))

    missing_evidence: list[str] = []

    has_p1_evidence = bool(p1_diagnostics_enabled and editorial_moment_understandings)
    if not has_p1_evidence:
        missing_evidence.append(MISSING_P1_UNAVAILABLE)

    has_p2_evidence = bool(p2_diagnostics_enabled and whole_video_editorial_reasoning_result is not None)
    if not has_p2_evidence:
        missing_evidence.append(MISSING_P2_UNAVAILABLE)

    moments_by_source: Mapping[str, Tuple] = {}
    local_groups_by_source: Mapping[str, Tuple] = {}
    moments_by_id: Mapping[str, object] = {}
    local_groups_by_id: Mapping[str, object] = {}
    sequence_ids_by_moment_id: Mapping[str, Tuple[str, ...]] = {}
    if has_p1_evidence:
        moments_by_source = {u.source_asset_id: u.moments for u in editorial_moment_understandings}
        local_groups_by_source = {u.source_asset_id: u.local_groups for u in editorial_moment_understandings}
        moments_by_id = {m.editorial_moment_id: m for u in editorial_moment_understandings for m in u.moments}
        local_groups_by_id = {g.group_id: g for u in editorial_moment_understandings for g in u.local_groups}
        sequence_lists: dict[str, list] = {}
        for u in editorial_moment_understandings:
            for seq in u.sequence_hypotheses:
                for moment_id in seq.moment_ids:
                    sequence_lists.setdefault(moment_id, []).append(seq.sequence_id)
        sequence_ids_by_moment_id = {mid: tuple(ids) for mid, ids in sequence_lists.items()}

    region_ids_by_moment_id: Mapping[str, Tuple[str, ...]] = {}
    supersession_hypotheses: Tuple = ()
    if has_p2_evidence:
        region_lists: dict[str, list] = {}
        for region in whole_video_editorial_reasoning_result.regions:
            for moment_id in region.moment_ids:
                region_lists.setdefault(moment_id, []).append(region.region_id)
        region_ids_by_moment_id = {mid: tuple(ids) for mid, ids in region_lists.items()}
        supersession_hypotheses = whole_video_editorial_reasoning_result.supersession_hypotheses

    units = build_ordering_units(
        realizations=selected_realizations,
        moments_by_source=moments_by_source,
        local_groups_by_source=local_groups_by_source,
        sequence_ids_by_moment_id=sequence_ids_by_moment_id,
        region_ids_by_moment_id=region_ids_by_moment_id,
    )
    region_ids_by_unit = {unit.realization_id: unit.p2_region_ids for unit in units}
    relations = build_ordering_relation_evidence(
        units=units,
        moments_by_id=moments_by_id,
        local_groups_by_id=local_groups_by_id,
        supersession_hypotheses=supersession_hypotheses,
        region_ids_by_unit=region_ids_by_unit,
    )
    baseline_plan = build_deterministic_ordering_plan(units=units, relations=relations)

    # D-207's own validator, exercised only through the deterministic,
    # provider-free existing-composer compatibility path above -- never
    # OpenAIComposerProvider, never a network call.
    composer_proposal = validate_composer_ordering_proposal(
        units=units,
        realizations=selected_realizations,
        relations=relations,
        baseline_plan=baseline_plan,
        provider=_DeterministicComposeSelectedProvider(),
    )

    capability_status = CAPABILITY_AVAILABLE if not missing_evidence else CAPABILITY_PARTIAL

    return OrderingLiveDiagnosticsResult(
        capability_status=capability_status,
        missing_evidence=tuple(missing_evidence),
        units=units,
        relations=relations,
        baseline_plan=baseline_plan,
        composer_proposal=composer_proposal,
        causal_validator_status=CAUSAL_VALIDATOR_NOT_INTEGRATED,
        provenance=("D208_LIVE_DIAGNOSTIC_INTEGRATION",),
    )


# ---------------------------------------------------------------------------
# Diagnostics / run summary -- tail-safe, counts/status-only, no
# transcript, no master score. Field names below mirror this task's own
# directive naming; ``ordering_unit_id``/``left_unit_id``/``right_unit_id``/
# ``relation``/``status`` are presentational aliases over D-206's own
# already-established ``realization_id``/``ordering_relation``/
# ``confidence`` fields -- this module mints no second identity for a
# unit or a relation.
# ---------------------------------------------------------------------------
def _live_unit_view(unit: OrderingUnit) -> dict:
    base = ordering_unit_diagnostics(unit)
    return {
        "ordering_unit_id": base["realization_id"],
        **base,
        "p1_local_group_ids": [base["p1_local_group_id"]] if base["p1_local_group_id"] else [],
    }


def _live_relation_view(relation: OrderingRelationEvidence) -> dict:
    base = ordering_relation_diagnostics(relation)
    return {
        **base,
        "left_unit_id": base["left_realization_id"],
        "right_unit_id": base["right_realization_id"],
        "relation": base["ordering_relation"],
        "status": base["confidence"],
    }


def ordering_live_diagnostics(result: OrderingLiveDiagnosticsResult) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "capability_status": result.capability_status,
        "missing_evidence": list(result.missing_evidence),
        "unit_count": len(result.units),
        "relation_count": len(result.relations),
        "units": [_live_unit_view(u) for u in result.units],
        "relations": [_live_relation_view(r) for r in result.relations],
        "baseline_plan": (
            ordered_realization_plan_diagnostics(result.baseline_plan)
            if result.baseline_plan is not None else None
        ),
        "proposal_validation": (
            ordering_composer_proposal_diagnostics(result.composer_proposal)
            if result.composer_proposal is not None else None
        ),
        "causal_validator_status": result.causal_validator_status,
        "provenance": list(result.provenance),
    }


def ordering_live_diagnostics_run_summary(result: OrderingLiveDiagnosticsResult) -> dict:
    relations = result.relations
    plan = result.baseline_plan
    proposal = result.composer_proposal
    unresolved = plan.unresolved_relation_ids if plan is not None else ()
    return {
        "schema_version": SCHEMA_VERSION,
        "ordering_unit_count": len(result.units),
        "ordering_relation_count": len(relations),
        "must_precede_count": sum(1 for r in relations if r.ordering_relation == RELATION_MUST_PRECEDE),
        "must_follow_count": sum(1 for r in relations if r.ordering_relation == RELATION_MUST_FOLLOW),
        "preserve_internal_order_count": sum(
            1 for r in relations if r.ordering_relation == RELATION_PRESERVE_INTERNAL_ORDER
        ),
        "conflicted_relation_count": sum(1 for r in relations if r.ordering_relation == RELATION_CONFLICTED),
        "unknown_relation_count": sum(1 for r in relations if r.ordering_relation == RELATION_UNKNOWN),
        "ordering_status": plan.ordering_status if plan is not None else CAPABILITY_NOT_EVALUABLE,
        "fallback_used": plan.fallback_used if plan is not None else False,
        "unresolved_relation_count": len(unresolved),
        "cycle_count": sum(1 for uid in unresolved if str(uid).startswith("CYCLE:")),
        "retry_survival_conflict_count": sum(
            1 for r in relations if "SUPERSESSION_SURVIVAL_CONFLICT" in r.conflict_flags
        ),
        "composite_internal_order_constraint_count": sum(
            1 for r in relations if r.ordering_reason == REASON_COMPOSITE_INTERNAL_ORDER
        ),
        "p1_sequence_constraint_count": sum(
            1 for r in relations if r.ordering_reason == REASON_P1_LOCAL_SEQUENCE
        ),
        "continuation_constraint_count": sum(
            1 for r in relations if r.ordering_reason == REASON_P1_CONTINUATION
        ),
        "correction_constraint_count": sum(
            1 for r in relations if r.ordering_reason == REASON_P1_CORRECTION
        ),
        "p2_constraint_count": sum(
            1 for r in relations if r.ordering_reason in (
                REASON_P2_PROPOSITION_PROGRESSION, REASON_P2_GLOBAL_CONTINUITY,
                REASON_MEANING_FIREWALL, REASON_UNIQUE_INFORMATION,
            )
        ),
        "composer_proposal_available": proposal is not None,
        "composer_proposal_status": proposal.proposal_status if proposal is not None else None,
        "composer_constraint_violation_count": proposal.constraint_violation_count if proposal is not None else 0,
    }
