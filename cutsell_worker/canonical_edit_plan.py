"""CanonicalEditPlan -- the single structured semantic handoff from Selection
to physical editing (Boundary/Renderer), per the agentic-editing architecture
upgrade in D-024.

Nothing here invents new semantic judgment. Every field is built from
evidence the pipeline already computed by the time this is called (right
after Final Story Coherence Validation, before Selection Freeze): the
draft's own `selected`/`discarded`/`alternates`, `take_judge_groups`
(IdeaClusterer's resolved retry families), `final_story_coherence_
validation`'s coverage/contradiction findings, and `hybrid_editorial_
chunks`/CompositeResolver's own diagnostics for composite provenance.

Why this exists: before this module, "what is the final semantic truth of
this edit" was implicit -- scattered across `draft.selected`, several
diagnostics keys, and the reader's own knowledge of pipeline order. Boundary
and Renderer already only ever consume `draft.selected` (Renderer renders
`selected` only, never `alternates` -- see D-021), so neither infers
semantic membership independently today; this module makes that semantic
snapshot explicit and inspectable as one object instead of leaving it
implicit in `draft.selected`'s shape, and gives FinalEditReviewer (D-024)
one canonical thing to review.

`annotations` is a deliberately empty, dormant extension point for a future
Sales/TikTok Shop composition layer (D-024's "FUTURE SALES / TIKTOK SHOP
EXTENSION CONTRACT"): per-clip optional tags like `semantic_role`,
`product_relevance`, `emphasis_opportunity`, etc. Clean Cut Core V1 never
populates it and nothing reads it; it exists so that layer, when built, has
a place to attach without another architecture reset.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

from .contracts import effective_parent_semantic_clip_id
from .selection_boundary_contract import semantic_token_stream


@dataclass(frozen=True)
class EditPlanClip:
    """One clip in the final KEEP sequence, with its semantic provenance."""

    clip_id: str
    source_asset_id: str
    source_order: int
    start: float
    end: float
    text: str
    idea_id: str | None
    is_composite_piece: bool
    protected_leading_word: str | None
    protected_trailing_word: str | None
    # Dormant Sales/TikTok Shop extension point -- see module docstring.
    # Never populated by Clean Cut Core V1.
    annotations: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EditPlanIdea:
    """One reconstructed idea/retry family and how it resolved."""

    idea_id: str
    winning_clip_ids: tuple[str, ...]
    is_composite: bool
    discarded_clip_ids: tuple[str, ...]
    coverage_status: str  # "complete" | "missing" | "unresolved_ambiguous"


@dataclass(frozen=True)
class DiscardRecord:
    """One discarded clip's provenance, best-effort from available diagnostics."""

    clip_id: str
    text: str
    source_asset_id: str
    start: float
    end: float


@dataclass(frozen=True)
class CanonicalEditPlan:
    """The single structured semantic handoff to physical editing.

    ``plan_id``/``plan_version``/``semantic_hash`` (D-025): SelectionFreeze
    must reference a specific, identified plan rather than freezing whatever
    ``draft.selected`` happens to contain -- see ``selection_boundary_
    contract.freeze_selection_contract``'s ``plan`` parameter, which records
    these onto its own diagnostics for traceability. ``plan_version`` is
    always 1 today: this cycle does not implement an automatic repair loop
    that would produce v2/v3 (see D-025's "not yet built" note) -- a human
    reviewing a FAIL is the only repair path right now, and a fresh pipeline
    run after a human-driven change produces a new plan_id (it is derived
    from the actual KEEP content), not a new version of the same one.
    """

    project_id: str
    plan_id: str
    plan_version: int
    semantic_hash: str
    ideas: tuple[EditPlanIdea, ...]
    keep_sequence: tuple[EditPlanClip, ...]
    discard_provenance: tuple[DiscardRecord, ...]
    lost_semantic_atoms: tuple[dict, ...]
    contradiction_findings: tuple[dict, ...]
    composite_provenance: tuple[dict, ...]
    possible_missing_story_ending: bool
    freeze_blocked: bool
    validation_state: str  # "frozen_ready" | "freeze_blocked_pending_review"
    # D-038: per-Idea critical-claim coverage findings (semantic_claims.py),
    # independent of lost_semantic_atoms's whole-KEEP-timeline vocabulary
    # comparison -- see final_story_coherence_validation._lost_critical_
    # claims's own docstring for why the two checks are not redundant.
    # Defaulted so every existing construction site (there is exactly one,
    # build_canonical_edit_plan below) and any external payload deserializer
    # stays valid unchanged.
    lost_critical_claims: tuple[dict, ...] = ()


def _composite_piece_ids(diagnostics: Mapping[str, object]) -> frozenset[str]:
    """Clip ids CompositeResolver marked as a composite piece (step 12/14's
    split_group_clip_ids), read from diagnostics it already produced. Also
    recognizes `claim_coverage_best_take.py`'s own narrow, D-038 claim-
    coverage-triggered composite fallback (a distinct, bounded mechanism
    from the general CompositeResolver above -- see that module's
    docstring), so an idea it resolved this way is correctly reported
    `is_composite: true` rather than `unresolved_ambiguous`."""
    ids: set[str] = set()
    for row in diagnostics.get("hybrid_editorial_chunks") or ():
        if not isinstance(row, dict):
            continue
        for key in ("hybrid_semantic_complementary_rescue", "hybrid_composite_best_take"):
            entry = row.get(key)
            if isinstance(entry, dict):
                ids.update(str(i) for i in (entry.get("split_group_clip_ids") or ()))
            elif isinstance(entry, list):
                for item in entry:
                    if isinstance(item, dict):
                        ids.update(str(i) for i in (item.get("split_group_clip_ids") or ()))
    claim_coverage_best_take = diagnostics.get("claim_coverage_best_take") or {}
    for composite in claim_coverage_best_take.get("composites") or ():
        if isinstance(composite, dict):
            ids.update(str(cid) for cid in (composite.get("clip_ids") or ()))
    return frozenset(ids)


def _composite_provenance_rows(diagnostics: Mapping[str, object]) -> tuple[dict, ...]:
    rows: list[dict] = []
    for row in diagnostics.get("hybrid_editorial_chunks") or ():
        if not isinstance(row, dict):
            continue
        entry = row.get("hybrid_composite_best_take")
        if isinstance(entry, dict) and entry.get("composite_replacements"):
            rows.extend(entry["composite_replacements"])
    return tuple(rows)


def build_canonical_edit_plan(draft) -> CanonicalEditPlan:
    """Build the CanonicalEditPlan from a draft that has already been through
    Final Story Coherence Validation (so `diagnostics["final_story_
    coherence_validation"]` and `take_judge_groups` are populated)."""
    diagnostics = draft.diagnostics or {}
    coherence = diagnostics.get("final_story_coherence_validation") or {}
    composite_ids = _composite_piece_ids(diagnostics)
    selected_ids = {clip.clip_id for clip in draft.selected}
    # D-046 FIX A: a `take_judge_groups` member that won Selection can be
    # physically split afterward (e.g. post_selection_interior_gap_trim's
    # speech-free-gap trim) into fragments whose own `clip_id`s differ from
    # the winning member's original id. Exact `clip_id` equality against
    # `draft.selected` alone would then find neither the original id nor
    # either fragment, and misreport a genuinely-surviving realization as
    # discarded -- see D-045 Case A. `parent_semantic_clip_id` is the
    # general, already-established (D-036) provenance link a physical split
    # is required to set; a member also counts as surviving when any
    # selected clip's `effective_parent_semantic_clip_id` names it. This is
    # scoped to `draft.selected` only, so a fragment of a genuinely-
    # discarded clip (if one ever existed) cannot revive it.
    selected_parent_ids = {
        pid for clip in draft.selected
        if (pid := effective_parent_semantic_clip_id(clip)) is not None
    }

    ideas: list[EditPlanIdea] = []
    for group in diagnostics.get("take_judge_groups") or ():
        member_ids = [str(row.get("clip_id") or "") for row in (group.get("ranked") or ())]
        winning = tuple(
            cid for cid in member_ids
            if cid in selected_ids or cid in selected_parent_ids
        )
        discarded = tuple(
            cid for cid in member_ids
            if cid not in selected_ids and cid not in selected_parent_ids
        )
        if not member_ids:
            continue
        is_accepted_composite = len(winning) >= 2 and all(cid in composite_ids for cid in winning)
        if not winning:
            coverage_status = "missing"
        elif len(winning) >= 2 and not is_accepted_composite:
            # D-025: 2+ surviving members is only ambiguous when they are
            # NOT all pieces of one CompositeResolver-accepted composite --
            # an accepted composite's 2+ components jointly realizing one
            # idea is the CORRECT, intended outcome, not an unresolved
            # retry contest. Without this check, every accepted composite
            # would incorrectly report DUPLICATE_IDEA/UNRESOLVED_RETRY.
            coverage_status = "unresolved_ambiguous"
        else:
            coverage_status = "complete"
        ideas.append(EditPlanIdea(
            idea_id=str(group.get("group_id") or ""),
            winning_clip_ids=winning,
            is_composite=is_accepted_composite,
            discarded_clip_ids=discarded,
            coverage_status=coverage_status,
        ))

    clip_to_idea: dict[str, str] = {
        cid: idea.idea_id for idea in ideas for cid in idea.winning_clip_ids
    }

    # D-025: iterate draft.selected in ITS OWN order, not re-sorted by
    # timestamp. render_plan.py renders `for clip in draft.selected` directly
    # -- that tuple's order IS the actual final rendered order (Composer is
    # explicitly allowed to reorder clips for pacing/narrative/sales logic),
    # so this is "the single semantic handoff to physical editing" only if
    # it reflects that real order. Re-sorting here would have made every
    # order-sensitive check (e.g. the composite continuity check below)
    # silently unable to ever detect a real reordering.
    keep_sequence = tuple(
        EditPlanClip(
            clip_id=clip.clip_id,
            source_asset_id=clip.source_asset_id,
            source_order=clip.source_order,
            start=clip.start,
            end=clip.end,
            text=clip.text,
            idea_id=clip_to_idea.get(clip.clip_id),
            is_composite_piece=clip.clip_id in composite_ids,
            protected_leading_word=(clip.words[0].text if clip.words else None),
            protected_trailing_word=(clip.words[-1].text if clip.words else None),
        )
        for clip in draft.selected
    )

    discard_provenance = tuple(
        DiscardRecord(
            clip_id=clip.clip_id,
            text=clip.text,
            source_asset_id=clip.source_asset_id,
            start=clip.start,
            end=clip.end,
        )
        for clip in sorted(draft.discarded, key=lambda c: (c.source_order, c.start, c.end, c.clip_id))
    )

    freeze_blocked = bool(coherence.get("freeze_blocked"))
    project_id = getattr(draft, "project_id", "")
    semantic_hash = hashlib.sha256(
        "\x1f".join(semantic_token_stream(draft.selected)).encode("utf-8")
    ).hexdigest()
    plan_id = "plan_" + hashlib.sha256(f"{project_id}|{semantic_hash}".encode()).hexdigest()[:16]
    return CanonicalEditPlan(
        project_id=project_id,
        plan_id=plan_id,
        plan_version=1,
        semantic_hash=semantic_hash,
        ideas=tuple(ideas),
        keep_sequence=keep_sequence,
        discard_provenance=discard_provenance,
        lost_semantic_atoms=tuple(coherence.get("lost_semantic_atoms") or ()),
        lost_critical_claims=tuple(coherence.get("lost_critical_claims") or ()),
        contradiction_findings=tuple(coherence.get("contradiction_findings") or ()),
        composite_provenance=_composite_provenance_rows(diagnostics),
        possible_missing_story_ending=bool(coherence.get("possible_missing_story_ending")),
        freeze_blocked=freeze_blocked,
        validation_state="freeze_blocked_pending_review" if freeze_blocked else "frozen_ready",
    )
