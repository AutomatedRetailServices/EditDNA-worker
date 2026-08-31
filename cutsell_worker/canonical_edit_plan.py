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

from dataclasses import dataclass, field
from typing import Mapping


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
    """The single structured semantic handoff to physical editing."""

    project_id: str
    ideas: tuple[EditPlanIdea, ...]
    keep_sequence: tuple[EditPlanClip, ...]
    discard_provenance: tuple[DiscardRecord, ...]
    lost_semantic_atoms: tuple[dict, ...]
    contradiction_findings: tuple[dict, ...]
    composite_provenance: tuple[dict, ...]
    possible_missing_story_ending: bool
    freeze_blocked: bool
    validation_state: str  # "frozen_ready" | "freeze_blocked_pending_review"


def _composite_piece_ids(diagnostics: Mapping[str, object]) -> frozenset[str]:
    """Clip ids CompositeResolver marked as a composite piece (step 12/14's
    split_group_clip_ids), read from diagnostics it already produced."""
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

    ideas: list[EditPlanIdea] = []
    for group in diagnostics.get("take_judge_groups") or ():
        member_ids = [str(row.get("clip_id") or "") for row in (group.get("ranked") or ())]
        winning = tuple(cid for cid in member_ids if cid in selected_ids)
        discarded = tuple(cid for cid in member_ids if cid not in selected_ids)
        if not member_ids:
            continue
        if not winning:
            coverage_status = "missing"
        elif len(winning) >= 2:
            coverage_status = "unresolved_ambiguous"
        else:
            coverage_status = "complete"
        ideas.append(EditPlanIdea(
            idea_id=str(group.get("group_id") or ""),
            winning_clip_ids=winning,
            is_composite=any(cid in composite_ids for cid in winning),
            discarded_clip_ids=discarded,
            coverage_status=coverage_status,
        ))

    clip_to_idea: dict[str, str] = {
        cid: idea.idea_id for idea in ideas for cid in idea.winning_clip_ids
    }

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
        for clip in sorted(draft.selected, key=lambda c: (c.source_order, c.start, c.end, c.clip_id))
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
    return CanonicalEditPlan(
        project_id=getattr(draft, "project_id", ""),
        ideas=tuple(ideas),
        keep_sequence=keep_sequence,
        discard_provenance=discard_provenance,
        lost_semantic_atoms=tuple(coherence.get("lost_semantic_atoms") or ()),
        contradiction_findings=tuple(coherence.get("contradiction_findings") or ()),
        composite_provenance=_composite_provenance_rows(diagnostics),
        possible_missing_story_ending=bool(coherence.get("possible_missing_story_ending")),
        freeze_blocked=freeze_blocked,
        validation_state="freeze_blocked_pending_review" if freeze_blocked else "frozen_ready",
    )
