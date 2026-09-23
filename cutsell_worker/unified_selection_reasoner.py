"""Unified whole-video Selection authority for CutSell Universal Clean Cut.

The legacy pipeline may produce useful local evidence, retry groups, Hybrid votes,
and provisional Selected/SWAP/Discarded buckets.  None of those buckets are final
when a UnifiedSelectionReasoner is active.  The reasoner sees the complete candidate
universe for the source at once and returns one editorial plan before Selection freeze.

This module is provider-neutral.  It owns validation and safe application only; it
contains no HTTP, vendor SDK, benchmark timestamp, phrase, or Human Gold rule.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from .contracts import DraftClip, DraftTimeline

_ALLOWED_ACTIONS = frozenset({"select", "swap", "discard"})
_ALLOWED_RELATIONS = frozenset({
    "independent",
    "retry_winner",
    "retry_alternate",
    "composite_piece",
    "continuation",
    "failed",
    "bts",
    "uncertain",
})
_ALLOWED_REASONS = frozenset({
    "best_complete_take",
    "independent_story_coverage",
    "composite_best_take_piece",
    "necessary_continuation",
    "usable_alternate",
    "redundant_retry",
    "failed_delivery",
    "recording_process_bts",
    "uncertain_preserve",
})


@dataclass(frozen=True)
class UnifiedSelectionDecision:
    clip_id: str
    action: str
    relation: str
    confidence: float
    family_index: int
    reason_code: str


@dataclass(frozen=True)
class UnifiedSelectionPlan:
    decisions: tuple[UnifiedSelectionDecision, ...]
    provider: str
    model: str
    requested: bool = True
    available: bool = True
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0


class UnifiedSelectionReasoner(Protocol):
    def reason(self, draft: DraftTimeline) -> UnifiedSelectionPlan: ...


def _all_clips(draft: DraftTimeline) -> tuple[DraftClip, ...]:
    """Return every unique semantic candidate in natural source order."""
    by_id: dict[str, DraftClip] = {}
    for clip in (*draft.selected, *draft.alternates, *draft.discarded):
        by_id.setdefault(str(clip.clip_id), clip)
    return tuple(sorted(
        by_id.values(),
        key=lambda clip: (clip.source_order, float(clip.start), float(clip.end), clip.clip_id),
    ))


def _bucket_map(draft: DraftTimeline) -> dict[str, str]:
    out = {clip.clip_id: "discard" for clip in draft.discarded}
    out.update({clip.clip_id: "swap" for clip in draft.alternates})
    out.update({clip.clip_id: "select" for clip in draft.selected})
    return out


def validate_unified_selection_plan(
    draft: DraftTimeline,
    plan: UnifiedSelectionPlan,
) -> UnifiedSelectionPlan:
    expected = {clip.clip_id for clip in _all_clips(draft)}
    seen: set[str] = set()
    normalized: list[UnifiedSelectionDecision] = []

    if not plan.available:
        raise ValueError("unified selection reasoner unavailable")
    if plan.estimated_input_tokens < 0 or plan.estimated_output_tokens < 0:
        raise ValueError("unified selection token estimates must be non-negative")

    for raw in plan.decisions:
        clip_id = str(raw.clip_id)
        if clip_id not in expected:
            raise ValueError("unified selection returned unknown clip id")
        if clip_id in seen:
            raise ValueError("unified selection returned duplicate clip id")
        action = str(raw.action).strip().lower()
        relation = str(raw.relation).strip().lower()
        reason_code = str(raw.reason_code).strip().lower()
        confidence = float(raw.confidence)
        family_index = int(raw.family_index)
        if action not in _ALLOWED_ACTIONS:
            raise ValueError("unified selection returned invalid action")
        if relation not in _ALLOWED_RELATIONS:
            raise ValueError("unified selection returned invalid relation")
        if reason_code not in _ALLOWED_REASONS:
            raise ValueError("unified selection returned invalid reason code")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("unified selection confidence outside 0..1")
        if family_index < 0:
            raise ValueError("unified selection family index must be non-negative")
        normalized.append(UnifiedSelectionDecision(
            clip_id=clip_id,
            action=action,
            relation=relation,
            confidence=confidence,
            family_index=family_index,
            reason_code=reason_code,
        ))
        seen.add(clip_id)

    if seen != expected:
        raise ValueError("unified selection reasoner omitted candidates")

    return UnifiedSelectionPlan(
        decisions=tuple(normalized),
        provider=str(plan.provider or "unknown")[:80],
        model=str(plan.model or "unknown")[:120],
        requested=bool(plan.requested),
        available=True,
        estimated_input_tokens=int(plan.estimated_input_tokens),
        estimated_output_tokens=int(plan.estimated_output_tokens),
    )


def _effective_action(decision: UnifiedSelectionDecision, current_bucket: str) -> tuple[str, str | None]:
    """Fail open on uncertainty without turning uncertainty into destructive deletion."""
    # A reason_code is the model's own explanation for a decision, and some
    # reason codes are self-describing about which tier they belong to per
    # the editorial contract itself: "usable_alternate" is SWAP-tier by
    # definition ("a usable alternative...that should not play by default"),
    # and "failed_delivery" is DISCARD-tier by definition ("failed/abandoned
    # delivery"). A `select` action paired with either directly contradicts
    # the model's own stated reason -- checked first, ahead of confidence, so
    # a high-confidence self-contradiction is still caught. General, not
    # Video00-specific: it depends only on the fixed reason_code vocabulary.
    if decision.action == "select" and decision.reason_code == "failed_delivery":
        return "discard", "failed_delivery_reason_overrides_select_action"
    if decision.action == "select" and decision.reason_code == "usable_alternate":
        return "swap", "usable_alternate_reason_overrides_select_action"
    if decision.relation == "uncertain" or decision.confidence < 0.70:
        if current_bucket == "select":
            return "select", "uncertain_preserved_current_selected"
        return "swap", "uncertain_preserved_as_swap"
    if decision.action == "discard" and decision.confidence < 0.80:
        return "swap", "low_confidence_discard_demoted_to_swap"
    return decision.action, None


def _enforce_single_retry_family_winner(
    clips: tuple[DraftClip, ...],
    decisions: dict[str, UnifiedSelectionDecision],
    actions: list[str],
    overrides: list[str | None],
) -> None:
    """Within one retry family, a genuine retry contest -- relation
    retry_winner or retry_alternate, i.e. candidates the model itself framed
    as competing takes of the same moment -- must produce at most one SELECT.
    More than one surviving SELECT there is always a policy error, never a
    legitimate composite: composites are relation composite_piece/
    continuation and are untouched by this pass, as is every independent
    story beat. Mutates `actions`/`overrides` in place; keeps the
    highest-confidence contender, demotes the rest to SWAP (never DISCARD --
    an alternate that was good enough to reach SELECT stays available for
    manual replacement, it is not thrown away)."""
    by_family: dict[int, list[int]] = {}
    for index, clip in enumerate(clips):
        decision = decisions[clip.clip_id]
        if decision.relation in ("retry_winner", "retry_alternate"):
            by_family.setdefault(decision.family_index, []).append(index)

    for indices in by_family.values():
        select_indices = [i for i in indices if actions[i] == "select"]
        if len(select_indices) <= 1:
            continue
        winner = max(select_indices, key=lambda i: decisions[clips[i].clip_id].confidence)
        for i in select_indices:
            if i != winner:
                actions[i] = "swap"
                overrides[i] = "retry_family_single_winner_enforced"


def apply_unified_selection_reasoner(
    draft: DraftTimeline,
    reasoner: UnifiedSelectionReasoner | None,
) -> DraftTimeline:
    """Apply one whole-video semantic plan. Provider errors leave the draft untouched."""
    if reasoner is None:
        return draft

    diagnostics = dict(draft.diagnostics or {})
    try:
        plan = validate_unified_selection_plan(draft, reasoner.reason(draft))
    except Exception as exc:
        diagnostics["unified_selection_reasoner"] = {
            "status": "provider_error_fail_open",
            "error": f"{exc.__class__.__name__}: {str(exc)[:240]}",
        }
        return replace(draft, diagnostics=diagnostics)

    clips = _all_clips(draft)
    current = _bucket_map(draft)
    decisions = {decision.clip_id: decision for decision in plan.decisions}

    actions: list[str] = []
    overrides: list[str | None] = []
    for clip in clips:
        decision = decisions[clip.clip_id]
        action, safety_override = _effective_action(decision, current.get(clip.clip_id, "swap"))
        actions.append(action)
        overrides.append(safety_override)

    _enforce_single_retry_family_winner(clips, decisions, actions, overrides)

    selected: list[DraftClip] = []
    alternates: list[DraftClip] = []
    discarded: list[DraftClip] = []
    audit: list[dict] = []

    for index, clip in enumerate(clips):
        decision = decisions[clip.clip_id]
        action = actions[index]
        normalized_clip = replace(clip, selected=(action == "select"))
        if action == "select":
            selected.append(normalized_clip)
        elif action == "swap":
            alternates.append(normalized_clip)
        else:
            discarded.append(normalized_clip)
        audit.append({
            "clip_id": clip.clip_id,
            "previous_bucket": current.get(clip.clip_id),
            "model_action": decision.action,
            "effective_action": action,
            "relation": decision.relation,
            "confidence": round(decision.confidence, 4),
            "family_index": decision.family_index,
            "reason_code": decision.reason_code,
            "safety_override": overrides[index],
        })

    diagnostics["unified_selection_reasoner"] = {
        "status": "applied",
        "provider": plan.provider,
        "model": plan.model,
        "candidate_count": len(clips),
        "selected_count": len(selected),
        "swap_count": len(alternates),
        "discarded_count": len(discarded),
        "estimated_input_tokens": plan.estimated_input_tokens,
        "estimated_output_tokens": plan.estimated_output_tokens,
        "decisions": audit,
    }
    return replace(
        draft,
        selected=tuple(selected),
        alternates=tuple(alternates),
        discarded=tuple(discarded),
        diagnostics=diagnostics,
    )
