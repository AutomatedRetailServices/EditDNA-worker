"""General causal/story order validation -- D-027.

STORY_ORDER_BREAK (final_edit_reviewer.py) is deliberately narrow: it only
checks that one accepted composite's OWN components stay in recording order
relative to each other. It never looks across independent Ideas, because the
Composer stage is explicitly allowed to reorder independent ideas for
pacing/sales logic -- a blanket "final order must match source order" check
across all ideas would false-positive on that legitimate behavior.

This module is the general, cross-idea complement the canonical directive
asks for: detect when a clip that is a DEPENDENT CONSEQUENCE or CONTINUATION
of an earlier clip (diagnosis after its test, CTA after its required body,
a "that's why"/"therefore" explanation after the fact it explains, a
continuation after its parent) ends up placed BEFORE the clip it depends on
in the final KEEP sequence, or has that required clip missing from KEEP
entirely (a detached, context-free explanation).

## What evidence this uses (per the canonical directive -- "use general
metadata", never Video00 facts/diseases/phrases/timestamps)

1. **Source chronology**: `EditPlanClip.source_asset_id` + `start` --
   dependency is only inferred between clips recorded in the SAME
   continuous source, close enough together (`_MAX_SOURCE_GAP_SEC`) to be
   the same train of thought -- exactly the same "continuous take" signal
   `take_grouping_provider.py` already uses for retry-family adjacency.
2. **Connector language**: a small, general (English + Spanish) lexicon of
   phrases that mark a clause as a dependent consequence/continuation of
   whatever preceded it ("therefore", "that's why", "and that confirmed",
   "por eso", "como resultado", ...). These are structural/grammatical
   connectors, not Video00 vocabulary -- the same kind of general lexical
   signal `take_grouping.py`'s `retry_similarity` already relies on.
3. **Idea/retry-family provenance**: `EditPlanClip.idea_id` is carried
   through so a finding can name which Idea is misplaced relative to which.

## Confidence and the bounded SemanticArbiter escalation path

A STRONG connector (an unambiguous causal/consequence marker) is enough
deterministic evidence on its own to flag a real order violation -- no
arbiter needed. A WEAK/generic connector is treated as insufficient
deterministic evidence on its own (matching CLAUDE.md's "WHEN UNCERTAIN,
KEEP" -- here, that means "do not claim a break you are not sure of"): it is
only escalated to `CausalOrderArbiter.check_dependency`, and only becomes a
finding if the arbiter confirms it. Fails open exactly like
`semantic_idea_equivalence.py`'s existing arbiter: no arbiter configured, or
any arbiter exception, means the weak/ambiguous case is silently DROPPED,
never flagged -- false-positive prevention takes priority, since a wrongly
blocked Freeze is a cost, but this module has no repair strategy of its own
either way (see final_edit_reviewer.py: CAUSAL_ORDER_BREAK routes to human
review, it is never auto-reordered -- a cross-idea reorder risks undoing an
intentional Composer pacing choice, unlike STORY_ORDER_BREAK's safe within-
composite reorder).

## Honest gap

`CausalOrderArbiter` is a real, usable Protocol (mirrors
`semantic_idea_equivalence.SemanticEquivalenceArbiter`'s shape and fail-open
contract exactly), and `find_causal_order_breaks` already calls it when one
is supplied. No live Gemini-backed implementation of it exists yet in this
codebase as of D-027 -- that is a new provider/prompt module, same shape of
work as `semantic_idea_equivalence_google.py`, and is explicitly not built
this cycle. Every caller defaults `causal_order_arbiter` to `None`, which is
exactly the already-established fail-open behavior for an absent arbiter
elsewhere in this codebase, not a degraded or partial mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .canonical_edit_plan import CanonicalEditPlan

# General, language-general (English + Spanish) connector phrases marking a
# clause as a dependent consequence/continuation of something said earlier.
# Matched only as a PREFIX of a clip's own text (after lowercasing/strip),
# never as a substring search over arbitrary content -- deliberately
# conservative so this cannot fire on a phrase merely mentioned mid-clip.
# No Video00-specific fact, disease, product, or phrase appears here.
_STRONG_DEPENDENCY_CONNECTORS: tuple[str, ...] = (
    "therefore", "as a result", "that's why", "that is why",
    "and that confirmed", "and that's how", "and that is how",
    "which confirmed", "and that showed", "as a consequence",
    "por lo tanto", "como resultado", "eso confirmó", "eso confirmo",
    "y eso confirmó", "y eso confirmo", "y así fue como", "y asi fue como",
    "lo cual confirmó", "lo cual confirmo",
)

_WEAK_DEPENDENCY_CONNECTORS: tuple[str, ...] = (
    "so ", "so,", "which means", "that means", "because of that",
    "because of this", "after that", "which showed",
    "entonces", "por eso", "así que", "asi que", "eso significa",
    "debido a eso", "debido a esto", "y por eso", "después de eso",
    "despues de eso", "lo que confirma", "lo que significa",
)

_STRONG_CONFIDENCE = 0.9
_WEAK_CONFIDENCE = 0.55
_STRONG_CONFIDENCE_THRESHOLD = 0.8  # at/above this, deterministic evidence stands alone
_MAX_SOURCE_GAP_SEC = 45.0


@dataclass(frozen=True)
class CausalDependency:
    """`dependent_clip_id` reads as a dependent consequence/continuation of
    `required_clip_id`, per the evidence recorded here."""

    dependent_clip_id: str
    required_clip_id: str
    dependent_idea_id: str | None
    required_idea_id: str | None
    evidence: str
    confidence: float
    resolved_by: str  # "deterministic_connector_language" | "semantic_arbiter"


class CausalOrderArbiter(Protocol):
    """Bounded semantic arbiter for exactly one narrow question per pair:
    does the second delivery read as a dependent consequence/continuation of
    the first, such that a viewer needs the first to make sense of the
    second? Text only, no clip identity -- mirrors
    `semantic_idea_equivalence.SemanticEquivalenceArbiter`'s contract shape
    so it cannot become a Video00-specific rule even by construction.
    Returns (is_dependent, confidence 0..1, short general reason)."""

    def check_dependency(self, required_text: str, dependent_text: str) -> tuple[bool, float, str]: ...


def _connector_prefix_hit(text: str) -> tuple[str, float] | None:
    lowered = (text or "").strip().lower()
    for phrase in _STRONG_DEPENDENCY_CONNECTORS:
        if lowered.startswith(phrase):
            return phrase, _STRONG_CONFIDENCE
    for phrase in _WEAK_DEPENDENCY_CONNECTORS:
        if lowered.startswith(phrase):
            return phrase, _WEAK_CONFIDENCE
    return None


@dataclass(frozen=True)
class _ClipRef:
    """One clip, kept or discarded, reduced to the fields this module needs
    to reconstruct source chronology. Discarded clips are included so a
    required-context clip that was discarded entirely (not just misordered)
    can still be found -- "a dependent explanation detached from the fact it
    explains" per the canonical directive."""

    clip_id: str
    source_asset_id: str
    start: float
    end: float
    text: str
    idea_id: str | None
    in_keep: bool


def _clip_pool(edit_plan: CanonicalEditPlan) -> list[_ClipRef]:
    pool = [
        _ClipRef(c.clip_id, c.source_asset_id, c.start, c.end, c.text, c.idea_id, True)
        for c in edit_plan.keep_sequence
    ]
    pool.extend(
        _ClipRef(r.clip_id, r.source_asset_id, r.start, r.end, r.text, None, False)
        for r in edit_plan.discard_provenance
    )
    return pool


def _find_deterministic_dependencies(pool: list[_ClipRef]) -> list[CausalDependency]:
    """Adjacent-in-source-recording pairs where a KEPT, later-recorded
    clip's text opens with a general dependency connector. Scoped to
    same-source, close-in-time pairs only -- the same continuous-take
    evidence `take_grouping_provider.py` already relies on for retry
    adjacency -- so this never fires across two ideas that merely happen to
    share a connector word but were recorded far apart or in different
    sources. The nearest-earlier-clip search includes discarded clips (see
    `_clip_pool`): only a KEPT clip can ever be the flagged dependent side,
    but its required context may have been discarded entirely."""
    by_source: dict[str, list[_ClipRef]] = {}
    for clip in pool:
        by_source.setdefault(clip.source_asset_id, []).append(clip)

    dependencies: list[CausalDependency] = []
    for clips in by_source.values():
        ordered = sorted(clips, key=lambda c: c.start)
        for index, later in enumerate(ordered):
            if not later.in_keep:
                continue  # only a clip that actually survived can be "out of order"
            hit = _connector_prefix_hit(later.text)
            if hit is None:
                continue
            connector, confidence = hit
            candidates = [
                c for c in ordered[:index]
                if c.end <= later.start and (later.start - c.end) <= _MAX_SOURCE_GAP_SEC
            ]
            if not candidates:
                continue
            earlier = max(candidates, key=lambda c: c.end)
            dependencies.append(CausalDependency(
                dependent_clip_id=later.clip_id,
                required_clip_id=earlier.clip_id,
                dependent_idea_id=later.idea_id,
                required_idea_id=earlier.idea_id if earlier.in_keep else None,
                evidence=f"connector_language:{connector.strip()}",
                confidence=confidence,
                resolved_by="deterministic_connector_language",
            ))
    return dependencies


def _resolve_ambiguous_with_arbiter(
    dependencies: list[CausalDependency],
    pool: list[_ClipRef],
    arbiter: CausalOrderArbiter | None,
) -> list[CausalDependency]:
    """Strong deterministic evidence is never second-guessed. A weak hit is
    insufficient deterministic evidence on its own: it is dropped silently
    unless a bounded arbiter is given AND confirms it -- fails open toward
    NOT flagging, exactly like `semantic_idea_equivalence.py`'s existing
    arbiter direction (absence of confirming evidence means "do not act")."""
    text_by_id = {c.clip_id: c.text for c in pool}
    resolved: list[CausalDependency] = []
    for dep in dependencies:
        if dep.confidence >= _STRONG_CONFIDENCE_THRESHOLD:
            resolved.append(dep)
            continue
        if arbiter is None:
            continue  # insufficient deterministic evidence, no arbiter available -- drop
        try:
            is_dependent, confidence, reason = arbiter.check_dependency(
                text_by_id.get(dep.required_clip_id, ""), text_by_id.get(dep.dependent_clip_id, ""),
            )
        except Exception:
            continue  # arbiter failure -- treat exactly like "not available"
        if not is_dependent:
            continue
        resolved.append(CausalDependency(
            dependent_clip_id=dep.dependent_clip_id,
            required_clip_id=dep.required_clip_id,
            dependent_idea_id=dep.dependent_idea_id,
            required_idea_id=dep.required_idea_id,
            evidence=f"semantic_arbiter:{str(reason)[:160]}",
            confidence=float(confidence),
            resolved_by="semantic_arbiter",
        ))
    return resolved


def find_causal_order_breaks(
    edit_plan: CanonicalEditPlan, *, arbiter: CausalOrderArbiter | None = None,
) -> tuple[CausalDependency, ...]:
    """Deterministic + bounded-arbiter general causal/story order check
    across the final KEEP sequence. Returns only dependencies that are
    actually violated in `keep_sequence`'s real order: the required clip is
    placed after the dependent clip, or is missing from KEEP entirely (a
    detached, context-free explanation) -- not every observed dependency."""
    position = {c.clip_id: i for i, c in enumerate(edit_plan.keep_sequence)}
    pool = _clip_pool(edit_plan)
    deterministic = _find_deterministic_dependencies(pool)
    resolved = _resolve_ambiguous_with_arbiter(deterministic, pool, arbiter)

    breaks: list[CausalDependency] = []
    for dep in resolved:
        dependent_pos = position.get(dep.dependent_clip_id)
        if dependent_pos is None:
            continue  # the dependent clip itself is not even kept -- not an order question
        required_pos = position.get(dep.required_clip_id)
        if required_pos is None or required_pos > dependent_pos:
            breaks.append(dep)
    return tuple(breaks)
