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

from .canonical_identity import mint_semantic_idea_id
from .contracts import effective_parent_semantic_clip_id
from .contradiction_signal import any_pair_contradicts
from .selection_boundary_contract import semantic_token_stream

# D-087 SINGLE-TRUTH CONTRACT (docs/CUTSELL_DECISIONS.md D-086/D-087):
#
# In AUTHORITATIVE resolver mode the Unified Realization Resolver is THE
# semantic authority -- it decides winner / composite / review-required per
# semantic idea (realization_resolver.apply_authoritative_realization_
# resolution), and every downstream stage runs on its resolved draft.
# CanonicalEditPlan is the execution-plan REPRESENTATION of that decision
# plus structural validation -- never a second semantic authority, never a
# post-resolver semantic mutator.
#
# Live defect this closes (D-086, run 33952982672, idea
# `idea_d35f7f3f98b9d13c3aa1`): the resolver resolved a diagnosis+hindsight
# family as RESOLVED_COMPOSITE [A, B] (all canonical claims covered, zero
# missing critical claims, overall status SEMANTICALLY_RESOLVED), but
# `build_canonical_edit_plan` derived `is_composite` ONLY from the legacy
# composite evidence sources (`_composite_piece_ids`: CompositeResolver's
# `hybrid_composite_best_take`/`hybrid_semantic_complementary_rescue` split
# ids and `claim_coverage_best_take.composites`). Two selected members with
# no legacy marker -> `unresolved_ambiguous` -> FinalEditReviewer
# DUPLICATE_IDEA + UNRESOLVED_RETRY -> repair loop NEEDS_HUMAN_REVIEW ->
# Freeze BLOCKED -- an already-correct semantic decision silently
# reinterpreted as an unresolved duplicate one layer downstream.
#
# Contract:
#   * `AuthoritativePlanSource` carries the resolver's own per-idea verdict
#     (status, winner/composite realization ids, candidate realizations,
#     covered/missing canonical claims, decision reason). It is built ONLY
#     in AUTHORITATIVE mode (universal_clean_cut.py's cutover branch) from
#     the same `AuthoritativeApplicationResult` + Semantic Ledger every
#     other authoritative stage already consumes. LEGACY/SHADOW never build
#     one, so `build_canonical_edit_plan(draft)` with no source is byte-
#     identical to every pre-D-087 call.
#   * A RESOLVED_COMPOSITE idea whose members all pass the structural
#     validity checks in `_validate_authoritative_composite` is represented
#     as `is_composite=True` / `coverage_status="complete"` with the
#     authoritative members (in the resolver's own member order, never
#     re-sorted by DeliveryScore or clip id) as `winning_clip_ids`.
#   * Any structural invariant failure FAILS CLOSED: the idea stays
#     `unresolved_ambiguous` (so FinalEditReviewer still blocks Freeze) and
#     the exact failures are recorded on the idea -- BLOCK and explain,
#     never invent an alternative semantic answer, never silently swap in a
#     different winner/composite.
#   * The legacy sources stay as evidence/diagnostics (`composite_
#     provenance`, `is_composite_piece`) and, for a group the resolver
#     recorded no decision for, as the fallback exactly as before.

PLAN_SOURCE_LEGACY = "legacy_composite_evidence"
PLAN_SOURCE_AUTHORITATIVE = "authoritative_realization_resolver"

_RESOLVED_WINNER = "RESOLVED_WINNER"
_RESOLVED_COMPOSITE = "RESOLVED_COMPOSITE"
_REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class AuthoritativeIdeaDecision:
    """One semantic idea's final authoritative verdict, exactly as the
    Unified Realization Resolver emitted it -- a read-only handoff record,
    never recomputed here."""

    semantic_idea_id: str
    decision_status: str  # RESOLVED_WINNER | RESOLVED_COMPOSITE | REVIEW_REQUIRED
    winner_realization_id: str | None
    composite_realization_ids: tuple[str, ...]
    candidate_realization_ids: tuple[str, ...]  # every realization the Ledger attributes to this idea
    covered_canonical_claim_ids: tuple[str, ...]
    missing_critical_claim_ids: tuple[str, ...]
    decision_reason: str


@dataclass(frozen=True)
class AuthoritativePlanSource:
    status: str  # SEMANTICALLY_RESOLVED | REVIEW_REQUIRED
    decisions: Mapping[str, AuthoritativeIdeaDecision]  # keyed by semantic_idea_id


def build_authoritative_plan_source(result, ledger=None) -> AuthoritativePlanSource:
    """Build the handoff record from `realization_resolver.Authoritative
    ApplicationResult` (duck-typed: `.status`, `.idea_outcomes`) plus the
    Semantic Ledger it was computed from (for each idea's full candidate
    realization set). Pure projection -- copies the resolver's verdict,
    decides nothing."""
    ledger_ideas = ledger.ideas() if ledger is not None else {}
    decisions: dict[str, AuthoritativeIdeaDecision] = {}
    for outcome in getattr(result, "idea_outcomes", ()) or ():
        idea_id = str(outcome.semantic_idea_id)
        ledger_idea = ledger_ideas.get(idea_id)
        candidates = tuple(dict.fromkeys((
            *(ledger_idea.realization_ids if ledger_idea is not None else ()),
            *(((outcome.winner_realization_id,) if outcome.winner_realization_id else ())),
            *outcome.composite_realization_ids,
            *outcome.discarded_realization_ids,
            *outcome.retained_for_contextual_value,
        )))
        decisions[idea_id] = AuthoritativeIdeaDecision(
            semantic_idea_id=idea_id,
            decision_status=str(outcome.decision_status),
            winner_realization_id=outcome.winner_realization_id,
            composite_realization_ids=tuple(str(r) for r in outcome.composite_realization_ids),
            candidate_realization_ids=tuple(str(r) for r in candidates),
            covered_canonical_claim_ids=tuple(str(c) for c in outcome.covered_canonical_claim_ids),
            missing_critical_claim_ids=tuple(str(c) for c in outcome.missing_critical_claim_ids),
            decision_reason=str(outcome.decision_reason or ""),
        )
    return AuthoritativePlanSource(status=str(getattr(result, "status", "") or ""), decisions=decisions)


def authoritative_plan_source_to_diagnostics(source: AuthoritativePlanSource) -> dict:
    """JSON-safe view stored at `diagnostics["authoritative_plan_source"]`
    (D-087 Section 15) -- also the form `build_canonical_edit_plan` reads
    back when a caller (e.g. live_render_qc's plan re-resolution) has the
    resolved draft but not the in-memory source object."""
    return {
        "schema_version": "cutsell.authoritative_plan_source.v1",
        "plan_semantic_source": PLAN_SOURCE_AUTHORITATIVE,
        "status": source.status,
        "ideas": [
            {
                "semantic_idea_id": d.semantic_idea_id,
                "decision_status": d.decision_status,
                "winner_realization_id": d.winner_realization_id,
                "composite_realization_ids": list(d.composite_realization_ids),
                "candidate_realization_ids": list(d.candidate_realization_ids),
                "covered_canonical_claim_ids": list(d.covered_canonical_claim_ids),
                "missing_critical_claim_ids": list(d.missing_critical_claim_ids),
                "decision_reason": d.decision_reason,
            }
            for d in source.decisions.values()
        ],
    }


def authoritative_plan_source_from_diagnostics(payload) -> AuthoritativePlanSource | None:
    """Inverse of `authoritative_plan_source_to_diagnostics`. Returns None
    for anything that is not a well-formed payload (LEGACY/SHADOW drafts
    never carry the key at all) -- never guesses."""
    if not isinstance(payload, Mapping) or payload.get("plan_semantic_source") != PLAN_SOURCE_AUTHORITATIVE:
        return None
    decisions: dict[str, AuthoritativeIdeaDecision] = {}
    for row in payload.get("ideas") or ():
        if not isinstance(row, Mapping) or not row.get("semantic_idea_id"):
            continue
        idea_id = str(row["semantic_idea_id"])
        winner = row.get("winner_realization_id")
        decisions[idea_id] = AuthoritativeIdeaDecision(
            semantic_idea_id=idea_id,
            decision_status=str(row.get("decision_status") or ""),
            winner_realization_id=str(winner) if winner else None,
            composite_realization_ids=tuple(str(r) for r in (row.get("composite_realization_ids") or ())),
            candidate_realization_ids=tuple(str(r) for r in (row.get("candidate_realization_ids") or ())),
            covered_canonical_claim_ids=tuple(str(c) for c in (row.get("covered_canonical_claim_ids") or ())),
            missing_critical_claim_ids=tuple(str(c) for c in (row.get("missing_critical_claim_ids") or ())),
            decision_reason=str(row.get("decision_reason") or ""),
        )
    return AuthoritativePlanSource(status=str(payload.get("status") or ""), decisions=decisions)


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
    # D-087 Section 15: where THIS idea's semantic decision came from and
    # whether the plan's structural validation of it passed. All defaulted
    # so every pre-D-087 construction site (and every LEGACY/SHADOW plan,
    # which never has an authoritative source) is unchanged.
    plan_semantic_source: str = PLAN_SOURCE_LEGACY
    authoritative_resolution_status: str | None = None
    authoritative_composite_realization_ids: tuple[str, ...] = ()
    authoritative_resolved_clip_ids: tuple[str, ...] = ()
    authoritative_claim_coverage: tuple[str, ...] = ()
    authoritative_decision_reason: str | None = None
    structural_validation_passed: bool | None = None
    structural_validation_failures: tuple[str, ...] = ()


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
    # D-087: which semantic source this plan's idea decisions were
    # represented from -- `legacy_composite_evidence` (LEGACY/SHADOW, or no
    # authoritative source available) or `authoritative_realization_
    # resolver`. Defaulted for every pre-D-087 construction site.
    plan_semantic_source: str = PLAN_SOURCE_LEGACY


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


def _composite_is_contradiction_free(winning_clip_ids: tuple[str, ...], clip_by_id: Mapping[str, object]) -> bool:
    """D-056.3 CONTRADICTION-SAFE COMPOSITE CONTRACT: the ONE structural gate
    that makes it impossible for a factually-contradictory pair to be
    reported `is_composite: true` / `coverage_status: complete` here --
    the field Selection Freeze/Boundary/Renderer actually consume, no
    matter which upstream mechanism (`claim_coverage_best_take.py`'s own
    2-piece fallback, `hybrid_composite_best_take`,
    `hybrid_semantic_complementary_rescue`, or any future one) proposed the
    composite. Live defect this closes (docs/CUTSELL_DECISIONS.md D-056.2/
    D-056.3): `_composite_piece_ids` above never checked for a factual
    contradiction between a composite's own members before this fix --
    D-056.2 Run B (`tg_539b31f663aaf9e13f`) and Run C
    (`tg_f4b9e7c1fe3e28a1af`) both a negation-conflicting pair reported
    `is_composite: true` here while FinalEditReviewer's independent
    CONTRADICTION check (reading `contradiction_findings` below) still
    flagged the exact same pair -- two safety layers that disagreed.

    Uses the SAME shared contradiction contract StoryValidator's own
    `_contradiction_findings`/`_resolve_residual_family` use (see
    `contradiction_signal.py`) -- never a second, independently-derived
    verdict. Members whose text cannot be resolved from `clip_by_id` are
    skipped, never invented; this only narrows available evidence, never
    widens what counts as a conflict."""
    texts = [str(clip.text) for cid in winning_clip_ids if (clip := clip_by_id.get(cid)) is not None]
    return not any_pair_contradicts(texts)


def _composite_provenance_rows(diagnostics: Mapping[str, object]) -> tuple[dict, ...]:
    rows: list[dict] = []
    for row in diagnostics.get("hybrid_editorial_chunks") or ():
        if not isinstance(row, dict):
            continue
        entry = row.get("hybrid_composite_best_take")
        if isinstance(entry, dict) and entry.get("composite_replacements"):
            rows.extend(entry["composite_replacements"])
    return tuple(rows)


def _all_draft_clips(draft) -> list:
    return [*draft.selected, *(getattr(draft, "alternates", ()) or ()), *draft.discarded]


def _clip_realization_id(clip, clip_by_id_all: Mapping[str, object]) -> str | None:
    """The realization a clip physically realizes: its own `realization_id`
    (D-050A -- a Boundary split leaves it equal to the parent's), else, for
    a D-036/D-046 fragment carrying only `parent_semantic_clip_id`, the
    parent clip's realization when that parent is still present in the
    draft. None when provenance cannot be resolved -- never guessed."""
    explicit = getattr(clip, "realization_id", None)
    if explicit:
        return str(explicit)
    parent_realization = getattr(clip, "parent_realization_id", None)
    if parent_realization:
        return str(parent_realization)
    parent_clip_id = effective_parent_semantic_clip_id(clip)
    if parent_clip_id is not None:
        parent = clip_by_id_all.get(parent_clip_id)
        if parent is None:
            return None  # stale provenance: the parent is gone from every bucket
        parent_explicit = getattr(parent, "realization_id", None)
        return str(parent_explicit) if parent_explicit else str(parent.clip_id)
    return str(clip.clip_id)


def _group_semantic_idea_id(group_id: str, member_clips: list) -> str:
    """The semantic idea a take_judge group realizes: the one
    `semantic_idea_id` its members were stamped with (D-050A), else the
    deterministic mint from the group id (the exact same function
    pipeline.py's `_draft_clip` used to stamp them)."""
    stamped = {
        str(getattr(c, "semantic_idea_id", None))
        for c in member_clips if getattr(c, "semantic_idea_id", None)
    }
    if len(stamped) == 1:
        return next(iter(stamped))
    return mint_semantic_idea_id(group_id)


def _validate_authoritative_composite(
    decision: AuthoritativeIdeaDecision,
    *,
    group_member_rids: frozenset,
    legacy_winning_clip_ids: tuple[str, ...],
    rids_for_member: Mapping[str, frozenset],
    selected_clips_by_rid: Mapping[str, list],
    known_rids: frozenset,
    clip_by_id: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """D-087 Section 4/6/10/11: structural validity of one RESOLVED_COMPOSITE
    against the resolved draft. Returns `(resolved_clip_ids, failures)`;
    the composite is accepted only when `failures` is empty. Every check
    fails closed; none of them re-decides semantics."""
    failures: list[str] = []
    members = tuple(decision.composite_realization_ids)
    if not members:
        failures.append("empty_composite")
        return (), tuple(failures)
    if len(set(members)) != len(members):
        failures.append("duplicate_composite_member_ids")
    if decision.decision_status != _RESOLVED_COMPOSITE:
        failures.append(f"decision_status_not_resolved_composite:{decision.decision_status}")
    if decision.missing_critical_claim_ids:
        failures.append("missing_critical_claim_ids_nonempty:" + ",".join(decision.missing_critical_claim_ids))
    candidates = frozenset(decision.candidate_realization_ids)
    resolved_clip_ids: list[str] = []
    for rid in dict.fromkeys(members):
        if rid not in known_rids:
            failures.append(f"unknown_realization:{rid}")
            continue
        if candidates and rid not in candidates:
            failures.append(f"realization_outside_semantic_idea:{rid}")
        if rid not in group_member_rids:
            failures.append(f"realization_outside_take_group:{rid}")
        selected_clips = selected_clips_by_rid.get(rid) or []
        if not selected_clips:
            failures.append(f"realization_not_selected:{rid}")
            continue
        for clip in selected_clips:
            stamped_idea = getattr(clip, "semantic_idea_id", None)
            if stamped_idea and str(stamped_idea) != decision.semantic_idea_id:
                failures.append(f"member_clip_stamped_with_other_idea:{clip.clip_id}")
            parent_realization = getattr(clip, "parent_realization_id", None)
            if parent_realization and str(parent_realization) != rid:
                failures.append(f"stale_fragment_provenance:{clip.clip_id}")
            resolved_clip_ids.append(clip.clip_id)
    resolved = tuple(dict.fromkeys(resolved_clip_ids))
    # Every legacy-visible surviving member of this group must be explained
    # by the composite -- an extra selected member the resolver never named
    # is a genuinely unresolved family, not a composite (Section 9).
    # Compared in REALIZATION terms so a member that survives only as
    # Boundary fragments (D-046 Case A: the pre-split clip id is what
    # `take_judge_groups` names) is explained by its realization, never
    # misreported as an extra member.
    member_set = frozenset(members)
    unexplained = tuple(
        cid for cid in legacy_winning_clip_ids
        if not (rids_for_member.get(cid, frozenset()) & member_set)
    )
    if unexplained:
        failures.append("selected_members_outside_composite:" + ",".join(unexplained))
    if resolved and not _composite_is_contradiction_free(resolved, clip_by_id):
        failures.append("composite_members_contradict")
    return resolved, tuple(failures)


@dataclass(frozen=True)
class AuthoritativeGroupAssessment:
    """D-090: one `take_judge_groups` family assessed against the resolved
    draft AND (when present) the D-087 authoritative verdict for its
    semantic idea. This is the ONE structural-membership assessment both
    `build_canonical_edit_plan` (the plan's idea rows) and the post-
    authority StoryValidator pass (`final_story_coherence_validation.
    apply_post_authority_story_validation`, family bookkeeping) consume, so
    the two can never disagree about whether a multi-member family is a
    valid authoritative composite or a genuinely unresolved contest.

    `accepted_as_resolved` is True only for a RESOLVED_COMPOSITE / RESOLVED_
    WINNER decision whose structural validation passed against the draft
    (every named member selected, none outside the idea/group, no extra
    unexplained selected member, contradiction-free). REVIEW_REQUIRED, a
    missing decision, or any structural failure leaves it False -- nothing
    is ever blindly exempted because a composite label exists."""
    group_id: str
    semantic_idea_id: str
    member_clip_ids: tuple[str, ...]
    legacy_winning_clip_ids: tuple[str, ...]
    legacy_discarded_clip_ids: tuple[str, ...]
    decision_status: str | None
    winner_realization_id: str | None
    composite_realization_ids: tuple[str, ...]
    covered_canonical_claim_ids: tuple[str, ...]
    decision_reason: str | None
    plan_semantic_source: str
    winning_clip_ids: tuple[str, ...]
    is_composite: bool
    coverage_status: str
    resolved_clip_ids: tuple[str, ...]
    structural_validation_passed: bool | None
    structural_validation_failures: tuple[str, ...]
    accepted_as_resolved: bool


def assess_authoritative_membership(
    draft, authoritative_source: AuthoritativePlanSource | None,
) -> dict[str, AuthoritativeGroupAssessment]:
    """Assess every `take_judge_groups` family of `draft` -- see
    `AuthoritativeGroupAssessment`. With `authoritative_source=None` the
    assessment is the legacy composite-evidence path only (D-025/D-056.3),
    byte-for-byte what the pre-D-087 plan computed."""
    diagnostics = draft.diagnostics or {}
    composite_ids = _composite_piece_ids(diagnostics)
    selected_ids = {clip.clip_id for clip in draft.selected}
    clip_by_id = {clip.clip_id: clip for clip in draft.selected}
    all_clips = _all_draft_clips(draft)
    clip_by_id_all = {clip.clip_id: clip for clip in all_clips}
    rid_by_clip: dict[str, str | None] = {}
    # D-046/D-050A: a member `take_judge_groups` names by its PRE-split
    # clip id may survive only as fragments naming it via
    # `parent_semantic_clip_id` -- those fragments' realizations are that
    # member's realizations too.
    rids_by_parent_clip: dict[str, set[str]] = {}
    selected_clips_by_rid: dict[str, list] = {}
    known_rids: set[str] = set()
    if authoritative_source is not None:
        for clip in all_clips:
            rid = _clip_realization_id(clip, clip_by_id_all)
            rid_by_clip[clip.clip_id] = rid
            if rid is not None:
                known_rids.add(rid)
                parent_clip_id = effective_parent_semantic_clip_id(clip)
                if parent_clip_id is not None:
                    rids_by_parent_clip.setdefault(parent_clip_id, set()).add(rid)
        for clip in draft.selected:  # draft.selected order == rendered order
            rid = rid_by_clip.get(clip.clip_id)
            if rid is not None:
                selected_clips_by_rid.setdefault(rid, []).append(clip)

    def _rids_for_member(cid: str) -> frozenset:
        rids = set(rids_by_parent_clip.get(cid, ()))
        own = rid_by_clip.get(cid)
        if own is not None:
            rids.add(own)
        return frozenset(rids)
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

    assessments: dict[str, AuthoritativeGroupAssessment] = {}
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
        is_accepted_composite = (
            len(winning) >= 2
            and all(cid in composite_ids for cid in winning)
            # D-056.3: an upstream mechanism marking every member a
            # "composite piece" is necessary but no longer sufficient --
            # the members must also not factually contradict each other.
            # See _composite_is_contradiction_free's own docstring for the
            # exact live defect this closes.
            and _composite_is_contradiction_free(winning, clip_by_id)
        )
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
        group_id = str(group.get("group_id") or "")
        member_clips = [clip_by_id_all[cid] for cid in member_ids if cid in clip_by_id_all]
        semantic_idea_id = _group_semantic_idea_id(group_id, member_clips)

        # D-087: in AUTHORITATIVE mode the resolver's verdict is the
        # canonical source of this idea's representation. Everything above
        # stays computed as legacy EVIDENCE (and as the fallback for a group
        # the resolver recorded no decision for).
        decision = None
        if authoritative_source is not None:
            decision = authoritative_source.decisions.get(semantic_idea_id)
        plan_semantic_source = PLAN_SOURCE_LEGACY
        structural_passed: bool | None = None
        structural_failures: tuple[str, ...] = ()
        resolved_clip_ids: tuple[str, ...] = ()
        accepted_as_resolved = False
        if decision is not None:
            plan_semantic_source = PLAN_SOURCE_AUTHORITATIVE
            rids_for_member = {cid: _rids_for_member(cid) for cid in member_ids}
            group_member_rids = frozenset().union(*rids_for_member.values()) if rids_for_member else frozenset()
            if decision.decision_status == _RESOLVED_COMPOSITE:
                resolved, failures = _validate_authoritative_composite(
                    decision,
                    group_member_rids=group_member_rids,
                    legacy_winning_clip_ids=winning,
                    rids_for_member=rids_for_member,
                    selected_clips_by_rid=selected_clips_by_rid,
                    known_rids=frozenset(known_rids),
                    clip_by_id=clip_by_id,
                )
                structural_passed = not failures
                structural_failures = tuple(failures)
                resolved_clip_ids = tuple(resolved)
                if not failures:
                    # Section 3: represent the resolver's composite --
                    # members in the resolver's own order, never re-sorted
                    # by DeliveryScore/clip id (Section 12).
                    winning = resolved
                    is_accepted_composite = True
                    coverage_status = "complete"
                    accepted_as_resolved = True
                else:
                    # Section 6: BLOCK and explain -- never invent an
                    # alternative semantic answer. Keeping the family
                    # `unresolved_ambiguous` is what makes FinalEditReviewer
                    # still block Freeze; the failures are carried on the
                    # idea so the block is explained, not silent.
                    is_accepted_composite = False
                    coverage_status = "unresolved_ambiguous" if winning else "missing"
            elif decision.decision_status == _RESOLVED_WINNER and decision.winner_realization_id:
                winner_clips = selected_clips_by_rid.get(decision.winner_realization_id) or []
                winner_clip_ids = tuple(c.clip_id for c in winner_clips)
                failures = []
                if decision.winner_realization_id not in known_rids:
                    failures.append(f"unknown_realization:{decision.winner_realization_id}")
                elif not winner_clips:
                    failures.append(f"realization_not_selected:{decision.winner_realization_id}")
                # Section 6: the surviving members must BE the resolver's
                # winner -- never a different winner represented silently.
                if winning and set(winning) != set(winner_clip_ids):
                    failures.append(
                        "selected_members_differ_from_authoritative_winner:" + ",".join(winning)
                    )
                structural_passed = not failures
                structural_failures = tuple(failures)
                resolved_clip_ids = winner_clip_ids
                if failures and winning:
                    is_accepted_composite = False
                    coverage_status = "unresolved_ambiguous"
                accepted_as_resolved = not failures and bool(winner_clip_ids)
            else:
                # REVIEW_REQUIRED (or a malformed status): the resolver
                # itself declined to decide -- the legacy representation
                # above stands, which keeps such a family unresolved.
                structural_passed = None
                structural_failures = ()
        elif authoritative_source is not None:
            structural_passed = None
            structural_failures = ("no_authoritative_decision_recorded_for_group",)
        assessments[group_id] = AuthoritativeGroupAssessment(
            group_id=group_id,
            semantic_idea_id=semantic_idea_id,
            member_clip_ids=tuple(member_ids),
            legacy_winning_clip_ids=tuple(
                cid for cid in member_ids if cid in selected_ids or cid in selected_parent_ids
            ),
            legacy_discarded_clip_ids=discarded,
            decision_status=decision.decision_status if decision is not None else None,
            winner_realization_id=decision.winner_realization_id if decision is not None else None,
            composite_realization_ids=tuple(decision.composite_realization_ids) if decision is not None else (),
            covered_canonical_claim_ids=tuple(decision.covered_canonical_claim_ids) if decision is not None else (),
            decision_reason=decision.decision_reason if decision is not None else None,
            plan_semantic_source=plan_semantic_source,
            winning_clip_ids=tuple(winning),
            is_composite=is_accepted_composite,
            coverage_status=coverage_status,
            resolved_clip_ids=resolved_clip_ids,
            structural_validation_passed=structural_passed,
            structural_validation_failures=structural_failures,
            accepted_as_resolved=accepted_as_resolved,
        )
    return assessments


def build_canonical_edit_plan(draft, *, authoritative_source: AuthoritativePlanSource | None = None) -> CanonicalEditPlan:
    """Build the CanonicalEditPlan from a draft that has already been through
    Final Story Coherence Validation (so `diagnostics["final_story_
    coherence_validation"]` and `take_judge_groups` are populated).

    D-087: `authoritative_source` (AUTHORITATIVE resolver mode only -- see
    the module-level SINGLE-TRUTH CONTRACT note) makes the Unified
    Realization Resolver's own per-idea verdict the canonical source of
    each idea's representation. When omitted, the draft's own
    `diagnostics["authoritative_plan_source"]` is consulted (present only
    on a draft the AUTHORITATIVE cutover produced); absent both, the
    legacy composite-evidence path below runs byte-for-byte unchanged.

    D-090: the per-family structural assessment lives in
    `assess_authoritative_membership` (shared with the post-authority
    StoryValidator pass); this function only turns it into plan rows."""
    diagnostics = draft.diagnostics or {}
    coherence = diagnostics.get("final_story_coherence_validation") or {}
    composite_ids = _composite_piece_ids(diagnostics)
    if authoritative_source is None:
        authoritative_source = authoritative_plan_source_from_diagnostics(
            diagnostics.get("authoritative_plan_source")
        )
    plan_semantic_source = PLAN_SOURCE_LEGACY if authoritative_source is None else PLAN_SOURCE_AUTHORITATIVE
    authoritative_piece_ids: set[str] = set()

    ideas: list[EditPlanIdea] = []
    for assessment in assess_authoritative_membership(draft, authoritative_source).values():
        idea_extra: dict = {}
        if assessment.decision_status is not None:
            idea_extra = {
                "plan_semantic_source": PLAN_SOURCE_AUTHORITATIVE,
                "authoritative_resolution_status": assessment.decision_status,
                "authoritative_composite_realization_ids": assessment.composite_realization_ids,
                "authoritative_claim_coverage": assessment.covered_canonical_claim_ids,
                "authoritative_decision_reason": assessment.decision_reason,
                "structural_validation_passed": assessment.structural_validation_passed,
                "structural_validation_failures": assessment.structural_validation_failures,
            }
            if assessment.decision_status in (_RESOLVED_COMPOSITE, _RESOLVED_WINNER):
                idea_extra["authoritative_resolved_clip_ids"] = assessment.resolved_clip_ids
            if assessment.decision_status == _RESOLVED_COMPOSITE and assessment.structural_validation_passed:
                authoritative_piece_ids.update(assessment.resolved_clip_ids)
        elif authoritative_source is not None:
            idea_extra = {
                "plan_semantic_source": PLAN_SOURCE_LEGACY,
                "structural_validation_passed": None,
                "structural_validation_failures": assessment.structural_validation_failures,
            }
        ideas.append(EditPlanIdea(
            idea_id=assessment.group_id,
            winning_clip_ids=assessment.winning_clip_ids,
            is_composite=assessment.is_composite,
            discarded_clip_ids=assessment.legacy_discarded_clip_ids,
            coverage_status=assessment.coverage_status,
            **idea_extra,
        ))

    clip_to_idea: dict[str, str] = {
        cid: idea.idea_id for idea in ideas for cid in idea.winning_clip_ids
    }
    composite_ids = frozenset(composite_ids | authoritative_piece_ids)

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
        plan_semantic_source=plan_semantic_source,
    )
