"""D-050B: the provider-neutral Semantic Ledger, in SHADOW MODE.

See docs/CUTSELL_DECISIONS.md D-050 (audit), D-050A (canonical identities),
and D-050B (this module) for full context.

SHADOW MODE CONTRACT (binding for every line in this file)
============================================================
The Ledger OBSERVES and RECORDS the current engine's already-made semantic
decisions. It never makes one. Nothing in the active pipeline (grouping,
DeliveryScorer, semantic best-take, ClaimCoverage, StoryValidator,
CanonicalEditPlan, FinalEditReviewer, Freeze, Boundary, Render/QC) reads
a `SemanticLedger` value to branch on. `build_semantic_ledger_shadow`
below is called exactly once, read-only, after every stage it observes
has already run and already written its own authoritative diagnostics --
see that function's own docstring for exactly which existing diagnostics
keys it reconstructs from, and its explicit list of best-effort/partial
reconstructions (this module reports what it cannot fully verify rather
than inventing certainty; see `LedgerParityReport`).

WRITE OWNERSHIP (directive Section 8)
======================================
`SemanticLedger`'s seven `register_*`/`record_*` methods are the ONLY
sanctioned way to mutate a Ledger. Every internal collection is name-
mangled and never handed out mutable; every read accessor returns an
immutable view (a `types.MappingProxyType` or a `tuple`). A caller that
wants to inspect Ledger state calls `realizations()`/`ideas()`/`claims()`/
`decisions()`/`discards()`/`composites()`/`coverage()` -- never touches
`_realizations` etc. directly.

`register_realization`/`register_semantic_idea`/`register_claim` are
idempotent when called twice with byte-identical data (a stage rebuilding
its own view mid-run is not an error) but raise `LedgerIntegrityError` the
moment two calls disagree about the same id's content -- "no stage may
silently create duplicate entries for the same semantic_idea_id" is
enforced structurally, not by convention.

PHYSICAL FRAGMENTS ARE NEVER NEW REALIZATIONS
==============================================
`record_physical_fragment` only ever attaches a `render_fragment_id` to an
EXISTING `RealizationRecord` (looked up by `realization_id`, which
D-050A's `parent_realization_id`/`realization_id` contract guarantees is
unchanged across a physical split). There is no method that mints a new
`RealizationRecord` from a fragment, so a fragment can never accidentally
become a second semantic identity for the same delivery.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import types
from typing import Iterable, Mapping, Sequence

from .canonical_identity import mint_canonical_claim_id
from .semantic_claims import Claim, extract_claims


class LedgerIntegrityError(ValueError):
    """Raised only when two writes disagree about the same id's content --
    never for merely re-observing the same fact twice."""


# ---------------------------------------------------------------------------
# Typed records (Section 1-7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RealizationRecord:
    realization_id: str
    semantic_idea_id: str | None
    retry_family_id: str | None
    source_span_ids: tuple[str, ...]
    attempt_id: str | None
    clip_ids: tuple[str, ...]          # every legacy clip_id seen for this realization
    text: str
    start: float                        # physical observation only -- never a decision input
    end: float                          # physical observation only -- never a decision input
    delivery_score: float | None
    state: str                          # "selected" | "discarded" | "alternate"
    discard_reason: str | None
    replacement_realization_id: str | None
    claim_ids: tuple[str, ...]
    render_fragment_ids: tuple[str, ...]
    # D-050C1.6 (F5, composite completeness safety): carried from
    # `DraftClip.complete_idea` (itself carried unchanged from the
    # CandidateTake this clip was built from -- see that field's own
    # docstring). `None` when the underlying clip predates this field or
    # a construction site never set it -- reported as UNKNOWN, never
    # guessed at as True.
    complete_idea: bool | None = None


@dataclass(frozen=True)
class CanonicalClaimRecord:
    canonical_claim_id: str
    claim_type: str
    content_tokens: frozenset
    importance: str
    source_realization_ids: tuple[str, ...]
    covered_by_realization_ids: tuple[str, ...]
    coverage_state: str                 # "covered" | "missing" | "unresolved"
    # D-050C1: the claim's own raw clause text (`semantic_claims.Claim.
    # text`), observation-only -- optional/defaulted so every existing
    # construction site stays valid unchanged. `content_tokens` alone
    # loses a short digit run ("5%", 2 characters) to `_content`'s own
    # >=3-character floor while keeping a longer one ("10%", "5-10%")
    # intact -- an asymmetry the resolver's quantitative-meaning dedup
    # check (realization_resolver.py) needs the raw text to see past.
    # Nothing in claim_coverage_best_take.py or any other authoritative
    # stage reads this field.
    text: str = ""
    # D-073: `semantic_claims.Claim.negation_role` (D-066), carried
    # through observation-only -- "" when absent/not-applicable, never
    # guessed. Lets realization_resolver.py's PATH B semantic replacement
    # certification reuse D-066's CONTRASTIVE_HINDSIGHT_NEGATION safe
    # contract on the Ledger's own claim representation, without this
    # module making any classification decision of its own.
    negation_role: str = ""


IDEA_STATUS_ACTIVE = "ACTIVE"
IDEA_STATUS_REMOVED_EXPLICITLY = "REMOVED_EXPLICITLY"
IDEA_STATUS_BLOCKED = "BLOCKED"

# D-050C1.6 F6/F7: the engine's own FINAL resolution shape for one idea --
# a typed, ground-truth representation computed directly from realization
# states + recorded composites (see `_finalize_engine_resolution` below),
# never from `current_winner_realization_id`'s own event-order history
# alone (a later composite silently superseding an earlier single-winner
# decision was exactly F6's bug). A comparison (`build_resolver_parity_
# report`) MUST branch on this field before ever trusting
# `current_winner_realization_id`/`composite_realization_ids` -- see each
# status's own meaning below.
ENGINE_RESOLVED_WINNER = "RESOLVED_WINNER"          # exactly one realization is `state == "selected"`, no composite recorded
ENGINE_RESOLVED_COMPOSITE = "RESOLVED_COMPOSITE"    # a CompositeRecord names this idea's winning member set
ENGINE_REVIEW_REQUIRED = "REVIEW_REQUIRED"          # zero realizations are `state == "selected"` for this idea
ENGINE_BLOCKED_UNRESOLVED = "BLOCKED_UNRESOLVED"    # >1 realizations are `state == "selected"` with no composite explaining it (e.g. freeze_blocked keeping multiple candidates pending human review)


@dataclass(frozen=True)
class SemanticIdeaRecord:
    semantic_idea_id: str
    retry_family_ids: tuple[str, ...]
    realization_ids: tuple[str, ...]
    canonical_claim_ids: tuple[str, ...]
    current_winner_realization_id: str | None
    composite_realization_ids: tuple[str, ...]
    coverage_status: str                # "complete" | "missing" | "unresolved_ambiguous" | "unknown"
    story_order_position: int | None
    status: str = IDEA_STATUS_ACTIVE
    # D-050C1.6 F6/F7 (see the ENGINE_* constants above): defaults to
    # "" (UNRESOLVED / not yet finalized) so every pre-existing
    # construction site (D-050B's own tests) stays valid unchanged;
    # `build_semantic_ledger_shadow` always finalizes this before
    # returning.
    engine_resolution_status: str = ""


# Decision types (Section 5) -- a closed, named vocabulary so a forensic
# reader (or a future D-050C automation) never has to guess a stage's
# own ad hoc string.
DELIVERY_SCORE_WINNER = "DELIVERY_SCORE_WINNER"
SEMANTIC_WINNER_OVERRIDE = "SEMANTIC_WINNER_OVERRIDE"
CLAIM_COVERAGE_OVERRIDE = "CLAIM_COVERAGE_OVERRIDE"
COMPOSITE_CREATED = "COMPOSITE_CREATED"
CLIP_DISCARDED = "CLIP_DISCARDED"
REPLACEMENT_DECLARED = "REPLACEMENT_DECLARED"
DRAFT_REVIEW_REMOVED = "DRAFT_REVIEW_REMOVED"
SEMANTIC_GROUP_MERGED = "SEMANTIC_GROUP_MERGED"
SEMANTIC_GROUP_SPLIT = "SEMANTIC_GROUP_SPLIT"


@dataclass(frozen=True)
class DecisionRecord:
    order_index: int
    stage: str
    decision_type: str
    subject_realization_id: str | None
    semantic_idea_id: str | None
    previous_state: str | None
    new_state: str | None
    reason: str
    evidence: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscardRecord:
    discarded_realization_id: str
    discarding_stage: str
    reason: str
    replacement_realization_id: str | None
    replacement_verified: bool
    coverage_after_discard: str | None = None
    # D-073: `hybrid_editorial_chunks[*].decisions[*].replacement_candidate_
    # clip_id_before_guard` (D-072), carried through observation-only when
    # this discard's own hybrid decision recorded one -- None whenever no
    # such candidate exists or predates D-072. This is the ONLY additional
    # evidence realization_resolver.py's PATH B semantic replacement
    # certification needs to find a starting candidate for a true orphan
    # (one with no semantic_idea_id, so no formal grouping relation to
    # consult) -- the Ledger itself makes no certification decision from
    # it, exactly like every other field on this record.
    pre_guard_candidate_clip_id: str | None = None


@dataclass(frozen=True)
class CompositeRecord:
    semantic_idea_id: str | None
    member_realization_ids: tuple[str, ...]
    composite_kind: str                 # "delivery_repair" | "claim_coverage_composite"
    reason: str


@dataclass(frozen=True)
class CoverageRecord:
    semantic_idea_id: str
    coverage_status: str
    missing_claim_ids: tuple[str, ...]
    winner_realization_id: str | None


@dataclass(frozen=True)
class ProvenanceEdge:
    child: str
    parent: str
    edge_type: str  # "source_span_to_attempt" | "attempt_to_realization" | "realization_to_idea" | "realization_to_fragment"


# ---------------------------------------------------------------------------
# The Ledger itself
# ---------------------------------------------------------------------------

class SemanticLedger:
    """See module docstring: WRITE OWNERSHIP. Construct with `SemanticLedger()`
    and populate only through the methods below."""

    def __init__(self) -> None:
        self.__realizations: dict[str, RealizationRecord] = {}
        self.__ideas: dict[str, SemanticIdeaRecord] = {}
        self.__claims: dict[str, CanonicalClaimRecord] = {}
        self.__decisions: list[DecisionRecord] = []
        self.__discards: list[DiscardRecord] = []
        self.__composites: list[CompositeRecord] = []
        self.__coverage: list[CoverageRecord] = []
        self.__edges: list[ProvenanceEdge] = []
        self.__fragment_owner: dict[str, str] = {}   # render_fragment_id -> realization_id
        self.__order = 0

    # ---- narrow write API (Section 8) --------------------------------

    def register_realization(self, record: RealizationRecord) -> None:
        existing = self.__realizations.get(record.realization_id)
        if existing is not None and existing != record:
            raise LedgerIntegrityError(
                f"realization_id {record.realization_id!r} already registered with different data"
            )
        self.__realizations[record.realization_id] = record
        for span_id in record.source_span_ids:
            self.__edges.append(ProvenanceEdge(record.realization_id, span_id, "source_span_to_attempt"))
        if record.attempt_id:
            self.__edges.append(ProvenanceEdge(record.realization_id, record.attempt_id, "attempt_to_realization"))
        if record.semantic_idea_id:
            # Recorded here too (not only in register_semantic_idea) so a
            # realization referencing an idea id that never actually gets
            # registered is still detectable by find_unknown_parent_ids --
            # exactly the "unknown parent id" case Section 7 asks for.
            self.__edges.append(ProvenanceEdge(record.realization_id, record.semantic_idea_id, "realization_to_idea"))
        for fragment_id in record.render_fragment_ids:
            self.__fragment_owner.setdefault(fragment_id, record.realization_id)
            self.__edges.append(ProvenanceEdge(fragment_id, record.realization_id, "realization_to_fragment"))

    def register_semantic_idea(self, record: SemanticIdeaRecord) -> None:
        existing = self.__ideas.get(record.semantic_idea_id)
        if existing is not None and existing != record:
            raise LedgerIntegrityError(
                f"semantic_idea_id {record.semantic_idea_id!r} already registered with different data"
            )
        self.__ideas[record.semantic_idea_id] = record
        for realization_id in record.realization_ids:
            self.__edges.append(ProvenanceEdge(realization_id, record.semantic_idea_id, "realization_to_idea"))

    def assign_retry_family(self, semantic_idea_id: str, retry_family_id: str) -> None:
        idea = self.__ideas.get(semantic_idea_id)
        if idea is None:
            raise LedgerIntegrityError(f"cannot assign retry_family to unknown semantic_idea_id {semantic_idea_id!r}")
        if retry_family_id in idea.retry_family_ids:
            return
        self.__ideas[semantic_idea_id] = replace(idea, retry_family_ids=(*idea.retry_family_ids, retry_family_id))

    def register_claim(self, record: CanonicalClaimRecord) -> None:
        existing = self.__claims.get(record.canonical_claim_id)
        if existing is not None:
            if existing.claim_type != record.claim_type or existing.content_tokens != record.content_tokens:
                raise LedgerIntegrityError(
                    f"canonical_claim_id {record.canonical_claim_id!r} already registered with different content"
                )
            merged_sources = tuple(dict.fromkeys((*existing.source_realization_ids, *record.source_realization_ids)))
            merged_covered = tuple(dict.fromkeys((*existing.covered_by_realization_ids, *record.covered_by_realization_ids)))
            self.__claims[record.canonical_claim_id] = replace(
                existing, source_realization_ids=merged_sources, covered_by_realization_ids=merged_covered,
                coverage_state=record.coverage_state or existing.coverage_state,
                text=existing.text or record.text,
            )
            return
        self.__claims[record.canonical_claim_id] = record

    def record_winner_decision(
        self, *, semantic_idea_id: str | None, realization_id: str, stage: str, decision_type: str,
        reason: str, evidence: Mapping[str, object] | None = None, previous_realization_id: str | None = None,
    ) -> None:
        if decision_type not in (DELIVERY_SCORE_WINNER, SEMANTIC_WINNER_OVERRIDE, CLAIM_COVERAGE_OVERRIDE):
            raise LedgerIntegrityError(f"not a winner decision type: {decision_type!r}")
        self.__decisions.append(DecisionRecord(
            order_index=self.__next_order(), stage=stage, decision_type=decision_type,
            subject_realization_id=realization_id, semantic_idea_id=semantic_idea_id,
            previous_state=previous_realization_id, new_state=realization_id,
            reason=reason, evidence=dict(evidence or {}),
        ))
        if semantic_idea_id is not None and semantic_idea_id in self.__ideas:
            idea = self.__ideas[semantic_idea_id]
            self.__ideas[semantic_idea_id] = replace(idea, current_winner_realization_id=realization_id)

    def record_discard(self, record: DiscardRecord, *, stage: str, semantic_idea_id: str | None = None) -> None:
        self.__discards.append(record)
        self.__decisions.append(DecisionRecord(
            order_index=self.__next_order(), stage=stage, decision_type=CLIP_DISCARDED,
            subject_realization_id=record.discarded_realization_id, semantic_idea_id=semantic_idea_id,
            previous_state="active", new_state="discarded", reason=record.reason,
            evidence={
                "replacement_realization_id": record.replacement_realization_id,
                "replacement_verified": record.replacement_verified,
            },
        ))
        if record.replacement_realization_id:
            self.__decisions.append(DecisionRecord(
                order_index=self.__next_order(), stage=stage, decision_type=REPLACEMENT_DECLARED,
                subject_realization_id=record.replacement_realization_id, semantic_idea_id=semantic_idea_id,
                previous_state=None, new_state=record.discarded_realization_id, reason=record.reason,
                evidence={"replacement_verified": record.replacement_verified},
            ))

    def record_composite(self, record: CompositeRecord, *, stage: str) -> None:
        self.__composites.append(record)
        self.__decisions.append(DecisionRecord(
            order_index=self.__next_order(), stage=stage, decision_type=COMPOSITE_CREATED,
            subject_realization_id=None, semantic_idea_id=record.semantic_idea_id,
            previous_state=None, new_state=",".join(record.member_realization_ids),
            reason=record.reason, evidence={"composite_kind": record.composite_kind},
        ))

    def record_coverage(self, record: CoverageRecord) -> None:
        self.__coverage.append(record)
        if record.semantic_idea_id in self.__ideas:
            idea = self.__ideas[record.semantic_idea_id]
            self.__ideas[record.semantic_idea_id] = replace(idea, coverage_status=record.coverage_status)

    def finalize_idea_engine_resolution(
        self, semantic_idea_id: str, *, status: str,
        winner_realization_id: str | None, composite_realization_ids: tuple[str, ...],
    ) -> None:
        """D-050C1.6 F6/F7: the ONE place `engine_resolution_status`/
        `current_winner_realization_id`/`composite_realization_ids` are
        ever written after an idea's initial registration -- always called
        LAST, after every stage `build_semantic_ledger_shadow` observes
        has already run, with values computed straight from ground-truth
        realization `state` + recorded composites (see
        `_finalize_engine_resolution`), never from decision-event order.
        Idempotent when called twice with identical values; safe to call
        even if the idea record doesn't exist (a no-op) so callers never
        need their own existence check."""
        idea = self.__ideas.get(semantic_idea_id)
        if idea is None:
            return
        self.__ideas[semantic_idea_id] = replace(
            idea, engine_resolution_status=status,
            current_winner_realization_id=winner_realization_id,
            composite_realization_ids=composite_realization_ids,
        )

    def record_physical_fragment(self, *, realization_id: str, render_fragment_id: str) -> None:
        if realization_id not in self.__realizations:
            raise LedgerIntegrityError(
                f"cannot attach fragment {render_fragment_id!r} to unknown realization_id {realization_id!r}"
            )
        record = self.__realizations[realization_id]
        if render_fragment_id in record.render_fragment_ids:
            return
        self.__realizations[realization_id] = replace(
            record, render_fragment_ids=(*record.render_fragment_ids, render_fragment_id),
        )
        self.__fragment_owner[render_fragment_id] = realization_id
        self.__edges.append(ProvenanceEdge(render_fragment_id, realization_id, "realization_to_fragment"))

    def record_decision(
        self, *, stage: str, decision_type: str, reason: str,
        subject_realization_id: str | None = None, semantic_idea_id: str | None = None,
        previous_state: str | None = None, new_state: str | None = None,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        """General-purpose decision-history write for decision types with
        no dedicated method (SEMANTIC_GROUP_MERGED/SEMANTIC_GROUP_SPLIT/
        DRAFT_REVIEW_REMOVED)."""
        self.__decisions.append(DecisionRecord(
            order_index=self.__next_order(), stage=stage, decision_type=decision_type,
            subject_realization_id=subject_realization_id, semantic_idea_id=semantic_idea_id,
            previous_state=previous_state, new_state=new_state, reason=reason, evidence=dict(evidence or {}),
        ))

    def __next_order(self) -> int:
        value = self.__order
        self.__order += 1
        return value

    # ---- read-only views -----------------------------------------------

    def realizations(self) -> Mapping[str, RealizationRecord]:
        return types.MappingProxyType(self.__realizations)

    def ideas(self) -> Mapping[str, SemanticIdeaRecord]:
        return types.MappingProxyType(self.__ideas)

    def claims(self) -> Mapping[str, CanonicalClaimRecord]:
        return types.MappingProxyType(self.__claims)

    def decisions(self) -> tuple[DecisionRecord, ...]:
        return tuple(sorted(self.__decisions, key=lambda d: d.order_index))

    def discards(self) -> tuple[DiscardRecord, ...]:
        return tuple(self.__discards)

    def composites(self) -> tuple[CompositeRecord, ...]:
        return tuple(self.__composites)

    def coverage(self) -> tuple[CoverageRecord, ...]:
        return tuple(self.__coverage)

    def provenance_edges(self) -> tuple[ProvenanceEdge, ...]:
        return tuple(self.__edges)

    def decision_history_for(self, realization_id: str) -> tuple[DecisionRecord, ...]:
        """Answers the directive's own forensic question -- "why is
        realization X no longer selected?" -- in one call, without
        reconstructing any diagnostics dict."""
        return tuple(
            d for d in self.decisions()
            if d.subject_realization_id == realization_id
            or (d.decision_type == REPLACEMENT_DECLARED and d.new_state == realization_id)
        )

    # ---- structural validation (Section 7 + 10): reports, never raises,
    # never mutates, never changes engine behavior ------------------------

    def find_orphan_realizations(self) -> list[str]:
        """A realization with no semantic_idea_id is only a real orphan if
        nothing explains its absence -- a DiscardRecord naming it is a
        legitimate explanation (it never reached grouping because an
        earlier stage removed it first, e.g. D-049 Case A)."""
        discarded_ids = {d.discarded_realization_id for d in self.__discards}
        return [
            realization_id for realization_id, record in self.__realizations.items()
            if record.semantic_idea_id is None and realization_id not in discarded_ids
        ]

    def find_unknown_parent_ids(self) -> list[str]:
        known_realizations = set(self.__realizations)
        known_ideas = set(self.__ideas)
        bad = []
        for edge in self.__edges:
            if edge.edge_type == "realization_to_idea" and edge.parent not in known_ideas:
                bad.append(edge.parent)
            elif edge.edge_type == "realization_to_fragment" and edge.parent not in known_realizations:
                bad.append(edge.parent)
        return sorted(set(bad))

    def find_duplicate_semantic_ids(self) -> list[str]:
        """Always empty by construction: `register_semantic_idea` raises
        `LedgerIntegrityError` the instant a second write disagrees with
        an existing one, and a Python dict key is unique by definition --
        there is no code path that leaves two conflicting
        `SemanticIdeaRecord`s stored under one id. Kept as an explicit,
        named check (rather than silently absent) so a future change to
        the write path that weakens this guarantee has a test to break."""
        return []

    def find_fragments_without_parent_realization(self) -> list[str]:
        return [
            fragment_id for fragment_id, owner in self.__fragment_owner.items()
            if owner not in self.__realizations
        ]

    def find_provenance_cycles(self) -> list[tuple[str, ...]]:
        graph: dict[str, set[str]] = {}
        for edge in self.__edges:
            graph.setdefault(edge.child, set()).add(edge.parent)
        cycles: list[tuple[str, ...]] = []
        visiting: set[str] = set()
        visited: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> None:
            if node in visiting:
                cycle_start = path.index(node)
                cycles.append(tuple(path[cycle_start:] + [node]))
                return
            if node in visited:
                return
            visiting.add(node)
            path.append(node)
            for parent in graph.get(node, ()):
                dfs(parent)
            path.pop()
            visiting.discard(node)
            visited.add(node)

        for node in list(graph):
            dfs(node)
        return cycles


# ---------------------------------------------------------------------------
# Shadow reconstruction driver (Section 9)
# ---------------------------------------------------------------------------

def _clip_realization_id(clip) -> str:
    return str(getattr(clip, "realization_id", None) or clip.clip_id)


def _all_clips(draft) -> list:
    return [*draft.selected, *draft.alternates, *draft.discarded]


def _extract_claims_for_clip(clip) -> tuple[Claim, ...]:
    try:
        return extract_claims(clip.clip_id, str(clip.text or ""))
    except Exception:  # pragma: no cover -- defensive: never let shadow observation break the run
        return ()


def build_semantic_ledger_shadow(draft) -> SemanticLedger:
    """Reconstructs a `SemanticLedger` from a fully-built `DraftTimeline`'s
    existing diagnostics -- read-only, called exactly once, after every
    stage it observes has already run (see the intended call site in
    universal_clean_cut.py, right after Final Story Coherence
    Validation/CanonicalEditPlan/FinalEditReviewer, before Freeze).

    WHY RECONSTRUCTION RATHER THAN LIVE HOOKS: every decision this
    function records (grouping membership, DeliveryScorer's provisional
    winner, the semantic best-take override, ClaimCoverage's override/
    composite/suppression, draft-review removal, hybrid_editorial_chunks'
    delete-with-or-without-replacement, CanonicalEditPlan/StoryValidator
    coverage) is ALREADY present, verbatim, in `draft.diagnostics` by the
    time this runs -- every one of those stages already writes its own
    authoritative diagnostics key today. Injecting eight separate live
    hooks into eight separate stages to observe the identical information
    a single read-only pass over the finished diagnostics dict already
    has would only add eight new places editorial behavior could
    accidentally change, for zero additional fidelity. This is the
    lower-risk implementation of "shadow integration" the D-050B
    directive asks for, not a shortcut around it.

    KNOWN BEST-EFFORT RECONSTRUCTIONS (reported, not hidden -- see
    `build_ledger_parity_report`):
    - A discarded clip's `semantic_idea_id` is not stamped by pipeline.py
      today (by design -- see D-050's own architecture audit). When the
      clip is a member of a `take_group_members` entry alongside at least
      one clip that DOES carry a stamped `semantic_idea_id` (selected or
      alternate), that id is borrowed. A clip discarded before ever
      reaching grouping (e.g. a hybrid_editorial_chunks delete -- exactly
      D-049 Case A's shape) legitimately has no `semantic_idea_id` at
      all; its `DiscardRecord` is what explains the absence (see
      `find_orphan_realizations`).
    - Per-realization claims are extracted fresh from each realization's
      own current text via `extract_claims` (the same function
      claim_coverage_best_take.py itself calls) rather than re-reading
      that module's internal, not-externally-diagnosed `Claim` objects.
    - `attempt_id`/`source_span_ids` on a `RealizationRecord` are only as
      complete as D-050A's own stamping reached that clip; a clip that
      predates D-050A stamping (or whose CandidateTake never carried the
      field through every transform) reports `None`/`()`, never a guess.
    """
    ledger = SemanticLedger()
    diagnostics = dict(draft.diagnostics or {})

    all_clips = _all_clips(draft)
    clip_by_id = {clip.clip_id: clip for clip in all_clips}

    # --- Section 3: realizations, deduped by realization_id -------------
    realization_clips: dict[str, list] = {}
    for clip in all_clips:
        realization_clips.setdefault(_clip_realization_id(clip), []).append(clip)

    hybrid_chunks = diagnostics.get("hybrid_editorial_chunks") or ()
    hybrid_delete_by_clip: dict[str, dict] = {}
    for chunk in hybrid_chunks:
        for decision in (chunk.get("decisions") or ()):
            clip_id = decision.get("clip_id")
            if clip_id and decision.get("applied_delete"):
                hybrid_delete_by_clip[clip_id] = decision

    selected_ids = {clip.clip_id for clip in draft.selected}
    draft_review_removed_ids = set(diagnostics.get("draft_review_removed_ids") or ())

    for realization_id, clips in realization_clips.items():
        primary = clips[0]
        state = "selected" if any(c.clip_id in selected_ids for c in clips) else (
            "alternate" if any(c in draft.alternates for c in clips) else "discarded"
        )
        discard_reason = None
        replacement_id = None
        pre_guard_candidate_clip_id = None
        if state == "discarded":
            hybrid_decision = next((hybrid_delete_by_clip.get(c.clip_id) for c in clips if c.clip_id in hybrid_delete_by_clip), None)
            if hybrid_decision:
                discard_reason = str(hybrid_decision.get("delete_basis") or "hybrid_editorial_delete")
                later_replacement_clip = hybrid_decision.get("later_retry_replacement_id")
                if later_replacement_clip and later_replacement_clip in clip_by_id:
                    replacement_id = _clip_realization_id(clip_by_id[later_replacement_clip])
                # D-073: carried through unconditionally (even when a
                # lexical replacement_id was already found above) -- an
                # observation, never a decision; PATH B only ever consults
                # it when the lexical path above left replacement_id None.
                pre_guard_candidate_clip_id = hybrid_decision.get(
                    "replacement_candidate_clip_id_before_guard"
                )
            elif any(c.clip_id in draft_review_removed_ids for c in clips):
                discard_reason = "draft_review_removed"
            else:
                discard_reason = "clean_cut_or_composite_resolution"

        claims = []
        for clip in clips:
            claims.extend(_extract_claims_for_clip(clip))
        source_span_ids = tuple(dict.fromkeys(
            str(getattr(c, "source_span_id", None)) for c in clips if getattr(c, "source_span_id", None)
        ))
        fragment_ids = tuple(dict.fromkeys(
            str(getattr(c, "render_fragment_id", None)) for c in clips if getattr(c, "render_fragment_id", None)
        ))

        ledger.register_realization(RealizationRecord(
            realization_id=realization_id,
            semantic_idea_id=str(getattr(primary, "semantic_idea_id", None)) if getattr(primary, "semantic_idea_id", None) else None,
            retry_family_id=str(getattr(primary, "retry_family_id", None)) if getattr(primary, "retry_family_id", None) else None,
            source_span_ids=source_span_ids,
            attempt_id=str(getattr(primary, "attempt_id", None)) if getattr(primary, "attempt_id", None) else None,
            clip_ids=tuple(dict.fromkeys(c.clip_id for c in clips)),
            text=str(primary.text or ""),
            start=float(primary.start), end=float(primary.end),
            delivery_score=None,
            state=state,
            discard_reason=discard_reason,
            replacement_realization_id=replacement_id,
            claim_ids=tuple(claim.canonical_claim_id for claim in claims),
            render_fragment_ids=fragment_ids,
            complete_idea=getattr(primary, "complete_idea", None),
        ))
        for claim in claims:
            ledger.register_claim(CanonicalClaimRecord(
                canonical_claim_id=claim.canonical_claim_id or mint_canonical_claim_id(claim.claim_type, claim.content_tokens),
                claim_type=claim.claim_type,
                content_tokens=claim.content_tokens,
                importance=claim.importance,
                source_realization_ids=(realization_id,),
                covered_by_realization_ids=(),
                coverage_state="unresolved",
                text=str(claim.text or ""),
                negation_role=str(getattr(claim, "negation_role", "") or ""),
            ))

        if discard_reason is not None:
            ledger.record_discard(
                DiscardRecord(
                    discarded_realization_id=realization_id,
                    discarding_stage="hybrid_editorial_chunks" if discard_reason not in ("draft_review_removed", "clean_cut_or_composite_resolution") else discard_reason,
                    reason=discard_reason,
                    replacement_realization_id=replacement_id,
                    replacement_verified=bool(replacement_id),
                    pre_guard_candidate_clip_id=pre_guard_candidate_clip_id,
                ),
                stage="hybrid_editorial_chunks" if discard_reason not in ("draft_review_removed", "clean_cut_or_composite_resolution") else discard_reason,
                semantic_idea_id=str(getattr(primary, "semantic_idea_id", None)) if getattr(primary, "semantic_idea_id", None) else None,
            )
            if discard_reason == "draft_review_removed":
                ledger.record_decision(
                    stage="draft_review", decision_type=DRAFT_REVIEW_REMOVED, reason="draft_review_removed",
                    subject_realization_id=realization_id,
                )

    # --- borrow semantic_idea_id for discarded group-mates that lack one
    # (see docstring: KNOWN BEST-EFFORT RECONSTRUCTIONS, first bullet) ----
    take_group_members = diagnostics.get("take_group_members") or ()
    for member_clip_ids in take_group_members:
        member_realizations = [
            _clip_realization_id(clip_by_id[cid]) for cid in member_clip_ids if cid in clip_by_id
        ]
        stamped_idea_ids = {
            ledger.realizations()[rid].semantic_idea_id
            for rid in member_realizations
            if rid in ledger.realizations() and ledger.realizations()[rid].semantic_idea_id
        }
        if len(stamped_idea_ids) != 1:
            continue
        (idea_id,) = tuple(stamped_idea_ids)
        for rid in member_realizations:
            record = ledger.realizations().get(rid)
            if record is not None and record.semantic_idea_id is None:
                ledger.register_realization(replace(record, semantic_idea_id=idea_id))

    # --- Section 2: semantic ideas ---------------------------------------
    # NOTE: coverage_status below is derived independently from the
    # Ledger's OWN winner-presence signal, deliberately NOT copied
    # verbatim from CanonicalEditPlan's own diagnostics -- copying would
    # make `build_ledger_parity_report`'s CanonicalEditPlan comparison
    # structurally unable to ever disagree, defeating the whole point of
    # an independent parity check (Section 10). A subsequent
    # `record_coverage` call (from claim_coverage_best_take's
    # unresolved_gaps or StoryValidator's missing_idea_coverage, both
    # below) can still refine this initial value.
    idea_members: dict[str, list[str]] = {}
    for realization_id, record in ledger.realizations().items():
        if record.semantic_idea_id:
            idea_members.setdefault(record.semantic_idea_id, []).append(realization_id)

    for idea_id, realization_ids in idea_members.items():
        winner_id = next(
            (rid for rid in realization_ids if ledger.realizations()[rid].state == "selected"), None,
        )
        claim_ids = tuple(dict.fromkeys(
            cid for rid in realization_ids for cid in ledger.realizations()[rid].claim_ids
        ))
        retry_family_ids = tuple(dict.fromkeys(
            ledger.realizations()[rid].retry_family_id for rid in realization_ids
            if ledger.realizations()[rid].retry_family_id
        ))
        ledger.register_semantic_idea(SemanticIdeaRecord(
            semantic_idea_id=idea_id,
            retry_family_ids=retry_family_ids,
            realization_ids=tuple(realization_ids),
            canonical_claim_ids=claim_ids,
            current_winner_realization_id=winner_id,
            composite_realization_ids=(),
            coverage_status="complete" if winner_id else "unresolved_ambiguous",
            story_order_position=None,
        ))
        if winner_id:
            ledger.record_winner_decision(
                semantic_idea_id=idea_id, realization_id=winner_id, stage="deterministic_best_take_authority",
                decision_type=DELIVERY_SCORE_WINNER, reason="best_take_authority_winner",
            )

    # --- Section 5/9: DeliveryScorer + semantic-best-take override history
    for group in (diagnostics.get("take_judge_groups") or ()):
        ranked = group.get("ranked") or ()
        if not ranked:
            continue
        local_winner_clip = group.get("local_selected_clip_id")
        semantic_winner_clip = group.get("selected_clip_id")
        idea_id = None
        if local_winner_clip in clip_by_id:
            realization_id = _clip_realization_id(clip_by_id[local_winner_clip])
            record = ledger.realizations().get(realization_id)
            idea_id = record.semantic_idea_id if record else None
            ledger.record_winner_decision(
                semantic_idea_id=idea_id, realization_id=realization_id, stage="take_judge_provider",
                decision_type=DELIVERY_SCORE_WINNER, reason="watch_listen_baseline_top_score",
                evidence={"ranked": list(ranked)},
            )
        if group.get("semantic_override_applied") and semantic_winner_clip in clip_by_id:
            override_realization_id = _clip_realization_id(clip_by_id[semantic_winner_clip])
            previous_realization_id = (
                _clip_realization_id(clip_by_id[local_winner_clip]) if local_winner_clip in clip_by_id else None
            )
            # D-058 Phase 2: carry the arbiter's own recorded confidence for
            # the winning clip through to the decision's evidence -- the
            # Resolver's own evidence hierarchy (realization_resolver.py's
            # `_pick_winner`) needs this to tell a HIGH-confidence semantic
            # winner from a merely-applied one; before this it was recorded
            # with no confidence at all, so a downstream reader could not
            # distinguish the two without re-deriving it from
            # `semantic_candidates` itself.
            override_confidence = next(
                (
                    float(candidate.get("confidence") or 0.0)
                    for candidate in (group.get("semantic_candidates") or ())
                    if candidate.get("clip_id") == semantic_winner_clip
                ),
                0.0,
            )
            ledger.record_winner_decision(
                semantic_idea_id=idea_id, realization_id=override_realization_id, stage="pipeline_semantic_best_take",
                decision_type=SEMANTIC_WINNER_OVERRIDE, reason="hybrid_semantic_label_override",
                evidence={"confidence": round(override_confidence, 4)},
                previous_realization_id=previous_realization_id,
            )

    # --- Section 5/9: ClaimCoverage overrides/composites/suppressions ----
    claim_coverage_diag = diagnostics.get("claim_coverage_best_take") or {}
    for override in (claim_coverage_diag.get("overrides") or ()):
        new_winner_clip = override.get("new_winner_clip_id")
        if new_winner_clip in clip_by_id:
            realization_id = _clip_realization_id(clip_by_id[new_winner_clip])
            record = ledger.realizations().get(realization_id)
            idea_id = record.semantic_idea_id if record else None
            ledger.record_winner_decision(
                semantic_idea_id=idea_id, realization_id=realization_id, stage="claim_coverage_best_take",
                decision_type=CLAIM_COVERAGE_OVERRIDE, reason=str(override.get("reason") or "claim_coverage_override"),
            )
    for composite in (claim_coverage_diag.get("composites") or ()):
        # D-050C1.6 F6: `claim_coverage_best_take.apply_claim_coverage_
        # best_take` writes each composite's member list under "clip_ids"
        # (see that module's own composites.append(...) call) -- this
        # used to read a "member_clip_ids" key that key never existed
        # under, silently reconstructing zero composites from every real
        # ClaimCoverage composite ever formed. Reading "clip_ids" (falling
        # back to the never-actually-used "member_clip_ids" only for any
        # hand-built test fixture that happened to use that name) is what
        # makes `finalize_idea_engine_resolution`'s ENGINE_RESOLVED_
        # COMPOSITE detection actually fire on real production shapes.
        member_ids = tuple(
            _clip_realization_id(clip_by_id[cid])
            for cid in (composite.get("clip_ids") or composite.get("member_clip_ids") or ())
            if cid in clip_by_id
        )
        if member_ids:
            first_idea = ledger.realizations().get(member_ids[0])
            ledger.record_composite(
                CompositeRecord(
                    semantic_idea_id=first_idea.semantic_idea_id if first_idea else None,
                    member_realization_ids=member_ids, composite_kind="claim_coverage_composite",
                    reason=str(composite.get("group_id") or "claim_coverage_composite"),
                ),
                stage="claim_coverage_best_take",
            )
    for suppressed in (claim_coverage_diag.get("suppressed_incidental_overrides") or ()):
        suppressed_clip = suppressed.get("suppressed_new_winner_clip_id")
        if suppressed_clip in clip_by_id:
            realization_id = _clip_realization_id(clip_by_id[suppressed_clip])
            record = ledger.realizations().get(realization_id)
            ledger.record_decision(
                stage="claim_coverage_best_take", decision_type="CLAIM_COVERAGE_OVERRIDE_SUPPRESSED",
                subject_realization_id=realization_id,
                semantic_idea_id=record.semantic_idea_id if record else None,
                reason="incidental_self_source_claim_suppressed",
            )
    for gap in (claim_coverage_diag.get("unresolved_gaps") or ()):
        winner_clip = gap.get("winner_clip_id")
        idea_id = None
        winner_realization_id = None
        if winner_clip in clip_by_id:
            winner_realization_id = _clip_realization_id(clip_by_id[winner_clip])
            record = ledger.realizations().get(winner_realization_id)
            idea_id = record.semantic_idea_id if record else None
        missing_ids = tuple(gap.get("missing_claim_ids") or ())
        if idea_id:
            ledger.record_coverage(CoverageRecord(
                semantic_idea_id=idea_id, coverage_status="unresolved_ambiguous",
                missing_claim_ids=missing_ids, winner_realization_id=winner_realization_id,
            ))

    # --- Section 11: StoryValidator / CanonicalEditPlan coverage ---------
    coherence_diag = diagnostics.get("final_story_coherence_validation") or {}
    for idea_id in (coherence_diag.get("missing_idea_coverage") or ()):
        if idea_id in ledger.ideas():
            ledger.record_coverage(CoverageRecord(
                semantic_idea_id=idea_id, coverage_status="missing",
                missing_claim_ids=(), winner_realization_id=ledger.ideas()[idea_id].current_winner_realization_id,
            ))

    # --- D-050C1.6 F6/F7: finalize each idea's ENGINE resolution shape
    # from ground-truth realization state + recorded composites -- always
    # last, always overriding whatever `current_winner_realization_id`
    # happened to be left at by the decision-event reconstruction above.
    # See `finalize_idea_engine_resolution`'s own docstring and the
    # ENGINE_* constants for exactly what each status means.
    composites_by_idea: dict[str, tuple[str, ...]] = {}
    for composite in ledger.composites():
        if composite.semantic_idea_id:
            composites_by_idea.setdefault(composite.semantic_idea_id, composite.member_realization_ids)
    for idea_id, idea in ledger.ideas().items():
        composite_members = composites_by_idea.get(idea_id, ())
        selected_ids = tuple(
            rid for rid in idea.realization_ids
            if rid in ledger.realizations() and ledger.realizations()[rid].state == "selected"
        )
        if composite_members:
            status, winner = ENGINE_RESOLVED_COMPOSITE, None
        elif len(selected_ids) == 1:
            status, winner, composite_members = ENGINE_RESOLVED_WINNER, selected_ids[0], ()
        elif len(selected_ids) == 0:
            status, winner = ENGINE_REVIEW_REQUIRED, None
        else:
            status, winner = ENGINE_BLOCKED_UNRESOLVED, None
        ledger.finalize_idea_engine_resolution(
            idea_id, status=status, winner_realization_id=winner, composite_realization_ids=composite_members,
        )

    return ledger


# ---------------------------------------------------------------------------
# Parity checker (Section 10): compares the Ledger's reconstructed view
# against today's authoritative outputs. Never fails production behavior --
# only ever reports.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LedgerMismatch:
    kind: str
    detail: str


@dataclass(frozen=True)
class LedgerParityReport:
    mismatches: tuple[LedgerMismatch, ...]

    @property
    def is_clean(self) -> bool:
        return not self.mismatches


def build_ledger_parity_report(ledger: SemanticLedger, draft) -> LedgerParityReport:
    mismatches: list[LedgerMismatch] = []
    diagnostics = dict(draft.diagnostics or {})

    for clip in draft.selected:
        rid = _clip_realization_id(clip)
        record = ledger.realizations().get(rid)
        if record is None:
            mismatches.append(LedgerMismatch("missing_realization", f"selected clip {clip.clip_id} has no ledger realization"))
        elif record.state != "selected":
            mismatches.append(LedgerMismatch("state_mismatch", f"selected clip {clip.clip_id} ledger state={record.state!r}"))
        fragment_id = getattr(clip, "render_fragment_id", None)
        if fragment_id and record is not None and fragment_id not in record.render_fragment_ids:
            mismatches.append(LedgerMismatch(
                "fragment_provenance_mismatch",
                f"clip {clip.clip_id} render_fragment_id {fragment_id!r} not attached to realization {rid!r}",
            ))

    for clip in draft.discarded:
        rid = _clip_realization_id(clip)
        record = ledger.realizations().get(rid)
        if record is None:
            mismatches.append(LedgerMismatch("missing_realization", f"discarded clip {clip.clip_id} has no ledger realization"))
        elif record.state == "selected":
            mismatches.append(LedgerMismatch("state_mismatch", f"discarded clip {clip.clip_id} ledger state={record.state!r}"))

    canonical_edit_plan = diagnostics.get("canonical_edit_plan") or {}
    for idea in (canonical_edit_plan.get("ideas") or ()):
        idea_id = idea.get("idea_id")
        expected_status = idea.get("coverage_status")
        ledger_idea = ledger.ideas().get(idea_id)
        if ledger_idea is None:
            mismatches.append(LedgerMismatch("missing_idea", f"CanonicalEditPlan idea {idea_id!r} absent from ledger"))
        elif ledger_idea.coverage_status != expected_status:
            mismatches.append(LedgerMismatch(
                "coverage_mismatch",
                f"idea {idea_id!r}: ledger coverage_status={ledger_idea.coverage_status!r}, CanonicalEditPlan={expected_status!r}",
            ))

    coherence = diagnostics.get("final_story_coherence_validation") or {}
    for idea_id in (coherence.get("missing_idea_coverage") or ()):
        ledger_idea = ledger.ideas().get(idea_id)
        if ledger_idea is None:
            mismatches.append(LedgerMismatch("missing_idea", f"StoryValidator-flagged idea {idea_id!r} absent from ledger"))
        elif ledger_idea.coverage_status != "missing":
            mismatches.append(LedgerMismatch(
                "coverage_mismatch",
                f"idea {idea_id!r}: StoryValidator says missing, ledger says {ledger_idea.coverage_status!r}",
            ))

    return LedgerParityReport(tuple(mismatches))


def build_semantic_ledger_diagnostics(ledger: SemanticLedger, parity: LedgerParityReport | None = None) -> dict:
    """A JSON-safe diagnostics view of a Ledger, for wiring into
    `draft.diagnostics["semantic_ledger"]` -- read-only, additive. Pass
    `parity` (from `build_ledger_parity_report`) to include it inline;
    omitted, `parity` reports as `null` rather than being silently
    absent."""
    return {
        "schema_version": "cutsell.semantic_ledger.v1",
        "realization_count": len(ledger.realizations()),
        "idea_count": len(ledger.ideas()),
        "claim_count": len(ledger.claims()),
        "decision_count": len(ledger.decisions()),
        "discard_count": len(ledger.discards()),
        "composite_count": len(ledger.composites()),
        "orphan_realizations": ledger.find_orphan_realizations(),
        "unknown_parent_ids": ledger.find_unknown_parent_ids(),
        "fragments_without_parent_realization": ledger.find_fragments_without_parent_realization(),
        "provenance_cycles": [list(cycle) for cycle in ledger.find_provenance_cycles()],
        "discards": [
            {
                "discarded_realization_id": d.discarded_realization_id,
                "discarding_stage": d.discarding_stage,
                "reason": d.reason,
                "replacement_realization_id": d.replacement_realization_id,
                "replacement_verified": d.replacement_verified,
                "pre_guard_candidate_clip_id": d.pre_guard_candidate_clip_id,
            }
            for d in ledger.discards()
        ],
        "parity": None if parity is None else {
            "is_clean": parity.is_clean,
            "mismatches": [{"kind": m.kind, "detail": m.detail} for m in parity.mismatches],
        },
    }
