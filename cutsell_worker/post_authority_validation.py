"""D-090 -- POST-RESOLVER STORYVALIDATOR IMMUTABILITY: the executable
authority boundary around the post-authoritative validation pass.

## Why this module exists (docs/CUTSELL_DECISIONS.md D-089 canary / D-090)

Run 33960713625 (engine 40dde20): the Unified Realization Resolver emitted a
two-member RESOLVED_COMPOSITE for one retry family, the AUTHORITATIVE
application put both members into `draft.selected`, and then StoryValidator's
residual-family resolution -- the SAME legacy code that is the right thing to
run BEFORE the resolver exists -- asked the semantic-equivalence arbiter
"same idea?", got "yes", and discarded the lower-ranked member. The
CanonicalEditPlan (D-087) correctly refused the mutated result
(`realization_not_selected`), so nothing wrong was rendered, but Freeze was
blocked by a defect StoryValidator itself introduced AFTER the one semantic
authority had already ruled.

That is a post-resolver semantic membership mutation: two authorities
disagreeing on the membership of the same family. The AUTHORITATIVE contract
(D-050C2 Section 3) is that `apply_authoritative_realization_resolution` is
THE ONE point the resolver's decision is applied and that "no other module
below this line mutates selection membership on the resolver's behalf".
StoryValidator's second pass was violating that contract by construction.

## The boundary

  1. `PostAuthorityValidationContext` -- an explicit, typed contract the
     post-resolver StoryValidator invocation MUST carry. It is built only
     from the very same `AuthoritativeApplicationResult` + `Authoritative
     PlanSource` (D-087) every other authoritative stage consumes. Its
     presence -- not a diagnostics key, not "two selected clips", not a
     composite label -- is what puts StoryValidator into VALIDATION-ONLY
     mode (`final_story_coherence_validation.apply_post_authority_story_
     validation`). A missing/invalid context is an integrity failure that
     fails closed; it NEVER falls back to the legacy resolving pass.

  2. `semantic_selection_signature` -- the ordered semantic selection
     projection (clip identity, realization/fragment provenance, semantic
     idea, source span, spoken content) captured right after authoritative
     application. `compare_selection_signatures` is the executable
     invariant: after post-authority validation the ORDER-SENSITIVE
     signature must be unchanged; after the bounded repair loop (whose one
     permitted repair is a story-order reorder) the ORDER-INSENSITIVE
     projection must be unchanged. Any drift is a named integrity failure
     (`POST_AUTHORITY_SELECTION_MUTATION`) reported by the caller, which
     must fail closed WITHOUT silently restoring the original selection and
     WITHOUT rebuilding the authoritative source from the mutated draft.

This module creates no selection authority: it only names the one that
already exists (the resolver, via its D-087 plan source) and checks that
nothing after it re-decides membership.
"""
from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from typing import Mapping

from .canonical_edit_plan import AuthoritativePlanSource

POST_AUTHORITY_VALIDATION_MODE = "post_authority_validation_only"
LEGACY_RESOLVING_MODE = "legacy_resolving"

INTEGRITY_FAILURE_MISSING_CONTEXT = "POST_AUTHORITY_CONTEXT_MISSING"
INTEGRITY_FAILURE_INVALID_CONTEXT = "POST_AUTHORITY_CONTEXT_INVALID"
INTEGRITY_FAILURE_SELECTION_MUTATION = "POST_AUTHORITY_SELECTION_MUTATION"

MUTATION_INVARIANT_PASS = "PASS"
MUTATION_INVARIANT_FAIL = "FAIL"
MUTATION_INVARIANT_NOT_EVALUATED = "NOT_EVALUATED"

PHASE_STORY_VALIDATION = "post_authority_story_validation"
PHASE_BOUNDED_REPAIR = "post_authority_bounded_repair"

_VALID_AUTHORITATIVE_STATUSES = frozenset({"SEMANTICALLY_RESOLVED", "REVIEW_REQUIRED"})


class PostAuthorityIntegrityError(ValueError):
    """Raised by `build_post_authority_validation_context` when the
    authoritative state the post-resolver pass requires is missing or
    malformed. The caller reports it as a named integrity failure and fails
    closed -- it must never be swallowed into a legacy resolving pass."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# Explicit post-authority validation context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PostAuthorityValidationContext:
    """The typed contract that puts StoryValidator into validation-only
    mode. `plan_source` IS the D-087 `AuthoritativePlanSource` (never a
    copy rebuilt from a draft); `source_identity` is its stable digest so
    the diagnostics can prove which authoritative verdict the pass was
    validating against."""
    authoritative_status: str
    plan_source: AuthoritativePlanSource
    source_identity: str
    decision_count: int
    unresolved_orphan_count: int = 0


def authoritative_source_identity(source: AuthoritativePlanSource) -> str:
    rows = []
    for idea_id in sorted(source.decisions):
        decision = source.decisions[idea_id]
        rows.append("|".join((
            str(decision.semantic_idea_id),
            str(decision.decision_status),
            str(decision.winner_realization_id or ""),
            ",".join(decision.composite_realization_ids),
            ",".join(sorted(decision.candidate_realization_ids)),
        )))
    payload = str(source.status) + "\x1e" + "\x1f".join(rows)
    return "authsrc_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_post_authority_validation_context(
    authoritative_result, plan_source: AuthoritativePlanSource | None,
) -> PostAuthorityValidationContext:
    """Build the context from the ONE authoritative application result and
    its D-087 plan source. Raises `PostAuthorityIntegrityError` (fail
    closed) when either is missing or structurally inconsistent."""
    if authoritative_result is None:
        raise PostAuthorityIntegrityError(
            INTEGRITY_FAILURE_MISSING_CONTEXT, "authoritative application result is absent",
        )
    if plan_source is None:
        raise PostAuthorityIntegrityError(
            INTEGRITY_FAILURE_MISSING_CONTEXT, "authoritative plan source is absent",
        )
    status = str(getattr(authoritative_result, "status", "") or "")
    if status not in _VALID_AUTHORITATIVE_STATUSES:
        raise PostAuthorityIntegrityError(
            INTEGRITY_FAILURE_INVALID_CONTEXT, f"unknown authoritative status {status!r}",
        )
    if str(plan_source.status) != status:
        raise PostAuthorityIntegrityError(
            INTEGRITY_FAILURE_INVALID_CONTEXT,
            f"plan source status {plan_source.status!r} disagrees with application status {status!r}",
        )
    decisions = plan_source.decisions or {}
    outcome_ids = {
        str(getattr(o, "semantic_idea_id", ""))
        for o in (getattr(authoritative_result, "idea_outcomes", ()) or ())
    }
    if outcome_ids != set(decisions):
        raise PostAuthorityIntegrityError(
            INTEGRITY_FAILURE_INVALID_CONTEXT,
            "plan source decisions do not match the applied idea outcomes",
        )
    return PostAuthorityValidationContext(
        authoritative_status=status,
        plan_source=plan_source,
        source_identity=authoritative_source_identity(plan_source),
        decision_count=len(decisions),
        unresolved_orphan_count=len(getattr(authoritative_result, "unresolved_orphan_realization_ids", ()) or ()),
    )


# ---------------------------------------------------------------------------
# Semantic selection signature + the executable mutation invariant
# ---------------------------------------------------------------------------

def _canonical_text(value: object) -> str:
    raw = unicodedata.normalize("NFKD", str(value or "").casefold())
    stripped = "".join(ch for ch in raw if not unicodedata.combining(ch))
    return " ".join(stripped.split())


@dataclass(frozen=True)
class SelectionSignatureEntry:
    clip_id: str
    realization_id: str | None
    parent_semantic_clip_id: str | None
    semantic_idea_id: str | None
    source_asset_id: str
    source_order: int
    start: float
    end: float
    text_digest: str

    def key(self) -> tuple:
        return (
            self.clip_id, self.realization_id, self.parent_semantic_clip_id,
            self.semantic_idea_id, self.source_asset_id, self.source_order,
            round(float(self.start), 3), round(float(self.end), 3), self.text_digest,
        )


@dataclass(frozen=True)
class SemanticSelectionSignature:
    entries: tuple[SelectionSignatureEntry, ...]
    ordered_digest: str
    membership_digest: str
    authority_identity: str = ""
    discarded_clip_ids: tuple[str, ...] = ()
    alternates_clip_ids: tuple[str, ...] = ()


def _entry(clip) -> SelectionSignatureEntry:
    realization = getattr(clip, "realization_id", None)
    parent = getattr(clip, "parent_semantic_clip_id", None)
    idea = getattr(clip, "semantic_idea_id", None)
    return SelectionSignatureEntry(
        clip_id=str(clip.clip_id),
        realization_id=str(realization) if realization else None,
        parent_semantic_clip_id=str(parent) if parent else None,
        semantic_idea_id=str(idea) if idea else None,
        source_asset_id=str(getattr(clip, "source_asset_id", "") or ""),
        source_order=int(getattr(clip, "source_order", 0) or 0),
        start=float(getattr(clip, "start", 0.0) or 0.0),
        end=float(getattr(clip, "end", 0.0) or 0.0),
        text_digest=hashlib.sha256(_canonical_text(getattr(clip, "text", "")).encode("utf-8")).hexdigest()[:16],
    )


def _digest(keys) -> str:
    payload = "\x1e".join("\x1f".join(str(part) for part in key) for key in keys)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def semantic_selection_signature(draft, *, authority_identity: str = "") -> SemanticSelectionSignature:
    """The semantic selection projection of `draft.selected`, in its own
    (rendered) order. `authority_identity` is the D-087 plan source digest
    the selection was produced under, so a signature also records WHICH
    authoritative verdict it corresponds to."""
    entries = tuple(_entry(clip) for clip in (draft.selected or ()))
    ordered_keys = [entry.key() for entry in entries]
    membership_keys = sorted(ordered_keys)
    return SemanticSelectionSignature(
        entries=entries,
        ordered_digest=_digest(ordered_keys),
        membership_digest=_digest(membership_keys),
        authority_identity=str(authority_identity or ""),
        discarded_clip_ids=tuple(sorted(str(c.clip_id) for c in (getattr(draft, "discarded", ()) or ()))),
        alternates_clip_ids=tuple(sorted(str(c.clip_id) for c in (getattr(draft, "alternates", ()) or ()))),
    )


@dataclass(frozen=True)
class SelectionMutationReport:
    phase: str
    order_sensitive: bool
    unchanged: bool
    added_clip_ids: tuple[str, ...] = ()
    removed_clip_ids: tuple[str, ...] = ()
    speech_changed_clip_ids: tuple[str, ...] = ()
    provenance_changed_clip_ids: tuple[str, ...] = ()
    order_changed: bool = False
    authority_changed: bool = False
    signature_before: str = ""
    signature_after: str = ""

    @property
    def status(self) -> str:
        return MUTATION_INVARIANT_PASS if self.unchanged else MUTATION_INVARIANT_FAIL

    @property
    def integrity_failure(self) -> str | None:
        return None if self.unchanged else INTEGRITY_FAILURE_SELECTION_MUTATION


def compare_selection_signatures(
    before: SemanticSelectionSignature,
    after: SemanticSelectionSignature,
    *,
    phase: str,
    order_sensitive: bool,
) -> SelectionMutationReport:
    """The executable invariant. `order_sensitive=True` (after StoryValidator:
    nothing at all may move); `order_sensitive=False` (after the bounded
    repair loop: the ONE permitted repair is a story-order reorder, so
    membership / speech / provenance / winner-composite composition must
    still be identical while order may differ)."""
    by_id_before = {e.clip_id: e for e in before.entries}
    by_id_after = {e.clip_id: e for e in after.entries}
    added = tuple(sorted(set(by_id_after) - set(by_id_before)))
    removed = tuple(sorted(set(by_id_before) - set(by_id_after)))
    speech_changed = tuple(sorted(
        cid for cid in set(by_id_before) & set(by_id_after)
        if by_id_before[cid].text_digest != by_id_after[cid].text_digest
    ))
    provenance_changed = tuple(sorted(
        cid for cid in set(by_id_before) & set(by_id_after)
        if (
            by_id_before[cid].realization_id, by_id_before[cid].parent_semantic_clip_id,
            by_id_before[cid].semantic_idea_id, by_id_before[cid].source_asset_id,
            round(by_id_before[cid].start, 3), round(by_id_before[cid].end, 3),
        ) != (
            by_id_after[cid].realization_id, by_id_after[cid].parent_semantic_clip_id,
            by_id_after[cid].semantic_idea_id, by_id_after[cid].source_asset_id,
            round(by_id_after[cid].start, 3), round(by_id_after[cid].end, 3),
        )
    ))
    membership_same = before.membership_digest == after.membership_digest
    order_changed = membership_same and before.ordered_digest != after.ordered_digest
    authority_changed = before.authority_identity != after.authority_identity
    if order_sensitive:
        unchanged = before.ordered_digest == after.ordered_digest and not authority_changed
        signature_before, signature_after = before.ordered_digest, after.ordered_digest
    else:
        unchanged = membership_same and not authority_changed
        signature_before, signature_after = before.membership_digest, after.membership_digest
    # A duplicate clip id appearing twice (or a clip dropped while another
    # identical id remains) changes the digest without changing the id
    # sets -- keep the digest as the authority, ids as the explanation.
    return SelectionMutationReport(
        phase=phase,
        order_sensitive=order_sensitive,
        unchanged=unchanged,
        added_clip_ids=added,
        removed_clip_ids=removed,
        speech_changed_clip_ids=speech_changed,
        provenance_changed_clip_ids=provenance_changed,
        order_changed=order_changed,
        authority_changed=authority_changed,
        signature_before=signature_before,
        signature_after=signature_after,
    )


def mutation_report_to_diagnostics(report: SelectionMutationReport | None) -> dict:
    if report is None:
        return {"status": MUTATION_INVARIANT_NOT_EVALUATED}
    return {
        "status": report.status,
        "phase": report.phase,
        "order_sensitive": report.order_sensitive,
        "integrity_failure": report.integrity_failure,
        "signature_before": report.signature_before,
        "signature_after": report.signature_after,
        "membership_added_clip_ids": list(report.added_clip_ids),
        "membership_removed_clip_ids": list(report.removed_clip_ids),
        "speech_changed_clip_ids": list(report.speech_changed_clip_ids),
        "provenance_changed_clip_ids": list(report.provenance_changed_clip_ids),
        "order_changed": report.order_changed,
        "authority_identity_changed": report.authority_changed,
    }


def signature_to_diagnostics(signature: SemanticSelectionSignature) -> dict:
    return {
        "ordered_digest": signature.ordered_digest,
        "membership_digest": signature.membership_digest,
        "authority_identity": signature.authority_identity,
        "selected_clip_ids": [e.clip_id for e in signature.entries],
        "selected_realization_ids": [e.realization_id for e in signature.entries],
        "selected_count": len(signature.entries),
        "discarded_count": len(signature.discarded_clip_ids),
        "alternates_count": len(signature.alternates_clip_ids),
    }


@dataclass(frozen=True)
class PostAuthorityBoundaryRecord:
    """Everything universal_clean_cut.py records under
    `diagnostics["post_authority_validation"]` (Section 9 observability)."""
    validation_mode: str
    context_status: str  # "present" | INTEGRITY_FAILURE_*
    context_detail: str
    authoritative_source_identity: str
    authoritative_status: str
    decision_count: int
    signature_after_authority: Mapping[str, object]
    signature_after_validation: Mapping[str, object] | None
    signature_after_repair: Mapping[str, object] | None
    validation_invariant: Mapping[str, object]
    repair_invariant: Mapping[str, object]
    integrity_failures: tuple[str, ...] = ()
    extra: Mapping[str, object] = field(default_factory=dict)

    @property
    def integrity_failed(self) -> bool:
        return bool(self.integrity_failures)

    def to_diagnostics(self) -> dict:
        return {
            "schema_version": "cutsell.post_authority_validation.v1",
            "validation_mode": self.validation_mode,
            "context_status": self.context_status,
            "context_detail": self.context_detail,
            "authoritative_source_identity": self.authoritative_source_identity,
            "authoritative_status": self.authoritative_status,
            "decision_count": self.decision_count,
            "signature_after_authority": dict(self.signature_after_authority),
            "signature_after_validation": dict(self.signature_after_validation) if self.signature_after_validation else None,
            "signature_after_repair": dict(self.signature_after_repair) if self.signature_after_repair else None,
            "validation_invariant": dict(self.validation_invariant),
            "repair_invariant": dict(self.repair_invariant),
            "integrity_failures": list(self.integrity_failures),
            "integrity_failed": self.integrity_failed,
            **dict(self.extra),
        }
