"""D-090 -- POST-RESOLVER STORYVALIDATOR IMMUTABILITY.

Evidence: D-089 canary run 33960713625 (engine 40dde20). The Unified
Realization Resolver emitted a two-member RESOLVED_COMPOSITE; StoryValidator's
residual-family resolution (arbiter merge 0.9) then discarded one member;
CanonicalEditPlan correctly failed closed (`realization_not_selected`);
Freeze was blocked by a post-resolver semantic membership mutation.

Every "full path" test here runs the REAL production import path:
`process_universal_clean_cut_sources` -> Ledger -> Resolver -> authoritative
application -> StoryValidator (post-authority) -> CanonicalEditPlan ->
FinalEditReviewer -> bounded repair -> Freeze gate, with only the physical
stages (ASR/Boundary polish) stubbed.

FIXTURE PROVENANCE (Section 2): the live run's texts/ids are NOT copied
here. The generic fixture reproduces the exact live SHAPE -- two retry-
family members each carrying a distinct CRITICAL claim of the SAME claim
type, so the legacy ClaimCoverageBestTake 2-piece composite declines
(`types_a & types_b`) exactly as it did live, while the resolver's own
composite rule composites them -- plus an always-merge semantic-equivalence
arbiter standing in for the live arbiter's 0.9 "same idea" answer.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import cutsell_worker.final_story_coherence_validation as fscv
import cutsell_worker.universal_clean_cut as universal
from cutsell_worker.canonical_edit_plan import (
    AuthoritativeIdeaDecision,
    AuthoritativePlanSource,
    assess_authoritative_membership,
    build_authoritative_plan_source,
    build_canonical_edit_plan,
)
from cutsell_worker.contracts import (
    SCHEMA_VERSION,
    DraftClip,
    DraftTimeline,
    EditStrategy,
    JobState,
    ProcessingResult,
)
from cutsell_worker.final_edit_reviewer import DUPLICATE_IDEA, UNRESOLVED_RETRY
from cutsell_worker.final_story_coherence_validation import (
    AUTHORITY_FAMILY_ACCEPTED,
    AUTHORITY_FAMILY_NO_DECISION,
    AUTHORITY_FAMILY_REVIEW_REQUIRED,
    AUTHORITY_FAMILY_STRUCTURAL_FAILURE,
    apply_final_story_coherence_validation,
    apply_post_authority_story_validation,
)
from cutsell_worker.post_authority_validation import (
    INTEGRITY_FAILURE_INVALID_CONTEXT,
    INTEGRITY_FAILURE_MISSING_CONTEXT,
    INTEGRITY_FAILURE_SELECTION_MUTATION,
    LEGACY_RESOLVING_MODE,
    PHASE_BOUNDED_REPAIR,
    PHASE_STORY_VALIDATION,
    POST_AUTHORITY_VALIDATION_MODE,
    PostAuthorityIntegrityError,
    build_post_authority_validation_context,
    compare_selection_signatures,
    semantic_selection_signature,
)
from cutsell_worker.realization_resolver import (
    RESOLVED_COMPOSITE,
    RESOLVED_WINNER,
    SEMANTICALLY_RESOLVED,
    apply_authoritative_realization_resolution,
    resolve_realizations_shadow,
)
from cutsell_worker.repair_loop import run_repair_loop
from cutsell_worker.resolver_mode import (
    ENV_VAR_NAME,
    RESOLVER_MODE_AUTHORITATIVE,
    RESOLVER_MODE_LEGACY,
    RESOLVER_MODE_SHADOW,
)
from cutsell_worker.semantic_idea_equivalence import IdeaEquivalenceDecision, IdeaEquivalenceResult
from cutsell_worker.semantic_ledger import build_semantic_ledger_shadow

# ---------------------------------------------------------------------------
# Generic fixtures (see module docstring: shape-exact, not data-exact)
# ---------------------------------------------------------------------------

IDEA = "idea_generic_diagnosis_family"
GROUP = "g_generic_diagnosis_family"
# Two distinct CRITICAL DIAGNOSIS_IDENTIFICATION claims -> same claim type.
TEXT_A = "The doctors confirmed the diagnosis was thyroid cancer."
TEXT_B = "The biopsy confirmed it was a papillary carcinoma."
# Same type but a NUMBER conflict -> a genuinely contradictory pair.
TEXT_A_NUM = "The scan confirmed a benign nodule of 2 centimeters."
TEXT_B_NUM = "Only 5 to 10 percent of cases are hereditary, science backs that."


class AlwaysMergeArbiter:
    """Stands in for the live arbiter's 'same idea, 0.9' answer -- and
    counts how often it is asked."""

    def __init__(self):
        self.calls = 0

    def check(self, request):
        self.calls += 1
        return IdeaEquivalenceResult(
            tuple(IdeaEquivalenceDecision(i, True, 0.9, "both recordings discuss the same statistic")
                  for i in range(len(request.pairs))),
            "fake", "fake", True, True,
        )


def _clip(cid, text, start, end, selected, idea=IDEA, **extra):
    return DraftClip(
        clip_id=cid, source_asset_id="src", source_order=0, start=start, end=end,
        text=text, caption_text=text, selected=selected,
        realization_id=extra.pop("realization_id", f"real_{cid}"), semantic_idea_id=idea,
        retry_family_id=idea, complete_idea=extra.pop("complete_idea", True), **extra,
    )


def _group(group_id, *ranked):
    return {"group_id": group_id, "ranked": [
        {"clip_id": cid, "score": score, "reason": "watch_listen_baseline"} for cid, score in ranked
    ]}


def _family_draft(text_a=TEXT_A, text_b=TEXT_B, *, b_selected=False, extra_groups=(), extra_selected=()):
    hook = _clip("hook", "Today I want to tell you my story.", 0.0, 2.0, True, "idea_hook")
    a = _clip("A", text_a, 10.0, 15.0, True)
    b = _clip("B", text_b, 16.0, 21.0, b_selected)
    cta = _clip("cta", "Take care of yourself and get checked.", 40.0, 45.0, True, "idea_cta")
    selected = [hook, a, *([b] if b_selected else []), *extra_selected, cta]
    discarded = [] if b_selected else [b]
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=tuple(selected), alternates=(), discarded=tuple(discarded),
        diagnostics={"take_judge_groups": [
            _group("g_hook", ("hook", 0.9)),
            # DeliveryScore ranks B above A: the legacy residual merge keeps B, drops A.
            _group(GROUP, ("B", 0.72), ("A", 0.66)),
            _group("g_cta", ("cta", 0.9)),
            *extra_groups,
        ]},
    )


def _run(monkeypatch, draft, *, env, arbiter=None):
    monkeypatch.setenv(ENV_VAR_NAME, env)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    monkeypatch.setattr(universal, "enforce_complete_idea_boundaries", lambda result, paths, **kw: result)

    def fake_process(request, local_paths, **kwargs):
        return ProcessingResult(
            schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY,
            draft=draft, stage_status={},
        )
    monkeypatch.setattr(universal, "process_local_sources", fake_process)
    return universal.process_universal_clean_cut_sources(
        object(), {}, asr_provider=object(), selection_reasoner=None,
        semantic_equivalence_arbiter=arbiter,
    )


def _selected_ids(result):
    return [c.clip_id for c in result.draft.selected]


def _plan_idea(result, idea_id=GROUP):
    return next(i for i in result.draft.diagnostics["canonical_edit_plan"]["ideas"] if i["idea_id"] == idea_id)


def _fam_authority(result):
    return next(i for i in result.draft.diagnostics["realization_resolver_authority"]["ideas"]
                if i.get("semantic_idea_id") == IDEA)


def _applied_composite(draft=None):
    """The resolver's real composite applied to the real draft -- the exact
    state the post-authority StoryValidator receives."""
    draft = draft or _family_draft()
    ledger = build_semantic_ledger_shadow(draft)
    report = resolve_realizations_shadow(ledger)
    assert report.idea_resolutions[IDEA].decision_status == RESOLVED_COMPOSITE
    applied = apply_authoritative_realization_resolution(draft, ledger, report)
    assert applied.status == SEMANTICALLY_RESOLVED
    source = build_authoritative_plan_source(applied, ledger)
    context = build_post_authority_validation_context(applied, source)
    return applied, source, context


# ---------------------------------------------------------------------------
# Section 2: RED reproduction (exact generic D-089 mutation)
# ---------------------------------------------------------------------------

def test_red_legacy_resolving_pass_discards_an_authoritative_composite_member():
    """Before D-090 the AUTHORITATIVE second pass WAS the legacy resolving
    pass: on the resolver's applied composite [A, B], an always-merge
    arbiter makes it discard A -- the D-089 live mutation."""
    applied, source, _ = _applied_composite()
    assert [c.clip_id for c in applied.draft.selected] == ["hook", "A", "B", "cta"]
    arbiter = AlwaysMergeArbiter()
    mutated = apply_final_story_coherence_validation(applied.draft, semantic_equivalence_arbiter=arbiter)
    assert arbiter.calls == 1
    assert [c.clip_id for c in mutated.selected] == ["hook", "B", "cta"]
    diag = mutated.diagnostics["final_story_coherence_validation"]
    assert diag["validation_mode"] == LEGACY_RESOLVING_MODE
    assert diag["resolved_families"][0]["discarded_clip_ids"] == ["A"]
    # ...and CanonicalEditPlan correctly refuses the mutated result.
    loop = run_repair_loop(mutated, authoritative_source=source)
    idea = next(i for i in loop.final_plan.ideas if i.idea_id == GROUP)
    assert idea.structural_validation_passed is False
    assert "realization_not_selected:real_A" in idea.structural_validation_failures
    assert loop.status == "NEEDS_HUMAN_REVIEW"
    assert {f.kind for f in loop.final_review.findings} >= {DUPLICATE_IDEA, UNRESOLVED_RETRY}


def test_green_post_authority_pass_keeps_the_composite_and_never_asks_the_merge_arbiter():
    applied, source, context = _applied_composite()
    arbiter = AlwaysMergeArbiter()
    validated = apply_post_authority_story_validation(
        applied.draft, context=context,
    )
    assert arbiter.calls == 0
    assert [c.clip_id for c in validated.selected] == ["hook", "A", "B", "cta"]
    diag = validated.diagnostics["final_story_coherence_validation"]
    assert diag["validation_mode"] == POST_AUTHORITY_VALIDATION_MODE
    assert diag["authoritative_source_identity"] == context.source_identity
    assert diag["residual_family_count"] == 1
    assert diag["resolved_families"] == [] and diag["resolved_family_count"] == 0
    accepted = diag["authoritative_families_accepted"]
    assert len(accepted) == 1 and accepted[0]["authority_status"] == AUTHORITY_FAMILY_ACCEPTED
    assert accepted[0]["still_selected_clip_ids"] == ["B", "A"]
    assert accepted[0]["resolved_clip_ids"] == ["A", "B"]
    assert diag["unresolved_families"] == [] and diag["authority_membership_findings"] == []
    assert diag["freeze_blocked"] is False
    assert diag["selection_mutation_self_check"]["status"] == "PASS"
    loop = run_repair_loop(validated, authoritative_source=source)
    idea = next(i for i in loop.final_plan.ideas if i.idea_id == GROUP)
    assert idea.is_composite is True and idea.coverage_status == "complete"
    assert idea.winning_clip_ids == ("A", "B")
    assert loop.status == "PASS"


def test_full_path_d089_shape_reaches_freeze_with_the_composite_intact(monkeypatch):
    arbiter = AlwaysMergeArbiter()
    result = _run(monkeypatch, _family_draft(), env=RESOLVER_MODE_AUTHORITATIVE, arbiter=arbiter)
    d = result.draft.diagnostics
    fam = _fam_authority(result)
    assert fam["decision_status"] == RESOLVED_COMPOSITE
    assert fam["composite_realization_ids"] == ["real_A", "real_B"]
    # Legacy ClaimCoverageBestTake declined the 2-piece composite (same claim type).
    assert (d.get("claim_coverage_best_take") or {}).get("composites") == []
    assert _selected_ids(result) == ["hook", "A", "B", "cta"]
    assert d["final_story_coherence_validation"]["validation_mode"] == POST_AUTHORITY_VALIDATION_MODE
    assert d["final_story_coherence_validation"]["resolved_families"] == []
    assert d["final_story_coherence_validation_legacy_evidence"]["validation_mode"] == LEGACY_RESOLVING_MODE
    plan_idea = _plan_idea(result)
    assert plan_idea["coverage_status"] == "complete" and plan_idea["structural_validation_passed"] is True
    assert list(plan_idea["winning_clip_ids"]) == ["A", "B"]
    assert d["final_edit_reviewer"]["status"] == "PASS"
    assert d["repair_loop"]["status"] == "PASS"
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is False
    assert result.stage_status["post_authority_integrity_failure"] is False
    pav = d["post_authority_validation"]
    assert pav["validation_mode"] == POST_AUTHORITY_VALIDATION_MODE
    assert pav["context_status"] == "present"
    assert pav["authoritative_source_identity"].startswith("authsrc_")
    assert pav["validation_invariant"]["status"] == "PASS" and pav["repair_invariant"]["status"] == "PASS"
    assert pav["validation_invariant"]["membership_removed_clip_ids"] == []
    assert pav["signature_after_authority"]["selected_clip_ids"] == ["hook", "A", "B", "cta"]
    assert pav["signature_after_authority"]["ordered_digest"] == pav["signature_after_validation"]["ordered_digest"]
    assert pav["integrity_failed"] is False
    assert pav["authoritative_families_accepted"][0]["group_id"] == GROUP


# ---------------------------------------------------------------------------
# Section 4/5: validate without editing; authoritative family bookkeeping
# ---------------------------------------------------------------------------

def test_valid_winner_survives_unchanged_and_is_not_treated_as_unresolved(monkeypatch):
    """A RESOLVED_WINNER family (B is a thinner restatement A fully covers)
    keeps exactly the resolver's winner: never residual, never re-merged,
    never treated as an unresolved contest."""
    draft = _family_draft("The doctors confirmed the diagnosis was thyroid cancer.",
                          "The diagnosis was thyroid cancer.", b_selected=False)
    result = _run(monkeypatch, draft, env=RESOLVER_MODE_AUTHORITATIVE, arbiter=AlwaysMergeArbiter())
    fam = _fam_authority(result)
    assert fam["decision_status"] == RESOLVED_WINNER and fam["winner_realization_id"] == "real_A"
    assert _selected_ids(result) == ["hook", "A", "cta"]
    assert _plan_idea(result)["coverage_status"] == "complete"
    diag = result.draft.diagnostics["final_story_coherence_validation"]
    assert diag["authority_membership_findings"] == [] and diag["resolved_families"] == []
    assert result.draft.diagnostics["post_authority_validation"]["validation_invariant"]["status"] == "PASS"


def test_fragment_identity_composite_member_split_into_fragments_stays_accepted():
    applied, source, context = _applied_composite()
    a = next(c for c in applied.draft.selected if c.clip_id == "A")
    frag1 = DraftClip(**{**a.__dict__, "clip_id": "A#1", "end": 12.5, "text": "The doctors confirmed",
                         "caption_text": "The doctors confirmed", "render_fragment_id": "A#1",
                         "parent_semantic_clip_id": "A", "fragment_index": 0, "fragment_count": 2})
    frag2 = DraftClip(**{**a.__dict__, "clip_id": "A#2", "start": 12.5, "text": "the diagnosis was thyroid cancer.",
                         "caption_text": "the diagnosis was thyroid cancer.", "render_fragment_id": "A#2",
                         "parent_semantic_clip_id": "A", "fragment_index": 1, "fragment_count": 2})
    selected = tuple(
        [c for c in applied.draft.selected if c.clip_id != "A"][:1] + [frag1, frag2]
        + [c for c in applied.draft.selected if c.clip_id not in ("A", "hook")]
    )
    split = DraftTimeline(**{**applied.draft.__dict__, "selected": selected})
    validated = apply_post_authority_story_validation(split, context=context)
    assert [c.clip_id for c in validated.selected] == ["hook", "A#1", "A#2", "B", "cta"]
    assessments = assess_authoritative_membership(validated, source)
    assert assessments[GROUP].accepted_as_resolved is True
    assert assessments[GROUP].resolved_clip_ids == ("A#1", "A#2", "B")
    plan = build_canonical_edit_plan(validated, authoritative_source=source)
    idea = next(i for i in plan.ideas if i.idea_id == GROUP)
    assert idea.is_composite and idea.coverage_status == "complete"
    sig = semantic_selection_signature(validated)
    assert [e.parent_semantic_clip_id for e in sig.entries] == [None, "A", "A", None, None]
    assert [e.realization_id for e in sig.entries][1:3] == ["real_A", "real_A"]


def test_review_required_family_remains_blocked_and_is_not_rewritten():
    applied, source, _ = _applied_composite()
    review = AuthoritativePlanSource(
        status="REVIEW_REQUIRED",
        decisions={IDEA: AuthoritativeIdeaDecision(
            semantic_idea_id=IDEA, decision_status="REVIEW_REQUIRED", winner_realization_id=None,
            composite_realization_ids=(), candidate_realization_ids=("real_A", "real_B"),
            covered_canonical_claim_ids=(), missing_critical_claim_ids=(), decision_reason="contradiction",
        )},
    )
    fake_result = SimpleNamespace(
        status="REVIEW_REQUIRED", idea_outcomes=(SimpleNamespace(semantic_idea_id=IDEA),),
        unresolved_orphan_realization_ids=(),
    )
    context = build_post_authority_validation_context(fake_result, review)
    validated = apply_post_authority_story_validation(applied.draft, context=context)
    assert [c.clip_id for c in validated.selected] == ["hook", "A", "B", "cta"]  # never collapsed
    diag = validated.diagnostics["final_story_coherence_validation"]
    assert diag["unresolved_families"][0]["authority_status"] == AUTHORITY_FAMILY_REVIEW_REQUIRED
    assert diag["authority_membership_findings"][0]["blocking"] is True
    assert diag["freeze_blocked"] is True
    loop = run_repair_loop(validated, authoritative_source=review)
    assert loop.status == "NEEDS_HUMAN_REVIEW"


def test_contradictory_composite_remains_blocked_without_silent_rewrite(monkeypatch):
    """Resolver composites two same-type claims that carry a number
    conflict: structural validation says `composite_members_contradict`;
    both members stay selected (nothing rewritten), Freeze blocked."""
    result = _run(
        monkeypatch, _family_draft(TEXT_A_NUM, TEXT_B_NUM),
        env=RESOLVER_MODE_AUTHORITATIVE, arbiter=AlwaysMergeArbiter(),
    )
    assert _fam_authority(result)["decision_status"] == RESOLVED_COMPOSITE
    assert _selected_ids(result) == ["hook", "A", "B", "cta"]
    plan_idea = _plan_idea(result)
    assert plan_idea["structural_validation_passed"] is False
    assert "composite_members_contradict" in plan_idea["structural_validation_failures"]
    diag = result.draft.diagnostics["final_story_coherence_validation"]
    assert diag["resolved_families"] == []
    assert diag["unresolved_families"][0]["authority_status"] == AUTHORITY_FAMILY_STRUCTURAL_FAILURE
    assert diag["contradiction_findings"] and diag["contradiction_findings"][0]["number_conflict"] is True
    assert diag["freeze_blocked"] is True
    assert result.draft.diagnostics["final_edit_reviewer"]["status"] == "FAIL"
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is True
    assert result.stage_status["post_authority_integrity_failure"] is False
    assert result.draft.diagnostics["post_authority_validation"]["validation_invariant"]["status"] == "PASS"


def test_missing_or_stale_member_remains_blocked_and_is_not_restored():
    applied, source, context = _applied_composite()
    stale = DraftTimeline(**{
        **applied.draft.__dict__,
        "selected": tuple(c for c in applied.draft.selected if c.clip_id != "B"),
        "discarded": tuple([*applied.draft.discarded, next(c for c in applied.draft.selected if c.clip_id == "B")]),
    })
    validated = apply_post_authority_story_validation(stale, context=context)
    assert [c.clip_id for c in validated.selected] == ["hook", "A", "cta"]  # NOT restored
    loop = run_repair_loop(validated, authoritative_source=source)
    idea = next(i for i in loop.final_plan.ideas if i.idea_id == GROUP)
    assert "realization_not_selected:real_B" in idea.structural_validation_failures
    assert loop.status == "NEEDS_HUMAN_REVIEW"


def test_extra_selected_member_remains_blocked():
    applied, source, context = _applied_composite()
    extra = _clip("C", "They said it was a thyroid problem, I think.", 22.0, 26.0, True)
    widened = DraftTimeline(**{
        **applied.draft.__dict__,
        "selected": tuple([*applied.draft.selected[:3], extra, *applied.draft.selected[3:]]),
        "diagnostics": {**applied.draft.diagnostics, "take_judge_groups": [
            _group("g_hook", ("hook", 0.9)), _group(GROUP, ("B", 0.72), ("A", 0.66), ("C", 0.5)), _group("g_cta", ("cta", 0.9)),
        ]},
    })
    validated = apply_post_authority_story_validation(widened, context=context)
    assert [c.clip_id for c in validated.selected] == ["hook", "A", "B", "C", "cta"]  # not pruned here
    diag = validated.diagnostics["final_story_coherence_validation"]
    finding = diag["authority_membership_findings"][0]
    assert finding["authority_status"] == AUTHORITY_FAMILY_STRUCTURAL_FAILURE
    assert any(f.startswith("selected_members_outside_composite:") and "C" in f for f in finding["structural_validation_failures"])
    assert diag["freeze_blocked"] is True
    assert run_repair_loop(validated, authoritative_source=source).status == "NEEDS_HUMAN_REVIEW"


def test_family_without_any_authoritative_decision_stays_unresolved():
    applied, _, _ = _applied_composite()
    empty = AuthoritativePlanSource(status=SEMANTICALLY_RESOLVED, decisions={})
    fake_result = SimpleNamespace(status=SEMANTICALLY_RESOLVED, idea_outcomes=(), unresolved_orphan_realization_ids=())
    context = build_post_authority_validation_context(fake_result, empty)
    validated = apply_post_authority_story_validation(applied.draft, context=context)
    diag = validated.diagnostics["final_story_coherence_validation"]
    assert diag["unresolved_families"][0]["authority_status"] == AUTHORITY_FAMILY_NO_DECISION
    assert diag["freeze_blocked"] is True
    assert [c.clip_id for c in validated.selected] == ["hook", "A", "B", "cta"]


def test_claim_and_atom_loss_checks_remain_active_in_the_post_authority_pass():
    applied, source, context = _applied_composite()
    lost = _clip("X", "The medication costs 40 dollars a month and it is not covered by insurance.",
                 30.0, 35.0, False, "idea_cost")
    with_loss = DraftTimeline(**{
        **applied.draft.__dict__,
        "discarded": tuple([*applied.draft.discarded, lost]),
        "diagnostics": {**applied.draft.diagnostics, "take_judge_groups": [
            *applied.draft.diagnostics["take_judge_groups"], _group("g_cost", ("X", 0.5)),
        ]},
    })
    validated = apply_post_authority_story_validation(with_loss, context=context)
    diag = validated.diagnostics["final_story_coherence_validation"]
    assert diag["missing_idea_coverage"] and diag["missing_idea_coverage"][0]["group_id"] == "g_cost"
    assert diag["lost_semantic_atoms"] or diag["lost_critical_claims"]
    assert diag["freeze_blocked"] is True
    assert [c.clip_id for c in validated.selected] == ["hook", "A", "B", "cta"]  # still no restore


def test_alternates_are_evaluated_as_discarded_but_buckets_are_returned_untouched():
    applied, source, context = _applied_composite()
    alt = _clip("ALT", "A small aside about my dog.", 36.0, 38.0, False, "idea_aside")
    with_alt = DraftTimeline(**{**applied.draft.__dict__, "alternates": (alt,)})
    validated = apply_post_authority_story_validation(with_alt, context=context)
    assert [c.clip_id for c in validated.alternates] == ["ALT"]
    assert "ALT" not in [c.clip_id for c in validated.discarded]
    diag = validated.diagnostics["final_story_coherence_validation"]
    assert diag["alternates_folded_for_evaluation_only"] is True
    assert diag["alternates_folded_clip_ids"] == ["ALT"]


# ---------------------------------------------------------------------------
# Section 3: explicit mode; missing context fails closed
# ---------------------------------------------------------------------------

def test_mode_is_decided_by_the_typed_context_not_by_diagnostics_keys_or_labels():
    applied, source, context = _applied_composite()
    with_keys = DraftTimeline(**{**applied.draft.__dict__, "diagnostics": {
        **applied.draft.diagnostics, "authoritative_plan_source": {"plan_semantic_source": "authoritative_realization_resolver"},
        "claim_coverage_best_take": {"composites": []},
    }})
    arbiter = AlwaysMergeArbiter()
    legacy = apply_final_story_coherence_validation(with_keys, semantic_equivalence_arbiter=arbiter)
    assert legacy.diagnostics["final_story_coherence_validation"]["validation_mode"] == LEGACY_RESOLVING_MODE
    assert [c.clip_id for c in legacy.selected] == ["hook", "B", "cta"]  # still the legacy resolving pass
    explicit = apply_final_story_coherence_validation(
        with_keys, semantic_equivalence_arbiter=arbiter, post_authority_context=context,
    )
    assert explicit.diagnostics["final_story_coherence_validation"]["validation_mode"] == POST_AUTHORITY_VALIDATION_MODE
    assert [c.clip_id for c in explicit.selected] == ["hook", "A", "B", "cta"]


def test_missing_context_is_an_integrity_failure_never_a_legacy_fallback():
    applied, _, _ = _applied_composite()
    out = apply_post_authority_story_validation(applied.draft, context=None)
    assert [c.clip_id for c in out.selected] == ["hook", "A", "B", "cta"]
    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["status"] == "integrity_failure"
    assert diag["integrity_failure"] == INTEGRITY_FAILURE_MISSING_CONTEXT
    assert diag["freeze_blocked"] is True
    assert diag["resolved_families"] == []


@pytest.mark.parametrize("bad", ["no_result", "no_source", "status_mismatch", "outcome_mismatch"])
def test_context_builder_fails_closed_on_missing_or_inconsistent_authoritative_state(bad):
    applied, source, _ = _applied_composite()
    if bad == "no_result":
        with pytest.raises(PostAuthorityIntegrityError) as exc:
            build_post_authority_validation_context(None, source)
        assert exc.value.code == INTEGRITY_FAILURE_MISSING_CONTEXT
    elif bad == "no_source":
        with pytest.raises(PostAuthorityIntegrityError) as exc:
            build_post_authority_validation_context(applied, None)
        assert exc.value.code == INTEGRITY_FAILURE_MISSING_CONTEXT
    elif bad == "status_mismatch":
        with pytest.raises(PostAuthorityIntegrityError) as exc:
            build_post_authority_validation_context(applied, AuthoritativePlanSource(status="REVIEW_REQUIRED", decisions=source.decisions))
        assert exc.value.code == INTEGRITY_FAILURE_INVALID_CONTEXT
    else:
        with pytest.raises(PostAuthorityIntegrityError) as exc:
            build_post_authority_validation_context(applied, AuthoritativePlanSource(status=source.status, decisions={}))
        assert exc.value.code == INTEGRITY_FAILURE_INVALID_CONTEXT


def test_full_path_missing_context_blocks_freeze_with_a_named_integrity_failure(monkeypatch):
    def broken(result, source):
        raise PostAuthorityIntegrityError(INTEGRITY_FAILURE_MISSING_CONTEXT, "injected")
    monkeypatch.setattr(universal, "build_post_authority_validation_context", broken)
    result = _run(monkeypatch, _family_draft(), env=RESOLVER_MODE_AUTHORITATIVE, arbiter=AlwaysMergeArbiter())
    assert _selected_ids(result) == ["hook", "A", "B", "cta"]  # never fell back to the legacy resolving pass
    diag = result.draft.diagnostics["final_story_coherence_validation"]
    assert diag["status"] == "integrity_failure" and diag["integrity_failure"] == INTEGRITY_FAILURE_MISSING_CONTEXT
    pav = result.draft.diagnostics["post_authority_validation"]
    assert pav["context_status"] == INTEGRITY_FAILURE_MISSING_CONTEXT and pav["context_detail"] == "injected"
    assert pav["integrity_failures"] == [INTEGRITY_FAILURE_MISSING_CONTEXT]
    assert result.stage_status["post_authority_integrity_failure"] is True
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is True
    assert result.draft.diagnostics["selection_boundary_contract"]["status"] == "not_frozen_post_authority_integrity_failure"


# ---------------------------------------------------------------------------
# Section 6: the executable boundary invariant
# ---------------------------------------------------------------------------

def test_injected_downstream_mutation_in_validation_triggers_the_invariant_and_is_not_restored(monkeypatch):
    real = universal.apply_post_authority_story_validation

    def mutating(draft, **kwargs):
        out = real(draft, **kwargs)
        return DraftTimeline(**{**out.__dict__, "selected": tuple(c for c in out.selected if c.clip_id != "A")})
    monkeypatch.setattr(universal, "apply_post_authority_story_validation", mutating)
    result = _run(monkeypatch, _family_draft(), env=RESOLVER_MODE_AUTHORITATIVE, arbiter=AlwaysMergeArbiter())
    pav = result.draft.diagnostics["post_authority_validation"]
    assert pav["validation_invariant"]["status"] == "FAIL"
    assert pav["validation_invariant"]["integrity_failure"] == INTEGRITY_FAILURE_SELECTION_MUTATION
    assert pav["validation_invariant"]["membership_removed_clip_ids"] == ["A"]
    # The drift is reported at the phase it happened AND is still visible
    # after bounded repair (both compare against the post-authority truth).
    assert pav["integrity_failures"][0] == f"{INTEGRITY_FAILURE_SELECTION_MUTATION}:{PHASE_STORY_VALIDATION}"
    assert f"{INTEGRITY_FAILURE_SELECTION_MUTATION}:{PHASE_BOUNDED_REPAIR}" in pav["integrity_failures"]
    assert result.stage_status["post_authority_integrity_failure"] is True
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is True
    assert result.draft.diagnostics["selection_boundary_contract"]["status"] == "not_frozen_post_authority_integrity_failure"
    # Not silently restored: the mutated selection is what is reported.
    assert "A" not in _selected_ids(result)
    # The authoritative source was NOT rebuilt from the mutated draft.
    source_rows = result.draft.diagnostics["authoritative_plan_source"]["ideas"]
    fam_row = next(r for r in source_rows if r["semantic_idea_id"] == IDEA)
    assert list(fam_row["composite_realization_ids"]) == ["real_A", "real_B"]


def test_injected_membership_mutation_in_bounded_repair_triggers_the_repair_invariant(monkeypatch):
    real = universal.run_repair_loop

    def mutating(draft, **kwargs):
        out = real(draft, **kwargs)
        dropped = DraftTimeline(**{**out.final_draft.__dict__, "selected": tuple(c for c in out.final_draft.selected if c.clip_id != "B")})
        return type(out)(status=out.status, final_draft=dropped, final_plan=out.final_plan, final_review=out.final_review, attempts=out.attempts)
    monkeypatch.setattr(universal, "run_repair_loop", mutating)
    result = _run(monkeypatch, _family_draft(), env=RESOLVER_MODE_AUTHORITATIVE, arbiter=AlwaysMergeArbiter())
    pav = result.draft.diagnostics["post_authority_validation"]
    assert pav["validation_invariant"]["status"] == "PASS"
    assert pav["repair_invariant"]["status"] == "FAIL"
    assert pav["repair_invariant"]["membership_removed_clip_ids"] == ["B"]
    assert pav["integrity_failures"] == [f"{INTEGRITY_FAILURE_SELECTION_MUTATION}:{PHASE_BOUNDED_REPAIR}"]
    assert result.stage_status["post_authority_integrity_failure"] is True


def test_a_pure_story_order_reorder_in_bounded_repair_passes_the_repair_projection(monkeypatch):
    real = universal.run_repair_loop

    def reordering(draft, **kwargs):
        out = real(draft, **kwargs)
        sel = list(out.final_draft.selected)
        sel[1], sel[2] = sel[2], sel[1]
        reordered = DraftTimeline(**{**out.final_draft.__dict__, "selected": tuple(sel)})
        return type(out)(status=out.status, final_draft=reordered, final_plan=out.final_plan, final_review=out.final_review, attempts=out.attempts)
    monkeypatch.setattr(universal, "run_repair_loop", reordering)
    result = _run(monkeypatch, _family_draft(), env=RESOLVER_MODE_AUTHORITATIVE, arbiter=AlwaysMergeArbiter())
    pav = result.draft.diagnostics["post_authority_validation"]
    assert pav["repair_invariant"]["status"] == "PASS"
    assert pav["repair_invariant"]["order_changed"] is True
    assert pav["integrity_failed"] is False


def test_signature_detects_speech_provenance_and_order_changes():
    applied, _, context = _applied_composite()
    base = semantic_selection_signature(applied.draft, authority_identity=context.source_identity)
    sel = list(applied.draft.selected)
    a = sel[1]
    reworded = DraftTimeline(**{**applied.draft.__dict__, "selected": tuple(
        [sel[0], DraftClip(**{**a.__dict__, "text": "The doctors never confirmed the diagnosis."}), *sel[2:]]
    )})
    rep = compare_selection_signatures(base, semantic_selection_signature(reworded, authority_identity=context.source_identity), phase="t", order_sensitive=False)
    assert rep.unchanged is False and rep.speech_changed_clip_ids == ("A",)
    reprov = DraftTimeline(**{**applied.draft.__dict__, "selected": tuple(
        [sel[0], DraftClip(**{**a.__dict__, "realization_id": "real_other"}), *sel[2:]]
    )})
    rep = compare_selection_signatures(base, semantic_selection_signature(reprov, authority_identity=context.source_identity), phase="t", order_sensitive=False)
    assert rep.unchanged is False and rep.provenance_changed_clip_ids == ("A",)
    swapped = DraftTimeline(**{**applied.draft.__dict__, "selected": tuple([sel[0], sel[2], sel[1], sel[3]])})
    strict = compare_selection_signatures(base, semantic_selection_signature(swapped, authority_identity=context.source_identity), phase="t", order_sensitive=True)
    loose = compare_selection_signatures(base, semantic_selection_signature(swapped, authority_identity=context.source_identity), phase="t", order_sensitive=False)
    assert strict.unchanged is False and strict.order_changed is True
    assert loose.unchanged is True and loose.order_changed is True
    other_authority = semantic_selection_signature(applied.draft, authority_identity="authsrc_other")
    assert compare_selection_signatures(base, other_authority, phase="t", order_sensitive=True).authority_changed is True


# ---------------------------------------------------------------------------
# Section 7: D-089 retention (placement + effective importance)
# ---------------------------------------------------------------------------

def test_d089_composite_contiguity_and_departed_slot_placement_survive_post_authority_validation(monkeypatch):
    IDEA_ACNE, IDEA_RES = "idea_acne", "idea_res"
    old = _clip("old_acne", "Seasonal back acne that I that I treated with resor", 166.0, 182.0, True, IDEA_ACNE, complete_idea=False)
    new = _clip("new_acne", "Seasonal back acne that I treated with resorcinol", 185.2, 189.8, False, IDEA_ACNE)
    res = _clip("res", "resorcinol.", 191.1, 191.7, True, IDEA_RES)
    hook = _clip("hook", "Today I want to tell you my story.", 0.0, 2.0, True, "idea_hook")
    a = _clip("A", TEXT_A, 10.0, 15.0, True)
    b = _clip("B", TEXT_B, 16.0, 21.0, False)
    cta = _clip("cta", "Take care of yourself and get checked.", 300.0, 305.0, True, "idea_cta")
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(hook, a, old, res, cta), alternates=(), discarded=(b, new),
        diagnostics={"take_judge_groups": [
            _group("g_hook", ("hook", 0.9)), _group(GROUP, ("B", 0.72), ("A", 0.66)),
            _group("g_acne", ("old_acne", 0.7), ("new_acne", 0.6)), _group("g_res", ("res", 0.9)), _group("g_cta", ("cta", 0.9)),
        ]},
    )
    result = _run(monkeypatch, draft, env=RESOLVER_MODE_AUTHORITATIVE, arbiter=AlwaysMergeArbiter())
    assert _fam_authority(result)["decision_status"] == RESOLVED_COMPOSITE
    assert _selected_ids(result) == ["hook", "A", "B", "new_acne", "res", "cta"]
    placement = result.draft.diagnostics["authoritative_story_placement"]
    assert placement["all_blocks_contiguous"] is True and placement["winner_replacement_count"] == 1
    assert "canonical_effective_importance" in result.draft.diagnostics
    pav = result.draft.diagnostics["post_authority_validation"]
    assert pav["validation_invariant"]["status"] == "PASS" and pav["repair_invariant"]["status"] == "PASS"
    assert result.draft.diagnostics["final_story_coherence_validation"]["resolved_families"] == []
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is False


def test_effective_importance_index_is_still_consumed_by_the_post_authority_pass():
    applied, _, context = _applied_composite()
    sentinel = {}
    seen = {}
    real = fscv._lost_critical_claims

    def spy(draft, **kwargs):
        seen["index"] = kwargs.get("canonical_effective_importance_index")
        return real(draft, **kwargs)
    fscv._lost_critical_claims = spy
    try:
        apply_post_authority_story_validation(applied.draft, context=context, canonical_effective_importance_index=sentinel)
    finally:
        fscv._lost_critical_claims = real
    assert seen["index"] is sentinel


# ---------------------------------------------------------------------------
# LEGACY / SHADOW parity + real import / wrapper path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [RESOLVER_MODE_LEGACY, RESOLVER_MODE_SHADOW])
def test_legacy_and_shadow_still_run_the_resolving_pass_and_carry_no_boundary_record(monkeypatch, mode):
    arbiter = AlwaysMergeArbiter()
    result = _run(monkeypatch, _family_draft(b_selected=True), env=mode, arbiter=arbiter)
    d = result.draft.diagnostics
    assert "post_authority_validation" not in d
    assert "final_story_coherence_validation_legacy_evidence" not in d
    assert d["final_story_coherence_validation"]["validation_mode"] == LEGACY_RESOLVING_MODE
    # The legacy resolving pass still merges the two selected members (B kept, A discarded).
    assert d["final_story_coherence_validation"]["resolved_families"][0]["discarded_clip_ids"] == ["A"]
    assert "A" not in _selected_ids(result)
    assert result.stage_status["post_authority_integrity_failure"] is False


def test_authoritative_path_uses_the_real_wrapper_from_the_package(monkeypatch):
    assert universal.apply_post_authority_story_validation is fscv.apply_post_authority_story_validation
    calls = []
    real = fscv.apply_post_authority_story_validation

    def spy(draft, **kwargs):
        calls.append(kwargs)
        return real(draft, **kwargs)
    monkeypatch.setattr(universal, "apply_post_authority_story_validation", spy)
    _run(monkeypatch, _family_draft(), env=RESOLVER_MODE_AUTHORITATIVE, arbiter=AlwaysMergeArbiter())
    assert len(calls) == 1
    ctx = calls[0]["context"]
    assert ctx is not None and ctx.authoritative_status == SEMANTICALLY_RESOLVED
    assert calls[0]["integrity_failure"] is None
    assert ctx.plan_source.decisions[IDEA].decision_status == RESOLVED_COMPOSITE


def test_plan_and_story_validator_share_one_structural_assessment():
    applied, source, context = _applied_composite()
    validated = apply_post_authority_story_validation(applied.draft, context=context)
    plan = build_canonical_edit_plan(validated, authoritative_source=source)
    assessment = assess_authoritative_membership(validated, source)[GROUP]
    idea = next(i for i in plan.ideas if i.idea_id == GROUP)
    assert idea.winning_clip_ids == assessment.winning_clip_ids == ("A", "B")
    assert idea.structural_validation_passed is assessment.structural_validation_passed is True
    accepted = validated.diagnostics["final_story_coherence_validation"]["authoritative_families_accepted"][0]
    assert accepted["resolved_clip_ids"] == list(assessment.resolved_clip_ids)
