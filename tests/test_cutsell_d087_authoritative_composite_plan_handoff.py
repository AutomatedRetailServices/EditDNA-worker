"""D-087 -- AUTHORITATIVE COMPOSITE -> CANONICAL EDIT PLAN HANDOFF.

Single-truth contract (docs/CUTSELL_DECISIONS.md D-086/D-087): in
AUTHORITATIVE resolver mode CanonicalEditPlan REPRESENTS the Unified
Realization Resolver's own per-idea verdict; it never reinterprets a valid
RESOLVED_COMPOSITE as an unresolved duplicate family, and it fails closed
(BLOCK and explain) on every structural-integrity failure instead of
inventing an alternative semantic answer.

No Video00 clip ids/text/timestamps in production logic; every fixture here
is generic (diagnosis+hindsight, product benefit+dosage, hook+CTA).
"""
from dataclasses import asdict

import pytest

import cutsell_worker.universal_clean_cut as universal
from cutsell_worker.canonical_edit_plan import (
    PLAN_SOURCE_AUTHORITATIVE,
    PLAN_SOURCE_LEGACY,
    AuthoritativeIdeaDecision,
    AuthoritativePlanSource,
    authoritative_plan_source_from_diagnostics,
    authoritative_plan_source_to_diagnostics,
    build_authoritative_plan_source,
    build_canonical_edit_plan,
)
from cutsell_worker.canonical_identity import mint_semantic_idea_id
from cutsell_worker.contracts import (
    DraftClip, DraftTimeline, EditStrategy, JobState, ProcessingResult, SCHEMA_VERSION,
)
from cutsell_worker.final_edit_reviewer import DUPLICATE_IDEA, UNRESOLVED_RETRY, review
from cutsell_worker.realization_resolver import (
    RESOLVED_COMPOSITE,
    RESOLVED_WINNER,
    REVIEW_REQUIRED,
    SEMANTICALLY_RESOLVED,
    _place_restored_clips_at_story_position,
    apply_authoritative_realization_resolution,
    resolve_realizations_shadow,
)
from cutsell_worker.repair_loop import run_repair_loop
from cutsell_worker.resolver_mode import (
    ENV_VAR_NAME, RESOLVER_MODE_AUTHORITATIVE, RESOLVER_MODE_LEGACY, RESOLVER_MODE_SHADOW,
)
from cutsell_worker.semantic_ledger import (
    CanonicalClaimRecord, RealizationRecord, SemanticIdeaRecord, SemanticLedger,
)


# ---------------------------------------------------------------------------
# Generic fixtures
# ---------------------------------------------------------------------------

IDEA = "idea_generic_diag_hindsight"
GROUP = "g_diag_hindsight"

# Realization A: critical diagnosis/result + supporting hindsight statement.
TEXT_A = "The scan confirmed the result was a benign nodule. Symptoms I had seemed normal but there were signs looking back."
# Realization B: clean complementary hindsight statement.
TEXT_B = "Symptoms that did not seem suspicious to me but now that I analyze it they were suspicious."


def _clip(clip_id, text, *, start, end, selected, realization_id=None, semantic_idea_id=IDEA, **extra):
    return DraftClip(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
        realization_id=realization_id or f"real_{clip_id}", semantic_idea_id=semantic_idea_id,
        retry_family_id=semantic_idea_id, complete_idea=True, **extra,
    )


def _draft(*, selected=(), discarded=(), alternates=(), groups=None, extra=None):
    diagnostics = {
        "take_judge_groups": list(groups or ()),
        "final_story_coherence_validation": {
            "freeze_blocked": False, "lost_semantic_atoms": [], "contradiction_findings": [],
        },
        "hybrid_editorial_chunks": [],
    }
    diagnostics.update(extra or {})
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=tuple(selected), alternates=tuple(alternates), discarded=tuple(discarded),
        diagnostics=diagnostics,
    )


def _group(group_id, *ranked):
    return {"group_id": group_id, "ranked": [
        {"clip_id": cid, "score": score, "reason": "watch_listen_baseline"} for cid, score in ranked
    ]}


def _decision(
    idea_id=IDEA, *, status=RESOLVED_COMPOSITE, composite=("real_A", "real_B"), winner=None,
    candidates=None, covered=("cclaim_diag", "cclaim_hind"), missing=(),
    reason="minimal_composite_covers_all_critical_claims",
):
    return AuthoritativeIdeaDecision(
        semantic_idea_id=idea_id, decision_status=status, winner_realization_id=winner,
        composite_realization_ids=tuple(composite),
        candidate_realization_ids=tuple(candidates if candidates is not None else composite),
        covered_canonical_claim_ids=tuple(covered), missing_critical_claim_ids=tuple(missing),
        decision_reason=reason,
    )


def _source(*decisions, status=SEMANTICALLY_RESOLVED):
    return AuthoritativePlanSource(status=status, decisions={d.semantic_idea_id: d for d in decisions})


def _diag_hindsight_draft(**overrides):
    a = _clip("A", TEXT_A, start=10.0, end=20.0, selected=True)
    b = _clip("B", TEXT_B, start=21.0, end=27.0, selected=True)
    kwargs = dict(
        selected=(a, b),
        # DeliveryScore ranks B ABOVE A -- Section 12 must never let that
        # reorder the composite's members.
        groups=[_group(GROUP, ("B", 0.72), ("A", 0.66))],
    )
    kwargs.update(overrides)
    return _draft(**kwargs)


def _idea_by_id(plan, idea_id):
    return next(i for i in plan.ideas if i.idea_id == idea_id)


# ---------------------------------------------------------------------------
# Section 8: D-086 generic regression
# ---------------------------------------------------------------------------

def test_section8_authoritative_resolved_composite_is_represented_as_complete_composite():
    draft = _diag_hindsight_draft()
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(_decision()))
    idea = _idea_by_id(plan, GROUP)

    assert plan.plan_semantic_source == PLAN_SOURCE_AUTHORITATIVE
    assert idea.plan_semantic_source == PLAN_SOURCE_AUTHORITATIVE
    assert idea.is_composite is True
    assert idea.coverage_status == "complete"
    assert idea.winning_clip_ids == ("A", "B")  # resolver member order, not DeliveryScore order
    assert idea.authoritative_resolution_status == RESOLVED_COMPOSITE
    assert idea.authoritative_composite_realization_ids == ("real_A", "real_B")
    assert idea.authoritative_resolved_clip_ids == ("A", "B")
    assert idea.authoritative_claim_coverage == ("cclaim_diag", "cclaim_hind")
    assert idea.authoritative_decision_reason == "minimal_composite_covers_all_critical_claims"
    assert idea.structural_validation_passed is True
    assert idea.structural_validation_failures == ()
    assert all(c.is_composite_piece for c in plan.keep_sequence if c.clip_id in ("A", "B"))

    result = review(plan)
    kinds = {f.kind for f in result.findings}
    assert DUPLICATE_IDEA not in kinds
    assert UNRESOLVED_RETRY not in kinds
    assert result.status == "PASS"
    assert plan.freeze_blocked is False


def test_section8_repair_loop_passes_and_freeze_not_blocked_by_this_family():
    draft = _diag_hindsight_draft()
    loop = run_repair_loop(draft, authoritative_source=_source(_decision()))
    assert loop.status == "PASS"
    assert loop.final_review.findings == ()
    assert _idea_by_id(loop.final_plan, GROUP).is_composite is True


def test_section8_same_draft_without_authoritative_source_still_reads_as_unresolved():
    """The exact pre-D-087 (legacy) behavior on the same shape -- proves the
    fix comes from consuming the resolver's verdict, not from weakening
    duplicate detection."""
    plan = build_canonical_edit_plan(_diag_hindsight_draft())
    idea = _idea_by_id(plan, GROUP)
    assert plan.plan_semantic_source == PLAN_SOURCE_LEGACY
    assert idea.coverage_status == "unresolved_ambiguous"
    assert idea.is_composite is False
    assert {f.kind for f in review(plan).findings} == {DUPLICATE_IDEA, UNRESOLVED_RETRY}


def test_section8_group_to_idea_mapping_falls_back_to_deterministic_mint_when_unstamped():
    a = _clip("A", TEXT_A, start=10.0, end=20.0, selected=True, semantic_idea_id=None)
    b = _clip("B", TEXT_B, start=21.0, end=27.0, selected=True, semantic_idea_id=None)
    draft = _draft(selected=(a, b), groups=[_group(GROUP, ("A", 0.7), ("B", 0.6))])
    minted = mint_semantic_idea_id(GROUP)
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(_decision(minted)))
    assert _idea_by_id(plan, GROUP).is_composite is True


# --- Section 8, end to end: real Ledger -> real resolver -> real application
#     -> plan source -> CanonicalEditPlan/FinalEditReviewer/repair loop.

def _claim(cid, tokens, importance="CRITICAL", claim_type="STATE_RESULT"):
    return CanonicalClaimRecord(
        canonical_claim_id=cid, claim_type=claim_type, content_tokens=frozenset(tokens),
        importance=importance, source_realization_ids=(), covered_by_realization_ids=(),
        coverage_state="unresolved",
    )


def _realization(rid, *, claim_ids, state, text, start, end, clip_ids=None):
    return RealizationRecord(
        realization_id=rid, semantic_idea_id=IDEA, retry_family_id=IDEA, source_span_ids=(),
        attempt_id=None, clip_ids=tuple(clip_ids or (rid.replace("real_", ""),)), text=text,
        start=start, end=end, delivery_score=None, state=state, discard_reason=None,
        replacement_realization_id=None, claim_ids=tuple(claim_ids), render_fragment_ids=(),
        complete_idea=True,
    )


def _ledger(realizations, claims, idea_realization_ids):
    ledger = SemanticLedger()
    for r in realizations:
        ledger.register_realization(r)
    for c in claims:
        ledger.register_claim(c)
    ledger.register_semantic_idea(SemanticIdeaRecord(
        semantic_idea_id=IDEA, retry_family_ids=(), realization_ids=tuple(idea_realization_ids),
        canonical_claim_ids=tuple(c.canonical_claim_id for c in claims),
        current_winner_realization_id=None, composite_realization_ids=(),
        coverage_status="unresolved_ambiguous", story_order_position=None,
    ))
    return ledger


def test_section8_end_to_end_real_resolver_composite_reaches_plan_as_complete():
    diag = _claim("cclaim_diag", {"scan", "confirmed", "benign", "nodule"})
    hind = _claim("cclaim_hind", {"symptoms", "seemed", "suspicious", "analyze"}, claim_type="NEGATION")
    ledger = _ledger(
        [
            _realization("real_A", claim_ids=("cclaim_diag",), state="selected", text=TEXT_A, start=10.0, end=20.0),
            _realization("real_B", claim_ids=("cclaim_hind",), state="discarded", text=TEXT_B, start=21.0, end=27.0),
        ],
        [diag, hind], ["real_A", "real_B"],
    )
    report = resolve_realizations_shadow(ledger)
    assert report.idea_resolutions[IDEA].decision_status == RESOLVED_COMPOSITE

    a = _clip("A", TEXT_A, start=10.0, end=20.0, selected=True)
    b = _clip("B", TEXT_B, start=21.0, end=27.0, selected=False)
    tail = _clip("Z", "closing call to action", start=40.0, end=45.0, selected=True, semantic_idea_id="idea_cta")
    draft = _draft(selected=(a, tail), discarded=(b,), groups=[_group(GROUP, ("A", 0.66), ("B", 0.72))])

    applied = apply_authoritative_realization_resolution(draft, ledger, report)
    assert applied.status == SEMANTICALLY_RESOLVED
    source = build_authoritative_plan_source(applied, ledger)
    assert source.decisions[IDEA].decision_status == RESOLVED_COMPOSITE
    assert set(source.decisions[IDEA].candidate_realization_ids) == {"real_A", "real_B"}

    loop = run_repair_loop(applied.draft, authoritative_source=source)
    idea = _idea_by_id(loop.final_plan, GROUP)
    assert idea.is_composite is True and idea.coverage_status == "complete"
    assert loop.status == "PASS"
    # Section 12: the restored member sits consecutively after its sibling,
    # not appended after the CTA.
    assert [c.clip_id for c in applied.draft.selected] == ["A", "B", "Z"]


# ---------------------------------------------------------------------------
# Section 9: true unresolved controls
# ---------------------------------------------------------------------------

def test_section9_resolver_review_required_keeps_family_unresolved_and_blocks():
    draft = _diag_hindsight_draft()
    decision = _decision(status=REVIEW_REQUIRED, composite=(), reason="no_single_or_composite_realization_covers_all_critical_claims")
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(decision, status="REVIEW_REQUIRED"))
    idea = _idea_by_id(plan, GROUP)
    assert idea.coverage_status == "unresolved_ambiguous"
    assert idea.is_composite is False
    assert idea.authoritative_resolution_status == REVIEW_REQUIRED
    result = review(plan)
    assert {f.kind for f in result.findings} == {DUPLICATE_IDEA, UNRESOLVED_RETRY}
    assert run_repair_loop(draft, authoritative_source=_source(decision)).status == "NEEDS_HUMAN_REVIEW"


def test_section9_two_selected_without_any_authoritative_composite_stay_unresolved():
    draft = _diag_hindsight_draft()
    # Source present (AUTHORITATIVE mode) but the resolver recorded nothing
    # for this group -- no composite to consume, legacy representation stands.
    plan = build_canonical_edit_plan(draft, authoritative_source=_source())
    idea = _idea_by_id(plan, GROUP)
    assert idea.coverage_status == "unresolved_ambiguous"
    assert idea.plan_semantic_source == PLAN_SOURCE_LEGACY
    assert idea.structural_validation_failures == ("no_authoritative_decision_recorded_for_group",)
    assert {f.kind for f in review(plan).findings} == {DUPLICATE_IDEA, UNRESOLVED_RETRY}


def test_section9_resolved_winner_with_two_selected_members_is_blocked_and_explained():
    draft = _diag_hindsight_draft()
    decision = _decision(status=RESOLVED_WINNER, winner="real_A", composite=(), candidates=("real_A", "real_B"))
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(decision))
    idea = _idea_by_id(plan, GROUP)
    assert idea.coverage_status == "unresolved_ambiguous"
    assert idea.structural_validation_passed is False
    assert any(f.startswith("selected_members_differ_from_authoritative_winner") for f in idea.structural_validation_failures)
    findings = review(plan).findings
    assert {f.kind for f in findings} == {DUPLICATE_IDEA, UNRESOLVED_RETRY}
    assert all(f.detail["reason"] == "authoritative_resolution_structural_validation_failed" for f in findings)


# ---------------------------------------------------------------------------
# Section 10: malformed composite controls -- all fail closed
# ---------------------------------------------------------------------------

def _assert_failed_closed(plan, expected_fragment):
    idea = _idea_by_id(plan, GROUP)
    assert idea.is_composite is False
    assert idea.coverage_status == "unresolved_ambiguous"
    assert idea.structural_validation_passed is False
    assert any(expected_fragment in f for f in idea.structural_validation_failures), idea.structural_validation_failures
    findings = review(plan).findings
    assert {f.kind for f in findings} == {DUPLICATE_IDEA, UNRESOLVED_RETRY}
    for f in findings:
        assert f.detail["reason"] == "authoritative_resolution_structural_validation_failed"
        assert f.owning_authority == "CanonicalEditPlan(structural_validation)"
        assert expected_fragment in " ".join(f.detail["structural_validation_failures"])
    return idea


def test_section10_missing_realization_id_fails_closed():
    plan = build_canonical_edit_plan(
        _diag_hindsight_draft(), authoritative_source=_source(_decision(composite=("real_A", "real_GHOST"), candidates=("real_A", "real_B", "real_GHOST"))),
    )
    _assert_failed_closed(plan, "unknown_realization:real_GHOST")


def test_section10_realization_not_selected_fails_closed():
    a = _clip("A", TEXT_A, start=10.0, end=20.0, selected=True)
    b = _clip("B", TEXT_B, start=21.0, end=27.0, selected=False)
    draft = _draft(selected=(a,), discarded=(b,), groups=[_group(GROUP, ("A", 0.7), ("B", 0.6))])
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(_decision()))
    idea = _idea_by_id(plan, GROUP)
    assert idea.is_composite is False
    assert idea.structural_validation_passed is False
    assert "realization_not_selected:real_B" in idea.structural_validation_failures
    # One surviving member: not a duplicate contest, but never a composite either.
    assert idea.winning_clip_ids == ("A",)


def test_section10_member_from_wrong_semantic_idea_fails_closed():
    a = _clip("A", TEXT_A, start=10.0, end=20.0, selected=True)
    b = _clip("B", TEXT_B, start=21.0, end=27.0, selected=True)
    other = _clip("X", "an unrelated product mention", start=30.0, end=33.0, selected=True, semantic_idea_id="idea_other")
    draft = _draft(selected=(a, b, other), groups=[_group(GROUP, ("A", 0.7), ("B", 0.6)), _group("g_other", ("X", 0.9))])
    plan = build_canonical_edit_plan(
        draft, authoritative_source=_source(_decision(composite=("real_A", "real_B", "real_X"))),
    )
    idea = _assert_failed_closed(plan, "realization_outside_take_group:real_X")
    assert "member_clip_stamped_with_other_idea:X" in idea.structural_validation_failures


def test_section10_member_outside_candidate_set_fails_closed():
    plan = build_canonical_edit_plan(
        _diag_hindsight_draft(), authoritative_source=_source(_decision(candidates=("real_A",))),
    )
    _assert_failed_closed(plan, "realization_outside_semantic_idea:real_B")


def test_section10_missing_critical_claim_hidden_by_composite_label_fails_closed():
    plan = build_canonical_edit_plan(
        _diag_hindsight_draft(), authoritative_source=_source(_decision(missing=("cclaim_dose",))),
    )
    _assert_failed_closed(plan, "missing_critical_claim_ids_nonempty:cclaim_dose")


def test_section10_contradictory_members_fail_closed():
    a = _clip("A", "only 5% of these cases are hereditary", start=10.0, end=12.0, selected=True)
    b = _clip("B", "only 10% of these cases are hereditary", start=13.0, end=15.0, selected=True)
    draft = _draft(selected=(a, b), groups=[_group(GROUP, ("A", 0.7), ("B", 0.6))])
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(_decision()))
    _assert_failed_closed(plan, "composite_members_contradict")


def test_section10_empty_composite_fails_closed():
    plan = build_canonical_edit_plan(_diag_hindsight_draft(), authoritative_source=_source(_decision(composite=())))
    _assert_failed_closed(plan, "empty_composite")


def test_section10_duplicate_member_ids_fail_closed():
    plan = build_canonical_edit_plan(
        _diag_hindsight_draft(), authoritative_source=_source(_decision(composite=("real_A", "real_A", "real_B"))),
    )
    _assert_failed_closed(plan, "duplicate_composite_member_ids")


def test_section10_wrong_status_labelled_composite_fails_closed():
    plan = build_canonical_edit_plan(
        _diag_hindsight_draft(),
        authoritative_source=_source(_decision(status=RESOLVED_WINNER, winner="real_A", composite=("real_A", "real_B"))),
    )
    # A RESOLVED_WINNER decision is validated on the winner path: two
    # survivors for a single-winner verdict is a structural failure.
    _assert_failed_closed(plan, "selected_members_differ_from_authoritative_winner")


def test_section10_stale_fragment_provenance_fails_closed():
    a = _clip("A", TEXT_A, start=10.0, end=20.0, selected=True)
    # A fragment claiming to realize real_B but whose provenance points at a
    # different (absent) parent realization -- stale mapping.
    b_frag = _clip(
        "B_frag", TEXT_B, start=21.0, end=27.0, selected=True, realization_id="real_B",
        render_fragment_id="rf_b1", parent_semantic_clip_id="B", parent_realization_id="real_OLD",
    )
    draft = _draft(selected=(a, b_frag), groups=[_group(GROUP, ("A", 0.7), ("B_frag", 0.6))])
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(_decision()))
    _assert_failed_closed(plan, "stale_fragment_provenance:B_frag")


def test_section10_fragment_whose_parent_vanished_and_carries_no_realization_id_fails_closed():
    a = _clip("A", TEXT_A, start=10.0, end=20.0, selected=True)
    orphan_frag = DraftClip(
        clip_id="B_frag", source_asset_id="src", source_order=0, start=21.0, end=27.0,
        text=TEXT_B, caption_text=TEXT_B, selected=True,
        render_fragment_id="rf_b1", parent_semantic_clip_id="B_gone", semantic_idea_id=IDEA,
    )
    draft = _draft(selected=(a, orphan_frag), groups=[_group(GROUP, ("A", 0.7), ("B_frag", 0.6))])
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(_decision()))
    # The fragment's realization cannot be resolved at all (parent gone, no
    # own realization_id) -> real_B is unknown to the draft -> fail closed,
    # and the unexplained fragment is named explicitly.
    idea = _assert_failed_closed(plan, "unknown_realization:real_B")
    assert "selected_members_outside_composite:B_frag" in idea.structural_validation_failures


def test_section10_extra_selected_member_not_named_by_composite_fails_closed():
    a = _clip("A", TEXT_A, start=10.0, end=20.0, selected=True)
    b = _clip("B", TEXT_B, start=21.0, end=27.0, selected=True)
    c = _clip("C", "a third take of the same beat", start=28.0, end=30.0, selected=True)
    draft = _draft(selected=(a, b, c), groups=[_group(GROUP, ("A", 0.7), ("B", 0.6), ("C", 0.5))])
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(_decision(candidates=("real_A", "real_B", "real_C"))))
    _assert_failed_closed(plan, "selected_members_outside_composite:C")


# ---------------------------------------------------------------------------
# Section 11: fragment / provenance support (D-046 / D-050A)
# ---------------------------------------------------------------------------

def test_section11_composite_member_split_into_selected_fragments_is_recognized():
    a = _clip("A", TEXT_A, start=10.0, end=20.0, selected=True)
    b1 = _clip(
        "B_f1", "Symptoms that did not seem suspicious to me", start=21.0, end=24.0, selected=True,
        realization_id="real_B", render_fragment_id="rf_b1", parent_semantic_clip_id="B",
        parent_realization_id="real_B", fragment_index=0, fragment_count=2,
    )
    b2 = _clip(
        "B_f2", "but now that I analyze it they were suspicious.", start=24.4, end=27.0, selected=True,
        realization_id="real_B", render_fragment_id="rf_b2", parent_semantic_clip_id="B",
        parent_realization_id="real_B", fragment_index=1, fragment_count=2,
    )
    # take_judge_groups still names the pre-split semantic member "B" (D-046 Case A shape).
    draft = _draft(selected=(a, b1, b2), groups=[_group(GROUP, ("B", 0.72), ("A", 0.66))])
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(_decision()))
    idea = _idea_by_id(plan, GROUP)
    assert idea.is_composite is True
    assert idea.coverage_status == "complete"
    assert idea.winning_clip_ids == ("A", "B_f1", "B_f2")
    assert idea.discarded_clip_ids == ()
    assert review(plan).status == "PASS"


def test_section11_legacy_d046_fragment_with_only_parent_clip_provenance_resolves_through_parent():
    a = _clip("A", TEXT_A, start=10.0, end=20.0, selected=True)
    parent_b = _clip("B", TEXT_B, start=21.0, end=27.0, selected=False)  # pre-split semantic clip, now discarded bucket
    frag = DraftClip(
        clip_id="B_f1", source_asset_id="src", source_order=0, start=21.0, end=27.0,
        text=TEXT_B, caption_text=TEXT_B, selected=True,
        render_fragment_id="rf_b1", parent_semantic_clip_id="B", semantic_idea_id=IDEA,
    )
    draft = _draft(selected=(a, frag), discarded=(parent_b,), groups=[_group(GROUP, ("A", 0.7), ("B", 0.6))])
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(_decision()))
    idea = _idea_by_id(plan, GROUP)
    assert idea.is_composite is True and idea.winning_clip_ids == ("A", "B_f1")


# ---------------------------------------------------------------------------
# Section 12: story order
# ---------------------------------------------------------------------------

def test_section12_composite_member_order_follows_resolver_not_delivery_score_or_clip_id():
    a = _clip("zzz_late_id", TEXT_A, start=10.0, end=20.0, selected=True, realization_id="real_A")
    b = _clip("aaa_early_id", TEXT_B, start=21.0, end=27.0, selected=True, realization_id="real_B")
    draft = _draft(selected=(a, b), groups=[_group(GROUP, ("aaa_early_id", 0.9), ("zzz_late_id", 0.5))])
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(_decision()))
    assert _idea_by_id(plan, GROUP).winning_clip_ids == ("zzz_late_id", "aaa_early_id")


def test_section12_restored_composite_member_is_placed_consecutively_after_its_sibling():
    a = _clip("A", TEXT_A, start=10.0, end=20.0, selected=True)
    b = _clip("B", TEXT_B, start=21.0, end=27.0, selected=False)
    hook = _clip("H", "hook", start=0.0, end=2.0, selected=True, semantic_idea_id="idea_hook")
    cta = _clip("Z", "call to action", start=40.0, end=45.0, selected=True, semantic_idea_id="idea_cta")
    placed = _place_restored_clips_at_story_position(
        [hook, a, cta], [b], (hook, a, cta),
        ideas_by_realization={"real_A": IDEA, "real_B": IDEA, "real_H": "idea_hook", "real_Z": "idea_cta"},
    )
    assert [c.clip_id for c in placed] == ["H", "A", "B", "Z"]


def test_section12_restored_member_recorded_before_its_sibling_is_placed_before_it():
    a = _clip("A", TEXT_A, start=21.0, end=27.0, selected=True)
    b = _clip("B", TEXT_B, start=10.0, end=20.0, selected=False)
    cta = _clip("Z", "call to action", start=40.0, end=45.0, selected=True, semantic_idea_id="idea_cta")
    placed = _place_restored_clips_at_story_position(
        [a, cta], [b], (a, cta), ideas_by_realization={"real_A": IDEA, "real_B": IDEA, "real_Z": "idea_cta"},
    )
    assert [c.clip_id for c in placed] == ["B", "A", "Z"]


def test_section12_winner_replacement_takes_departing_siblings_slot():
    thin = _clip("thin", "diagnosis", start=10.0, end=12.0, selected=True)
    rich = _clip("rich", "the scan confirmed the full diagnosis", start=13.0, end=18.0, selected=False)
    hook = _clip("H", "hook", start=0.0, end=2.0, selected=True, semantic_idea_id="idea_hook")
    cta = _clip("Z", "call to action", start=40.0, end=45.0, selected=True, semantic_idea_id="idea_cta")
    # `thin` departed selected (resolver discarded it); `rich` is restored.
    placed = _place_restored_clips_at_story_position(
        [hook, cta], [rich], (hook, thin, cta),
        ideas_by_realization={"real_thin": IDEA, "real_rich": IDEA, "real_H": "idea_hook", "real_Z": "idea_cta"},
    )
    assert [c.clip_id for c in placed] == ["H", "rich", "Z"]


def test_section12_restored_clip_with_no_sibling_context_keeps_append_behavior():
    lone = _clip("L", "lone restored realization", start=5.0, end=8.0, selected=False, semantic_idea_id="idea_lone")
    cta = _clip("Z", "call to action", start=40.0, end=45.0, selected=True, semantic_idea_id="idea_cta")
    placed = _place_restored_clips_at_story_position([cta], [lone], (cta,), ideas_by_realization={})
    assert [c.clip_id for c in placed] == ["Z", "L"]


def test_section12_placement_never_changes_membership():
    a = _clip("A", TEXT_A, start=10.0, end=20.0, selected=True)
    b = _clip("B", TEXT_B, start=21.0, end=27.0, selected=False)
    placed = _place_restored_clips_at_story_position([a], [b], (a,), ideas_by_realization={"real_A": IDEA, "real_B": IDEA})
    assert {c.clip_id for c in placed} == {"A", "B"} and len(placed) == 2


# ---------------------------------------------------------------------------
# Section 13: human choice contract -- a valid composite is decided, not a choice
# ---------------------------------------------------------------------------

def test_section13_valid_authoritative_composite_is_not_review_required_or_human_choice():
    loop = run_repair_loop(_diag_hindsight_draft(), authoritative_source=_source(_decision()))
    assert loop.status == "PASS"  # never NEEDS_HUMAN_REVIEW
    assert loop.attempts == ()
    assert loop.final_plan.validation_state == "frozen_ready"


# ---------------------------------------------------------------------------
# Section 14: sales / UGC generalization
# ---------------------------------------------------------------------------

def test_section14_product_benefit_plus_required_dosage_detail_kept_as_one_resolved_unit():
    benefit = _clip("BEN", "this serum cleared my skin in two weeks", start=5.0, end=9.0, selected=True, semantic_idea_id="idea_prod")
    dosage = _clip("DOSE", "you apply two drops every night before bed", start=9.5, end=13.0, selected=True, semantic_idea_id="idea_prod")
    draft = _draft(selected=(benefit, dosage), groups=[_group("g_prod", ("DOSE", 0.8), ("BEN", 0.7))])
    decision = _decision("idea_prod", composite=("real_BEN", "real_DOSE"), covered=("cclaim_benefit", "cclaim_dose"))
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(decision))
    idea = _idea_by_id(plan, "g_prod")
    assert idea.is_composite is True and idea.coverage_status == "complete"
    assert idea.winning_clip_ids == ("BEN", "DOSE")
    assert review(plan).status == "PASS"


def test_section14_hook_and_cta_are_distinct_resolved_winners_never_a_retry_composite():
    hook = _clip("HOOK", "this product changed my routine", start=0.0, end=3.0, selected=True, semantic_idea_id="idea_hook")
    cta = _clip("CTA", "get this product with the link below", start=50.0, end=54.0, selected=True, semantic_idea_id="idea_cta")
    draft = _draft(selected=(hook, cta), groups=[_group("g_hook", ("HOOK", 0.8)), _group("g_cta", ("CTA", 0.8))])
    source = _source(
        _decision("idea_hook", status=RESOLVED_WINNER, winner="real_HOOK", composite=(), candidates=("real_HOOK",), covered=("cclaim_hook",), reason="single_realization_full_critical_coverage"),
        _decision("idea_cta", status=RESOLVED_WINNER, winner="real_CTA", composite=(), candidates=("real_CTA",), covered=("cclaim_cta",), reason="single_realization_full_critical_coverage"),
    )
    plan = build_canonical_edit_plan(draft, authoritative_source=source)
    for gid, cid in (("g_hook", "HOOK"), ("g_cta", "CTA")):
        idea = _idea_by_id(plan, gid)
        assert idea.is_composite is False
        assert idea.coverage_status == "complete"
        assert idea.winning_clip_ids == (cid,)
        assert idea.authoritative_resolution_status == RESOLVED_WINNER
        assert idea.structural_validation_passed is True
    assert review(plan).status == "PASS"


# ---------------------------------------------------------------------------
# Section 5 / 15 / 16: legacy unchanged, diagnostics, no new authority
# ---------------------------------------------------------------------------

def test_section5_legacy_plan_carries_default_source_fields_and_identical_shape():
    a = _clip("A", "the winning delivery", start=0.0, end=5.0, selected=True)
    b = _clip("B", "a weaker retry", start=5.0, end=10.0, selected=False)
    draft = _draft(selected=(a,), discarded=(b,), groups=[_group(GROUP, ("A", 0.9), ("B", 0.5))])
    plan = build_canonical_edit_plan(draft)
    idea = plan.ideas[0]
    assert plan.plan_semantic_source == PLAN_SOURCE_LEGACY
    assert idea.plan_semantic_source == PLAN_SOURCE_LEGACY
    assert idea.authoritative_resolution_status is None
    assert idea.structural_validation_passed is None
    assert idea.structural_validation_failures == ()
    assert idea.coverage_status == "complete" and idea.winning_clip_ids == ("A",)


def test_section5_legacy_composite_evidence_path_unchanged_with_no_source():
    a = _clip("A", "first half of the beat", start=0.0, end=5.0, selected=True)
    b = _clip("B", "second half of the beat", start=5.0, end=10.0, selected=True)
    draft = _draft(
        selected=(a, b), groups=[_group(GROUP, ("A", 0.7), ("B", 0.6))],
        extra={"hybrid_editorial_chunks": [{"hybrid_composite_best_take": {"split_group_clip_ids": ["A", "B"]}}]},
    )
    plan = build_canonical_edit_plan(draft)
    assert plan.ideas[0].is_composite is True and plan.ideas[0].coverage_status == "complete"
    assert plan.ideas[0].plan_semantic_source == PLAN_SOURCE_LEGACY


def test_section15_plan_source_diagnostics_round_trip_and_fallback_from_draft_diagnostics():
    source = _source(_decision())
    payload = authoritative_plan_source_to_diagnostics(source)
    assert payload["plan_semantic_source"] == PLAN_SOURCE_AUTHORITATIVE
    assert payload["ideas"][0]["composite_realization_ids"] == ["real_A", "real_B"]
    restored = authoritative_plan_source_from_diagnostics(payload)
    assert restored.decisions[IDEA] == source.decisions[IDEA]
    assert authoritative_plan_source_from_diagnostics({"unrelated": 1}) is None
    assert authoritative_plan_source_from_diagnostics(None) is None

    # A resolved draft carrying the key (what live_render_qc re-resolves
    # from) yields the same representation without the in-memory object.
    draft = _diag_hindsight_draft(extra={"authoritative_plan_source": payload})
    plan = build_canonical_edit_plan(draft)
    assert plan.plan_semantic_source == PLAN_SOURCE_AUTHORITATIVE
    assert _idea_by_id(plan, GROUP).is_composite is True
    as_dict = asdict(plan)
    idea_row = next(i for i in as_dict["ideas"] if i["idea_id"] == GROUP)
    for key in (
        "plan_semantic_source", "authoritative_resolution_status", "authoritative_composite_realization_ids",
        "authoritative_resolved_clip_ids", "authoritative_claim_coverage", "structural_validation_passed",
        "structural_validation_failures",
    ):
        assert key in idea_row


def test_section16_plan_never_changes_membership_or_invents_a_composite():
    """No new authority: whatever the source says, `winning_clip_ids` is
    always a subset of the draft's own selected clips, and an accepted
    composite is exactly the resolver's members -- never a different set."""
    draft = _diag_hindsight_draft()
    selected_ids = {c.clip_id for c in draft.selected}
    for decision in (
        _decision(),
        _decision(composite=("real_A", "real_GHOST")),
        _decision(status=REVIEW_REQUIRED, composite=()),
        _decision(status=RESOLVED_WINNER, winner="real_A", composite=()),
    ):
        plan = build_canonical_edit_plan(draft, authoritative_source=_source(decision))
        idea = _idea_by_id(plan, GROUP)
        assert set(idea.winning_clip_ids) <= selected_ids
        if idea.is_composite:
            assert idea.winning_clip_ids == ("A", "B")
            assert decision.decision_status == RESOLVED_COMPOSITE


def test_section16_explicit_source_takes_precedence_over_stale_draft_diagnostics():
    stale_payload = authoritative_plan_source_to_diagnostics(_source(_decision(status=REVIEW_REQUIRED, composite=())))
    draft = _diag_hindsight_draft(extra={"authoritative_plan_source": stale_payload})
    plan = build_canonical_edit_plan(draft, authoritative_source=_source(_decision()))
    assert _idea_by_id(plan, GROUP).is_composite is True


# ---------------------------------------------------------------------------
# Section 5 end to end: only AUTHORITATIVE mode carries/consumes the source
# ---------------------------------------------------------------------------

def _run(monkeypatch, draft, *, env):
    monkeypatch.setenv(ENV_VAR_NAME, env)
    monkeypatch.setattr(universal, "polish_human_boundaries_v5", lambda result, paths: result)
    monkeypatch.setattr(universal, "enforce_complete_idea_boundaries", lambda result, paths, **kw: result)

    def fake_process(request, local_paths, **kwargs):
        return ProcessingResult(
            schema_version=SCHEMA_VERSION, project_id="p1", state=JobState.DRAFT_READY,
            draft=draft, stage_status={},
        )
    monkeypatch.setattr(universal, "process_local_sources", fake_process)
    return universal.process_universal_clean_cut_sources(object(), {}, asr_provider=object(), selection_reasoner=None)


@pytest.mark.parametrize("mode", [RESOLVER_MODE_LEGACY, RESOLVER_MODE_SHADOW])
def test_section5_legacy_and_shadow_modes_never_carry_the_authoritative_plan_source(monkeypatch, mode):
    a = _clip("A", "the clean complete idea", start=0.0, end=2.0, selected=True, semantic_idea_id="idea_1")
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=(a,), alternates=(), discarded=(), diagnostics={},
    )
    result = _run(monkeypatch, draft, env=mode)
    assert "authoritative_plan_source" not in result.draft.diagnostics
    assert result.draft.diagnostics["canonical_edit_plan"]["plan_semantic_source"] == PLAN_SOURCE_LEGACY


def test_section5_authoritative_mode_carries_and_consumes_the_plan_source(monkeypatch):
    a = _clip("A", "the clean complete idea", start=0.0, end=2.0, selected=True, semantic_idea_id="idea_1")
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=(a,), alternates=(), discarded=(),
        diagnostics={"take_judge_groups": [_group("g1", ("A", 0.9))]},
    )
    result = _run(monkeypatch, draft, env=RESOLVER_MODE_AUTHORITATIVE)
    payload = result.draft.diagnostics["authoritative_plan_source"]
    assert payload["plan_semantic_source"] == PLAN_SOURCE_AUTHORITATIVE
    assert payload["status"] == SEMANTICALLY_RESOLVED
    plan_diag = result.draft.diagnostics["canonical_edit_plan"]
    assert plan_diag["plan_semantic_source"] == PLAN_SOURCE_AUTHORITATIVE
    assert result.draft.diagnostics["canonical_edit_plan_legacy_evidence"]["plan_semantic_source"] == PLAN_SOURCE_LEGACY
    assert result.stage_status["freeze_blocked_pending_coherence_review"] is False
