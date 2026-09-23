"""D-056.3 CONTRADICTION-SAFE COMPOSITE CONTRACT.

Root defect (docs/CUTSELL_DECISIONS.md D-056.2/D-056.3, live evidence: Run B
`tg_539b31f663aaf9e13f`, Run C `tg_f4b9e7c1fe3e28a1af`): an upstream
composite-forming mechanism (`claim_coverage_best_take.py`'s own narrow
2-piece fallback is one; `hybrid_composite_best_take`/`hybrid_semantic_
complementary_rescue` are others) could mark a 2+-member group's combined
claim coverage "complete" and get it accepted as `is_composite: true` /
`coverage_status: complete` in `canonical_edit_plan.py` WITHOUT ever
checking whether the members actually contradict each other -- while
StoryValidator's own `contradiction_findings`/FinalEditReviewer's
independent CONTRADICTION check still (correctly) flagged the exact same
pair. Two safety layers disagreeing instead of one shared, structurally
enforced contract.

This file is entirely generic -- no Video00 clip ids or phrases -- and
covers, per D-056.3 Section 6's required matrix:
  1. valid complementary composite -> accepted (PASS-shaped)
  2. positive vs negative -> REVIEW_REQUIRED (unresolved_ambiguous)
  3. incompatible numbers (5 vs 10) -> REVIEW_REQUIRED
  4. same number restated -> stays valid
  5. causal inversion (expressed via negation, the only contradiction
     signal this codebase's existing primitive detects -- general
     non-numeric/non-negation contradiction remains the same honest,
     documented gap `final_story_coherence_validation.py` already
     declares, not silently claimed solved here) -> REVIEW_REQUIRED
  6. safe chronological complementary pair -> valid
  7/8. wrong-order / redundant-member pairs -> already covered,
     unmodified, by `tests/test_cutsell_claim_coverage_best_take.py`'s own
     `test_composite_skipped_when_candidates_overlap_in_time` and
     `test_composite_skipped_when_unique_contributions_share_a_claim_type`
     -- this directive does not touch ClaimCoverage's own formation logic
     (Section 7: "Do not change ... ClaimCoverage"), so those invariants
     are referenced, not duplicated, here.
  9. critical-coverage-complete but contradictory -> invalid (the exact
     D-056.2 generic shape, reproduced end to end)
  10. StoryValidator's unresolved-family bookkeeping retains the family
     when the composite is invalid
  11. FinalEditReviewer independently agrees with the invalid-composite
     finding (same shared `contradiction_findings` list all three layers
     now read -- structurally cannot disagree)

Layers exercised together, in real pipeline order, for every scenario:
StoryValidator (`apply_final_story_coherence_validation`) -> CanonicalEditPlan
(`build_canonical_edit_plan`) -> FinalEditReviewer (`review`).
"""
from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.contradiction_signal import TextContradiction, any_pair_contradicts, detect_text_contradiction
from cutsell_worker.final_edit_reviewer import CONTRADICTION, DUPLICATE_IDEA, UNRESOLVED_RETRY, review
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation


def clip(clip_id, start, end, text, *, selected, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def draft(*, selected=(), discarded=(), take_judge_groups=(), claim_coverage_composites=()):
    diagnostics = {"take_judge_groups": list(take_judge_groups)}
    if claim_coverage_composites:
        diagnostics["claim_coverage_best_take"] = {
            "status": "applied",
            "composites": list(claim_coverage_composites),
        }
    return DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id="p",
        strategy=EditStrategy.STORYTELLING,
        selected=selected,
        alternates=(),
        discarded=discarded,
        diagnostics=diagnostics,
    )


def ranked_row(clip_id, score):
    return {"clip_id": clip_id, "score": score, "reason": "watch_listen_baseline"}


def run_pipeline(d):
    """StoryValidator -> CanonicalEditPlan -> FinalEditReviewer, in real
    pipeline order -- exactly the chain D-056.3 Section 5 requires to agree."""
    d = apply_final_story_coherence_validation(d)
    plan = build_canonical_edit_plan(d)
    result = review(plan)
    return d, plan, result


def _two_member_composite_draft(text_a: str, text_b: str, *, group_id="g1"):
    a = clip("a", 0.0, 5.0, text_a, selected=True)
    b = clip("b", 5.0, 10.0, text_b, selected=True)
    return draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": group_id, "ranked": [ranked_row("a", 0.7), ranked_row("b", 0.65)]}],
        claim_coverage_composites=[{"group_id": group_id, "clip_ids": ["a", "b"], "reason": "claim_coverage_complementary"}],
    )


# --- contradiction_signal.py primitive -------------------------------------

def test_detect_text_contradiction_negation_mismatch():
    signal = detect_text_contradiction("she is the only one affected", "she is not the only one affected")
    assert signal.negation_conflict is True
    assert signal.has_conflict is True


def test_detect_text_contradiction_number_mismatch():
    signal = detect_text_contradiction("about 5 percent of cases", "about 10 percent of cases")
    assert signal.number_conflict is True
    assert signal.has_conflict is True


def test_detect_text_contradiction_same_number_is_not_a_conflict():
    signal = detect_text_contradiction("about 5 percent of cases, per her doctor", "her doctor said 5 percent, roughly")
    assert signal.number_conflict is False
    assert signal.has_conflict is False


def test_detect_text_contradiction_no_signal_present_is_not_a_conflict():
    signal = detect_text_contradiction("she felt tired that week", "she also had a mild headache")
    assert signal == TextContradiction(number_conflict=False, negation_conflict=False)


def test_any_pair_contradicts_checks_every_pair():
    assert any_pair_contradicts([
        "it happened once during the trial",
        "it happened once during the trial",
        "it never happened during the trial",
    ]) is True
    assert any_pair_contradicts([
        "it happened once during the trial",
        "it happened twice too during the trial",
        "yes it did happen during the trial",
    ]) is False


def test_any_pair_contradicts_empty_or_single_is_false():
    assert any_pair_contradicts([]) is False
    assert any_pair_contradicts(["only one text here"]) is False


# --- Composite rule matrix (D-056.3 Section 3/6) ---------------------------

def test_valid_complementary_composite_is_accepted():
    d = _two_member_composite_draft(
        "she first noticed the symptom in the spring",
        "by summer it had gotten noticeably worse",
    )
    _, plan, result = run_pipeline(d)

    idea = plan.ideas[0]
    assert idea.is_composite is True
    assert idea.coverage_status == "complete"
    assert not any(f.kind == CONTRADICTION for f in result.findings)
    assert not any(f.kind in (DUPLICATE_IDEA, UNRESOLVED_RETRY) for f in result.findings)


def test_positive_vs_negative_composite_is_rejected():
    d = _two_member_composite_draft(
        "she is the only one in her family with this condition",
        "she is not the only one in her family with this condition",
    )
    _, plan, result = run_pipeline(d)

    idea = plan.ideas[0]
    assert idea.is_composite is False
    assert idea.coverage_status == "unresolved_ambiguous"
    kinds = {f.kind for f in result.findings}
    assert CONTRADICTION in kinds
    assert DUPLICATE_IDEA in kinds
    assert UNRESOLVED_RETRY in kinds


def test_incompatible_number_composite_is_rejected():
    d = _two_member_composite_draft(
        "roughly 5 percent of cases are hereditary",
        "roughly 10 percent of cases are hereditary",
    )
    _, plan, result = run_pipeline(d)

    idea = plan.ideas[0]
    assert idea.is_composite is False
    assert idea.coverage_status == "unresolved_ambiguous"
    assert any(f.kind == CONTRADICTION for f in result.findings)


def test_same_number_restated_composite_stays_valid():
    d = _two_member_composite_draft(
        "roughly 5 percent of cases are hereditary, her doctor explained",
        "her doctor said 5 percent of cases run in families",
    )
    _, plan, result = run_pipeline(d)

    idea = plan.ideas[0]
    assert idea.is_composite is True
    assert idea.coverage_status == "complete"
    assert not any(f.kind == CONTRADICTION for f in result.findings)


def test_causal_inversion_expressed_as_negation_is_rejected():
    d = _two_member_composite_draft(
        "stress causes the flare-ups, according to her notes",
        "stress does not cause the flare-ups, according to her notes",
    )
    _, plan, result = run_pipeline(d)

    idea = plan.ideas[0]
    assert idea.is_composite is False
    assert idea.coverage_status == "unresolved_ambiguous"
    assert any(f.kind == CONTRADICTION for f in result.findings)


def test_safe_chronological_complementary_pair_is_valid():
    a = clip("a", 0.0, 5.0, "first she describes the initial checkup", selected=True)
    b = clip("b", 5.05, 10.0, "then she describes the follow-up visit", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.7), ranked_row("b", 0.65)]}],
        claim_coverage_composites=[{"group_id": "g1", "clip_ids": ["a", "b"]}],
    )
    _, plan, result = run_pipeline(d)

    idea = plan.ideas[0]
    assert idea.is_composite is True
    assert idea.coverage_status == "complete"
    assert not any(f.kind == CONTRADICTION for f in result.findings)


# --- D-056.2 generic shape, reproduced end to end (D-056.3 Section 1/6.9) ---

def test_critical_coverage_complete_but_contradictory_composite_is_invalid():
    """The exact generic shape D-056.2 found live: realization A + B, same
    semantic idea/retry family, an upstream mechanism already recorded them
    as a claim-coverage-complete composite (`claim_coverage_best_take.
    composites`) -- but A and B contain a real contradiction. Before
    D-056.3, CanonicalEditPlan accepted this as `is_composite: true` while
    FinalEditReviewer's independent CONTRADICTION check still fired --
    exactly the dual truth this directive makes structurally impossible."""
    d = _two_member_composite_draft(
        "the patient reported no history of the condition in the family",
        "the patient reported a history of the condition in the family",
    )
    _, plan, result = run_pipeline(d)

    idea = plan.ideas[0]
    # CompositeResolver's acceptance gate: REJECTED, not RESOLVED_COMPOSITE.
    assert idea.is_composite is False
    assert idea.coverage_status == "unresolved_ambiguous"
    # FinalEditReviewer: independently agrees, same finding set as any other
    # unresolved retry family with a real contradiction.
    kinds = {f.kind for f in result.findings}
    assert {CONTRADICTION, DUPLICATE_IDEA, UNRESOLVED_RETRY}.issubset(kinds)


def test_storyvalidator_retains_unresolved_family_when_composite_invalid():
    """D-056.3 Section 4: StoryValidator must NOT drop a family from
    unresolved_families/residual_family_count merely because an upstream
    mechanism claims a composite exists -- only a contradiction-free
    composite may resolve it."""
    d = _two_member_composite_draft(
        "the patient reported no history of the condition in the family",
        "the patient reported a history of the condition in the family",
    )
    out, _plan, _result = run_pipeline(d)

    coherence = out.diagnostics["final_story_coherence_validation"]
    assert coherence["residual_family_count"] == 1
    assert coherence["unresolved_family_count"] == 1
    assert coherence["resolved_family_count"] == 0
    assert coherence["unresolved_families"][0]["group_id"] == "g1"
    assert sorted(coherence["unresolved_families"][0]["still_selected_clip_ids"]) == ["a", "b"]
    assert coherence["freeze_blocked"] is True


def test_storyvalidator_exempts_family_when_composite_is_actually_safe():
    """Mirror of the above: a legitimately safe composite (no contradiction)
    IS exempted from unresolved-family bookkeeping, same as before D-056.3
    -- the fix narrows what upstream's claim is trusted for, it does not
    make every composite unresolved."""
    d = _two_member_composite_draft(
        "she first noticed the symptom in the spring",
        "by summer it had gotten noticeably worse",
    )
    out, _plan, _result = run_pipeline(d)

    coherence = out.diagnostics["final_story_coherence_validation"]
    assert coherence["residual_family_count"] == 0
    assert coherence["unresolved_family_count"] == 0
    assert coherence["freeze_blocked"] is False


def test_final_edit_reviewer_agrees_with_invalid_composite_finding():
    """D-056.3 Section 5/6.11: FinalEditReviewer remains the independent
    safety net, unmodified -- and now structurally CANNOT disagree with
    StoryValidator/CanonicalEditPlan, since all three read the same
    contradiction_findings list built from the one shared primitive."""
    d = _two_member_composite_draft(
        "roughly 5 percent of cases are hereditary",
        "roughly 10 percent of cases are hereditary",
    )
    out, plan, result = run_pipeline(d)

    coherence = out.diagnostics["final_story_coherence_validation"]
    contradiction_group_ids = {row["group_id"] for row in coherence["contradiction_findings"]}
    review_contradiction_ideas = {f.idea_id for f in result.findings if f.kind == CONTRADICTION}
    assert contradiction_group_ids == {"g1"}
    assert review_contradiction_ideas == {"g1"}
    # And CanonicalEditPlan's own field agrees too -- three layers, one answer.
    assert plan.ideas[0].coverage_status == "unresolved_ambiguous"
    assert result.status == "FAIL"


def test_three_member_composite_any_contradicting_pair_is_rejected():
    """The contract is not bounded to exactly 2 members -- any pair among a
    larger accepted composite must be contradiction-free."""
    a = clip("a", 0.0, 5.0, "she described the first symptom", selected=True)
    b = clip("b", 5.05, 10.0, "she said it was never that severe", selected=True)
    c = clip("c", 10.05, 15.0, "she said it was in fact quite severe", selected=True)
    d = draft(
        selected=(a, b, c),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("a", 0.7), ranked_row("b", 0.65), ranked_row("c", 0.6)],
        }],
        claim_coverage_composites=[{"group_id": "g1", "clip_ids": ["a", "b", "c"]}],
    )
    _, plan, result = run_pipeline(d)

    idea = plan.ideas[0]
    assert idea.is_composite is False
    assert idea.coverage_status == "unresolved_ambiguous"
    assert any(f.kind == CONTRADICTION for f in result.findings)
