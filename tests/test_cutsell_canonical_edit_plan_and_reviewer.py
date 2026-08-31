"""CanonicalEditPlan + FinalEditReviewer (D-024).

Both modules invent no new semantic detection -- they read evidence
final_story_coherence_validation.py already computed. These tests focus on
the plan-building/finding-mapping logic itself, not re-testing StoryValidator's
own detection (covered in tests/test_cutsell_final_story_coherence_validation.py).
"""
from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_edit_reviewer import (
    CONTRADICTION,
    DUPLICATE_IDEA,
    IDEA_COVERAGE_LOST,
    REQUIRED_CONTINUATION_LOST,
    UNIQUE_FACT_LOST,
    UNRESOLVED_RETRY,
    review,
)


def clip(clip_id, start, end, text, *, selected, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def draft(*, selected=(), discarded=(), take_judge_groups=(), coherence=None):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=discarded,
        diagnostics={
            "take_judge_groups": list(take_judge_groups),
            "final_story_coherence_validation": coherence or {},
        },
    )


def ranked_row(clip_id, score):
    return {"clip_id": clip_id, "score": score, "reason": "watch_listen_baseline"}


def test_clean_pass_produces_pass_status_and_no_findings():
    a = clip("a", 0.0, 5.0, "the winning delivery", selected=True)
    b = clip("b", 5.0, 10.0, "a weaker retry", selected=False)
    d = draft(
        selected=(a,), discarded=(b,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9), ranked_row("b", 0.5)]}],
        coherence={"freeze_blocked": False, "lost_semantic_atoms": [], "contradiction_findings": []},
    )

    plan = build_canonical_edit_plan(d)
    result = review(plan)

    assert plan.freeze_blocked is False
    assert plan.validation_state == "frozen_ready"
    assert plan.ideas[0].coverage_status == "complete"
    assert result.status == "PASS"
    assert result.findings == ()


def test_unresolved_ambiguous_idea_yields_duplicate_idea_and_unresolved_retry_findings():
    a = clip("a", 0.0, 5.0, "take one", selected=True)
    b = clip("b", 5.0, 10.0, "take two", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.6), ranked_row("b", 0.55)]}],
        coherence={"freeze_blocked": False, "lost_semantic_atoms": [], "contradiction_findings": []},
    )

    plan = build_canonical_edit_plan(d)
    result = review(plan)

    assert plan.ideas[0].coverage_status == "unresolved_ambiguous"
    assert result.status == "FAIL"
    kinds = {f.kind for f in result.findings}
    assert kinds == {DUPLICATE_IDEA, UNRESOLVED_RETRY}
    assert all(f.owning_authority for f in result.findings)


def test_missing_idea_coverage_yields_idea_coverage_lost_finding():
    a = clip("a", 0.0, 5.0, "take one", selected=False)
    b = clip("b", 5.0, 10.0, "take two", selected=False)
    d = draft(
        discarded=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.6), ranked_row("b", 0.55)]}],
        coherence={"freeze_blocked": True, "lost_semantic_atoms": [], "contradiction_findings": []},
    )

    plan = build_canonical_edit_plan(d)
    result = review(plan)

    assert plan.ideas[0].coverage_status == "missing"
    assert result.status == "FAIL"
    assert result.findings[0].kind == IDEA_COVERAGE_LOST


def test_contradiction_and_lost_semantic_atoms_pass_through_from_coherence_diagnostics():
    a = clip("a", 0.0, 5.0, "it happened in 2019", selected=True)
    b = clip("b", 5.0, 10.0, "it happened in 2020", selected=True)
    lost = clip("lost", 10.0, 15.0, "a unique fact nobody else mentions", selected=False)
    d = draft(
        selected=(a, b), discarded=(lost,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.6), ranked_row("b", 0.6)]}],
        coherence={
            "freeze_blocked": True,
            "contradiction_findings": [{"group_id": "g1", "left_clip_id": "a", "right_clip_id": "b", "number_conflict": True}],
            "lost_semantic_atoms": [{"clip_id": "lost", "text": lost.text, "missing_critical_atoms": []}],
        },
    )

    plan = build_canonical_edit_plan(d)
    result = review(plan)

    assert plan.contradiction_findings and plan.lost_semantic_atoms
    kinds = {f.kind for f in result.findings}
    assert CONTRADICTION in kinds
    assert UNIQUE_FACT_LOST in kinds
    assert result.status == "FAIL"


def test_possible_missing_story_ending_is_a_non_blocking_warning():
    a = clip("a", 0.0, 5.0, "the only take", selected=True)
    d = draft(
        selected=(a,),
        coherence={"freeze_blocked": False, "possible_missing_story_ending": True, "lost_semantic_atoms": [], "contradiction_findings": []},
    )

    plan = build_canonical_edit_plan(d)
    result = review(plan)

    assert plan.possible_missing_story_ending is True
    assert result.status == "PASS"
    assert result.findings == ()
    assert result.warnings and result.warnings[0].kind == REQUIRED_CONTINUATION_LOST


def test_keep_sequence_and_discard_provenance_reflect_the_actual_final_draft():
    a = clip("a", 0.0, 5.0, "kept text", selected=True)
    b = clip("b", 5.0, 10.0, "discarded text", selected=False)
    d = draft(selected=(a,), discarded=(b,))

    plan = build_canonical_edit_plan(d)

    assert [c.clip_id for c in plan.keep_sequence] == ["a"]
    assert plan.keep_sequence[0].text == "kept text"
    assert [r.clip_id for r in plan.discard_provenance] == ["b"]
    assert plan.discard_provenance[0].text == "discarded text"


def test_review_never_mutates_the_plan_or_returns_a_membership_edit():
    # "Reviewer routes failure rather than directly editing membership":
    # review() takes a frozen CanonicalEditPlan and returns findings only --
    # there is no code path in it that can construct a new draft, a new
    # plan, or otherwise change what is selected/discarded. Calling it
    # twice on the same plan must be side-effect-free and idempotent.
    a = clip("a", 0.0, 5.0, "take one", selected=True)
    b = clip("b", 5.0, 10.0, "take two", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.6), ranked_row("b", 0.55)]}],
        coherence={"freeze_blocked": False, "lost_semantic_atoms": [], "contradiction_findings": []},
    )
    plan = build_canonical_edit_plan(d)

    first = review(plan)
    second = review(plan)

    assert first == second
    assert plan.ideas[0].winning_clip_ids == ("a", "b")  # untouched by review()


def test_final_edit_reviewer_has_no_detector_for_incomplete_delivery_orphan_fragment_or_incompatible_composite():
    # Honest gap, not silently-absent coverage: these three finding kinds
    # (plus STORY_ORDER_BREAK) are recognized in the vocabulary but no
    # deterministic detector exists for them anywhere in this pipeline, so
    # review() can never emit them today. This fixture is exactly the kind
    # of case a real detector would need to catch (an incomplete, clearly
    # unfinished delivery that is nonetheless the ONLY thing selected for
    # its idea) and pins that, right now, it passes silently.
    from cutsell_worker.final_edit_reviewer import _UNIMPLEMENTED_KINDS

    incomplete = DraftClip(
        clip_id="incomplete", source_asset_id="src", source_order=0,
        start=0.0, end=2.0, text="so basically what happens is",
        caption_text="so basically what happens is", selected=True,
    )
    d = draft(
        selected=(incomplete,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("incomplete", 0.9)]}],
        coherence={"freeze_blocked": False, "lost_semantic_atoms": [], "contradiction_findings": []},
    )

    plan = build_canonical_edit_plan(d)
    result = review(plan)

    assert result.status == "PASS"  # no detector exists -- this is the gap, not a pass
    assert not any(f.kind in _UNIMPLEMENTED_KINDS for f in (*result.findings, *result.warnings))
