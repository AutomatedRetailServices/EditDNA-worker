"""Unit tests for D-038's CRITICAL_CLAIM_LOST chain:
final_story_coherence_validation._lost_critical_claims (per-Idea claim
coverage, scoped to that idea's OWN winning realization only) and its
wiring through canonical_edit_plan.CanonicalEditPlan.lost_critical_claims
into final_edit_reviewer.review()'s CRITICAL_CLAIM_LOST finding.

The core property under test throughout: whole-video vocabulary presence
must never falsely satisfy a different idea's missing claim -- the exact
blind spot RAW 33423953391 exposed in the older whole-KEEP-timeline
`_lost_semantic_atoms` check.
"""
from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_edit_reviewer import CRITICAL_CLAIM_LOST, review
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation


def clip(clip_id, start, end, text, *, selected, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def draft(*, selected=(), alternates=(), discarded=(), take_judge_groups=()):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=alternates, discarded=discarded,
        diagnostics={"take_judge_groups": list(take_judge_groups)},
    )


def ranked_row(clip_id, score):
    return {"clip_id": clip_id, "score": score, "reason": "watch_listen_baseline"}


DIAGNOSIS_TEXT = "The biopsy confirmed it was a benign tumor."


# --- _lost_critical_claims (via apply_final_story_coherence_validation) -----

def test_flags_critical_claim_missing_from_idea_own_winning_realization():
    winner = clip("winner", 0.0, 5.0, "So that was my experience overall.", selected=True)
    complete_loser = clip("complete_loser", 5.0, 10.0, DIAGNOSIS_TEXT, selected=False)
    d = draft(
        selected=(winner,), discarded=(complete_loser,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("complete_loser", 0.5)]}],
    )

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]

    assert diag["freeze_blocked"] is True
    assert len(diag["lost_critical_claims"]) == 1
    row = diag["lost_critical_claims"][0]
    assert row["idea_id"] == "g1"
    assert row["source_clip_id"] == "complete_loser"
    assert row["winning_clip_ids"] == ["winner"]
    assert row["blocking"] is True
    assert row["owning_authority"] == "BestTakeResolver"


def test_whole_video_vocabulary_elsewhere_does_not_falsely_satisfy_the_idea():
    # RAW 33423953391's exact blind spot: "biopsy"/"tumor" tokens recur in a
    # DIFFERENT, unrelated selected clip belonging to no idea contest here --
    # per-idea scoping must still flag the loss for g1.
    winner = clip("winner", 0.0, 5.0, "So that was my experience overall.", selected=True)
    complete_loser = clip("complete_loser", 5.0, 10.0, DIAGNOSIS_TEXT, selected=False)
    unrelated_elsewhere = clip(
        "unrelated", 20.0, 25.0,
        "We also talked about biopsy risks and tumor screening in general.",
        selected=True,
    )
    d = draft(
        selected=(winner, unrelated_elsewhere), discarded=(complete_loser,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("complete_loser", 0.5)]}],
    )

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]

    assert diag["freeze_blocked"] is True
    assert len(diag["lost_critical_claims"]) == 1
    assert diag["lost_critical_claims"][0]["idea_id"] == "g1"


def test_winner_that_actually_covers_the_claim_is_not_flagged():
    winner = clip("winner", 0.0, 5.0, DIAGNOSIS_TEXT, selected=True)
    weaker = clip("weaker", 5.0, 10.0, "It was fine I guess.", selected=False)
    d = draft(
        selected=(winner,), discarded=(weaker,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("weaker", 0.4)]}],
    )

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]

    assert diag["lost_critical_claims"] == []
    assert diag["freeze_blocked"] is False


def test_composite_winners_jointly_covering_the_claim_is_not_flagged():
    piece_a = clip("piece_a", 0.0, 5.0, "So that was my experience overall.", selected=True)
    piece_b = clip("piece_b", 5.0, 10.0, DIAGNOSIS_TEXT, selected=True)
    d = draft(
        selected=(piece_a, piece_b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("piece_a", 0.9), ranked_row("piece_b", 0.85)]}],
    )

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]

    assert diag["lost_critical_claims"] == []


def test_single_member_group_is_not_a_contest_and_is_skipped():
    a = clip("a", 0.0, 5.0, DIAGNOSIS_TEXT, selected=True)
    d = draft(
        selected=(a,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9)]}],
    )

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]

    assert diag["lost_critical_claims"] == []


def test_fully_discarded_idea_is_missing_idea_coverages_job_not_this_checks():
    a = clip("a", 0.0, 5.0, DIAGNOSIS_TEXT, selected=False)
    b = clip("b", 5.0, 10.0, "a weaker retry", selected=False)
    d = draft(
        discarded=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9), ranked_row("b", 0.4)]}],
    )

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]

    assert diag["lost_critical_claims"] == []


# --- CRITICAL_CLAIM_LOST wiring in final_edit_reviewer.review() -------------

def _plan_with_lost_critical_claim():
    a = clip("winner", 0.0, 5.0, "So that was my experience overall.", selected=True)
    b = clip("complete_loser", 5.0, 10.0, DIAGNOSIS_TEXT, selected=False)
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(a,), alternates=(), discarded=(b,),
        diagnostics={
            "take_judge_groups": [{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("complete_loser", 0.5)]}],
            "final_story_coherence_validation": {
                "freeze_blocked": True,
                "lost_semantic_atoms": [],
                "contradiction_findings": [],
                "lost_critical_claims": [{
                    "idea_id": "g1",
                    "claim_id": "claim_abc123",
                    "claim_type": "DIAGNOSIS_IDENTIFICATION",
                    "claim_text": DIAGNOSIS_TEXT,
                    "importance": "CRITICAL",
                    "source_clip_id": "complete_loser",
                    "winning_clip_ids": ["winner"],
                    "coverage_against_winning_realization": 0.0,
                    "owning_authority": "BestTakeResolver",
                    "blocking": True,
                }],
            },
        },
    )
    return build_canonical_edit_plan(d)


def test_canonical_edit_plan_carries_lost_critical_claims_through():
    plan = _plan_with_lost_critical_claim()
    assert len(plan.lost_critical_claims) == 1
    assert plan.lost_critical_claims[0]["claim_id"] == "claim_abc123"
    assert plan.freeze_blocked is True


def test_review_emits_critical_claim_lost_finding_always_blocking():
    plan = _plan_with_lost_critical_claim()
    result = review(plan)

    assert result.status == "FAIL"
    matches = [f for f in result.findings if f.kind == CRITICAL_CLAIM_LOST]
    assert len(matches) == 1
    finding = matches[0]
    assert finding.blocking is True
    assert finding.idea_id == "g1"
    assert finding.clip_ids == ("winner",)
    assert finding.owning_authority == "BestTakeResolver"
    assert finding.detail["claim_id"] == "claim_abc123"
    assert finding.detail["source_clip_id"] == "complete_loser"
    # Never routed as a warning -- always blocking by construction.
    assert not any(w.kind == CRITICAL_CLAIM_LOST for w in result.warnings)


def test_no_lost_critical_claims_means_no_such_finding():
    a = clip("a", 0.0, 5.0, "the winning delivery", selected=True)
    b = clip("b", 5.0, 10.0, "a weaker retry", selected=False)
    d = draft(
        selected=(a,), discarded=(b,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9), ranked_row("b", 0.5)]}],
    )
    out = apply_final_story_coherence_validation(d)
    plan = build_canonical_edit_plan(out)
    result = review(plan)

    assert not any(f.kind == CRITICAL_CLAIM_LOST for f in (*result.findings, *result.warnings))
