"""D-046 FIX A -- fragment-provenance-aware CanonicalEditPlan / StoryValidator.

D-045 Case A: a retry-family winner survived Selection but was physically
split afterward (post_selection_interior_gap_trim), so its own `clip_id`
never appears in `draft.selected` (only its fragments' derived ids do).
Both `canonical_edit_plan.py`'s winning/discarded derivation and
`final_story_coherence_validation.py`'s `_missing_idea_coverage` used exact
`clip_id` equality against `draft.selected`, and so wrongly reported the
idea as vanished. The fix: recognize a selected fragment's
`parent_semantic_clip_id` (D-036's existing, general provenance link,
already used by human_boundary_polish_v5.py) as covering its original
member id -- not a Video00-specific patch.

These tests exercise the fix directly with synthetic fixtures; they do not
depend on any live pipeline run or Modal/RunPod result.
"""
from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_story_coherence_validation import (
    _missing_idea_coverage,
    apply_final_story_coherence_validation,
)


def clip(clip_id, start, end, text, *, selected, source="src", parent_semantic_clip_id=None):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
        parent_semantic_clip_id=parent_semantic_clip_id,
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


# --- 1. lone retry-family winner selected unchanged -> still winner -------

def test_unsplit_winner_still_counts_as_winning():
    a = clip("winner", 0.0, 5.0, "the winning delivery", selected=True)
    b = clip("loser", 5.0, 10.0, "a weaker retry", selected=False)
    d = draft(
        selected=(a,), discarded=(b,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("loser", 0.5)]}],
    )

    plan = build_canonical_edit_plan(d)

    assert plan.ideas[0].winning_clip_ids == ("winner",)
    assert plan.ideas[0].discarded_clip_ids == ("loser",)
    assert plan.ideas[0].coverage_status == "complete"


# --- 2. lone winner split into one child fragment -> parent still covered -

def test_winner_split_into_one_fragment_still_covers_parent_idea():
    frag = clip(
        "winner__psiglabc123", 0.0, 3.0, "the winning delivery",
        selected=True, parent_semantic_clip_id="winner",
    )
    loser = clip("loser", 5.0, 10.0, "a weaker retry", selected=False)
    d = draft(
        selected=(frag,), discarded=(loser,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("loser", 0.5)]}],
    )

    plan = build_canonical_edit_plan(d)

    assert plan.ideas[0].winning_clip_ids == ("winner",)
    assert plan.ideas[0].discarded_clip_ids == ("loser",)
    assert plan.ideas[0].coverage_status == "complete"


# --- 3. lone winner split into multiple children -> covered exactly once -

def test_winner_split_into_multiple_fragments_covers_parent_idea_once():
    left = clip("winner__psiglabc", 0.0, 3.0, "part one", selected=True, parent_semantic_clip_id="winner")
    right = clip("winner__psigrdef", 3.0, 6.0, "part two", selected=True, parent_semantic_clip_id="winner")
    loser = clip("loser", 6.0, 10.0, "a weaker retry", selected=False)
    d = draft(
        selected=(left, right), discarded=(loser,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("loser", 0.5)]}],
    )

    plan = build_canonical_edit_plan(d)

    assert plan.ideas[0].winning_clip_ids == ("winner",)
    assert plan.ideas[0].winning_clip_ids.count("winner") == 1
    assert plan.ideas[0].coverage_status == "complete"


# --- 4. discarded sibling with no descendants remains discarded -----------

def test_discarded_sibling_without_descendants_stays_discarded():
    frag = clip("winner__psiglabc", 0.0, 3.0, "winning part", selected=True, parent_semantic_clip_id="winner")
    loser = clip("loser", 5.0, 10.0, "a weaker retry", selected=False)
    d = draft(
        selected=(frag,), discarded=(loser,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("loser", 0.5)]}],
    )

    plan = build_canonical_edit_plan(d)

    assert plan.ideas[0].discarded_clip_ids == ("loser",)
    assert "loser" not in plan.ideas[0].winning_clip_ids


# --- 5. a discarded clip's own fragment must not accidentally revive it ---
#     unless that fragment is itself actually selected.

def test_fragment_of_discarded_clip_not_in_selected_does_not_revive_it():
    winner = clip("winner", 0.0, 5.0, "the winning delivery", selected=True)
    # A fragment of the LOSING clip that some hypothetical trim produced,
    # but which never made it into draft.selected (e.g. it was itself
    # judged not worth keeping) -- appears only in draft.discarded.
    loser_frag = clip(
        "loser__psiglxyz", 5.0, 7.0, "loser fragment",
        selected=False, parent_semantic_clip_id="loser",
    )
    d = draft(
        selected=(winner,), discarded=(loser_frag,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("loser", 0.5)]}],
    )

    plan = build_canonical_edit_plan(d)

    assert plan.ideas[0].winning_clip_ids == ("winner",)
    assert plan.ideas[0].discarded_clip_ids == ("loser",)


# --- 6. fragment provenance through one additional physical transform -----
#     (chained split: the fragment's parent_semantic_clip_id already points
#     at the ORIGINAL root, not an intermediate fragment id, mirroring how
#     human_boundary_polish_v5.py/post_selection_interior_gap_trim.py both
#     resolve `root_parent` before re-splitting an already-split piece).

def test_fragment_provenance_survives_a_second_chained_physical_split():
    twice_split = clip(
        "winner__psiglabc__hbp5def", 0.0, 2.0, "smallest surviving piece",
        selected=True, parent_semantic_clip_id="winner",
    )
    loser = clip("loser", 5.0, 10.0, "a weaker retry", selected=False)
    d = draft(
        selected=(twice_split,), discarded=(loser,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("loser", 0.5)]}],
    )

    plan = build_canonical_edit_plan(d)

    assert plan.ideas[0].winning_clip_ids == ("winner",)
    assert plan.ideas[0].coverage_status == "complete"


# --- 7. mixed selected original + fragment does not double-count ----------

def test_original_and_fragment_both_present_does_not_double_count():
    # Defensive/edge case: should never happen in practice (a split hook
    # always removes the pre-split clip from draft.selected), but the
    # winning tuple must still name the member id exactly once even if it.
    original_survivor = clip("winner", 0.0, 1.0, "leftover original", selected=True)
    frag = clip("winner__psiglabc", 1.0, 3.0, "fragment", selected=True, parent_semantic_clip_id="winner")
    d = draft(
        selected=(original_survivor, frag), discarded=(),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9)]}],
    )

    plan = build_canonical_edit_plan(d)

    assert plan.ideas[0].winning_clip_ids == ("winner",)
    assert plan.ideas[0].winning_clip_ids.count("winner") == 1


# --- 8. CanonicalEditPlan coverage_status stays covered --------------------

def test_coverage_status_is_complete_not_missing_after_split():
    frag_l = clip("winner__psiglabc", 0.0, 2.0, "part one", selected=True, parent_semantic_clip_id="winner")
    frag_r = clip("winner__psigrdef", 2.0, 4.0, "part two", selected=True, parent_semantic_clip_id="winner")
    other_loser = clip("loser", 5.0, 10.0, "a weaker retry", selected=False)
    d = draft(
        selected=(frag_l, frag_r), discarded=(other_loser,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.95), ranked_row("loser", 0.65)]}],
        coherence={"freeze_blocked": False, "lost_semantic_atoms": [], "contradiction_findings": []},
    )

    plan = build_canonical_edit_plan(d)

    assert plan.ideas[0].coverage_status == "complete"
    assert plan.freeze_blocked is False
    assert plan.validation_state == "frozen_ready"


# --- 9. Freeze no longer blocks on the false missing_idea_coverage --------

def test_missing_idea_coverage_is_empty_when_winner_survived_as_fragments():
    frag_l = clip("winner__psiglabc", 0.0, 2.0, "part one", selected=True, parent_semantic_clip_id="winner")
    frag_r = clip("winner__psigrdef", 2.0, 4.0, "part two", selected=True, parent_semantic_clip_id="winner")
    other_loser = clip("loser", 5.0, 10.0, "a weaker retry", selected=False)
    d = draft(
        selected=(frag_l, frag_r), discarded=(other_loser,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.95), ranked_row("loser", 0.65)]}],
    )

    assert _missing_idea_coverage(d) == []

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["missing_idea_coverage"] == []
    assert diag["freeze_blocked"] is False


# --- 10. an unrelated, genuinely-vanished idea still correctly blocks -----

def test_genuinely_missing_idea_still_blocks_freeze():
    a = clip("a", 0.0, 5.0, "take one", selected=False)
    b = clip("b", 5.0, 10.0, "take two", selected=False)
    # A different, healthy idea elsewhere in the same edit, split into
    # fragments -- present to prove the fix does not mask an UNRELATED
    # idea's real, total loss.
    frag_l = clip("winner__psiglabc", 20.0, 22.0, "part one", selected=True, parent_semantic_clip_id="winner")
    frag_r = clip("winner__psigrdef", 22.0, 24.0, "part two", selected=True, parent_semantic_clip_id="winner")
    d = draft(
        selected=(frag_l, frag_r), discarded=(a, b),
        take_judge_groups=[
            {"group_id": "g1", "ranked": [ranked_row("a", 0.60), ranked_row("b", 0.55)]},
            {"group_id": "g2", "ranked": [ranked_row("winner", 0.95)]},
        ],
    )

    missing = _missing_idea_coverage(d)
    assert missing == [{"group_id": "g1", "member_clip_ids": ["a", "b"]}]

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True

    plan = build_canonical_edit_plan(d)
    ideas_by_id = {idea.idea_id: idea for idea in plan.ideas}
    assert ideas_by_id["g1"].coverage_status == "missing"
    assert ideas_by_id["g2"].coverage_status == "complete"


# --- Regression lock: the literal D-045 Case A incident ------------------
#
# Offline replay of the exact real shape observed in the D-044 confirmatory
# Modal Video00 result (benchmark_id video00-modal-33648172326-1),
# reconstructed from the real, unmasked diagnostics fetched via the
# read-only cutsell-video00-d044-forensic-extract.yml workflow's
# trace_clip_ids output -- not a live pipeline re-run. This locks the fix
# against the exact reported incident, in addition to the generic synthetic
# fixtures above that prove the fix is general (not Video00-specific).

def test_d045_case_a_incident_no_longer_reports_missing_idea_coverage():
    real_winner_id = "clip_42b0b7919d9f9d025e86"
    real_loser_id = "clip_a6a6f4d1cffd6c94115a"
    real_group_id = "tg_28298998766ee0c8f1"
    fragment_left = clip(
        f"{real_winner_id}__psiglcad3722cd281", 95.58, 104.02,
        "Al terminar mi contrato, cambié de ginecóloga y le pedí que me hiciera un test.",
        selected=True, parent_semantic_clip_id=real_winner_id,
    )
    fragment_right = clip(
        f"{real_winner_id}__psigr1015a2ec8b00", 104.02, 107.48,
        "Ahí me mandó a hacer sonografías.",
        selected=True, parent_semantic_clip_id=real_winner_id,
    )
    loser = clip(
        real_loser_id, 82.4, 90.36,
        "Al terminar mi contrato, hablé con mi ginecóloga y le pedí todos los test que existían.",
        selected=False,
    )
    d = draft(
        selected=(fragment_left, fragment_right), discarded=(loser,),
        take_judge_groups=[{
            "group_id": real_group_id,
            "ranked": [ranked_row(real_winner_id, 0.6594), ranked_row(real_loser_id, 0.6855)],
        }],
    )

    assert _missing_idea_coverage(d) == []

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["missing_idea_coverage"] == []

    plan = build_canonical_edit_plan(d)
    idea = next(i for i in plan.ideas if i.idea_id == real_group_id)
    assert idea.coverage_status == "complete"
    assert idea.winning_clip_ids == (real_winner_id,)
    assert idea.discarded_clip_ids == (real_loser_id,)
