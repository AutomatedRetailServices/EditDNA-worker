"""run_repair_loop (D-026) -- bounded, targeted semantic repair.

Only STORY_ORDER_BREAK (composite reordering) has a real repair strategy;
see repair_loop.py's own docstring for why every other finding kind
routes straight to NEEDS_HUMAN_REVIEW by design.
"""
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.repair_loop import run_repair_loop


def clip(clip_id, start, end, text, *, selected=True, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def _composite_hybrid_editorial_chunks(clip_ids):
    return [{"hybrid_composite_best_take": {"split_group_clip_ids": list(clip_ids)}}]


def draft(*, selected, take_judge_groups, hybrid_editorial_chunks=(), coherence=None):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=(),
        diagnostics={
            "take_judge_groups": list(take_judge_groups),
            "final_story_coherence_validation": coherence or {"freeze_blocked": False, "lost_semantic_atoms": [], "contradiction_findings": []},
            "hybrid_editorial_chunks": list(hybrid_editorial_chunks),
        },
    )


def _ranked(clip_id, score=0.7):
    return {"clip_id": clip_id, "score": score, "reason": "x"}


def test_composite_out_of_order_is_repaired_without_touching_an_unrelated_idea():
    # Idea "story" (a real, unrelated, single-winner idea) sits alongside
    # the disordered composite. It must be byte-identical after repair.
    story = clip("story", 20.0, 23.0, "an unrelated independent idea", source="src")
    piece_a = clip("piece_a", 0.0, 3.0, "first half of the idea")
    piece_b = clip("piece_b", 10.0, 13.0, "second half of the idea")
    d = draft(
        selected=(story, piece_b, piece_a),  # composite rendered out of order
        take_judge_groups=[
            {"group_id": "g_story", "ranked": [_ranked("story")]},
            {"group_id": "g_composite", "ranked": [_ranked("piece_a"), _ranked("piece_b")]},
        ],
        hybrid_editorial_chunks=_composite_hybrid_editorial_chunks(["piece_a", "piece_b"]),
    )

    result = run_repair_loop(d)

    assert result.status == "PASS"
    assert len(result.attempts) == 1
    assert result.attempts[0].repaired is True
    assert result.attempts[0].finding_kind == "STORY_ORDER_BREAK"
    assert result.attempts[0].unaffected_ideas_changed is False
    # The unrelated idea's clip is untouched -- same object, same position.
    assert result.final_draft.selected[0] is story
    # The composite is now in recording order.
    assert [c.clip_id for c in result.final_draft.selected[1:]] == ["piece_a", "piece_b"]


def test_valid_composite_survives_repair_of_a_different_disordered_composite():
    good_a = clip("good_a", 0.0, 3.0, "first half of a valid composite")
    good_b = clip("good_b", 3.0, 6.0, "second half of a valid composite")
    bad_a = clip("bad_a", 30.0, 33.0, "first half of a disordered composite")
    bad_b = clip("bad_b", 40.0, 43.0, "second half of a disordered composite")
    d = draft(
        selected=(good_a, good_b, bad_b, bad_a),  # bad composite reversed
        take_judge_groups=[
            {"group_id": "g_good", "ranked": [_ranked("good_a"), _ranked("good_b")]},
            {"group_id": "g_bad", "ranked": [_ranked("bad_a"), _ranked("bad_b")]},
        ],
        hybrid_editorial_chunks=_composite_hybrid_editorial_chunks(["good_a", "good_b"])
        + _composite_hybrid_editorial_chunks(["bad_a", "bad_b"]),
    )

    result = run_repair_loop(d)

    assert result.status == "PASS"
    assert [c.clip_id for c in result.final_draft.selected] == ["good_a", "good_b", "bad_a", "bad_b"]
    # The valid composite's own clip objects were never touched.
    assert result.final_draft.selected[0] is good_a
    assert result.final_draft.selected[1] is good_b


def test_plan_version_increments_across_a_repair():
    piece_a = clip("piece_a", 0.0, 3.0, "first half")
    piece_b = clip("piece_b", 10.0, 13.0, "second half")
    d = draft(
        selected=(piece_b, piece_a),
        take_judge_groups=[{"group_id": "g1", "ranked": [_ranked("piece_a"), _ranked("piece_b")]}],
        hybrid_editorial_chunks=_composite_hybrid_editorial_chunks(["piece_a", "piece_b"]),
    )

    result = run_repair_loop(d)

    assert result.attempts[0].previous_plan_version == 1
    assert result.attempts[0].new_plan_version == 2
    assert result.final_plan.plan_version == 2


def test_semantic_hash_unchanged_by_a_pure_reorder_repair():
    # A reorder repair touches ordering only, not text/membership --
    # semantic_hash (built from a timestamp-sorted token stream) must not
    # change just because plan_version did.
    piece_a = clip("piece_a", 0.0, 3.0, "first half")
    piece_b = clip("piece_b", 10.0, 13.0, "second half")
    d = draft(
        selected=(piece_b, piece_a),
        take_judge_groups=[{"group_id": "g1", "ranked": [_ranked("piece_a"), _ranked("piece_b")]}],
        hybrid_editorial_chunks=_composite_hybrid_editorial_chunks(["piece_a", "piece_b"]),
    )

    result = run_repair_loop(d)

    assert result.final_plan.plan_version == 2
    from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
    v1_hash = build_canonical_edit_plan(d).semantic_hash
    assert result.final_plan.semantic_hash == v1_hash


def test_repeated_unresolvable_failure_terminates_safely_as_needs_human_review():
    # DUPLICATE_IDEA/UNRESOLVED_RETRY has no repair strategy -- the loop
    # must not spin, and must never claim PASS.
    a = clip("a", 0.0, 5.0, "take one")
    b = clip("b", 5.0, 10.0, "take two")
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [_ranked("a", 0.6), _ranked("b", 0.58)]}],
    )

    result = run_repair_loop(d, max_attempts=3)

    assert result.status == "NEEDS_HUMAN_REVIEW"
    assert len(result.attempts) == 1  # stops immediately, does not spin to max_attempts
    assert result.attempts[0].repaired is False
    assert result.final_review.status == "FAIL"


def test_needs_human_review_result_must_never_be_treated_as_pass():
    a = clip("a", 0.0, 5.0, "take one")
    b = clip("b", 5.0, 10.0, "take two")
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [_ranked("a", 0.6), _ranked("b", 0.58)]}],
    )

    result = run_repair_loop(d)

    assert result.status != "PASS"


def test_clean_plan_with_no_findings_passes_with_zero_repair_attempts():
    a = clip("a", 0.0, 5.0, "the only take")
    d = draft(selected=(a,), take_judge_groups=[{"group_id": "g1", "ranked": [_ranked("a")]}])

    result = run_repair_loop(d)

    assert result.status == "PASS"
    assert result.attempts == ()
