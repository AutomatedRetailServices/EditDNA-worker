"""Architecture rebalance Phase 0/1: promote the deterministic take_judge
Best-Take ranking to real authority for clear-cut retry-family contests,
without inventing any new similarity/completeness heuristic.
"""
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.deterministic_best_take_authority import (
    CLEAR_WINNER_MINIMUM_GAP,
    apply_deterministic_best_take_authority,
)


def clip(clip_id, start, end, text, *, selected):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        caption_text=text,
        selected=selected,
    )


def draft(*, selected=(), alternates=(), discarded=(), take_judge_groups=()):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id="p",
        strategy=EditStrategy.STORYTELLING,
        selected=selected,
        alternates=alternates,
        discarded=discarded,
        diagnostics={"take_judge_groups": list(take_judge_groups)},
    )


def ranked_row(clip_id, score, reason="watch_listen_baseline"):
    return {"clip_id": clip_id, "score": score, "reason": reason}


def test_no_take_judge_groups_is_a_pure_noop():
    d = draft(selected=(clip("a", 0.0, 5.0, "hello", selected=True),))
    out = apply_deterministic_best_take_authority(d)
    assert out is d


def test_clear_winner_both_wrongly_selected_gets_corrected_to_one_select_one_swap():
    # Simulates exactly the RAW #122-class bug: Unified Selection put two
    # members of one retry family into SELECT even though the deterministic
    # ranker was decisive (gap 0.94 - 0.50 = 0.44 >= 0.30).
    a = clip("winner", 0.0, 5.0, "the clean complete take", selected=True)
    b = clip("loser", 5.0, 10.0, "a weaker retry of the same idea", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("winner", 0.94), ranked_row("loser", 0.50)],
        }],
    )

    out = apply_deterministic_best_take_authority(d)

    assert [c.clip_id for c in out.selected] == ["winner"]
    assert [c.clip_id for c in out.alternates] == ["loser"]
    assert out.discarded == ()
    diag = out.diagnostics["deterministic_best_take_authority"]
    assert diag["clear_winner_minimum_gap"] == CLEAR_WINNER_MINIMUM_GAP == 0.30
    reasons = {row["clip_id"]: row["reason"] for row in diag["moves"]}
    assert reasons["loser"] == "deterministic_legitimate_alternate_not_additional_select"


def test_ambiguous_thin_gap_family_is_left_completely_untouched():
    a = clip("a", 0.0, 5.0, "take one", selected=True)
    b = clip("b", 5.0, 10.0, "take two", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("a", 0.60), ranked_row("b", 0.55)],  # gap 0.05 < 0.30
        }],
    )

    out = apply_deterministic_best_take_authority(d)

    assert out is d


def test_loser_with_failure_evidence_is_discarded_not_swapped():
    winner = clip("winner", 0.0, 5.0, "the complete clean delivery", selected=True)
    fragment = clip("fragment", 5.0, 6.0, "the com-", selected=True)
    d = draft(
        selected=(winner, fragment),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [
                ranked_row("winner", 0.90),
                ranked_row("fragment", 0.30, reason="watch_listen_baseline+material_prefix_fragment_penalty"),
            ],
        }],
    )

    out = apply_deterministic_best_take_authority(d)

    assert [c.clip_id for c in out.selected] == ["winner"]
    assert out.alternates == ()
    assert [c.clip_id for c in out.discarded] == ["fragment"]
    diag = out.diagnostics["deterministic_best_take_authority"]
    reasons = {row["clip_id"]: row["reason"] for row in diag["moves"]}
    assert reasons["fragment"] == "deterministic_failed_or_incomplete_evidence"


def test_already_discarded_loser_is_never_resurrected_to_swap():
    winner = clip("winner", 0.0, 5.0, "the complete clean delivery", selected=True)
    already_discarded = clip("already_discarded", 5.0, 6.0, "uh a broken take", selected=False)
    d = draft(
        selected=(winner,),
        discarded=(already_discarded,),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("winner", 0.90), ranked_row("already_discarded", 0.20)],
        }],
    )

    out = apply_deterministic_best_take_authority(d)

    assert [c.clip_id for c in out.selected] == ["winner"]
    assert out.alternates == ()
    assert [c.clip_id for c in out.discarded] == ["already_discarded"]
    # No move recorded for a clip that was already in its correct bucket.
    diag = out.diagnostics.get("deterministic_best_take_authority")
    if diag is not None:
        assert not any(row["clip_id"] == "already_discarded" for row in diag["moves"])


def test_winner_itself_carrying_failure_evidence_fails_open_no_changes():
    # Even a "clear" gap must not be trusted if the top scorer is itself a
    # proven fragment -- ambiguity here fails open, exactly like every other
    # authority in this codebase.
    a = clip("a", 0.0, 5.0, "the com-", selected=True)
    b = clip("b", 5.0, 10.0, "something else entirely", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [
                ranked_row("a", 0.90, reason="watch_listen_baseline+repetitive_restart_fragment_penalty"),
                ranked_row("b", 0.40),
            ],
        }],
    )

    out = apply_deterministic_best_take_authority(d)

    assert out is d


def test_singleton_group_is_never_touched():
    a = clip("a", 0.0, 5.0, "only take", selected=True)
    d = draft(
        selected=(a,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.90)]}],
    )

    out = apply_deterministic_best_take_authority(d)

    assert out is d


def test_three_way_family_keeps_exactly_one_select_rest_swap():
    a = clip("a", 0.0, 5.0, "clean winner", selected=True)
    b = clip("b", 5.0, 10.0, "usable alternate one", selected=True)
    c = clip("c", 10.0, 15.0, "usable alternate two", selected=False)
    d = draft(
        selected=(a, b),
        alternates=(c,),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("a", 0.95), ranked_row("b", 0.55), ranked_row("c", 0.50)],
        }],
    )

    out = apply_deterministic_best_take_authority(d)

    assert [clip_.clip_id for clip_ in out.selected] == ["a"]
    assert sorted(clip_.clip_id for clip_ in out.alternates) == ["b", "c"]
    assert out.discarded == ()
