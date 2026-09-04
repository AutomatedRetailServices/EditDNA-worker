"""Unit tests for claim_coverage_best_take.py -- D-038.

Mirrors the fixture style of test_cutsell_deterministic_best_take_
authority.py (same take_judge_groups-driven contract). No Video00-specific
fact is referenced -- these are generic sentences exercising the same
claim-coverage-override shapes RAW 33423953391 exposed.
"""
from cutsell_worker.claim_coverage_best_take import apply_claim_coverage_best_take
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION


def clip(clip_id, start, end, text, *, selected, source="src", complete_idea=None):
    return DraftClip(
        clip_id=clip_id,
        source_asset_id=source,
        source_order=0,
        start=start,
        end=end,
        text=text,
        caption_text=text,
        selected=selected,
        complete_idea=complete_idea,
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
    d = draft(selected=(clip("a", 0.0, 5.0, "hello there friend", selected=True),))
    out = apply_claim_coverage_best_take(d)
    assert out is d


def test_single_member_group_is_skipped():
    a = clip("a", 0.0, 5.0, "The biopsy confirmed it was a benign tumor.", selected=True)
    d = draft(selected=(a,), take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9)]}])
    out = apply_claim_coverage_best_take(d)
    assert out is d


def test_ambiguous_two_current_winners_resolved_by_critical_coverage_dominance():
    # D-063: `a` strictly dominates `b` on CRITICAL-claim coverage (covers
    # `b`'s ENTIRE content -- b has no critical claim of its own at all --
    # plus the diagnosis fact `b` lacks). Before D-063 this was left
    # untouched as "not this module's job"; now the dominant candidate
    # becomes the family's sole winner instead of falling through to
    # DUPLICATE_IDEA+UNRESOLVED_RETRY.
    a = clip("a", 0.0, 5.0, "The biopsy confirmed it was a benign tumor.", selected=True)
    b = clip("b", 5.0, 10.0, "It was just a routine checkup.", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9), ranked_row("b", 0.85)]}],
    )
    out = apply_claim_coverage_best_take(d)

    assert [c.clip_id for c in out.selected] == ["a"]
    assert [c.clip_id for c in out.discarded] == ["b"]
    diag = out.diagnostics["claim_coverage_best_take"]
    assert len(diag["dominance_resolutions"]) == 1
    resolution = diag["dominance_resolutions"][0]
    assert resolution["group_id"] == "g1"
    assert resolution["winner_clip_id"] == "a"
    assert resolution["discarded_clip_ids"] == ["b"]
    assert resolution["reason"] == "critical_coverage_dominance"


def test_dominance_ignores_ranked_delivery_score_ordering():
    # D-063 test matrix item 2: `b` has the HIGHER ranked/delivery score,
    # but `a` still wins on CRITICAL_COVERAGE_DOMINANCE -- this module
    # never consults score for this branch, only claim coverage.
    a = clip("a", 0.0, 5.0, "The biopsy confirmed it was a benign tumor.", selected=True)
    b = clip("b", 5.0, 10.0, "It was just a routine checkup.", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("b", 0.95), ranked_row("a", 0.4)]}],
    )
    out = apply_claim_coverage_best_take(d)

    assert [c.clip_id for c in out.selected] == ["a"]
    assert [c.clip_id for c in out.discarded] == ["b"]


def test_identical_critical_coverage_is_not_dominance_left_untouched():
    # D-063 test matrix item 3: both candidates cover the exact same
    # CRITICAL claim -- a proper superset can never hold between equal
    # sets, so no forced pick here; delivery may decide it elsewhere, this
    # module leaves the family exactly as it was.
    a = clip("a", 0.0, 5.0, "The biopsy confirmed it was a benign tumor.", selected=True)
    b = clip("b", 5.0, 10.0, "The biopsy confirmed it was a benign tumor, cleanly delivered.", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9), ranked_row("b", 0.85)]}],
    )
    out = apply_claim_coverage_best_take(d)
    assert out is d


def test_extra_contextual_only_claim_does_not_dominate():
    # D-063 test matrix item 4: `a` has an extra sentence, but it is
    # SUPPORTING/CONTEXTUAL importance (an incidental year), not CRITICAL --
    # coverage over CRITICAL claims alone is identical between `a` and `b`,
    # so this must NOT be treated as dominance.
    a = clip(
        "a", 0.0, 5.0,
        "The biopsy confirmed it was a benign tumor. It happened back in 2019.",
        selected=True,
    )
    b = clip("b", 5.0, 10.0, "The biopsy confirmed it was a benign tumor.", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9), ranked_row("b", 0.85)]}],
    )
    out = apply_claim_coverage_best_take(d)
    assert out is d


def test_extra_critical_claim_but_genuine_contradiction_blocks_dominance():
    # D-063 test matrix item 5: `a` technically covers an extra CRITICAL
    # claim (5 percent vs 10 percent), but the two candidates directly
    # contradict each other on it -- safety must never be overridden by
    # coverage dominance (module docstring / Section 3).
    a = clip(
        "a", 0.0, 5.0,
        "The biopsy confirmed it was a benign tumor. Only 5 percent of cases are hereditary.",
        selected=True,
    )
    b = clip(
        "b", 5.0, 10.0,
        "The biopsy confirmed it was a benign tumor. Only 10 percent of cases are hereditary.",
        selected=True,
    )
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9), ranked_row("b", 0.85)]}],
    )
    out = apply_claim_coverage_best_take(d)
    assert out is d


def test_extra_critical_claim_but_incomplete_delivery_is_a_safe_fallback():
    # D-063 test matrix item 6: `a` structurally dominates `b` on coverage,
    # but `a` is itself a proven-incomplete/failed delivery -- do not
    # blindly prefer a richer candidate that cannot actually deliver those
    # claims usably. Left untouched (existing completeness evidence wins).
    a = clip("a", 0.0, 5.0, "The biopsy confirmed it was a benign tumor.", selected=True, complete_idea=False)
    b = clip("b", 5.0, 10.0, "It was just a routine checkup.", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9), ranked_row("b", 0.85)]}],
    )
    out = apply_claim_coverage_best_take(d)
    assert out is d


def test_disjoint_unique_critical_claims_no_dominance_stays_review_required():
    # D-063 test matrix items 7 & 9: each candidate carries a DIFFERENT
    # critical claim the other entirely lacks -- neither is a superset of
    # the other, so there is no dominance either way. This is a genuine
    # ambiguity: left exactly as it was, still falling through to
    # StoryValidator/FinalEditReviewer's DUPLICATE_IDEA+UNRESOLVED_RETRY
    # block (this codebase's production analog of REVIEW_REQUIRED).
    a = clip("a", 0.0, 5.0, "The biopsy confirmed it was a benign tumor.", selected=True)
    b = clip("b", 5.0, 10.0, "Only 5 percent of patients ever see this reaction.", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9), ranked_row("b", 0.85)]}],
    )
    out = apply_claim_coverage_best_take(d)
    assert out is d


def test_negation_contradiction_blocks_dominance_even_with_coverage_gap():
    # D-063 test matrix item 10: semantic-winner-style evidence alone
    # (whichever candidate "looks" more complete) must never override a
    # genuine safety conflict -- here a negation-polarity contradiction --
    # same posture as the numeric case above, different conflict shape.
    a = clip(
        "a", 0.0, 5.0,
        "The biopsy confirmed it was a benign tumor. She is not the only one in her family with this.",
        selected=True,
    )
    b = clip(
        "b", 5.0, 10.0,
        "The biopsy confirmed it was a benign tumor. She is the only one in her family with this.",
        selected=True,
    )
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9), ranked_row("b", 0.85)]}],
    )
    out = apply_claim_coverage_best_take(d)
    assert out is d


def test_d0621_generic_regression_fixture_auto_resolves_via_dominance():
    # D-063 test matrix item 11: the D-062.1 forensic's generic shape --
    # candidate A carries a distinct critical fact PLUS the shared
    # reflective statement both sides carry; candidate B carries only the
    # shared reflective statement. No Video00-specific text/ids -- this is
    # the general pattern D-062.1 identified, not the literal fixture.
    a = clip(
        "a", 0.0, 5.0,
        "The biopsy confirmed it was a benign tumor. "
        "Looking back, I am really glad we caught it early.",
        selected=True,
    )
    b = clip(
        "b", 5.0, 10.0,
        "Looking back, I am really glad we caught it early.",
        selected=True,
    )
    d = draft(
        selected=(a, b),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9), ranked_row("b", 0.88)]}],
    )
    out = apply_claim_coverage_best_take(d)

    assert [c.clip_id for c in out.selected] == ["a"]
    assert [c.clip_id for c in out.discarded] == ["b"]
    diag = out.diagnostics["claim_coverage_best_take"]
    assert diag["dominance_resolutions"][0]["winner_clip_id"] == "a"


def test_no_critical_claims_in_group_is_untouched():
    a = clip("a", 0.0, 5.0, "I walked into the room and sat down.", selected=True)
    b = clip("b", 5.0, 10.0, "I stood by the door for a while.", selected=False)
    d = draft(
        selected=(a,), discarded=(b,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9), ranked_row("b", 0.4)]}],
    )
    out = apply_claim_coverage_best_take(d)
    assert out is d


def test_winner_already_covers_every_critical_claim_is_untouched():
    a = clip("a", 0.0, 5.0, "The biopsy confirmed it was a benign tumor.", selected=True)
    b = clip("b", 5.0, 10.0, "The biopsy confirmed it was a benign tumor, cleanly delivered.", selected=False)
    d = draft(
        selected=(a,), discarded=(b,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9), ranked_row("b", 0.6)]}],
    )
    out = apply_claim_coverage_best_take(d)
    assert out is d


def test_single_candidate_full_coverage_overrides_incomplete_winner():
    # Exactly the RAW 33423953391 shape: the current winner is the cleaner
    # take but drops the diagnosis claim; a discarded sibling has it intact.
    winner = clip("winner", 0.0, 5.0, "So that was my experience at the clinic.", selected=True)
    complete = clip("complete", 5.0, 10.0, "The biopsy confirmed it was a benign tumor.", selected=False)
    d = draft(
        selected=(winner,), discarded=(complete,),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("winner", 0.9), ranked_row("complete", 0.5)],
        }],
    )
    out = apply_claim_coverage_best_take(d)

    assert [c.clip_id for c in out.selected] == ["complete"]
    assert [c.clip_id for c in out.discarded] == ["winner"]
    diag = out.diagnostics["claim_coverage_best_take"]
    assert diag["status"] == "applied"
    assert len(diag["overrides"]) == 1
    override = diag["overrides"][0]
    assert override["group_id"] == "g1"
    assert override["previous_winner_clip_id"] == "winner"
    assert override["new_winner_clip_id"] == "complete"


def test_two_piece_composite_when_claims_are_complementary_and_time_compatible():
    winner = clip("winner", 0.0, 5.0, "So that was my experience overall.", selected=True)
    piece_a = clip("piece_a", 5.0, 10.0, "The biopsy confirmed it was a benign tumor.", selected=False)
    piece_b = clip("piece_b", 10.0, 15.0, "Only 5 percent of patients ever see this.", selected=False)
    d = draft(
        selected=(winner,), discarded=(piece_a, piece_b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("winner", 0.9), ranked_row("piece_a", 0.6), ranked_row("piece_b", 0.55)],
        }],
    )
    out = apply_claim_coverage_best_take(d)

    assert [c.clip_id for c in out.selected] == ["piece_a", "piece_b"]
    assert [c.clip_id for c in out.discarded] == ["winner"]
    diag = out.diagnostics["claim_coverage_best_take"]
    assert len(diag["composites"]) == 1
    assert diag["composites"][0]["clip_ids"] == ["piece_a", "piece_b"]
    assert diag["composites"][0]["reason"] == "claim_coverage_complementary"


def test_composite_pieces_kept_in_recording_time_order_regardless_of_ranked_order():
    winner = clip("winner", 0.0, 5.0, "So that was my experience overall.", selected=True)
    # piece_b was RECORDED first (earlier start) but ranked second.
    piece_b = clip("piece_b", 5.0, 10.0, "Only 5 percent of patients ever see this.", selected=False)
    piece_a = clip("piece_a", 10.0, 15.0, "The biopsy confirmed it was a benign tumor.", selected=False)
    d = draft(
        selected=(winner,), discarded=(piece_a, piece_b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("winner", 0.9), ranked_row("piece_a", 0.6), ranked_row("piece_b", 0.55)],
        }],
    )
    out = apply_claim_coverage_best_take(d)
    assert [c.clip_id for c in out.selected] == ["piece_b", "piece_a"]


def test_composite_skipped_when_candidates_overlap_in_time():
    winner = clip("winner", 0.0, 5.0, "So that was my experience overall.", selected=True)
    # These two overlap (piece_b starts before piece_a ends) -- not safe to
    # place side by side, so this must fall through to unresolved_gaps.
    piece_a = clip("piece_a", 5.0, 12.0, "The biopsy confirmed it was a benign tumor.", selected=False)
    piece_b = clip("piece_b", 10.0, 15.0, "Only 5 percent of patients ever see this.", selected=False)
    d = draft(
        selected=(winner,), discarded=(piece_a, piece_b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("winner", 0.9), ranked_row("piece_a", 0.6), ranked_row("piece_b", 0.55)],
        }],
    )
    out = apply_claim_coverage_best_take(d)

    assert [c.clip_id for c in out.selected] == ["winner"]
    diag = out.diagnostics["claim_coverage_best_take"]
    assert diag["composites"] == []
    assert len(diag["unresolved_gaps"]) == 1
    assert diag["unresolved_gaps"][0]["winner_clip_id"] == "winner"


def test_composite_skipped_when_candidates_are_from_different_sources():
    winner = clip("winner", 0.0, 5.0, "So that was my experience overall.", selected=True)
    piece_a = clip("piece_a", 5.0, 10.0, "The biopsy confirmed it was a benign tumor.", selected=False, source="camA")
    piece_b = clip("piece_b", 10.0, 15.0, "Only 5 percent of patients ever see this.", selected=False, source="camB")
    d = draft(
        selected=(winner,), discarded=(piece_a, piece_b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("winner", 0.9), ranked_row("piece_a", 0.6), ranked_row("piece_b", 0.55)],
        }],
    )
    out = apply_claim_coverage_best_take(d)

    assert [c.clip_id for c in out.selected] == ["winner"]
    diag = out.diagnostics["claim_coverage_best_take"]
    assert diag["composites"] == []
    assert len(diag["unresolved_gaps"]) == 1


def test_composite_skipped_when_unique_contributions_share_a_claim_type():
    # Two NEGATION claims worded differently are more likely one idea's
    # coarse-classifier-split paraphrase than genuinely complementary facts
    # -- must fall through to unresolved_gaps rather than a false composite
    # (this is the exact shape of a real regression this guard was added
    # for: a multilingual paraphrased retry, both sides negating the same
    # thing in different words, was being frozen as a fake composite
    # instead of correctly collapsing to one winner upstream).
    winner = clip("winner", 0.0, 5.0, "So that's been my take on it overall.", selected=True)
    a = clip("a", 5.0, 10.0, "We never thought to get it checked at all.", selected=False)
    b = clip("b", 10.0, 15.0, "It did not occur to us to have it examined.", selected=False)
    d = draft(
        selected=(winner,), discarded=(a, b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("winner", 0.9), ranked_row("a", 0.6), ranked_row("b", 0.55)],
        }],
    )
    out = apply_claim_coverage_best_take(d)

    assert [c.clip_id for c in out.selected] == ["winner"]
    diag = out.diagnostics["claim_coverage_best_take"]
    assert diag["composites"] == []
    assert len(diag["unresolved_gaps"]) == 1


def test_unresolved_gap_when_no_single_or_pair_covers_everything():
    winner = clip("winner", 0.0, 5.0, "So that was my experience overall.", selected=True)
    # Three claims split across sibling members such that no single member
    # or pair jointly covers the full set -- left exactly as upstream
    # decided, flagged for observability only.
    a = clip("a", 5.0, 10.0, "The biopsy confirmed it was a benign tumor.", selected=False)
    b = clip("b", 10.0, 15.0, "The test did not show any infection at all.", selected=False)
    c = clip("c", 15.0, 20.0, "Only 5 percent of patients ever see this reaction.", selected=False)
    d = draft(
        selected=(winner,), discarded=(a, b, c),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("winner", 0.9), ranked_row("a", 0.6), ranked_row("b", 0.55), ranked_row("c", 0.5)],
        }],
    )
    out = apply_claim_coverage_best_take(d)

    assert [c2.clip_id for c2 in out.selected] == ["winner"]
    diag = out.diagnostics["claim_coverage_best_take"]
    assert diag["overrides"] == []
    assert diag["composites"] == []
    assert len(diag["unresolved_gaps"]) == 1


def test_move_only_ever_produces_select_or_discard_never_swap():
    # D-019: no SWAP in this module's own bucket moves either.
    winner = clip("winner", 0.0, 5.0, "So that was my experience overall.", selected=True)
    complete = clip("complete", 5.0, 10.0, "The biopsy confirmed it was a benign tumor.", selected=False)
    d = draft(
        selected=(winner,), discarded=(complete,),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("winner", 0.9), ranked_row("complete", 0.5)],
        }],
    )
    out = apply_claim_coverage_best_take(d)
    assert out.alternates == ()
    assert all(c.selected for c in out.selected)
    assert not any(c.selected for c in out.discarded)


def test_multiple_groups_independently_evaluated():
    winner1 = clip("winner1", 0.0, 5.0, "So that was my experience overall.", selected=True)
    complete1 = clip("complete1", 5.0, 10.0, "The biopsy confirmed it was a benign tumor.", selected=False)
    winner2 = clip("winner2", 20.0, 25.0, "clean unrelated content", selected=True)
    loser2 = clip("loser2", 25.0, 30.0, "a weaker retry", selected=False)
    d = draft(
        selected=(winner1, winner2), discarded=(complete1, loser2),
        take_judge_groups=[
            {"group_id": "g1", "ranked": [ranked_row("winner1", 0.9), ranked_row("complete1", 0.5)]},
            {"group_id": "g2", "ranked": [ranked_row("winner2", 0.9), ranked_row("loser2", 0.4)]},
        ],
    )
    out = apply_claim_coverage_best_take(d)
    selected_ids = {c.clip_id for c in out.selected}
    assert selected_ids == {"complete1", "winner2"}


# --- D-061 Phase 2: this module is the SECOND consumer of claim_equivalence_
#     arbiter (alongside StoryValidator's _lost_critical_claims) that D-061's
#     BrainRuntime/universal_clean_cut_validation.py wiring activates for the
#     first time in production -- both share the one wired instance. No new
#     heuristic here: `_covered_claim_ids` already called `resolve_ambiguous_
#     coverage` with whatever arbiter it was given; it was simply always
#     None before D-061. -------------------------------------------------

def test_paraphrased_winner_claim_without_arbiter_flags_unresolved_gap():
    """Unchanged, pre-D-061 behavior: a winner whose own paraphrase of a
    critical claim lands in the ambiguous coverage band, with no arbiter
    available, is NOT credited -- fails open exactly as before, surfacing
    an unresolved_gap (which StoryValidator's own, always-on backstop would
    also independently catch as CRITICAL_CLAIM_LOST)."""
    winner = clip("winner", 0.0, 5.0, "About 5 to 10 percent of cancers are hereditary, according to her doctor.", selected=True)
    complete = clip("complete", 5.0, 10.0, "Only 5 to 10 percent of these cases are hereditary in nature.", selected=False)
    d = draft(
        selected=(winner,), discarded=(complete,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("complete", 0.5)]}],
    )
    out = apply_claim_coverage_best_take(d)

    assert [c.clip_id for c in out.selected] == ["winner"]  # no override -- no safe single/pair fix
    diag = out.diagnostics["claim_coverage_best_take"]
    assert diag["overrides"] == []
    assert len(diag["unresolved_gaps"]) == 1


def test_paraphrased_winner_claim_with_wired_arbiter_recognizes_coverage():
    """New, D-061 behavior: the SAME ambiguous-band paraphrase, now with a
    wired arbiter confirming it, is correctly recognized as already covered
    by the winner -- no override needed, no unresolved gap raised. Safer
    than before, not less safe: catching this earlier (at BestTake) means
    StoryValidator never even needs to flag it downstream."""
    winner = clip("winner", 0.0, 5.0, "About 5 to 10 percent of cancers are hereditary, according to her doctor.", selected=True)
    complete = clip("complete", 5.0, 10.0, "Only 5 to 10 percent of these cases are hereditary in nature.", selected=False)
    d = draft(
        selected=(winner,), discarded=(complete,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("complete", 0.5)]}],
    )

    class _AlwaysCoveredArbiter:
        def claim_covered(self, claim_text, winning_realization_text):
            return True, 0.9, "paraphrase confirmed"

    out = apply_claim_coverage_best_take(d, claim_equivalence_arbiter=_AlwaysCoveredArbiter())

    assert [c.clip_id for c in out.selected] == ["winner"]
    # A true no-op: nothing needed fixing, so no diagnostics key is even
    # written (mirrors test_winner_already_covers_every_critical_claim_is_
    # untouched's own assertion shape for the "nothing to do" case).
    assert out is d
