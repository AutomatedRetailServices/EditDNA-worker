from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation
from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceResult,
)


def clip(clip_id, start, end, text, *, selected, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
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


class ConfirmSameIdeaArbiter:
    def __init__(self):
        self.calls = 0

    def check(self, request):
        self.calls += 1
        decisions = tuple(
            IdeaEquivalenceDecision(pair_index=i, same_idea=True, confidence=0.9, reason="same beat")
            for i in range(len(request.pairs))
        )
        return IdeaEquivalenceResult(decisions=decisions, provider="fake", model="fake", requested=True, available=True)


class DenyArbiter:
    def __init__(self):
        self.calls = 0

    def check(self, request):
        self.calls += 1
        decisions = tuple(
            IdeaEquivalenceDecision(pair_index=i, same_idea=False, confidence=0.9, reason="different beat")
            for i in range(len(request.pairs))
        )
        return IdeaEquivalenceResult(decisions=decisions, provider="fake", model="fake", requested=True, available=True)


def test_alternates_always_fold_into_discarded():
    a = clip("a", 0.0, 5.0, "kept", selected=True)
    b = clip("b", 5.0, 10.0, "not winning but not touched by any authority", selected=False)
    d = draft(selected=(a,), alternates=(b,))

    out = apply_final_story_coherence_validation(d)

    assert out.alternates == ()
    assert [c.clip_id for c in out.discarded] == ["b"]
    assert out.diagnostics["final_story_coherence_validation"]["alternates_folded_into_discard"] is True


def test_no_residual_family_is_a_clean_pass_through():
    a = clip("a", 0.0, 5.0, "only take", selected=True)
    d = draft(selected=(a,), take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("a", 0.9)]}])

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["residual_family_count"] == 0
    assert diag["resolved_family_count"] == 0
    assert diag["unresolved_family_count"] == 0


def test_residual_ambiguous_family_without_arbiter_fails_open_and_is_flagged():
    a = clip("a", 0.0, 5.0, "take one of the ambiguous idea", selected=True)
    b = clip("b", 5.0, 10.0, "take two of the ambiguous idea", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("a", 0.60), ranked_row("b", 0.55)],
        }],
    )

    out = apply_final_story_coherence_validation(d, semantic_equivalence_arbiter=None)

    assert sorted(c.clip_id for c in out.selected) == ["a", "b"]
    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["residual_family_count"] == 1
    assert diag["resolved_family_count"] == 0
    assert diag["unresolved_family_count"] == 1
    assert diag["unresolved_families"][0]["still_selected_clip_ids"] == ["a", "b"]


def test_residual_family_resolved_when_arbiter_confirms_same_idea():
    a = clip("a", 0.0, 5.0, "take one of the ambiguous idea", selected=True)
    b = clip("b", 5.0, 10.0, "take two of the ambiguous idea", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("a", 0.60), ranked_row("b", 0.55)],
        }],
    )
    arbiter = ConfirmSameIdeaArbiter()

    out = apply_final_story_coherence_validation(d, semantic_equivalence_arbiter=arbiter)

    # Higher-ranked member ("a", score 0.60) is kept; the confirmed-same-idea
    # loser is discarded, never left as an alternate/SWAP.
    assert [c.clip_id for c in out.selected] == ["a"]
    assert out.alternates == ()
    assert [c.clip_id for c in out.discarded] == ["b"]
    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["resolved_family_count"] == 1
    assert diag["unresolved_family_count"] == 0
    assert arbiter.calls == 1


def test_residual_family_left_alone_when_arbiter_denies_same_idea():
    a = clip("a", 0.0, 5.0, "take one of a genuinely different idea", selected=True)
    b = clip("b", 5.0, 10.0, "take two of a genuinely different idea", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("a", 0.60), ranked_row("b", 0.55)],
        }],
    )
    arbiter = DenyArbiter()

    out = apply_final_story_coherence_validation(d, semantic_equivalence_arbiter=arbiter)

    assert sorted(c.clip_id for c in out.selected) == ["a", "b"]
    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["unresolved_family_count"] == 1


def test_three_member_residual_family_keeps_only_top_ranked_member():
    a = clip("a", 0.0, 5.0, "take one", selected=True)
    b = clip("b", 5.0, 10.0, "take two", selected=True)
    c = clip("c", 10.0, 15.0, "take three", selected=True)
    d = draft(
        selected=(a, b, c),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("a", 0.60), ranked_row("b", 0.58), ranked_row("c", 0.55)],
        }],
    )
    arbiter = ConfirmSameIdeaArbiter()

    out = apply_final_story_coherence_validation(d, semantic_equivalence_arbiter=arbiter)

    assert [clip_.clip_id for clip_ in out.selected] == ["a"]
    assert sorted(clip_.clip_id for clip_ in out.discarded) == ["b", "c"]


def test_possible_missing_story_ending_flagged_when_last_take_discarded():
    a = clip("a", 0.0, 5.0, "opening", selected=True)
    b = clip("b", 5.0, 10.0, "closing cta", selected=False)
    d = draft(selected=(a,), discarded=(b,))

    out = apply_final_story_coherence_validation(d)

    assert out.diagnostics["final_story_coherence_validation"]["possible_missing_story_ending"] is True


def test_no_missing_story_ending_flag_when_last_take_is_selected():
    a = clip("a", 0.0, 5.0, "opening", selected=True)
    b = clip("b", 5.0, 10.0, "closing cta", selected=True)
    d = draft(selected=(a, b))

    out = apply_final_story_coherence_validation(d)

    assert out.diagnostics["final_story_coherence_validation"]["possible_missing_story_ending"] is False


# --- Contradiction invariant ---


def test_contradiction_found_blocks_freeze_and_is_not_auto_resolved():
    # Both survive Best-Take authority (e.g. authority left an ambiguous
    # family untouched) but disagree on a hard fact (an explicit negation
    # present in only one of the two) -- must never be silently resolved;
    # must set freeze_blocked.
    a = clip("a", 0.0, 5.0, "No soy la unica en mi familia con este tipo de cancer.", selected=True)
    b = clip("b", 5.0, 10.0, "Soy la unica en mi familia con este tipo de cancer.", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("a", 0.60), ranked_row("b", 0.58)],
        }],
    )

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    assert len(diag["contradiction_findings"]) == 1
    finding = diag["contradiction_findings"][0]
    assert finding["negation_conflict"] is True
    # Never auto-resolved: both remain selected, neither discarded on our say-so.
    assert sorted(c.clip_id for c in out.selected) == ["a", "b"]


def test_numeric_contradiction_between_still_selected_members_blocks_freeze():
    a = clip("a", 0.0, 5.0, "Solo un 5 por ciento de los casos son hereditarios.", selected=True)
    b = clip("b", 5.0, 10.0, "Solo un 20 por ciento de los casos son hereditarios.", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("a", 0.60), ranked_row("b", 0.58)],
        }],
    )

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    assert diag["contradiction_findings"][0]["number_conflict"] is True


def test_no_contradiction_no_freeze_block():
    a = clip("a", 0.0, 5.0, "take one of the ambiguous idea", selected=True)
    b = clip("b", 5.0, 10.0, "take two of the ambiguous idea", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("a", 0.60), ranked_row("b", 0.55)],
        }],
    )

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is False
    assert diag["contradiction_findings"] == []


def test_resolved_arbiter_confirmed_family_has_no_residual_contradiction_check():
    # Once the arbiter resolves a family to a single winner, only one member
    # remains selected -- there is no longer a still-co-selected pair to
    # even evaluate for contradiction.
    a = clip("a", 0.0, 5.0, "take one of the ambiguous idea", selected=True)
    b = clip("b", 5.0, 10.0, "take two of the ambiguous idea", selected=True)
    d = draft(
        selected=(a, b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("a", 0.60), ranked_row("b", 0.55)],
        }],
    )
    arbiter = ConfirmSameIdeaArbiter()

    out = apply_final_story_coherence_validation(d, semantic_equivalence_arbiter=arbiter)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is False
    assert len(out.selected) == 1


# --- Idea coverage invariant ---


def test_missing_idea_coverage_blocks_freeze_when_whole_family_discarded():
    a = clip("a", 0.0, 5.0, "take one", selected=False)
    b = clip("b", 5.0, 10.0, "take two", selected=False)
    d = draft(
        discarded=(a, b),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("a", 0.60), ranked_row("b", 0.55)],
        }],
    )

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    assert diag["missing_idea_coverage"] == [{"group_id": "g1", "member_clip_ids": ["a", "b"]}]


def test_idea_coverage_fine_when_one_member_still_selected():
    a = clip("a", 0.0, 5.0, "take one", selected=True)
    b = clip("b", 5.0, 10.0, "take two", selected=False)
    d = draft(
        selected=(a,), discarded=(b,),
        take_judge_groups=[{
            "group_id": "g1",
            "ranked": [ranked_row("a", 0.90), ranked_row("b", 0.50)],
        }],
    )

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is False
    assert diag["missing_idea_coverage"] == []


# --- Lost semantic atoms: general coverage ledger vs. the ACTUAL final KEEP
# timeline, independent of take_judge_groups. Motivated by RAW 33345946000:
# hybrid_session_cleanup deleted candidates before IdeaClusterer/grouping
# ever ran, so they never entered any take_judge_groups entry and
# _missing_idea_coverage reported nothing missing despite real content loss.


def test_lost_semantic_atoms_blocks_freeze_for_content_discarded_before_any_grouping():
    # No take_judge_groups entry at all for this clip -- simulates a clip
    # deleted upstream (e.g. by hybrid_session_cleanup) before grouping ever
    # ran, which _missing_idea_coverage cannot see by construction.
    kept = clip("kept", 0.0, 5.0, "the gynecologist sent me for other tests", selected=True)
    lost = clip(
        "lost", 5.0, 10.0,
        "the biopsy confirmed it was papillary thyroid cancer",
        selected=False,
    )
    d = draft(selected=(kept,), discarded=(lost,))

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    finding_ids = [row["clip_id"] for row in diag["lost_semantic_atoms"]]
    assert finding_ids == ["lost"]


def test_lost_semantic_atoms_does_not_flag_correctly_discarded_redundant_retry():
    # A genuine weaker retry of the SAME idea shares most of its topic
    # vocabulary with the surviving winner even when reworded -- this must
    # not be mistaken for information loss.
    kept = clip("kept", 0.0, 5.0, "the serum cleared my skin in two weeks", selected=True)
    redundant = clip(
        "redundant", 5.0, 10.0,
        "so the serum basically cleared my skin after two weeks",
        selected=False,
    )
    d = draft(selected=(kept,), discarded=(redundant,))

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is False
    assert diag["lost_semantic_atoms"] == []


def test_lost_semantic_atoms_flags_missing_numeric_fact_despite_high_overlap():
    # High lexical overlap overall, but the discarded clip carries a number
    # never repeated anywhere in the final KEEP text -- a critical fact, not
    # just a phrasing difference, so it is flagged unconditionally even
    # though the broader content-coverage floor alone would not catch it.
    kept = clip("kept", 0.0, 5.0, "i had surgery to remove it", selected=True)
    unique_fact = clip(
        "unique_fact", 5.0, 10.0,
        "i had surgery to remove it, my 3rd surgery that year",
        selected=False,
    )
    d = draft(selected=(kept,), discarded=(unique_fact,))

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    finding = diag["lost_semantic_atoms"][0]
    assert finding["clip_id"] == "unique_fact"
    assert finding["missing_critical_atoms"]


def test_lost_semantic_atoms_does_not_block_freeze_for_a_contextual_incidental_year():
    # D-031 / RAW 33402023395: an incidental year in an ordinary temporal
    # aside, on a discarded clip whose core claim (diagnosis + treatment) is
    # otherwise fully present in the surviving winner, must not block
    # Freeze by itself -- the Human Gold oracle itself does not preserve
    # that year either. Generic subject matter, no Video00 wording.
    kept = clip(
        "kept", 0.0, 5.0,
        "I had digestion problems and it turned out to be gastritis, nothing severe.",
        selected=True,
    )
    # Near-identical wording to `kept` plus one leading incidental-year
    # aside -- isolates the atom-level check from the separate, coarser
    # content-vocabulary floor (which needs real divergence to fire).
    lost_year = clip(
        "lost_year", 5.0, 10.0,
        "During one period in 2023 I had digestion problems and it turned out to be gastritis, nothing severe.",
        selected=False,
    )
    d = draft(selected=(kept,), discarded=(lost_year,))

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is False
    finding = diag["lost_semantic_atoms"][0]
    assert finding["blocking"] is False
    classes = {c["importance"] for c in finding["atom_classifications"]}
    assert classes == {"CONTEXTUAL"}


def test_lost_semantic_atoms_still_blocks_freeze_for_a_critical_measurement():
    # A measurement atom lost alongside an incidental year must still block
    # -- CONTEXTUAL-only tolerance never weakens a genuinely critical atom.
    kept = clip("kept", 0.0, 5.0, "they found something during the scan", selected=True)
    lost = clip(
        "lost", 5.0, 10.0,
        "during a scan in 2023 they found a lump measuring 3 centimeters",
        selected=False,
    )
    d = draft(selected=(kept,), discarded=(lost,))

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    finding = diag["lost_semantic_atoms"][0]
    assert finding["blocking"] is True
    classes = {c["importance"] for c in finding["atom_classifications"]}
    assert "CRITICAL" in classes


def test_lost_semantic_atoms_ignores_short_filler_discard():
    # Too short to safely judge as carrying a distinct idea/fact -- avoids
    # flagging BTS/false-start scraps that legitimately have near-zero
    # overlap with the eventual complete delivery.
    kept = clip("kept", 0.0, 5.0, "the serum cleared my skin in two weeks", selected=True)
    filler = clip("filler", 5.0, 6.0, "okay wait let me redo that", selected=False)
    d = draft(selected=(kept,), discarded=(filler,))

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is False
    assert diag["lost_semantic_atoms"] == []


# --- Composite preserves all unique semantic atoms: the coverage ledger
# checks the UNION of the final selected text, so a two-piece composite
# (both pieces individually selected by CompositeResolver) that jointly
# covers a discarded delivery's facts must not be flagged, while one that
# genuinely drops a fact still must be -- exactly the "composite preserves
# all unique semantic atoms" CleanCutBench category, at the level this
# repo's coverage ledger can actually check (composite mechanics themselves
# are covered separately in tests/test_cutsell_hybrid_composite_best_take.py,
# which cannot be exercised in isolation through this module -- see
# tests/test_cutsell_clean_cut_core_evaluation_suite.py's own scoping note).


def test_two_piece_composite_jointly_covering_a_discarded_facts_is_not_flagged():
    piece_a = clip("piece_a", 0.0, 3.0, "i had thyroid surgery last year", selected=True)
    piece_b = clip("piece_b", 3.0, 6.0, "it was my 3rd surgery and it went well", selected=True)
    superseded = clip(
        "superseded", 6.0, 10.0,
        "i had thyroid surgery last year, my 3rd surgery, and it went well",
        selected=False,
    )
    d = draft(selected=(piece_a, piece_b), discarded=(superseded,))

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is False
    assert diag["lost_semantic_atoms"] == []


def test_two_piece_composite_missing_a_fact_the_superseded_delivery_had_is_flagged():
    piece_a = clip("piece_a", 0.0, 3.0, "i had thyroid surgery last year", selected=True)
    piece_b = clip("piece_b", 3.0, 6.0, "and it went well overall", selected=True)
    superseded = clip(
        "superseded", 6.0, 10.0,
        "i had thyroid surgery last year, my 3rd surgery, and it went well overall",
        selected=False,
    )
    d = draft(selected=(piece_a, piece_b), discarded=(superseded,))

    out = apply_final_story_coherence_validation(d)

    diag = out.diagnostics["final_story_coherence_validation"]
    assert diag["freeze_blocked"] is True
    finding = diag["lost_semantic_atoms"][0]
    assert finding["clip_id"] == "superseded"
    assert "3rd" in finding["missing_critical_atoms"]
