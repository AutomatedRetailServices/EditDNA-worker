"""D-061 Phase 1 -- SAME-IDEA PARAPHRASE CREDIT for _lost_semantic_atoms.

Root defect (docs/CUTSELL_DECISIONS.md D-060's forensic): `_lost_semantic_
atoms` is deliberately whole-video/bag-of-words/idea-membership-blind, which
is correct for its original purpose (catching content deleted BEFORE
grouping ever saw it) but makes it flag a discarded clip as UNIQUE_FACT_LOST
even when a SELECTED member of its own retry family already carries the
identical proposition in different words -- the live D-060 shape: two clips
merged by the semantic-equivalence arbiter at 0.9 confidence shared almost
no raw vocabulary, so the discarded one scored only 0.4 whole-video coverage
and blocked Freeze for content that was never actually lost.

Fixed by `_same_idea_paraphrase_credit`: a POST-GROUPING discarded clip (a
member of a genuine 2+-member take_judge_groups retry contest) whose own
family has a SELECTED sibling is credited as covered only when an existing
`semantic_idea_equivalence` merge record for that exact pair already
confirms it at high confidence -- no new paid call, reusing evidence
already computed during grouping. See that function's module comment for
why a deterministic idea-scoped word-overlap fallback is deliberately NOT
also implemented: it is mathematically inert here (idea-scoped coverage can
never exceed whole-video coverage, since the winner's own text is already
part of the whole-video comparison), so a genuine near-duplicate is already
handled correctly by the existing, unmodified whole-video check with no fix
needed -- it never even reaches this credit path in the first place. A
genuinely additive fact, a different idea, or a clip that never reached
grouping at all (pre-group reject / hybrid delete with no verified
replacement) is unaffected -- current fail-closed behavior is unchanged for
all of those.

Entirely generic -- no Video00 clip ids or phrases.
"""
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation


def clip(clip_id, start, end, text, *, selected, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def ranked_row(clip_id, score):
    return {"clip_id": clip_id, "score": score, "reason": "watch_listen_baseline"}


def draft(*, selected, discarded, take_judge_groups=(), merges=()):
    diagnostics = {"take_judge_groups": list(take_judge_groups)}
    if merges:
        diagnostics["semantic_idea_equivalence"] = {"status": "applied", "merges": list(merges)}
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=discarded, diagnostics=diagnostics,
    )


def _row_for(diag, clip_id):
    return next((r for r in diag["lost_semantic_atoms"] if r["clip_id"] == clip_id), None)


# --- 1. Same idea + explicit high-confidence semantic-equivalence evidence -
#        (the exact D-060 live shape: near-zero raw overlap) -> NOT blocking

def test_same_idea_explicit_semantic_equivalence_evidence_not_blocking():
    winner = clip(
        "winner", 0.0, 3.0,
        "My skin looked fine to me back then, but pictures from that year tell a different story.",
        selected=True,
    )
    discard = clip(
        "discard", 3.5, 6.0,
        "Photos from around that time show something I completely missed while it was happening.",
        selected=False,
    )
    d = draft(
        selected=(winner,), discarded=(discard,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.8), ranked_row("discard", 0.6)]}],
        merges=[{"left_clip_id": "discard", "right_clip_id": "winner", "confidence": 0.9,
                 "reason": "Both reflect on unrecognized signs viewed in retrospect."}],
    )

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    row = _row_for(diag, "discard")

    assert row is not None
    assert row["blocking"] is False
    assert row["classification"] == "SEMANTICALLY_COVERED_BY_SELECTED_REALIZATION"
    assert row["content_loss_suppressed_by"] == "same_idea_semantic_equivalence"
    assert diag["lost_critical_claims"] == []
    assert diag["freeze_blocked"] is False


# --- 2. Same idea but genuinely additive fact -> still blocking ------------
#        (no semantic_idea_equivalence merge record for this pair exists --
#        there is nothing to reuse, and the additional content is real)

def test_same_idea_genuinely_additive_fact_still_blocking():
    winner = clip(
        "winner", 0.0, 3.0,
        "I switched to a gentler cleanser for my face after a few rough weeks.",
        selected=True,
    )
    discard = clip(
        "discard", 3.5, 6.0,
        "I switched to a gentler cleanser for my face and started a daily stretching routine for my knees.",
        selected=False,
    )
    d = draft(
        selected=(winner,), discarded=(discard,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.8), ranked_row("discard", 0.6)]}],
    )

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    row = _row_for(diag, "discard")

    assert row is not None
    assert row["blocking"] is True
    assert row["classification"] == "REAL_CONTENT_LOSS"
    assert "content_loss_suppressed_by" not in row
    assert diag["freeze_blocked"] is True


# --- 3. Different idea -> still blocking -----------------------------------
#        (credit is scoped to the clip's OWN retry family only -- its own
#        group's winner is unrelated, even with no evidence to the contrary)

def test_different_idea_still_blocking():
    winner_a = clip("winner_a", 0.0, 3.0, "I cleaned out my garage over the weekend and gave away an old bike.", selected=True)
    discard = clip(
        "discard", 3.5, 6.0,
        "My face cleared up nicely after I started using the gentler cleanser every morning.",
        selected=False,
    )
    d = draft(
        selected=(winner_a,), discarded=(discard,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner_a", 0.9), ranked_row("discard", 0.5)]}],
    )

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    row = _row_for(diag, "discard")

    assert row is not None
    assert row["blocking"] is True
    assert row["classification"] == "REAL_CONTENT_LOSS"


# --- 4. Pre-group rejected content -> current fail-closed behavior --------

def test_pre_group_rejected_content_fail_closed_preserved():
    winner = clip("winner", 0.0, 3.0, "This whole thing was a long journey from beginning to end for me.", selected=True)
    # Never appears in any take_judge_groups ranked list -- e.g. rejected
    # before grouping ever ran.
    discard = clip(
        "discard", 20.0, 23.0,
        "Looking back there were small clues around that period I connected much later.",
        selected=False,
    )
    d = draft(
        selected=(winner,), discarded=(discard,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9)]}],
    )

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    row = _row_for(diag, "discard")

    assert row is not None
    assert row["blocking"] is True
    assert row["classification"] == "REAL_CONTENT_LOSS"
    assert "content_loss_suppressed_by" not in row


# --- 5. Hybrid delete without verified replacement -> still blocking ------

def test_hybrid_delete_without_verified_replacement_still_blocking():
    winner = clip("winner", 0.0, 3.0, "That pretty much sums up my whole experience with this from start to finish.", selected=True)
    # Simulates a hybrid_session_cleanup pre-grouping delete: no take_judge_
    # groups entry mentions it at all, and nothing selected covers its
    # content -- must remain blocking exactly as before D-061.
    discard = clip(
        "discard", 45.0, 50.0,
        "A specialist confirmed it was a rare condition nobody around me had heard of before.",
        selected=False,
    )
    d = draft(
        selected=(winner,), discarded=(discard,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.9)]}],
    )

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    row = _row_for(diag, "discard")

    assert row is not None
    assert row["blocking"] is True
    assert row["classification"] == "REAL_CONTENT_LOSS"


# --- 6/7. Selected exact duplicate / semantic paraphrase -> not blocking --
#          These never reach the content_loss trigger at all: the winner's
#          own text is always part of the whole-video comparison, so a
#          genuine near-duplicate or close paraphrase already scores well
#          above the coverage floor under the existing, unmodified check --
#          confirming D-061 introduces no regression for the easy case.

def test_selected_exact_duplicate_not_blocking():
    winner = clip(
        "winner", 0.0, 3.0,
        "I felt tired all the time and eventually went to see a doctor about it.",
        selected=True,
    )
    discard = clip(
        "discard", 3.5, 6.0,
        "I felt tired all the time and eventually went to go see a doctor about it too.",
        selected=False,
    )
    d = draft(
        selected=(winner,), discarded=(discard,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.8), ranked_row("discard", 0.6)]}],
    )

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    row = _row_for(diag, "discard")

    assert row is None or row["blocking"] is False


def test_selected_semantic_paraphrase_not_blocking():
    winner = clip(
        "winner", 0.0, 3.0,
        "I felt worn out every single day and eventually booked a visit with a doctor.",
        selected=True,
    )
    discard = clip(
        "discard", 3.5, 6.0,
        "Feeling worn out every single day, I eventually went and visited a doctor myself.",
        selected=False,
    )
    d = draft(
        selected=(winner,), discarded=(discard,),
        take_judge_groups=[{"group_id": "g1", "ranked": [ranked_row("winner", 0.8), ranked_row("discard", 0.6)]}],
    )

    out = apply_final_story_coherence_validation(d)
    diag = out.diagnostics["final_story_coherence_validation"]
    row = _row_for(diag, "discard")

    assert row is None or row["blocking"] is False
