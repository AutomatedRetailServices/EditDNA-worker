"""D-056.5 PROPOSITION-COMPLETE NEGATION CONTRADICTION CONTRACT.

Root false positive (docs/CUTSELL_DECISIONS.md D-056.4's forensic, live
evidence run 33794783794 `tg_48c0c0027c22862ff2`): the D-056.3 shared
contradiction primitive's `negation_conflict` signal fired whenever exactly
one of two texts contained ANY negation-marker token anywhere in the whole
clip -- even when that token was attached to a DIFFERENT, adjacent clause
than the content actually shared with the other text (a rhetorical
negation), or when the OTHER text was an incomplete retry that trailed off
before ever reaching an equivalent clause at all. Fixed by scoping the
negation-token comparison to the sentence/clause that contains content
shared between the two texts (preferring a shared NUMBER as the most
specific anchor) -- entirely inside `contradiction_signal.detect_text_
contradiction`, so every existing caller (StoryValidator's two checks, its
residual-family exemption, CanonicalEditPlan's composite-acceptance gate)
benefits automatically with zero caller-side changes.

This file is entirely generic -- no Video00 phrases in the fixtures beyond
what plain Spanish negation-marker vocabulary testing requires (ordinary
sentences, not tied to any specific video's content) -- and no Video00 text
anywhere in production code (contradiction_signal.py itself). Covers the
full D-056.5 Section 6 matrix plus a structural regression fixture matching
the D-056.4 shape, and (Section 7) proves the fix does not reduce true-
contradiction recall or weaken FinalEditReviewer as an independent layer.
"""
from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.contradiction_signal import any_pair_contradicts, detect_text_contradiction
from cutsell_worker.final_edit_reviewer import CONTRADICTION, DUPLICATE_IDEA, UNRESOLVED_RETRY, review
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation


# --- pipeline helpers (same convention as test_cutsell_d056_3_...) ---------

def clip(clip_id, start, end, text, *, selected, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def draft(*, selected=(), take_judge_groups=(), claim_coverage_composites=()):
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
        discarded=(),
        diagnostics=diagnostics,
    )


def ranked_row(clip_id, score):
    return {"clip_id": clip_id, "score": score, "reason": "watch_listen_baseline"}


def run_pipeline(d):
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
        claim_coverage_composites=[{"group_id": group_id, "clip_ids": ["a", "b"]}],
    )


# --- Section 1/6.11: D-056.4 shape reproduced generically ------------------

def test_d056_4_shape_reproduced_generically_no_longer_conflicts():
    """Generic paraphrase of the exact D-056.4 live structure: a complete
    realization rhetorically negates a BROADER, unrelated claim in an
    earlier clause, then restates the shared figure in a later clause
    without negation; the other realization never reaches its own
    completion of that later clause at all."""
    a = (
        "This is my experience. I am the only one in my family with this diagnosis. "
        "That is why I do not believe, and science backs this up, that these conditions "
        "are broadly hereditary. Rather, only about 10 percent are hereditary in nature. "
        "Mostly it comes down to lifestyle, so take care of yourself."
    )
    b = (
        "I am the first one in my family with this diagnosis. Nobody in my family has "
        "this specific rare subtype or suffers from related issues. So I am convinced, "
        "and the science supports it, that only about 10 percent of"
    )
    result = detect_text_contradiction(a, b)
    assert result.negation_conflict is False
    assert result.number_conflict is False
    assert result.has_conflict is False


# --- Section 6 matrix -------------------------------------------------------

def test_complete_positive_vs_complete_negative_contradicts():
    a = "the medication worked well for her symptoms."
    b = "the medication never worked well for her symptoms."
    result = detect_text_contradiction(a, b)
    assert result.negation_conflict is True
    assert result.has_conflict is True


def test_complete_negative_vs_incomplete_same_direction_retry_no_conflict():
    a = "nobody in her family had this condition before her diagnosis, as far as she knew."
    b = "going back through the family history on her mother's side there was"
    result = detect_text_contradiction(a, b)
    assert result.negation_conflict is False


def test_incomplete_negative_fragment_that_completes_the_claim_still_contradicts():
    """D-056.5 Section 4: do NOT treat every incomplete clip as safe -- if
    the negation-bearing clause itself is fully stated before the recording
    trails off elsewhere, the contradiction must remain valid."""
    a = "the treatment worked for every patient in the trial."
    b = "the treatment never worked for any patient in the trial, and the team then"
    result = detect_text_contradiction(a, b)
    assert result.negation_conflict is True
    assert result.has_conflict is True


def test_rhetorical_negation_outside_shared_proposition_no_conflict():
    a = "honestly I did not expect this outcome at first. anyway, the biopsy showed it was benign."
    b = "the biopsy result showed the growth was benign, according to her doctor."
    result = detect_text_contradiction(a, b)
    assert result.negation_conflict is False


def test_nadie_ni_sin_nunca_not_never_all_recognized_as_polarity_markers():
    positive = "la familia tiene antecedentes de esta condicion."
    variants = {
        "nadie": "nadie en la familia tiene antecedentes de esta condicion.",
        "ni": "la familia no tiene antecedentes de esta condicion ni de otras similares.",
        "sin": "la familia esta sin antecedentes de esta condicion.",
        "nunca": "la familia nunca tuvo antecedentes de esta condicion.",
        "not": "the family does not have a history of this condition.",
        "never": "the family never had a history of this condition.",
    }
    pos_en = "the family has a history of this condition."
    for marker, negative in variants.items():
        base = positive if marker in ("nadie", "ni", "sin", "nunca") else pos_en
        result = detect_text_contradiction(base, negative)
        assert result.negation_conflict is True, f"marker {marker!r} not recognized"


def test_same_fact_restated_differently_no_conflict():
    a = "roughly 10 percent of cases are hereditary, according to her doctor."
    b = "her doctor said about 10 percent of cases run in families."
    result = detect_text_contradiction(a, b)
    assert result.has_conflict is False


def test_incompatible_numbers_still_contradicts():
    a = "roughly 5 percent of cases are hereditary."
    b = "roughly 10 percent of cases are hereditary."
    result = detect_text_contradiction(a, b)
    assert result.number_conflict is True
    assert result.has_conflict is True


def test_explicit_correction_of_the_same_number_still_contradicts():
    a = "it was about 5 percent of cases in the study."
    b = "actually, it was about 10 percent of cases in the study."
    result = detect_text_contradiction(a, b)
    assert result.number_conflict is True


def test_different_propositions_each_containing_negation_no_conflict():
    a = "she never smoked in her life."
    b = "she does not drink coffee in the mornings."
    result = detect_text_contradiction(a, b)
    assert result.negation_conflict is False


def test_shared_proposition_absent_on_one_side_no_conflict():
    a = "no one in her family had this condition before, as far as she knew."
    b = "the weather that week had been unusually mild for the season."
    result = detect_text_contradiction(a, b)
    assert result.negation_conflict is False


def test_explicit_causal_inversion_via_negation_still_contradicts():
    a = "stress triggers the flare-ups every time, according to her notes."
    b = "stress never triggers the flare-ups, according to her notes."
    result = detect_text_contradiction(a, b)
    assert result.negation_conflict is True


def test_any_pair_contradicts_still_finds_the_completeness_gated_conflict():
    texts = [
        "the medication worked well for her symptoms.",
        "the medication never worked well for her symptoms.",
        "she also mentioned mild side effects at first.",
    ]
    assert any_pair_contradicts(texts) is True


# --- Section 7: no safety weakening, end to end ----------------------------

def test_true_contradiction_still_blocks_freeze_end_to_end():
    d = _two_member_composite_draft(
        "the medication worked well for her symptoms.",
        "the medication never worked well for her symptoms.",
    )
    out, plan, result = run_pipeline(d)

    idea = plan.ideas[0]
    assert idea.is_composite is False
    assert idea.coverage_status == "unresolved_ambiguous"
    coherence = out.diagnostics["final_story_coherence_validation"]
    assert coherence["freeze_blocked"] is True
    assert coherence["unresolved_family_count"] == 1
    kinds = {f.kind for f in result.findings}
    assert {CONTRADICTION, DUPLICATE_IDEA, UNRESOLVED_RETRY}.issubset(kinds)
    assert result.status == "FAIL"


def test_d056_4_shape_end_to_end_no_longer_blocks_on_a_false_contradiction():
    """The exact class of case D-056.4 found live, run through the full
    StoryValidator -> CanonicalEditPlan -> FinalEditReviewer chain: no
    longer produces a CONTRADICTION finding, and the composite is accepted
    (both realizations complete the SAME shared claim, no genuine
    polarity conflict)."""
    a = (
        "This is my experience. I am the only one in my family with this diagnosis. "
        "That is why I do not believe, and science backs this up, that these conditions "
        "are broadly hereditary. Rather, only about 10 percent are hereditary in nature."
    )
    b = "I am the first one in my family with this diagnosis, and only about 10 percent are hereditary."
    d = _two_member_composite_draft(a, b)
    out, plan, result = run_pipeline(d)

    coherence = out.diagnostics["final_story_coherence_validation"]
    assert coherence["contradiction_findings"] == []
    assert not any(f.kind == CONTRADICTION for f in result.findings)


def test_final_edit_reviewer_still_independently_agrees_after_the_gate():
    """FinalEditReviewer reads the same fields StoryValidator/CanonicalEditPlan
    populate -- unmodified by D-056.5, and still structurally in agreement."""
    d = _two_member_composite_draft(
        "roughly 5 percent of cases are hereditary.",
        "roughly 10 percent of cases are hereditary.",
    )
    out, plan, result = run_pipeline(d)

    coherence = out.diagnostics["final_story_coherence_validation"]
    contradiction_group_ids = {row["group_id"] for row in coherence["contradiction_findings"]}
    review_contradiction_ideas = {f.idea_id for f in result.findings if f.kind == CONTRADICTION}
    assert contradiction_group_ids == {"g1"}
    assert review_contradiction_ideas == {"g1"}
    assert plan.ideas[0].coverage_status == "unresolved_ambiguous"
