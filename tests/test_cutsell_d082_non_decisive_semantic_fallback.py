"""D-082: NON-DECISIVE SEMANTIC LABEL FALLBACK -- AUTHORITATIVE BEST-TAKE STABILITY.

Generic (English) fixtures only -- reproduces the D-080 sonography SHAPE
(a shorter, incomplete retry with a HIGHER DeliveryScorer score vs. a
complete retry with a LOWER score, and semantic labels that degrade from
decisive to non-actionable across two hypothetical runs) rather than the
literal Video00 clause.

D-080 proved: ``pipeline.py::_semantic_best_take`` fell straight through to
the raw, completeness-blind local (DeliveryScorer) rank whenever
``take_judge_groups``'s semantic-candidate labels were non-decisive (zero or
2+ "winner" labels) -- byte-identical DeliveryScorer scores across two live
runs, but the winning candidate flipped because the LLM label set flipped
from a decisive {"failed", "winner"} to a non-actionable {"keep", "keep"}.

This suite proves the D-082 cutover: when labels are decisive, behavior is
byte-identical to before (Section 2); when they are not, deterministic
evidence this codebase already computes elsewhere (D-081's
semantic_delete_recommended, CandidateTake.complete_idea, and D-063/D-065/
D-066 CRITICAL_COVERAGE_DOMINANCE, reused verbatim) is consulted BEFORE the
raw DeliveryScorer rank ever gets the final word -- and delivery only
settles a GENUINE tie, never an asymmetric/disjoint split.
"""
from cutsell_worker.contracts import CandidateTake, RankedTake
from cutsell_worker.pipeline import _semantic_best_take


def take(clip_id: str, text: str, *, complete_idea: bool = True, start: float = 0.0) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=start + 4.0,
        text=text,
        complete_idea=complete_idea,
    )


def ranked(*pairs: tuple[str, float]) -> tuple[RankedTake, ...]:
    return tuple(RankedTake(clip_id, score, "watch_listen_baseline") for clip_id, score in pairs)


# --- Section 4/3: the core D-082 sonography stability regression ----------

_A_SHORT_INCOMPLETE = "We never thought to check because it always came back fine."
_B_COMPLETE = "We never thought to check because it always came back fine on the routine tests, so this was a surprise."


def _sonography_pair():
    a = take("A", _A_SHORT_INCOMPLETE, complete_idea=False)
    b = take("B", _B_COMPLETE, complete_idea=True, start=4.0)
    r = ranked(("A", 0.6817), ("B", 0.6211))
    return a, b, r


def test_sonography_regression_decisive_labels_select_complete_candidate():
    a, b, r = _sonography_pair()
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("failed", 0.90), "B": ("winner", 0.95)}, "A", r,
    )
    assert selected == "B"
    assert reason == "single_semantic_winner"


def test_sonography_regression_non_decisive_labels_still_select_complete_candidate():
    """The core D-082 regression: identical deterministic candidates, but
    labels degrade to a non-actionable {"keep","keep"} -- final selection
    must remain B, not silently flip to the higher-raw-score incomplete A."""
    a, b, r = _sonography_pair()
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("keep", 0.92), "B": ("keep", 0.95)}, "A", r,
    )
    assert selected == "B"
    assert reason != "local_fallback"


# --- Section 5/9: winner/winner and full model-variance matrix ------------

_THIN_TEXT = "I felt a bit off for a while."
_RICH_TEXT = "The test confirmed it was a mild vitamin D deficiency, and I felt a bit off for a while."


def _completeness_dominance_pair():
    # Raw delivery favors the THINNER candidate (A) -- proves dominance,
    # not delivery, decides once a critical (DIAGNOSIS_IDENTIFICATION) fact
    # is at stake. B is a strict superset of A's own content plus the
    # diagnosis confirmation, so D-063's dominance -- reused, not
    # reimplemented -- finds it unambiguously.
    a = take("A", _THIN_TEXT)
    b = take("B", _RICH_TEXT, start=4.0)
    r = ranked(("A", 0.90), ("B", 0.70))
    return a, b, r


def test_winner_winner_variance_prefers_objectively_more_complete_candidate():
    a, b, r = _completeness_dominance_pair()
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("winner", 0.90), "B": ("winner", 0.90)}, "A", r,
    )
    assert selected == "B"
    assert reason == "critical_coverage_dominance"


def test_model_label_variance_matrix_invariant_final_selection():
    """Section 9: sweep failed/winner, keep/keep, winner/winner, keep/winner,
    failed/keep over the SAME deterministic candidates. Whenever
    deterministic semantic evidence (here: critical coverage dominance)
    clearly favors B, the final selection must remain B regardless of the
    mocked LLM label shape."""
    a, b, r = _completeness_dominance_pair()
    label_shapes = [
        {"A": ("failed", 0.90), "B": ("winner", 0.95)},
        {"A": ("keep", 0.90), "B": ("keep", 0.90)},
        {"A": ("winner", 0.90), "B": ("winner", 0.90)},
        {"A": ("keep", 0.90), "B": ("winner", 0.90)},
        {"A": ("failed", 0.90), "B": ("keep", 0.90)},
    ]
    for labels in label_shapes:
        selected, _preferred, _reason = _semantic_best_take((a, b), labels, "A", r)
        assert selected == "B", labels


# --- Section 6: critical coverage precedence over delivery ----------------

def test_critical_coverage_dominance_wins_before_delivery_is_ever_consulted():
    a, b, r = _completeness_dominance_pair()
    # Non-decisive labels; delivery alone would pick A (0.90 > 0.70).
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("keep", 0.88), "B": ("keep", 0.91)}, "A", r,
    )
    assert selected == "B"
    assert reason == "critical_coverage_dominance"


# --- Section 7: unique-fact safety -- genuine asymmetry stays unresolved --

def test_two_distinct_critical_facts_neither_dominates_stays_unresolved():
    a = take("A", "The test confirmed it was a mild vitamin D deficiency.")
    b = take("B", "The test confirmed it was low iron.", start=4.0)
    r = ranked(("A", 0.80), ("B", 0.75))
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("keep", 0.90), "B": ("keep", 0.90)}, "A", r,
    )
    # Must not force a winner between two disjoint diagnosis claims.
    assert selected == "A"  # local_selected_clip_id, the pre-D-082 safe default
    assert preferred is None
    assert reason == "unresolved_unique_fact_asymmetry"


# --- Section 11: failed/incomplete safety ----------------------------------

def test_richer_but_incomplete_candidate_never_dominates_solely_on_word_count():
    """A is longer, carries a critical numeric fact, but is EXPLICITLY
    truncated/incomplete. B is shorter, complete, and safely delivers the
    required proposition without the extra (unusable) content. B must win
    -- richness/claim count never overrides proven incompleteness."""
    a = take(
        "A",
        "So basically what happened is you take two of these every single morning and then",
        complete_idea=False,
    )
    b = take("B", "You take one capsule every morning with breakfast.", start=4.0)
    r = ranked(("A", 0.95), ("B", 0.60))
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("keep", 0.90), "B": ("keep", 0.90)}, "A", r,
    )
    assert selected == "B"


# --- Section 8/10: delivery still decides a genuine tie --------------------

def test_delivery_decides_when_content_is_effectively_tied():
    a = take("A", "I really loved this product overall and would buy it again.")
    b = take("B", "I really loved this product overall, and would buy it again.", start=4.0)
    r = ranked(("A", 0.55), ("B", 0.91))
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("keep", 0.90), "B": ("keep", 0.90)}, "A", r,
    )
    assert selected == "B"
    assert reason == "delivery_tie_break_among_survivors"


# --- Section 12: D-081 semantic_delete_recommended evidence integration ---

def test_semantic_delete_recommended_evidence_soft_excludes_a_candidate():
    a = take("A", "I really loved this product overall and would buy it again.")
    b = take("B", "I really loved this product overall, and would buy it again.", start=4.0)
    # A has the HIGHER raw delivery score but carries D-081's negative
    # semantic evidence -- it must not win via the delivery tie-break.
    r = ranked(("A", 0.91), ("B", 0.55))
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("keep", 0.90), "B": ("keep", 0.90)}, "A", r,
        semantic_delete_recommended={"A": True, "B": False},
    )
    assert selected == "B"


def test_semantic_delete_recommended_never_recreates_irreversible_delete_authority():
    """If EVERY candidate carries the negative evidence, it must not
    eliminate the whole group (fail-open, same posture as every other step)
    -- D-081's evidence informs, it never became a second destructive
    authority."""
    a = take("A", "I really loved this product overall and would buy it again.")
    b = take("B", "I really loved this product overall, and would buy it again.", start=4.0)
    r = ranked(("A", 0.91), ("B", 0.55))
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("keep", 0.90), "B": ("keep", 0.90)}, "A", r,
        semantic_delete_recommended={"A": True, "B": True},
    )
    assert selected in {"A", "B"}  # never crashes, never empties the group


# --- Section 13: sales/UGC generalization ----------------------------------

def test_ugc_required_claim_wins_even_with_lower_delivery_score():
    """Generic stand-in for a required dosage/offer/CTA claim (D-082 Section
    13): the candidate carrying it must win even against a higher raw
    delivery score on the take that omits it."""
    a = take("A", "These gummies helped my bloating.")
    b = take(
        "B",
        "The test confirmed it was a mild sensitivity, and these gummies helped my bloating.",
        start=4.0,
    )
    r = ranked(("A", 0.88), ("B", 0.62))  # raw delivery would wrongly favor A
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("keep", 0.90), "B": ("keep", 0.90)}, "A", r,
    )
    assert selected == "B"
    assert reason == "critical_coverage_dominance"


def test_ugc_delivery_may_favor_cleaner_take_when_dosage_not_at_stake():
    a = take("A", "These gummies helped my bloating within a week.")
    b = take("B", "These gummies helped my bloating within a week, honestly.", start=4.0)
    r = ranked(("A", 0.91), ("B", 0.58))
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("keep", 0.90), "B": ("keep", 0.90)}, "A", r,
    )
    assert selected == "A"
    assert reason == "delivery_tie_break_among_survivors"


def test_ugc_price_offer_claim_wins_over_higher_delivery_score():
    a = take("A", "This blender is powerful and looks great on the counter.")
    b = take("B", "This blender is powerful and it's on sale for 20 percent off right now.", start=4.0)
    r = ranked(("A", 0.90), ("B", 0.65))
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("keep", 0.90), "B": ("keep", 0.90)}, "A", r,
    )
    assert selected == "B"
    assert reason == "critical_coverage_dominance"


# --- Section 2: decisive labels remain byte-identical to pre-D-082 --------

def test_decisive_labels_unchanged_even_when_raw_delivery_disagrees():
    a, b, r = _completeness_dominance_pair()
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("alternate", 0.85), "B": ("winner", 0.92)}, "A", r,
    )
    assert selected == "B"
    assert preferred == "B"
    assert reason == "single_semantic_winner"


# --- QA_ENGINE hardening: a genuine contradiction must never reach delivery,
#     even if a claim-identity collision ever made two coverage sets look
#     identical -- reuses `contradiction_signal.any_pair_contradicts`
#     directly, the same gate `_critical_coverage_dominant_candidate` itself
#     already applies internally. -----------------------------------------

def test_contradiction_never_reaches_delivery_even_with_a_large_score_gap():
    a = take("A", "The test confirmed it was gastritis.")
    b = take("B", "The test confirmed it was not gastritis.", start=4.0)
    r = ranked(("A", 0.60), ("B", 0.95))  # delivery strongly favors B
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("keep", 0.90), "B": ("keep", 0.90)}, "A", r,
    )
    assert selected == "A"  # stays at the safe local default, never B
    assert preferred is None
    assert reason in {"unresolved_unique_fact_asymmetry", "unresolved_contradiction"}
