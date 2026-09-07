"""D-103 (P0 follow-up): PAPILLARY REQUIRED-MEANING GENERAL SAFETY EXTENSION.

D-102's real-media qualification proved `_single_winner_safety_veto`'s
existing CRITICAL-claim check (condition d) cannot catch every required-
meaning loss: the D-101 papillary continuation realization carries ZERO
claims of any type under `semantic_claims.py`'s classifier, so there is
nothing for `critical_coverage_sets` to compare. Investigation (see the
D-103 decision entry) found no already-production-authoritative
representation captures this either -- `semantic_ledger.py` and
`realization_resolver.py`'s `RequirementGroup` machinery are both
explicitly SHADOW-ONLY and cannot be bridged into a live veto without a
separate authority-cutover directive.

This suite proves the MINIMUM general extension added instead: a sibling
expressing a before/after retrospective realization about its own
condition/state (the SAME general marker vocabulary `semantic_claims.py`
already uses for `CONTRASTIVE_HINDSIGHT_NEGATION`, but without requiring
its negation marker) that the proposed winner does not substantially
overlap with is now protected -- both at the single-winner fast path
(veto) AND at the general ladder's own survivor pool (so dominance,
which is driven by the SAME blind claim classifier, cannot silently
re-select the excluded winner anyway).

Generic (English) fixtures only -- no Video00 wording, matching every
other test file in this family (D-101/D-100/D-082).
"""
from cutsell_worker import pipeline
from cutsell_worker.contracts import CandidateTake, RankedTake
from cutsell_worker.pipeline import (
    _is_retrospective_condition_realization,
    _members_missing_required_condition_realization,
    _single_winner_safety_veto,
)


def take(clip_id: str, text: str, *, complete_idea: bool | None = True, start: float = 0.0) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=start, end=start + 4.0, text=text, words=(), signals=None, complete_idea=complete_idea,
    )


def ranked(*pairs: tuple[str, float]) -> tuple[RankedTake, ...]:
    return tuple(RankedTake(clip_id, score, "watch_listen_baseline") for clip_id, score in pairs)


# The exact structural shape from the D-101/D-102 papillary cluster,
# reproduced with generic content:
#   Candidate A: model-labelled winner, unrelated content, own CRITICAL
#   NEGATION claim (a CONTRASTIVE_HINDSIGHT_NEGATION shape).
#   Candidate B: contains the required diagnosis-continuation meaning
#   (a before/after realization about one's own condition), but the
#   existing CRITICAL-claim classifier finds zero claims in it at all.
_CANDIDATE_A_UNRELATED_WINNER = (
    "The other symptoms never seemed unusual to me, but now that I think "
    "about it they really were unusual."
)
_CANDIDATE_B_REQUIRED_CONTINUATION = (
    "The symptoms I had, I thought I was fine at the time, but there were "
    "signs looking back on it."
)


def test_a_positive_control_required_realization_survives_the_authoritative_path():
    """BEFORE (pre-D-103): the fast path would trust A unconditionally and
    B's required meaning would be lost, even though A outscores B on raw
    delivery. AFTER: the required-realization safety excludes A from the
    survivor pool entirely (not merely a veto flag), so B survives through
    the FULL authoritative selection path -- proven with A given the
    HIGHER delivery score, so the outcome cannot be attributed to delivery
    quality alone."""
    a = take("a", _CANDIDATE_A_UNRELATED_WINNER)
    b = take("b", _CANDIDATE_B_REQUIRED_CONTINUATION, start=5.0)
    decisions = {"a": ("winner", 0.95), "b": ("failed", 0.80)}
    r = ranked(("a", 0.90), ("b", 0.60))  # A scores higher -- old behavior would keep A twice over

    selected, preferred, reason = pipeline._semantic_best_take((a, b), decisions, "a", r)

    assert selected == "b"
    assert preferred == "b"
    assert reason != "single_semantic_winner"


def test_a_veto_reports_the_specific_reason():
    a = take("a", _CANDIDATE_A_UNRELATED_WINNER)
    b = take("b", _CANDIDATE_B_REQUIRED_CONTINUATION, start=5.0)
    veto = _single_winner_safety_veto("a", (a, b), None)
    assert veto == "winner_missing_required_condition_realization"


def test_a_helper_detects_the_asymmetric_exclusion():
    a = take("a", _CANDIDATE_A_UNRELATED_WINNER)
    b = take("b", _CANDIDATE_B_REQUIRED_CONTINUATION, start=5.0)
    missing = _members_missing_required_condition_realization(["a", "b"], {"a": a, "b": b})
    assert missing == {"a"}


def test_pattern_helper_requires_both_belief_and_retrospective_markers():
    assert _is_retrospective_condition_realization(_CANDIDATE_B_REQUIRED_CONTINUATION) is True
    assert _is_retrospective_condition_realization("The jacket comes in three colors.") is False
    assert _is_retrospective_condition_realization("I thought the store was closed.") is False  # belief only
    assert _is_retrospective_condition_realization("Looking back, the weather was great.") is False  # retrospective only


# ---------------------------------------------------------------------------
# Required negative controls
# ---------------------------------------------------------------------------

def test_negative_control_low_value_unique_detail_never_vetoes_a_genuine_winner():
    """A sibling with a unique but LOW-VALUE supporting detail (no
    retrospective-realization shape at all) must never trigger this
    safety class -- it is not "every unique fact is critical"."""
    winner = take("winner", "The specialist confirmed she was diagnosed with a rare autoimmune condition.")
    sibling = take("sibling", "It was raining outside that day and traffic was bad on the way there.", start=5.0)
    decisions = {"winner": ("winner", 0.95), "sibling": ("failed", 0.5)}
    r = ranked(("winner", 10.0), ("sibling", 3.0))
    selected, preferred, reason = pipeline._semantic_best_take((winner, sibling), decisions, "winner", r)
    assert selected == "winner"
    assert reason == "single_semantic_winner"
    assert _single_winner_safety_veto("winner", (winner, sibling), None) is None


def test_negative_control_rephrase_already_preserved_by_winner_does_not_force_keep():
    """When the winner's OWN text already substantially overlaps with the
    sibling's retrospective realization (a paraphrase, not a loss), the
    safety class must not fire -- reusing the same content already
    expressed is not a meaning loss."""
    winner = take(
        "winner",
        "The symptoms I had, I thought I was fine at the time, but looking back "
        "there were clear signs I missed the first time around.",
    )
    sibling = take("sibling", "I thought I was fine, but there were signs looking back.", start=5.0)
    decisions = {"winner": ("winner", 0.95), "sibling": ("failed", 0.6)}
    r = ranked(("winner", 10.0), ("sibling", 3.0))
    selected, preferred, reason = pipeline._semantic_best_take((winner, sibling), decisions, "winner", r)
    assert selected == "winner"
    assert reason == "single_semantic_winner"


def test_negative_control_genuine_safe_single_winner_still_uses_fast_path():
    """Positive control for the fast path itself: a genuinely safe,
    non-contradictory, complete single winner is completely unaffected by
    this new safety class."""
    winner = take("winner", "The specialist confirmed she was diagnosed with a rare autoimmune condition.")
    alternate = take("alternate", "It was a really long day at the clinic.", start=5.0)
    decisions = {"winner": ("winner", 0.95), "alternate": ("failed", 0.5)}
    r = ranked(("winner", 10.0), ("alternate", 3.0))
    selected, preferred, reason = pipeline._semantic_best_take((winner, alternate), decisions, "winner", r)
    assert selected == "winner"
    assert reason == "single_semantic_winner"


def test_negative_control_two_redundant_complete_retries_never_become_keep_both():
    """Two candidates that both happen to carry SOME realization language
    about the same general topic must still resolve to exactly ONE
    selection -- this function never composites or keeps both, regardless
    of which safety class fires."""
    a = take("a", "I thought everything was fine at first, but looking back there were signs I missed.")
    b = take("b", "I believed I was fine, but now that I think about it there were definitely signs.", start=5.0)
    decisions = {"a": ("winner", 0.9), "b": ("failed", 0.7)}
    r = ranked(("a", 9.0), ("b", 5.0))
    selected, preferred, reason = pipeline._semantic_best_take((a, b), decisions, "a", r)
    assert selected in ("a", "b")  # exactly one, never both/composite (return type is a single clip id)
    assert isinstance(selected, str)


def test_negative_control_complementary_content_is_not_forced_into_a_composite_here():
    """Two genuinely complementary (different-topic) realizations still
    resolve to exactly one `_semantic_best_take` pick -- this function
    never composites either way; downstream `claim_coverage_best_take.py`
    composite/sufficiency logic (unaffected by this change) remains the
    sole place complementary pairs are ever combined."""
    a = take("a", "The jacket comes in three different colors for everyone to choose from.")
    b = take("b", "I thought the sizing would run small, but looking back it actually fit true to size.", start=5.0)
    decisions = {"a": ("winner", 0.9), "b": ("failed", 0.6)}
    r = ranked(("a", 9.0), ("b", 5.0))
    selected, preferred, reason = pipeline._semantic_best_take((a, b), decisions, "a", r)
    assert selected in ("a", "b")


def test_negative_control_contradiction_safety_still_checked_first_in_the_fast_path():
    """The existing contradiction veto (condition c) is checked BEFORE this
    new required-realization check inside `_single_winner_safety_veto` --
    a contradicting pair is vetoed for that reason, never silently
    overridden by this new class."""
    a = take("a", "The discount applies at a rate of 5 percent storewide for everyone today.")
    b = take("b", "The discount applies at a rate of 5 percent, though loyalty members instead get 10 percent.", start=5.0)
    veto = _single_winner_safety_veto("a", (a, b), None)
    assert veto == "members_contradict"


def test_family_context_d102_fixture_unaffected_by_d103():
    """D-102's own family-context fixture (the '5-10%' partial-paraphrase
    pair) must still resolve identically -- neither candidate matches the
    D-103 retrospective-realization pattern, so this new check never
    engages for that family."""
    winner_text = (
        "Esta es mi experiencia. Soy la única en mi familia que tiene este tipo de cáncer. "
        "Por eso no creo y está comprobado científicamente que los cánceres son hereditarios. "
        "Más bien solo un 5 -10 % son de carácter hereditario. Mayormente son nuestras "
        "elecciones de vida. Así que cuídate."
    )
    loser_text = "Así que estoy convencida y la ciencia lo avala que solo un 5 -10 % de los"
    assert _is_retrospective_condition_realization(winner_text) is False
    assert _is_retrospective_condition_realization(loser_text) is False
    winner = take("clip_62e16", winner_text)
    loser = take("clip_a5a66", loser_text, start=5.0)
    decisions = {"clip_62e16": ("winner", 0.95), "clip_a5a66": ("failed", 0.9)}
    selected, preferred, reason = pipeline._semantic_best_take((winner, loser), decisions, "clip_62e16")
    assert selected == "clip_62e16"
    assert reason != "single_semantic_winner"  # D-102 Fix A's own veto (condition d) still fires
