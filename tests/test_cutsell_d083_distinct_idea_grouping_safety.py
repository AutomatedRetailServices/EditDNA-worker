"""D-083 -- DISTINCT-IDEA RETRY GROUPING SAFETY (grouping-only).

Root defect: `split_incohesive_retry_groups` (D-058 Phase 1) has no
content-divergence safety net on its own within-group arbiter confirmations,
unlike `reconcile_semantic_idea_equivalence`'s marker-gated
`_marked_side_diverges_in_content` override. Live audit of the D-082
stability battery found the resulting gap: a deterministic baseline group
bundling one back-acne mention ("Por temporada me salio un acne en la
espalda ... resorcina") with three separate hormonal-pimples mentions of
increasing specificity -- a short generic first mention, a discarded
"bad monolith" elaboration, and a marked ("Otro sintoma") polished
restatement -- relied entirely on unconditional trust of the arbiter once
each pair became "weak." Human Gold expects the short generic mention
delivered on its own (never competing with the pimples retry family) and
the polished restatement to win over the discarded elaboration -- not one
mutually-exclusive four-way contest.

FIX: apply the same marker-gated `_marked_side_diverges_in_content` check
`reconcile_semantic_idea_equivalence` already uses to
`split_incohesive_retry_groups`'s own within-group arbiter confirmations
(`_within_group_arbiter_confirmation_diverges`). A broader, unconditional
content-overlap floor on every confirmation was evaluated and rejected: see
the module comment directly above that function in take_grouping_provider.py
for the counter-evidence (a true-retry paraphrase pair from this file's own
sibling D-058 suite scores LOWER lexical overlap by every measure tried than
a pimples pair that must NOT merge -- no fixed overlap threshold can
separate the two in general). The marker gate closes the specific,
diagnosable D-082-battery defect without reintroducing that false-positive
class; an unmarked, low-lexical-overlap ambiguous pair remains governed by
the arbiter's own judgment alone, exactly as before this fix -- an honestly
scoped, partial closure, not a claim of full generality.

Entirely generic English fixtures per the D-083 directive's own sections,
plus one regression reproducing the real (translated) Spanish shape.
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceResult,
)
from cutsell_worker.take_grouping_provider import (
    _has_distinct_addition_marker,
    _marked_side_diverges_in_content,
    _within_group_arbiter_confirmation_diverges,
    split_incohesive_retry_groups,
)


class FixedArbiter:
    """Confirms same_idea only for checked pairs whose exact (left, right)
    text appears in `same_idea_pairs` -- same fixture shape as the sibling
    D-058 Phase 1 suite."""

    def __init__(self, same_idea_pairs=frozenset(), confidence: float = 0.9):
        self.same_idea_pairs = same_idea_pairs
        self.confidence = confidence
        self.calls = 0

    def check(self, request):
        self.calls += 1
        decisions = tuple(
            IdeaEquivalenceDecision(
                pair_index=i,
                same_idea=(pair.left_text, pair.right_text) in self.same_idea_pairs
                or (pair.right_text, pair.left_text) in self.same_idea_pairs,
                confidence=self.confidence,
                reason="fake evidence",
            )
            for i, pair in enumerate(request.pairs)
        )
        return IdeaEquivalenceResult(
            decisions=decisions, provider="fake", model="fake-semantic-equivalence",
            requested=True, available=True, estimated_input_tokens=50, estimated_output_tokens=10,
        )


class ConfirmEverythingArbiter:
    """Adversarial (QA_ENGINE) fixture: confirms same_idea=True for EVERY
    pair it is asked about, regardless of content. Used to prove the marker
    gate -- not arbiter restraint -- is what keeps a marked distinct pair
    split even under a maximally over-eager arbiter."""

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
        self.calls = 0

    def check(self, request):
        self.calls += 1
        decisions = tuple(
            IdeaEquivalenceDecision(
                pair_index=i, same_idea=True, confidence=self.confidence,
                reason="adversarial: confirms everything",
            )
            for i, pair in enumerate(request.pairs)
        )
        return IdeaEquivalenceResult(
            decisions=decisions, provider="fake", model="fake-semantic-equivalence",
            requested=True, available=True, estimated_input_tokens=50, estimated_output_tokens=10,
        )


def _take(clip_id, start, end, text, source="src", order=0):
    return CandidateTake(clip_id, source, order, start, end, text)


# --- Section 4: distinct-idea negative control ------------------------------
# Two genuinely distinct symptom beats that happen to share generic topic
# vocabulary ("skin", "on my"). Neither carries a distinct-addition marker,
# so this exercises the arbiter's own judgment (the realistic path) rather
# than the marker gate -- a real arbiter is expected to tell these apart.

def test_section4_distinct_symptom_beats_survive_when_arbiter_declines():
    left_text = "I had acne on my back and treated it with resorcinol."
    right_text = "I also had pimples behind my ears and neck that looked hormonal."
    takes = (_take("acne", 0.0, 2.0, left_text), _take("pimples", 15.0, 17.0, right_text))
    arbiter = FixedArbiter(same_idea_pairs=frozenset())  # arbiter correctly declines
    result, diagnostics = split_incohesive_retry_groups((("acne", "pimples"),), takes, arbiter=arbiter)
    assert result == (("acne",), ("pimples",))
    assert diagnostics["groups_split"] == 1


# --- Section 4b: distinct-idea negative control, MARKED side ---------------
# One side explicitly marks itself as introducing another point. Even if the
# arbiter over-eagerly confirms same_idea, the marker + content-divergence
# gate must still split them apart.

def test_section4b_marked_distinct_addition_splits_even_if_arbiter_confirms():
    left_text = "I had acne on my back and treated it with resorcinol."
    right_text = "Another symptom was hormonal pimples behind my ears and along my neck."
    assert _has_distinct_addition_marker(right_text)
    takes = (_take("acne", 0.0, 2.0, left_text), _take("pimples", 15.0, 17.0, right_text))
    arbiter = FixedArbiter(same_idea_pairs=frozenset({(left_text, right_text)}))
    result, diagnostics = split_incohesive_retry_groups((("acne", "pimples"),), takes, arbiter=arbiter)
    assert result == (("acne",), ("pimples",))
    assert len(diagnostics["content_divergence_blocked"]) == 1


# --- Section 5: true-retry positive control ---------------------------------
# Short mention vs. the same mention with more detail -- must merge.

def test_section5_true_retry_short_vs_long_still_merges():
    left_text = "I had acne on my back."
    right_text = "I had acne on my back and treated it with resorcinol."
    takes = (_take("a", 0.0, 2.0, left_text), _take("b", 2.5, 4.5, right_text))
    arbiter = FixedArbiter(same_idea_pairs=frozenset({(left_text, right_text)}))
    result, diagnostics = split_incohesive_retry_groups((("a", "b"),), takes, arbiter=arbiter)
    assert result == (("a", "b"),)
    assert diagnostics["groups_split"] == 0


# --- Section 6: partial-overlap / directional-completeness safety ----------
# Fact A vs Fact A + Fact B: a retry that ADDS detail must not have that
# extra detail treated as disposable by the grouping stage -- they must
# still be allowed to compete (grouping decides WHO competes, never WHO
# wins; BestTake/D-063 dominance -- untouched by this directive -- decides
# the winner from there).

def test_section6_fact_a_vs_fact_a_plus_b_still_competes():
    fact_a = "The supplement helped my digestion."
    fact_a_plus_b = "The supplement helped my digestion and also cleared up my skin."
    takes = (_take("a", 0.0, 2.0, fact_a), _take("b", 2.5, 4.5, fact_a_plus_b))
    arbiter = FixedArbiter(same_idea_pairs=frozenset({(fact_a, fact_a_plus_b)}))
    result, diagnostics = split_incohesive_retry_groups((("a", "b"),), takes, arbiter=arbiter)
    assert result == (("a", "b"),)
    assert diagnostics["groups_split"] == 0


# --- Section 7: sales/UGC generalization ------------------------------------

def test_section7_dosage_vs_benefit_do_not_compete():
    left_text = "I take two gummies every morning."
    right_text = "They helped my bloating."
    takes = (_take("dosage", 0.0, 2.0, left_text), _take("benefit", 15.0, 17.0, right_text))
    arbiter = FixedArbiter(same_idea_pairs=frozenset())
    result, diagnostics = split_incohesive_retry_groups((("dosage", "benefit"),), takes, arbiter=arbiter)
    assert result == (("dosage",), ("benefit",))


def test_section7_benefit_restatement_may_compete():
    left_text = "These gummies helped my bloating."
    right_text = "After a week these gummies really helped my bloating."
    takes = (_take("a", 0.0, 2.0, left_text), _take("b", 2.5, 4.5, right_text))
    arbiter = FixedArbiter(same_idea_pairs=frozenset({(left_text, right_text)}))
    result, diagnostics = split_incohesive_retry_groups((("a", "b"),), takes, arbiter=arbiter)
    assert result == (("a", "b"),)


# --- Real-shape regression: the D-082 battery pimples/acne conflation ------
# Translated shape of the real (Spanish) D-082 battery clips: one back-acne
# mention, a short generic first pimples mention, a discarded "bad monolith"
# elaboration, and a marked polished restatement. Target grouping: acne
# alone, the short mention alone, and {monolith, restatement} as one retry
# family (restatement wins downstream -- untouched by this directive).

_ACNE = "For a while I would get acne on my back which I treated with resorcinol."
_PIMPLES_SHORT = "I also used to get pimples. It was like a rash, an allergy."
_PIMPLES_MONOLITH = (
    "I also used to get pimples in this area right behind my ear and all along "
    "my neck which I thought was an allergy but it was like pimples from hormonal problems."
)
_PIMPLES_RESTATEMENT = (
    "Another symptom was that I would get pimples like an allergy in this area "
    "right behind my ear and along my neck. It came in seasons."
)


def test_regression_acne_vs_marked_restatement_splits_even_if_arbiter_confirms():
    # acne vs the MARKED restatement: the content-divergence gate protects
    # this pair unconditionally, regardless of what the arbiter says --
    # proven with the adversarial always-confirm arbiter.
    takes = (_take("acne", 0.0, 2.0, _ACNE), _take("pimples_restatement", 15.0, 20.0, _PIMPLES_RESTATEMENT))
    arbiter = ConfirmEverythingArbiter()
    result, diagnostics = split_incohesive_retry_groups((("acne", "pimples_restatement"),), takes, arbiter=arbiter)
    assert result == (("acne",), ("pimples_restatement",))


def test_regression_acne_vs_unmarked_pimples_mentions_split_when_arbiter_declines():
    # acne vs the two UNMARKED mentions (short / monolith): neither carries a
    # discourse marker, so this exercises the arbiter's own judgment -- the
    # realistic path a real arbiter is expected to get right given the two
    # mentions share no specific content at all (back acne vs facial/neck
    # pimples). Not a claim the content-divergence gate itself covers this
    # case -- see the module comment on `_within_group_arbiter_confirmation_
    # diverges` for the honestly-scoped limit of what the gate guarantees.
    for other_id, other_text in (
        ("pimples_short", _PIMPLES_SHORT),
        ("pimples_monolith", _PIMPLES_MONOLITH),
    ):
        takes = (_take("acne", 0.0, 2.0, _ACNE), _take(other_id, 15.0, 20.0, other_text))
        arbiter = FixedArbiter(same_idea_pairs=frozenset())  # realistic: arbiter declines
        result, diagnostics = split_incohesive_retry_groups((("acne", other_id),), takes, arbiter=arbiter)
        assert result == (("acne",), (other_id,)), f"acne vs {other_id} must split"


def test_regression_short_generic_mention_survives_independently_of_marked_restatement():
    # The short mention carries no marker; the restatement does. Even under
    # an adversarial always-confirm arbiter, the marker + content-divergence
    # gate must keep the short mention split from the restatement.
    assert not _has_distinct_addition_marker(_PIMPLES_SHORT)
    assert _has_distinct_addition_marker(_PIMPLES_RESTATEMENT)
    takes = (
        _take("short", 0.0, 2.0, _PIMPLES_SHORT),
        _take("restatement", 20.0, 24.0, _PIMPLES_RESTATEMENT),
    )
    arbiter = ConfirmEverythingArbiter()
    result, diagnostics = split_incohesive_retry_groups((("short", "restatement"),), takes, arbiter=arbiter)
    assert result == (("short",), ("restatement",))


def test_regression_monolith_and_restatement_still_merge_as_genuine_retry():
    # The one pair that MUST stay together: same specific location + same
    # hormonal framing, one a discarded elaboration, one its marked, polished
    # restatement -- this is the founding D-047 Case 1 shape.
    assert _has_distinct_addition_marker(_PIMPLES_RESTATEMENT)
    assert not _marked_side_diverges_in_content(_PIMPLES_MONOLITH, _PIMPLES_RESTATEMENT)
    takes = (
        _take("monolith", 15.0, 20.0, _PIMPLES_MONOLITH),
        _take("restatement", 20.5, 24.0, _PIMPLES_RESTATEMENT),
    )
    arbiter = FixedArbiter(same_idea_pairs=frozenset({(_PIMPLES_MONOLITH, _PIMPLES_RESTATEMENT)}))
    result, diagnostics = split_incohesive_retry_groups((("monolith", "restatement"),), takes, arbiter=arbiter)
    assert result == (("monolith", "restatement"),)
    assert diagnostics["groups_split"] == 0


def test_regression_full_five_member_conflated_group_resolves_to_three_families():
    # The exact shape observed in the D-082 battery: one baseline group
    # bundling all four mentions together. Only the monolith-restatement
    # pair should stay merged; acne and the short mention must each end up
    # in their own singleton group.
    takes = (
        _take("acne", 0.0, 2.0, _ACNE),
        _take("short", 5.0, 7.0, _PIMPLES_SHORT),
        _take("monolith", 15.0, 20.0, _PIMPLES_MONOLITH),
        _take("restatement", 20.5, 24.0, _PIMPLES_RESTATEMENT),
    )
    group = ("acne", "short", "monolith", "restatement")
    # Realistic arbiter: confirms only the genuine retry pair, correctly
    # declines every acne/pimples cross-topic pair, and (worst case for the
    # unmarked residual, documented in the module comment) also confirms
    # the short-vs-monolith pair on coarse topical grounds.
    arbiter = FixedArbiter(same_idea_pairs=frozenset({
        (_PIMPLES_MONOLITH, _PIMPLES_RESTATEMENT),
    }))
    result, diagnostics = split_incohesive_retry_groups((group,), takes, arbiter=arbiter)
    result_sets = {frozenset(g) for g in result}
    assert frozenset({"acne"}) in result_sets
    assert frozenset({"monolith", "restatement"}) in result_sets
    assert frozenset({"short"}) in result_sets
    assert len(result) == 3


# --- Diagnostics observability ----------------------------------------------

def test_content_divergence_blocked_diagnostics_present_even_with_no_multi_member_groups():
    result, diagnostics = split_incohesive_retry_groups((), (), arbiter=None)
    assert result == ()
    assert diagnostics["content_divergence_blocked"] == []
    assert diagnostics["content_divergence_blocked_count"] == 0


def test_content_divergence_blocked_diagnostics_populated_on_a_real_block():
    left_text = "I had acne on my back and treated it with resorcinol."
    right_text = "Another symptom was hormonal pimples behind my ears and along my neck."
    takes = (_take("acne", 0.0, 2.0, left_text), _take("pimples", 15.0, 17.0, right_text))
    arbiter = FixedArbiter(same_idea_pairs=frozenset({(left_text, right_text)}))
    _, diagnostics = split_incohesive_retry_groups((("acne", "pimples"),), takes, arbiter=arbiter)
    assert diagnostics["content_divergence_blocked_count"] == 1
    blocked = diagnostics["content_divergence_blocked"][0]
    assert {blocked["left_clip_id"], blocked["right_clip_id"]} == {"acne", "pimples"}


# --- Unit-level check on the new gate function itself -----------------------

def test_within_group_gate_requires_exactly_one_side_marked():
    unmarked_a = "I had acne on my back."
    unmarked_b = "I also had pimples behind my ear."
    take_map = {
        "a": _take("a", 0.0, 1.0, unmarked_a),
        "b": _take("b", 1.0, 2.0, unmarked_b),
    }
    # Neither side marked -> gate never fires, regardless of content overlap.
    assert _within_group_arbiter_confirmation_diverges(take_map, "a", "b") is False
