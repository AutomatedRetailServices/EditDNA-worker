"""D-058 Phase 1 -- DISTINCT-IDEA GROUPING SAFETY.

Root defect (docs/CUTSELL_DECISIONS.md D-057's forensic on the D-056.6
pimples/acne shape): a deterministic multi-member `take_judge_groups` group
is trusted as one mutually-exclusive retry family with no re-validation of
that assumption anywhere downstream -- `reconcile_semantic_idea_equivalence`
only ever MERGES groups across boundaries, never re-examines an already-
multi-member group's own internal cohesion. Fixed by
`split_incohesive_retry_groups`: every pair inside an already-multi-member
group must show strong deterministic lexical evidence of being the same
retry, OR explicit arbiter confirmation for that specific pair -- neither
temporal proximity nor shared vocabulary alone is ever sufficient.

Entirely generic -- no Video00 clip ids or phrases; the back-acne-vs-pimples
shape is reproduced with a paraphrased, non-Video00 "skin symptom" pair.
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceResult,
)
from cutsell_worker.take_grouping_provider import split_incohesive_retry_groups


class FixedArbiter:
    """Confirms same_idea only for checked pairs whose exact (left, right)
    text appears in `same_idea_pairs` -- same fixture shape already used by
    test_cutsell_semantic_idea_equivalence_grouping.py."""

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


class NoneAvailableArbiter:
    """Simulates provider failure/uncertainty: always fails open (unavailable)."""

    def check(self, request):
        raise RuntimeError("provider down")


def _take(clip_id, start, end, text, source="src", order=0):
    return CandidateTake(clip_id, source, order, start, end, text)


# --- 1. Same symptom, same proposition retry -> stays grouped --------------

def test_same_proposition_retry_stays_grouped():
    takes = (
        _take("a", 0.0, 2.0, "I had a rash on my back that flared up every summer"),
        _take("b", 2.5, 4.5, "I used to get a rash on my back that flared up every summer"),
    )
    groups = (("a", "b"),)
    result, diagnostics = split_incohesive_retry_groups(groups, takes, arbiter=None)
    assert result == (("a", "b"),)
    assert diagnostics["groups_split"] == 0


# --- 2. Related topic but distinct symptom -> separated ---------------------

def test_related_topic_distinct_symptom_is_separated():
    takes = (
        _take("back", 0.0, 3.0, "I had seasonal back acne that I treated with an ointment"),
        _take("neck", 3.5, 6.5, "I also got small bumps behind my ear and neck I thought were an allergy"),
    )
    groups = (("back", "neck"),)
    # No arbiter available -- deterministic evidence between these two is weak
    # (different specific symptom and location), so they must not compete.
    result, diagnostics = split_incohesive_retry_groups(groups, takes, arbiter=None)
    assert set(result) == {("back",), ("neck",)}
    assert diagnostics["groups_split"] == 1


# --- 3. Temporal proximity alone is insufficient ----------------------------

def test_temporal_proximity_alone_is_insufficient():
    takes = (
        _take("back", 0.0, 2.0, "I had seasonal back acne that I treated with an ointment"),
        # Immediately adjacent in time (no gap at all), but a distinct claim.
        _take("neck", 2.0, 4.0, "I also got small bumps behind my ear and neck I thought were an allergy"),
    )
    groups = (("back", "neck"),)
    result, diagnostics = split_incohesive_retry_groups(groups, takes, arbiter=None)
    assert set(result) == {("back",), ("neck",)}


# --- 4. Shared vocabulary alone is insufficient -----------------------------

def test_shared_vocabulary_alone_is_insufficient():
    takes = (
        _take(
            "skin_a", 0.0, 2.0,
            "My skin would break out on my back every summer and I used an ointment for it",
        ),
        # Shares only generic connective vocabulary ("skin", "around")
        # with skin_a -- a different, unrelated symptom and body part, not
        # a retry of the same statement.
        _take(
            "skin_b", 10.0, 12.0,
            "I noticed some dry patches on my elbows around the same time but never really did anything about them",
        ),
    )
    groups = (("skin_a", "skin_b"),)
    result, diagnostics = split_incohesive_retry_groups(groups, takes, arbiter=None)
    assert set(result) == {("skin_a",), ("skin_b",)}


# --- 5. Semantic arbiter confirms retry -> stays grouped --------------------

def test_arbiter_confirmed_retry_stays_grouped():
    left_text = "I had seasonal back acne that I treated with an ointment"
    right_text = "Every season I would get back breakouts and I used an ointment for it"
    takes = (
        _take("a", 0.0, 2.0, left_text),
        _take("b", 15.0, 17.0, right_text),
    )
    groups = (("a", "b"),)
    arbiter = FixedArbiter(same_idea_pairs=frozenset({(left_text, right_text)}))
    result, diagnostics = split_incohesive_retry_groups(groups, takes, arbiter=arbiter)
    assert result == (("a", "b"),)
    assert arbiter.calls == 1
    assert diagnostics["arbiter_confirmed_pairs"]


# --- 6. Arbiter does not confirm -> separate when deterministic evidence ----
#        is weak (arbiter available but declines/unavailable) ---------------

def test_arbiter_unavailable_separates_when_deterministic_evidence_weak():
    takes = (
        _take("back", 0.0, 2.0, "I had seasonal back acne that I treated with an ointment"),
        _take("neck", 3.0, 5.0, "I also got small bumps behind my ear and neck I thought were an allergy"),
    )
    groups = (("back", "neck"),)
    result, diagnostics = split_incohesive_retry_groups(groups, takes, arbiter=NoneAvailableArbiter())
    assert set(result) == {("back",), ("neck",)}


def test_arbiter_explicitly_declines_separates_the_pair():
    takes = (
        _take("back", 0.0, 2.0, "I had seasonal back acne that I treated with an ointment"),
        _take("neck", 3.0, 5.0, "I also got small bumps behind my ear and neck I thought were an allergy"),
    )
    groups = (("back", "neck"),)
    arbiter = FixedArbiter(same_idea_pairs=frozenset())  # confirms nothing
    result, diagnostics = split_incohesive_retry_groups(groups, takes, arbiter=arbiter)
    assert set(result) == {("back",), ("neck",)}
    assert arbiter.calls == 1


# --- 7. Full D-057 shape: two true retries plus one distinct beat ----------

def test_d057_shape_reproduced_generically_true_retries_stay_distinct_beat_splits():
    """Generic paraphrase of the exact D-057 structure: two real retakes of
    one statement ("back acne treated with resorcina") plus a genuinely
    different, additional symptom ("pimples behind the ear/neck, thought it
    was an allergy") all bundled into one deterministic group -- the true
    retry pair must stay together (single winner contest), the distinct
    beat must be split out so it gets its own independent chance to survive
    Selection rather than losing a contest it was never actually part of."""
    back_1 = "Every season I would get some acne on my back and I treated it with an ointment"
    back_2 = "I also used to get acne on my back sometimes and I treated it with an ointment"
    neck = "Another thing was I would get small pimples behind my ear and down my neck that I thought was an allergy"
    takes = (
        _take("back_1", 0.0, 3.0, back_1),
        _take("back_2", 3.5, 6.5, back_2),
        _take("neck", 7.0, 10.0, neck),
    )
    groups = (("back_1", "back_2", "neck"),)
    # The arbiter, exactly like the live run, only ever confirms the genuine
    # back-acne retry pair -- it is never asked to (or never confirms)
    # merging the neck beat with either back-acne clip as the same idea.
    arbiter = FixedArbiter(same_idea_pairs=frozenset({(back_1, back_2)}))
    result, diagnostics = split_incohesive_retry_groups(groups, takes, arbiter=arbiter)
    groups_by_membership = {frozenset(group) for group in result}
    assert frozenset({"back_1", "back_2"}) in groups_by_membership
    assert frozenset({"neck"}) in groups_by_membership
    assert diagnostics["groups_split"] == 1


# --- 8. Protected (accepted composite) ids are never re-split --------------

def test_protected_ids_are_never_split_even_with_weak_evidence():
    takes = (
        _take("back", 0.0, 2.0, "I had seasonal back acne that I treated with an ointment"),
        _take("neck", 3.0, 5.0, "I also got small bumps behind my ear and neck I thought were an allergy"),
    )
    groups = (("back", "neck"),)
    result, diagnostics = split_incohesive_retry_groups(
        groups, takes, arbiter=None, protected_ids=frozenset({"back", "neck"}),
    )
    assert result == (("back", "neck"),)


# --- 9. Singleton and already-cohesive groups are left untouched -----------

def test_singleton_groups_are_never_touched():
    takes = (_take("solo", 0.0, 2.0, "I had a rash on my back that flared up every summer"),)
    groups = (("solo",),)
    result, diagnostics = split_incohesive_retry_groups(groups, takes, arbiter=None)
    assert result == (("solo",),)
    assert diagnostics["status"] == "not_requested"
