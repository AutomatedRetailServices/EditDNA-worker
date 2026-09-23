"""D-048 FIX 1 -- content-divergence-gated distinct-addition guard.

D-047 Case 1: the D-039 guard (take_grouping_provider._DISTINCT_ADDITION_
MARKERS) blocked a semantic merge purely because exactly one candidate
opened with a "new/additional item" discourse marker ("Otro sintoma..."),
even though the arbiter confirmed same_idea at 0.95 confidence and the two
texts shared the same specific symptom AND the same specific location
("detras de la oreja"/"cuello") -- the marker was a narrative restart into
a restatement, not evidence of a genuinely new point.

The fix does not remove the guard (the founding RAW 33432104336 incident --
two mentions of red spots on two DIFFERENT body parts, arm vs leg -- must
still block). It gates the block on real content divergence: the marked
side must lack enough of the unmarked side's own specific content (beyond
generic scaffolding/stopwords) to trust the marker as evidence of a
distinct point, using the same order-independent bag-of-words machinery
the rest of cutsell_worker already relies on for this kind of comparison
(final_sibling_grouping._content's own pattern, kept local here to avoid a
circular import -- final_sibling_grouping imports FROM this module).
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_grouping_provider import reconcile_semantic_idea_equivalence


class FixedArbiter:
    """Confirms same_idea for every checked pair (this suite always wants
    the arbiter's own verdict to be same_idea=True -- these tests are about
    the guard's OWN decision on top of that, not the arbiter)."""

    def __init__(self, confidence: float = 0.9):
        self.confidence = confidence
        self.calls = 0

    def check(self, request):
        self.calls += 1
        from cutsell_worker.semantic_idea_equivalence import (
            IdeaEquivalenceDecision,
            IdeaEquivalenceResult,
        )
        decisions = tuple(
            IdeaEquivalenceDecision(
                pair_index=i, same_idea=True, confidence=self.confidence, reason="fake evidence",
            )
            for i, _ in enumerate(request.pairs)
        )
        return IdeaEquivalenceResult(
            decisions=decisions, provider="fake", model="fake", requested=True, available=True,
        )


def _take(clip_id, start, end, text, source="src", order=0):
    return CandidateTake(clip_id, source, order, start, end, text)


def _merged(left_text, right_text, *, confidence=0.9):
    groups = (("a",), ("b",))
    takes = (
        _take("a", 0.0, 2.0, left_text),
        _take("b", 5.0, 7.0, right_text),
    )
    arbiter = FixedArbiter(confidence=confidence)
    merged, diagnostics = reconcile_semantic_idea_equivalence(groups, takes, arbiter)
    return len(merged) == 1, diagnostics


# 1. Founding D-039 case: genuinely distinct locations -- remains BLOCKED.

def test_founding_case_distinct_location_still_blocked():
    is_merged, diag = _merged(
        "También tenía manchas rojas en la piel del brazo.",
        "Otro síntoma que noté fueron manchas rojas en la piel de la pierna.",
    )
    assert is_merged is False
    assert diag["status"] == "checked_no_merge"
    assert len(diag["distinct_addition_blocked"]) == 1


# 2. D-047 Case 1 shape: high specific-content overlap -- NOT blocked.

def test_high_overlap_retry_with_marker_is_not_blocked():
    is_merged, diag = _merged(
        "También me salían espinillas en esta parte de aquí detrás de la "
        "oreja y todo el cuello que yo pensaba que era alergia.",
        "Otro síntoma era que me salían espinillas como si fuera una "
        "alergia de esta parte aquí detrás de la oreja y en el cuello.",
    )
    assert is_merged is True
    assert diag["status"] == "applied"
    assert diag.get("distinct_addition_blocked") == []


# 3. English "another symptom" -- same-content retry -- NOT blocked.

def test_english_another_symptom_same_content_retry_not_blocked():
    is_merged, _ = _merged(
        "I noticed breakouts near my jawline behind my ear that felt like an allergy.",
        "Another symptom was breakouts near my jawline behind my ear that felt like an allergy.",
    )
    assert is_merged is True


# 4. English "another symptom" -- genuinely new fact -- stays BLOCKED.

def test_english_another_symptom_genuinely_new_fact_blocked():
    is_merged, diag = _merged(
        "I noticed breakouts near my jawline behind my ear that felt like an allergy.",
        "Another symptom was joint pain in my knees every single morning.",
    )
    assert is_merged is False
    assert len(diag["distinct_addition_blocked"]) == 1


# 5. Additive framing ("on top of that") -- same-content restatement -- NOT blocked.

def test_additive_framing_same_content_restatement_not_blocked():
    is_merged, _ = _merged(
        "I noticed hair thinning around my temples after finishing the treatment.",
        "On top of that, I noticed hair thinning around my temples after finishing the treatment.",
    )
    assert is_merged is True


# 6. Additive framing -- genuine addition -- stays BLOCKED.

def test_additive_framing_genuine_addition_blocked():
    is_merged, _ = _merged(
        "I noticed hair thinning around my temples after finishing the treatment.",
        "On top of that, I started getting migraines every single week.",
    )
    assert is_merged is False


# 7. Both sides marked -- existing behavior preserved (guard never applies;
#    the arbiter's own same_idea verdict decides).

def test_both_sides_marked_existing_behavior_preserved():
    is_merged, diag = _merged(
        "Otro síntoma fue que me dolía la cabeza todos los días.",
        "Otro problema fue que me dolía la cabeza todos los días.",
    )
    assert is_merged is True
    assert diag.get("distinct_addition_blocked", []) == []


# 8. Neither side marked -- existing behavior preserved.

def test_neither_side_marked_existing_behavior_preserved():
    is_merged, diag = _merged(
        "Me dolía la cabeza todos los días después del tratamiento.",
        "Tenía dolores de cabeza diarios después de terminar el tratamiento.",
    )
    assert is_merged is True
    assert diag.get("distinct_addition_blocked", []) == []


# 9. High topical overlap but a different specific noun/entity -- block remains.

def test_high_topical_overlap_different_specific_entity_still_blocked():
    is_merged, diag = _merged(
        "Tenía hinchazón en las manos por las mañanas de forma constante.",
        "Otro síntoma fue hinchazón en los pies por las mañanas de forma constante.",
    )
    assert is_merged is False
    assert len(diag["distinct_addition_blocked"]) == 1


# 10. Strong specific overlap + high arbiter confidence -- merge allowed
#     (confidence is supporting evidence only, overlap decides).

def test_strong_overlap_high_confidence_merge_allowed():
    is_merged, _ = _merged(
        "También me salían espinillas en esta parte de aquí detrás de la "
        "oreja y todo el cuello que yo pensaba que era alergia.",
        "Otro síntoma era que me salían espinillas como si fuera una "
        "alergia de esta parte aquí detrás de la oreja y en el cuello.",
        confidence=0.98,
    )
    assert is_merged is True


# 11. Weak overlap + high arbiter confidence -- guard may still block
#     (high confidence does not rescue a genuinely distinct addition).

def test_weak_overlap_high_confidence_guard_still_blocks():
    is_merged, _ = _merged(
        "Tenía hinchazón en las manos por las mañanas de forma constante.",
        "Otro síntoma fue hinchazón en los pies por las mañanas de forma constante.",
        confidence=0.98,
    )
    assert is_merged is False


# 12. No regression to lexical-only grouping behavior (baseline tier,
#     independent of the arbiter/guard entirely).

def test_lexical_only_grouping_unaffected_by_the_guard_refinement():
    from cutsell_worker.take_grouping_provider import safe_group_takes

    takes = (
        _take("x", 0.0, 3.0, "the exact same delivery repeated verbatim here"),
        _take("y", 4.0, 7.0, "the exact same delivery repeated verbatim here"),
        _take("z", 8.0, 11.0, "a completely unrelated statement about something else"),
    )
    result = safe_group_takes(None, takes)
    group_sizes = sorted(len(group) for group in result.groups)
    assert group_sizes == [1, 2]
