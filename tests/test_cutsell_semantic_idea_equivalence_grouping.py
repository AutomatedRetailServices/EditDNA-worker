from cutsell_worker.contracts import CandidateTake
from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceResult,
    SemanticEquivalenceGatePolicy,
)
from cutsell_worker.take_grouping_provider import (
    _cross_group_candidate_pairs,
    reconcile_semantic_idea_equivalence,
    safe_group_takes,
)


class FixedArbiter:
    """Confirms same_idea only for checked pairs whose exact (left, right) text
    appears in `same_idea_pairs` -- matching on the full pair, not just one
    side, so a test can pin exactly which cross-group pair merges."""

    def __init__(self, same_idea_pairs=frozenset(), confidence: float = 0.9):
        self.same_idea_pairs = same_idea_pairs
        self.confidence = confidence
        self.calls = 0
        self.last_request = None

    def check(self, request):
        self.calls += 1
        self.last_request = request
        decisions = tuple(
            IdeaEquivalenceDecision(
                pair_index=i,
                same_idea=(pair.left_text, pair.right_text) in self.same_idea_pairs,
                confidence=self.confidence,
                reason="fake evidence",
            )
            for i, pair in enumerate(request.pairs)
        )
        return IdeaEquivalenceResult(
            decisions=decisions,
            provider="fake",
            model="fake-semantic-equivalence",
            requested=True,
            available=True,
            estimated_input_tokens=100,
            estimated_output_tokens=20,
        )


class NoneAvailableArbiter:
    """Simulates provider failure/uncertainty: always fails open (unavailable)."""

    def __init__(self):
        self.calls = 0

    def check(self, request):
        self.calls += 1
        raise RuntimeError("provider down")


def _take(clip_id, start, end, text, source="src", order=0):
    return CandidateTake(clip_id, source, order, start, end, text)


def test_cross_group_pairs_skips_pairs_beyond_gap_ceiling():
    groups = (("a",), ("b",))
    takes = (
        _take("a", 0.0, 2.0, "we launched the new product line today"),
        _take("b", 40.0, 42.0, "we launched the new product line today again"),
    )
    take_map = {t.clip_id: t for t in takes}
    pairs = _cross_group_candidate_pairs(groups, take_map, maximum_gap_sec=30.0)
    assert pairs == ()


def test_cross_group_pairs_skips_short_phrase_floor():
    groups = (("a",), ("b",))
    takes = (
        _take("a", 0.0, 1.0, "yeah okay"),
        _take("b", 2.0, 3.0, "sure thing"),
    )
    take_map = {t.clip_id: t for t in takes}
    pairs = _cross_group_candidate_pairs(groups, take_map, maximum_gap_sec=30.0)
    assert pairs == ()


def test_cross_group_pairs_skips_cross_source():
    groups = (("a",), ("b",))
    takes = (
        _take("a", 0.0, 2.0, "we launched the new product line today", source="src-1"),
        _take("b", 3.0, 5.0, "we launched the new product line today", source="src-2"),
    )
    take_map = {t.clip_id: t for t in takes}
    pairs = _cross_group_candidate_pairs(groups, take_map, maximum_gap_sec=30.0)
    assert pairs == ()


def test_cross_group_pairs_eligible_within_gap_same_source_long_enough():
    groups = (("a",), ("b",))
    takes = (
        _take("a", 0.0, 2.0, "we launched the new product line today"),
        _take("b", 5.0, 7.0, "today we finally launched our new product line"),
    )
    take_map = {t.clip_id: t for t in takes}
    pairs = _cross_group_candidate_pairs(groups, take_map, maximum_gap_sec=30.0)
    assert pairs == ((0, 1, "a", "b"),)


def test_reconcile_merges_groups_arbiter_confirms_same_idea():
    groups = (("a",), ("b",), ("c",))
    takes = (
        _take("a", 0.0, 2.0, "we launched the new product line today"),
        _take("b", 5.0, 7.0, "today we finally launched our new product line"),
        _take("c", 10.0, 12.0, "here is how the packaging looks up close"),
    )
    arbiter = FixedArbiter(same_idea_pairs={
        ("we launched the new product line today", "today we finally launched our new product line"),
    })
    merged, diagnostics = reconcile_semantic_idea_equivalence(groups, takes, arbiter)
    assert diagnostics["status"] == "applied"
    assert diagnostics["merged_pair_count"] == 1
    assert set(merged) == {("a", "b"), ("c",)}


def test_reconcile_never_remerges_protected_composite_pieces_even_if_arbiter_confirms_same_idea():
    # D-025: RAW 33366538992 -- CompositeResolver already accepted "a" and
    # "b" as a two-piece composite and forced them into singleton groups.
    # This step's OWN, separate arbiter call would otherwise confirm they
    # are the same idea (true! that's exactly why they're a composite) and
    # re-merge them into one ordinary retry contest -- which a third clip
    # can then win outright, discarding both accepted composite pieces.
    # protected_ids must make that impossible.
    groups = (("a",), ("b",), ("c",))
    takes = (
        _take("a", 0.0, 2.0, "también me salían espinillas era como una alergia"),
        _take("b", 5.0, 7.0, "otro síntoma que tenía eran espinillas detrás de la oreja"),
        _take("c", 10.0, 12.0, "por temporada me salió un acné en la espalda"),
    )
    arbiter = FixedArbiter(same_idea_pairs={
        ("también me salían espinillas era como una alergia", "otro síntoma que tenía eran espinillas detrás de la oreja"),
        ("también me salían espinillas era como una alergia", "por temporada me salió un acné en la espalda"),
    })

    merged, diagnostics = reconcile_semantic_idea_equivalence(
        groups, takes, arbiter, protected_ids=frozenset({"a", "b"}),
    )

    assert diagnostics["status"] == "no_eligible_pairs"
    assert set(merged) == {("a",), ("b",), ("c",)}
    assert arbiter.calls == 0  # never even asked -- protected pairs are filtered before the call


def test_reconcile_protects_composite_pieces_from_merging_into_unrelated_groups_too():
    # Not just from re-merging with each other -- a protected clip must not
    # merge into ANY other group either, since that would also silently
    # remove it from its accepted composite.
    groups = (("a",), ("c",))
    takes = (
        _take("a", 0.0, 2.0, "también me salían espinillas era como una alergia"),
        _take("c", 10.0, 12.0, "por temporada me salió un acné en la espalda"),
    )
    arbiter = FixedArbiter(same_idea_pairs={
        ("también me salían espinillas era como una alergia", "por temporada me salió un acné en la espalda"),
    })

    merged, diagnostics = reconcile_semantic_idea_equivalence(
        groups, takes, arbiter, protected_ids=frozenset({"a"}),
    )

    assert diagnostics["status"] == "no_eligible_pairs"
    assert set(merged) == {("a",), ("c",)}


def test_reconcile_never_merges_when_arbiter_says_different_idea():
    groups = (("a",), ("b",))
    takes = (
        _take("a", 0.0, 2.0, "we launched the new product line today"),
        _take("b", 5.0, 7.0, "here is how the packaging looks up close"),
    )
    arbiter = FixedArbiter(same_idea_pairs=frozenset())
    merged, diagnostics = reconcile_semantic_idea_equivalence(groups, takes, arbiter)
    assert diagnostics["status"] == "checked_no_merge"
    assert merged == groups


def test_reconcile_fails_open_preserving_groups_when_arbiter_is_none():
    groups = (("a",), ("b",))
    takes = (
        _take("a", 0.0, 2.0, "we launched the new product line today"),
        _take("b", 5.0, 7.0, "today we finally launched our new product line"),
    )
    merged, diagnostics = reconcile_semantic_idea_equivalence(groups, takes, None)
    assert merged == groups
    assert diagnostics["status"] == "not_requested"


def test_reconcile_fails_open_preserving_groups_when_provider_errors():
    groups = (("a",), ("b",))
    takes = (
        _take("a", 0.0, 2.0, "we launched the new product line today"),
        _take("b", 5.0, 7.0, "today we finally launched our new product line"),
    )
    arbiter = NoneAvailableArbiter()
    merged, diagnostics = reconcile_semantic_idea_equivalence(groups, takes, arbiter)
    assert merged == groups
    assert diagnostics["status"] == "arbiter_unavailable"
    assert arbiter.calls == 1


def test_reconcile_no_eligible_pairs_when_only_one_group():
    takes = (_take("a", 0.0, 2.0, "we launched the new product line today"),)
    merged, diagnostics = reconcile_semantic_idea_equivalence((("a",),), takes, FixedArbiter())
    assert merged == (("a",),)
    assert diagnostics["status"] == "not_requested"


def test_reconcile_transitively_merges_three_groups_via_union_find():
    groups = (("a",), ("b",), ("c",))
    takes = (
        _take("a", 0.0, 2.0, "we launched the new product line today"),
        _take("b", 5.0, 7.0, "today we finally launched our new product line"),
        _take("c", 10.0, 12.0, "we launched our new product line earlier today"),
    )
    a_text, b_text, c_text = takes[0].text, takes[1].text, takes[2].text
    # Arbiter confirms a<->b and b<->c as same idea, but NOT a<->c; union-find
    # must still merge all three transitively through the shared b link.
    arbiter = FixedArbiter(same_idea_pairs={(a_text, b_text), (b_text, c_text)})
    merged, diagnostics = reconcile_semantic_idea_equivalence(groups, takes, arbiter)
    assert diagnostics["merged_pair_count"] >= 1
    assert len(merged) == 1
    assert set(merged[0]) == {"a", "b", "c"}


def test_reconcile_truncates_candidate_pairs_to_policy_maximum():
    groups = tuple((clip_id,) for clip_id in "abcd")
    takes = tuple(
        _take(clip_id, float(i) * 3.0, float(i) * 3.0 + 2.0, f"we launched the new product line today variant {i}")
        for i, clip_id in enumerate("abcd")
    )
    arbiter = FixedArbiter(same_idea_pairs=frozenset())
    policy = SemanticEquivalenceGatePolicy(max_pairs_per_request=2)
    _, diagnostics = reconcile_semantic_idea_equivalence(groups, takes, arbiter, policy=policy)
    assert diagnostics["checked_pair_count"] == 2
    assert len(arbiter.last_request.pairs) == 2


def test_reconcile_blocks_merge_when_one_side_signals_a_distinct_additional_item():
    # Offline audit of RAW 33432104336: the arbiter confirmed "otro sintoma
    # ..." ("ANOTHER symptom...") as the same idea as an earlier, unrelated
    # pimples mention -- purely topical similarity, missing the speaker's
    # own explicit "this is a different/additional point" discourse marker.
    # No protected_ids involved here (that guard only fires once
    # CompositeResolver has already recognized these as composite pieces --
    # this general marker-based guard is independent of that and fires even
    # when nothing upstream has flagged anything).
    groups = (("a",), ("b",))
    takes = (
        _take("a", 0.0, 2.0, "también me salían espinillas era como una alergia"),
        _take("b", 5.0, 7.0, "otro síntoma que tenía eran espinillas detrás de la oreja"),
    )
    arbiter = FixedArbiter(same_idea_pairs={
        ("también me salían espinillas era como una alergia", "otro síntoma que tenía eran espinillas detrás de la oreja"),
    })

    merged, diagnostics = reconcile_semantic_idea_equivalence(groups, takes, arbiter)

    assert diagnostics["status"] == "checked_no_merge"
    assert set(merged) == {("a",), ("b",)}
    assert len(diagnostics["distinct_addition_blocked"]) == 1
    assert diagnostics["distinct_addition_blocked"][0]["left_clip_id"] == "a"
    assert diagnostics["distinct_addition_blocked"][0]["right_clip_id"] == "b"


def test_reconcile_still_merges_when_neither_side_has_a_distinct_addition_marker():
    # The guard must not become a blanket ban on merging -- an ordinary
    # paraphrased retry with no "another X" marker on either side merges
    # exactly as before.
    groups = (("a",), ("b",))
    takes = (
        _take("a", 0.0, 2.0, "we launched the new product line today"),
        _take("b", 5.0, 7.0, "today we finally launched our new product line"),
    )
    arbiter = FixedArbiter(same_idea_pairs={
        ("we launched the new product line today", "today we finally launched our new product line"),
    })

    merged, diagnostics = reconcile_semantic_idea_equivalence(groups, takes, arbiter)

    assert diagnostics["status"] == "applied"
    assert diagnostics["distinct_addition_blocked"] == []
    assert ("a", "b") in merged


def test_reconcile_still_merges_when_both_sides_have_a_distinct_addition_marker():
    # Both sides equally signalling "another point" is not evidence of a
    # DIFFERENCE between the two -- only an imbalance (one side marked, the
    # other not) is. This must not become a false block on, say, two
    # attempts at introducing the SAME "another symptom" transition.
    groups = (("a",), ("b",))
    takes = (
        _take("a", 0.0, 2.0, "another symptom I had was joint pain in my knees"),
        _take("b", 5.0, 7.0, "one more thing, I also had joint pain in my knees"),
    )
    arbiter = FixedArbiter(same_idea_pairs={
        ("another symptom I had was joint pain in my knees", "one more thing, I also had joint pain in my knees"),
    })

    merged, diagnostics = reconcile_semantic_idea_equivalence(groups, takes, arbiter)

    assert diagnostics["status"] == "applied"
    assert diagnostics["distinct_addition_blocked"] == []
    assert ("a", "b") in merged


def test_reconcile_applied_after_safe_group_takes_strengthens_lexical_grouping():
    # Deliberately low lexical overlap so the baseline local grouper leaves
    # these as two separate singleton groups -- only the semantic-equivalence
    # arbiter can merge them, proving Phase 2 actually strengthens grouping
    # for a "same idea, very different wording" pair the lexical layer alone
    # cannot catch (see reconcile_semantic_idea_equivalence's docstring).
    # This mirrors pipeline.py's own integration: reconcile runs directly on
    # safe_group_takes's resolved output, not threaded through it as a
    # parameter (see safe_group_takes's docstring for why).
    left_text = "we launched the new product line today"
    right_text = "today marks the official debut of our newest product"
    takes = (
        _take("a", 0.0, 2.0, left_text),
        _take("b", 25.0, 27.0, right_text),
    )
    baseline_only = safe_group_takes(None, takes)
    assert baseline_only.groups == (("a",), ("b",))

    arbiter = FixedArbiter(same_idea_pairs={(left_text, right_text)})
    merged, diagnostics = reconcile_semantic_idea_equivalence(baseline_only.groups, takes, arbiter)
    assert arbiter.calls == 1
    assert diagnostics.get("status") == "applied"
    assert ("a", "b") in merged
