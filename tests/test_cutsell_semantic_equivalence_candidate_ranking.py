"""Targeted coverage for the priority-ranked candidate-pair fix.

An offline audit of a real RAW run found the root cause of low semantic-
equivalence coverage: _cross_group_candidate_pairs enumerated ALL eligible
group-index pairs in plain chronological order, and reconcile_semantic_idea_
equivalence truncated to the batch budget by array position -- so on a video
dense enough to exceed the budget, pairs late in the timeline were
systematically less likely to ever be proposed to the arbiter, regardless of
how obvious a retry they were. These tests pin the fix: ranking by priority
(temporal proximity + raw lexical overlap + continuation/restart evidence)
before truncating, not truncating in enumeration order.
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceResult,
    SemanticEquivalenceGatePolicy,
)
from cutsell_worker.take_grouping_provider import (
    _continuation_or_restart_bonus,
    _cross_group_candidate_pairs,
    _pair_priority_score,
    _raw_content_overlap,
    _rank_candidate_pairs,
    reconcile_semantic_idea_equivalence,
)


def _take(clip_id, start, end, text, *, complete=True, source="src", order=0):
    return CandidateTake(clip_id, source, order, start, end, text, complete_idea=complete)


class ConfirmAllArbiter:
    """Confirms same_idea for every pair it is actually asked about --
    lets tests isolate WHICH pairs made it into the request, not whether
    the arbiter itself classifies correctly (that's covered elsewhere)."""

    def __init__(self):
        self.last_request = None

    def check(self, request):
        self.last_request = request
        decisions = tuple(
            IdeaEquivalenceDecision(pair_index=i, same_idea=True, confidence=0.9, reason="confirmed")
            for i in range(len(request.pairs))
        )
        return IdeaEquivalenceResult(
            decisions=decisions, provider="fake", model="fake", requested=True, available=True,
        )


def test_raw_content_overlap_distinguishes_low_overlap_from_zero_overlap():
    # retry_similarity()'s own containment<0.60 floor would score both of
    # these identically at 0.0 -- this raw signal must not collapse them,
    # since collapsing is exactly why a hard similarity gate was rejected.
    zero = _raw_content_overlap(
        "we launched the new product line today",
        "packaging looks great up close",
    )
    low = _raw_content_overlap(
        "we launched the new product line today",
        "today marks the official debut of our newest product",
    )
    assert zero == 0.0
    assert low > zero


def test_continuation_bonus_rewards_incomplete_take_and_prefix_relationship():
    complete_a = _take("a", 0.0, 2.0, "we launched the new product line today")
    complete_b = _take("b", 3.0, 5.0, "here is how the packaging looks up close")
    incomplete = _take("c", 0.0, 1.0, "we launched the new—", complete=False)
    prefix = _take("d", 0.0, 1.0, "we launched")
    full = _take("e", 0.0, 2.0, "we launched the new product line today with a big event")

    assert _continuation_or_restart_bonus(complete_a, complete_b) == 0.0
    assert _continuation_or_restart_bonus(incomplete, complete_a) > 0.0
    assert _continuation_or_restart_bonus(prefix, full) > 0.0


def test_pair_priority_score_favors_closer_more_overlapping_pairs():
    take_a = _take("a", 0.0, 2.0, "we launched the new product line today")
    close_match = _take("b", 3.0, 5.0, "today we finally launched our new product line")
    far_unrelated = _take("c", 25.0, 27.0, "here is how the packaging looks up close")

    close_score = _pair_priority_score(take_a, close_match, gap_sec=1.0)
    far_score = _pair_priority_score(take_a, far_unrelated, gap_sec=20.0)
    assert close_score > far_score


def test_rank_candidate_pairs_reorders_low_priority_pair_first_input_behind():
    # Construct raw pairs in an order that deliberately puts a low-priority
    # (far, no overlap) pair FIRST and a high-priority (close, overlapping)
    # pair LAST -- exactly the appearance-order bias the audit found.
    takes = {
        "a": _take("a", 0.0, 2.0, "we launched the new product line today"),
        "b": _take("b", 3.0, 5.0, "today we finally launched our new product line"),
        "x": _take("x", 0.0, 2.0, "unrelated filler content about something else"),
        "y": _take("y", 25.0, 27.0, "another unrelated filler statement entirely"),
    }
    raw_pairs = ((5, 6, "x", "y"), (0, 1, "a", "b"))
    ranked = _rank_candidate_pairs(raw_pairs, takes)
    assert ranked[0] == (0, 1, "a", "b")


def test_reconcile_prioritizes_late_high_priority_pair_over_early_low_priority_ones_under_budget():
    # 5 groups: group 0 is far from everything and shares no vocabulary with
    # anything (low priority, but enumerated FIRST against every other
    # group by _cross_group_candidate_pairs's left-to-right loop). Group 3
    # and group 4 are close together and share vocabulary (high priority,
    # but enumerated LAST). With a budget of exactly 1 pair, the fix must
    # spend it on the high-priority pair, not whichever came first.
    groups = (("g0",), ("g1",), ("g2",), ("g3",), ("g4",))
    takes = (
        _take("g0", 0.0, 2.0, "totally unrelated filler statement about nothing"),
        _take("g1", 5.0, 7.0, "another unrelated filler statement here too"),
        _take("g2", 10.0, 12.0, "yet another unrelated filler statement present"),
        _take("g3", 15.0, 17.0, "we launched the new product line today"),
        _take("g4", 18.0, 20.0, "today we finally launched our new product line"),
    )
    arbiter = ConfirmAllArbiter()
    policy = SemanticEquivalenceGatePolicy(max_pairs_per_request=1)

    merged, diagnostics = reconcile_semantic_idea_equivalence(groups, takes, arbiter, policy=policy)

    assert diagnostics["checked_pair_count"] == 1
    checked_ids = {arbiter.last_request.pairs[0].left_text, arbiter.last_request.pairs[0].right_text}
    assert checked_ids == {takes[3].text, takes[4].text}
    assert ("g3", "g4") in merged
