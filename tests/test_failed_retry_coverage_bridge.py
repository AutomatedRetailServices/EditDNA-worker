from cutsell_worker.contracts import CandidateTake
from cutsell_worker.failed_retry_coverage_bridge import (
    _internal_restart_suffix_covered,
    failed_retry_coverage_pairs,
)
from cutsell_worker.hybrid_retry_winner_authority import _same_retry_attempt
from cutsell_worker.semantic_idea_equivalence import IdeaEquivalenceDecision, IdeaEquivalenceResult
from cutsell_worker.take_grouping_provider import reconcile_semantic_idea_equivalence


def take(cid, text, start, end, *, complete=True, source="src"):
    return CandidateTake(cid, source, 0, start, end, text, complete_idea=complete)


FAILED_TEXT = (
    "different salons gave me foot fungus and I recommend this treatment "
    "wait dad I am recording a video for a friend"
)
CLEAN_TEXT = (
    "different salons can give you foot fungus so I recommend this treatment "
    "and here is how to use it"
)


def row(cid, **overrides):
    base = {
        "clip_id": cid,
        "label": "failed",
        "proposed_label": "failed",
        "confidence": 0.95,
        "semantic_delete_recommended": True,
        "local_failure_corroborated": True,
        "dense_semantic_failure_cluster": True,
        "recording_word_ranges": [],
    }
    base.update(overrides)
    return base


class CoverageArbiter:
    def __init__(self, *, confidence=0.95, conflict=False, left_covered=True, same_idea=False):
        self.confidence = confidence
        self.conflict = conflict
        self.left_covered = left_covered
        self.same_idea = same_idea
        self.last_request = None

    def check(self, request):
        self.last_request = request
        return IdeaEquivalenceResult(
            decisions=tuple(
                IdeaEquivalenceDecision(
                    pair_index=index,
                    same_idea=self.same_idea,
                    confidence=self.confidence,
                    reason="failed delivery is covered by the later clean realization",
                    meaning_conflict=self.conflict,
                    left_covered_by_right=self.left_covered,
                    right_covered_by_left=False,
                )
                for index, _ in enumerate(request.pairs)
            ),
            provider="fake",
            model="fake",
            requested=True,
            available=True,
        )


def evidence():
    return ({"decisions": [
        row("failed"),
        row(
            "clean",
            label="uncertain",
            proposed_label="winner",
            semantic_delete_recommended=False,
            local_failure_corroborated=False,
            dense_semantic_failure_cluster=False,
        ),
    ]},)


def test_strong_failed_take_and_later_proposed_winner_become_comparison_pair():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    pairs = failed_retry_coverage_pairs(
        (failed, clean), evidence(), partition_by_id={"failed": 0, "clean": 0},
    )
    assert pairs == frozenset({("failed", "clean")})


def test_complete_failed_pitch_with_large_shared_core_can_reach_coverage_arbiter():
    failed = take(
        "failed",
        "if you visited many salons they caused fungus on your feet because tools were not washed "
        "I have the solution first stop changing salons use your own product second buy ClearFoot "
        "put it on the sole toes and eradicate it",
        20, 39, complete=True,
    )
    clean = take(
        "clean",
        "has this happened to you salons infected you with foot fungus because they fail to disinfect "
        "beauty tools keep your own pedicure kit and use ClearFoot on the sole of the foot to eradicate "
        "fungus apply balm nightly then click the cart for delivery",
        70, 101, complete=True,
    )
    windows = ({"partition_index": 0, "member_ids": ["failed", "clean"], "decisions": [
        row("failed", confidence=0.90),
        row("clean", label="winner", proposed_label="winner", confidence=0.95,
            semantic_delete_recommended=False),
    ]},)
    same_attempt, evidence_row = _same_retry_attempt(failed, clean)
    assert same_attempt is False
    assert evidence_row["shared_count"] >= 8
    assert 0.30 <= evidence_row["failed_coverage"] < 0.45
    pairs = failed_retry_coverage_pairs(
        (failed, clean), windows, partition_by_id={"failed": 0, "clean": 0},
    )
    assert pairs == frozenset({("failed", "clean")})
    merged, diagnostics = reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",)), (failed, clean), CoverageArbiter(confidence=0.90),
        failed_retry_coverage_pairs=pairs,
    )
    assert merged == (("failed", "clean"),)
    assert diagnostics["merges"][0]["accepted_by"] == "failed_attempt_directional_coverage"


def test_complete_candidate_with_internal_restart_suffix_can_reach_coverage_arbiter():
    failed = take(
        "failed",
        "if you are like me that you... if this happened to you from visiting different salons",
        20, 28, complete=True,
    )
    clean = take(
        "clean",
        "if this happened to you because they infected you with fungus from visiting different salons use your own kit",
        70, 92, complete=True,
    )
    windows = ({"partition_index": 0, "member_ids": ["failed", "clean"], "decisions": [
        row("failed", confidence=0.90),
        row("clean", label="winner", proposed_label="winner", confidence=0.95,
            semantic_delete_recommended=False),
    ]},)
    assert failed_retry_coverage_pairs(
        (failed, clean), windows, partition_by_id={"failed": 0, "clean": 0},
    ) == frozenset({("failed", "clean")})


def test_internal_restart_suffix_requires_long_ordered_coverage_and_preserves_numbers():
    clean = take(
        "clean", "if this happened to you from visiting two different salons use your own kit tonight",
        70, 92, complete=True,
    )
    unsafe = (
        "unrelated setup... if this happened",  # generic short suffix
        "unrelated setup... salons different visiting from you to happened this if",  # reordered
        "unrelated setup... if this happened to you from visiting three different salons",  # changed fact
    )
    for index, text in enumerate(unsafe):
        failed = take(f"failed-{index}", text, 20, 28, complete=True)
        assert _internal_restart_suffix_covered(failed, clean) is False


def test_confirmed_retry_family_can_supply_anchor_for_second_pass_clean_delivery():
    intro = take("intro", "many salons can give your feet fungus and I have a solution", 5, 13)
    failed = take(
        "failed", "many salons gave me fungus use ClearFoot on your feet and then...", 27, 46,
    )
    summary = take(
        "summary", "salons gave you fungus I recommend ClearFoot with this you can", 58, 64,
        complete=False,
    )
    clean = take(
        "clean", "salons gave you fungus I recommend ClearFoot with this you can eradicate it and order today",
        101, 132,
    )
    windows = ({"partition_index": 0, "member_ids": ["intro", "failed", "summary", "clean"], "decisions": [
        row("intro", label="alternate", proposed_label="alternate", confidence=0.80,
            semantic_delete_recommended=False),
        row("failed", confidence=0.85),
        row("summary", label="alternate", proposed_label="alternate", confidence=0.80,
            semantic_delete_recommended=False),
        row("clean", label="winner", proposed_label="winner", confidence=0.95,
            semantic_delete_recommended=False),
    ]},)
    # The first pass can establish this early family semantically. On the
    # bounded recovery pass its literal summary member supplies the relation
    # anchor, while the arbiter still receives the complete original texts.
    pairs = failed_retry_coverage_pairs(
        (intro, failed, summary, clean), windows,
        partition_by_id={cid: 0 for cid in ("intro", "failed", "summary", "clean")},
        retry_groups=(("intro", "failed", "summary"), ("clean",)),
    )
    assert ("summary", "clean") in pairs


def test_coverage_only_reconcile_never_reasks_or_merges_ordinary_pairs():
    failed = take("failed", "this backpack can hold", 0, 4, complete=False)
    clean = take("clean", "this backpack can hold two laptops", 20, 28)
    ordinary = take("ordinary", "this backpack comes in blue", 30, 36)
    arbiter = CoverageArbiter(confidence=0.95, left_covered=True, same_idea=True)
    groups, diagnostics = reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",), ("ordinary",)),
        (failed, clean, ordinary),
        arbiter,
        failed_retry_coverage_pairs=frozenset({("failed", "clean")}),
        coverage_pairs_only=True,
    )
    assert len(arbiter.last_request.pairs) == 1
    assert arbiter.last_request.pairs[0].left_text == failed.text
    assert arbiter.last_request.pairs[0].right_text == clean.text
    assert any(set(group) == {"failed", "clean"} for group in groups)
    assert ("ordinary",) in groups
    assert diagnostics["candidate_pair_count"] == 1


def test_shared_core_fallback_is_not_available_for_small_overlap_or_numeric_conflict():
    failed = take(
        "failed", "salons spread 3 foot fungus problems because tools are not cleaned use this product on the feet",
        20, 39, complete=True,
    )
    candidates = (
        take("small", "salons discuss fungus and this completely different price offer", 70, 105, complete=True),
        take("number", "salons spread 2 foot fungus problems because tools are not cleaned use this product on the feet every night with a complete kit and delivery", 70, 105, complete=True),
    )
    for clean in candidates:
        windows = ({"partition_index": 0, "member_ids": ["failed", clean.clip_id], "decisions": [
            row("failed", confidence=0.90),
            row(clean.clip_id, label="winner", proposed_label="winner", confidence=0.95,
                semantic_delete_recommended=False),
        ]},)
        assert not failed_retry_coverage_pairs(
            (failed, clean), windows,
            partition_by_id={"failed": 0, clean.clip_id: 0},
        )


def test_consistent_failed_windows_allow_bounded_confidence_and_one_delete_vote():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    windows = (
        {"partition_index": 0, "member_ids": ["failed", "clean"], "decisions": [
            row("failed", confidence=0.85),
            row(
                "clean", label="winner", proposed_label="winner",
                semantic_delete_recommended=False, confidence=0.95,
            ),
        ]},
        {"partition_index": 0, "member_ids": ["failed", "clean"], "decisions": [
            row("failed", confidence=0.80, semantic_delete_recommended=False),
            row(
                "clean", label="winner", proposed_label="winner",
                semantic_delete_recommended=False, confidence=0.95,
            ),
        ]},
    )
    assert failed_retry_coverage_pairs(
        (failed, clean), windows, partition_by_id={"failed": 0, "clean": 0},
    ) == frozenset({("failed", "clean")})


def test_failed_bridge_uses_corroborated_failed_label_for_comparison_only():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    clean_row = row(
        "clean", label="winner", proposed_label="winner",
        semantic_delete_recommended=False, confidence=0.95,
    )
    no_delete = ({"decisions": [
        row("failed", confidence=0.90, semantic_delete_recommended=False), clean_row,
    ]},)
    assert failed_retry_coverage_pairs(
        (failed, clean), no_delete, partition_by_id={"failed": 0, "clean": 0},
    ) == frozenset({("failed", "clean")})


def test_failed_bridge_still_rejects_uncorroborated_or_sub_point_eight_failure():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    clean_row = row(
        "clean", label="winner", proposed_label="winner",
        semantic_delete_recommended=False, confidence=0.95,
    )
    uncorroborated = ({"decisions": [
        row("failed", confidence=0.90, local_failure_corroborated=False), clean_row,
    ]},)
    low_confidence_view = ({"decisions": [
        row("failed", confidence=0.90), clean_row,
    ]}, {"decisions": [
        row("failed", confidence=0.79, semantic_delete_recommended=False), clean_row,
    ]})
    weak_delete_vote = ({"decisions": [
        row("failed", confidence=0.79), clean_row,
    ]},)
    for windows in (uncorroborated, low_confidence_view, weak_delete_vote):
        assert not failed_retry_coverage_pairs(
            (failed, clean), windows, partition_by_id={"failed": 0, "clean": 0},
        )


def test_complete_substantially_fuller_alternate_can_be_coverage_candidate():
    failed = take(
        "failed", "different salons can give you foot fungus so I recommend this treatment",
        0, 10, complete=False,
    )
    clean = take("clean", CLEAN_TEXT + " with complete application steps and a final call to action", 50, 75)
    windows = ({"partition_index": 0, "member_ids": ["failed", "clean"], "decisions": [
        row("failed", confidence=0.80, semantic_delete_recommended=False),
        row(
            "clean", label="alternate", proposed_label="alternate", confidence=0.75,
            semantic_delete_recommended=False,
            local_failure_corroborated=True, dense_semantic_failure_cluster=True,
        ),
    ]},)
    assert failed_retry_coverage_pairs(
        (failed, clean), windows, partition_by_id={"failed": 0, "clean": 0},
    ) == frozenset({("failed", "clean")})


def test_alternate_fallback_requires_substantial_fullness_and_no_recording_scope():
    failed = take("failed", FAILED_TEXT, 0, 10, complete=False)
    short = take("short", CLEAN_TEXT, 50, 61)
    scoped = take("scoped", CLEAN_TEXT + " with complete instructions and closing", 70, 95)
    for delivery, extra in (
        (short, {}),
        (scoped, {"recording_suffix_words": 2}),
        (scoped, {"content_role": "recording_only"}),
    ):
        windows = ({"partition_index": 0, "member_ids": ["failed", delivery.clip_id], "decisions": [
            row("failed", confidence=0.80, semantic_delete_recommended=False),
            row(
                delivery.clip_id, label="alternate", proposed_label="alternate", confidence=0.75,
                semantic_delete_recommended=False,
                local_failure_corroborated=True, dense_semantic_failure_cluster=True,
                **extra,
            ),
        ]},)
        assert not failed_retry_coverage_pairs(
            (failed, delivery), windows,
            partition_by_id={"failed": 0, delivery.clip_id: 0},
        )


def test_pair_requires_consistent_failure_and_same_session():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    inconsistent = (*evidence(), {"decisions": [row(
        "failed", label="keep", proposed_label="keep",
        semantic_delete_recommended=False,
    )]})
    assert not failed_retry_coverage_pairs(
        (failed, clean), inconsistent, partition_by_id={"failed": 0, "clean": 0},
    )
    assert not failed_retry_coverage_pairs(
        (failed, clean), evidence(), partition_by_id={"failed": 0, "clean": 1},
    )


def test_stronger_failed_vote_dominates_weaker_overlapping_keep_or_alternate():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    clean_row = row(
        "clean", label="winner", proposed_label="winner",
        semantic_delete_recommended=False, confidence=0.95,
    )
    for protective_label in ("keep", "alternate", "uncertain"):
        windows = (
            {"decisions": [
                row("failed", confidence=0.85), clean_row,
            ]},
            {"decisions": [
                row(
                    "failed", label=protective_label,
                    proposed_label=protective_label, confidence=0.70,
                    semantic_delete_recommended=False,
                ),
                clean_row,
            ]},
        )
        assert failed_retry_coverage_pairs(
            (failed, clean), windows, partition_by_id={"failed": 0, "clean": 0},
        ) == frozenset({("failed", "clean")})


def test_equal_or_stronger_protective_vote_still_blocks_destructive_comparison():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    clean_row = row(
        "clean", label="winner", proposed_label="winner",
        semantic_delete_recommended=False, confidence=0.95,
    )
    for protective_label in ("keep", "alternate", "uncertain"):
        for confidence in (0.85, 0.90):
            windows = (
                {"decisions": [row("failed", confidence=0.85), clean_row]},
                {"decisions": [
                    row(
                        "failed", label=protective_label,
                        proposed_label=protective_label,
                        confidence=confidence,
                        semantic_delete_recommended=False,
                    ),
                    clean_row,
                ]},
            )
            assert not failed_retry_coverage_pairs(
                (failed, clean), windows,
                partition_by_id={"failed": 0, "clean": 0},
            )


def test_lower_confidence_alternate_nominates_fragment_for_later_covered_retry():
    failed = take(
        "failed",
        "different salons damaged the shoes so I recommend this repair kit "
        "with this you can",
        58.62,
        64.02,
    )
    clean = take(
        "clean",
        "has it happened to you that different salons damaged your shoes "
        "because they did not clean their tools I recommend this repair kit "
        "and you apply it to the damaged area",
        101.491,
        132.0,
    )
    windows = (
        {"partition_index": 0, "member_ids": ["failed", "clean"], "decisions": [
            row(
                "failed", label="alternate", proposed_label="alternate",
                confidence=0.70, semantic_delete_recommended=False,
            ),
            row(
                "clean", label="winner", proposed_label="winner",
                confidence=0.95, semantic_delete_recommended=False,
            ),
        ]},
        {"partition_index": 0, "member_ids": ["failed", "clean"], "decisions": [
            row("failed", confidence=0.85),
            row(
                "clean", label="winner", proposed_label="winner",
                confidence=0.95, semantic_delete_recommended=False,
            ),
        ]},
    )
    assert failed_retry_coverage_pairs(
        (failed, clean), windows, partition_by_id={"failed": 0, "clean": 0},
    ) == frozenset({("failed", "clean")})


def test_failed_family_member_can_use_safe_sibling_as_relation_anchor():
    failed = take(
        "failed", "this product attempt contains several unique broken words",
        20, 35,
    )
    anchor = take(
        "anchor", "different salons gave me foot fungus and I recommend this treatment",
        40, 48,
    )
    clean = take("clean", CLEAN_TEXT, 80, 95)
    windows = ({"partition_index": 0, "member_ids": ["failed", "anchor", "clean"], "decisions": [
        row("failed", confidence=0.90),
        row(
            "anchor", label="alternate", proposed_label="alternate",
            confidence=0.85, semantic_delete_recommended=False,
        ),
        row(
            "clean", label="uncertain", proposed_label="winner",
            confidence=0.95, semantic_delete_recommended=False,
        ),
    ]},)
    assert failed_retry_coverage_pairs(
        (failed, anchor, clean), windows,
        partition_by_id={"failed": 0, "anchor": 0, "clean": 0},
        retry_groups=(("failed", "anchor"), ("clean",)),
    ) == frozenset({("anchor", "clean")})


def test_winner_consensus_allows_one_point_nine_view_with_point_nine_five_peak():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    windows = (
        {"decisions": [
            row("failed"),
            row(
                "clean", label="uncertain", proposed_label="winner",
                confidence=0.90, semantic_delete_recommended=False,
            ),
        ]},
        {"decisions": [
            row("failed"),
            row(
                "clean", label="uncertain", proposed_label="winner",
                confidence=0.95, semantic_delete_recommended=False,
            ),
        ]},
    )
    assert failed_retry_coverage_pairs(
        (failed, clean), windows, partition_by_id={"failed": 0, "clean": 0},
    ) == frozenset({("failed", "clean")})


def test_family_relation_anchor_cannot_be_winner_bts_or_uncorroborated():
    failed = take("failed", "broken words with no literal relation", 20, 35)
    anchor = take("anchor", FAILED_TEXT, 40, 48)
    clean = take("clean", CLEAN_TEXT, 80, 95)
    clean_row = row(
        "clean", label="uncertain", proposed_label="winner",
        confidence=0.95, semantic_delete_recommended=False,
    )
    for overrides in (
        {"label": "winner", "proposed_label": "winner"},
        {"label": "bts", "proposed_label": "bts"},
        {"local_failure_corroborated": False},
        {"dense_semantic_failure_cluster": False},
    ):
        anchor_overrides = {
            "label": "alternate",
            "proposed_label": "alternate",
            "semantic_delete_recommended": False,
            **overrides,
        }
        windows = ({"decisions": [
            row("failed"),
            row("anchor", **anchor_overrides),
            clean_row,
        ]},)
        assert not failed_retry_coverage_pairs(
            (failed, anchor, clean), windows,
            partition_by_id={"failed": 0, "anchor": 0, "clean": 0},
            retry_groups=(("failed", "anchor"), ("clean",)),
        )


def test_winner_consensus_still_requires_one_point_nine_five_peak():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    windows = tuple({"decisions": [
        row("failed"),
        row(
            "clean", label="uncertain", proposed_label="winner",
            confidence=0.90, semantic_delete_recommended=False,
        ),
    ]} for _ in range(2))
    assert not failed_retry_coverage_pairs(
        (failed, clean), windows, partition_by_id={"failed": 0, "clean": 0},
    )


def test_family_anchor_cannot_inherit_failure_from_other_source_session_or_future():
    anchor = take("anchor", FAILED_TEXT, 40, 48)
    clean = take("clean", CLEAN_TEXT, 80, 95)
    clean_row = row(
        "clean", label="uncertain", proposed_label="winner",
        confidence=0.95, semantic_delete_recommended=False,
    )
    windows = ({"partition_index": 0, "member_ids": ["anchor", "clean"], "decisions": [
        row("failed"),
        row(
            "anchor", label="alternate", proposed_label="alternate",
            confidence=0.85, semantic_delete_recommended=False,
        ),
        clean_row,
    ]},)
    for failed, failed_partition in (
        (take("failed", "broken words", 20, 35, source="other"), 0),
        (take("failed", "broken words", 20, 35), 1),
        (take("failed", "broken words", 140, 150), 0),
    ):
        assert not failed_retry_coverage_pairs(
            (failed, anchor, clean), windows,
            partition_by_id={
                "failed": failed_partition, "anchor": 0, "clean": 0,
            },
            retry_groups=(("failed", "anchor"), ("clean",)),
        )
    recorded_partition_mismatch = (
        {"partition_index": 1, "member_ids": ["failed"], "decisions": [
            row("failed"),
        ]},
        {"partition_index": 0, "member_ids": ["anchor", "clean"], "decisions": [
            row(
                "anchor", label="alternate", proposed_label="alternate",
                confidence=0.85, semantic_delete_recommended=False,
            ),
            clean_row,
        ]},
    )
    assert not failed_retry_coverage_pairs(
        (take("failed", "broken words", 20, 35), anchor, clean),
        recorded_partition_mismatch,
        partition_by_id={"failed": 0, "anchor": 0, "clean": 0},
        retry_groups=(("failed", "anchor"), ("clean",)),
    )


def test_incomplete_failed_take_can_use_discarded_immediate_continuation_as_relation_evidence():
    failed = take(
        "failed", "has this happened to you after visiting different",
        49.0, 56.0, complete=False,
    )
    continuation = take(
        "continuation",
        "different salons gave me foot fungus and I recommend this treatment",
        58.0, 64.0, complete=False,
    )
    clean = take("clean", CLEAN_TEXT, 101.0, 132.0)
    windows = ({"partition_index": 0, "member_ids": [
        "failed", "continuation", "clean",
    ], "decisions": [
        row("failed", confidence=0.90),
        row("continuation", confidence=0.90),
        row(
            "clean", label="winner", proposed_label="winner",
            confidence=0.95, semantic_delete_recommended=False,
        ),
    ]},)
    assert failed_retry_coverage_pairs(
        (failed, clean), windows,
        partition_by_id={"failed": 0, "clean": 0},
        relation_takes=(failed, continuation, clean),
        relation_partition_by_id={"failed": 0, "continuation": 0, "clean": 0},
    ) == frozenset({("failed", "clean")})


def test_original_whole_session_identity_overrides_later_heuristic_resplits():
    failed = take("failed", "abandoned opening with missing ending", 27, 46)
    anchor = take("anchor", FAILED_TEXT, 58, 64)
    clean = take("clean", CLEAN_TEXT, 101, 132)
    windows = ({
        "partition_index": 0,
        "member_ids": ["failed", "anchor", "clean"],
        "decisions": [
            row("failed", confidence=0.90),
            row(
                "anchor", label="alternate", proposed_label="alternate",
                confidence=0.82, semantic_delete_recommended=False,
            ),
            row(
                "clean", label="winner", proposed_label="winner",
                confidence=0.95, semantic_delete_recommended=False,
            ),
        ],
    },)
    assert failed_retry_coverage_pairs(
        (failed, anchor, clean),
        windows,
        partition_by_id={"failed": 0, "anchor": 1, "clean": 2},
        retry_groups=(("failed", "anchor"), ("clean",)),
        relation_takes=(failed, anchor, clean),
        relation_partition_by_id={"failed": 0, "anchor": 1, "clean": 2},
    ) == frozenset({("anchor", "clean")})


def test_partial_recorded_identity_does_not_override_heuristic_session_split():
    failed = take("failed", FAILED_TEXT, 27, 46)
    clean = take("clean", CLEAN_TEXT, 101, 132)
    windows = ({
        "partition_index": 0,
        "member_ids": ["failed"],
        "decisions": [
            row("failed", confidence=0.90),
            row(
                "clean", label="winner", proposed_label="winner",
                confidence=0.95, semantic_delete_recommended=False,
            ),
        ],
    },)
    assert not failed_retry_coverage_pairs(
        (failed, clean), windows,
        partition_by_id={"failed": 0, "clean": 1},
    )


def test_discarded_continuation_bridge_rejects_complete_donor_large_gap_or_wrong_session():
    clean = take("clean", CLEAN_TEXT, 101.0, 132.0)
    clean_row = row(
        "clean", label="winner", proposed_label="winner",
        confidence=0.95, semantic_delete_recommended=False,
    )
    for failed, continuation, continuation_partition in (
        (
            take("failed", "valid complete statement", 49, 56, complete=True),
            take("continuation", FAILED_TEXT, 58, 64, complete=False), 0,
        ),
        (
            take("failed", "incomplete statement", 49, 56, complete=False),
            take("continuation", FAILED_TEXT, 61, 67, complete=False), 0,
        ),
        (
            take("failed", "incomplete statement", 49, 56, complete=False),
            take("continuation", FAILED_TEXT, 58, 64, complete=False), 1,
        ),
    ):
        windows = ({"decisions": [
            row("failed"), row("continuation"), clean_row,
        ]},)
        assert not failed_retry_coverage_pairs(
            (failed, clean), windows,
            partition_by_id={"failed": 0, "clean": 0},
            relation_takes=(failed, continuation, clean),
            relation_partition_by_id={
                "failed": 0, "continuation": continuation_partition, "clean": 0,
            },
        )


def test_original_recorded_partition_cannot_be_erased_by_reduced_pool_repartitioning():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    windows = (
        {"partition_index": 0, "member_ids": ["failed"], "decisions": [row("failed")]},
        {"partition_index": 1, "member_ids": ["clean"], "decisions": [row(
            "clean", label="uncertain", proposed_label="winner",
            semantic_delete_recommended=False,
            local_failure_corroborated=False,
            dense_semantic_failure_cluster=False,
        )]},
    )
    assert not failed_retry_coverage_pairs(
        (failed, clean), windows, partition_by_id={"failed": 0, "clean": 0},
    )


def test_recording_prefix_or_suffix_blocks_proposed_winner_eligibility():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    for field in ("recording_prefix_words", "recording_suffix_words"):
        windows = ({"decisions": [
            row("failed"),
            row(
                "clean", label="uncertain", proposed_label="winner",
                semantic_delete_recommended=False,
                local_failure_corroborated=False,
                dense_semantic_failure_cluster=False,
                **{field: 5},
            ),
        ]},)
        assert not failed_retry_coverage_pairs(
            (failed, clean), windows, partition_by_id={"failed": 0, "clean": 0},
        )


def test_directional_coverage_can_bridge_beyond_normal_time_window():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    arbiter = CoverageArbiter()
    merged, diagnostics = reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",)),
        (failed, clean),
        arbiter,
        failed_retry_coverage_pairs=frozenset({("failed", "clean")}),
    )
    assert merged == (("failed", "clean"),)
    assert diagnostics["failed_retry_coverage_pair_count"] == 1
    assert diagnostics["merges"][0]["accepted_by"] == "failed_attempt_directional_coverage"


def test_directional_coverage_accepts_point_nine_with_all_safety_fields():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    merged, diagnostics = reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",)),
        (failed, clean),
        CoverageArbiter(confidence=0.90),
        failed_retry_coverage_pairs=frozenset({("failed", "clean")}),
    )
    assert merged == (("failed", "clean"),)
    assert diagnostics["merges"][0]["accepted_by"] == "failed_attempt_directional_coverage"


def test_directional_bridge_fails_closed_on_conflict_low_confidence_or_missing_coverage():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    for arbiter in (
        CoverageArbiter(conflict=True),
        CoverageArbiter(conflict=True, same_idea=True),
        CoverageArbiter(confidence=0.89),
        CoverageArbiter(left_covered=False),
    ):
        merged, diagnostics = reconcile_semantic_idea_equivalence(
            (("failed",), ("clean",)),
            (failed, clean),
            arbiter,
            failed_retry_coverage_pairs=frozenset({("failed", "clean")}),
        )
        assert merged == (("failed",), ("clean",))
        assert diagnostics["merged_pair_count"] == 0


def test_unlisted_pair_does_not_gain_directional_coverage_authority():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 20, 35)
    merged, _ = reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",)), (failed, clean), CoverageArbiter(),
    )
    assert merged == (("failed",), ("clean",))


def test_listed_pair_cannot_bypass_coverage_through_deterministic_restart_fast_path():
    text = "this exact complete product statement is repeated word for word"
    failed = take("failed", text, 0, 10)
    clean = take("clean", text, 50, 65)
    merged, diagnostics = reconcile_semantic_idea_equivalence(
        (("failed",), ("clean",)),
        (failed, clean),
        CoverageArbiter(conflict=True, left_covered=False, same_idea=True),
        failed_retry_coverage_pairs=frozenset({("failed", "clean")}),
    )
    assert merged == (("failed",), ("clean",))
    assert diagnostics["merged_pair_count"] == 0
    assert diagnostics["restart_evidence_merges"] == []
