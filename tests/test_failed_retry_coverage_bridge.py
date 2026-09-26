from cutsell_worker.contracts import CandidateTake
from cutsell_worker.failed_retry_coverage_bridge import failed_retry_coverage_pairs
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


def test_failed_bridge_still_rejects_no_delete_vote_or_uncorroborated_window():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    clean_row = row(
        "clean", label="winner", proposed_label="winner",
        semantic_delete_recommended=False, confidence=0.95,
    )
    no_delete = ({"decisions": [
        row("failed", confidence=0.90, semantic_delete_recommended=False), clean_row,
    ]},)
    uncorroborated = ({"decisions": [
        row("failed", confidence=0.90, local_failure_corroborated=False), clean_row,
    ]},)
    low_confidence_view = ({"decisions": [
        row("failed", confidence=0.90), clean_row,
    ]}, {"decisions": [
        row("failed", confidence=0.79, semantic_delete_recommended=False), clean_row,
    ]})
    weak_delete_vote = ({"decisions": [
        row("failed", confidence=0.84), clean_row,
    ]},)
    for windows in (no_delete, uncorroborated, low_confidence_view, weak_delete_vote):
        assert not failed_retry_coverage_pairs(
            (failed, clean), windows, partition_by_id={"failed": 0, "clean": 0},
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


def test_directional_bridge_fails_closed_on_conflict_low_confidence_or_missing_coverage():
    failed = take("failed", FAILED_TEXT, 0, 10)
    clean = take("clean", CLEAN_TEXT, 50, 65)
    for arbiter in (
        CoverageArbiter(conflict=True),
        CoverageArbiter(conflict=True, same_idea=True),
        CoverageArbiter(confidence=0.94),
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
