from cutsell_worker import hybrid_composite_best_take, hybrid_semantic_complementary_rescue
from cutsell_worker.composite_resolver import _composite_split_ids, apply_composite_group_split
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.take_grouping_provider import TakeGroupingProviderResult
from cutsell_worker.contracts import CandidateTake


def _take(clip_id: str, start: float) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=start + 4.0,
        text=f"complete delivery {clip_id}",
        words=(),
        signals=None,
        complete_idea=True,
    )


def _grouping(*groups: tuple[str, ...]) -> TakeGroupingProviderResult:
    return TakeGroupingProviderResult(
        groups=tuple(groups),
        status=ProviderStatus(
            provider="test",
            requested=False,
            available=True,
            status="baseline",
            reason="test",
        ),
        reason="test",
    )


def test_semantic_rescue_is_restoration_not_best_take_immunity():
    rescue_token = hybrid_semantic_complementary_rescue._SPLIT_IDS.set(
        frozenset({"conclusion-a", "conclusion-b"})
    )
    composite_token = hybrid_composite_best_take._COMPOSITE_SPLIT_IDS.set(frozenset())
    try:
        split_ids = _composite_split_ids()

        assert split_ids == frozenset()
        assert hybrid_semantic_complementary_rescue._SPLIT_IDS.get() == frozenset()
        assert hybrid_composite_best_take._COMPOSITE_SPLIT_IDS.get() == frozenset()
    finally:
        hybrid_semantic_complementary_rescue._SPLIT_IDS.reset(rescue_token)
        hybrid_composite_best_take._COMPOSITE_SPLIT_IDS.reset(composite_token)


def test_true_composite_keeps_best_take_immunity():
    rescue_token = hybrid_semantic_complementary_rescue._SPLIT_IDS.set(frozenset())
    composite_token = hybrid_composite_best_take._COMPOSITE_SPLIT_IDS.set(
        frozenset({"piece-a", "piece-b"})
    )
    try:
        split_ids = _composite_split_ids()

        assert split_ids == frozenset({"piece-a", "piece-b"})

        takes = (_take("piece-a", 0.0), _take("piece-b", 5.0), _take("other", 10.0))
        grouping = _grouping(("piece-a", "piece-b", "other"))
        repaired = apply_composite_group_split(grouping, takes, split_ids)

        assert repaired.groups == (("piece-a",), ("piece-b",), ("other",))
        assert "composite_resolver_group_split:2" in repaired.reason
    finally:
        hybrid_semantic_complementary_rescue._SPLIT_IDS.reset(rescue_token)
        hybrid_composite_best_take._COMPOSITE_SPLIT_IDS.reset(composite_token)


def test_rescue_and_true_composite_overlap_returns_only_true_composite_ids():
    rescue_token = hybrid_semantic_complementary_rescue._SPLIT_IDS.set(
        frozenset({"rescued-whole", "piece-a"})
    )
    composite_token = hybrid_composite_best_take._COMPOSITE_SPLIT_IDS.set(
        frozenset({"piece-a", "piece-b"})
    )
    try:
        split_ids = _composite_split_ids()

        assert split_ids == frozenset({"piece-a", "piece-b"})
        assert "rescued-whole" not in split_ids
    finally:
        hybrid_semantic_complementary_rescue._SPLIT_IDS.reset(rescue_token)
        hybrid_composite_best_take._COMPOSITE_SPLIT_IDS.reset(composite_token)
