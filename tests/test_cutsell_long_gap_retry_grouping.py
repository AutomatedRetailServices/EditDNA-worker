from cutsell_worker.contracts import CandidateTake
from cutsell_worker.local_retry_grouping import _adjacent_reformulated_retries


def _take(clip_id: str, start: float, text: str) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=start + 4.0,
        text=text,
        complete_idea=True,
    )


def test_reformulated_same_idea_retry_can_bridge_twenty_five_second_pause():
    first = _take(
        "a",
        0.0,
        "this product helps improve sleep quality every single night",
    )
    retry = _take(
        "b",
        29.0,
        "this product helps improve sleep quality every night for me",
    )
    groups, changed = _adjacent_reformulated_retries(
        ((first.clip_id,), (retry.clip_id,)),
        (first, retry),
    )
    assert changed is True
    assert groups == (("a", "b"),)


def test_long_gap_does_not_merge_broad_topic_when_semantic_opening_differs():
    first = _take(
        "a",
        0.0,
        "this product helps improve sleep quality every single night",
    )
    new_detail = _take(
        "b",
        29.0,
        "another benefit is the soft washable cover for travel",
    )
    groups, changed = _adjacent_reformulated_retries(
        ((first.clip_id,), (new_detail.clip_id,)),
        (first, new_detail),
    )
    assert changed is False
    assert groups == (("a",), ("b",))
