from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_grouping import group_takes, retry_similarity


def _take(clip_id, text, start):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src-1",
        source_order=0,
        start=start,
        end=start + 2.0,
        text=text,
    )


def test_retry_similarity_groups_small_wording_changes():
    score = retry_similarity(
        "This serum changed my skin completely",
        "This serum completely changed my skin",
    )
    assert score >= 0.72


def test_group_takes_keeps_different_sales_ideas_separate():
    groups = group_takes((
        _take("a", "This serum completely changed my skin", 0.0),
        _take("b", "This serum changed my skin completely", 3.0),
        _take("c", "Tap the cart before the deal ends", 6.0),
    ))
    sizes = sorted(len(group) for group in groups.values())
    assert sizes == [1, 2]


def test_short_generic_phrases_are_not_fuzzy_grouped():
    groups = group_takes((
        _take("a", "buy it now", 0.0),
        _take("b", "buy this now", 3.0),
    ))
    assert len(groups) == 2
