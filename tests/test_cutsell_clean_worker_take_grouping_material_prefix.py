from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_grouping_provider import (
    _groups_should_reconcile,
    _is_material_prefix_fragment,
)


def _take(clip_id: str, start: float, end: float, text: str, *, complete: bool = True) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src-1",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete,
    )


def test_material_prefix_can_be_neutral_even_when_complete_idea_is_true():
    prefix = _take(
        "prefix",
        0.0,
        2.0,
        "people ask me all the time do",
        complete=True,
    )
    full_first = _take(
        "full-first",
        2.4,
        7.4,
        "people ask me all the time do you actually have fun doing your job and the answer is yes",
    )
    full_retry = _take(
        "full-retry",
        10.0,
        15.0,
        "people ask me all the time do you actually have fun doing your job and the answer is yes obviously",
    )
    take_map = {take.clip_id: take for take in (prefix, full_first, full_retry)}

    assert _is_material_prefix_fragment(prefix, full_first) is True
    assert _groups_should_reconcile((prefix.clip_id, full_first.clip_id), (full_retry.clip_id,), take_map) is True


def test_near_full_prefix_is_not_neutralized():
    almost_full = _take(
        "almost-full",
        0.0,
        4.4,
        "people ask me all the time do you actually have fun doing your job",
    )
    full = _take(
        "full",
        4.8,
        9.6,
        "people ask me all the time do you actually have fun doing your job and yes",
    )

    assert _is_material_prefix_fragment(almost_full, full) is False


def test_same_duration_prefix_is_not_neutralized():
    prefix = _take(
        "prefix",
        0.0,
        4.5,
        "people ask me all the time do",
    )
    full = _take(
        "full",
        4.6,
        9.0,
        "people ask me all the time do you actually have fun doing your job and the answer is yes",
    )

    assert _is_material_prefix_fragment(prefix, full) is False


def test_different_source_never_reconciles():
    left = _take("left", 0.0, 4.0, "this is the same complete retry sentence for testing")
    right = CandidateTake(
        clip_id="right",
        source_asset_id="src-2",
        source_order=1,
        start=5.0,
        end=9.0,
        text="this is the same complete retry sentence for testing",
        complete_idea=True,
    )
    take_map = {left.clip_id: left, right.clip_id: right}

    assert _groups_should_reconcile((left.clip_id,), (right.clip_id,), take_map) is False
