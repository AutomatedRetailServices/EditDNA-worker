from __future__ import annotations

import pytest

from cutsell_worker.validation_job import (
    BLOOPER_NEGATIVE_PREFIX,
    CLEAN_CUT_GOLD_PREFIX,
    _clean_cut_gold_prefix,
    _gold_video_limit,
)


def test_clean_cut_gold_prefix_is_longform_validation() -> None:
    assert CLEAN_CUT_GOLD_PREFIX == "Editdna/ longform validation/"
    assert _clean_cut_gold_prefix(None) == CLEAN_CUT_GOLD_PREFIX
    assert _clean_cut_gold_prefix(CLEAN_CUT_GOLD_PREFIX) == CLEAN_CUT_GOLD_PREFIX


def test_bloopers_only_are_rejected_as_clean_cut_gold() -> None:
    assert BLOOPER_NEGATIVE_PREFIX == "Editdna bloopers videos/"
    with pytest.raises(ValueError, match="negative-behavior"):
        _clean_cut_gold_prefix(BLOOPER_NEGATIVE_PREFIX)


def test_other_prefixes_are_rejected_as_clean_cut_gold() -> None:
    with pytest.raises(ValueError, match="must use"):
        _clean_cut_gold_prefix("some-other-folder/")


def test_gold_video_limit_accepts_real_batch_size() -> None:
    assert _gold_video_limit(None) == 16
    assert _gold_video_limit(10) == 10
    assert _gold_video_limit(16) == 16


@pytest.mark.parametrize("value", [0, 33, -1])
def test_gold_video_limit_is_bounded(value: int) -> None:
    with pytest.raises(ValueError, match="between"):
        _gold_video_limit(value)
