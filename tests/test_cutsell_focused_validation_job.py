import pytest

from cutsell_worker.focused_validation_job import _requested_source_keys


def test_focused_source_keys_preserve_exact_human_review_order():
    keys = (
        "Editdna longform validation/VIDEO-2026-07-30-09-18-03.mp4",
        "Editdna longform validation/VIDEO-2026-07-30-09-24-13.mp4",
        "Editdna longform validation/VIDEO-2026-07-30-10-22-46.mp4",
    )
    assert _requested_source_keys(keys) == keys


def test_focused_source_keys_reject_negative_or_external_dataset():
    with pytest.raises(ValueError):
        _requested_source_keys(("Editdna bloopers videos/blooper.mp4",))


def test_focused_source_keys_reject_duplicates():
    key = "Editdna longform validation/VIDEO-2026-07-30-09-18-03.mp4"
    with pytest.raises(ValueError):
        _requested_source_keys((key, key))


def test_focused_source_keys_reject_too_many_sources():
    keys = tuple(
        f"Editdna longform validation/video-{index}.mp4"
        for index in range(7)
    )
    with pytest.raises(ValueError):
        _requested_source_keys(keys)
