import pytest

from worker.draft_edits import DraftEditError, remove_clip, reorder_clips, restore_clip, swap_take


def _item(cid, selected):
    return {"clip_id": cid, "selected": selected, "clean_cut_keep": True, "slot": "OTHER"}


def _draft():
    return {
        "schema_version": "v1",
        "mode": "human",
        "selected_clip_ids": ["a", "b"],
        "selected": [_item("a", True), _item("b", True)],
        "alternates": [_item("c", False)],
        "discarded": [],
        "boundary_discards": [],
        "counts": {"selected": 2, "alternates": 1, "discarded": 0},
    }


def test_swap_preserves_old_take_as_alternate():
    original = _draft()
    edited = swap_take(original, "a", "c")
    assert edited["selected_clip_ids"] == ["c", "b"]
    assert {x["clip_id"] for x in edited["alternates"]} == {"a"}
    assert original["selected_clip_ids"] == ["a", "b"]


def test_remove_and_restore_are_reversible():
    removed = remove_clip(_draft(), "a")
    assert removed["selected_clip_ids"] == ["b"]
    restored = restore_clip(removed, "a", position=0)
    assert restored["selected_clip_ids"] == ["a", "b"]


def test_reorder_requires_exact_selected_set():
    edited = reorder_clips(_draft(), ["b", "a"])
    assert edited["selected_clip_ids"] == ["b", "a"]
    with pytest.raises(DraftEditError):
        reorder_clips(_draft(), ["a"])
