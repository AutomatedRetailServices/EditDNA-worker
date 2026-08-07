import pytest
from fastapi.testclient import TestClient

import cutsell_app.main as api
from cutsell_worker.draft_edits import (
    DraftEditError,
    patch_captions,
    remove_clip,
    reorder_clips,
    restore_clip,
    swap_take,
)


def _clip(clip_id, group_id, text, *, selected):
    return {
        "clip_id": clip_id,
        "source_asset_id": "src-1",
        "source_order": 0,
        "start": 0.0,
        "end": 1.0,
        "text": text,
        "caption_text": text,
        "semantic_role": "BENEFITS",
        "take_group_id": group_id,
        "selected": selected,
    }


def _draft():
    return {
        "schema_version": "cutsell.v1",
        "project_id": "p1",
        "strategy": "mixed",
        "selected": [
            _clip("a", "g1", "Wow, it's so good.", selected=True),
            _clip("c", "g2", "Check it out.", selected=True),
        ],
        "alternates": [
            _clip("b", "g1", "Wow, it's so good.", selected=False),
            _clip("d", "g3", "Different idea.", selected=False),
        ],
        "discarded": [],
        "diagnostics": {},
    }


def test_swap_only_allows_same_take_group_and_preserves_old_take_as_alternate():
    original = _draft()
    updated = swap_take(original, "a", "b")
    assert [item["clip_id"] for item in updated["selected"]] == ["b", "c"]
    assert {item["clip_id"] for item in updated["alternates"]} == {"a", "d"}
    assert updated["selected"][0]["selected"] is True
    assert original["selected"][0]["clip_id"] == "a"
    with pytest.raises(DraftEditError, match="same take group"):
        swap_take(original, "a", "d")


def test_remove_and_restore_round_trip_without_duplicate_take_group():
    removed = remove_clip(_draft(), "a")
    assert [item["clip_id"] for item in removed["selected"]] == ["c"]
    restored = restore_clip(removed, "a", 0)
    assert [item["clip_id"] for item in restored["selected"]] == ["a", "c"]
    with pytest.raises(DraftEditError, match="use swap"):
        restore_clip(_draft(), "b")


def test_reorder_requires_every_selected_clip_once():
    updated = reorder_clips(_draft(), ["c", "a"])
    assert [item["clip_id"] for item in updated["selected"]] == ["c", "a"]
    with pytest.raises(DraftEditError):
        reorder_clips(_draft(), ["a", "a"])


def test_caption_patch_preserves_transcript_text():
    updated = patch_captions(_draft(), [{"clip_id": "a", "text": "WOW — so good!"}])
    item = updated["selected"][0]
    assert item["text"] == "Wow, it's so good."
    assert item["caption_text"] == "WOW — so good!"


def test_mobile_draft_edit_api_swaps_real_siblings_only():
    client = TestClient(api.app)
    response = client.post("/v1/draft-edits/swap", json={
        "draft": _draft(),
        "selected_clip_id": "a",
        "replacement_clip_id": "b",
    })
    assert response.status_code == 200
    assert response.json()["selected"][0]["clip_id"] == "b"

    bad = client.post("/v1/draft-edits/swap", json={
        "draft": _draft(),
        "selected_clip_id": "a",
        "replacement_clip_id": "d",
    })
    assert bad.status_code == 409
