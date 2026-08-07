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
    trim_clip,
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


def test_trim_changes_only_media_boundaries_and_preserves_source_and_text():
    original = _draft()
    updated = trim_clip(original, "a", start=0.2, end=0.8)
    item = updated["selected"][0]
    assert item["start"] == 0.2
    assert item["end"] == 0.8
    assert item["source_asset_id"] == "src-1"
    assert item["text"] == "Wow, it's so good."
    assert item["caption_text"] == "Wow, it's so good."
    assert original["selected"][0]["start"] == 0.0
    assert original["selected"][0]["end"] == 1.0


def test_trim_cannot_expand_outside_source_interval_or_create_microfragment():
    with pytest.raises(DraftEditError, match="inside"):
        trim_clip(_draft(), "a", start=0.0, end=1.1)
    with pytest.raises(DraftEditError, match="microfragment"):
        trim_clip(_draft(), "a", start=0.5, end=0.55)


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


def test_mobile_trim_api_returns_trimmed_draft_and_rejects_unsafe_range():
    client = TestClient(api.app)
    response = client.post("/v1/draft-edits/trim", json={
        "draft": _draft(),
        "clip_id": "a",
        "start": 0.2,
        "end": 0.8,
    })
    assert response.status_code == 200
    assert response.json()["selected"][0]["start"] == 0.2
    assert response.json()["selected"][0]["end"] == 0.8

    bad = client.post("/v1/draft-edits/trim", json={
        "draft": _draft(),
        "clip_id": "a",
        "start": 0.5,
        "end": 0.55,
    })
    assert bad.status_code == 409
