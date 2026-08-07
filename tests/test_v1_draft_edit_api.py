from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.routes_v1 import router


def _draft():
    return {
        "schema_version": "v1",
        "mode": "human",
        "selected_clip_ids": ["a", "b"],
        "selected": [
            {"clip_id": "a", "selected": True, "text": "A"},
            {"clip_id": "b", "selected": True, "text": "B"},
        ],
        "alternates": [
            {"clip_id": "c", "selected": False, "text": "C"},
        ],
        "discarded": [],
        "boundary_discards": [],
        "counts": {"selected": 2, "alternates": 1, "discarded": 0},
    }


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_v1_health_exposes_stateless_draft_capability():
    response = _client().get("/v1/healthz")
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "api_version": "v1",
        "draft_edits": ["swap", "remove", "restore", "reorder", "captions"],
        "persistence": False,
    }


def test_swap_route_preserves_old_selection_as_alternate():
    response = _client().post(
        "/v1/draft-edits/swap",
        json={
            "draft": _draft(),
            "selected_clip_id": "a",
            "replacement_clip_id": "c",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "v1"
    assert body["draft"]["selected_clip_ids"] == ["c", "b"]
    assert {item["clip_id"] for item in body["draft"]["alternates"]} == {"a"}


def test_remove_restore_and_reorder_routes_round_trip():
    client = _client()
    removed = client.post(
        "/v1/draft-edits/remove",
        json={"draft": _draft(), "clip_id": "a"},
    )
    assert removed.status_code == 200
    removed_draft = removed.json()["draft"]
    assert removed_draft["selected_clip_ids"] == ["b"]

    restored = client.post(
        "/v1/draft-edits/restore",
        json={"draft": removed_draft, "clip_id": "a", "position": 0},
    )
    assert restored.status_code == 200
    restored_draft = restored.json()["draft"]
    assert restored_draft["selected_clip_ids"] == ["a", "b"]

    reordered = client.post(
        "/v1/draft-edits/reorder",
        json={"draft": restored_draft, "ordered_clip_ids": ["b", "a"]},
    )
    assert reordered.status_code == 200
    assert reordered.json()["draft"]["selected_clip_ids"] == ["b", "a"]


def test_caption_route_preserves_transcript_and_sets_caption_text():
    response = _client().post(
        "/v1/draft-edits/captions",
        json={
            "draft": _draft(),
            "edits": [
                {"clip_id": "a", "text": "Caption A"},
                {"clip_id": "b", "text": ""},
            ],
        },
    )
    assert response.status_code == 200
    selected = response.json()["draft"]["selected"]
    assert selected[0]["text"] == "A"
    assert selected[0]["caption_text"] == "Caption A"
    assert selected[1]["text"] == "B"
    assert selected[1]["caption_text"] == ""


def test_caption_route_rejects_nonselected_clip():
    response = _client().post(
        "/v1/draft-edits/captions",
        json={"draft": _draft(), "edits": [{"clip_id": "c", "text": "alternate"}]},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "caption clip not found in selected draft"


def test_invalid_edit_returns_conflict_not_server_error():
    response = _client().post(
        "/v1/draft-edits/swap",
        json={
            "draft": _draft(),
            "selected_clip_id": "missing",
            "replacement_clip_id": "c",
        },
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "selected clip not found"
