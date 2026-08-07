import json

from fastapi.testclient import TestClient

import cutsell_app.main as api
from cutsell_worker.draft_store import (
    DraftConflictError,
    DraftHistoryError,
    create_initial_draft,
    get_draft_snapshot,
    redo_draft_snapshot,
    save_draft_snapshot,
    undo_draft_snapshot,
)


def _draft(text="hello", project_id="project-1"):
    return {
        "schema_version": "cutsell.v1",
        "project_id": project_id,
        "strategy": "mixed",
        "selected": [{
            "clip_id": "clip-1",
            "source_asset_id": "src-1",
            "source_order": 0,
            "start": 0.0,
            "end": 1.0,
            "text": text,
            "caption_text": text,
            "semantic_role": "OTHER",
            "selected": True,
        }],
        "alternates": [],
        "discarded": [],
        "diagnostics": {},
    }


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.pending = []
    def watch(self, _key):
        return None
    def get(self, key):
        return self.redis.get(key)
    def multi(self):
        return None
    def set(self, key, value):
        self.pending.append((key, value))
    def execute(self):
        for key, value in self.pending:
            self.redis.data[key] = value
        self.pending = []
        return [True]
    def reset(self):
        self.pending = []


class FakeRedis:
    def __init__(self):
        self.data = {}
    def get(self, key):
        return self.data.get(key)
    def set(self, key, value, nx=False):
        if nx and key in self.data:
            return False
        self.data[key] = value
        return True
    def pipeline(self):
        return FakePipeline(self)


def _init(client):
    return create_initial_draft(
        user_id="user-1",
        project_id="project-1",
        draft=_draft("AI first"),
        sources=[{"source_asset_id": "src-1", "uri": "s3://bucket/source.mov"}],
        client=client,
    )


def test_initial_ai_draft_is_create_only_and_does_not_overwrite_existing_draft():
    client = FakeRedis()
    first, created = _init(client)
    assert created is True
    assert first["revision"] == 1
    assert first["undo"] == []
    assert first["redo"] == []

    existing, created_again = create_initial_draft(
        user_id="user-1",
        project_id="project-1",
        draft=_draft("AI retry must not overwrite"),
        sources=[{"source_asset_id": "src-1", "uri": "s3://bucket/source.mov"}],
        client=client,
    )
    assert created_again is False
    assert existing["revision"] == 1
    assert existing["draft"]["selected"][0]["text"] == "AI first"


def test_autosave_increments_revision_preserves_sources_and_creates_undo_state():
    client = FakeRedis()
    _init(client)
    saved = save_draft_snapshot(
        user_id="user-1",
        project_id="project-1",
        draft=_draft("human edit"),
        expected_revision=1,
        client=client,
    )
    assert saved["revision"] == 2
    assert saved["draft"]["selected"][0]["text"] == "human edit"
    assert saved["sources"][0]["source_asset_id"] == "src-1"
    assert saved["undo"][-1]["selected"][0]["text"] == "AI first"
    assert saved["redo"] == []


def test_persistent_undo_redo_sequence_survives_multiple_revisions():
    client = FakeRedis()
    _init(client)
    save_draft_snapshot(user_id="user-1", project_id="project-1", draft=_draft("edit A"), expected_revision=1, client=client)
    save_draft_snapshot(user_id="user-1", project_id="project-1", draft=_draft("edit B"), expected_revision=2, client=client)

    undo1 = undo_draft_snapshot(user_id="user-1", project_id="project-1", expected_revision=3, client=client)
    assert undo1["revision"] == 4
    assert undo1["draft"]["selected"][0]["text"] == "edit A"

    undo2 = undo_draft_snapshot(user_id="user-1", project_id="project-1", expected_revision=4, client=client)
    assert undo2["revision"] == 5
    assert undo2["draft"]["selected"][0]["text"] == "AI first"

    redo1 = redo_draft_snapshot(user_id="user-1", project_id="project-1", expected_revision=5, client=client)
    assert redo1["revision"] == 6
    assert redo1["draft"]["selected"][0]["text"] == "edit A"


def test_new_autosave_after_undo_clears_redo_branch():
    client = FakeRedis()
    _init(client)
    save_draft_snapshot(user_id="user-1", project_id="project-1", draft=_draft("edit A"), expected_revision=1, client=client)
    save_draft_snapshot(user_id="user-1", project_id="project-1", draft=_draft("edit B"), expected_revision=2, client=client)
    undo_draft_snapshot(user_id="user-1", project_id="project-1", expected_revision=3, client=client)
    saved = save_draft_snapshot(user_id="user-1", project_id="project-1", draft=_draft("new branch"), expected_revision=4, client=client)
    assert saved["redo"] == []
    try:
        redo_draft_snapshot(user_id="user-1", project_id="project-1", expected_revision=5, client=client)
    except DraftHistoryError as exc:
        assert "nothing to redo" in str(exc)
    else:
        raise AssertionError("redo must be cleared after a new autosave")


def test_stale_autosave_cannot_overwrite_newer_revision():
    client = FakeRedis()
    _init(client)
    save_draft_snapshot(user_id="user-1", project_id="project-1", draft=_draft("newer"), expected_revision=1, client=client)
    try:
        save_draft_snapshot(user_id="user-1", project_id="project-1", draft=_draft("stale"), expected_revision=1, client=client)
    except DraftConflictError as exc:
        assert "current 2" in str(exc)
    else:
        raise AssertionError("stale autosave must not overwrite a newer draft")


def test_recovery_api_returns_saved_snapshot(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_draft_snapshot",
        lambda **kwargs: {
            "schema_version": "cutsell.draft.v1",
            "project_id": kwargs["project_id"],
            "user_id": kwargs["user_id"],
            "revision": 3,
            "saved_at": "2026-08-07T00:00:00+00:00",
            "draft": _draft(),
            "sources": [{"source_asset_id": "src-1"}],
            "undo": [],
            "redo": [],
        },
    )
    response = TestClient(api.app).get("/v1/projects/project-1/draft?user_id=user-1")
    assert response.status_code == 200
    assert response.json()["revision"] == 3
    assert response.json()["sources"][0]["source_asset_id"] == "src-1"


def test_autosave_api_returns_409_for_stale_revision(monkeypatch):
    def conflict(**_kwargs):
        raise DraftConflictError("draft revision conflict: expected 1, current 2")
    monkeypatch.setattr(api, "save_draft_snapshot", conflict)
    response = TestClient(api.app).put("/v1/projects/project-1/draft", json={
        "user_id": "user-1",
        "expected_revision": 1,
        "draft": _draft(),
    })
    assert response.status_code == 409
    assert "current 2" in response.json()["detail"]


def test_undo_redo_api_contract(monkeypatch):
    monkeypatch.setattr(api, "undo_draft_snapshot", lambda **kwargs: {"revision": kwargs["expected_revision"] + 1, "draft": _draft("undo")})
    monkeypatch.setattr(api, "redo_draft_snapshot", lambda **kwargs: {"revision": kwargs["expected_revision"] + 1, "draft": _draft("redo")})
    client = TestClient(api.app)
    undone = client.post("/v1/projects/project-1/draft/undo", json={"user_id": "user-1", "expected_revision": 4})
    redone = client.post("/v1/projects/project-1/draft/redo", json={"user_id": "user-1", "expected_revision": 5})
    assert undone.status_code == 200 and undone.json()["revision"] == 5
    assert redone.status_code == 200 and redone.json()["revision"] == 6
