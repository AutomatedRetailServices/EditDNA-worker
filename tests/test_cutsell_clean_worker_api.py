from types import SimpleNamespace

from fastapi.testclient import TestClient

import cutsell_app.main as api


def test_healthz_exposes_clean_service_readiness(monkeypatch):
    monkeypatch.setattr(
        api,
        "load_runtime_config",
        lambda: SimpleNamespace(queue_ready=True, storage_ready=True, semantic_ready=False),
    )
    response = TestClient(api.app).get("/v1/healthz")
    assert response.status_code == 200
    assert response.json()["service"] == "cutsell-api"
    assert response.json()["queue_ready"] is True
    assert response.json()["storage_ready"] is True


def test_flow_b_submit_generates_source_ids_and_enqueues(monkeypatch):
    captured = {}
    def fake_enqueue(payload):
        captured.update(payload)
        return SimpleNamespace(job_id="job-1", queue_name="cutsell")
    monkeypatch.setattr(api, "enqueue_flow_b", fake_enqueue)

    response = TestClient(api.app).post("/v1/flow-b/jobs", json={
        "project_id": "project-1",
        "user_id": "user-1",
        "language_hint": "en",
        "sources": [
            {"original_name": "one.mov", "uri": "s3://bucket/one.mov", "source_order": 0},
            {"original_name": "two.mov", "uri": "s3://bucket/two.mov", "source_order": 1},
        ],
    })
    assert response.status_code == 202
    assert response.json() == {"job_id": "job-1", "queue": "cutsell", "state": "uploaded"}
    assert len(captured["sources"]) == 2
    assert captured["sources"][0]["source_asset_id"].startswith("src_")
    assert captured["sources"][0]["source_asset_id"] != captured["sources"][1]["source_asset_id"]


def test_flow_b_submit_rejects_duplicate_source_order(monkeypatch):
    monkeypatch.setattr(api, "enqueue_flow_b", lambda payload: None)
    response = TestClient(api.app).post("/v1/flow-b/jobs", json={
        "project_id": "project-1",
        "user_id": "user-1",
        "sources": [
            {"original_name": "one.mov", "uri": "s3://bucket/one.mov", "source_order": 0},
            {"original_name": "two.mov", "uri": "s3://bucket/two.mov", "source_order": 0},
        ],
    })
    assert response.status_code == 409
