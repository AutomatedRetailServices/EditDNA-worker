from types import SimpleNamespace

from fastapi.testclient import TestClient

import cutsell_app.main as api
from cutsell_worker.jobs import JobSnapshot


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


def test_presign_upload_returns_direct_s3_contract(monkeypatch):
    monkeypatch.setattr(
        api,
        "create_presigned_upload",
        lambda **kwargs: {
            "method": "POST",
            "upload_url": "https://upload.invalid",
            "fields": {"key": "cutsell/uploads/x/video.mov"},
            "source_uri": "s3://bucket/cutsell/uploads/x/video.mov",
            "object_key": "cutsell/uploads/x/video.mov",
            "content_type": "video/quicktime",
            "max_bytes": kwargs["size_bytes"],
            "expires_in": 900,
        },
    )
    response = TestClient(api.app).post("/v1/uploads/presign", json={
        "project_id": "project-1",
        "user_id": "user-1",
        "original_name": "video.mov",
        "content_type": "video/quicktime",
        "size_bytes": 12345,
    })
    assert response.status_code == 200
    assert response.json()["method"] == "POST"
    assert response.json()["source_uri"].startswith("s3://bucket/cutsell/uploads/")


def test_flow_b_submit_generates_source_ids_and_enqueues(monkeypatch):
    captured = {}
    def fake_enqueue(payload):
        captured.update(payload)
        return SimpleNamespace(job_id="job-1", queue_name="cutsell")
    monkeypatch.setattr(api, "enqueue_flow_b", fake_enqueue)
    monkeypatch.setattr(api, "validate_product_source_uri", lambda uri, **kwargs: ("bucket", uri))

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
    monkeypatch.setattr(api, "validate_product_source_uri", lambda uri, **kwargs: ("bucket", uri))
    response = TestClient(api.app).post("/v1/flow-b/jobs", json={
        "project_id": "project-1",
        "user_id": "user-1",
        "sources": [
            {"original_name": "one.mov", "uri": "s3://bucket/one.mov", "source_order": 0},
            {"original_name": "two.mov", "uri": "s3://bucket/two.mov", "source_order": 0},
        ],
    })
    assert response.status_code == 409


def test_flow_b_submit_rejects_source_outside_product_upload_scope(monkeypatch):
    def reject(_uri, **_kwargs):
        raise ValueError("source uri is outside allowed CutSell upload scope")
    monkeypatch.setattr(api, "validate_product_source_uri", reject)
    response = TestClient(api.app).post("/v1/flow-b/jobs", json={
        "project_id": "project-1",
        "user_id": "user-1",
        "sources": [
            {"original_name": "one.mov", "uri": "s3://bucket/legacy/one.mov", "source_order": 0},
        ],
    })
    assert response.status_code == 422
    assert "outside allowed CutSell upload scope" in response.json()["detail"]


def test_get_job_returns_progress_without_exposing_internal_tracebacks(monkeypatch):
    monkeypatch.setattr(
        api,
        "fetch_job_snapshot",
        lambda job_id: JobSnapshot(job_id, "analyzing", progress=42),
    )
    response = TestClient(api.app).get("/v1/jobs/job-1")
    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-1",
        "state": "analyzing",
        "progress": 42,
        "result": None,
        "error": None,
    }


def test_get_job_returns_404_for_unknown_job(monkeypatch):
    def missing(_job_id):
        raise KeyError("missing")
    monkeypatch.setattr(api, "fetch_job_snapshot", missing)
    response = TestClient(api.app).get("/v1/jobs/missing")
    assert response.status_code == 404


def test_cancel_job_returns_canceled_state(monkeypatch):
    monkeypatch.setattr(
        api,
        "cancel_job",
        lambda job_id: JobSnapshot(job_id, "canceled"),
    )
    response = TestClient(api.app).post("/v1/jobs/job-1/cancel")
    assert response.status_code == 200
    assert response.json()["state"] == "canceled"
