import json

from fastapi.testclient import TestClient

import cutsell_app.batch_routes as routes
import cutsell_app.main as api
import cutsell_worker.batch as batch
import cutsell_worker.batch_job as batch_job
from cutsell_worker.queueing import QueueSubmission


class FakeRedis:
    def __init__(self):
        self.data = {}
    def get(self, key):
        return self.data.get(key)
    def set(self, key, value):
        self.data[key] = value
        return True


def _payload(project_id):
    return {
        "project_id": project_id,
        "user_id": "u1",
        "sources": [{"source_asset_id": f"src-{project_id}", "uri": "s3://bucket/key"}],
    }


def test_batch_requires_one_to_ten_and_scopes_items_to_same_user():
    redis = FakeRedis()
    record = batch.create_batch(user_id="u1", payloads=[_payload("p1"), _payload("p2")], client=redis)
    assert record["items"][0]["state"] == "queued"
    assert record["items"][1]["state"] == "waiting"
    public = batch.get_batch(user_id="u1", batch_id=record["batch_id"], client=redis)
    assert "payloads" not in public
    try:
        batch.create_batch(user_id="u1", payloads=[_payload(str(i)) for i in range(11)], client=redis)
    except ValueError as exc:
        assert "1 to 10" in str(exc)
    else:
        raise AssertionError("batch over ten must be rejected")


def test_failed_batch_item_still_enqueues_next(monkeypatch):
    record = {"batch_id": "batch_1", "payloads": [_payload("p1"), _payload("p2")]}
    monkeypatch.setattr(batch_job, "get_batch", lambda **kwargs: record)
    updates = []
    monkeypatch.setattr(batch_job, "update_batch_item", lambda **kwargs: updates.append(kwargs) or {})
    monkeypatch.setattr(batch_job, "run_flow_b_job", lambda payload: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(batch_job, "get_queue", lambda: object())
    enqueued = []
    monkeypatch.setattr(
        batch_job,
        "enqueue_batch_item",
        lambda **kwargs: enqueued.append(kwargs) or QueueSubmission("job-2", "cutsell"),
    )
    import sys
    from types import ModuleType, SimpleNamespace
    rq = ModuleType("rq")
    rq.get_current_job = lambda: SimpleNamespace(id="job-1")
    monkeypatch.setitem(sys.modules, "rq", rq)

    result = batch_job.run_batch_item("batch_1", "u1", 0)
    assert result["state"] == "failed"
    assert enqueued[0]["index"] == 1
    assert any(item["state"] == "failed" for item in updates)
    assert any(item["index"] == 1 and item["state"] == "queued" for item in updates)


def test_batch_api_accepts_two_projects_and_starts_only_first(monkeypatch):
    monkeypatch.setattr(routes, "validate_product_source_uri", lambda uri, **kwargs: ("bucket", "key"))
    monkeypatch.setattr(
        routes,
        "create_batch",
        lambda **kwargs: {"batch_id": "batch_1", "items": [{}, {}]},
    )
    seen = []
    monkeypatch.setattr(
        routes,
        "enqueue_batch_item",
        lambda **kwargs: seen.append(kwargs) or QueueSubmission("job-1", "cutsell"),
    )
    response = TestClient(api.app).post("/v1/batches", json={
        "user_id": "u1",
        "projects": [
            {"project_id": "p1", "sources": [{"original_name": "a.mp4", "uri": "s3://bucket/a", "source_order": 0}]},
            {"project_id": "p2", "sources": [{"original_name": "b.mp4", "uri": "s3://bucket/b", "source_order": 0}]},
        ],
    })
    assert response.status_code == 202
    assert response.json()["item_count"] == 2
    assert seen == [{"batch_id": "batch_1", "user_id": "u1", "index": 0}]
