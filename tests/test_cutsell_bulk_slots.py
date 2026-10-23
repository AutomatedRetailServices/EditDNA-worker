"""23 Oct: a Bulk send gets one processing slot per video (Batch API waits up to 30 min each); the limit answers 429."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cutsell_worker import queueing, usage_limits


class FakeRedis:
    def __init__(self):
        self.n = 0

    def eval(self, script, nkeys, key, limit, ttl):
        if self.n >= int(limit):
            return [0, self.n]
        self.n += 1
        return [1, self.n]


def _payload(bulk_size=None):
    meta = {"bulk_size": str(bulk_size), "bulk_id": "b1"} if bulk_size else {}
    return {"user_id": "u1", "project_id": "p", "sources": [{"uri": "x", "metadata": meta}]}


@pytest.mark.parametrize("size,extra", [(None, 0), (1, 0), (2, 1), (5, 4), (10, 9), (50, 9)])
def test_bulk_extra_slots(size, extra):
    assert queueing.bulk_extra_slots(_payload(size)) == extra


def test_bad_bulk_size_is_ignored():
    p = _payload()
    p["sources"][0]["metadata"] = {"bulk_size": "lots"}
    assert queueing.bulk_extra_slots(p) == 0


def _run(monkeypatch, sizes):
    redis = FakeRedis()
    monkeypatch.setattr(usage_limits, "load_runtime_config",
                        lambda: SimpleNamespace(redis_url="redis://x", max_concurrent_jobs_per_user=2))
    real = usage_limits.reserve_processing_slot
    monkeypatch.setattr(queueing, "reserve_processing_slot", lambda **kw: real(client=redis, **kw))
    monkeypatch.setattr(queueing, "wake_worker", lambda: None)
    q = SimpleNamespace(name="cutsell", enqueue=lambda *a, **k: SimpleNamespace(id="j"))
    out = []
    for s in sizes:
        try:
            queueing.enqueue_flow_b(_payload(s), queue=q)
            out.append("ok")
        except queueing.ProcessingLimitError:
            out.append("limit")
    return out


def test_bulk_of_five_all_accepted(monkeypatch):
    assert _run(monkeypatch, [5] * 5) == ["ok"] * 5


def test_singles_keep_the_limit_of_two(monkeypatch):
    assert _run(monkeypatch, [None, None, None]) == ["ok", "ok", "limit"]


def test_limit_error_is_still_a_runtime_error():
    assert issubclass(queueing.ProcessingLimitError, RuntimeError)


def test_api_answers_429(monkeypatch):
    from fastapi.testclient import TestClient
    import cutsell_app.main as api

    def full(_payload):
        raise queueing.ProcessingLimitError("processing_concurrency_limit")

    monkeypatch.setattr(api, "enqueue_flow_b", full)
    monkeypatch.setattr(api, "validate_product_source_uri", lambda uri, **kwargs: ("bucket", uri))
    r = TestClient(api.app).post("/v1/flow-b/jobs", json={
        "project_id": "project-1", "user_id": "user-1",
        "sources": [{"original_name": "one.mov", "uri": "s3://bucket/one.mov", "source_order": 0}],
    })
    assert r.status_code == 429
    assert r.json()["detail"] == "processing_concurrency_limit"
    assert r.headers.get("retry-after") == "60"
