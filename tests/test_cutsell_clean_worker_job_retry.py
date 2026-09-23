from types import SimpleNamespace

from fastapi.testclient import TestClient

import cutsell_app.job_retry_routes as routes
import cutsell_app.main as api
import cutsell_worker.jobs as jobs
from cutsell_worker.queueing import QueueSubmission


class FakeQueue:
    name = "cutsell"
    connection = object()


class FakeJob:
    def __init__(self, *, status="failed", func_name="cutsell_worker.worker_job.run_flow_b_job", payload=None):
        self.id = "old-job"
        self.func_name = func_name
        self.args = (payload or {"user_id": "user-1", "project_id": "p1"},)
        self._status = status
    def get_status(self, refresh=True):
        return self._status


def test_retry_flow_b_creates_fresh_job_from_failed_payload(monkeypatch):
    fake = FakeJob()
    monkeypatch.setattr(jobs, "_fetch_job", lambda job_id, connection: fake)
    seen = []
    monkeypatch.setattr(
        jobs,
        "enqueue_flow_b",
        lambda payload, queue=None: seen.append((payload, queue)) or QueueSubmission("new-job", "cutsell"),
    )
    result = jobs.retry_job("old-job", user_id="user-1", queue=FakeQueue())
    assert result.job_id == "new-job"
    assert seen[0][0]["project_id"] == "p1"
    assert seen[0][1].name == "cutsell"


def test_retry_export_routes_to_export_worker_and_rejects_wrong_owner_or_active_job(monkeypatch):
    export_job = FakeJob(func_name="cutsell_worker.export_job.run_export_job")
    monkeypatch.setattr(jobs, "_fetch_job", lambda job_id, connection: export_job)
    monkeypatch.setattr(
        jobs,
        "enqueue_export",
        lambda payload, queue=None: QueueSubmission("new-export", "cutsell"),
    )
    assert jobs.retry_job("old-job", user_id="user-1", queue=FakeQueue()).job_id == "new-export"

    try:
        jobs.retry_job("old-job", user_id="other-user", queue=FakeQueue())
    except PermissionError:
        pass
    else:
        raise AssertionError("retry must enforce user ownership")

    active = FakeJob(status="started")
    monkeypatch.setattr(jobs, "_fetch_job", lambda job_id, connection: active)
    try:
        jobs.retry_job("old-job", user_id="user-1", queue=FakeQueue())
    except ValueError as exc:
        assert "failed or canceled" in str(exc)
    else:
        raise AssertionError("active jobs must not be duplicated by retry")


def test_retry_api_returns_new_job_and_maps_ownership_failure(monkeypatch):
    monkeypatch.setattr(
        routes,
        "retry_job",
        lambda job_id, user_id: QueueSubmission("fresh-job", "cutsell"),
    )
    response = TestClient(api.app).post("/v1/jobs/old-job/retry", json={"user_id": "user-1"})
    assert response.status_code == 200
    assert response.json()["job_id"] == "fresh-job"
    assert response.json()["original_job_id"] == "old-job"

    def forbidden(job_id, user_id):
        raise PermissionError("no")
    monkeypatch.setattr(routes, "retry_job", forbidden)
    denied = TestClient(api.app).post("/v1/jobs/old-job/retry", json={"user_id": "other"})
    assert denied.status_code == 403
