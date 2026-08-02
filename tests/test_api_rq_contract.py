from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from rq import Retry
from rq.exceptions import NoSuchJobError

import jobs
import web.routes_render as routes
from main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def enqueue(monkeypatch):
    calls = []

    def fake_enqueue(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id="job-123")

    monkeypatch.setattr(routes, "enqueue_render", fake_enqueue)
    return calls


@pytest.mark.parametrize("field", ["files", "file_urls"])
def test_render_accepts_both_url_fields(client, enqueue, field):
    response = client.post(
        "/render",
        json={"session_id": "session-1", field: ["https://example.com/video.mp4"], "mode": "clean"},
    )
    assert response.status_code == 202
    assert response.json() == {
        "job_id": "job-123", "session_id": "session-1", "status": "queued", "mode": "clean"
    }
    assert enqueue == [{
        "session_id": "session-1",
        "files": None,
        "file_urls": ["https://example.com/video.mp4"],
        "mode": "clean",
    }]


def test_real_main_application_registers_job_routes():
    paths = app.openapi()["paths"]
    assert "post" in paths["/render"]
    assert "get" in paths["/jobs/{job_id}"]


def test_legacy_input_url_reaches_enqueue(client, enqueue):
    response = client.post("/render", json={"input_url": "https://example.com/legacy.mp4"})
    assert response.status_code == 202
    assert enqueue[0]["files"] is None
    assert enqueue[0]["file_urls"] == ["https://example.com/legacy.mp4"]


def test_input_aliases_are_combined_and_deduplicated(client, enqueue):
    response = client.post(
        "/render",
        json={
            "input_url": "https://example.com/one.mp4",
            "file_urls": ["https://example.com/two.mp4", "https://example.com/one.mp4"],
            "files": ["https://example.com/two.mp4", "https://example.com/three.mp4"],
        },
    )
    assert response.status_code == 202
    assert enqueue[0]["file_urls"] == [
        "https://example.com/one.mp4",
        "https://example.com/two.mp4",
        "https://example.com/three.mp4",
    ]


def test_render_rejects_missing_inputs_and_invalid_mode(client, enqueue):
    assert client.post("/render", json={"mode": "human"}).status_code == 422
    assert client.post("/render", json={"files": ["https://example.com/a.mp4"], "mode": "other"}).status_code == 422
    assert enqueue == []


def test_render_generates_session_and_does_not_run_pipeline(client, enqueue, monkeypatch):
    monkeypatch.setattr("tasks.run_pipeline", lambda **_kwargs: pytest.fail("pipeline ran synchronously"))
    response = client.post("/render", json={"files": ["https://example.com/a.mp4"]})
    assert response.status_code == 202
    assert response.json()["session_id"].startswith("render-")
    assert enqueue[0]["session_id"] == response.json()["session_id"]


def test_enqueue_configuration_and_arguments(monkeypatch):
    recorded = {}
    monkeypatch.setenv("RQ_JOB_TIMEOUT", "120")
    monkeypatch.setenv("RQ_RESULT_TTL", "240")
    monkeypatch.setenv("RQ_FAILURE_TTL", "480")
    monkeypatch.setenv("RQ_MAX_RETRIES", "3")

    class Queue:
        def enqueue(self, function, **kwargs):
            recorded.update(function=function, **kwargs)
            return SimpleNamespace(id="rq-job")

    monkeypatch.setattr(jobs, "get_queue", Queue)
    job = jobs.enqueue_render("sid", files=["https://example.com/a.mp4"], mode="blooper")
    assert job.id == "rq-job"
    assert recorded["function"] == "tasks.job_render"
    assert recorded["kwargs"] == {
        "session_id": "sid", "files": ["https://example.com/a.mp4"], "file_urls": None, "mode": "blooper"
    }
    assert (recorded["job_timeout"], recorded["result_ttl"], recorded["failure_ttl"]) == (120, 240, 480)
    assert isinstance(recorded["retry"], Retry) and recorded["retry"].max == 3


@pytest.mark.parametrize(
    ("rq_status", "result", "expected_result", "expected_error"),
    [("queued", None, None, None), ("finished", {"url": "safe"}, {"url": "safe"}, None),
     ("failed", None, None, "Render job failed")],
)
def test_job_status(client, monkeypatch, rq_status, result, expected_result, expected_error):
    job = SimpleNamespace(id="job-123", result=result, get_status=lambda refresh: rq_status)
    monkeypatch.setattr(routes, "get_queue", lambda: SimpleNamespace(connection=object()))
    monkeypatch.setattr(routes.Job, "fetch", lambda *_args, **_kwargs: job)
    response = client.get("/jobs/job-123")
    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-123", "status": rq_status, "result": expected_result, "error": expected_error
    }


def test_unknown_job_returns_404(client, monkeypatch):
    monkeypatch.setattr(routes, "get_queue", lambda: SimpleNamespace(connection=object()))

    def missing(*_args, **_kwargs):
        raise NoSuchJobError

    monkeypatch.setattr(routes.Job, "fetch", missing)
    assert client.get("/jobs/missing").status_code == 404


def test_direct_single_file_job_compatibility(monkeypatch):
    import tasks

    monkeypatch.setattr(tasks, "run_pipeline", lambda **kwargs: kwargs)
    assert tasks.job_render("legacy", files=["https://example.com/one.mp4"])["files"] == [
        "https://example.com/one.mp4"
    ]
