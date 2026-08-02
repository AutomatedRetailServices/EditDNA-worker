import copy
import importlib.util
import os
import sys
import types

import pytest
from fastapi.testclient import TestClient
from rq.exceptions import NoSuchJobError


def _stub(name, **attrs):
    if name not in sys.modules and importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module


_stub("requests")
_stub("boto3")
_stub("clip")
_stub("faster_whisper", WhisperModel=object)

import job_progress
import tasks
import web.routes_render as routes
from main import app
from worker import pipeline


class FakeJob:
    def __init__(self, status="queued", meta=None, result=None):
        self.id = "job-123"
        self.status = status
        self.meta = dict(meta or {})
        self.result = result
        self.saved = 0
        self.canceled = 0

    def get_status(self, refresh=True):
        return self.status

    def save_meta(self):
        self.saved += 1

    def cancel(self):
        self.canceled += 1
        self.status = "canceled"

    def refresh(self):
        return None


@pytest.fixture
def client():
    return TestClient(app)


def _install_job(monkeypatch, job):
    monkeypatch.setattr(routes, "get_queue", lambda: types.SimpleNamespace(connection=object()))
    monkeypatch.setattr(routes.Job, "fetch", lambda *_args, **_kwargs: job)


def test_default_queued_progress(client, monkeypatch):
    _install_job(monkeypatch, FakeJob())
    body = client.get("/jobs/job-123").json()
    assert (body["stage"], body["progress"], body["message"]) == ("queued", 0, "Render queued")
    assert body["cancel_requested"] is False


@pytest.mark.parametrize(("status", "code"), [("finished", 409), ("failed", 409)])
def test_terminal_jobs_cannot_be_canceled(client, monkeypatch, status, code):
    job = FakeJob(status)
    _install_job(monkeypatch, job)
    assert client.post("/jobs/job-123/cancel").status_code == code
    assert job.saved == job.canceled == 0


def test_cancel_unknown_job(client, monkeypatch):
    monkeypatch.setattr(routes, "get_queue", lambda: types.SimpleNamespace(connection=object()))
    monkeypatch.setattr(routes.Job, "fetch", lambda *_args, **_kwargs: (_ for _ in ()).throw(NoSuchJobError()))
    assert client.post("/jobs/missing/cancel").status_code == 404


def test_cancel_queued_job_and_repeat_is_idempotent(client, monkeypatch):
    job = FakeJob()
    _install_job(monkeypatch, job)
    expected = {"job_id": "job-123", "status": "canceled", "cancel_requested": True}
    assert client.post("/jobs/job-123/cancel").json() == expected
    assert job.canceled == 1 and job.meta["stage"] == "canceled"
    assert client.post("/jobs/job-123/cancel").json() == expected
    assert job.canceled == 1


def test_started_job_receives_cooperative_request(client, monkeypatch):
    job = FakeJob("started", {"stage": "analyzing", "progress": 30})
    _install_job(monkeypatch, job)
    assert client.post("/jobs/job-123/cancel").json() == {
        "job_id": "job-123", "status": "started", "cancel_requested": True
    }
    assert job.meta["cancel_requested"] is True and job.saved == 1 and job.canceled == 0


@pytest.fixture
def configured(monkeypatch, tmp_path):
    state = {"render": 0, "upload": 0, "dirs": []}
    clip = {"id": "chosen", "start": 0.0, "end": 1.0, "text": "x", "slot": "STORY",
            "semantic_score": 1.0, "source_index": 0, "meta": {"keep": True}}

    def session_dir(_sid):
        path = tmp_path / f"job-{len(state['dirs'])}"
        path.mkdir(); state["dirs"].append(path)
        return str(path)

    def download(_url, path):
        with open(path, "wb") as output: output.write(b"source")

    def render(_sources, directory, _clips, _ids):
        state["render"] += 1
        output = os.path.join(directory, "final.mp4")
        with open(output, "wb") as stream: stream.write(b"render")
        return output

    monkeypatch.setattr(pipeline, "ensure_session_dir", session_dir)
    monkeypatch.setattr(pipeline, "download_to_local", download)
    monkeypatch.setattr(pipeline, "probe_duration", lambda _path: 1.0)
    monkeypatch.setattr(pipeline, "run_asr", lambda _path: [{}])
    monkeypatch.setattr(pipeline, "sentence_boundary_micro_cuts", lambda _asr: [copy.deepcopy(clip)])
    monkeypatch.setattr(pipeline, "merge_incomplete_phrases", lambda x: x)
    monkeypatch.setattr(pipeline, "enrich_clips_semantic", lambda _x: False)
    monkeypatch.setattr(pipeline, "dedupe_clips", lambda x: x)
    monkeypatch.setattr(pipeline, "run_visual_pass", lambda *_x: False)
    monkeypatch.setattr(pipeline, "build_slots_dict", lambda _x: {})
    monkeypatch.setattr(pipeline, "build_composer", lambda _x, mode: {"mode": mode, "used_clip_ids": ["chosen"]})
    monkeypatch.setattr(pipeline, "pretty_print_composer", lambda *_x: "diagnostic")
    monkeypatch.setattr(pipeline, "render_funnel_video", render)
    monkeypatch.setattr(pipeline, "save_result_json_to_s3", lambda _x: None)
    monkeypatch.setattr(pipeline, "S3_BUCKET", "bucket")

    def upload(*_args):
        state["upload"] += 1
        return "https://output.example/final.mp4"
    monkeypatch.setattr(pipeline, "upload_to_s3", upload)
    return state


def test_pipeline_reports_all_stages_and_monotonic_multi_file_analysis(configured):
    events = []
    result = pipeline.run_pipeline("session", files=["one", "two"], progress=lambda *event: events.append(event))
    assert result["ok"] is True
    assert {stage for stage, _, _ in events} >= {"downloading", "analyzing", "selecting", "rendering", "uploading"}
    percents = [percent for _, percent, _ in events]
    assert percents == sorted(percents)
    analysis = [percent for stage, percent, _ in events if stage == "analyzing"]
    assert analysis == sorted(analysis) and len(set(analysis)) == 3
    assert ("selecting", 60) in [(s, p) for s, p, _ in events]
    assert ("rendering", 65) in [(s, p) for s, p, _ in events]
    assert ("uploading", 92) in [(s, p) for s, p, _ in events]


@pytest.mark.parametrize(("cancel_at", "renders", "uploads"), [(1, 0, 0), (4, 0, 0), (7, 1, 0)])
def test_cooperative_cancellation_boundaries_cleanup_and_never_succeed(configured, cancel_at, renders, uploads):
    checks = 0
    def check():
        nonlocal checks
        checks += 1
        if checks == cancel_at:
            raise pipeline.JobCanceledError("canceled")

    with pytest.raises(pipeline.JobCanceledError):
        pipeline.run_pipeline("session", files=["one"], check_canceled=check)
    assert configured["render"] == renders
    assert configured["upload"] == uploads
    assert not configured["dirs"][-1].exists()


def test_progress_save_failure_does_not_fail_render(monkeypatch):
    job = FakeJob("started")
    job.save_meta = lambda: (_ for _ in ()).throw(OSError("redis unavailable"))
    reporter = job_progress.RQProgressReporter(job)
    reporter.update("rendering", 65, "Rendering video")
    assert job.meta["stage"] == "rendering"


def test_direct_non_rq_job_render_and_finished_progress(monkeypatch):
    monkeypatch.setattr(job_progress, "current_rq_job", lambda: None)
    monkeypatch.setattr(tasks, "run_pipeline", lambda **_kwargs: {"ok": True, "composer_mode": "human"})
    assert tasks.job_render("direct", files=["input"])["ok"] is True

    job = FakeJob("started")
    monkeypatch.setattr(job_progress, "current_rq_job", lambda: job)
    tasks.job_render("rq", files=["input"])
    assert job.meta["stage"] == "finished" and job.meta["progress"] == 100
