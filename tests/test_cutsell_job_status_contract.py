from enum import Enum
from types import SimpleNamespace

from cutsell_worker.jobs import cancel_job, fetch_job_snapshot


class FakeJobStatus(Enum):
    QUEUED = "queued"
    STARTED = "started"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELED = "canceled"


class Queue:
    connection = object()


def _job(monkeypatch, status, *, result=None, meta=None):
    job = SimpleNamespace(
        id="job-1",
        result=result,
        meta=meta or {},
        args=({"user_id": "user-1"},),
        get_status=lambda refresh=True: status,
        cancel=lambda: None,
    )
    monkeypatch.setattr("cutsell_worker.jobs._fetch_job", lambda *_args, **_kwargs: job)
    return job


def test_job_status_enum_is_exposed_as_stable_public_string(monkeypatch):
    _job(monkeypatch, FakeJobStatus.FINISHED, result={"state": "finished"}, meta={"progress_percent": 100})
    snap = fetch_job_snapshot("job-1", user_id="user-1", queue=Queue())
    assert snap.state == "finished"
    assert snap.progress == 100
    assert snap.result == {"state": "finished"}


def test_started_enum_maps_to_analyzing(monkeypatch):
    _job(monkeypatch, FakeJobStatus.STARTED, meta={"progress_percent": 58})
    snap = fetch_job_snapshot("job-1", user_id="user-1", queue=Queue())
    assert snap.state == "analyzing"
    assert snap.progress == 58


def test_stringified_rq_enum_is_normalized(monkeypatch):
    _job(monkeypatch, "JobStatus.FINISHED", result={"ok": True})
    snap = fetch_job_snapshot("job-1", user_id="user-1", queue=Queue())
    assert snap.state == "finished"
    assert snap.result == {"ok": True}


def test_cancel_finished_enum_does_not_cancel(monkeypatch):
    job = _job(monkeypatch, FakeJobStatus.FINISHED, result={"ok": True})
    job.cancel = lambda: (_ for _ in ()).throw(AssertionError("finished job was canceled"))
    snap = cancel_job("job-1", user_id="user-1", queue=Queue())
    assert snap.state == "finished"
    assert snap.result == {"ok": True}
