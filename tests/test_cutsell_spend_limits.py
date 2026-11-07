"""PO 5 Nov (security review 4 Nov): every door that queues paid work has a limit, the worker checks the REAL video
length before anything paid runs, and the Bulk size is counted by the server instead of believed."""
from __future__ import annotations

from types import SimpleNamespace

import fakeredis
import pytest

from cutsell_worker import queueing, usage_limits


@pytest.fixture()
def redis(monkeypatch):
    client = fakeredis.FakeRedis()
    monkeypatch.setattr(usage_limits, "_redis", lambda c=None: c or client)
    monkeypatch.setattr(usage_limits, "load_runtime_config", lambda: SimpleNamespace(
        redis_url="redis://x", max_concurrent_jobs_per_user=2, max_source_minutes=0, database_url=None,
        monthly_processing_minutes=300))
    return client


def _bulk(size, bulk_id="b1", user="u1"):
    return {"user_id": user, "project_id": "p", "sources": [{"uri": "x", "metadata": {"bulk_size": str(size),
                                                                                      "bulk_id": bulk_id}}]}


def test_one_request_claiming_a_bulk_of_ten_gets_no_extra_slot(redis):
    assert queueing.bulk_extra_slots(_bulk(10)) == 0


def test_the_nth_video_of_a_bulk_gets_n_slots_never_more_than_announced(redis):
    assert [queueing.bulk_extra_slots(_bulk(3)) for _ in range(5)] == [0, 1, 2, 2, 2]
    assert queueing.bulk_extra_slots(_bulk(3, bulk_id="other")) == 0          # another send starts again
    assert queueing.bulk_extra_slots(_bulk(3, user="u2")) == 0               # and another person


def test_a_real_bulk_of_ten_still_runs_side_by_side(redis, monkeypatch):
    active = {"n": 0}

    def reserve(*, user_id, extra=0, **kw):                 # the Lua script, minus Lua (fakeredis has none)
        if active["n"] >= 2 + extra:
            return {"allowed": False}
        active["n"] += 1
        return {"allowed": True}

    monkeypatch.setattr(queueing, "reserve_processing_slot", reserve)
    monkeypatch.setattr(queueing, "wake_worker", lambda: None)
    q = SimpleNamespace(name="cutsell", enqueue=lambda *a, **k: SimpleNamespace(id="j"))
    for i in range(10):
        p = _bulk(10)
        p["project_id"] = f"p{i}"
        queueing.enqueue_flow_b(p, queue=q)                                   # none refused
    with pytest.raises(queueing.ProcessingLimitError):
        queueing.enqueue_flow_b({"user_id": "u1", "project_id": "x", "sources": [{"uri": "x"}]}, queue=q)


@pytest.mark.parametrize("kind", sorted(usage_limits.HOURLY_LIMITS))
def test_every_kind_of_paid_work_has_an_hourly_ceiling(redis, kind):
    limit = usage_limits.HOURLY_LIMITS[kind]
    for _ in range(limit):
        assert usage_limits.work_gate(user_id="u1", kind=kind) is None
    assert usage_limits.work_gate(user_id="u1", kind=kind) == f"{kind}_hourly_limit"
    assert usage_limits.work_gate(user_id="u2", kind=kind) is None            # someone else is not affected


def test_used_up_month_blocks_new_cutting_work_not_her_finished_videos(redis, monkeypatch):
    monkeypatch.setattr(usage_limits, "check_processing_allowance", lambda **kw: usage_limits.UsageDecision(
        True, "allowed", 0.0, 300.0, 300.0))
    for kind in ("mix", "batch", "retry"):                                  # new cutting work: blocked
        assert usage_limits.work_gate(user_id="u1", kind=kind) == "monthly_processing_limit"
    for kind in ("export", "clean_audio", "video_ready"):                  # her finished videos: still allowed
        assert usage_limits.work_gate(user_id="u1", kind=kind) is None
    monkeypatch.setattr(queueing, "wake_worker", lambda: None)
    q = SimpleNamespace(name="cutsell", enqueue=lambda *a, **k: SimpleNamespace(id="j"))
    with pytest.raises(queueing.ProcessingLimitError):
        queueing.enqueue_mix({"user_id": "u1", "mix_id": "m"}, queue=q)


def test_measured_length_is_checked_not_the_declared_one(monkeypatch):
    monkeypatch.setattr(usage_limits, "load_runtime_config", lambda: SimpleNamespace(
        max_source_minutes=0, database_url=None, monthly_processing_minutes=300))
    usage_limits.check_measured(user_id="u1", durations_sec=[45.0, 300.0])           # fine
    with pytest.raises(usage_limits.MeasuredLimitExceeded, match="source_duration_limit"):
        usage_limits.check_measured(user_id="u1", durations_sec=[3 * 3600.0])        # "0 s" declared, 3 h real


def test_worker_refuses_a_too_long_video_before_speech_to_text(monkeypatch, tmp_path):
    """The worker probes the real file, then refuses before the engine (Deepgram / Claude) is called."""
    from cutsell_worker import worker_job
    src = tmp_path / "long.mp4"
    src.write_bytes(b"x")
    called = {"engine": 0}
    monkeypatch.setenv("CUTSELL_ENGINE", "simple")
    monkeypatch.setattr(worker_job, "validate_product_source_uri", lambda uri, **kw: None)
    monkeypatch.setattr(worker_job, "download_source", lambda uri, dest, **kw: str(src))
    monkeypatch.setattr(worker_job, "probe_media", lambda path: SimpleNamespace(duration_sec=3 * 3600.0))
    monkeypatch.setattr(worker_job, "load_runtime_config", lambda: SimpleNamespace(asr_model="tiny"))
    monkeypatch.setattr(worker_job, "safe_update_project", lambda **kw: {"status": "saved"})
    monkeypatch.setattr(worker_job, "publish_notification", lambda **kw: {"notification_id": "n1"})
    monkeypatch.setattr(worker_job, "record_processing_minutes", lambda **kw: None)
    monkeypatch.setattr(worker_job, "release_processing_slot", lambda **kw: None)
    monkeypatch.setattr(usage_limits, "load_runtime_config", lambda: SimpleNamespace(
        max_source_minutes=0, database_url=None, monthly_processing_minutes=300))
    import cutsell_worker.simple_engine_adapter as adapter
    monkeypatch.setattr(adapter, "process_with_simple_engine", lambda *a, **k: called.__setitem__("engine", 1))
    with pytest.raises(usage_limits.MeasuredLimitExceeded):
        worker_job.run_flow_b_job({"project_id": "p1", "user_id": "u1", "sources": [
            {"source_asset_id": "s1", "uri": "s3://b/k.mp4", "original_name": "k.mp4", "duration_sec": 0}]})
    assert called["engine"] == 0


def test_api_answers_429_when_a_door_is_limited(monkeypatch):
    from fastapi.testclient import TestClient
    import cutsell_app.main as api
    from cutsell_app import job_retry_routes

    def full(*a, **k):
        raise queueing.ProcessingLimitError("retry_hourly_limit")

    monkeypatch.setattr(job_retry_routes, "retry_job", full)
    r = TestClient(api.app).post("/v1/jobs/j1/retry", json={"user_id": "u1"})
    assert r.status_code == 429 and r.json()["detail"] == "retry_hourly_limit"
    assert r.headers.get("retry-after") == "60"


def test_any_route_raising_the_limit_answers_429(monkeypatch):
    """The app-wide handler: export, video-ready, Bulk and Clean up voice raise ProcessingLimitError -> 429."""
    from fastapi.testclient import TestClient
    import cutsell_app.main as api

    def full(*a, **k):
        raise queueing.ProcessingLimitError("video_ready_hourly_limit")

    monkeypatch.setattr(api, "enqueue_video_followup", full)
    monkeypatch.setattr(api, "validate_product_source_uri", lambda uri, **kwargs: ("bucket", uri))
    import cutsell_app.mix_routes as mix_routes
    monkeypatch.setattr(mix_routes, "is_mix", lambda user, project: False)
    r = TestClient(api.app).post("/v1/projects/p1/video-ready", json={"user_id": "u1", "uri": "s3://b/k.mp4"})
    assert r.status_code == 429 and r.json()["detail"] == "video_ready_hourly_limit"


def test_the_monthly_meter_fails_closed_and_counts_in_redis(monkeypatch):
    """Security review 4 Nov, point 9: no readable meter -> refused, never free processing."""
    import fakeredis
    from types import SimpleNamespace
    from cutsell_worker import usage_limits as ul
    config = SimpleNamespace(database_url=None, redis_url="redis://x", monthly_processing_minutes=10,
                             max_source_minutes=0)
    monkeypatch.setattr(ul, "load_runtime_config", lambda: config)
    meter = fakeredis.FakeRedis()
    monkeypatch.setattr(ul, "_meter_client", lambda client=None: meter)
    assert ul.check_processing_allowance(user_id="u", durations_sec=[300]).allowed
    assert ul.record_processing_minutes(user_id="u", project_id="p", minutes=8)["status"] == "recorded_month_counter"
    late = ul.check_processing_allowance(user_id="u", durations_sec=[300])
    assert not late.allowed and late.reason == "monthly_processing_limit" and late.monthly_used_minutes == 8
    assert ul.check_processing_allowance(user_id="other", durations_sec=[300]).allowed

    class Down:
        def get(self, *a):
            raise ConnectionError

        def pipeline(self):
            raise ConnectionError
    monkeypatch.setattr(ul, "_meter_client", lambda client=None: Down())
    refused = ul.check_processing_allowance(user_id="u", durations_sec=[60])
    assert not refused.allowed and refused.reason == "usage_meter_unavailable"
